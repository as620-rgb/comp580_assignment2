"""Run the Spartacus agent over Assignment 2 datasets, with or without MinHash reuse.

Usage:
    python run.py --datasets tickets configs --config baseline --rpm 14
    python run.py --datasets tickets --config both --limit 5 --keep
    python run.py --datasets tickets configs --config baseline --resume   # after a stop

Configs: baseline (no reuse), task (policy.py only), content (content.py only), both.

Each incoming task, in `seq` order, gets a fresh working directory holding exactly
what the agent may see: INSTRUCTIONS.md, history.jsonl, reference files (db.json),
task.json and, for corpus datasets, the files of the task's scenario. hidden/ is
never copied, and verify.py runs as a separate process, so labels never enter this
one. Per-task metrics go to results/<config>/<dataset>.jsonl.

Model calls are paced to --rpm and a rate-limited call (HTTP 429) is retried after a
pause; time spent waiting is reported as throttle_s and left out of a task's seconds.
An API error that outlasts the retries stops the run instead of being scored as a
wrong answer; --resume continues from the last completed task.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(HERE, "Assignment2_COMP_580", "assignment2", "datasets")
sys.path.insert(0, os.path.join(HERE, "Spartacus"))

from spartacus import Harness  # noqa: E402
from spartacus import config as sp_config  # noqa: E402
from spartacus import provider as sp_provider  # noqa: E402
from spartacus import tools as sp_tools  # noqa: E402

import content  # noqa: E402
import policy  # noqa: E402

MODEL = "gemini-3.5-flash-lite"
PROMPT = ("Read INSTRUCTIONS.md, then complete the request in task.json. "
          "Write your answer to answer.json exactly as the instructions specify.")
CONFIGS = {"baseline": (False, False), "task": (True, False),
           "content": (False, True), "both": (True, True)}
CONTENT_TOOLS = ("read_file", "grep", "bash")  # the tools whose output can repeat


class Throttle:
    """Paces model calls to `rpm` per minute and waits out HTTP 429s, timing the waits."""

    def __init__(self, rpm, retries=8, pause=60):
        self.gap = 60.0 / rpm if rpm else 0.0
        self.retries, self.pause = retries, pause
        self.next = 0.0
        self.waited = 0.0  # seconds spent pacing or backing off, excluded from task time

    def wrap(self, complete):
        def call(*args, **kwargs):
            for attempt in range(self.retries):
                delay = self.next - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                    self.waited += delay
                self.next = time.monotonic() + self.gap
                start = time.monotonic()
                try:
                    return complete(*args, **kwargs)
                except RuntimeError as error:
                    if "HTTP 429" not in str(error) or attempt == self.retries - 1:
                        raise
                    print("  rate limited; waiting %ds" % self.pause, flush=True)
                    time.sleep(self.pause)
                    self.waited += time.monotonic() - start  # the failed call and the pause
        return call


def load_tasks(env):
    folder = os.path.join(env, "incoming")
    tasks = []
    for name in os.listdir(folder):
        if name.endswith(".json"):
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                tasks.append(json.load(handle))
    return sorted(tasks, key=lambda task: (task.get("seq", 0), task["tid"]))


def _norm(text):
    """Scenario ids are spelled both ways (CFG1000 / CFG-1000)."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def make_workdir(env, task, root):
    work = tempfile.mkdtemp(prefix=task["tid"] + "-", dir=root)
    for name in os.listdir(env):
        if name in ("incoming", "scenarios"):
            continue
        src = os.path.join(env, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(work, name))
        else:
            shutil.copy2(src, work)
    if task.get("scenario"):
        scenarios = os.path.join(env, "scenarios")
        match = [d for d in os.listdir(scenarios) if _norm(d) == _norm(task["scenario"])]
        if not match:
            raise FileNotFoundError("no scenario directory for %r" % task["scenario"])
        shutil.copytree(os.path.join(scenarios, match[0]), work, dirs_exist_ok=True)
    with open(os.path.join(work, "task.json"), "w", encoding="utf-8") as handle:
        json.dump(task, handle, indent=2)
    return work


def verify(dataset, tid, answer_path):
    """(passed, verifier output); a subprocess so hidden labels stay out of this process."""
    script = os.path.join(DATASETS, dataset, "verify.py")
    proc = subprocess.run([sys.executable, script, "--tid", tid, "--answer", answer_path],
                          capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def read_answer(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def content_tools(workdir, cache, row):
    """The core tools whose output passes through content.py; they replace the originals."""
    wrapped = []
    for original in sp_tools.core_tools(workdir):
        if original.name not in CONTENT_TOOLS:
            continue

        def run(_original=original, **args):
            output = str(_original.run(**args))
            start = time.perf_counter()
            sent = cache.process(_original.name, args, output)
            row["content_s"] += time.perf_counter() - start
            row["chars_raw"] += len(output)
            row["chars_sent"] += len(sent)
            return sent

        wrapped.append(sp_tools.Tool(original.name, original.spec, run))
    return wrapped


def run_task(dataset, env, task, args, index, root):
    work = make_workdir(env, task, root)
    answer_path = os.path.join(work, "answer.json")
    row = {"tid": task["tid"], "reused": False, "tool_calls": 0, "model_requests": 0,
           "input_tokens": 0, "output_tokens": 0, "policy_s": 0.0, "content_s": 0.0,
           "chars_raw": 0, "chars_sent": 0, "error": None}
    start, waited = time.perf_counter(), args.throttle.waited

    answer = None
    if index is not None:
        t0 = time.perf_counter()
        answer = index.lookup(task)
        row["policy_s"] = time.perf_counter() - t0
    if answer is not None:
        row["reused"] = True
        row["match"] = getattr(index, "last", None)  # (record id, similarity)
        with open(answer_path, "w", encoding="utf-8") as handle:
            json.dump(answer, handle)
    else:
        def on_event(kind, payload):
            if kind == "assistant":
                usage = payload.get("usage") or {}
                row["model_requests"] += 1
                row["input_tokens"] += usage.get("input", 0)
                row["output_tokens"] += usage.get("output", 0)
            elif kind == "tool_start":
                row["tool_calls"] += 1
                row.setdefault("calls", []).append(
                    "%s %s" % (payload["name"], json.dumps(payload["args"])[:160]))

        extra = content_tools(work, content.ContentCache(**args.content_opts), row) \
            if args.use_content else None
        agent = Harness(work, model=args.model, extra_tools=extra, on_event=on_event,
                        max_turns=args.max_turns, persist=False, enable_subagents=False)
        try:
            agent.run(PROMPT)
        except Exception as error:  # a failed API call is a failed task, not a crashed run
            row["error"] = "%s: %s" % (type(error).__name__, error)
    # Includes MinHash overhead; excludes rate-limit waits and verification.
    row["throttle_s"] = args.throttle.waited - waited
    row["seconds"] = time.perf_counter() - start - row["throttle_s"]
    if row["error"]:
        return row  # not scored: the caller stops the run

    row["passed"], row["verdict"] = verify(dataset, task["tid"], answer_path)
    row["answer"] = read_answer(answer_path)
    if index is not None and row["passed"] and not row["reused"]:
        index.add(task, row["answer"])  # only verified answers are indexed
    return row


def run_dataset(dataset, args, root):
    env = os.path.join(DATASETS, dataset, "env")
    tasks = load_tasks(env)[:args.limit]
    index = policy.TaskIndex(dataset, env, **args.task_opts) if args.use_task else None
    out_dir = os.path.join(args.out, args.config)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, dataset + ".jsonl")
    done = {}
    if args.resume and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            done = {row["tid"]: row for row in map(json.loads, handle)}
    rows = []
    with open(path, "a" if done else "w", encoding="utf-8") as out:
        for n, task in enumerate(tasks, 1):
            if task["tid"] in done:  # replay it, so the index is what it was
                row = done[task["tid"]]
                if index is not None and row["passed"] and not row["reused"]:
                    index.add(task, row.get("answer"))
                rows.append(row)
                continue
            row = run_task(dataset, env, task, args, index, root)
            if row["error"]:
                raise SystemExit("%s/%s %s: %s\nStopped; fix the cause and rerun with --resume."
                                 % (args.config, dataset, task["tid"], row["error"][:300]))
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print("[%s/%s %d/%d] %s %s%s %.1fs" % (
                args.config, dataset, n, len(tasks), row["tid"],
                "PASS" if row["passed"] else "FAIL", " (reused)" if row["reused"] else "",
                row["seconds"]), flush=True)
    return summarize(dataset, rows)


def summarize(dataset, rows):
    n = max(1, len(rows))
    keys = ("tool_calls", "model_requests", "input_tokens", "output_tokens",
            "seconds", "throttle_s", "policy_s", "content_s", "chars_raw", "chars_sent")
    summary = {"dataset": dataset, "tasks": len(rows),
               "accuracy": sum(r["passed"] for r in rows) / n,
               "reused": sum(r["reused"] for r in rows),
               "reused_wrong": sum(r["reused"] and not r["passed"] for r in rows),
               "errors": sum(bool(r["error"]) for r in rows)}
    summary.update({key: sum(r[key] for r in rows) for key in keys})
    return summary


def parse_opts(pairs):
    """KEY=VALUE pairs -> kwargs, numbers parsed, for threshold sweeps."""
    opts = {}
    for pair in pairs or []:
        key, value = pair.split("=", 1)
        try:
            opts[key] = json.loads(value)
        except ValueError:
            opts[key] = value
    return opts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--config", choices=CONFIGS, default="baseline")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None, help="first N tasks only")
    parser.add_argument("--task-opt", action="append", help="KEY=VALUE for policy.TaskIndex")
    parser.add_argument("--content-opt", action="append", help="KEY=VALUE for content.ContentCache")
    parser.add_argument("--out", default=os.path.join(HERE, "results"))
    parser.add_argument("--keep", action="store_true", help="keep the per-task working directories")
    parser.add_argument("--rpm", type=float, default=14, help="model requests per minute (0: no pacing)")
    parser.add_argument("--resume", action="store_true", help="skip tasks already in the results file")
    args = parser.parse_args(argv)
    sp_config.load(os.path.join(HERE, ".env"))  # GEMINI_API_KEY=...; exported variables win
    args.throttle = Throttle(args.rpm)
    sp_provider.complete = args.throttle.wrap(sp_provider.complete)  # the loop calls it by name
    args.use_task, args.use_content = CONFIGS[args.config]
    args.task_opts, args.content_opts = parse_opts(args.task_opt), parse_opts(args.content_opt)

    root = tempfile.mkdtemp(prefix="a2-runs-")
    try:
        summaries = [run_dataset(dataset, args, root) for dataset in args.datasets]
    finally:
        if args.keep:
            print("working directories kept in", root)
        else:
            shutil.rmtree(root, ignore_errors=True)
    path = os.path.join(args.out, args.config, "summary.json")
    merged = {}
    if os.path.exists(path):  # one entry per dataset, across separate invocations
        with open(path, encoding="utf-8") as handle:
            merged = {s["dataset"]: s for s in json.load(handle)}
    merged.update({s["dataset"]: s for s in summaries})
    with open(path, "w", encoding="utf-8") as out:
        json.dump(sorted(merged.values(), key=lambda s: s["dataset"]), out, indent=2)
    for s in summaries:
        print("%-10s acc=%.3f reused=%d (wrong %d) tools=%d requests=%d in_tok=%d time=%.1fs"
              % (s["dataset"], s["accuracy"], s["reused"], s["reused_wrong"], s["tool_calls"],
                 s["model_requests"], s["input_tokens"], s["seconds"]))


if __name__ == "__main__":
    sys.exit(main())
