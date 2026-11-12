"""Offline analyses for the report: no model calls, no API key.

Usage: python offline.py            # writes results/offline.json

1. Task-level threshold sweep, on every incoming task of each dataset: the index holds
   history.jsonl only (no agent answers exist for tasks that were not run), so this
   counts reuse of history duplicates. Each threshold is run twice: with the validation
   key gate (policy.py as submitted) and with similarity alone, to show what the gate
   prevents. Reused answers are scored by the dataset's own verify.py -- evaluation only.
2. Content-level upper bound: every file of every scenario read once, in path order,
   through content.py, at several thresholds; each fold is checked to rebuild exactly.
"""

import importlib.util
import json
import os
import time

import content
import minhash
import policy
import run
from spartacus import tools as sp_tools

TASK_DATASETS = ("tickets", "bugs", "docs", "citations", "configs", "hosts", "code", "revisions")
CORPUS_DATASETS = ("configs", "hosts", "code", "revisions", "threads", "logs")
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
CONTENT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def verifier(dataset):
    """The dataset's verify.py `shortfall` (failing checks; empty means PASS)."""
    path = os.path.join(run.DATASETS, dataset, "verify.py")
    spec = importlib.util.spec_from_file_location("verify_" + dataset, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.shortfall


def sweep_task(dataset):
    env = os.path.join(run.DATASETS, dataset, "env")
    tasks = run.load_tasks(env)
    shortfall = verifier(dataset)
    graded = {}

    def correct(tid, answer):
        key = (tid, json.dumps(answer, sort_keys=True))
        if key not in graded:
            graded[key] = not shortfall(tid, answer)
        return graded[key]

    def score(decide):
        reused = wrong = 0
        for task in tasks:
            answer = decide(task)
            if answer is not None:
                reused += 1
                wrong += not correct(task["tid"], answer)
        return reused, wrong

    def similarity_only(index):
        def decide(task):
            sig = index.minhash.of_text(index._text(task))
            hits = index.lsh.query(sig, index.threshold)
            return index.items[hits[0][0]][2] if hits else None
        return decide

    out = {"tasks": len(tasks), "sweep": []}
    for t in THRESHOLDS:
        gated = policy.TaskIndex(dataset, env, threshold=t)
        g_reused, g_wrong = score(gated.lookup)
        s_reused, s_wrong = score(similarity_only(gated))
        out["sweep"].append({"threshold": t, "gated_reused": g_reused, "gated_wrong": g_wrong,
                             "sim_reused": s_reused, "sim_wrong": s_wrong})
    calibrated = policy.TaskIndex(dataset, env)
    start = time.perf_counter()
    reused, wrong = score(calibrated.lookup)
    out["calibrated"] = {"threshold": calibrated.threshold, "reused": reused, "wrong": wrong,
                         "ms_per_lookup": 1000 * (time.perf_counter() - start) / len(tasks)}
    return out


def sweep_content(dataset):
    scenarios = os.path.join(run.DATASETS, dataset, "env", "scenarios")
    reads = []  # (scenario, path, output) in reading order
    for sid in sorted(os.listdir(scenarios)):
        root = os.path.join(scenarios, sid)
        read = {t.name: t for t in sp_tools.core_tools(root)}["read_file"]
        for base, _, names in sorted(os.walk(root)):
            for name in sorted(names):
                rel = os.path.relpath(os.path.join(base, name), root)
                reads.append((sid, rel, read.run(path=rel)))
    out = {"reads": len(reads), "chars": sum(len(o) for _, _, o in reads), "sweep": []}
    for t in CONTENT_THRESHOLDS:
        sent = folds = exact = 0
        start = time.perf_counter()
        cache, current = None, None
        for sid, rel, output in reads:
            if sid != current:  # one trajectory per scenario
                cache, current, full = content.ContentCache(threshold=t), sid, {}
            got = cache.process("read_file", {"path": rel}, output)
            sent += len(got)
            if got == output:
                full[rel] = content.lines_of(output)
            else:
                folds += 1
                exact += _rebuilds(got, full, content.lines_of(output))
        out["sweep"].append({"threshold": t, "sent": sent, "folds": folds, "exact": exact,
                             "saved_pct": 100.0 * (1 - sent / out["chars"]),
                             "ms_per_read": 1000 * (time.perf_counter() - start) / len(reads)})
    return out


def _rebuilds(fold, full, lines):
    """Rebuild a rendered [reuse] fold from its reference, as the model must read it."""
    head, *body = fold.split("\n")
    ref = head.split(" ")[3].rstrip(",") if "identical to" in head else head.split(" = ")[1].split(" (")[0]
    if "identical to" in head:
        return full[ref] == lines
    parts = []
    for line in body:
        if line.startswith("@@ "):
            old = line.split()[1][1:].split(",")
            start, count = int(old[0]) - 1, int(old[1])
            parts.append((start, start + count, []))
        elif line.startswith("+ "):
            parts[-1][2].append(line[2:])
    return content.apply(full[ref], parts) == lines


def main():
    report = {"minhash": {"num_perm": minhash.NUM_PERM, "shingle": minhash.SHINGLE},
              "task": {}, "content": {}}
    for dataset in TASK_DATASETS:
        report["task"][dataset] = sweep_task(dataset)
        c = report["task"][dataset]["calibrated"]
        print("task    %-10s calibrated t=%.2f reused %d wrong %d (%.2f ms/lookup)"
              % (dataset, c["threshold"], c["reused"], c["wrong"], c["ms_per_lookup"]))
    for dataset in CORPUS_DATASETS:
        report["content"][dataset] = sweep_content(dataset)
        d = next(s for s in report["content"][dataset]["sweep"] if s["threshold"] == 0.5)
        print("content %-10s t=0.5 saved %.0f%%, %d folds (%d rebuild exactly), %.1f ms/read"
              % (dataset, d["saved_pct"], d["folds"], d["exact"], d["ms_per_read"]))
    path = os.path.join(run.HERE, "results", "offline.json")
    with open(path, "w", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
