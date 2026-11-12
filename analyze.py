"""Summarise results/<config>/<dataset>.jsonl into the comparison tables of the report.

Usage: python analyze.py [--limit 10] [--out results]
Only the first `limit` tasks of each dataset (in seq order) are counted, so every
configuration is compared on exactly the same tasks.
"""

import argparse
import json
import os

import run

CONFIGS = ("baseline", "task", "content", "both")
DATASETS = ("tickets", "bugs", "docs", "citations", "configs", "hosts", "code", "revisions")
METRICS = ("tool_calls", "model_requests", "input_tokens", "seconds")


def load(out, config, dataset, limit):
    """{tid: row} for the first `limit` tasks, or None when the run is incomplete."""
    path = os.path.join(out, config, dataset + ".jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        rows = {row["tid"]: row for row in map(json.loads, handle)}
    tids = [t["tid"] for t in run.load_tasks(os.path.join(run.DATASETS, dataset, "env"))[:limit]]
    return [rows[tid] for tid in tids] if all(tid in rows for tid in tids) else None


def totals(rows):
    t = {m: sum(r[m] for r in rows) for m in METRICS}
    t["tasks"] = len(rows)
    t["passed"] = sum(r["passed"] for r in rows)
    t["reused"] = sum(r["reused"] for r in rows)
    t["reused_wrong"] = sum(r["reused"] and not r["passed"] for r in rows)
    t["overhead_s"] = sum(r["policy_s"] + r["content_s"] for r in rows)
    t["chars_raw"] = sum(r["chars_raw"] for r in rows)
    t["chars_sent"] = sum(r["chars_sent"] for r in rows)
    return t


def table(results, datasets):
    head = "| dataset | config | acc | reused (wrong) | tool calls | requests | input tokens | time (s) | MinHash overhead (ms) |"
    lines = [head, "|---" * (head.count("|") - 1) + "|"]
    for dataset in datasets + ["ALL"]:
        base = results[("baseline", dataset)]
        for config in CONFIGS:
            t = results[(config, dataset)]
            delta = lambda m: "" if config == "baseline" or not base[m] else \
                " (%+.0f%%)" % (100.0 * (t[m] - base[m]) / base[m])
            lines.append("| %s | %s | %d/%d | %d (%d) | %d%s | %d%s | %d%s | %.0f%s | %.1f |" % (
                dataset if config == "baseline" else "", config, t["passed"], t["tasks"],
                t["reused"], t["reused_wrong"], t["tool_calls"], delta("tool_calls"),
                t["model_requests"], delta("model_requests"), t["input_tokens"],
                delta("input_tokens"), t["seconds"], delta("seconds"), 1000 * t["overhead_s"]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", default=os.path.join(run.HERE, "results"))
    args = parser.parse_args(argv)
    results, missing = {}, []
    for dataset in DATASETS:
        for config in CONFIGS:
            rows = load(args.out, config, dataset, args.limit)
            if rows is None:
                missing.append("%s/%s" % (config, dataset))
            else:
                results[(config, dataset)] = rows
    done = [d for d in DATASETS if all((c, d) in results for c in CONFIGS)]
    for config in CONFIGS:
        results[(config, "ALL")] = sum((results[(config, d)] for d in done), [])
    summary = {key: totals(rows) for key, rows in results.items()}
    print(table(summary, done))
    if missing:
        print("\nincomplete (left out):", ", ".join(missing))
    with open(os.path.join(args.out, "summary_limit%d.json" % args.limit), "w", encoding="utf-8") as out:
        json.dump({"%s/%s" % key: t for key, t in summary.items()}, out, indent=2)


if __name__ == "__main__":
    main()
