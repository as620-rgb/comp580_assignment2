"""Figures for the report, from results/ (run.py, analyze.py) and results/offline.json.

Usage: python plots.py [--limit 10]     # writes figures/*.png
"""

import argparse
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import analyze  # noqa: E402
import run  # noqa: E402

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
COLORS = {"baseline": "#a3a29d", "task": "#2a78d6", "content": "#eb6834", "both": "#1baf7a"}
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3, validated all-pairs

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8, "font.size": 9, "lines.linewidth": 2,
})


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_tradeoff(summary, out):
    """Efficiency vs correctness, all datasets pooled."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    base = summary["baseline/ALL"]
    for config in analyze.CONFIGS:
        t = summary["%s/ALL" % config]
        x = 100.0 * t["input_tokens"] / base["input_tokens"]
        y = 100.0 * t["passed"] / t["tasks"]
        ax.scatter(x, y, s=70, color=COLORS[config], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate("%s\n%.0f%% tokens, %d/%d correct" % (config, x, t["passed"], t["tasks"]),
                    (x, y), xytext=(8, -4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(40, 115)
    ax.set_ylim(80, 100)
    ax.set_xlabel("input tokens (% of baseline)  ← cheaper")
    ax.set_ylabel("verified accuracy (%)")
    ax.set_title("Efficiency vs. correctness (%d tasks per config)" % base["tasks"], loc="left")
    save(fig, out, "fig1_tradeoff.png")


def fig_datasets(summary, out):
    """Input tokens per dataset relative to baseline."""
    datasets = [d for d in analyze.DATASETS if "baseline/%s" % d in summary]
    configs = ("task", "content", "both")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.26
    for k, config in enumerate(configs):
        xs = [i + (k - 1) * width for i in range(len(datasets))]
        ys = [100.0 * summary["%s/%s" % (config, d)]["input_tokens"] /
              summary["baseline/%s" % d]["input_tokens"] for d in datasets]
        ax.bar(xs, ys, width, color=COLORS[config], edgecolor=SURFACE, linewidth=1, label=config)
        for x, y in zip(xs, ys):
            ax.text(x, y + 2, "%.0f" % y, ha="center", fontsize=6.5, color=INK_2)
    ax.axhline(100, color=INK_2, linewidth=1, linestyle="--")
    ax.text(len(datasets) - 0.5, 102, "baseline = 100", ha="right", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set_ylabel("input tokens (% of baseline)")
    ax.set_title("Input tokens by dataset (first 10 tasks each)", loc="left")
    ax.legend(ncol=3, loc="upper left")
    fig.text(0.01, -0.04, "Per-dataset totals over 10 tasks are dominated by run-to-run variance "
             "(temperature 0.4, no seed); content folds removed only 0-10% of tool output (Fig. 5).",
             fontsize=7, color=INK_2)
    save(fig, out, "fig2_datasets.png")


def fig_threshold(offline, out):
    """Reuse decisions vs similarity threshold, with and without the validation key."""
    datasets = list(offline["task"])
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.8), sharex=True)
    for ax, dataset in zip(axes.flat, datasets):
        data = offline["task"][dataset]
        ts = [s["threshold"] for s in data["sweep"]]
        ax.plot(ts, [s["gated_reused"] - s["gated_wrong"] for s in data["sweep"]],
                color=SERIES[0], label="correct reuses (key + similarity)")
        ax.plot(ts, [s["gated_wrong"] for s in data["sweep"]],
                color=SERIES[1], label="wrong reuses (key + similarity)")
        ax.plot(ts, [s["sim_wrong"] for s in data["sweep"]],
                color=SERIES[2], linestyle="--", label="wrong reuses (similarity only)")
        cal = data["calibrated"]["threshold"]
        ax.axvline(cal, color=INK_2, linewidth=1, linestyle=":")
        ax.text(cal + 0.02, ax.get_ylim()[1] * 0.97, "calibrated %.2f" % cal, fontsize=6.5,
                color=INK_2, va="top", bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
        ax.set_title("%s (%d tasks)" % (dataset, data["tasks"]), loc="left", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("similarity threshold")
    for ax in axes[:, 0]:
        ax.set_ylabel("tasks")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Task-level reuse vs. similarity threshold (history index, all incoming tasks)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "fig3_threshold.png")


def fig_overhead(results, out):
    """MinHash overhead per task vs agent time saved per reused task (both config)."""
    datasets, overhead, saved = [], [], []
    for dataset in analyze.DATASETS:
        both, base = results.get(("both", dataset)), results.get(("baseline", dataset))
        if not both or not base:
            continue
        base_by = {r["tid"]: r for r in base}
        reused = [base_by[r["tid"]]["seconds"] for r in both if r["reused"]]
        datasets.append(dataset)
        overhead.append(statistics.mean(r["policy_s"] + r["content_s"] for r in both))
        saved.append(statistics.mean(reused) if reused else 0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.38
    for k, (ys, label, color) in enumerate(((overhead, "MinHash overhead per task", SERIES[0]),
                                            (saved, "agent time saved per reused task", SERIES[1]))):
        xs = [i + (k - 0.5) * width for i in range(len(datasets))]
        ax.bar(xs, ys, width, color=color, edgecolor=SURFACE, linewidth=1, label=label)
        for x, y in zip(xs, ys):
            if y:
                ax.text(x, y * 1.15, "%.0f ms" % (1000 * y) if y < 1 else "%.0f s" % y,
                        ha="center", fontsize=6.5, color=INK_2)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 200)
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set_ylabel("seconds (log scale)")
    ax.set_title("Computational overhead vs. savings (both config)", loc="left")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.28))
    save(fig, out, "fig4_overhead.png")


def fig_content(offline, results, out):
    """Content-level savings: offline upper bound vs what the live agent triggered."""
    datasets = list(offline["content"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    width = 0.38
    bound = [next(s for s in offline["content"][d]["sweep"] if s["threshold"] == 0.5)["saved_pct"]
             for d in datasets]
    live = []
    for d in datasets:
        rows = results.get(("content", d))
        raw = sum(r["chars_raw"] for r in rows) if rows else 0
        live.append(100.0 * (1 - sum(r["chars_sent"] for r in rows) / raw) if raw else None)
    for k, (ys, label, color) in enumerate(((bound, "upper bound: every file read once", SERIES[0]),
                                            (live, "live agent (content config)", SERIES[1]))):
        xs = [i + (k - 0.5) * width for i in range(len(datasets))]
        for x, y in zip(xs, ys):
            if y is None:
                ax.text(x, 1, "not run", ha="center", fontsize=6.5, color=MUTED, rotation=90)
                continue
            ax.bar(x, y, width, color=color, edgecolor=SURFACE, linewidth=1,
                   label=label if x == xs[0] else None)
            ax.text(x, y + 1.5, "%.0f%%" % y, ha="center", fontsize=6.5, color=INK_2)
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set_ylabel("tool-output characters saved (%)")
    ax.set_ylim(0, 75)
    ax.set_title("Content-level reuse: potential vs. realised", loc="left")
    ax.legend(loc="upper right")
    save(fig, out, "fig5_content.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    res = os.path.join(run.HERE, "results")
    out = os.path.join(run.HERE, "figures")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(res, "summary_limit%d.json" % args.limit), encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(os.path.join(res, "offline.json"), encoding="utf-8") as handle:
        offline = json.load(handle)
    results = {}
    for config in analyze.CONFIGS:
        for dataset in set(analyze.DATASETS) | set(offline["content"]):
            rows = analyze.load(res, config, dataset, args.limit)
            if rows:
                results[(config, dataset)] = rows
    fig_tradeoff(summary, out)
    fig_datasets(summary, out)
    fig_threshold(offline, out)
    fig_overhead(results, out)
    fig_content(offline, results, out)


if __name__ == "__main__":
    main()
