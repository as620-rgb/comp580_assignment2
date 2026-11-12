# MinHash for Efficient Agent Trajectories

## The three implementation files

| File | What it does |
|---|---|
| `minhash.py` | 3-word shingles, 128-function MinHash (implemented here), banded LSH index |
| `policy.py` | Task-level reuse: validation-key gate + per-dataset threshold calibrated on `history.jsonl` + abstain on disagreement; indexes new answers only after `verify.py` passes |
| `content.py` | Content-level reuse: folds a near-duplicate tool output into an exact line diff marked `[reuse]`, or returns the original |

Supporting scripts: `run.py` (experiment runner), `run_all.sh` / `auto_run.sh` (all configurations),
`analyze.py` (tables), `offline.py` (threshold sweep and content upper bound, no API calls),
`plots.py` (figures).

## Setup

Python 3.12; the code uses only the standard library (matplotlib only for the figures).

```sh
git clone <this repository> a2 && cd a2
git clone https://github.com/sahiljoshi515/Spartacus.git
git clone https://github.com/sahiljoshi515/Assignment2_COMP_580.git

conda create -n comp580_a2 python=3.12 -y
conda activate comp580_a2
pip install matplotlib

echo 'GEMINI_API_KEY=your-key' > .env    # read by run.py; git-ignored
```

The two cloned repositories must sit in the repository root as `Spartacus/` and
`Assignment2_COMP_580/`; `run.py` finds them there.

## Reproduce the experiments

All runs use `gemini-3.5-flash-lite`, the same prompt, and the first 10 incoming tasks of each of
the 8 datasets (tickets, bugs, docs, citations, configs, hosts, code, revisions).

```sh
# 1. Agent runs: 4 configurations x 8 datasets (baseline, task, content, both).
#    Paid tier, no client-side pacing:
sh run_all.sh --limit 10 --rpm 0
#    Free tier (15 requests/min, 500/day): paces itself and resumes after the daily reset.
nohup caffeinate -i sh auto_run.sh >> results/run_all.log 2>&1 &

# 2. Tables: results/summary_limit10.json (and the comparison table on stdout)
python analyze.py --limit 10

# 3. Offline analyses, no API calls (~1 min): results/offline.json
python offline.py

# 4. Figures: figures/fig1..fig5
python plots.py --limit 10
```

Single runs, for example one dataset and one configuration:

```sh
python run.py --datasets tickets --config baseline --limit 10
python run.py --datasets configs --config both --limit 5 --keep          # keep the work dirs
python run.py --datasets tickets --config task --task-opt threshold=0.5  # fixed threshold
python run.py --datasets hosts --config content --content-opt show_old=true
```

`run.py` options: `--config {baseline,task,content,both}`, `--limit N` (first N tasks in `seq`
order), `--rpm` (model requests per minute, 0 = no pacing), `--resume` (skip tasks already in the
results file), `--task-opt` / `--content-opt KEY=VALUE` (passed to `TaskIndex` / `ContentCache`),
`--max-turns` (default 40), `--model`.

## Outputs

- `results/<config>/<dataset>.jsonl`: one line per task with pass/fail, verifier output, the answer,
  tool calls (with arguments), model requests, input/output tokens, completion time (includes
  MinHash overhead; excludes rate-limit waits and verification), reuse match and similarity, and
  tool-output characters before/after content folding.
- `results/summary_limit10.json`, `results/offline.json`: aggregates behind the report's tables.
- `figures/`: the report's figures.
