#!/bin/sh
# Reproduce every experiment: 4 configurations x 8 datasets, in seq order.
# Usage: sh run_all.sh [extra run.py flags], e.g. sh run_all.sh --limit 20
# Dataset by dataset, so whatever has finished is a complete 4-way comparison.
# Safe to rerun after a stop (quota, network): --resume skips finished tasks,
# and a larger --limit later extends each dataset from where it ended.
set -e
DATASETS="tickets bugs docs citations configs hosts code revisions"
for dataset in $DATASETS; do
    for config in baseline task content both; do
        python run.py --datasets "$dataset" --config "$config" --resume "$@"
    done
done
