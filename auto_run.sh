#!/bin/sh
# Unattended driver for the free tier (500 requests/day): run_all.sh at --limit 10,
# retrying every 30 minutes after a stop (daily quota).
# Usage: nohup caffeinate -i sh auto_run.sh [run.py flags, e.g. --rpm 0] >> results/run_all.log 2>&1 &
cd "$(dirname "$0")"
for limit in 10; do
    until sh run_all.sh --limit "$limit" "$@"; do
        echo "=== $(date '+%m-%d %H:%M') stopped; retrying in 30 min"
        sleep 1800
    done
    osascript -e "display notification \"limit $limit: all 8 datasets x 4 configs done\" with title \"COMP 580 A3 experiment\" sound name \"Glass\""
done
