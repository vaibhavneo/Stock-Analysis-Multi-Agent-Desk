#!/bin/bash
# Daily heartbeat wrapper for cron.
#
# WHY A WRAPPER, NOT A RAW CRON LINE
#   launchd runs with a minimal environment and an unpredictable working
#   directory. This pins both, so the job behaves the same as an interactive
#   run instead of failing in ways only a scheduled context can produce.
#
# LOCATION MATTERS. This repo must NOT live under ~/Desktop, ~/Documents or
#   ~/Downloads: macOS TCC denies scheduled processes access to those, and the
#   failure is silent-ish (exit 126, "Operation not permitted"). Verified by
#   experiment - an identical LaunchAgent ran fine from outside those folders
#   and failed inside them.
#
# TIMING — once per trading day, AFTER the close, never intraday.
#   An intraday run would freeze a prediction against a price that is still
#   moving, so two runs the same day would disagree about price_at_call and
#   the ledger would hold two versions of "today's call". Running after the
#   close gives one clean observation point per day, which is exactly what the
#   independence calculation depends on.
#
# PORTABLE TIMESTAMPS. macOS ships BSD date, which has no `-Is`; the explicit
#   format string below works on both BSD and GNU. The first real cron run
#   logged "date: invalid argument 's' for -I" until this was fixed.
#
# NO API KEY REQUIRED. The forecast path is deterministic and keyless
#   (verified with DEEPSEEK_API_KEY/ANTHROPIC_API_KEY/OPENAI_API_KEY all unset),
#   so no secret is placed in the cron environment.

set -uo pipefail

# Self-locating: derive the repo from this script's own path rather than
# hardcoding it. The previous hardcoded ~/Desktop path broke when the repo had
# to be moved out of Desktop (macOS TCC denies scheduled jobs access to
# Desktop/Documents/Downloads), and a hardcoded path would break again on the
# next move. Resolves symlinks so `launchctl` invoking a linked copy still works.
SCRIPT="${BASH_SOURCE[0]}"
while [ -L "$SCRIPT" ]; do SCRIPT="$(readlink "$SCRIPT")"; done
REPO="$(cd "$(dirname "$SCRIPT")" && pwd -P)"
PY="/usr/bin/python3"
LOG_DIR="$REPO/logs"
LOG="$LOG_DIR/heartbeat-$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"
cd "$REPO" || { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') FATAL: cannot cd to $REPO" >>"$LOG_DIR/heartbeat-error.log"; exit 2; }

{
  echo "===== heartbeat start $(date '+%Y-%m-%dT%H:%M:%S%z') ====="
  "$PY" run_heartbeat.py --file watchlist.txt
  rc=$?
  echo "===== heartbeat end   $(date '+%Y-%m-%dT%H:%M:%S%z')  exit=$rc ====="
  # A non-zero exit is surfaced in a separate always-checked file, because a
  # silent cron failure is worse than a loud one.
  if [ $rc -ne 0 ]; then
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') heartbeat exited $rc — see $LOG" >>"$LOG_DIR/heartbeat-error.log"
  fi
  exit $rc
} >>"$LOG" 2>&1
