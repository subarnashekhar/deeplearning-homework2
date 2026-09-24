#!/usr/bin/env bash
set -euo pipefail

# Run Question 5 Task 1: simplified AlexNet model summary.
if [[ -x "/usr/local/bin/python3.11" ]]; then
  PY="/usr/local/bin/python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "Python 3 not found in PATH." >&2
  exit 1
fi

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$WORKDIR/Q5_Task1_AlexNet.py"
LOGFILE="$WORKDIR/q5_task1_alexnet_run.log"
PIDFILE="$WORKDIR/q5_task1_alexnet.pid"

if ! "$PY" -c "import tensorflow" >/dev/null 2>&1; then
  echo "TensorFlow is not installed. Installing it now..."
  "$PY" -m pip install tensorflow
fi

nohup "$PY" "$SCRIPT" "$@" > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

echo "Started Q5 Task 1 AlexNet summary -> log: $LOGFILE (pid $(cat "$PIDFILE"))"
echo "To watch output: cat \"$LOGFILE\""
echo "To stop: kill \$(cat \"$PIDFILE\") || true"
