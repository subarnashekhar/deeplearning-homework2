#!/usr/bin/env bash
set -euo pipefail

# Run Question 4 Task 2: max pooling and average pooling.
if [[ -x "/usr/local/bin/python3.11" ]]; then
  PY="/usr/local/bin/python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "Python 3 not found in PATH." >&2
  exit 1
fi

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$WORKDIR/Q4_Task2_Pooling_Operations.py"
LOGFILE="$WORKDIR/q4_task2_pooling_run.log"
PIDFILE="$WORKDIR/q4_task2_pooling.pid"

# TensorFlow provides the Keras pooling layers used by this demonstration.
if ! "$PY" -c "import numpy, tensorflow" >/dev/null 2>&1; then
  echo "NumPy or TensorFlow is not installed. Installing dependencies..."
  "$PY" -m pip install 'numpy<2' tensorflow
fi

# Run in the background like the other homework launchers.
nohup "$PY" "$SCRIPT" "$@" > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

echo "Started Q4 Task 2 pooling operations -> log: $LOGFILE (pid $(cat "$PIDFILE"))"
echo "To watch output: cat \"$LOGFILE\""
echo "To stop: kill \$(cat \"$PIDFILE\") || true"
