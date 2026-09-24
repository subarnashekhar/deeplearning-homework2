#!/usr/bin/env bash
set -euo pipefail

# Run the Q1 LSTM text generation solution.
if [[ -x "/usr/local/bin/python3.11" ]]; then
  PY="/usr/local/bin/python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "Python 3 not found in PATH." >&2
  exit 1
fi

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$WORKDIR/Q1_Implement_RNN_For_Text_Generation.py"
LOGFILE="$WORKDIR/q1_rnn_text_generation_run.log"
PIDFILE="$WORKDIR/q1_rnn_text_generation.pid"

# TensorFlow is installed only if it is missing. Installation errors are shown
# later by Python, instead of making this launcher fail in a confusing way.
if ! "$PY" -c "import tensorflow" >/dev/null 2>&1; then
  echo "TensorFlow is not installed. Installing it now..."
  "$PY" -m pip install tensorflow
fi

# Run in the background so training can continue while the terminal is free.
nohup "$PY" "$SCRIPT" "$@" > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

echo "Started Q1 LSTM text generation -> log: $LOGFILE (pid $(cat "$PIDFILE"))"
echo "To watch output: tail -f \"$LOGFILE\""
echo "To stop: kill \$(cat \"$PIDFILE\") || true"