#!/usr/bin/env bash
set -euo pipefail

# Run the Q2 IMDB sentiment classification solution.
if [[ -x "/usr/local/bin/python3.11" ]]; then
  PY="/usr/local/bin/python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "Python 3 not found in PATH." >&2
  exit 1
fi

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$WORKDIR/Q2_Implement_RNN_For_Sentiment_Classification.py"
LOGFILE="$WORKDIR/q2_rnn_sentiment_classification_run.log"
PIDFILE="$WORKDIR/q2_rnn_sentiment_classification.pid"

# Install missing packages so the script can be started from a clean setup.
if ! "$PY" -c "import tensorflow, sklearn" >/dev/null 2>&1; then
  echo "TensorFlow or scikit-learn is not installed. Installing dependencies..."
  "$PY" -m pip install tensorflow scikit-learn
fi

# Run in the background so the terminal remains available during training.
nohup "$PY" "$SCRIPT" "$@" > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

echo "Started Q2 IMDB sentiment classification -> log: $LOGFILE (pid $(cat "$PIDFILE"))"
echo "To watch output: tail -f \"$LOGFILE\""
echo "To stop: kill \$(cat \"$PIDFILE\") || true"
