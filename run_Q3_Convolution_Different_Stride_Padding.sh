#!/usr/bin/env bash
set -euo pipefail

# Run the Q3 convolution solution.
if [[ -x "/usr/local/bin/python3.11" ]]; then
  PY="/usr/local/bin/python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "Python 3 not found in PATH." >&2
  exit 1
fi

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$WORKDIR/Q3_Convolution_Different_Stride_Padding.py"
LOGFILE="$WORKDIR/q3_convolution_run.log"
PIDFILE="$WORKDIR/q3_convolution.pid"

# Install TensorFlow if it is missing from the selected Python environment.
if ! "$PY" -c "import numpy, tensorflow" >/dev/null 2>&1; then
  echo "NumPy or TensorFlow is not installed. Installing dependencies..."
  "$PY" -m pip install numpy tensorflow
fi

# Keep the same background execution pattern as the other homework questions.
nohup "$PY" "$SCRIPT" "$@" > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

echo "Started Q3 convolution operations -> log: $LOGFILE (pid $(cat "$PIDFILE"))"
echo "To watch output: cat \"$LOGFILE\""
echo "To stop: kill \$(cat \"$PIDFILE\") || true"
