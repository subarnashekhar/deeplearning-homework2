#!/usr/bin/env bash
set -euo pipefail

# Run Question 4 Task 1: Sobel edge detection.
if [[ -x "/usr/local/bin/python3.11" ]]; then
  PY="/usr/local/bin/python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "Python 3 not found in PATH." >&2
  exit 1
fi

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$WORKDIR/Q4_Task1_Sobel_Edge_Detection.py"
LOGFILE="$WORKDIR/q4_task1_sobel_run.log"
PIDFILE="$WORKDIR/q4_task1_sobel.pid"

# Install compatible packages needed for image loading and Sobel filtering.
# TensorFlow 2.16 on Python 3.11 requires NumPy below version 2. OpenCV 4.10
# avoids the NumPy 2 requirement used by newer OpenCV wheels.
if ! "$PY" -c "import cv2, numpy; assert tuple(map(int, numpy.__version__.split('.')[:2])) < (2, 0); assert tuple(map(int, cv2.__version__.split('.')[:2])) < (4, 11)" >/dev/null 2>&1; then
  echo "Installing compatible OpenCV and NumPy dependencies..."
  "$PY" -m pip install --force-reinstall --no-deps 'numpy==1.26.4' 'opencv-python==4.10.0.84'
fi

# A background process cannot receive the keypress required by cv2.waitKey.
# Save images without opening windows; direct Python execution can display them.
nohup "$PY" "$SCRIPT" --no-display "$@" > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

echo "Started Q4 Task 1 Sobel edge detection -> log: $LOGFILE (pid $(cat "$PIDFILE"))"
echo "To watch output: cat \"$LOGFILE\""
echo "To stop: kill \$(cat \"$PIDFILE\") || true"
