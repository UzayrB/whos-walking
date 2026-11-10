#!/bin/zsh
# Double-click to start the Open Day demo (live, full screen). Extra arguments pass through, e.g. --windowed.
# caffeinate keeps the Mac and its screen awake for as long as the app runs.
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/caffeinate -dims /opt/anaconda3/bin/python3 -m openday live "$@"
echo
echo "The app has stopped. Close this window, or double-click run_openday.command to start again."
