#!/bin/zsh
# Double-click for a solo test at home: the app exactly as on the day, but recording the raw Wi-Fi data
# (red RECORDING banner) into bench/solo_<date>/ so it can be analysed afterwards. No names are recorded.
cd "$(dirname "$0")"
exec ./run_openday.command --record "bench/solo_$(date +%Y%m%d)"
