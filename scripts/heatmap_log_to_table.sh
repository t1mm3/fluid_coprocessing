#!/bin/bash
# Arguments:
#  1. Heatmap.log file
grep -e "--tw" $1 | awk '{printf("%s%s | tail -n2\n", $0, " --print_bf_conf=1")}' | bash 