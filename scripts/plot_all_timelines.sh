#!/bin/bash
echo "Sel 1"
scripts/plot_timeline.py --infile timeline_sel1.time  --num=10 --outfile timeline_sel1.pgf
#scripts/plot_timeline.py --infile timeline_sel1_4w.time  --num=10 --outfile timeline_sel1_4w.pgf

echo "Sel 5"
scripts/plot_timeline.py --infile timeline_sel5.time  --num=10 --outfile timeline_sel5.pgf
scripts/plot_timeline.py --infile timeline_sel5_4w.time  --num=10 --outfile timeline_sel5_4w.pgf

echo "Sel 10"
scripts/plot_timeline.py --infile timeline_sel10.time  --num=10 --outfile timeline_sel10.pgf
# scripts/plot_timeline.py --infile timeline_sel10_4w.time  --num=10 --outfile timeline_sel10_4w.pgf
