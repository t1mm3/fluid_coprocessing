#!/bin/env python2

import matplotlib as mpl
mpl.use('pgf')
pgf_with_pgflatex = {
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         # r"\usepackage{cmbright}",
         ]
}
mpl.rcParams.update(pgf_with_pgflatex)
mpl.rcParams['axes.axisbelow'] = True

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.figure import figaspect

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
hatches = ["//", "--", "\\\\", "xx", "||", "++"]
markers = ['o', 'x', '^', '+']

import matplotlib.patches as mpatches

def tag2color(t):
    if t == "CPUJOIN":
        return colors[1]
    elif t == "CPUWORK":
        return colors[0]
    elif t == "SCHEDPROBE":
        return colors[2]
    else:
        return colors[3]

def tagIsGpu(t):
    return t == "SCHEDPROBE" or t == "FINISHPROBE"


def plot_timeline(num, tfile, ofile):
    times = pd.read_csv(tfile,
        sep='|', names=["ID", "TABS", "TREL", "NAME", "OFFSET", "NUM_TUPLES", "PROBE_PTR"], header=None)

    w, h = figaspect(0.5)
    (fig, ax1) = plt.subplots(figsize=(w,h))

    ax1.set_ylabel('Worker')
    ax1.set_xlabel('Time (in cycles)')


    for tid in range(0, num):
        df = times[times['ID'] == tid]
        df = df.sort_values('TABS')

        lengths = df[['TABS']]

        lengths = lengths.diff(periods=-1)

        lengths["LEN"] = -lengths["TABS"]
        #with pd.option_context('display.max_rows', 10, 'display.max_columns', 100):
        #    print lengths

        tuples = df.join(lengths, lsuffix='', rsuffix='_l')

        plot_tuples = []
        prev_name = ""
        first_tabs = 0
        sum_len = 0

        facecolors = []

        for (tabs, tlen, name) in list(zip(tuples["TABS"], tuples["LEN"], tuples["NAME"])):
            if prev_name == name:
                # aggregate
                sum_len = sum_len + tlen
            else:
                if len(prev_name) > 0:
                    plot_tuples.append((first_tabs, sum_len))
                    facecolors.append(tag2color(prev_name))
                prev_name = name
                first_tabs = tabs
                sum_len = tlen

        if len(prev_name) > 0:
            plot_tuples.append((first_tabs, sum_len))
            facecolors.append(tag2color(prev_name))

        # print(plot_tuples)

        ax1.broken_barh(plot_tuples, ( tid + 0.5, 0.5), facecolors=facecolors)



    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    NA = mpatches.Patch(color=tag2color("CPUJOIN"), label='Join')
    EU = mpatches.Patch(color=tag2color("CPUWORK"), label='CPU Pipeline')
    AP = mpatches.Patch(color=tag2color("SCHEDPROBE"), label='Schedule GPU Probe')
    SA = mpatches.Patch(color=tag2color("OTHER"), label='Other')
    plt.legend(handles=[NA,EU,AP,SA], loc='upper center', ncol=4)

    ax1.set_ylim(0, num + 1)


    # Put a legend below current axis
    #legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
    #          fancybox=False, ncol=1)
    # ax1.legend(loc='upper left', ncol=1)

    fig.tight_layout()
    #,legend2
    fig.savefig(ofile, bbox_extra_artists=(), bbox_inches='tight')
    plt.close(fig)


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int)
    parser.add_argument("--outfile")
    parser.add_argument("--infile")
    args = parser.parse_args()

    num = 10
    infile = "time.csv"
    outfile = "time.pgf"

    if args.num is not None:
        num = args.num
    if args.outfile is not None:
        outfile = args.outfile
    if args.infile is not None:
        infile = args.infile


    mpl.rcParams.update({'font.size': 15})
    plot_timeline(num, infile, outfile)


if __name__ == '__main__':
    main()