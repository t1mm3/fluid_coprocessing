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
from matplotlib.ticker import MultipleLocator, FuncFormatter

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
hatches = ["//", "--", "\\\\", "xx", "||", "++"]


result_path = "results/"

def plot_sel():
    df = pd.read_csv("{}/selectivity/results-selectivity_cpu.csv".format(result_path, ), sep='|',
        names=["PipelineCycles", "PipelineSumThreadCycles", "PipelineTime", "CPUJoinTime", "GPUProbeTime",
        "CPUGPUTime", "PreFilterTuples", "FilteredTuples", "PreJoinTuples" , "PostJoinTuples", "Selectivity"],
        header=None, skiprows=1)

    (fig, ax1) = plt.subplots()

    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print df

    ofilename = "plot_sel.pgf"
    # plt.title("Breakdown for \\emph{{{}}}".format(wbname))
    # plt.xlabel("Query")

    ax1.set_ylabel('Time')
    ax1.set_xlabel('Selectivity (in \\%)')
    # ax1.grid(True)

    ax1.plot(df['Selectivity'], df['PipelineCycles'], linestyle='--', marker='o', color=colors[0], label="Probe pipeline")
    ax1.plot(df['Selectivity'], df['CPUJoinTime'], linestyle='--', marker='o', color=colors[1], label="CPU \fjoin")
    ax1.plot(df['Selectivity'], df['PipelineSumThreadCycles']/16.0, linestyle='--', marker='o', color=colors[2], label="Avg worker")

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # ax1.set_ylim([0, 100])  

    # Put a legend below current axis
    legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=False, ncol=2)

    fig.tight_layout()
    #,legend2
    fig.savefig(ofilename, bbox_extra_artists=(), bbox_inches='tight')
    plt.close(fig)



def main():
    mpl.rcParams.update({'font.size': 15})

    plot_sel()

if __name__ == '__main__':
    main()