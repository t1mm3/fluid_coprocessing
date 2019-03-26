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

kibi = 1024.0
mebi = kibi*1024.0
gibi = mebi*1024.0

kilo = 1000.0
mega = kilo * 1000.0
giga = mega * 1000.0

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math
from matplotlib.ticker import MultipleLocator, FuncFormatter

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
hatches = ["//", "--", "\\\\", "xx", "||", "++"]


framework_columns = ["PipelineCycles", "PipelineSumThreadCycles", "PipelineTime", "CPUTime", "CPUJoinTime", "CPUExpOpTime", "GPUProbeTime",
        "CPUGPUTime", "PreFilterTuples", "FilteredTuples", "PreJoinTuples" , "PostJoinTuples", "CPUBloomFilter", "FilterSize", "Slowdown", "Selectivity"]
result_path = "results/"

def plot_sel(slowdown, filter_size):
    df = pd.read_csv("{}/selectivity/results-selectivity_cpu.csv".format(result_path),
        sep='|', names=framework_columns, header=None, skiprows=1)

    df = df[df['Slowdown']==slowdown]
    df = df[df['FilterSize']==filter_size]

    filter0 = df[df['CPUBloomFilter']==0]
    filter1 = df[df['CPUBloomFilter']==1]

    (fig, ax1) = plt.subplots()

    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print df

    ofilename = "plot_sel_sd{}_fs{}.pgf".format(slowdown, filter_size)
    # plt.title("Breakdown for \\emph{{{}}}".format(wbname))
    # plt.xlabel("Query")

    ax1.set_ylabel('Time')
    ax1.set_xlabel('Selectivity (in \\%)')
    # ax1.grid(True)

    # ax1.plot(df['Selectivity'], df['PipelineCycles'], linestyle='--', marker='o', color=colors[0], label="Probe pipeline")
    #ax1.plot(filter0['Selectivity'], filter0['CPUJoinTime'], linestyle='--', marker='o', color=colors[1], label="CPU \fjoin, no Bloom filter")
    ax1.plot(filter0['Selectivity'], filter0['PipelineSumThreadCycles'], linestyle='--', marker='o', color=colors[1], label="Probe pipeline, no CPU filter")

    #ax1.plot(filter1['Selectivity'], filter1['CPUJoinTime'], linestyle='--', marker='o', color=colors[3], label="CPU \fjoin, CPU filter")
    ax1.plot(filter1['Selectivity'], filter1['PipelineSumThreadCycles'], linestyle='--', marker='o', color=colors[2], label="Probe pipeline CPU filter")

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=False, ncol=2)

    fig.tight_layout()
    #,legend2
    fig.savefig(ofilename, bbox_extra_artists=(), bbox_inches='tight')
    plt.close(fig)



def plot_bloomfilter():
    df = pd.read_csv("bloom_size.csv".format(result_path),
        sep='|', names=["NAME", "BFSIZE", "TPUT"], header=None, skiprows=0)

    cpu = df[df['NAME']=="CPU"]
    gpu = df[df['NAME']=="GPU-Naive"]
    gpu_cluster = df[df['NAME']=="GPU-Clustering"]

    (fig, ax1) = plt.subplots()

    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print gpu

    ofilename = "plot_bf.pgf"
    # plt.title("Breakdown for \\emph{{{}}}".format(wbname))
    # plt.xlabel("Query")

    ax1.set_ylabel('Throughput (MProbe/s)')
    ax1.set_xlabel('Bloom filter size (MiB)')
    # ax1.grid(True)

    sz_div = mebi * 8.0
    tp_div = mega

    ax1.ticklabel_format(axis='x', style='plain')

    ax1.set_xlim(1, 2**9)

    ax1.loglog(cpu['BFSIZE'] / sz_div, cpu['TPUT'] / tp_div, linestyle='--', marker='o', color=colors[0], label="CPU", basex=2)
    ax1.loglog(gpu['BFSIZE'] / sz_div, gpu['TPUT'] / tp_div, linestyle='--', marker='o', color=colors[1], label="GPU Naive", basex=2)
    ax1.loglog(gpu_cluster['BFSIZE'] / sz_div, gpu_cluster['TPUT']  / tp_div, linestyle='--', marker='o', color=colors[2], label="GPU Radix", basex=2)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=False, ncol=3)

    fig.tight_layout()
    #,legend2
    fig.savefig(ofilename, bbox_extra_artists=(), bbox_inches='tight')
    plt.close(fig)

def main():
    mpl.rcParams.update({'font.size': 15})

    if False:
        plot_sel(0, 1*mebi)
        plot_sel(0, 64*mebi*8)
        if False:
            plot_sel(1024, 64*mebi*8)
            plot_sel(1024, 1*mebi)

    plot_bloomfilter()

if __name__ == '__main__':
    main()