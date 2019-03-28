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
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator, FuncFormatter

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
hatches = ["//", "--", "\\\\", "xx", "||", "++"]


framework_columns = ["PipelineCycles", "PipelineSumThreadCycles", "PipelineTime", "CPUTime", "CPUJoinTime",
        "CPUExpOpTime", "GPUProbeTime", "CPUGPUTime", "PreFilterTuples", "FilteredTuples", "PreJoinTuples",
        "PostJoinTuples", "CPUBloomFilter", "FilterSize", "Slowdown", "CPUMorselSize", "GPUMorselsize",
        "Selectivity"]


result_path = "results/"

def plot_sel():
    cpu = pd.read_csv("{}/selectivity/results-selectivity_cpu.csv".format(result_path),
        sep='|', names=framework_columns, header=None, skiprows=1)
    gpu = pd.read_csv("{}/selectivity/results-selectivity_gpu.csv".format(result_path),
        sep='|', names=framework_columns, header=None, skiprows=1)

    cpu=cpu.sort_values(by=['Selectivity'])
    gpu=gpu.sort_values(by=['Selectivity'])

    cpu_filter = cpu[cpu['CPUBloomFilter']==1]
    cpu_nofilter = cpu[cpu['CPUBloomFilter']==0]

    (fig, ax1) = plt.subplots()

    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print gpu

    ofilename = "plot_sel.pgf"
    # plt.title("Breakdown for \\emph{{{}}}".format(wbname))
    # plt.xlabel("Query")

    ax1.set_ylabel('Time (in s)')
    ax1.set_xlabel('Selectivity (in \\%)')
    # ax1.grid(True)

    # ax1.plot(df['Selectivity'], df['PipelineCycles'], linestyle='--', marker='o', color=colors[0], label="Probe pipeline")
    #ax1.plot(filter0['Selectivity'], filter0['CPUJoinTime'], linestyle='--', marker='o', color=colors[1], label="CPU \fjoin, no Bloom filter")
    ax1.semilogy(cpu_nofilter['Selectivity'], cpu_nofilter['PipelineTime'], linestyle='--', marker='o', color=colors[1], label="CPU-only, no BF")
    ax1.semilogy(cpu_filter['Selectivity'], cpu_filter['PipelineTime'], linestyle='--', marker='x', color=colors[3], label="CPU only, BF")
    # PipelineSumThreadCycles

    #ax1.plot(filter1['Selectivity'], filter1['CPUJoinTime'], linestyle='--', marker='o', color=colors[3], label="CPU \fjoin, CPU filter")
    ax1.semilogy(gpu['Selectivity'], gpu['PipelineTime'], linestyle='--', marker='^', color=colors[2], label="GPU+CPU, BF")

    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax1.yaxis.get_major_formatter().set_useOffset(False)
    ax1.yaxis.set_minor_formatter(mticker.ScalarFormatter())

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

    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.xaxis.get_major_formatter().set_scientific(False)
    ax1.xaxis.get_major_formatter().set_useOffset(False)
    ax1.xaxis.set_minor_formatter(mticker.ScalarFormatter())

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


def plot_expensiveop(sel):
    cpu = pd.read_csv("{}/expensiveop/results-expensiveop_cpu.csv".format(result_path),
        sep='|', names=framework_columns, header=None, skiprows=1)
    gpu = pd.read_csv("{}/expensiveop/results-expensiveop_gpu.csv".format(result_path),
        sep='|', names=framework_columns, header=None, skiprows=1)

    cpu = cpu[cpu['Selectivity']==sel]
    gpu = gpu[gpu['Selectivity']==sel]

    (fig, ax1) = plt.subplots()

    cpu_filter = cpu[cpu['CPUBloomFilter']==1]
    cpu_nofilter = cpu[cpu['CPUBloomFilter']==0]

    ofilename = "plot_expensiveop_sel{}.pgf".format(sel)

    ax1.set_ylabel('Time (in s)')
    ax1.set_xlabel('Slowdown')
    # ax1.grid(True)

    ax1.semilogx(cpu_nofilter['Slowdown'], cpu_nofilter['PipelineTime'], linestyle='--', marker='o', color=colors[0], label="CPU, no BF")
    ax1.semilogx(cpu_filter['Slowdown'], cpu_filter['PipelineTime'], linestyle='--', marker='o', color=colors[1], label="CPU, BF")
    ax1.semilogx(gpu['Slowdown'], gpu['PipelineTime'], linestyle='--', marker='x', color=colors[2], label="GPU+CPU, BF")


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
    # plot_sel()
    # plot_bloomfilter()
    plot_expensiveop(1)
    plot_expensiveop(5)

if __name__ == '__main__':
    main()