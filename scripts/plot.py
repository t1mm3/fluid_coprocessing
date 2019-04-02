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


result_path = "results"

max_sel = 70
min_sel = 1

def df_filter_sel(df):
    df = df[df['Selectivity'] <= max_sel]
    return df[df['Selectivity'] >= min_sel]

def plot_sel():
    cpu = pd.read_csv("{}/selectivity/results-selectivity_cpu.csv".format(result_path),
        sep='|', names=framework_columns, header=None, skiprows=1)
    gpu = pd.read_csv("{}/selectivity/results-selectivity_gpu.csv".format(result_path),
        sep='|', names=framework_columns, header=None, skiprows=1)
    gpukeys = pd.read_csv("{}/selectivity/results-selectivity_gpukeys.csv".format(result_path),
        sep='|', names=framework_columns, header=None, skiprows=1)

    cpu = df_filter_sel(cpu)
    gpu = df_filter_sel(gpu)
    gpukeys = df_filter_sel(gpukeys)

    cpu=cpu.sort_values(by=['Selectivity'])
    gpu=gpu.sort_values(by=['Selectivity'])
    gpukeys=gpukeys.sort_values(by=['Selectivity'])

    cpu_filter = cpu[cpu['CPUBloomFilter']==1]
    cpu_nofilter = cpu[cpu['CPUBloomFilter']==0]

    gpu = gpu[gpu['CPUBloomFilter']==1]
    gpukeys = gpukeys[gpukeys['CPUBloomFilter']==1]

    (fig, ax1) = plt.subplots()

    #with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
    #    print gpu

    ofilename = "plot_sel.pgf"
    # plt.title("Breakdown for \\emph{{{}}}".format(wbname))
    # plt.xlabel("Query")

    ax1.set_ylabel('Time (in s)')
    ax1.set_xlabel('Selectivity (in \\%)')
    # ax1.grid(True)

    # ax1.plot(df['Selectivity'], df['PipelineCycles'], linestyle='--', marker='o', color=colors[0], label="Probe pipeline")
    #ax1.plot(filter0['Selectivity'], filter0['CPUJoinTime'], linestyle='--', marker='o', color=colors[1], label="CPU \fjoin, no Bloom filter")
    ax1.plot(cpu_nofilter['Selectivity'], cpu_nofilter['PipelineTime'], linestyle='--', marker='o', color=colors[0], label="CPU, no BF")
    ax1.plot(cpu_filter['Selectivity'], cpu_filter['PipelineTime'], linestyle='--', marker='x', color=colors[1], label="CPU, BF")
    # PipelineSumThreadCycles

    #ax1.plot(filter1['Selectivity'], filter1['CPUJoinTime'], linestyle='--', marker='o', color=colors[3], label="CPU \fjoin, CPU filter")
    ax1.plot(gpu['Selectivity'], gpu['PipelineTime'], linestyle='--', marker='^', color=colors[2], label="GPU+CPU, BF")
    ax1.plot(gpukeys['Selectivity'], gpukeys['PipelineTime'], linestyle='--', marker='+', color=colors[3], label="GPU+CPU, BF (cached)")

    if False:
        ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax1.yaxis.get_major_formatter().set_scientific(False)
        ax1.yaxis.get_major_formatter().set_useOffset(False)
        ax1.yaxis.set_minor_formatter(mticker.ScalarFormatter())

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    #legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
    #          fancybox=False, ncol=3)
    ax1.legend(loc='lower right', ncol=1)

    fig.tight_layout()
    #,legend2
    fig.savefig(ofilename, bbox_extra_artists=(), bbox_inches='tight')
    plt.close(fig)

def df_div(df, colA, colB):
    return df[colA] / df[colB]

def df_prejointuples(df):
    return df['FilteredTuples'].replace(0, 1.0*1024.0*1024.0*1024.0)

def df_joinspeed(df):
    return df['CPUJoinTime'] / df_prejointuples(df)


def plot_joinspeed():
    cpu = pd.read_csv("{}/selectivity/results-selectivity_cpu.csv".format(result_path),
        sep='|', names=framework_columns, header=None, skiprows=1)
    gpu = pd.read_csv("{}/selectivity/results-selectivity_gpu.csv".format(result_path),
        sep='|', names=framework_columns, header=None, skiprows=1)
    gpukeys = pd.read_csv("{}/selectivity/results-selectivity_gpukeys.csv".format(result_path),
        sep='|', names=framework_columns, header=None, skiprows=1)

    cpu = df_filter_sel(cpu)
    gpu = df_filter_sel(gpu)
    gpukeys = df_filter_sel(gpukeys)

    cpu=cpu.sort_values(by=['Selectivity'])
    gpu=gpu.sort_values(by=['Selectivity'])
    gpukeys=gpukeys.sort_values(by=['Selectivity'])

    cpu_filter = cpu[cpu['CPUBloomFilter']==1]
    cpu_nofilter = cpu[cpu['CPUBloomFilter']==0]

    gpu = gpu[gpu['CPUBloomFilter']==1]
    gpukeys = gpukeys[gpukeys['CPUBloomFilter']==1]

    (fig, ax1) = plt.subplots()

    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print(gpu)
    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print(cpu_filter)

    ofilename = "plot_joinspeed.pgf"
    # plt.title("Breakdown for \\emph{{{}}}".format(wbname))
    # plt.xlabel("Query")

    ax1.set_ylabel('Join probe time (in cycles/tuple)')
    ax1.set_xlabel('Selectivity (in \\%)')
    # ax1.grid(True)

    # ax1.plot(df['Selectivity'], df['PipelineCycles'], linestyle='--', marker='o', color=colors[0], label="Probe pipeline")
    #ax1.plot(filter0['Selectivity'], filter0['CPUJoinTime'], linestyle='--', marker='o', color=colors[1], label="CPU \fjoin, no Bloom filter")
    

    #ax1.plot(cpu_nofilter['Selectivity'], df_joinspeed(cpu_nofilter),
    #    linestyle='--', marker='o', color=colors[0], label="CPU, no BF")
    
    ax1.plot(cpu_filter['Selectivity'], df_joinspeed(cpu_filter),
        linestyle='--', marker='x', color=colors[1], label="CPU, BF")
    # PipelineSumThreadCycles

    #ax1.plot(filter1['Selectivity'], filter1['CPUJoinTime'], linestyle='--', marker='o', color=colors[3], label="CPU \fjoin, CPU filter")
    ax1.plot(gpu['Selectivity'], df_joinspeed(gpu),
        linestyle='--', marker='^', color=colors[2], label="GPU+CPU, BF")

    ax1.plot(gpukeys['Selectivity'], df_joinspeed(gpukeys),
        linestyle='--', marker='+', color=colors[3], label="GPU+CPU, BF (cached)")

    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax1.yaxis.get_major_formatter().set_useOffset(False)
    ax1.yaxis.set_minor_formatter(mticker.ScalarFormatter())

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    #legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
    #          fancybox=False, ncol=3)
    ax1.legend(loc='upper center', ncol=1)

    fig.tight_layout()
    #,legend2
    fig.savefig(ofilename, bbox_extra_artists=(), bbox_inches='tight')
    plt.close(fig)


def plot_bloomfilter():
    df = pd.read_csv("bloom_size.csv".format(result_path),
        sep='|', names=["NAME", "BFSIZE", "TPUT"], header=None, skiprows=0)

    cpu = df[df['NAME']=="CPU"]
    gpu = df[df['NAME']=="GPU-Naive"]
    #gpu_cluster = df[df['NAME']=="GPU-Clustering"]
    #gpu_cluster_only = df[df['NAME']=="GPU-Clustering_only"]

    (fig, ax1) = plt.subplots()

    #with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
    #    print gpu

    ofilename = "plot_bf.pgf"

    ax1.set_ylabel('Throughput (MProbe/s)')
    ax1.set_xlabel('Bloom filter size (MiB)')
    # ax1.grid(True)

    sz_div = mebi * 8.0
    tp_div = mega

    ax1.ticklabel_format(axis='x', style='plain')

    ax1.set_xlim(1, 2**9)

    ax1.loglog(cpu['BFSIZE'] / sz_div, cpu['TPUT'] / tp_div, linestyle='--', marker='o', color=colors[0], label="CPU", basex=2)
    ax1.loglog(gpu['BFSIZE'] / sz_div, gpu['TPUT'] / tp_div, linestyle='--', marker='x', color=colors[1], label="GPU", basex=2)
    #ax1.loglog(gpu_cluster['BFSIZE'] / sz_div, gpu_cluster['TPUT']  / tp_div, linestyle='--', marker='^', color=colors[2], label="GPU Radix", basex=2)
    #ax1.loglog(gpu_cluster_only['BFSIZE'] / sz_div, gpu_cluster_only['TPUT']  / tp_div, linestyle='--', marker='+', color=colors[3], label="GPU Radix (only)", basex=2)

    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.xaxis.get_major_formatter().set_scientific(False)
    ax1.xaxis.get_major_formatter().set_useOffset(False)
    ax1.xaxis.set_minor_formatter(mticker.ScalarFormatter())

    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax1.yaxis.get_major_formatter().set_useOffset(False)
    ax1.yaxis.set_minor_formatter(mticker.ScalarFormatter())

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    # legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
    #          fancybox=False, ncol=3)
    ax1.legend(loc='lower left', ncol=1)

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
    ax1.set_xlabel('Additional Pipeline Cost $c_A$')
    # ax1.grid(True)
    #with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
    #    print cpu_nofilter

    ax1.plot(cpu_nofilter['Slowdown'], cpu_nofilter['PipelineTime'], linestyle='--', marker='o', color=colors[0], label="CPU, no BF")
    ax1.plot(cpu_filter['Slowdown'], cpu_filter['PipelineTime'], linestyle='--', marker='x', color=colors[1], label="CPU, BF")
    ax1.plot(gpu['Slowdown'], gpu['PipelineTime'], linestyle='--', marker='^', color=colors[2], label="GPU+CPU, BF")

    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.xaxis.get_major_formatter().set_scientific(False)
    ax1.xaxis.get_major_formatter().set_useOffset(False)
    ax1.xaxis.set_minor_formatter(mticker.ScalarFormatter())


    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    #legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
    #          fancybox=False, ncol=1)
    ax1.legend(loc='upper left', ncol=1)

    fig.tight_layout()
    #,legend2
    fig.savefig(ofilename, bbox_extra_artists=(), bbox_inches='tight')
    plt.close(fig)

from pandas import DataFrame

def plot_heatmap(sel, file, rbar, lbar, cpubf):
    cpu = pd.read_csv("{p}/op_vs_bfsize/results-op_vs_bfsize{file}.csv".format(
        p=result_path, file=file),
        sep='|', names=framework_columns, header=None, skiprows=1)
    #gpu = pd.read_csv("{}/op_vs_bfsize/results-op_vs_bfsizegpu.csv".format(result_path),
    #    sep='|', names=framework_columns, header=None, skiprows=1)
    (fig, ax1) = plt.subplots()

    ax1.set_aspect(1.0)

    cpu = cpu[cpu['Selectivity']==sel]

    if cpubf:
        cpu = cpu[cpu['CPUBloomFilter']==1]
    else:
        cpu = cpu[cpu['CPUBloomFilter']==0]


    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print(cpu)

    if not cpubf:
        file = file + "_nocpubf"
    ofilename = "plot-heat{sel}-{file}.pgf".format(sel=sel, file=file)

    #Index= ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
    #Cols = ['A', 'B', 'C', 'D']
    #df = DataFrame(abs(np.random.randn(5, 4)), index=Index, columns=Cols)


    cpu['NormalizedPipelineTime'] = cpu['PipelineTime'] / ((cpu['FilterSize'] / (8.0 * 1024))) * 3.0 * 1000.0 * 1000.0 # * 1000.0 * 1000.0 * 1000.0 * 1000.0
    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print(cpu)    

    df = pd.pivot_table(cpu, values="NormalizedPipelineTime",index=["FilterSize"], columns=["Slowdown"], fill_value=0)
    # df = cpu.pivot("FilterSize", "Slowdown", "PipelineTime")

    c = plt.pcolor(df, cmap="plasma", vmin=0.5, vmax=900)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)

    ax1.set_xlabel("Additonal Pipeline Cost $c_A$")

    if not rbar:
        plt.setp(ax1.get_yticklabels(), visible=False)
    else:
        ax1.set_ylabel("Inner Relation Cardinality (MiTuples)")
        l = [8, 16, 32, 64,
            128, 256, 512, 1024,
            2*1024, 4*1024]
        p = ["1", "2", "4", "8",
            "16", "32", "64", "128",
            "256", "512"]
        ax1.set_yticklabels(p)

    if lbar:
        fig.colorbar(c, ax=ax1)

    fig.tight_layout()
    #,legend2
    fig.savefig(ofilename, bbox_extra_artists=(), bbox_inches='tight')
    plt.close(fig)

def main():
    mpl.rcParams.update({'font.size': 15})
    plot_sel()
    plot_joinspeed()
    plot_bloomfilter()

    for sel in [1]: #, 5]:
        for file in ["cpu", "gpu"]: #, "gpuonly"]:
            right = file == "cpu"
            left = file == "gpu"
            cpubf = True
            if file == "gpuonly":
                file = "gpu"
                cpubf = False

            plot_heatmap(sel, file, right, left, cpubf)


    plot_expensiveop(1)
    plot_expensiveop(5)

if __name__ == '__main__':
    main()
