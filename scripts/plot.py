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
        "Selectivity", "Streams", "TuplesGpuProbe", "TuplesGpuConsume"]


result_path = "results"

max_sel = 70
min_sel = 0

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


framework_columns2 = ["PipelineCycles", "PipelineSumThreadCycles", "PipelineTime", "CPUTime", "CPUJoinTime",
        "CPUExpOpTime", "GPUProbeTime", "CPUGPUTime", "PreFilterTuples", "FilteredTuples", "PreJoinTuples",
        "PostJoinTuples", "CPUBloomFilter", "FilterSize", "Slowdown", "CPUMorselSize", "GPUMorselsize",
        "Selectivity", "NumStreams", "GPUConsumed", "GPUProduced"]


def frac_tuples_gpu(df):
    return (100.0 * df['GPUConsumed']) / (1024.0*1024.0*1024.0)

def plot_utilization(cached, runtime):
    # keys on gpu
    if cached:
        gpu = pd.read_csv("{}/op_vs_bfsize/results-stream_newnot_gpu.csv".format(result_path),
            sep='|', names=framework_columns2, header=None, skiprows=1)
    else:
        gpu = pd.read_csv("{}/op_vs_bfsize/results-stream_newgpu.csv".format(result_path),
            sep='|', names=framework_columns2, header=None, skiprows=1)

    gpu = gpu.sort_values(['Selectivity'], ascending=[True])

    gpu = gpu[gpu['CPUBloomFilter']==1]
    gpu = df_filter_sel(gpu)


    gpu1 = gpu[gpu['NumStreams']==1]
    gpu2 = gpu[gpu['NumStreams']==2]
    gpu4 = gpu[gpu['NumStreams']==4]
    gpu8 = gpu[gpu['NumStreams']==8]
    #gpu = gpu[gpu['CPUBloomFilter']==cpubf]
    #gpukeys = gpukeys[gpukeys['CPUBloomFilter']==cpubf]

    (fig, ax1) = plt.subplots()

    #with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
    #    print gpukeys8

    ofilename = "plot_streams{}{}.pgf".format(
        "_cached" if cached else "",
        "_time" if runtime else "")
    # plt.title("Breakdown for \\emph{{{}}}".format(wbname))
    # plt.xlabel("Query")

    if runtime:
        ax1.set_ylabel('Time (in s)')
        # ax1.set_ylim(40, 100)
    else:
        ax1.set_ylabel('GPU Utilization (in \\%)')
        ax1.set_ylim(40, 100)
    ax1.set_xlabel('Selectivity (in \\%)')
    # ax1.set_xlim(0, 70)

    # ax2 = ax1.twinx()
    # ax1.grid(True)

    # ax1.plot(df['Selectivity'], df['PipelineCycles'], linestyle='--', marker='o', color=colors[0], label="Probe pipeline")
    #ax1.plot(filter0['Selectivity'], filter0['CPUJoinTime'], linestyle='--', marker='o', color=colors[1], label="CPU \fjoin, no Bloom filter")

    if runtime:
        ax1.plot(gpu1['Selectivity'], gpu1["PipelineTime"], linestyle='--', marker='o', color=colors[0], label="1 Stream")
        ax1.plot(gpu2['Selectivity'], gpu2["PipelineTime"], linestyle='--', marker='x', color=colors[1], label="2 Streams")
        ax1.plot(gpu4['Selectivity'], gpu4["PipelineTime"], linestyle='--', marker='^', color=colors[2], label="4 Streams")
        ax1.plot(gpu8['Selectivity'], gpu8["PipelineTime"], linestyle='--', marker='+', color=colors[3], label="8 Streams")
    else:
        ax1.plot(gpu1['Selectivity'], frac_tuples_gpu(gpu1), linestyle='--', marker='o', color=colors[0], label="1 Stream")
        ax1.plot(gpu2['Selectivity'], frac_tuples_gpu(gpu2), linestyle='--', marker='x', color=colors[1], label="2 Streams")
        ax1.plot(gpu4['Selectivity'], frac_tuples_gpu(gpu4), linestyle='--', marker='^', color=colors[2], label="4 Streams")
        ax1.plot(gpu8['Selectivity'], frac_tuples_gpu(gpu8), linestyle='--', marker='+', color=colors[3], label="8 Streams")

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    #legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
    #          fancybox=False, ncol=3)
    

    if runtime:
        ax1.legend(loc='upper left', ncol=1)
    else:
        ax1.legend(loc='lower right', ncol=1)

    #ax2.legend(loc='lower left', ncol=1)

    fig.tight_layout()
    #,legend2
    fig.savefig(ofilename, bbox_extra_artists=(), bbox_inches='tight')
    plt.close(fig)

def plot_streams(cpubf, fraction):
    # keys on gpu
    gpukeys = pd.read_csv("{}/op_vs_bfsize/results-stream_newnot_gpu.csv".format(result_path),
        sep='|', names=framework_columns2, header=None, skiprows=1)
    gpu = pd.read_csv("{}/op_vs_bfsize/results-stream_newgpu.csv".format(result_path),
        sep='|', names=framework_columns2, header=None, skiprows=1)


    gpu = gpu[gpu['CPUBloomFilter']==cpubf]
    gpukeys = gpukeys[gpukeys['CPUBloomFilter']==cpubf]

    (fig, ax1) = plt.subplots()

    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print gpu

    ofilename = "plot_streams_cpubf{}_frac{}.pgf".format(cpubf, fraction)
    # plt.title("Breakdown for \\emph{{{}}}".format(wbname))
    # plt.xlabel("Query")

    if fraction:
        ax1.set_ylabel('Tuples processed on GPU (in \\%)')
        ax1.set_ylim(0, 100)
    else:
        ax1.set_ylabel('Time (in s)')
    ax1.set_xlabel('\\#Streams')

    # ax2 = ax1.twinx()
    # ax1.grid(True)

    # ax1.plot(df['Selectivity'], df['PipelineCycles'], linestyle='--', marker='o', color=colors[0], label="Probe pipeline")
    #ax1.plot(filter0['Selectivity'], filter0['CPUJoinTime'], linestyle='--', marker='o', color=colors[1], label="CPU \fjoin, no Bloom filter")

    if fraction:
        ax1.plot(gpu['NumStreams'], frac_tuples_gpu(gpu), linestyle='--', marker='+', color=colors[0], label="GPU+CPU")
        ax1.plot(gpukeys['NumStreams'], frac_tuples_gpu(gpukeys), linestyle='--', marker='x', color=colors[1], label="GPU+CPU (cached)  ")
    else:
        ax1.plot(gpu['NumStreams'], gpu['PipelineTime'], linestyle='--', marker='+', color=colors[0], label="GPU+CPU")
        ax1.plot(gpukeys['NumStreams'], gpukeys['PipelineTime'], linestyle='--', marker='x', color=colors[1], label="GPU+CPU (cached)  ")


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
    

    if fraction:
        ax1.legend(loc='lower right', ncol=1)
    else:
        ax1.legend(loc='center right', ncol=1)

    #ax2.legend(loc='lower left', ncol=1)

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
        print(gpukeys)

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

    pointA = (10, 510)
    pointB = (10, 610)
    ax1.annotate('', xytext=pointA, xy=pointB,
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.5), size=15, horizontalalignment='right', verticalalignment='top',
            )
    ax1.annotate('', xytext=pointB, xy=pointA,
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.5), size=15, horizontalalignment='right', verticalalignment='top',
            )

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
    #arrow with text
    ax1.annotate('', xytext=(256, 300), xy=(250, 1400),
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.5), size=15, horizontalalignment='right', verticalalignment='top',
            )
    ax1.annotate('6x', xy=(170, 600),)
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

    cpu['NormalizedPipelineTime'] = cpu['PipelineTime']

    invalid = np.nan
    if not lbar:
        # Mask Data because some points do not make sense on our hardware...
        # This happens when the query with BF would be slower than without
        for index, row in cpu.iterrows():
            # Hard coded values
            if row['Slowdown'] == 0:
                if row['FilterSize'] / 8 >= 8 * 1024 * 1024:
                    cpu.at[index, 'NormalizedPipelineTime'] = invalid
                    print("yes")

            if row['Slowdown'] == 50:
                if row['FilterSize'] / 8 >= 256 * 1024 * 1024:
                    cpu.at[index, 'NormalizedPipelineTime'] = invalid
                    print("yes")
        # cpu = np.ma.masked_invalid(cpu)

    if False:
        with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
            print(cpu)    

    df = pd.pivot_table(cpu, values="NormalizedPipelineTime",index=["FilterSize"], columns=["Slowdown"], fill_value=invalid)

    print(df)

    # df = cpu.pivot("FilterSize", "Slowdown", "PipelineTime")

    cmap = mpl.cm.plasma
    cmap.set_bad('white',1.)

    df = np.ma.masked_invalid(df)

    #if cpubf:
    #    exit(1)

    # Plot heatmap
    c = plt.pcolor(df, cmap=cmap, vmin=0.5, vmax=20)

    p = ["1", "2", "4", "8",
            "16", "32", "64", "128",
            "256", "512"]
    plt.yticks(np.arange(0.5, len(p), 1), p)

    cols = ["0", "50", "100", "200", "400", "800"]
    plt.xticks(np.arange(0.5, len(cols), 1), cols)

    ax1.set_xlabel("Additional Pipeline Cost $c_A$")

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

import copy

def autofix_unit(x):
    units = [
            (1024*1024*1024, "{div}~Gi"),
            (1024*1024, "{div}~Mi"),
            (1024, "{div}~Ki")
        ]

    flt = False

    txt = x.get_text()
    try:
        for scale, label in units:
            if flt:
                num = float(txt)
            else:
                num = int(txt)

            if num >= scale:
                x.set_text(label.format(div=num / scale))
                return x
    except ValueError:
        # Cannot cast
        return x

    # Not found
    return x


def morselsize_labels(labels):
    return list(map(lambda x: autofix_unit(x), labels))


def plot_morselsizes():
    coproc = pd.read_csv("{p}/morsel/results-morsel.csv".format(
        p=result_path),
        sep='|', names=framework_columns, header=None, skiprows=1)
    #gpu = pd.read_csv("{}/op_vs_bfsize/results-op_vs_bfsizegpu.csv".format(result_path),
    #    sep='|', names=framework_columns, header=None, skiprows=1)
    (fig, ax1) = plt.subplots()

    ax1.set_aspect(1.0)

    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print(coproc)

    ofilename = "plot-morsel.pgf"

    #Index= ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
    #Cols = ['A', 'B', 'C', 'D']
    #df = DataFrame(abs(np.random.randn(5, 4)), index=Index, columns=Cols)

    coproc['NormalizedPipelineTime'] = coproc['PipelineTime']
    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print(coproc)    

    df = pd.pivot_table(coproc, values="NormalizedPipelineTime",index=["CPUMorselSize"], columns=["GPUMorselsize"], fill_value=0)
    # df = cpu.pivot("FilterSize", "Slowdown", "PipelineTime")

    c = plt.pcolor(df, cmap="plasma")
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)

    plt.xticks(rotation=30)

    locs, labels = plt.yticks()
    labels = morselsize_labels(labels)
    ax1.set_yticklabels(labels)

    locs, labels = plt.xticks()
    labels = morselsize_labels(labels)
    #for label in labels:
    #    txt = int(label.get_text()) / 1024
    #    label.set_text("{}".format(txt))
    ax1.set_xticklabels(labels)

    ax1.set_ylabel("CPU morsel size")
    ax1.set_xlabel("GPU morsel size")

    fig.colorbar(c, ax=ax1, orientation='horizontal')

    fig.tight_layout()
    #,legend2
    fig.savefig(ofilename, bbox_extra_artists=(), bbox_inches='tight')
    plt.close(fig)

def main():
    print("PLOT SEL")
    mpl.rcParams.update({'font.size': 15})

    print("1")
    plot_joinspeed()
    print("2")
    plot_sel()


    mpl.rcParams.update({'font.size': 20})

    for cached in [True, False]:
        for time in [True, False]:
            plot_utilization(cached, time)

    mpl.rcParams.update({'font.size': 15})
    print("3")
    plot_bloomfilter()
    print("4")


    print("PLOT HEATMAP")
    for sel in [1]: #, 5]:
        mpl.rcParams.update({'font.size': 15})
        for file in ["cpu", "gpu"]: #, "gpuonly"]:
            right = file == "cpu"
            left = file == "gpu"
            cpubf = True
            if file == "gpuonly":
                file = "gpu"
                cpubf = False
            plot_heatmap(sel, file, right, left, cpubf)

    exit(0)
    print("PLOT STREAMS")
    mpl.rcParams.update({'font.size': 20})

    if False:
        for cpubf in [1]: 
            for frac in [True, False]:
                print("PLOT {} {}".format(cpubf, frac))
                plot_streams(cpubf, frac)

    exit(0)

    if False:
        plot_expensiveop(1)
        plot_expensiveop(5)

    mpl.rcParams.update({'font.size': 10})

    plot_morselsizes()

if __name__ == '__main__':
    main()
