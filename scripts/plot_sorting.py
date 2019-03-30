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


framework_columns = ["Bloom Filter size (MiB)"," Block size (bytes)", "bits to sort", "Probe size", "Hash time (ms)", "Sort time (ms)", "Probe time (ms)", "Total throughput"]
result_path = "results/"


def plot_sorting_throughput():
    df = pd.read_csv("{}/bench_bits.csv".format(result_path), sep=';', usecols=['Bloom filter size (MiB)','bits to sort','Total throughput'])
    print(df)
    bf16 = df[df['Bloom filter size (MiB)']==16]
    #bf32 = df[df['Bloom filter size (MiB)']==32]
    bf64 = df[df['Bloom filter size (MiB)']==64]
   # bf128 = df[df['Bloom filter size (MiB)']==128]
    #bf256 = df[df['Bloom filter size (MiB)']==256]
    bf512 = df[df['Bloom filter size (MiB)']==512]
    print(bf16)

    (fig, ax1) = plt.subplots()

    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print bf16

    ofilename = "plot_bf_sort_throughput.pgf"

    ax1.set_ylabel('Throughput (GProbe/s)')
    ax1.set_xlabel('Sorted bits')
    # ax1.grid(True)

    sz_div = mebi * 8.0
    tp_div = giga

    ax1.ticklabel_format(axis='x', style='plain')

    ax1.set_xlim(1, 32, auto=True)

    ax1.loglog(bf16['bits to sort'] , bf16['Total throughput'] / tp_div, linestyle='--', marker='o', color=colors[0], label="BF Size 16MiB", basex=2)
    #ax1.loglog(bf32['bits to sort'] , bf32['Total throughput'] / tp_div, linestyle='--', marker='o', color=colors[1], label="BF Size 32MiB", basex=2)
    ax1.loglog(bf64['bits to sort'] , bf64['Total throughput'] / tp_div, linestyle='--', marker='x', color=colors[1], label="BF Size 64MiB", basex=2)
    #ax1.loglog(bf128['bits to sort'] , bf128['Total throughput'] / tp_div, linestyle='--', marker='o', color=colors[3], label="BF Size 128MiB", basex=2)
    #ax1.loglog(bf256['bits to sort'] , bf256['Total throughput'] / tp_div, linestyle='--', marker='o', color=colors[4], label="BF Size 256MiB", basex=2)
    ax1.loglog(bf512['bits to sort'] , bf512['Total throughput'] / tp_div, linestyle='--', marker='^', color=colors[2], label="BF Size 512MiB", basex=2)



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


def plot_sorting_time():
    df = pd.read_csv("{}/bench_bits.csv".format(result_path), sep=';', usecols=['Bloom filter size (MiB)','bits to sort','Sort time (ms)', 'Probe time (ms)', 'Total throughput'])
    bf16 = df[df['Bloom filter size (MiB)']==16]
    bf64 = df[df['Bloom filter size (MiB)']==64]
    bf512 = df[df['Bloom filter size (MiB)']==512]
    sort = df[df['Bloom filter size (MiB)']==512]

    (fig, ax1) = plt.subplots(2, 1,sharex=True)
    fig_size = fig.get_figheight()
    fig.set_figheight(fig_size * 1.5)
    with pd.option_context('display.max_rows', None, 'display.max_columns', 100):
        print bf16

    ofilename = "plot_bf_sort_time.pgf"

    ax1[1].set_ylabel('Time (ms)')
    ax1[1].set_xlabel('Sorted bits')
    ax1[1].grid(True)
    ax1[0].set_ylabel('Throughput (GProbe/s)')
    ax1[0].grid(True)

    sz_div = mebi * 8.0
    tp_div = giga

    ax1[1].set_xlim(1, 35)
    ax1[0].set_xlim(1, 35)
    ax1[1].set_ylim(10, 350)
    ax1[1].xaxis.set_ticks(np.arange(0, 33, 8))
    ax1[0].xaxis.set_ticks(np.arange(0, 33, 8))

    ax1[1].semilogy(bf16['bits to sort'] , bf16['Probe time (ms)'] , linestyle='--', marker='o', color=colors[0], label="BF Size 16MiB")
    ax1[1].semilogy(bf64['bits to sort'] , bf64['Probe time (ms)'] , linestyle='--', marker='x', color=colors[1], label="BF Size 64MiB")
    ax1[1].semilogy(bf512['bits to sort'] , bf512['Probe time (ms)'] , linestyle='--', marker='^', color=colors[2], label="BF Size 512MiB")
    ax1[1].semilogy(sort['bits to sort'] , sort['Sort time (ms)'] , linestyle='--', marker='+', color=colors[3], label="Sorting")


    ax1[0].semilogy(bf16['bits to sort'] , bf16['Total throughput'] / tp_div, linestyle='--', marker='o', color=colors[0], label="BF Size 16MiB")
    ax1[0].semilogy(bf64['bits to sort'] , bf64['Total throughput'] / tp_div, linestyle='--', marker='x', color=colors[1], label="BF Size 64MiB")
    ax1[0].semilogy(bf512['bits to sort'] , bf512['Total throughput'] / tp_div, linestyle='--', marker='^', color=colors[2], label="BF Size 512MiB")

    ax1[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1[0].yaxis.get_major_formatter().set_scientific(False)
    ax1[0].yaxis.get_major_formatter().set_useOffset(False)
    ax1[0].yaxis.set_minor_formatter(mticker.ScalarFormatter())

    ax1[1].yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1[1].yaxis.get_major_formatter().set_scientific(False)
    ax1[1].yaxis.get_major_formatter().set_useOffset(False)
    ax1[1].yaxis.set_minor_formatter(mticker.ScalarFormatter())

    # Put a legend below current axis
    handles, labels = ax1[1].get_legend_handles_labels()
    plt.legend( handles, labels, loc = 'lower center', bbox_to_anchor = (0,-0.1,1,1),ncol=2,
            bbox_transform = plt.gcf().transFigure )

    fig.tight_layout()
    #,legend2
    fig.savefig(ofilename, bbox_extra_artists=(), bbox_inches='tight')
    plt.close(fig)

def main():
    mpl.rcParams.update({'font.size': 15})
    plot_sorting_throughput()
    plot_sorting_time()

if __name__ == '__main__':
    main()
