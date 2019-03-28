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


def plot_bloomfilter():
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

    ofilename = "plot_bf_sort.pgf"

    ax1.set_ylabel('Throughput (GProbe/s)')
    ax1.set_xlabel('Sorted bits')
    # ax1.grid(True)

    sz_div = mebi * 8.0
    tp_div = giga

    ax1.ticklabel_format(axis='x', style='plain')

    ax1.set_xlim(1, 32, auto=True)

    ax1.loglog(bf16['bits to sort'] , bf16['Total throughput'] / tp_div, linestyle='--', marker='o', color=colors[0], label="BF Size 16MiB", basex=2)
    #ax1.loglog(bf32['bits to sort'] , bf32['Total throughput'] / tp_div, linestyle='--', marker='o', color=colors[1], label="BF Size 32MiB", basex=2)
    ax1.loglog(bf64['bits to sort'] , bf64['Total throughput'] / tp_div, linestyle='--', marker='o', color=colors[2], label="BF Size 64MiB", basex=2)
    #ax1.loglog(bf128['bits to sort'] , bf128['Total throughput'] / tp_div, linestyle='--', marker='o', color=colors[3], label="BF Size 128MiB", basex=2)
    #ax1.loglog(bf256['bits to sort'] , bf256['Total throughput'] / tp_div, linestyle='--', marker='o', color=colors[4], label="BF Size 256MiB", basex=2)
    ax1.loglog(bf512['bits to sort'] , bf512['Total throughput'] / tp_div, linestyle='--', marker='o', color=colors[5], label="BF Size 512MiB", basex=2)



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

def main():
    mpl.rcParams.update({'font.size': 15})
    plot_bloomfilter()

if __name__ == '__main__':
    main()