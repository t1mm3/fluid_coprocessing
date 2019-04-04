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


def amdahl_speedup(tseq, tpar, npar):
    ttot = tseq + tpar
    timp = tseq + (tpar / npar)

    return ttot / timp

def amdahl_fraction_par(frac, npar):
    return amdahl_speedup(1.0 - frac, frac, npar)

markers = ['o', 'x', '^', '+']

def plot_amdahl(crazy):
    if not crazy:
        npars = [5, 10, 20]
        percent = range(0, 100, 5)
    else:
        npars = [50, 100]
        percent = range(80, 100, 1)

    
    percent.append(100)

    fractions = list(map(lambda x: x/100.0, percent))

    ofilename = "plot_amdahl1{}.pgf".format("crazy" if crazy else "")

    (fig, ax1) = plt.subplots()

    ax1.set_ylabel('Maximal Speedup')
    ax1.set_xlabel('Tuples processed on Accelerator (in \\%s)')

    for i, npar in enumerate(npars):
        mu = list(map(lambda x: amdahl_fraction_par(x, npar), fractions))

        ax1.plot(percent, mu, linestyle='--', marker=markers[i], color=colors[i], label="{}$\\times$".format(npar))

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

def main():
    mpl.rcParams.update({'font.size': 20})

    for crazy in [False, True]:
        plot_amdahl(crazy)

    mpl.rcParams.update({'font.size': 15})

if __name__ == '__main__':
    main()
