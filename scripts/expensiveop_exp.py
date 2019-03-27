#!/usr/bin/python3

import os
from common_exp import run_test

selectivities = [1] + range(0, 100, 5)
cpu_filter_values = [1]
gpu_values = [None, 1]

os.system('make')
os.system('mkdir -p results/expensiveop')

for s in [0, 10, 100, 1000, 10000]:
	for gpu in gpu_values:
		file = "expensiveop/results-expensiveop.csv".format(postfix)
		run_test(fname=file, selectivity="1", slowdown=s)

		file = "expensiveop/results-expensiveop.csv".format(postfix)
		run_test(fname=file, selectivity="5", slowdown=s)		