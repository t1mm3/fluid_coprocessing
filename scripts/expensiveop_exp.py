#!/usr/bin/python3

import os
from common_exp import run_test

cpu_filter_values = [0, 1]
gpu_values = [None, 1]

os.system('make')
os.system('mkdir -p results/expensiveop')

for cpu_filter in cpu_filter_values:
	for gpu in gpu_values:
		if gpu == 1 and cpu_filter == 0:
			continue

		postfix = "gpu" if gpu is not None else "cpu"

		for s in [0, 5, 10, 25, 50, 75, 100]:
			file = "expensiveop/results-expensiveop_{}.csv".format(postfix)
			run_test(fname=file, selectivity="1", slowdown=s, gpu_devices=gpu,
				cpu_filter=cpu_filter)

			file = "expensiveop/results-expensiveop_{}.csv".format(postfix)
			run_test(fname=file, selectivity="5", slowdown=s, gpu_devices=gpu,
				cpu_filter=cpu_filter)		