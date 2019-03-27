#!/usr/bin/python3
import os
from common_exp import run_test

selectivities = list(range(0, 100, 5))
selectivities.append(1)
cpu_filter_values = [0, 1]
gpu_values = [None, 1]


os.system('make')
os.system('mkdir -p results/selectivity')

for cpu_filter in cpu_filter_values:
	for gpu in gpu_values:
		if gpu == 1 and cpu_filter == 0:
			continue

		postfix = "gpu" if gpu is not None else "cpu"
		for selectivity in selectivities:
			file = "selectivity/results-selectivity_{}.csv".format(postfix)
			run_test(fname=file, selectivity=selectivity, gpu_devices=gpu,
				cpu_filter=cpu_filter)
