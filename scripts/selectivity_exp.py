#!/usr/bin/python3
import os
from common_exp import run_test

selectivities = list(range(0, 100, 5))
selectivities.append(1)
cpu_filter_values = [0, 1]
gpu_values = [None, 1]


os.system('make')
os.system('mkdir -p results/selectivity')

# only keys on GPU
only_keys_on_gpu = True

if only_keys_on_gpu:
	gpu = 1
	cpu_filter = 1
	postfix = "cpu"
	if gpu is not None:
		postfix = "gpu"
	if keys_on_gpu:
		postfix = postfix + "keys"

	for selectivity in selectivities:
		file = "selectivity/results-selectivity_{}.csv".format(postfix)
		run_test(fname=file, selectivity=selectivity, gpu_devices=gpu,
			cpu_filter=cpu_filter, keys_on_gpu=keys_on_gpu)



if not only_keys_on_gpu
	for keys_on_gpu in [True, None]:
		for cpu_filter in cpu_filter_values:
			for gpu in gpu_values:
				if gpu == 1 and cpu_filter == 0:
					continue

				if gpu is None and keys_on_gpu is not None:
					continue

				postfix = "cpu"
				if gpu is not None:
					postfix = "gpu"
				if keys_on_gpu:
					postfix = postfix + "keys"

				for selectivity in selectivities:
					file = "selectivity/results-selectivity_{}.csv".format(postfix)
					run_test(fname=file, selectivity=selectivity, gpu_devices=gpu,
						cpu_filter=cpu_filter, keys_on_gpu=keys_on_gpu)

