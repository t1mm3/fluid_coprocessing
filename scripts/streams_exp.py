#!/usr/bin/python3

import os
from common_exp import run_test
from common_exp import default_filter_size

cpu_filter_values = [0,1]
gpu_values = [None,1]

streams = [1, 2, 4, 6, 8, 16]
keys_on_gpu = [0,1]
selectivities = list(range(0, 100, 5))
selectivities.append(1)
cpu_filter = 1
gpu = 1


os.system('make')
os.system('mkdir -p results/op_vs_bfsize')

for select in selectivities:
	for stream in streams:
		for key_on_gpu in keys_on_gpu:
			postfix = "gpu" if key_on_gpu == 0 else "not_gpu"
			file = "op_vs_bfsize/results-stream_new{}.csv".format(postfix)
			run_test(fname=file, selectivity=select, keys_on_gpu=key_on_gpu, gpu_devices=1, cpu_filter=cpu_filter, streams=stream)