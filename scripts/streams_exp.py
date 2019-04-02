#!/usr/bin/python3

import os
from common_exp import run_test
from common_exp import default_filter_size

cpu_filter_values = [0,1]
gpu_values = [None, 1]

streams = range(1,36,1)
default_probe_scale = 64
keys_on_gpu = [0,1]

os.system('make')
os.system('mkdir -p results/op_vs_bfsize')

for stream in streams:
	for key_on_gpu in keys_on_gpu:
		for cpu_filter in cpu_filter_values:
			for gpu in gpu_values:
				postfix = "gpu" if gpu is not None else "cpu"
				file = "op_vs_bfsize/results-stream{}.csv".format(postfix)
				run_test(fname=file, selectivity="1", keys_on_gpu=key_on_gpu, gpu_devices=gpu, cpu_filter=cpu_filter, streams=stream)