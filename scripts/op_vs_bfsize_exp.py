#!/usr/bin/python3

import os
from common_exp import run_test
from common_exp import default_filter_size

cpu_filter_values = [1]
gpu_values = [None, 1]

filter_sizes = [default_filter_size / 64, default_filter_size / 32,
				default_filter_size / 16, default_filter_size / 8,
				default_filter_size / 4, default_filter_size / 2,
				default_filter_size, default_filter_size * 2,
				default_filter_size * 4, default_filter_size * 8]

os.system('make')
os.system('mkdir -p results/op_vs_bfsize')

for cpu_filter in cpu_filter_values:
	for gpu in gpu_values:
		postfix = "gpu" if gpu is not None else "cpu"

		for s in [0, 50, 100, 200, 400, 800]:
			for filter_size in filter_sizes:
				file = "op_vs_bfsize/results-op_vs_bfsize{}.csv".format(postfix)
				run_test(fname=file, selectivity="1", slowdown=s, gpu_devices=gpu,
					cpu_filter=cpu_filter, filter_size=int(filter_size),build_size=int(input_size))

				file = "op_vs_bfsize/results-op_vs_bfsize{}.csv".format(postfix)
				run_test(fname=file, selectivity="5", slowdown=s, gpu_devices=gpu,
					cpu_filter=cpu_filter, filter_size=int(filter_size), build_size=int(input_size))		