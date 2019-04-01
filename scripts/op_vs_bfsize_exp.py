#!/usr/bin/python3

import os
from common_exp import run_test
from common_exp import default_filter_size

cpu_filter_values = [1]
gpu_values = [None, 1]
build_size = 1024

my_dict = {
    ('8388608')   :build_size, 
	('16777216')  :build_size, 
    ('33554432')  :build_size,
	('67108864')  :build_size,
	('134217728') :build_size, 
	('268435456') :build_size,
	('536870912') :build_size, 
	('1073741824'):build_size,
	('2147483648'):build_size, 
	('4294967296'):build_size
}

os.system('make')
os.system('mkdir -p results/op_vs_bfsize')

for cpu_filter in cpu_filter_values:
	for gpu in gpu_values:
		postfix = "gpu" if gpu is not None else "cpu"

		for s in [0, 50, 100, 200, 400, 800]:
			for filter_size,build_size in my_dict.items():
				file = "op_vs_bfsize/results-op_vs_bfsize{}.csv".format(postfix)
				run_test(fname=file, selectivity="1", slowdown=s, gpu_devices=gpu,
					cpu_filter=cpu_filter, filter_size=int(filter_size),build_size=int(build_size))

				file = "op_vs_bfsize/results-op_vs_bfsize{}.csv".format(postfix)
				run_test(fname=file, selectivity="5", slowdown=s, gpu_devices=gpu,
					cpu_filter=cpu_filter, filter_size=int(filter_size), build_size=int(build_size))		