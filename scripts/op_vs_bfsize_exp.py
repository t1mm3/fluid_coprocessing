#!/usr/bin/python3

import os
from common_exp import run_test
from common_exp import default_filter_size

cpu_filter_values = [1]
gpu_values = [None, 1]

my_dict = {
	('4294967296'):536870912,
	('2147483648'):268435456, 
	('1073741824'):134217728,
	('536870912') :67108864,
	('268435456') :33554432,
	('134217728') :16777216,
	('67108864')  :8388608,
    ('33554432')  :4194304,
	('16777216')  :2097152,
    ('8388608')   :1048576
}
default_probe_scale = 64

os.system('make')
os.system('mkdir -p results/op_vs_bfsize')

for cpu_filter in cpu_filter_values:
	for gpu in gpu_values:
		postfix = "gpu" if gpu is not None else "cpu"

		for s in [0, 50, 100, 200, 400, 800]:
			for filter_size,build_size in my_dict.items():
				file = "op_vs_bfsize/results-op_vs_bfsize{}.csv".format(postfix)
				run_test(fname=file, selectivity="1", slowdown=s, gpu_devices=gpu,
					cpu_filter=cpu_filter, filter_size=int(filter_size), build_size=int(build_size), probe_size=int(536870912 * default_probe_scale))

				#file = "op_vs_bfsize/results-op_vs_bfsize{}.csv".format(postfix)
				#run_test(fname=file, selectivity="5", slowdown=s, gpu_devices=gpu,
				#cpu_filter=cpu_filter, filter_size=int(filter_size), build_size=int(build_size), probe_size=int(build_size * default_probe_scale))		