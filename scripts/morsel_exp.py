#!/usr/bin/python3

import os
from common_exp import run_test

os.system('make')
os.system('mkdir -p results/morsel')

for cms in [16*1024, 128*1024, 1024*1024]:
	for gms in [128*1024, 1024*1024, 16*1024*1024, 128*1024*1024]:
		file = "morsel/results-morsel.csv"
		run_test(fname=file, selectivity="5", gpu_devices="1",
			cpu_filter="1", cpu_morsel_size=cms, gpu_morsel_size=gms)
