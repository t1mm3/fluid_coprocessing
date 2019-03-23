#!/usr/bin/python

import os

binary = "build/release/main_cu"

kibi = 1024
mebi = 1024*kibi
gibi = 1024*mebi

default_filter_size = 536870912    #64  MiB
default_streams = 4
default_probe_size = 2*gibi        #2   G keys
default_build_size = 4194304   	   #4   M keys
default_num_threads = 16
default_gpu_morsel_size = 16777216 #16  M keys
default_cpu_morsel_size = 16384	   #16  K keys
default_gpu_devices = 0
default_selectivity = 1
default_cpu_filter = 1
default_slowdown = 0

selectivities = [0, 1, 5, 10, 15] + range(20, 100, 10)
cpu_filter_values = [0, 1]
gpu_values = [None]
slowdown_values = [0, 1024, 1024*1024]


def syscall(cmd):
	print(cmd)
	os.system(cmd)

def run_test(fname = None, probe_size = None, streams = None, filter_size = None, build_size = None, gpu_morsel_size = None, cpu_morsel_size = None, gpu_devices = None, selectivity = None, threads = None, cpu_filter = None, slowdown = None):
	if fname is None: raise Exception("No filename provided")
	if probe_size is None: probe_size = default_probe_size
	if streams is None: streams = default_streams
	if build_size is None: build_size = default_build_size
	if gpu_morsel_size is None: gpu_morsel_size = default_gpu_morsel_size
	if cpu_morsel_size is None: cpu_morsel_size = default_cpu_morsel_size
	if gpu_devices is None: gpu_devices = default_gpu_devices
	if selectivity is None: selectivity = default_selectivity
	if threads is None: threads = default_num_threads
	if filter_size is None: filter_size = default_filter_size
	if cpu_filter is None: cpu_filter = default_cpu_filter
	if slowdown is None: slowdown = default_slowdown

	# Execute Experiment
	syscall("""{BINARY} --filter_size={FILTER_SIZE} --probe_size={PROBE_SIZE} --build_size={BUILD_SIZE} --gpu_morsel_size={GPU_MORSEL_SIZE} --cpu_morsel_size={CPU_MORSEL_SIZE} --gpu={DEVICES} --selectivity={SELECTIVITY} --num_threads={THREADS} --cpu_bloomfilter={CPU_FILTER} --slowdown={SLOWDOWN}""".format(
		BINARY=binary, FILTER_SIZE=filter_size, PROBE_SIZE=probe_size, BUILD_SIZE=build_size, GPU_MORSEL_SIZE=gpu_morsel_size, CPU_MORSEL_SIZE=cpu_morsel_size,
		DEVICES=gpu_devices, SELECTIVITY=selectivity, THREADS=threads, CPU_FILTER=cpu_filter, SLOWDOWN=slowdown))

	# We include the header in the first time
	if not os.path.isfile(os.path.join('results', fname)):
		os.system('mv results.csv %s' % os.path.join('results', fname))
	else:
		os.system('sed -n \'2,3p\' results.csv >> %s' % os.path.join('results', fname))


os.system('make')
os.system('mkdir -p results')






os.system('mkdir -p results/selectivity')
for cpu_filter in cpu_filter_values:
	for gpu in gpu_values:
		for slowdown in slowdown_values:
			postfix = "gpu" if gpu is not None else "cpu"

			for selectivity in selectivities:
				file = "selectivity/results-selectivity_{}.csv".format(postfix)
				run_test(fname=file, selectivity=selectivity, gpu_devices=gpu, cpu_filter=cpu_filter)
