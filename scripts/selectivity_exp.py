#!/usr/bin/python

import os

binary = "build/release/main_cu"

default_filter_size = 536870912    #64  MiB
default_streams = 4
default_probe_size = 419430400     #400 M keys
default_build_size = 4194304   	   #4   M keys
default_num_threads = 16
default_gpu_morsel_size = 16777216 #16  M keys
default_cpu_morsel_size = 16384	   #16  K keys
default_gpu_devices = 0
default_selectivity = 1

selectivities = [1 for y in range(3)]

def syscall(cmd):
	print(cmd)
	os.system(cmd)

def run_test(fname = None, probe_size = None, streams = None, filter_size = None, build_size = None, gpu_morsel_size = None, cpu_morsel_size = None, gpu_devices = None, selectivity = None, threads = None):
	if not fname:
		raise Exception("No filename provided")
	if not probe_size: probe_size = default_probe_size
	if not streams: streams = default_streams
	if not build_size: build_size = default_build_size
	if not gpu_morsel_size: gpu_morsel_size = default_gpu_morsel_size
	if not cpu_morsel_size: cpu_morsel_size = default_cpu_morsel_size
	if not gpu_devices: gpu_devices = default_gpu_devices
	if not selectivity: selectivity = default_selectivity
	if not threads: threads = default_num_threads
	if not filter_size: filter_size = default_filter_size

	# Execute Experiment
	syscall("""${BINARY} --filter_size=${FILTER_SIZE} --probe_size=${PROBE_SIZE} --build_size=${BUILD_SIZE} --gpu_morsel_size=${GPU_MORSEL_SIZE} --cpu_morsel_size=${CPU_MORSEL_SIZE} --gpu=${DEVICES} --selectivity=${SELECTIVITY} --num_threads=${THREADS}""".replace(
		"${BINARY}", binary).replace(
		"${FILTER_SIZE}", str(filter_size)).replace(
		"${PROBE_SIZE}", str(probe_size)).replace(
		"${BUILD_SIZE}", str(build_size)).replace(
		"${GPU_MORSEL_SIZE}", str(gpu_morsel_size)).replace(
		"${CPU_MORSEL_SIZE}", str(cpu_morsel_size)).replace(
		"${DEVICES}", str(gpu_devices)).replace(
		"${SELECTIVITY}", str(selectivity)).replace(
		"${THREADS}", str(threads)))
	# We include the header in the first time
	if not os.path.isfile(os.path.join('results', fname)):
		os.system('mv results.csv %s' % os.path.join('results', fname))
	else:
		os.system('sed -n \'2,3p\' results.csv >> %s' % os.path.join('results', fname))


os.system('make')
os.system('mkdir -p results')

os.system('mkdir -p results/selectivity')
for selectivity in selectivities:
	run_test(fname="selectivity/results-selectivity_cpu.csv", selectivity=selectivity)

os.system('mkdir -p results/selectivity')
for selectivity in selectivities:
	run_test(fname="selectivity/results-selectivity_gpu.csv", selectivity=selectivity, gpu_devices=1)

