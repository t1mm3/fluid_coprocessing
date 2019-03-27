#!/usr/bin/python3

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

filter_size_values = [default_filter_size / 8, default_filter_size, 8*default_filter_size]


def run_timeout(cmd, timeout):
	# inspired by https://stackoverflow.com/questions/36952245/subprocess-timeout-failure
	import os
	import signal
	from subprocess import Popen, PIPE, TimeoutExpired
	from time import monotonic as timer

	start = timer()
	with Popen(cmd, shell=True, stdout=PIPE, preexec_fn=os.setsid) as process:
		try:
			output = process.communicate(timeout=timeout)[0]
			return True
		except TimeoutExpired:
			print('Timeout for {}'.format(cmd))
			os.killpg(process.pid, signal.SIGINT) # send signal to the process group
			output = process.communicate()[0]
			return False

def syscall(cmd):
	print(cmd)
	# os.system(cmd)

	timed_out = True
	# 100 seconds for large run * 20 reps * 2 (overalloc)
	time_out_seconds = 100 * 20 * 2 # 30*60
	iterations = 0

	while timed_out:
		assert(iterations < 10)
		timed_out = not run_timeout(cmd, time_out_seconds)
		iterations = iterations + 1

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
os.system('mkdir -p results/morsel')

for cms in [16*1024, 128*1024, 1024*1024]:
	for gms in [128*1024, 1024*1024, 16*1024*1024, 128*1024*1024]:
		file = "morsel/results-morsel.csv"
		run_test(fname=file, selectivity="5", gpu_devices="1",
			cpu_filter="1", cpu_morsel_size=cms)
