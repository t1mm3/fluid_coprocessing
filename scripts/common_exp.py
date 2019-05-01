import os

binary = "build/release/main_cu"

kibi = 1024
mebi = 1024*kibi
gibi = 1024*mebi

default_filter_size = 536870912    #64  MiB
default_streams = 4
default_probe_scale = 128
default_build_size = 4194304   	   #4   M keys
default_probe_size = 1*gibi
default_num_threads = 10
default_gpu_morsel_size = 16777216 #16  M keys
default_cpu_morsel_size = 16384	   #16  K keys
default_gpu_devices = 0
default_selectivity = 1
default_cpu_filter = 1
default_slowdown = 0
default_keys_on_gpu = 0
default_repetitions = 3
default_num_payloads = 32

def run_timeout(cmd, timeout):
	# inspired by https://stackoverflow.com/questions/36952245/subprocess-timeout-failure
	import os
	import signal
	from subprocess import Popen, PIPE, TimeoutExpired
	from time import monotonic as timer

	start = timer()
	with Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, preexec_fn=os.setsid) as process:
		try:
			output, err = process.communicate(timeout=timeout)
			print(err)
			return output
		except TimeoutExpired:
			print('Timeout for {}'.format(cmd))
			os.killpg(process.pid, signal.SIGINT) # send signal to the process group
			output = process.communicate()[0]
			return None

def syscall(cmd):
	print(cmd)
	# os.system(cmd)

	timed_out = True
	# 100 seconds for large run * 20 reps * 2 (overalloc)
	time_out_seconds = 30*60
	iterations = 0
	output = None

	while timed_out:
		assert(iterations < 10)
		output = run_timeout(cmd, time_out_seconds)
		timed_out = output is None
		iterations = iterations + 1

	assert(output is not None)
	return output

def run_test(fname = None, probe_size = None, streams = None, filter_size = None, probe_scale = None, num_payloads = None, build_size = None, gpu_morsel_size = None, cpu_morsel_size = None, gpu_devices = None, selectivity = None, threads = None, cpu_filter = None, slowdown = None, keys_on_gpu = None, perf_optimal_bloomfilters = None):
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
	if keys_on_gpu is None: keys_on_gpu = default_keys_on_gpu
	if probe_scale is None: probe_scale = default_probe_scale
	if num_payloads is None: num_payloads = default_num_payloads
	if perf_optimal_bloomfilters is None: perf_optimal_bloomfilters = False

	repetitions=default_repetitions


	cmd = """{BINARY} --repetitions={REPS} --streams={STREAM} --filter_size={FILTER_SIZE} --probe_size={PROBE_SIZE} --build_size={BUILD_SIZE} --probe_scale={PROBE_SCALE} --num_payloads={NUM_PAYLOADS} --gpu_morsel_size={GPU_MORSEL_SIZE} --cpu_morsel_size={CPU_MORSEL_SIZE} --gpu={DEVICES} --selectivity={SELECTIVITY} --num_threads={THREADS} --cpu_bloomfilter={CPU_FILTER} --slowdown={SLOWDOWN} --in_gpu_keys={KEYS_ON_GPU}""".format(
		BINARY=binary, STREAM=streams, REPS=repetitions, FILTER_SIZE=filter_size, PROBE_SIZE=probe_size, BUILD_SIZE=build_size, GPU_MORSEL_SIZE=gpu_morsel_size, CPU_MORSEL_SIZE=cpu_morsel_size,
		DEVICES=gpu_devices, SELECTIVITY=selectivity, THREADS=threads, CPU_FILTER=cpu_filter, SLOWDOWN=slowdown, KEYS_ON_GPU=keys_on_gpu, PROBE_SCALE=probe_scale, NUM_PAYLOADS=num_payloads)

	if perf_optimal_bloomfilters:
		# Measure tw
		script = cmd + " --measure_tw=1"
		output = syscall(script)
		words = str(output).split(" ")
		print(words)
		tw_pos = words.index("TW")
		tw = words[tw_pos+1]
		tw_float = float(tw)
		print("Using tw {}".format(tw_float))

	# Execute Experiment
	syscall(cmd)

	# We include the header in the first time
	if not os.path.isfile(os.path.join('results', fname)):
		os.system('mv results.csv %s' % os.path.join('results', fname))
	else:
		os.system('sed -n \'2,3p\' results.csv >> %s' % os.path.join('results', fname))

