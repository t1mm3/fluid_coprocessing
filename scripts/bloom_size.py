import os

binary="build/release/bloom_size_benchmark"

default_bloom_size = 1*1024*1024*8
default_bits = 6
bloom_sizes = [1 * 1024 * 1024 * 8,
	2 * 1024 * 1024 * 8,
	4 * 1024 * 1024 * 8,
	8 * 1024 * 1024 * 8,
	16 * 1024 * 1024 * 8,
	32 * 1024 * 1024 * 8,
	64 * 1024 * 1024 * 8,
	128 * 1024 * 1024 * 8,
	256 * 1024 * 1024 * 8,
	512 * 1024 * 1024 * 8]
bits_to_sort = [6, 6, 6, 6, 6, 6, 6, 6, 8, 12]
file = "bloom_size/bloom_size_gpu"

def syscall(cmd):
	print(cmd)
	os.system(cmd)

def run_experiment(bloom_size = None, bits_to_sort = None, result_file = None):
	if not result_file: raise Exception("No file provided")
	if not bloom_size: bloom_size = default_bloom_size
	if not bits_to_sort: bits_to_sort = default_bits
	syscall("""{BINARY} --bf_size={BF_SIZE} --bits_to_sort={BITS}""".format(BINARY=binary, BF_SIZE=bloom_size, BITS=bits_to_sort))

	# We include the header in the first time
	if not os.path.isfile(os.path.join('results', result_file)):
		os.system('mv results.csv %s' % os.path.join('results', result_file))
	else:
		os.system('sed -n \'2,3p\' results.csv >> %s' % os.path.join('results', result_file))

os.system('make bench_size')
os.system('mkdir results/bloom_size')

for size,bits in zip(bloom_sizes, bits_to_sort):
	run_experiment(bloom_size=size,bits_to_sort=bits,result_file=file)
