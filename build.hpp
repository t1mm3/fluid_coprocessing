#pragma once

#define NO_INLINE __attribute__((noinline))
#define R __restrict

#include <stdint.h>

static uint64_t rdtsc() {
	unsigned hi, lo;
	__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	return ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
}
