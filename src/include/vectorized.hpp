/* Copyright (c) 2019 by Tim Gubner, CWI 
 * Licensed under GPLv3
 */

#pragma once
#include "build.hpp"
#include <stdint.h>
#include <algorithm>
#include <string>
#include <cassert>

#define bucket_t uint32_t


#define kVecSize 1024

struct Vectorized {
	template<typename T>static void
	map(int* sel, int num, T&& fun) {
		if (sel) {
			for (int i=0; i<num; i++) {
				fun(sel[i]);
			}
		} else {
			for (int i=0; i<num; i++) {
				fun(i);
			}
		}
	}

	template<typename T>static int
	select(int* osel, int* sel, int num, T&& fun) {
		int res = 0;

		if (sel) {
			for (int i=0; i<num; i++) {
				if (fun(sel[i])) {
					osel[res++] = sel[i];
				}
			}
		} else {
			for (int i=0; i<num; i++) {
				if (fun(i)) {
					osel[res++] = i;
				}
			}
		}

		return res;
	}

	static void NO_INLINE map_not_match_bucket_t(bool* R out, bucket_t* R a,
			bucket_t b, int* R sel, int num);

	static int NO_INLINE select_match(int* R osel, bool* R b, int* R sel,
			int num);
	static int NO_INLINE select_not_match(int* R osel, bool* R b, int* R sel,
			int num);
	static int NO_INLINE select_match_bit(int* R osel, uint8_t* R a, int num);

	inline static uint32_t hash32(uint32_t a) {
		return a * 2654435761;
	}

	static void NO_INLINE map_hash(uint32_t* R out, int32_t* R a, int* R sel,
			int num);

	static void NO_INLINE glob_sum(int64_t* R out, int32_t* R a, int* R sel,
			int num);

	template<typename T>
	static void NO_INLINE check(bool* R match, T* R keys, T* R table, bucket_t* R idx,
			size_t stride, int* R sel, int num) {
		if (stride > 1) {
			map(sel, num, [&] (auto i) { match[i] = table[idx[i] * stride] == keys[i]; });
		} else {
			map(sel, num, [&] (auto i) { match[i] = table[idx[i] * 1] == keys[i]; });
		}
	}

	template<typename T>
	static void NO_INLINE write(T* R table, T* R a, size_t R idx,
			size_t stride, int* R sel, int num) {
		if (stride > 1) {
			map(sel, num, [&] (auto i) { table[(idx+i) * stride] = a[i]; });
		} else {
			map(sel, num, [&] (auto i) { table[(idx+i) * 1] = a[i]; });
		}
	}

	template<typename T>
	static void NO_INLINE gather(T* R out, T* R table, bucket_t* R idx,
			size_t stride, int* R sel, int num) {
		if (stride > 1) {
			map(sel, num, [&] (auto i) { out[i] = table[idx[i] * stride]; });
		} else {
			map(sel, num, [&] (auto i) { out[i] = table[idx[i] * 1]; });
		}
	}

	static void NO_INLINE gather_next(bucket_t* R out, bucket_t* R table, bucket_t* R idx,
			size_t stride, int* R sel, int num) {
		Vectorized::gather<bucket_t>(out, table, idx, stride, sel, num);
	}

	template<typename T>
	static void NO_INLINE read(T* R out, T* R table, size_t R idx,
			size_t stride, int* R sel, int num) {
		if (stride > 1) {
			map(sel, num, [&] (auto i) { out[i] = table[(idx+i) * stride]; });
		} else {
			map(sel, num, [&] (auto i) { out[i] = table[(idx+i) * 1]; });
		}
	}

	template<typename T>
	static void chunk(size_t offset, size_t size, T&& fun, int vsize = kVecSize) {
		const size_t end = size + offset;
		size_t i = offset;
		size_t total_num = 0;
		while (i < end) {
			size_t num = std::min((size_t)vsize, end-i);
			fun(i, num);
			total_num += num;
			assert(total_num <= size);
			i+=num;
		}

		assert(total_num == size);
	}
};
