/* Copyright (c) 2019 by Tim Gubner, CWI
 * Licensed under GPLv3
 */

#pragma once
#include "build.hpp"

#include <algorithm>
#include <cassert>
#include <stdint.h>
#include <string>
#include "constants.hpp"

#define bucket_t uint32_t

struct Vectorized {
	template <typename T> static void map(int *sel, int num, T &&fun) {
		if (sel) {
			for (int i = 0; i < num; i++) {
				fun(sel[i]);
			}
		} else {
			for (int i = 0; i < num; i++) {
				fun(i);
			}
		}
	}

	template <typename T> static int select(int *osel, int *sel, int num, T &&fun) {
		int res = 0;

		if (sel) {
			for (int i = 0; i < num; i++) {
				if (fun(sel[i])) {
					osel[res++] = sel[i];
				}
			}
		} else {
			for (int i = 0; i < num; i++) {
				if (fun(i)) {
					osel[res++] = i;
				}
			}
		}

		return res;
	}

	static void NO_INLINE map_not_match_bucket_t(bool *CPU_R out, bucket_t *CPU_R a, bucket_t b, int *CPU_R sel,
	                                             int num) {
		map(sel, num, [&](auto i) { out[i] = a[i] != b; });
	}

	static int NO_INLINE select_match(int *CPU_R osel, bool *CPU_R b, int *CPU_R sel, int num) {
		return select(osel, sel, num, [&](auto i) { return b[i]; });
	}
	static int NO_INLINE select_not_match(int *CPU_R osel, bool *CPU_R b, int *CPU_R sel, int num) {
		return select(osel, sel, num, [&](auto i) { return !b[i]; });
	}

	static int NO_INLINE select_match_bit(int *CPU_R osel, uint8_t *CPU_R a, int num) {
		int res = 0;
		int i = 0;
#define A(z)                                                                                                           \
	{                                                                                                                  \
		int i = z;                                                                                                     \
		int w = a[i / 8];                                                                                              \
		uint8_t m = 1 << (i % 8);                                                                                      \
		if (w & m) {                                                                                                   \
			osel[res++] = i;                                                                                           \
		}                                                                                                              \
	}

		for (; i + 8 < num; i += 8) {
			A(i);
			A(i + 1);
			A(i + 2);
			A(i + 3);
			A(i + 4);
			A(i + 5);
			A(i + 6);
			A(i + 7);
		}
		for (; i < num; i++) {
			A(i);
		}
#undef A
		return res;
	}

	inline static uint32_t hash32(uint32_t a) {
		return a * 2654435761;
	}

	static void NO_INLINE map_hash(uint32_t *CPU_R out, int32_t *CPU_R a, int *CPU_R sel, int num) {
		map(sel, num, [&](auto i) { out[i] = hash32((uint32_t)(a[i])); });
	}

	static void NO_INLINE glob_sum(uint64_t *CPU_R out, int32_t *CPU_R a, int *CPU_R sel, int num) {
		uint64_t p = 0;
		map(sel, num, [&](auto i) { p += a[i]; });
		*out = *out + p;
	}

	template <typename T>
	static void NO_INLINE check(bool *CPU_R match, T *CPU_R keys, T *CPU_R table, bucket_t *CPU_R idx, size_t stride,
	                            int *CPU_R sel, int num) {
		if (stride > 1) {
			map(sel, num, [&](auto i) { match[i] = table[idx[i] * stride] == keys[i]; });
		} else {
			map(sel, num, [&](auto i) { match[i] = table[idx[i] * 1] == keys[i]; });
		}
	}

	template <typename T>
	static void NO_INLINE write(T *CPU_R table, T *CPU_R a, size_t CPU_R idx, size_t stride, int *CPU_R sel, int num) {
		if (stride > 1) {
			map(sel, num, [&](auto i) { table[(idx + i) * stride] = a[i]; });
		} else {
			map(sel, num, [&](auto i) { table[(idx + i) * 1] = a[i]; });
		}
	}

	template <typename T>
	static void NO_INLINE gather(T *CPU_R out, T *CPU_R table, bucket_t *CPU_R idx, size_t stride, int *CPU_R sel,
	                             int num) {
		if (stride > 1) {
			map(sel, num, [&](auto i) { out[i] = table[idx[i] * stride]; });
		} else {
			map(sel, num, [&](auto i) { out[i] = table[idx[i] * 1]; });
		}
	}

	static void NO_INLINE gather_next(bucket_t *CPU_R out, bucket_t *CPU_R table, bucket_t *CPU_R idx, size_t stride,
	                                  int *CPU_R sel, int num) {
		Vectorized::gather<bucket_t>(out, table, idx, stride, sel, num);
	}

	template <typename T>
	static void NO_INLINE read(T *CPU_R out, T *CPU_R table, size_t CPU_R idx, size_t stride, int *CPU_R sel, int num) {
		if (stride > 1) {
			map(sel, num, [&](auto i) { out[i] = table[(idx + i) * stride]; });
		} else {
			map(sel, num, [&](auto i) { out[i] = table[(idx + i) * 1]; });
		}
	}

	template <typename T> static void chunk(size_t offset, size_t size, T &&fun, int vsize = kVecSize) {
		const size_t end = size + offset;
		size_t i = offset;
		size_t total_num = 0;
		while (i < end) {
			size_t num = std::min((size_t)vsize, end - i);
			fun(i, num);
			total_num += num;
			assert(total_num <= size);
			i += num;
		}

		assert(total_num == size);
	}
};
