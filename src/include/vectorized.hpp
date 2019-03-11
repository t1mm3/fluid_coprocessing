/* Copyright (c) 2019 by Tim Gubner, CWI 
 * Licensed under GPLv3
 */

#pragma once
#include "build.hpp"
#include <stdint.h>
#include <algorithm>
#include <string>

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

	static void NO_INLINE map_not_match_void(bool* R out, void** R a,
			void* R b, int* R sel, int num) {
		map(sel, num, [&] (auto i) { out[i] = a[i] != b; });
	}

	static int NO_INLINE select_match(int* R osel, bool* R b, int* R sel,
			int num) {
		return select(osel, sel, num, [&] (auto i) { return b[i]; });
	}

	static int NO_INLINE select_not_match(int* R osel, bool* R b, int* R sel,
			int num) {
		return select(osel, sel, num, [&] (auto i) { return !b[i]; });
	}

	static int NO_INLINE select_match_bit(int* R osel, uint8_t* R a, int num) {
		int res=0;
		int i=0;
#define A(z) { int i=z; int w=a[i/8]; uint8_t m=1 << (i % 8); if (w & m) { osel[res++] = i;}}

		for (; i+8<num; i+=8) {
			A(i);
			A(i+1);
			A(i+2);
			A(i+3);
			A(i+4);
			A(i+5);
			A(i+6);
			A(i+7);
		}
		for (; i<num; i++) { A(i); }
#undef A
		return res;
	}

	inline static uint32_t hash32(uint32_t a) {
		return a * 2654435761;
	}

	static void NO_INLINE map_hash(uint32_t* R out, int32_t* R a, int* R sel,
			int num) {
		map(sel, num, [&] (auto i) { out[i] = hash32((uint32_t)(a[i])); });
	}

	static void NO_INLINE glob_sum(int64_t* R out, int32_t* R a, int* R sel,
			int num) {

		int64_t p=0;
		map(sel, num, [&] (auto i) { p+= a[i]; });
		*out = *out + p;
	}

	template<typename T>
	static void NO_INLINE check(bool* R match, T* R keys, T* R table, size_t* R idx,
			size_t stride, int* R sel, int num) {
		if (stride > 1) {
			map(sel, num, [&] (auto i) { match[i] = table[idx[i] * stride] == keys[i]; });
		} else {
			map(sel, num, [&] (auto i) { match[i] = table[idx[i] * 1] == keys[i]; });
		}
	}

	template<typename T>
	static void NO_INLINE check_ptr(bool* R match, T* R keys, T** R ptrs, int* R sel, int num) {
		map(sel, num, [&] (auto i) { match[i] = (*ptrs[i]) == keys[i]; });
	}

	template<typename T>
	static void NO_INLINE scatter(T* R table, T* R a, size_t* R idx,
			size_t stride, int* R sel, int num) {
		if (stride > 1) {
			map(sel, num, [&] (auto i) { table[idx[i] * stride] = a[i]; });
		} else {
			map(sel, num, [&] (auto i) { table[idx[i] * 1] = a[i]; });
		}
	}

	template<typename T>
	static void NO_INLINE gather(T* R out, T* R table, size_t* R idx,
			size_t stride, int* R sel, int num) {
		if (stride > 1) {
			map(sel, num, [&] (auto i) { out[i] = table[idx[i] * stride]; });
		} else {
			map(sel, num, [&] (auto i) { out[i] = table[idx[i] * 1]; });
		}
	}

	template<typename T>
	static void NO_INLINE gather_ptr(T* R out, T** R ptrs, int offset, int* R sel, int num) {
		map(sel, num, [&] (auto i) { out[i] = *(ptrs[i] + offset); });
	}

	template<typename T>
	static void chunk(size_t offset, size_t size, T&& fun, int vsize = kVecSize) {
		const size_t end = size + offset;
		size_t i = offset;
		while (i < end) {
			size_t num = std::min((size_t)vsize, end-i);
			fun(i, num);
		}
	}
};
