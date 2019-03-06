#pragma once
#include "build.hpp"
#include <stdint.h>
#include <string>

struct Vectorized {
	static constexpr int kVecSize = 1024;

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

	inline static uint32_t hash32(uint32_t a) {
		return a * 2654435761;
	}

	static void NO_INLINE map_hash(uint32_t* R out, int32_t* R a, int* R sel,
			int num) {
		map(sel, num, [&] (auto i) { out[i] = hash32((uint32_t)(a[i])); });
	}

	template<typename T>
	static int NO_INLINE check(bool* R match, T* R keys, T* R table, size_t* R idx,
			size_t stride, int* R sel, int num) {
		if (stride > 1) {
			map(sel, num, [&] (auto i) { match[i] = table[idx[i] * stride] == keys[i]; });
		} else {
			map(sel, num, [&] (auto i) { match[i] = table[idx[i] * 1] == keys[i]; });
		}
	}

	template<typename T>
	static int NO_INLINE check_ptr(bool* R match, T* R keys, T** R ptrs, int* R sel, int num) {
		map(sel, num, [&] (auto i) { match[i] = (*ptrs[i]) == keys[i]; });
	}

	template<typename T>
	static void chunk(size_t offset, size_t size, T&& fun, int vsize = kVecSize) {
		const size_t end = size + offset;
		size_t i = offset;
		while (i < end) {
			size_t num = std::min(vsize, end-i);
			fun(i, num);
		}
	}
};
