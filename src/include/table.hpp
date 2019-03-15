/* Copyright (c) 2019 by Tim Gubner, CWI
 * Licensed under GPLv3
 */

#pragma once

#include <atomic>
#include <cassert>
#include <stdint.h>
#include <vector>
#include <iostream>

struct Table {
private:
	int64_t capacity;
	std::atomic<int64_t> start; //! Current start offset
	std::atomic<int64_t> done;  //!< Current finished offset (after morsel completion)

public:
	std::vector<void *> columns;

public:
	int64_t size() const {
		return capacity;
	}

	Table(int64_t num_cols, int64_t cap) {
		start = 0;
		done = 0;
		capacity = cap;

		for (int64_t i = 0; i < num_cols; i++) {
			size_t bytes = sizeof(int32_t) * cap;
			void *col = malloc(bytes);
			memset(col, 0, bytes);
			columns.push_back(col);
		}
	}
	void reset() {
		done = 0;
		start = 0;

		for (auto &col : columns) {
			free(col);
		}
	}

	template <typename T> void fill_columns(T &&f) {
		for (size_t i = 0; i < columns.size(); i++) {
			f(i, columns[i], this);
		}
	}

	bool get_range(int64_t &onum, int64_t &ostart, int64_t num) {
		if (start >= capacity) {
			return false;
		}

		ostart = std::atomic_fetch_add(&start, num);
		if (ostart >= capacity) {
			return false;
		}

		const int64_t todo = capacity - ostart;
		assert(todo > 0);

		onum = std::min(todo, num);
		//std::cout << " get range " << onum << " " << ostart << std::endl;
		return onum > 0;
	}

private:
	/* "Consumes" a range of the table
	        \param ostart Offset of range
	        \param onum Number of tuples in range
	        \param num Preferred size of range
	        \returns True, if successful (and valid range); False, otherwise
	*/

public:
	template <typename T, typename S> void chunk(T &&morsel, S &&finished, int64_t morsel_size = 16 * 1024) {
		while (1) {
			int64_t num = 0;
			int64_t offset = 0;
			bool has_range = get_range(num, offset, morsel_size);

			if (!has_range) {
				break;
			}

			assert(num <= morsel_size);

			morsel(&columns[0], columns.size(), offset, num);

			done += num;
			if (done >= capacity) {
				finished();
			}
		}
	}
};
