/* Copyright (c) 2019 by Tim Gubner, CWI 
 * Licensed under GPLv3
 */

#pragma once

#include <stdint.h>
#include <atomic>
#include <vector>
#include <cassert>

struct Table {
private:
	int64_t capacity;
	std::atomic<int64_t> start; //! Current start offset
	std::atomic<int64_t> done; //!< Current finished offset (after morsel completion)

public:
	std::vector<void*> columns;

public:
	int64_t size() const { return capacity; }

	Table(int64_t num_cols, int64_t cap);
	void reset();

	template<typename T>void
	fill_columns(T&& f) {
		for (size_t i=0; i<columns.size(); i++) {
			f(i, columns[i], this);
		}
	}

private:
	/* "Consumes" a range of the table
		\param ostart Offset of range
		\param onum Number of tuples in range
		\param num Preferred size of range
		\returns True, if successful (and valid range); False, otherwise
	*/
	bool get_range(int64_t& onum, int64_t& ostart, int64_t num);

public:
	template<typename T, typename S>
	void chunk(T&& morsel, S&& finished, int64_t morsel_size = 16*1024) {
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
