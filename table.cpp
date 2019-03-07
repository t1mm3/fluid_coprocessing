/* Copyright (c) 2019 by Tim Gubner, CWI 
 * Licensed under GPLv3
 */

#include "table.hpp"
#include <malloc.h>
#include <cstring>

Table::Table(int64_t num_cols, int64_t cap)
{
	start = 0;
	done = 0;
	capacity = cap;

	for (int64_t i=0; i<num_cols; i++) {
		size_t bytes = sizeof(int32_t)*cap;
		void* col = malloc(bytes);
		memset(col, 0, bytes);
		columns.push_back(col);
	}
}

void
Table::reset() {
	done = 0;
	start = 0;

	for (auto& col : columns) {
		free(col);
	}
}

bool
Table::get_range(int64_t& onum, int64_t& ostart, int64_t num)
{
	if (start > capacity) {
		return false;
	}

	ostart = std::atomic_fetch_add(&start, num);
	if (ostart >= capacity) {
		return false;
	}

	const int64_t todo = capacity - ostart;
	assert(todo > 0);

	onum = std::min(todo, num);
	return onum > 0;
}