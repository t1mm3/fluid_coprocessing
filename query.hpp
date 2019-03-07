/* Copyright (c) 2019 by Tim Gubner, CWI 
 * Licensed under GPLv3
 */

#pragma once

#include "vectorized.hpp"
#include "hash_table.hpp"
#include "table.hpp"

struct Query {
private:
	uint32_t hashs[kVecSize];
	bool matches[kVecSize];
	int sel2[kVecSize];
	int64_t ksum = 0;

	HashTablinho::StaticProbeContext<kVecSize> ctx;

	void build_vec(HashTablinho* ht, int32_t* keys, uint32_t* hashs,
			int* sel, int num);

	void probe_vec(HashTablinho* ht, int32_t* keys, uint32_t* hashs,
			int* sel, int num, HashTablinho::ProbeContext& ctx);
public:
	void build(HashTablinho* ht, Table& t, int64_t morsel_size, size_t vsize);
	void probe(HashTablinho* ht, Table& t, int64_t morsel_size, size_t vsize);
};
