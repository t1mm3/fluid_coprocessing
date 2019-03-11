/* Copyright (c) 2019 by Tim Gubner, CWI 
 * Licensed under GPLv3
 */

#include "query.hpp"

void
Query::build_vec(HashTablinho* ht, int32_t* keys, uint32_t* hashs,
int* sel, int num)
{
	Vectorized::map_hash(hashs, keys, sel, num);
	ht->Insert(keys, hashs, sel, num);
}

void
Query::probe_vec(HashTablinho* ht, int32_t* keys, uint32_t* hashs,
	int* sel, int num, HashTablinho::ProbeContext& ctx)
{
	// BloomProbe


	// HashSemiJoin
	Vectorized::map_hash(hashs, keys, sel, num);
	ht->Probe(ctx, matches, keys, hashs, sel, num);

	num = Vectorized::select_match(sel2, matches, sel, num);
	sel = &sel2[0];

	// SUM(key)
	Vectorized::glob_sum(&ksum, keys, sel, num);
}

void
Query::build(HashTablinho* ht, Table& t, int64_t morsel_size, size_t vsize)
{
	t.chunk([&] (auto columns, auto num_columns, auto offset, auto num) {
		int32_t* tkeys = (int32_t*)columns[0];
		Vectorized::chunk(offset, num, [&] (auto offset, auto num) {
			build_vec(ht, &tkeys[offset], hashs, nullptr, num);
		}, vsize);
	}, [&] () {
		// finished
		ht->FinalizeBuild();
	});
}

void
Query::probe(HashTablinho* ht, Table& t, int64_t morsel_size, size_t vsize)
{
	t.chunk([&] (auto columns, auto num_columns, auto offset, auto num) {
		int32_t* tkeys = (int32_t*)columns[0];

		Vectorized::chunk(offset, num, [&] (auto offset, auto num) {
			probe_vec(ht, &tkeys[offset], hashs, nullptr, num, ctx);
		}, vsize);
	}, [&] () { });
}