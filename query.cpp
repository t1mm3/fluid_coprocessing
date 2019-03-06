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
	bool matches[num];

	Vectorized::map_hash(hashs, keys, sel, num);
	ht->Probe(ctx, matches, keys, hashs, sel, num);
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
