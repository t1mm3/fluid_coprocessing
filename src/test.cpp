#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "hash_table.hpp"

#include <map>

TEST_CASE("hash table", "[hj]") {
	std::map<int32_t, int32_t> build_tuples;

	SECTION("64k") {
		const int32_t cap = 64 * 1024;
		HashTablinho ht(64, cap);
		HashTablinho::StaticProbeContext<kVecSize> probe;

		REQUIRE(true);

		for (int32_t i = 0; i < cap; i++) {
			build_tuples.insert({i, i});
		}
		int32_t keys[kVecSize];
		uint32_t hashs[kVecSize];
		bool matches[kVecSize];

		Vectorized::chunk(0, cap, [&](auto offset, auto num) {
			// printf("insert offset %d\n", offset);
			for (int i = 0; i < num; i++) {
				keys[i] = i + offset;
				assert(build_tuples[i + offset] == i + offset);
			}

			Vectorized::map_hash(hashs, keys, nullptr, num);

			ht.Insert(keys, hashs, nullptr, num);
		});

		ht.FinalizeBuild();

		// check whether the keys are inside
		Vectorized::chunk(0, 2 * cap, [&](auto offset, auto num) {
			// printf("probe offset %d\n", offset);
			for (int i = 0; i < num; i++) {
				keys[i] = i + offset;
			}

			Vectorized::map_hash(hashs, keys, nullptr, num);

			ht.Probe(probe, matches, keys, hashs, nullptr, num);

			for (int i = 0; i < num; i++) {
				if (offset + i < cap) {
					REQUIRE(matches[i]);
				} else {
					REQUIRE(!matches[i]);
				}

				matches[i] = false;
			}
		});
	}

	SECTION("2k one bucket") {
		const int32_t cap = 2 * 1024;
		HashTablinho ht(64, cap);
		HashTablinho::StaticProbeContext<kVecSize> probe;

		REQUIRE(true);

		for (int32_t i = 0; i < cap; i++) {
			build_tuples.insert({i, i});
		}
		int32_t keys[kVecSize];
		uint32_t hashs[kVecSize];
		bool matches[kVecSize];

		Vectorized::chunk(0, cap, [&](auto offset, auto num) {
			// printf("insert offset %d\n", offset);
			for (int i = 0; i < num; i++) {
				keys[i] = i + offset;
				assert(build_tuples[i + offset] == i + offset);
				hashs[i] = 0;
			}

			ht.Insert(keys, hashs, nullptr, num);
		});

		ht.FinalizeBuild();

		// check whether the keys are inside
		Vectorized::chunk(0, 2 * cap, [&](auto offset, auto num) {
			// printf("probe offset %d\n", offset);
			for (int i = 0; i < num; i++) {
				keys[i] = i + offset;
				hashs[i] = 0;
			}

			ht.Probe(probe, matches, keys, hashs, nullptr, num);

			for (int i = 0; i < num; i++) {
				if (offset + i < cap) {
					REQUIRE(matches[i]);
				} else {
					REQUIRE(!matches[i]);
				}

				matches[i] = false;
			}
		});
	}
}
