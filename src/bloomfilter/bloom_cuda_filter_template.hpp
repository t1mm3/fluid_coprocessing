#pragma once

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_sgew.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_sltw.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_block_logic_zoned.hpp>
#include <dtl/filter/blocked_bloomfilter/blocked_bloomfilter_logic.hpp>
#include <dtl/filter/blocked_bloomfilter/hash_family.hpp>

//===----------------------------------------------------------------------===//
// Typedefs.
using filter_key_t = $u32;
using hash_value_t = $u32;
using word_t = $u32;
//===----------------------------------------------------------------------===//
/// Type switch for the different block types. - The proper block type is chosen
/// based on the parameters word_cnt and zone_cnt.
template <u32 word_cnt, u32 sector_cnt, u32 zone_cnt, u32 k> struct bbf_block_switch {
	/// The first hash function to use inside the block. Note: 0 is used for block
	/// addressing
	static constexpr u32 block_hash_fn_idx = 1;

	// DEPRECATED
	static constexpr u1 early_out = false;

	/// Classic blocked Bloom filter, where the block size exceeds the size of a
	/// word.
	using blocked = dtl::multisector_block<key_t, word_t, word_cnt, 1, k, dtl::hasher, hash_value_t, block_hash_fn_idx,
	                                       0, sector_cnt, early_out>;

	/// Sectorized blocked Bloom filter, where the number of sectors is less than
	/// the number of words per block (causing a random block access pattern).
	///
	/// This block type is know to be dominated by classic-blocked wrt. precision
	/// and by register-blocked and cache-sectorized wrt. performance on CPUs.
	using sectorized_rnd = dtl::multiword_block<key_t, word_t, word_cnt, sector_cnt, k, dtl::hasher, hash_value_t,
	                                            block_hash_fn_idx, 0, word_cnt, early_out>;

	/// Sectorized blocked Bloom filter, where the number of sectors is equal to
	/// the number of words per block.
	using sectorized_seq = dtl::multiword_block<key_t, word_t, word_cnt, sector_cnt, k, dtl::hasher, hash_value_t,
	                                            block_hash_fn_idx, 0, word_cnt, early_out>;

	static_assert(!(zone_cnt > 1) || word_cnt == sector_cnt,
	              "The number of words must be equal to the number of sectors.");

	/// Cache-Sectorized (aka zoned) blocked Bloom filter, where the number of
	/// sectors is equal to the number of words per block.
	using zoned = dtl::multizone_block<filter_key_t, word_t, word_cnt, zone_cnt, k, dtl::hasher, hash_value_t,
	                                   block_hash_fn_idx, 0, zone_cnt, early_out>;

	/// Refers to the implementation.
	using type =
	    typename std::conditional<(zone_cnt > 1), zoned,
	                              typename std::conditional<(sector_cnt >= word_cnt), sectorized_seq,
	                                                        typename std::conditional<(sector_cnt > 1), sectorized_rnd,
	                                                                                  blocked>::type>::type>::type;

	/// The number of accessed words per lookup. Actually, the number of LOAD-CMP
	/// sequences.
	static constexpr std::size_t word_access_cnt = (zone_cnt > 1) ? zone_cnt : (sector_cnt >= word_cnt) ? word_cnt : k;
};
//===----------------------------------------------------------------------===//
/// The template for (almost all kinds of) blocked Bloom filters.
template <u32 word_cnt, u32 sector_cnt, u32 zone_cnt, u32 k,
          dtl::block_addressing addr = dtl::block_addressing::POWER_OF_TWO, u1 early_out = false>
using bbf_t = dtl::blocked_bloomfilter_logic<filter_key_t, dtl::hasher,
                                             typename bbf_block_switch<word_cnt, sector_cnt, zone_cnt, k>::type, addr>;
//===----------------------------------------------------------------------===//
/// Returns the number of accessed words per lookup. Actually, the number of
/// LOAD-CMP sequences.
static constexpr std::size_t get_word_access_cnt(u32 word_cnt, u32 sector_cnt, u32 zone_cnt, u32 k) {
	return (zone_cnt > 1) ? zone_cnt : (sector_cnt >= word_cnt) ? word_cnt : k;
};
//===----------------------------------------------------------------------===//
