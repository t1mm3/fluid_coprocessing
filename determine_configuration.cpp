#include <cstdint>

#include <dtl/filter/blocked_bloomfilter/math.hpp>

/// The possible filter sizes in bits.
const std::vector<std::size_t>
filter_sizes {
    1ull * 1024 * 1024 * 8,
    2ull * 1024 * 1024 * 8,
    4ull * 1024 * 1024 * 8,
    8ull * 1024 * 1024 * 8,
   16ull * 1024 * 1024 * 8,
   32ull * 1024 * 1024 * 8,
   64ull * 1024 * 1024 * 8,
  128ull * 1024 * 1024 * 8,
  256ull * 1024 * 1024 * 8,
  512ull * 1024 * 1024 * 8,
};

/// Lookup table for the average time a lookup takes. Entry i corresponds to
/// the filter size given in filter_sizes[i].
const std::vector<double>
nanos_per_lookup_gtx_1080 {
    0.3362,
    0.3363,
    0.4563,
    0.6000,
    0.6870,
    0.7363,
    0.7625,
    0.7761,
    0.7831,
    0.7864,
};

const std::vector<double>
nanos_per_lookup_gtx_1080_cached { // no PCI data transfer
    0.1553,
    0.1844,
    0.3866,
    0.5551,
    0.6556,
    0.7134,
    0.7445,
    0.7606,
    0.7688,
    0.7728,
};

const std::vector<double>
nanos_per_lookup_gtx_1080_ti {
    0.3419,
    0.3384,
    0.3399,
    0.3639,
    0.4333,
    0.4746,
    0.5049,
    0.5104,
    0.5166,
    0.5204,
};

const std::vector<double>
nanos_per_lookup_gtx_1080_ti_cached { // no PCI data transfer
    0.1358,
    0.1252,
    0.1921,
    0.3154,
    0.3986,
    0.4474,
    0.4759,
    0.4910,
    0.4987,
    0.5033,
};

const std::vector<double>
nanos_per_lookup_gtx_1050_ti {
    0.3420,
    0.3393,
    0.3401,
    0.3759,
    0.4469,
    0.4856,
    0.5009,
    0.5244,
    0.5268,
    0.5250,
};

const std::vector<double>
nanos_per_lookup_gtx_1050_ti_cached { // no PCI data transfer
    0.1358,
    0.1254,
    0.1935,
    0.3177,
    0.3961,
    0.4521,
    0.4759,
    0.4910,
    0.5045,
    0.5026,
};

/// The Bloom filter's block size. Fixed to 128 bytes, which is the cache-line
/// size in NVidia GPUs.
const std::size_t block_size_bits = 128 * 8;

/// Computes the filter overhead for a given configuration F.
///   lookup_time(F) + fpr(F) ∗ work_time
double
filter_overhead(
    const std::size_t m, // filter size in bits
    const std::size_t n, // the number of keys
    const int32_t k,     // the number of hash functions
    const double tw,     // work time in nano seconds
    const std::vector<double>& nanos_per_lookup_lut
  ) {
  auto search = std::lower_bound(filter_sizes.begin(), filter_sizes.end(), m);
  if (search == filter_sizes.end()) {
    throw std::invalid_argument(
        "Cannot determine the filter overhead for the given filter size.");
  }
  const auto lut_idx = std::distance(filter_sizes.begin(), search);

  // Note: In case of a GTX 10xy GPU, the (amortized) lookup time mostly
  // depends on the filter size. The other parameters, such as the number of
  // hash functions or the block size, have a negligible influence on the
  // overall lookup performance and can therefore be ignored.
  const auto lookup_time = nanos_per_lookup_lut[lut_idx];

  // Compute the FPR.
  const auto fpr = dtl::bloomfilter::fpr_blocked(m, n, k, block_size_bits);

  // Compute the filter overhead.
  const auto overhead = lookup_time + fpr * tw;
  return overhead;
}

struct filter_info {
  /// Filter size bits.
  std::size_t m = 0;
  /// The number of hash functions.
  uint32_t k = 1;
  /// The expected false positive rate.
  double f = 1.0;
  /// The filter overhead in nanos.
  double overhead = std::numeric_limits<double>::max();
};

filter_info
determine_filter_params(
    const std::size_t n, // filter size in bits
    const double tw,     // work time in nano seconds
    const std::vector<double>& nanos_per_lookup_lut
    ) {

  filter_info opt;
  opt.overhead = std::numeric_limits<double>::max();

  for (auto m : filter_sizes) {
    for (uint32_t k = 1; k < 16; ++k) {
      const auto o = filter_overhead(m, n, k, tw, nanos_per_lookup_lut);
      if (o < opt.overhead) {
        opt.overhead = o;
        opt.m = m;
        opt.k = k;
        opt.f = dtl::bloomfilter::fpr_blocked(m, n, k, block_size_bits);
      }
    }
  }
  return opt;
}

/// The different types of GPUs we support.
enum class gpu_t {
  GTX_1050_Ti,
  GTX_1080,
  GTX_1080_Ti,
};

enum class gpu_mode_t {
  /// Probe keys a transferred via PCIe.
  PCIe,
  /// Probe keys reside in GPU memory.
  Cached,
};

filter_info
determine_filter_params(
    const std::size_t n,         // filter size in bits
    const double tw,             // work time in nano seconds
    const gpu_t gpu_type,        // the GPU type
    const gpu_mode_t gpu_mode    // are the keys transferred to the GPu or are they already there?
    ) {

  if (gpu_mode == gpu_mode_t::PCIe) {
    switch (gpu_type) {
      case gpu_t::GTX_1050_Ti: return determine_filter_params(n, tw, nanos_per_lookup_gtx_1050_ti);
      case gpu_t::GTX_1080:    return determine_filter_params(n, tw, nanos_per_lookup_gtx_1080);
      case gpu_t::GTX_1080_Ti: return determine_filter_params(n, tw, nanos_per_lookup_gtx_1080_ti);
    }
  }
  if (gpu_mode == gpu_mode_t::Cached) {
    switch (gpu_type) {
      case gpu_t::GTX_1050_Ti: return determine_filter_params(n, tw, nanos_per_lookup_gtx_1050_ti_cached);
      case gpu_t::GTX_1080:    return determine_filter_params(n, tw, nanos_per_lookup_gtx_1080_cached);
      case gpu_t::GTX_1080_Ti: return determine_filter_params(n, tw, nanos_per_lookup_gtx_1080_ti_cached);
    }
  }
  throw std::invalid_argument("Failed to find an appropriate lookup table.");
}

int32_t main() {
  // The number of keys.
  const std::size_t n = 10000000;
  // The time we safe, with each filtered tuple.
  const auto work_time_nanos = 1.0;

  {
    const auto info = determine_filter_params(n, work_time_nanos, gpu_t::GTX_1080, gpu_mode_t::Cached);
    std::cout << "m=" << info.m <<  " (=" << (info.m /1024/1024/8) << "MiB)"
        << ", k=" << info.k
        << " (fpr=" << info.f
        << ", overhead=" << info.overhead << "ns)" << std::endl;
  }

  {
    const auto info = determine_filter_params(n, work_time_nanos, gpu_t::GTX_1050_Ti, gpu_mode_t::PCIe);
    std::cout << "m=" << info.m <<  " (=" << (info.m /1024/1024/8) << "MiB)"
        << ", k=" << info.k
        << " (fpr=" << info.f
        << ", overhead=" << info.overhead << "ns)" << std::endl;
  }
}
