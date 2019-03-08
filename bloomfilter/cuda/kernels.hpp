//===----------------------------------------------------------------------===//
template <typename filter_t>
__global__ void contains_naive_kernel(
    const filter_t filter,
    const typename filter_t::word_t *__restrict__ word_array,
    u32 *__restrict__ keys, u32 key_cnt, $u32 *__restrict__ result_bitmap) {
    // who am I?
    u32 wid = global_warp_id();
    u32 lid = warp_local_thread_id();

    // the output this executing thread wi  ll write later on (kept in a register)
    $u32 thread_local_bitmap = 0u;  // out  put

    constexpr u32 elements_per_thread = warp_size;  // ... processed sequentially
    constexpr u32 elements_per_warp = elements_per_thread * warp_size;

    // where to star    t?
    $u32 read_pos = wid * elements_per_warp + lid;

    // each thread processes multiple elements sequentially
    for ($u32 i = 0; i != elements_per_thread; i++) {
        auto is_contained = (read_pos < key_cnt) ? filter.contains(word_array, keys[read_pos]) : false;
        u32 bitmap = __ballot_sync(0xffffffff, is_contained);
        thread_local_bitmap = (lid == i) ? bitmap : thread_local_bitmap;
        read_pos += warp_size;
    }
    __syncwarp();
    // every thread writes a single word of the output bitmap
    u32 write_pos = global_thread_id();
    result_bitmap[write_pos] = thread_local_bitmap;
}

//===----------------------------------------------------------------------===//
template <typename filter_t>
__global__ void contains_clustered_kernel(
    const filter_t filter,
    const typename filter_t::word_t *__restrict__ word_array,
    u32 *__restrict__ keys,
        u32 *__restrict__ hash_keys,
    u32 key_cnt, $u32 *__restrict__ result_bitmap,
    $u32 *__restrict__ positions) {
    // who am I?
    u32 warp_id = global_warp_id();
    u32 local_thread_id = warp_local_thread_id();

    // the output this executing thread will write later on (kept in a register)
    $u32 thread_local_bitmap = 0u; // output

    constexpr u32 elements_per_thread = warp_size; // ... processed sequentially
    constexpr u32 elements_per_warp = elements_per_thread * warp_size;

    // where to start?
    $u32 read_pos = warp_id * elements_per_warp + local_thread_id;

    // each thread processes multiple elements sequentially
    for ($u32 i = 0; i != elements_per_thread; i++) {
      // 
        if (read_pos < key_cnt){
            result_bitmap[read_pos] = filter.contains_with_hash(word_array, hash_keys[read_pos]);
        }
        read_pos += warp_size;
    }
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
template <typename filter_t>
__global__ void calculate_hash_kernel(
    const filter_t filter, u32 *__restrict__ keys, u32 key_cnt,
    $u32 *__restrict__ hash_block_vector,
    $u32 *__restrict__ positions_block_vector) {

    int read_pos = blockIdx.x *blockDim.x + threadIdx.x;
    size_t stride = (blockDim.x * gridDim.x); // Grid-Stride
    for (size_t i = read_pos; i < key_cnt; i += stride) {
        hash_block_vector[i] = ($u32)filter.hash(keys[i]);
        positions_block_vector[i] = i;
    }
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
template <typename filter_t>
__global__ void contains_shared(
    const filter_t filter,
    const typename filter_t::word_t *__restrict__ word_array,
    u32 *__restrict__ keys, u32 key_cnt, $u32 *__restrict__ result_bitmap) {
  // who am I?
  u32 wid = global_warp_id();
  u32 lid = warp_local_thread_id();

  // the output this executing thread will write later on (kept in a register)
  $u32 thread_local_bitmap = 0u;  // output

  constexpr u32 elements_per_thread = warp_size;  // ... processed sequentially
  constexpr u32 elements_per_warp = elements_per_thread * warp_size;

  // where to start?
  $u32 read_pos = wid * elements_per_warp + lid;

  // each thread processes multiple elements sequentially
  for ($u32 i = 0; i != elements_per_thread; i++) {
    /*  	const size_t bitvector_word_idx = keys[read_pos]
                                       << filter.word_cnt_per_block_log2;
   const auto block_ptr = &word_array[bitvector_word_idx];*/
    auto is_contained = (read_pos < key_cnt)
                            ? filter.contains_sorted(word_array, keys[read_pos])
                            : false;
    u32 bitmap = __ballot_sync(0xffffffff, is_contained);
    thread_local_bitmap = (lid == i) ? bitmap : thread_local_bitmap;
    read_pos += warp_size;
  }
  __syncwarp();
  // every thread writes a single word of the output bitmap
  u32 write_pos = global_thread_id();
  result_bitmap[write_pos] = thread_local_bitmap;
}
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
template <typename filter_t>
__global__ void build_candidate_list($u32 *__restrict__ result_bitmap,
                                     $u32 *__restrict__ positions)
{
    
}