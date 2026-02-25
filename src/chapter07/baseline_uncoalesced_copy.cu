// baseline_copy_uncoalesced.cu -- uncoalesced identity copy baseline for
// Chapter 7.

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"
#include "bench_utils.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Transpose access pattern: view the array as a 32 x (n/32) matrix and read
// it column-by-column while writing row-by-row.
//
// For a warp of 32 threads with consecutive i values:
//   col = {0, 1, ..., 31}  →  src idx = {0, 32, 64, ..., 31*32}
//
// Each thread touches a different 128-byte cache line → 32 transactions/warp
// instead of 1, the worst-case uncoalesced scenario.
//
// The mapping (row, col) → col*32 + row is a bijection over [0, n), so every
// element is read and written exactly once (correct copy, max diff = 0).
__global__ void uncoalescedCopy(const float *__restrict__ in,
                                float *__restrict__ out, int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_stride = gridDim.x * blockDim.x;
  const int ncols = n / 32; // assumes n % 32 == 0

  for (int i = tid; i < n; i += grid_stride) {
    const int row = i / ncols;
    const int col = i % ncols;
    out[i] = in[col * 32 + row]; // gather: uncoalesced read, coalesced write
  }
}

void uncoalescedCopyBench(nvbench::state &state) {
  const auto num_floats = static_cast<int>(state.get_int64("NumFloats"));
  const auto block_size = static_cast<int>(state.get_int64("BlockSize"));

  // ---- allocate and initialize ---------------------------------------------
  NVTX_RANGE("initialize");
  thrust::host_vector<float> h_data(num_floats);
  for (int i = 0; i < num_floats; ++i) {
    h_data[i] = static_cast<float>(i);
  }

  thrust::device_vector<float> d_in(h_data); // H2D copy
  thrust::device_vector<float> d_out(num_floats);

  dim3 block(block_size);
  dim3 grid((num_floats + block_size - 1) / block_size);

  // ---- tell nvbench the memory traffic for bandwidth reporting --------------
  state.add_global_memory_reads<float>(num_floats, "Read");
  state.add_global_memory_writes<float>(num_floats, "Write");

  // ---- benchmark -----------------------------------------------------------
  NVTX_RANGE("compute_kernel:uncoalescedCopy");
  state.exec([&](nvbench::launch &launch) {
    uncoalescedCopy<<<grid, block, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()), num_floats);
  });

#ifdef VERIFY
  thrust::host_vector<float> h_verify(d_out); // D2H copy
  const auto verify_checksum = checksum(h_verify);
  const auto data_checksum = checksum(h_data);
  printf("Output checksum: %.6f\n", verify_checksum);
  printf("Input checksum: %.6f\n", data_checksum);
  printf("Abs diff vs src checkdum: %.6e\n",
         abs(verify_checksum - data_checksum));
#endif
}

NVBENCH_BENCH(uncoalescedCopyBench)
    .add_int64_axis("NumFloats", {16 << 20, 32 << 20, 64 << 20})
    .add_int64_axis("BlockSize", {128, 256, 512});
