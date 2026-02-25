#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"
#include "bench_utils.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

struct alignas(32) Float8 {
  float elems[8];
};
static_assert(sizeof(Float8) == 32, "Float8 must be 32 bytes");
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

constexpr int kVectorPerThread = 4;

__global__ void copyScalarVec8(const Float8 *__restrict__ in,
                               Float8 *__restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  const int n_vec = n / 8; // n floats → n/8 Float8 elements
  for (int i = idx * kVectorPerThread; i < n_vec;
       i += stride * kVectorPerThread) {
#pragma unroll
    for (int j = 0; j < kVectorPerThread; j++) {
      out[i + j] = in[i + j]; // 32-byte load/store per iter, 128B per thread
    }
  }
}

void copyScalarVec8HBMBench(nvbench::state &state) {
  const auto num_floats = static_cast<int>(state.get_int64("NumFloats"));
  const auto grid_size = static_cast<int>(state.get_int64("GridSize"));
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
  dim3 grid(grid_size);

  // ---- tell nvbench the memory traffic for bandwidth reporting --------------
  state.add_global_memory_reads<float>(num_floats, "Read");
  state.add_global_memory_writes<float>(num_floats, "Write");

  // ---- benchmark -----------------------------------------------------------
  NVTX_RANGE("compute_kernel:copyScalarVec8");
  state.exec([&](nvbench::launch &launch) {
    copyScalarVec8<<<grid, block, 0, launch.get_stream()>>>(
        reinterpret_cast<const Float8 *>(thrust::raw_pointer_cast(d_in.data())),
        reinterpret_cast<Float8 *>(thrust::raw_pointer_cast(d_out.data())),
        num_floats);
  });

#ifdef VERIFY
  thrust::host_vector<float> h_verify(d_out); // D2H copy
  printf("Max abs diff vs src: %.6e\n", max_abs_diff(h_data, h_verify));
#endif
}

NVBENCH_BENCH(copyScalarVec8HBMBench)
    .add_int64_axis("NumFloats", {16 << 20, 32 << 20, 64 << 20})
    .add_int64_axis("GridSize", {512, 1024, 2048})
    .add_int64_axis("BlockSize", {128, 256, 512});
