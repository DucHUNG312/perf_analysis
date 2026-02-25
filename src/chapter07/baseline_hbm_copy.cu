#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"
#include "bench_utils.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

__global__ void copyScalar(const float *__restrict__ in,
                           float *__restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = idx; i < n; i += stride) {
    out[i] = in[i]; // 4-byte transactions
  }
}

void copyScalarHBMBench(nvbench::state &state) {
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
  NVTX_RANGE("compute_kernel:copyScalar");
  state.exec([&](nvbench::launch &launch) {
    copyScalar<<<grid, block, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()), num_floats);
  });

#ifdef VERIFY
  thrust::host_vector<float> h_verify(d_out); // D2H copy
  printf("Max abs diff vs src: %.6e\n", max_abs_diff(h_data, h_verify));
#endif
}

NVBENCH_BENCH(copyScalarHBMBench)
    .add_int64_axis("NumFloats", {16 << 20, 32 << 20, 64 << 20})
    .add_int64_axis("GridSize", {1024, 2048})
    .add_int64_axis("BlockSize", {128, 256, 512});
