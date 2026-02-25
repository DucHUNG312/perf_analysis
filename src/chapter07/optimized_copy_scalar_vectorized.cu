#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"
#include "bench_utils.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

__global__ void copyScalarVectorized(const float *__restrict__ in,
                                     float *__restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx * 4 < n) {
    *reinterpret_cast<float4 *>(&out[idx * 4]) =
        *reinterpret_cast<const float4 *>(&in[idx * 4]); // 16-byte load/store
  }
}

void copyScalarVectorizedBench(nvbench::state &state) {
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
  dim3 grid(((num_floats / 4) + block_size - 1) / block_size);

  // ---- tell nvbench the memory traffic for bandwidth reporting --------------
  state.add_global_memory_reads<float>(num_floats, "Read");
  state.add_global_memory_writes<float>(num_floats, "Write");

  // ---- benchmark -----------------------------------------------------------
  NVTX_RANGE("compute_kernel:copyScalarVectorized");
  state.exec([&](nvbench::launch &launch) {
    copyScalarVectorized<<<grid, block, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()), num_floats);
  });

#ifdef VERIFY
  thrust::host_vector<float> h_verify(d_out); // D2H copy
  printf("Max abs diff vs src: %.6e\n", max_abs_diff(h_data, h_verify));
#endif
}

NVBENCH_BENCH(copyScalarVectorizedBench)
    .add_int64_axis("NumFloats", {16 << 20, 32 << 20, 64 << 20})
    .add_int64_axis("BlockSize", {128, 256, 512});
