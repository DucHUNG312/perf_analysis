#include "../core/common/cuda_check.h"
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"
#include "bench_utils.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

constexpr int TILE_SIZE = 4096; // Larger tiles = more latency to hide

void light_compute_ref(const thrust::host_vector<float> &in,
                       thrust::host_vector<float> &out) {
  for (int i = 0; i < in.size(); i++) {
    float x = in[i];
    float y = x * 2.0f + 1.0f;
    y = y * y - x;
    y = y * 0.5f + x * 0.25f;
    out[i] = y;
  }
}

__device__ __forceinline__ float light_compute(float x) {
  // Simple arithmetic that completes quickly, exposing memory stalls
  float y = x * 2.0f + 1.0f;
  y = y * y - x;
  y = y * 0.5f + x * 0.25f;
  return y;
}

__global__ void nativeTiled(const float *__restrict__ in,
                            float *__restrict__ out, int n, int total_tiles) {
  extern __shared__ float smem[];

  // Process tiles assigned to this block
  for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    int tile_offset = tile * TILE_SIZE;
    int tile_elems = min(TILE_SIZE, n - tile_offset);

    // load tile to shared mem
    for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
      smem[i] = in[tile_offset + i];
    }
    __syncthreads(); // Must wait for all loads before compute

    // compute
    for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
      float v = smem[i];
      v = light_compute(v);
      out[tile_offset + i] = v;
    }
  }
}

void nativeTiledBench(nvbench::state &state) {
  const auto num_floats = static_cast<int>(state.get_int64("NumFloats"));
  const auto block_size = static_cast<int>(state.get_int64("BlockSize"));

  // ---- allocate and initialize ---------------------------------------------
  NVTX_RANGE("initialize");
  thrust::host_vector<float> h_data(num_floats);
  thrust::host_vector<float> h_data_ref(num_floats);
  for (int i = 0; i < num_floats; ++i) {
    h_data[i] = static_cast<float>(i % 1000) / 1000.0f;
  }
  light_compute_ref(h_data, h_data_ref);

  thrust::device_vector<float> d_in(h_data); // H2D copy
  thrust::device_vector<float> d_out(num_floats);

  const int total_tiles = (num_floats + TILE_SIZE - 1) / TILE_SIZE;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const size_t shared_bytes = TILE_SIZE * sizeof(float);
  int max_active_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_per_sm, nativeTiled, block_size, shared_bytes));
  const int grid_size = std::min(total_tiles, prop.multiProcessorCount *
                                                  max_active_blocks_per_sm);

  dim3 block(block_size);
  dim3 grid(grid_size);

  // ---- tell nvbench the memory traffic for bandwidth reporting --------------
  state.add_global_memory_reads<float>(num_floats, "Read");
  state.add_global_memory_writes<float>(num_floats, "Write");

  // ---- benchmark -----------------------------------------------------------
  NVTX_RANGE("compute_kernel:nativeTiled");
  state.exec([&](nvbench::launch &launch) {
    nativeTiled<<<grid, block, shared_bytes, launch.get_stream()>>>(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()), num_floats, total_tiles);
  });

#ifdef VERIFY
  thrust::host_vector<float> h_verify(d_out); // D2H copy
  printf("Max abs diff vs src: %.6e\n", max_abs_diff(h_data_ref, h_verify));
  printf("Max rel diff vs src: %.6e\n", max_rel_diff(h_data_ref, h_verify));
#endif
}

NVBENCH_BENCH(nativeTiledBench)
    .add_int64_axis("NumFloats", {16 << 20, 32 << 20, 64 << 20})
    .add_int64_axis("BlockSize", {64, 128, 256, 512});
