#include "../core/common/cuda_check.h"
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"
#include "bench_utils.h"
#include <cooperative_groups.h>
#include <cstdio>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

namespace cg = cooperative_groups;

constexpr int TILE_SIZE = 4096; // Larger tiles = more latency to hide
constexpr int PIPELINE_STAGES = 2;

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

// Optimized: Double-buffered pipeline overlaps load(tile N+1) with compute(tile
// N)
__global__ void pipelinedKernel(const float *__restrict__ in,
                                float *__restrict__ out, int n,
                                int total_tiles) {
  extern __shared__ float smem[];
  float *stage_buf[PIPELINE_STAGES];
  for (int s = 0; s < PIPELINE_STAGES; s++) {
    stage_buf[s] = smem + s * TILE_SIZE;
  }

  cg::thread_block block = cg::this_thread_block();

  // set up pipeline with PIPELINE_STAGES stages
  using SharedState =
      cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_STAGES>;
  __shared__ alignas(
      SharedState) unsigned char pipe_storage[sizeof(SharedState)];
  auto *pipe_state = reinterpret_cast<SharedState *>(pipe_storage);
  if (threadIdx.x == 0) {
    new (pipe_state) SharedState();
  }
  block.sync();
  auto pipe = cuda::make_pipeline(block, pipe_state);

  // calculate tiles for this block
  const int tiles_per_block = (total_tiles + gridDim.x - 1) / gridDim.x;
  const int first_tile = blockIdx.x * tiles_per_block;
  const int last_tile = min(first_tile + tiles_per_block, total_tiles);

  auto issue_load = [&](int tile, int stage) {
    int tile_offset = tile * TILE_SIZE;
    int tile_elems = min(TILE_SIZE, n - tile_offset);
    // we need to ensure there is available slot in stage buffer
    pipe.producer_acquire();
    if (tile_elems > 0) {
      cuda::memcpy_async(block, stage_buf[stage], in + tile_offset,
                         tile_elems * sizeof(float), pipe);
    }
    pipe.producer_commit(); // this slot is now IN_FLIGHT, consumer can wait on
                            // it
  };

  // Prime the pipeline: load first PIPELINE_STAGES tiles
  for (int i = 0; i < PIPELINE_STAGES && (first_tile + i) < last_tile; ++i) {
    issue_load(first_tile + i, i);
  }

  // Main loop: compute current tile while loading next
  for (int tile = first_tile; tile < last_tile; ++tile) {
    int stage = (tile - first_tile) % PIPELINE_STAGES;
    int tile_offset = tile * TILE_SIZE;
    int tile_elems = min(TILE_SIZE, n - tile_offset);

    pipe.consumer_wait(); // Wait for this tile's data (block until the oldest
                          // committed load is done in smem)
    block.sync(); // ensuring every thread in the block sees the freshly written
                  // smem data

    // Compute while next tile is loading (overlap!)
    for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
      float v = stage_buf[stage][i];
      v = light_compute(v);
      out[tile_offset + i] = v;
    }

    // done compute, release so that we can start issue load the next tile
    pipe.consumer_release();

    // Issue load for tile + PIPELINE_STAGES (non-blocking)
    int next = tile + PIPELINE_STAGES;
    if (next < last_tile) {
      issue_load(next, stage);
    }

    // block.sync();
  }
}

void pipelinedKernelBench(nvbench::state &state) {
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
  const size_t shared_bytes = PIPELINE_STAGES * TILE_SIZE * sizeof(float);
  int max_active_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_per_sm, pipelinedKernel, block_size, shared_bytes));
  const int grid_size = std::min(total_tiles, prop.multiProcessorCount *
                                                  max_active_blocks_per_sm);

  dim3 block(block_size);
  dim3 grid(grid_size);

  // ---- tell nvbench the memory traffic for bandwidth reporting --------------
  state.add_global_memory_reads<float>(num_floats, "Read");
  state.add_global_memory_writes<float>(num_floats, "Write");

  // ---- benchmark -----------------------------------------------------------
  NVTX_RANGE("compute_kernel:pipelinedKernel");
  state.exec([&](nvbench::launch &launch) {
    pipelinedKernel<<<grid, block, shared_bytes, launch.get_stream()>>>(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()), num_floats, total_tiles);
  });

#ifdef VERIFY
  thrust::host_vector<float> h_verify(d_out); // D2H copy
  printf("Max abs diff vs src: %.6e\n", max_abs_diff(h_data_ref, h_verify));
  printf("Max rel diff vs src: %.6e\n", max_rel_diff(h_data_ref, h_verify));
#endif
}

NVBENCH_BENCH(pipelinedKernelBench)
    .add_int64_axis("NumFloats", {16 << 20, 32 << 20, 64 << 20})
    .add_int64_axis("BlockSize", {64, 128, 256, 512});
