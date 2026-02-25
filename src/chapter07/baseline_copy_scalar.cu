// baseline_copy_scalar.cu - Scalar memory copy (Ch7)
//
// WHAT: Naive scalar loads - 1 float (4 bytes) per memory operation.
// Simple memory copy benchmark for comparing scalar vs vectorized.
//
// WHY THIS IS SLOWER:
//   - Each thread issues individual 4-byte loads/stores
//   - High instruction count per byte transferred
//   - Does NOT saturate HBM bandwidth

#include "../core/common/cuda_check.h"
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Scalar copy: 1 float per thread
__global__ void copyScalar(const float *__restrict__ in,
                           float *__restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx]; // 4-byte load, 4-byte store
  }
}

void copy_scalar_bench(nvbench::state &state) {
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
  NVTX_RANGE("compute_kernel:copyScalar");
  state.exec([&](nvbench::launch &launch) {
    copyScalar<<<grid, block, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()), num_floats);
  });

#ifdef VERIFY
  thrust::host_vector<float> h_verify(d_out); // D2H copy
  float checksum = 0.0f;
  VERIFY_CHECKSUM(thrust::raw_pointer_cast(h_verify.data()), num_floats,
                  &checksum);
  VERIFY_PRINT_CHECKSUM(checksum);
#endif
}

// nvbench runs the full Cartesian product of all axes:
//   3 NumFloats  x  3 BlockSize  =  9 configurations
NVBENCH_BENCH(copy_scalar_bench)
    .add_int64_axis("NumFloats", {16 << 20, 32 << 20, 64 << 20})
    .add_int64_axis("BlockSize", {128, 256, 512});
