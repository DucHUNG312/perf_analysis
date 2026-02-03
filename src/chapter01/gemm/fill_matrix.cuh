#pragma once

#include "../../core/common/cuda_check.h"

struct alignas(32) Float8 {
  float v[8];
};
static_assert(sizeof(Float8) == 32, "Float8 must pack eight floats");
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

__global__ void fill_matrix_kernel(float *data, int elements, float value) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int vec_elems = elements / 8;
  Float8 vec_value;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    vec_value.v[i] = value;
  }
  Float8 *vec_ptr = reinterpret_cast<Float8 *>(data);
  for (int idx = thread_id; idx < vec_elems; idx += stride) {
    vec_ptr[idx] = vec_value;
  }
  const int tail_start = vec_elems * 8;
  for (int idx = tail_start + thread_id; idx < elements; idx += stride) {
    data[idx] = value;
  }
}

void fill_matrix(float *data, int elements, float value) {
  if (elements <= 0) {
    return;
  }
  const int threads = 256;
  int blocks = (elements + threads * 8 - 1) / (threads * 8);
  if (blocks <= 0) {
    blocks = 1;
  }
  fill_matrix_kernel<<<blocks, threads>>>(data, elements, value);
  CUDA_CHECK(cudaGetLastError());
}