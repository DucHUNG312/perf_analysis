#pragma once

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"
#include <thrust/host_vector.h>
#include <vector>

inline float checksum(thrust::host_vector<float> &data) {
  double acc = 0.0;
  for (float v : data) {
    NVTX_RANGE("verify");
    acc += static_cast<double>(v);
  }
  return static_cast<float>(acc / static_cast<double>(data.size()));
}

inline float max_abs_diff(thrust::host_vector<float> &a,
                          thrust::host_vector<float> &b) {
  float max_err = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    max_err = fmaxf(max_err, fabsf(a[i] - b[i]));
  }
  return max_err;
}

inline float max_rel_diff(const thrust::host_vector<float> &a,
                          const thrust::host_vector<float> &b) {
  float max_err = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    float denom = fmaxf(fabsf(a[i]), 1.0f); // avoid div-by-zero
    max_err = fmaxf(max_err, fabsf(a[i] - b[i]) / denom);
  }
  return max_err;
}