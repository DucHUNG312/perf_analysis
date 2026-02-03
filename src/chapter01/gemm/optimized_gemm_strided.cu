// optimized_gemm_strided.cu - Strided Batched GEMM (most optimized)
// Best for uniform matrices: contiguous memory layout, single kernel launch

#include "../../core/common/cuda_check.h"
#include "../../core/common/nvtx_utils.cuh"
#include "fill_matrix.cuh"
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main() {
  NVTX_RANGE("main");
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

  int m = 32;
  int n = 256;
  int k = 256;
  int batch_count = 40;

  // Allocate contiguous memory for all matrices
  float *d_A, *d_B, *d_C;
  size_t stride_A = m * k;
  size_t stride_B = k * n;
  size_t stride_C = m * n;

  CUDA_CHECK(cudaMalloc(&d_A, stride_A * batch_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, stride_B * batch_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, stride_C * batch_count * sizeof(float)));

  fill_matrix(d_A, stride_A * batch_count, 1.0f);
  fill_matrix(d_B, stride_B * batch_count, 1.0f);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Warmup
  const float alpha = 1.0f, beta = 0.0f;
  const cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_TF32;
  const cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, CUDA_R_32F, n,
      static_cast<long long>(stride_B), d_A, CUDA_R_32F, k,
      static_cast<long long>(stride_A), &beta, d_C, CUDA_R_32F, n,
      static_cast<long long>(stride_C), batch_count, compute, algo));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < 100; ++iter) {
    NVTX_RANGE("compute_math:gemmstridedbatched");
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, CUDA_R_32F, n,
        static_cast<long long>(stride_B), d_A, CUDA_R_32F, k,
        static_cast<long long>(stride_A), &beta, d_C, CUDA_R_32F, n,
        static_cast<long long>(stride_C), batch_count, compute, algo));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  float time_strided = ms / 100.0f;
  float tflops_strided =
      (2.0f * m * n * k * batch_count) / (time_strided * 1e9);

  std::printf(
      "=== Optimized: Strided Batched GEMM (cublasGemmStridedBatchedEx) ===\n");
  std::printf("Matrix dimensions: M=%d, N=%d, K=%d, Batch=%d\n", m, n, k,
              batch_count);
  std::printf("Time: %.3f ms\n", time_strided);
  std::printf("Performance: %.2f TFLOPS\n", tflops_strided);
  std::printf(
      "Kernel launches: 1 per iteration (strided, contiguous memory)\n");

  // Cleanup
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUBLAS_CHECK(cublasDestroy(handle));

  return 0;
}
