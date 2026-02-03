#include "../../core/common/cuda_check.h"
#include "../../core/common/nvtx_utils.cuh"
#include "fill_matrix.cuh"
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

// Benchmark individual GEMM calls (simulating original PyTorch behavior)
int main() {
  NVTX_RANGE("main");
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

  // Simulate the workload from profiling: 40 GEMMs with typical NN dimensions
  int m = 32;  // batch size
  int n = 256; // output features
  int k = 256; // input features
  int batch_count = 40;

  std::vector<float *> d_A(batch_count), d_B(batch_count), d_C(batch_count);

  // Allocate matrices for each GEMM
  for (int i = 0; i < batch_count; ++i) {
    NVTX_RANGE("setup");
    CUDA_CHECK(cudaMalloc(&d_A[i], m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B[i], k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C[i], m * n * sizeof(float)));

    // Initialize with dummy data
    fill_matrix(d_A[i], m * k, 1.0f);
    fill_matrix(d_B[i], k * n, 1.0f);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Warmup
  const float alpha = 1.0f, beta = 0.0f;
  for (int i = 0; i < batch_count; ++i) {
    NVTX_RANGE("compute_math:sgemm");
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                             d_B[i], n, d_A[i], k, &beta, d_C[i], n));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < 100; ++iter) {
    NVTX_RANGE("batch");
    for (int i = 0; i < batch_count; ++i) {
      NVTX_RANGE("compute_math:sgemm");
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                               &alpha, d_B[i], n, d_A[i], k, &beta, d_C[i], n));
    }
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  float time_individual = ms / 100.0f;
  float tflops_individual =
      (2.0f * m * n * k * batch_count) / (time_individual * 1e9);

  std::printf("=== Baseline: Individual GEMM Calls ===\n");
  std::printf("Matrix dimensions: M=%d, N=%d, K=%d, Batch=%d\n", m, n, k,
              batch_count);
  std::printf("Time: %.3f ms\n", time_individual);
  std::printf("Performance: %.2f TFLOPS\n", tflops_individual);
  std::printf("Kernel launches: %d per iteration\n", batch_count);

  // Cleanup
  for (int i = 0; i < batch_count; ++i) {
    NVTX_RANGE("cleanup");
    CUDA_CHECK(cudaFree(d_A[i]));
    CUDA_CHECK(cudaFree(d_B[i]));
    CUDA_CHECK(cudaFree(d_C[i]));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUBLAS_CHECK(cublasDestroy(handle));

  return 0;
}