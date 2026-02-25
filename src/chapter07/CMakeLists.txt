include(cc_binary)

cc_binary(
  NAME
    baseline_copy_scalar
  SRCS
    baseline_copy_scalar.cu
  DEPS
    nvbench::main
)

cc_binary(
  NAME
    baseline_hbm_copy
  SRCS
    baseline_hbm_copy.cu
  DEPS
    nvbench::main
)

cc_binary(
  NAME
    baseline_uncoalesced_copy
  SRCS
    baseline_uncoalesced_copy.cu
  DEPS
    nvbench::main
)

cc_binary(
  NAME
    optimized_copy_scalar_vectorized
  SRCS
    optimized_copy_scalar_vectorized.cu
  DEPS
    nvbench::main
)

cc_binary(
  NAME
    optimized_hbm_copy
  SRCS
    optimized_hbm_copy.cu
  DEPS
    nvbench::main
)

cc_binary(
  NAME
    baseline_async_prefetch
  SRCS
    baseline_async_prefetch.cu
  DEPS
    nvbench::main
)

cc_binary(
  NAME
    optimized_async_prefetch
  SRCS
    optimized_async_prefetch.cu
  DEPS
    nvbench::main
)
