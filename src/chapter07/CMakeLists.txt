include(cc_binary)

cc_binary(
  NAME
    baseline_copy_scalar
  SRCS
    baseline_copy_scalar.cu
  DEPS
    nvbench::main
)