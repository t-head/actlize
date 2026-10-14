# ACTLIZE Changelog

## 0.5.0+v0.2.0

Changes since 0.5.0+v0.1.0 for the ACTLIZE 0.5.0 series.

- Fixed intermediate 32-bit overflow when computing output byte strides by converting the layout stride to `LongIndex` before multiplication.
  + Updated the [predicated tile iterator](./include/cutlass/epilogue/threadblock/predicated_tile_iterator.h).
  + Applied the same fix to the [direct-convolution](./include/cutlass/epilogue/threadblock/predicated_tile_iterator_direct_conv.h) and [strided](./include/cutlass/epilogue/threadblock/predicated_tile_iterator_strided.h) output iterators.
- Replaced compiler-version checks with `HGGCRT_VERSION` checks in affected headers, including [numeric conversions](./include/cutlass/numeric_conversion.h), [FP8 conversions](./include/cutlass/float8.h), and [memory operations](./include/cutlass/arch/memory.h).
- Updated [128-bit integer exception handling](./include/cutlass/uint128.h) to use `__brkpt()` on the device while retaining the host-side exception. Guarded [`device_breakpoint()`](./include/cutlass/arch/arch.h) so that `__brkpt()` is compiled only for the device.
- Removed commented-out prefetch code from the [implicit GEMM convolution](./include/cutlass/conv/kernel/implicit_gemm_convolution.h) and [fused convolution](./include/cutlass/conv/kernel/implicit_gemm_convolution_fusion.h) kernels, and removed disabled upstream macros.
- Corrected the [README](./README.md) by removing FP8 and FP4 MMA support claims from the ACTLIZE 0.5.0 feature list.
