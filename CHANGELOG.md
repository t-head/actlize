# ACTLIZE Changelog

## 1.0.0+v0.2.0

Changes since 1.0.0+v0.1.0 for the ACTLIZE 1.0.0 series.

- Fixed the global atomic-add instruction used by the [inter-threadblock barrier](./include/cutlass/barrier.h).
- Fixed shared-memory stride selection for AIU loads with swizzling disabled in [GEMM operand configuration](./include/cutlass/gemm/config/gemm_operands.hpp).
- Updated device-property queries for HGGCRT 13 in the [profiler](./tools/profiler/src/options.cu) and [unit-test utilities](./test/unit/common/filter_architecture.cpp).
  + Added [`acDeviceProp` and `acGetDeviceProperties`](./include/cutlass/ac_device_properties.h) to query device and memory clock rates through runtime attributes on HGGCRT 13 and newer.
  + Preserved the device-property field-based queries for earlier runtime versions.
- Replaced compiler-version checks with `HGGCRT_VERSION` checks in affected headers, including [numeric conversions](./include/cutlass/numeric_conversion.h), [FP8 conversions](./include/cutlass/float8.h), and the [PPU host adapter](./include/cutlass/ppu_host_adapter.hpp).
- Updated [128-bit integer exception handling](./include/cutlass/uint128.h) to use `__brkpt()` on the device while retaining `abort()` on the host. Guarded [`device_breakpoint()`](./include/cutlass/arch/arch.h) so that `__brkpt()` is compiled only for the device.
- Updated PPU architecture names and the GEMM instruction-shape description in the [README](./README.md), and revised source comments to use PPU tensor-cell terminology.
