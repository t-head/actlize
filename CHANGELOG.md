# ACTLIZE Changelog

## 0.8.0+v0.2.0

Changes since 0.8.0+v0.1.0 for the ACTLIZE 0.8.0 series.

- Added [`PPU0015_TSM_LD_SWZL_CVT`](./include/ppu/cute/arch/copy_ppu0015_aiu.hpp) and its [CuTe copy traits](./include/ppu/cute/atom/copy_traits_ppu0015_aiu.hpp) for paired-interleaved shared-memory layouts designed to avoid TSM group conflicts.
  + Supports 2-byte elements with a 16-by-64 cube shape; the device implementation requires `ACOMPUTE_VERSION >= 10500`.
  + Fixed address calculation for the designated `OddTile` in the non-transposed path and corrected parity calculation across pipeline stages.
  + Added a non-transposed specialization for `OddTile == -1` and explicitly restricted the transposed specialization to `OddTile == -1`.
- Replaced compiler-version checks with `HGGCRT_VERSION` checks in affected headers, including [memory operations](./include/cutlass/arch/memory.h), [FP8 conversions](./include/cutlass/float8.h), and [kernel launch support](./include/cutlass/device_kernel.h).
- Updated [128-bit integer exception handling](./include/cutlass/uint128.h) to use `__brkpt()` on the device while retaining `abort()` on the host. Guarded [`device_breakpoint()`](./include/cutlass/arch/arch.h) so that `__brkpt()` is compiled only for the device.
- Removed disabled upstream helper macros and replaced inline breakpoint assembly with `__brkpt()` in [CUTLASS barrier error paths](./include/cutlass/arch/barrier.h).
- Updated PPU architecture names in the [README](./README.md) and revised source comments to use PPU tensor-cell terminology.
