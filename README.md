![ALT](/media/images/gemm-hierarchy-with-epilogue-no-labels-ppu.png "Complete GEMM decomposition")

# ACTLIZE 0.8.0 for PPU

## General Introduction for ACTLIZE

_ACTLIZE 0.8.0 - PPU Adaptation (July 2026)_

ACTLIZE 0.8.0 for PPU is a specialized adaptation of [CUTLASS 3.4.1](https://github.com/NVIDIA/cutlass),
originally developed by NVIDIA, re-engineered for PPU architectures.


## PPU-Specific Notes and Important Information

### Overview

The changes focused on enabling high-performance GEMM operations on PPU hardware, particularly through AIU tensor cell operations.

### Important Notes

**The currently open-sourced PPU ACTLIZE 0.8.0 is retained solely to ensure the completeness of existing projects that depend on it (e.g., flash-attention2.7.2, FlashMLA). PPU support for this version is incomplete — ACTLIZE codes not instantiated by afore-mentioned projects are not guaranteed for correctness or performance. Developers building new features depend on PPU ACTLIZE are encouraged to use ACTLIZE 1.0.0 instead**

**Performance Warning**: Upstream examples (e.g., using standard Tensor cell instructions) are not optimized for PPU hardware and should not be used for performance evaluation on PPU


### Prerequisites

- **PPU Hardware:** ZW 610 / 610E / 810 / 810E / M890
- **PPU SDK**
- **CMake:** 3.19+
- **Host Compiler:** C++17 capable (GCC 9.4+)


### Key PPU-Specific Features

- PPU1.0/1.5 Hardware Capabilities
  - AIU load/tsm load swizzle operations
  - PPU1.0/1.5-specific MMA traits

- Collective Kernels
  - Normal GEMM with PPU collectives
  - Array, group, and StreamK GEMM kernels with multi-stage pipeline

- Data Type Support
  - TF32/FP16/BF16/INT8 Tensor OP GEMM for PPU1.0/1.5
  - Mixed datatype input GEMM
  - FP8 GEMM for PPU1.5
  - FP4 GEMM for PPU1.5

- Epilogue & Activation
  - PPU-optimized epilogue operations

- Testing & Examples
  - AIU unit tests
  - Example applications


## Performance Notes

**Important**: Upstream examples in this repository are provided for reference only. Even if they pass verification on PPU hardware, they do not represent expected or optimized performance. The PPU-specific implementations in `include/ppu/` should be used for production workloads if needed.


## Quick start to build and run ACTLIZE

**Important**: The PPU ACTLIZE is designed to be built with the PPU SDK. The default build configuration only enables PPU-specific examples and disables the CUTLASS Library and Profiler which are not applicable to PPU.

Create a build directory within the ACTLIZE project, then run CMake.

```bash
$ mkdir build && cd build
# examples/ut only support PPU1.0
$ cmake .. -DCUTLASS_PPU_ARCHS=ppu0010
$ make -j8
```
To simply run the examples/ut:

```bash
$ ./run.sh
```

All tests should pass on supported platforms.


# Copyright


Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.

This product contains various third-party components under other open source licenses:

Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

Portions (acrand_kernel.h and related files) are derived from [rocRAND](https://github.com/ROCm/rocRAND).

Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```