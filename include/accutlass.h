/*
 * Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#pragma once

// macros for cutlass changes required by ali customized logic, must be defined before cutlass includes
#ifndef ACOMPUTE_VERSION
#define ACOMPUTE_VERSION 10000
#endif
#define SAIL_PPU_MMA                 1
// [TODO] workaround code temporarily, will double check later.
#define SAIL_TMP_WORKAROUND          1

#define SAIL_ENABLE_CUTLASS_SIMT     0
#define SAIL_SIMULATE_CUTLASS_MMA    0
#define SAIL_CUSTOMIZE_CUTLASS       1
#define SAIL_DGRAD_STRIDE_OPT        1
#define SAIL_WGRAD_ITER_OPT          1
#define SAIL_VALID_COPY_ONLY         0  // avoid invalid copy in pipeline/multistage, have some problem when main loop < stage
#define SAIL_EPILOGUE_OPT            2  // >=1: skipped identity output op; >=2: bring type convert before writing to shm
#define SAIL_TYPE_CONVERT            1  // 0: use c-code RTN; 1: use build-in intrinsic rna; 2: use build-in intrinsic, 3: use c-code RTN w/o check inf.
#define SAIL_REDUCE_SPLITK_OPT       1
#define SAIL_FUSE_OP_EXT             1
#define SAIL_FUSE_ALIGN_CUDNN        1  // align to cudnn's fusion flow: conv_result(FP32 -> FP16) + Add(FP16) + BiasAdd(FP16) + ReLu
