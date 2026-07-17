# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set(CUTLASS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_custom_target(cutlass_examples)

function(cutlass_example_add_executable NAME)

  cutlass_add_executable(${NAME} ${ARGN})

  add_dependencies(cutlass_examples ${NAME})

  target_link_libraries(
    ${NAME}
    PRIVATE
    CUTLASS
    cutlass_tools_util_includes
    ppu::driver
    )

  target_include_directories(
    ${NAME}
    PRIVATE
    ${CUTLASS_EXAMPLES_COMMON_SOURCE_DIR}
    ${CUTLASS_EXAMPLES_UTILS_DIR}
    )

  install(
    TARGETS ${NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

endfunction()

foreach(EXAMPLE
  00_ppu_basic_tensor_op_gemm
  01_ppu_gemm_with_collective_builder
  02_ppu_mixed_dtype_gemm
  03_ppu_ptr_array_batched_gemm
  04_ppu_grouped_gemm
  05_ppu_cute_gemm
  )

  add_subdirectory(${EXAMPLE})

endforeach()
