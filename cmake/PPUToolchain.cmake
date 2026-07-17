# Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
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

#
# PPUToolchain.cmake
# ------------------
# Single source of truth for the PPU device-toolchain bootstrap used by
# this open-source cutlass fork.  The device-compile driver is the SDK's
# hgcc binary (the PPU device and host compilation driver).
#
# Build model:
#   * Host .cpp files are compiled by the system CXX compiler (g++).
#   * Device .cu files are compiled via add_custom_command(COMMAND hgcc ...)
#     producing .o object files that link directly into the CXX targets.
#
# Responsibilities:
#   * locate the PPU SDK installation,
#   * provide add_custom_command-based .cu compilation via hgcc,
#   * declare imported library targets backed by the PPU runtime libs,
#   * provide the helper functions (cutlass_add_library,
#     cutlass_add_executable, cutlass_target_sources)
#     that the rest of the build expects.
#



###############################################################################
# 1. Locate the PPU SDK
###############################################################################

if (NOT DEFINED PPU_SDK_ROOT)
  if (DEFINED ENV{PPU_SDK})
    set(PPU_SDK_ROOT "$ENV{PPU_SDK}")
  elseif (DEFINED ENV{PPU_HOME})
    set(PPU_SDK_ROOT "$ENV{PPU_HOME}")
  else()
    set(PPU_SDK_ROOT "/usr/local/PPU_SDK")
  endif()
endif()

set(PPU_SDK_BIN     "${PPU_SDK_ROOT}/bin")
set(PPU_SDK_INCLUDE "${PPU_SDK_ROOT}/include")
set(PPU_SDK_LIB     "${PPU_SDK_ROOT}/lib")

set(PPU_SDK_TARGETS_INCLUDE "${PPU_SDK_ROOT}/targets/x86_64-linux/include")
if (NOT EXISTS "${PPU_SDK_TARGETS_INCLUDE}")
  message(WARNING
    "PPU SDK 'targets/x86_64-linux/include' missing at ${PPU_SDK_ROOT}; "
    "some SDK-internal headers may not be found.")
endif()

# Locate hgcc via system PATH (set by envsetup.sh) or PPU_SDK bin directory.
find_program(PPU_DEVICE_HGCC_REAL
  NAMES hgcc
  PATHS "${PPU_SDK_BIN}"
  REQUIRED
)

message(STATUS "PPU SDK Root    : ${PPU_SDK_ROOT}")
message(STATUS "PPU hgcc        : ${PPU_DEVICE_HGCC_REAL}")

###############################################################################
# 2. PPU build-mode flags
###############################################################################

set(CUTLASS_PPU_DEVICE_COMPILE ON CACHE BOOL
    "Using PPU device driver for compilation" FORCE)
set(CUTLASS_PPU_HOST_COMPILE   ON CACHE BOOL
    "PPU build is active (host=g++, device=hgcc)" FORCE)

###############################################################################
# 3. PPU device-side compiler flags
###############################################################################

set(_PPU_ALLOWED_ARCHS ppu0010 ppu0015)

if (DEFINED CUTLASS_PPU_ARCHS AND NOT "${CUTLASS_PPU_ARCHS}" STREQUAL "")
  string(REPLACE "," ";" _PPU_ARCHS_LIST "${CUTLASS_PPU_ARCHS}")
  string(REPLACE " " ";" _PPU_ARCHS_LIST "${_PPU_ARCHS_LIST}")
  list(FILTER _PPU_ARCHS_LIST EXCLUDE REGEX "^$")
else()
  set(_PPU_ARCHS_LIST "ppu0010;ppu0015")
endif()

foreach(_ARCH IN LISTS _PPU_ARCHS_LIST)
  if (NOT _ARCH IN_LIST _PPU_ALLOWED_ARCHS)
    message(FATAL_ERROR
      "CUTLASS_PPU_ARCHS contains '${_ARCH}' which is not in ${_PPU_ALLOWED_ARCHS}")
  endif()
endforeach()

set(_PPU_HGCC_ARCH_FLAGS)
foreach(_ARCH IN LISTS _PPU_ARCHS_LIST)
  if (_ARCH STREQUAL "ppu0010")
    list(APPEND _PPU_HGCC_ARCH_FLAGS "-arch=ppu_10")
  elseif (_ARCH STREQUAL "ppu0015")
    list(APPEND _PPU_HGCC_ARCH_FLAGS "-arch=ppu_15")
  endif()
endforeach()

list(GET _PPU_ARCHS_LIST 0 _PPU_ARCH_PRIMARY)
set(CUTLASS_PPU_ARCH "${_PPU_ARCH_PRIMARY}" CACHE STRING
    "Primary PPU architecture (first entry of CUTLASS_PPU_ARCHS)." FORCE)
unset(_PPU_ARCH_PRIMARY)

message(STATUS "PPU device archs: ${_PPU_ARCHS_LIST} (flags: ${_PPU_HGCC_ARCH_FLAGS})")

set(CUTLASS_PPU_ARCHS "${_PPU_ARCHS_LIST}" CACHE STRING
    "PPU architectures to build (resolved from input or default)." FORCE)

# Device compilation flags for hgcc.
#   -x hg                   : selects the PPU device language
#   --forward-unknown-to-host-compiler : let hgcc forward unknown host flags
#   --forward-unknown-to-host-linker   : same for link phase
#   -arch=ppu_XX            : selects the PPU device architecture
#   --expt-relaxed-constexpr : host/device constexpr interop
#
# IMPORTANT: do NOT inject -D__HGGC_ARCH__ on the command line.  The PPU
# device frontend auto-defines __HGGC_ARCH__ ONLY in its device path.
set(CUTLASS_PPU_EXTRA_HGCC_FLAGS
    "-x" "hg"
    "--forward-unknown-to-host-compiler"
    "--forward-unknown-to-host-linker"
    ${_PPU_HGCC_ARCH_FLAGS}
    "-DSWITCH_TO_HGGCRT"
    "-DHGGC_SDK_MODE"
    "-DUSE_CLANG"
    "-Xcompiler" "-ftemplate-depth=8192"
    "--expt-relaxed-constexpr"
    CACHE INTERNAL "PPU-specific hgcc device flags")

###############################################################################
# 4. PPU runtime libraries -> imported targets
###############################################################################

find_library(PPU_HG_WRAPPER_LIB   NAMES hg_wrapper   PATHS "${PPU_SDK_LIB}" NO_DEFAULT_PATH)
find_library(PPU_HGGC_WRAPPER_LIB NAMES hggc_wrapper PATHS "${PPU_SDK_LIB}" NO_DEFAULT_PATH)
find_library(PPU_HGGCRT1_LIB      NAMES hggcrt1      PATHS "${PPU_SDK_LIB}" NO_DEFAULT_PATH)
find_library(PPU_HGGC_LIB         NAMES hggc         PATHS "${PPU_SDK_LIB}" NO_DEFAULT_PATH)

message(STATUS "PPU libhg_wrapper   : ${PPU_HG_WRAPPER_LIB}")
message(STATUS "PPU libhggc_wrapper : ${PPU_HGGC_WRAPPER_LIB}")
message(STATUS "PPU libhggcrt1      : ${PPU_HGGCRT1_LIB}")
message(STATUS "PPU libhggc         : ${PPU_HGGC_LIB}")

set(PPU_RUNTIME_LIBS "")
if (PPU_HG_WRAPPER_LIB)
  list(APPEND PPU_RUNTIME_LIBS "${PPU_HG_WRAPPER_LIB}")
endif()
if (PPU_HGGC_WRAPPER_LIB)
  list(APPEND PPU_RUNTIME_LIBS "${PPU_HGGC_WRAPPER_LIB}")
endif()
if (PPU_HGGCRT1_LIB)
  list(APPEND PPU_RUNTIME_LIBS "${PPU_HGGCRT1_LIB}")
endif()
if (PPU_HGGC_LIB)
  list(APPEND PPU_RUNTIME_LIBS "${PPU_HGGC_LIB}")
endif()

if (NOT TARGET ppu_runtime)
  add_library(ppu_runtime INTERFACE IMPORTED GLOBAL)
  set_target_properties(ppu_runtime PROPERTIES
    INTERFACE_LINK_LIBRARIES "${PPU_RUNTIME_LIBS}"
    INTERFACE_INCLUDE_DIRECTORIES "${PPU_SDK_INCLUDE};${PPU_SDK_TARGETS_INCLUDE}")
  add_library(ppu::runtime ALIAS ppu_runtime)
endif()

###############################################################################
# 5. Helper functions (add_custom_command for .cu)
###############################################################################

function(_cutlass_split_sources _cu_var _cxx_var)
  set(_cu_list)
  set(_cxx_list)
  foreach(_s IN LISTS ARGN)
    if (_s MATCHES "\\.cu$")
      list(APPEND _cu_list "${_s}")
    elseif(NOT "${_s}" STREQUAL "")
      list(APPEND _cxx_list "${_s}")
    endif()
  endforeach()
  set(${_cu_var}  ${_cu_list}  PARENT_SCOPE)
  set(${_cxx_var} ${_cxx_list} PARENT_SCOPE)
endfunction()

macro(cutlass_build_dev_kernels source_list_var obj_list_var)
  set(_extra_dev_flags ${ARGN})
  set(_ppu_obj_dir "${CMAKE_CURRENT_BINARY_DIR}/ppu_obj")
  file(MAKE_DIRECTORY "${_ppu_obj_dir}")

  foreach(_ppu_src IN LISTS ${source_list_var})
    get_filename_component(_ppu_src_abs "${_ppu_src}" ABSOLUTE
                           BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    get_filename_component(_ppu_name    "${_ppu_src_abs}" NAME_WE)
    get_filename_component(_ppu_src_dir "${_ppu_src_abs}" DIRECTORY)
    string(SHA256 _ppu_hash "${_ppu_src_abs}")
    string(SUBSTRING "${_ppu_hash}" 0 8 _ppu_hash8)
    set(_ppu_obj "${_ppu_obj_dir}/${_ppu_name}_${_ppu_hash8}.o")
    add_custom_command(
      OUTPUT "${_ppu_obj}"
      MAIN_DEPENDENCY "${_ppu_src_abs}"
      COMMAND "${PPU_DEVICE_HGCC_REAL}"
              ${CUTLASS_PPU_EXTRA_HGCC_FLAGS}
              ${CUTLASS_PPU_DEV_INCLUDE_FLAGS}
              ${_extra_dev_flags}
              "-I${_ppu_src_dir}"
              "-I${CMAKE_CURRENT_SOURCE_DIR}"
              "-fPIC"
              -c "${_ppu_src_abs}" -o "${_ppu_obj}"
      COMMENT "[hgcc] ${_ppu_src}"
      VERBATIM
    )
    list(APPEND ${obj_list_var} "${_ppu_obj}")
  endforeach()
endmacro()

function(cutlass_add_library NAME)
  set(options)
  set(oneValueArgs EXPORT_NAME)
  set(multiValueArgs EXTRA_DEV_INCLUDES)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  _cutlass_split_sources(_cu_sources _cxx_sources ${__UNPARSED_ARGUMENTS})

  set(_dev_objs)
  if (_cu_sources)
    cutlass_build_dev_kernels(_cu_sources _dev_objs ${__EXTRA_DEV_INCLUDES})
  endif()

  add_library(${NAME} ${_cxx_sources} ${_dev_objs})
  set_target_properties(${NAME} PROPERTIES LINKER_LANGUAGE CXX)
  cutlass_apply_standard_compile_options(${NAME})
  target_compile_features(${NAME} INTERFACE cxx_std_11)

  if(__EXPORT_NAME)
    add_library(cutlass::${__EXPORT_NAME} ALIAS ${NAME})
    set_target_properties(${NAME} PROPERTIES EXPORT_NAME ${__EXPORT_NAME})
  endif()
endfunction()

function(cutlass_add_executable NAME)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs EXTRA_DEV_INCLUDES)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  _cutlass_split_sources(_cu_sources _cxx_sources ${__UNPARSED_ARGUMENTS})

  set(_dev_objs)
  if (_cu_sources)
    cutlass_build_dev_kernels(_cu_sources _dev_objs ${__EXTRA_DEV_INCLUDES})
  endif()

  add_executable(${NAME} ${_cxx_sources} ${_dev_objs})
  set_target_properties(${NAME} PROPERTIES LINKER_LANGUAGE CXX)
  cutlass_apply_standard_compile_options(${NAME})
  target_compile_features(${NAME} INTERFACE cxx_std_11)
endfunction()

function(cutlass_target_sources NAME)
  _cutlass_split_sources(_cu_sources _cxx_sources ${ARGN})

  set(_dev_objs)
  if (_cu_sources)
    cutlass_build_dev_kernels(_cu_sources _dev_objs)
  endif()

  target_sources(${NAME} PRIVATE ${_cxx_sources} ${_dev_objs})
endfunction()

function(cutlass_apply_standard_compile_options TARGET)
  target_compile_options(
    ${TARGET}
    PRIVATE
    -ftemplate-depth=8192
    )
endfunction()

