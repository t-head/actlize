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

cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
cmake_policy(SET CMP0112 NEW)

if(cutlass_LOADED)
  # If CUTLASS has been previously fetched and loaded, don't do it again.
  return()
else()
  set(cutlass_LOADED ON)
  set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "CUTLASS Repository Directory")
endif()

message(STATUS "CMake Version: ${CMAKE_VERSION}")

# To reduce duplicate version locations, parse the version out of the
# main versions.h file and reuse it here.

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/cutlass/version.h VERSION_FILE_CONTENTS)
string(REGEX MATCH "#define CUTLASS_MAJOR ([0-9]+)" _CUTLASS_VERSION_MAJOR "${VERSION_FILE_CONTENTS}")
set(_CUTLASS_VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define CUTLASS_MINOR ([0-9]+)" _CUTLASS_VERSION_MINOR "${VERSION_FILE_CONTENTS}")
set(_CUTLASS_VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define CUTLASS_PATCH ([0-9]+)" _CUTLASS_VERSION_PATCH "${VERSION_FILE_CONTENTS}")
set(_CUTLASS_VERSION_PATCH ${CMAKE_MATCH_1})

message(STATUS "CUTLASS ${_CUTLASS_VERSION_MAJOR}.${_CUTLASS_VERSION_MINOR}.${_CUTLASS_VERSION_PATCH}")

## CUTLASS PROJECT #############################################################

project(CUTLASS VERSION ${_CUTLASS_VERSION_MAJOR}.${_CUTLASS_VERSION_MINOR}.${_CUTLASS_VERSION_PATCH} LANGUAGES CXX)

################################################################################

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PPUToolchain.cmake)

################################################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX install CACHE PATH "Default installation location." FORCE)
endif()

message(STATUS "Default Install Location: ${CMAKE_INSTALL_PREFIX}")

set(CUTLASS_ENABLE_EXAMPLES ON CACHE BOOL "Enable CUTLASS Examples")
################################################################################

# CUTLASS_PPU_ARCHS is validated and populated by PPUToolchain.cmake.
set(CUTLASS_PPU_ARCHS_ENABLED ${CUTLASS_PPU_ARCHS} CACHE STRING "The PPU architectures to build code for.")

include(GNUInstallDirs)

link_directories(${PPU_SDK_LIB})

###################################################################################################
#
# Configure CMake variables
#
###################################################################################################

message(STATUS "Device Compilation Architectures: ${CUTLASS_PPU_ARCHS_ENABLED}")

if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
  # By default we want to build in Release mode to ensure that we're getting best performance.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build level" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_VERSIONS_GENERATED")

#
# CUTLASS NAMESPACE
#
set(CUTLASS_NAMESPACE "cutlass" CACHE STRING "Top level namespace of CUTLASS")

set(CUTLASS_ENABLE_F16C OFF CACHE BOOL "Enable F16C x86 extensions in host code.")

################################################################################

# Trace levels for debugging
set(CUTLASS_DEBUG_TRACE_LEVEL "0" CACHE STRING "Level of debug tracing to perform.")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_DEBUG_TRACE_LEVEL=${CUTLASS_DEBUG_TRACE_LEVEL}")
list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS "-DCUTLASS_DEBUG_TRACE_LEVEL=${CUTLASS_DEBUG_TRACE_LEVEL}")

set(CUTLASS_SKIP_REDUCTION_INIT OFF CACHE BOOL "Disable init reduction workspace")

###################################################################################################
#
# Configure device-side build options
#
###################################################################################################

if (CUTLASS_SKIP_REDUCTION_INIT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_SKIP_REDUCTION_INIT=1")
  list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS "-DCUTLASS_SKIP_REDUCTION_INIT=1")
endif()

if (CUTLASS_ENABLE_F16C AND NOT CMAKE_CROSSCOMPILING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_ENABLE_F16C=1 -mf16c")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion -fno-strict-aliasing")

# Warning suppressions for host compilation (g++).
if (CUTLASS_PPU_HOST_COMPILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-conversion -Wno-unused-parameter -Wno-strict-aliasing")
endif()

if (CUTLASS_PPU_DEVICE_COMPILE)
  if (NOT CUTLASS_PPU_HOST_COMPILE)
    message(FATAL_ERROR "PPU device compilation requires the PPU toolchain (PPUToolchain.cmake).")
  endif()

  # Force RPATH (not RUNPATH) so libraries co-located with the PPU
  # runtime can be found at load time.
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")

  link_libraries(ppu::runtime)
  link_libraries(ppu::driver)
endif()

message(STATUS "Host CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Device hgcc flags: ${CUTLASS_PPU_EXTRA_HGCC_FLAGS}")

function(cutlass_apply_standard_compile_options TARGET)
  # Device flags are applied via add_custom_command in cutlass_build_dev_kernels
  # (PPUToolchain.cmake).  This function only links the CUTLASS header target.
  target_link_libraries(${TARGET} PRIVATE CUTLASS)
endfunction()

#
# The following items should eventually be pushed into cutlass/CMakeLists.txt
#

# GLOB for CUTLASS header files. Should we use a static list instead?
file(GLOB_RECURSE CUTLASS_INCLUDE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/cutlass/*.h)
file(GLOB_RECURSE CUTLASS_CUTLASS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include include/cutlass/*.h include/cutlass/*.hpp include/cutlass/*.inl)
file(GLOB_RECURSE CUTLASS_CUTE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include include/cute/*.h*)

###################################################################################################
#
# Define build targets
#
###################################################################################################

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include REGULAR_EXPRESSION ".*\.h")

add_library(CUTLASS INTERFACE)
add_library(cutlass3::cutlass ALIAS CUTLASS)
set_target_properties(CUTLASS PROPERTIES EXPORT_NAME cutlass)

set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "CUTLASS Header Library")

set(CUTLASS_TOOLS_UTIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/util/include CACHE INTERNAL "")

include_directories(${CUTLASS_INCLUDE_DIR})

# Populate device-compile include flags for hgcc add_custom_command.
set(CUTLASS_PPU_DEV_INCLUDE_FLAGS
    "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
    "-I${CMAKE_CURRENT_BINARY_DIR}/include"
    "-I${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/examples/common"
    "-I${PPU_SDK_INCLUDE}"
    "-I${PPU_SDK_TARGETS_INCLUDE}"
    CACHE INTERNAL "Include flags (-I...) for hgcc device compilation")

target_compile_features(CUTLASS INTERFACE cxx_std_11)

if (NOT CUTLASS_NAMESPACE STREQUAL "cutlass")
  target_compile_definitions(CUTLASS INTERFACE CUTLASS_NAMESPACE=${CUTLASS_NAMESPACE})
endif()

if (NOT DEFINED CUTLASS_REVISION)

  find_package(Git QUIET)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    RESULT_VARIABLE CUTLASS_REVISION_RESULT
    OUTPUT_VARIABLE CUTLASS_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (CUTLASS_REVISION_RESULT)
    message(STATUS "CUTLASS Revision: Unable to detect, Git returned code ${CUTLASS_REVISION_RESULT}.")
  else()
    message(STATUS "CUTLASS Revision: ${CUTLASS_REVISION}")
  endif()

endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_extended.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/cutlass/version_extended.h
  @ONLY)

target_include_directories(
  CUTLASS
  INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CUTLASS_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  )

# Mark SDK headers as system to suppress warnings from them
target_include_directories(
  CUTLASS
  SYSTEM INTERFACE
  $<BUILD_INTERFACE:${PPU_SDK_ROOT}/include>
  )

install(
  DIRECTORY
  ${CUTLASS_INCLUDE_DIR}/
  ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(
  TARGETS CUTLASS
  EXPORT Cutlass3
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

################################################################################

# Add common library search paths so executables and libraries can load and run
# without LD_LIBRARY_PATH being set.
link_libraries(
  "-Wl,-rpath,'$ORIGIN'"
  "-Wl,-rpath,'$ORIGIN/../lib64'"
  "-Wl,-rpath,'$ORIGIN/../lib'"
  "-Wl,-rpath,'${PPU_SDK_LIB}'"
  ${CMAKE_DL_LIBS}
  )

################################################################################

add_subdirectory(tools)

if (CUTLASS_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

################################################################################

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/Cutlass3ConfigVersion.cmake
  COMPATIBILITY AnyNewerVersion)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Cutlass3Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/Cutlass3Config.cmake
  @ONLY
  )

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/Cutlass3Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/Cutlass3ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cutlass3/
  )

install(
  EXPORT Cutlass3
  NAMESPACE cutlass3::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cutlass3/
  FILE Cutlass3Targets.cmake
  )
