# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

if(NOT DEFINED HGCC OR NOT DEFINED SOURCE_ROOT OR NOT DEFINED OUTPUT_ROOT OR NOT DEFINED TEST_MACRO)
  message(FATAL_ERROR "HGCC, SOURCE_ROOT, OUTPUT_ROOT and TEST_MACRO are required")
endif()

if(TEST_MACRO STREQUAL "ACTLIZE_TEST_STATIC_TAIL")
  set(expected "Unchecked store fragment must divide static problem N.")
elseif(TEST_MACRO STREQUAL "ACTLIZE_TEST_FRAGMENT_MISMATCH")
  set(expected "Unchecked store fragment must divide output alignment.")
elseif(TEST_MACRO STREQUAL "ACTLIZE_TEST_NO_EVT")
  set(expected "Safe store requires the PPU EVT epilogue.")
elseif(TEST_MACRO STREQUAL "ACTLIZE_TEST_ZERO_ALIGNMENT")
  set(expected "Epilogue alignments must be positive.")
else()
  message(FATAL_ERROR "Unknown compile-fail case: ${TEST_MACRO}")
endif()

file(MAKE_DIRECTORY "${OUTPUT_ROOT}")
execute_process(
  COMMAND "${HGCC}" -std=c++17 -arch=ppu_10 -O3 -DUSE_HGGC -DUSE_CLANG -DUSE_ACWRAPPER
    -DNDEBUG -DACTLIZE_EPILOGUE_STANDALONE "-D${TEST_MACRO}"
    --expt-relaxed-constexpr --expt-extended-lambda -ftemplate-depth=8192
    "-I${SOURCE_ROOT}/include" "-I${SOURCE_ROOT}/tools/util/include"
    -x hg -c "${SOURCE_ROOT}/test/unit/gemm_ppu/device/ppu0010_epilogue_vectorized_evt.cu"
    -o "${OUTPUT_ROOT}/${TEST_MACRO}.o"
  RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr
)
set(diagnostic "${stdout}\n${stderr}")
file(WRITE "${OUTPUT_ROOT}/${TEST_MACRO}.log" "${diagnostic}")
string(FIND "${diagnostic}" "${expected}" matched)
if(result STREQUAL "0" OR matched EQUAL -1)
  message(FATAL_ERROR "Expected compilation failure with '${expected}', got ${result}:\n${diagnostic}")
endif()
message(STATUS "${TEST_MACRO}: expected static_assert matched")
