/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Basic include for CUTLASS.
*/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

#include <hggc.h>
#include <sstream>
#include <fstream>

#ifndef checkHggcDrvErrors
#define checkHggcDrvErrors(expr)                                                          \
  do {                                                                                    \
    HGresult __ret = expr;                                                                \
    if (__ret != HGGC_SUCCESS) {                                                          \
      const char *perror = nullptr;                                                       \
      hgGetErrorString(__ret, &perror);                                                   \
      fprintf(stderr, "device Driver error at %s:%d code=%d(%s) \"%s\" \n",                 \
              __FILE__, __LINE__, static_cast<int>(__ret), perror, #expr);                \
      throw std::runtime_error("device driver failed.");                                    \
    }                                                                                     \
  } while (0)
#endif

#ifndef checkHggcErrors
#define checkHggcErrors(expr)                                                             \
  do {                                                                                    \
    expr;                                                                                 \
    hggcError_t __err = hggcGetLastError();                                               \
    if (__err != hggcSuccess) {                                                           \
      fprintf(stderr, "device error at %s:%d code=%d(%s) \"%s\" \n", __FILE__, __LINE__,    \
              static_cast<int>(__err), hggcGetErrorString(__err), #expr);                 \
      throw std::runtime_error("device runtime failed.");                                   \
    }                                                                                     \
  } while (0)
#endif

static HGmodule upload(const char *filename) {
  HGmodule module;
  std::ostringstream fatbin;
  std::ifstream fileModule(filename, std::ios::binary);
  fatbin << fileModule.rdbuf();
  checkHggcDrvErrors(hgModuleLoadDataEx(&module, fatbin.str().c_str(), 0, nullptr, nullptr));
  return module;
}

static void showFuncAttrib(HGfunction kernel) {
  int attr = 0;
  checkHggcDrvErrors(hgFuncGetAttribute(&attr, HG_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel));
  std::cout << "[rtc] HG_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = " << attr << std::endl;
  checkHggcDrvErrors(hgFuncGetAttribute(&attr, HG_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES    , kernel));
  std::cout << "[rtc] HG_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES     = " << attr << std::endl;
  checkHggcDrvErrors(hgFuncGetAttribute(&attr, HG_FUNC_ATTRIBUTE_CONST_SIZE_BYTES     , kernel));
  std::cout << "[rtc] HG_FUNC_ATTRIBUTE_CONST_SIZE_BYTES      = " << attr << std::endl;
  checkHggcDrvErrors(hgFuncGetAttribute(&attr, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES     , kernel));
  std::cout << "[rtc] HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES      = " << attr << std::endl;
  checkHggcDrvErrors(hgFuncGetAttribute(&attr, HG_FUNC_ATTRIBUTE_NUM_REGS             , kernel));
  std::cout << "[rtc] HG_FUNC_ATTRIBUTE_NUM_REGS              = " << attr << std::endl;
  checkHggcDrvErrors(hgFuncGetAttribute(&attr, HG_FUNC_ATTRIBUTE_ASM_VERSION          , kernel));
  std::cout << "[rtc] HG_FUNC_ATTRIBUTE_ASM_VERSION           = " << attr << std::endl;
  checkHggcDrvErrors(hgFuncGetAttribute(&attr, HG_FUNC_ATTRIBUTE_BINARY_VERSION       , kernel));
  std::cout << "[rtc] HG_FUNC_ATTRIBUTE_BINARY_VERSION        = " << attr << std::endl;
}

#if CUTLASS_ENABLE_RTC

#include <vector>
#include <hgrtc.h>
#include "hgrtc/assert.h"
#include "hgrtc/stdint.h"
#include "hgrtc/device_kernel.h"

#ifndef checkHgrtcErrors
#define checkHgrtcErrors(expr)                                                            \
  do {                                                                                    \
    hgrtcResult __ret = expr;                                                             \
    if (__ret != HGRTC_SUCCESS) {                                                         \
      fprintf(stderr, "HGRTC error at %s:%d code=%d(%s) \"%s\" \n", __FILE__, __LINE__,   \
              static_cast<int>(__ret), hgrtcGetErrorString(__ret), #expr);                \
      throw std::runtime_error("HGRTC failed.");                                          \
    }                                                                                     \
  } while (0)
#endif

static HGfunction rtCompile(const std::string &prog_src, const std::string &kernel_instantiation) {
    char const *stdHeaders[] = {
      cutlass::hgrtc::assert_h,
      cutlass::hgrtc::stdint_h,
    };

    char const *stdHeaderNames[] = {
      "assert.h",
      "stdint.h",
    };

    const char *ppu_home = std::getenv("PPU_HOME");
    if (!ppu_home) {
      std::cerr << "[ERROR] Cannot find enviorment: PPU_HOME" << std::endl;
      exit(1);
    }

    const char *device_path = "/usr/local/hggc";

    const std::string opt_device_inc    = "--include-path=" + std::string(device_path) + "/include";
    const std::string opt_device_std    = "--include-path=" + std::string(device_path) + "/include/hggc/std";
    const std::string opt_cutlass     = "--include-path=" + std::string(ppu_home) + "/cutlass";
    const std::string opt_cutlass_inc = "--include-path=" + std::string(ppu_home) + "/cutlass/include";

    std::vector<const char*> options = {
      "--gpu-architecture=ppu_10",
      "--std=c++11",
      "--generate-line-info",
      opt_device_inc.c_str(),
      opt_device_std.c_str(),
      opt_cutlass.c_str(),
      opt_cutlass_inc.c_str(),
    };

    hgrtcProgram program;
    checkHgrtcErrors(hgrtcCreateProgram(&program, prog_src.c_str(), "kernel.hg", 2, stdHeaders, stdHeaderNames));

    checkHgrtcErrors(hgrtcAddNameExpression(program, kernel_instantiation.c_str()));
    if (hgrtcCompileProgram(program, static_cast<int>(options.size()), options.data()) != HGRTC_SUCCESS) {
      size_t log_size;
      checkHgrtcErrors(hgrtcGetProgramLogSize(program, &log_size));
      std::vector<char> log(log_size);
      checkHgrtcErrors(hgrtcGetProgramLog(program, log.data()));
      std::cerr << "Compile Failed:" << std::endl << log.data() << std::endl;
    }
    std::cout << "[rtc] compile success." << std::endl;

    size_t hg_size;
    checkHgrtcErrors(hgrtcGetHGBINSize(program, &hg_size));
    std::vector<char> hg(hg_size);
    checkHgrtcErrors(hgrtcGetHGBIN(program, hg.data()));

    char const *kernel_lowered;
    checkHgrtcErrors(hgrtcGetLoweredName(program, kernel_instantiation.c_str(), &kernel_lowered));

    HGmodule module;
    HGfunction kernel;
    checkHggcDrvErrors(hgModuleLoadDataEx(&module, hg.data(), 0, nullptr, nullptr));
    checkHggcDrvErrors(hgModuleGetFunction(&kernel, module, kernel_lowered));

    checkHgrtcErrors(hgrtcDestroyProgram(&program));

    return kernel;
}

#endif // #if CUTLASS_ENABLE_RTC
////////////////////////////////////////////////////////////////////////////////////////////////////
