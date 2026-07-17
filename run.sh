#!/bin/bash

set -ex
set -o pipefail

current_dir=$(pwd)
echo "Current path is: ${current_dir}"
if [[ ! $current_dir =~ /build$ ]]; then
  if [ -d "./build" ]; then
    cd ./build
    echo "Change to path: ${current_dir}/build"
  else
    echo "ERROR: build directory not found! Please check."
    exit 1
  fi
fi

./examples/00_basic_gemm/00_basic_gemm
./examples/01_cutlass_utilities/01_cutlass_utilities
./examples/09_gemm_bias_leakyrelu/09_gemm_bias_leakyrelu
./examples/10_gemm_add_biasadd_leakyrelu/10_gemm_add_biasadd_leakyrelu
./examples/11_gemm_add_biasadd_mul_biasadd_leakyrelu/11_gemm_add_biasadd_mul_biasadd_leakyrelu
./examples/12_gemm_add_scale_biasadd_leakyrelu/12_gemm_add_scale_biasadd_leakyrelu
./examples/13_gemm_add_rsqrt_mul_biasadd_leakyrelu/13_gemm_add_rsqrt_mul_biasadd_leakyrelu
./examples/14_splitK_gemm_add_biasadd_mul_biasadd_leakyrelu/14_splitK_gemm_add_biasadd_mul_biasadd_leakyrelu
./examples/15_gemm_biasadd_gelu/15_gemm_biasadd_gelu
./examples/16_batched_gemm_biasadd_relu/16_batched_gemm_biasadd_relu
./examples/18_conv_add_biasadd_scale_biasadd_relu/18_conv_add_biasadd_scale_biasadd_relu
./examples/17_conv_add_biasadd_relu/17_conv_add_biasadd_relu
./examples/19_conv_scale_biasadd_relu/19_conv_scale_biasadd_relu
./examples/20_conv_abs_scale_biasadd_relu/20_conv_abs_scale_biasadd_relu
./examples/21_conv_add_biasadd_scale_biasadd_relu_legacy/21_conv_add_biasadd_scale_biasadd_relu_legacy
./examples/22_conv_scale_biasadd_add_relu/22_conv_scale_biasadd_add_relu
./examples/23_conv_scale_biasadd_relu/23_conv_scale_biasadd_relu
./examples/24_conv_biasadd_relu/24_conv_biasadd_relu
