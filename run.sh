#!/bin/bash

set -e

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

export TraceEventPath=$(pwd)
export HGGC_PROFILE_MODE=4

echo "run start"
./examples/05_ppu_cute_gemm/05_ppu_cute_gemm --m=256 --n=256 --k=64 --iterations=1
./examples/00_ppu_basic_tensor_op_gemm/00_ppu_basic_tensor_op_gemm --m=256 --n=256 --k=64 --iterations=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=128 --n=128 --k=2048 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=128 --n=128 --k=4608 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=128 --n=128 --k=256 --l=1

./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=1024 --n=1024 --k=1024 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=2048 --n=2048 --k=1024 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=4096 --n=4096 --k=512 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=4096 --n=4096 --k=1024 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=4096 --n=4096 --k=2048 --l=1

./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=1024 --n=1024 --k=4096 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=2048 --n=2048 --k=4096 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=2048 --n=2176 --k=4096 --l=1

./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=1024 --n=1024 --k=30528 --l=1

./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=2048 --n=2176 --k=4096 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=2048 --n=2560 --k=4096 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=2048 --n=2688 --k=4096 --l=1


./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=1024 --n=1152 --k=256 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=1024 --n=1280 --k=256 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=1024 --n=2048 --k=256 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=2048 --n=2048 --k=256 --l=1
./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=2048 --n=2048 --k=256 --l=1 --beta=0.1

./examples/01_ppu_gemm_with_collective_builder/01_collective_builder --m=2048 --n=2048 --k=256 --l=2

./examples/02_ppu_mixed_dtype_gemm/02_ppu_mixed_dtype_gemm --m=256 --n=256 --k=128 --l=1 --alpha=2 --beta=0 --mode=0 --iterations=0
./examples/02_ppu_mixed_dtype_gemm/02_ppu_mixed_dtype_gemm --m=256 --n=256 --k=2048 --l=1 --alpha=2 --beta=0 --mode=0 --iterations=0
./examples/02_ppu_mixed_dtype_gemm/02_ppu_mixed_dtype_gemm --m=256 --n=4096 --k=2048 --l=1 --alpha=2 --beta=0 --mode=0 --iterations=0
./examples/02_ppu_mixed_dtype_gemm/02_ppu_mixed_dtype_gemm --m=1024 --n=4096 --k=2048 --l=1 --alpha=2 --beta=0 --mode=0 --iterations=0

./examples/03_ppu_ptr_array_batched_gemm/03_ppu_ptr_array_batched_gemm --m=256 --n=256 --k=64 --l=1
./examples/03_ppu_ptr_array_batched_gemm/03_ppu_ptr_array_batched_gemm --m=256 --n=256 --k=64 --l=2
./examples/03_ppu_ptr_array_batched_gemm/03_ppu_ptr_array_batched_gemm --m=128 --n=128 --k=64 --l=4

echo "run done"

