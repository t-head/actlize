#!/bin/bash

set -ex
set -o pipefail

rm -rf build && mkdir -p build
cd build

cmake .. -DCUTLASS_PPU_ARCHS=ppu0010

rm -f build.log
make -j8 2>&1 | tee -a build.log
