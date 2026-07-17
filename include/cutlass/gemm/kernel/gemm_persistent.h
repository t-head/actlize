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

/*! \file
    \brief implementations for persistent GEMMs
*/

#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumerated type describing the type of Persistent/Atomic Gemm work mode
enum class PersistentGemmMode {
    kNonAtomic = 0,     // Perform Gemm with any Sync in/out
    kProducer,          // Perform Gemm wiht Sync out
    kConsumer           // Perform Gemm wiht Sync in
};

/// Enumerated type describing the type of Chunk Direction
enum class AtomicDirection {
    kRow = 0,           // divide Chunks by Row
    kColumn             // divide Chunks by Column
};


template<
    typename ThreadblockShape,
    int ThreadCount,
    bool Transposed,
    PersistentGemmMode PersistentGemmMode_,
    AtomicDirection Direction = AtomicDirection::kRow,
    int Bins = 64
    >
struct GemmPersistentTileVisitor
{
    static bool const kTransposed = Transposed;
    static PersistentGemmMode const kPersistentGemmMode = PersistentGemmMode_;
    static bool const kAtomicMode = (kPersistentGemmMode != PersistentGemmMode::kNonAtomic);
    static int const kBins = Bins;

    struct TileVisitorLayout {
        alignas(8) uint32_t block_counter;
        alignas(8) int32_t  mutex;
        alignas(8) uint32_t complated_bin[Bins];
    };

    struct Params {
        cutlass::gemm::GemmCoord problem_size;
        uint64_t ub_timeout;
        int32_t *sync_counter;
        int32_t num_chunks;

        CUTLASS_HOST_DEVICE Params(): ub_timeout(1e8), num_chunks(1), sync_counter(nullptr) { }

        CUTLASS_HOST_DEVICE Params(cutlass::gemm::GemmCoord problem_size_, int32_t num_chunks_, int32_t *sync_counter_, uint64_t ub_timeout_):
                                    problem_size(problem_size_), num_chunks(num_chunks_), sync_counter(sync_counter_), ub_timeout(ub_timeout_)
        { }
    };

    const int32_t threadblock_count;
    const int32_t num_chunks;
    cutlass::gemm::GemmCoord problem_size;

    uint32_t chunk_idx;
    uint32_t tiles_per_chunk;
    uint32_t tile_idx;
    uint32_t ending_tile;

    uint32_t *block_counter;
    int32_t  *mutex;
    uint32_t *complated_bin;
    int32_t *atomic_sync_out;
    int32_t *atomic_sync_in;

    uint64_t ub_timeout;

    struct SharedStorage {
        AlignedBuffer<uint32_t, 1> prefetch_tile_id;
    };
    SharedStorage &shared_storage;

    template <typename T>
    CUTLASS_HOST_DEVICE void swap(T &lhs, T &rhs) {
        T tmp = lhs; lhs = rhs; rhs = tmp;
    }

    CUTLASS_DEVICE GemmPersistentTileVisitor(Params const &params_, SharedStorage &shared_storage_, void *workspace, int32_t block_idx):
                                                chunk_idx(0), tile_idx(block_idx), problem_size(params_.problem_size), shared_storage(shared_storage_),
                                                block_counter(nullptr), mutex(nullptr), ub_timeout(params_.ub_timeout), complated_bin(nullptr),
                                                atomic_sync_out(params_.sync_counter), atomic_sync_in(params_.sync_counter),
                                                threadblock_count(gridDim.x), tiles_per_chunk(0), num_chunks(params_.num_chunks) {
        if constexpr (kTransposed) {
            swap(problem_size.m(), problem_size.n());
        }

        auto internal_space = reinterpret_cast<TileVisitorLayout *>(workspace);
        mutex           = &(internal_space->mutex);
        block_counter   = &(internal_space->block_counter);

        if constexpr (PersistentGemmMode::kProducer == kPersistentGemmMode) {
            complated_bin = internal_space->complated_bin;
        }

        ending_tile = tile_count(params_.problem_size);
        tiles_per_chunk = ending_tile / num_chunks;
    }

    CUTLASS_DEVICE void _lock_mutex() {
        while(0 != atomicCAS(mutex, 0, 1)) { }
    }

    CUTLASS_DEVICE void _unlock_mutex() {
        atomicExch(mutex, 0);
        // __threadfence();
    }

    CUTLASS_HOST_DEVICE static uint32_t tile_count(const cutlass::gemm::GemmCoord & problem_size_) {
         cutlass::gemm::GemmCoord gridshape = grid_shape(problem_size_);
         return gridshape.m() * gridshape.n();
    }

    CUTLASS_HOST_DEVICE static cutlass::gemm::GemmCoord grid_shape(const cutlass::gemm::GemmCoord & problem_size) {
        return cutlass::gemm::GemmCoord(
        ((problem_size.m() - 1 + ThreadblockShape::kM) / ThreadblockShape::kM),
        ((problem_size.n() - 1 + ThreadblockShape::kN) / ThreadblockShape::kN),
        1);
    }

    CUTLASS_HOST_DEVICE uint32_t threadblock_idx() const { return tile_idx; }

    CUTLASS_DEVICE void reset_counter() { /* No Op*/ }

    CUTLASS_DEVICE bool next_tile() {
        if(threadIdx.x == 0) {
            _lock_mutex();
            // Critical code
            auto prefetch_tile_id = atomicAdd(block_counter, 1);
            _unlock_mutex();

            chunk_idx = prefetch_tile_id / tiles_per_chunk;
            auto share_ptr = shared_storage.prefetch_tile_id.data();
            share_ptr[0] = prefetch_tile_id;

            if constexpr (PersistentGemmMode::kConsumer == kPersistentGemmMode) {
                while(0 != *reinterpret_cast<volatile int32_t *>(&(atomic_sync_in[chunk_idx]))) { }
            }
        }
        __syncthreads();

        auto share_ptr = shared_storage.prefetch_tile_id.data();
        tile_idx = share_ptr[0];
        return tile_idx < ending_tile;
    }

    CUTLASS_DEVICE void advance() {
        if constexpr (PersistentGemmMode::kProducer == kPersistentGemmMode) {
            __threadfence();
            if(threadIdx.x == 0) {
                auto completed_tiles = atomicAdd(&(complated_bin[chunk_idx]), 1);
                if (tiles_per_chunk-1 == completed_tiles) {
                    atomic_sync_out[chunk_idx] = 0;
                    complated_bin[chunk_idx] = 0;
                }
            }
            __syncthreads();
        }
    }
};

template <
    typename Mma_,
    typename Epilogue_,
    typename ThreadblockSwizzle_,
    bool Transposed = false,
    PersistentGemmMode PersistentGemmMode_ = PersistentGemmMode::kNonAtomic,
    AtomicDirection Direction = AtomicDirection::kRow>
struct GemmPersistent
{
public:
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using EpilogueOutputOp = typename Epilogue::OutputOp;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    static const AtomicDirection kDirection = Direction;
    using ThreadblockShape = typename Mma::Shape;
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;
    using PersistentTileVisitor = GemmPersistentTileVisitor<ThreadblockShape, kThreadCount, Transposed, PersistentGemmMode_, Direction>;
    using TileVisitorLayout = typename PersistentTileVisitor::TileVisitorLayout;
    static bool const kAtomicMode = PersistentTileVisitor::kAtomicMode;
    static bool const kTransposed = Transposed;
    using MapArguments = kernel::detail::MapArguments<
        typename Mma::IteratorA::Element,
        typename Mma::IteratorA::Layout,
        Mma::IteratorA::AccessType::kElements,
        typename Mma::IteratorB::Element,
        typename Mma::IteratorB::Layout,
        Mma::IteratorB::AccessType::kElements,
        typename Mma::LayoutC,
        kTransposed
    >;
    using ElementA = typename MapArguments::ElementA;
    using LayoutA = typename MapArguments::LayoutA;
    using ElementB = typename MapArguments::ElementB;
    using LayoutB = typename MapArguments::LayoutB;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename MapArguments::LayoutC;


    struct Arguments {
        GemmCoord problem_size;
        int threadblock_count;
        typename EpilogueOutputOp::Params output_op;
        typename PersistentTileVisitor::Params & visitor_param;
        ElementA *ptr_A;
        ElementB *ptr_B;
        ElementC *ptr_C;
        ElementC *ptr_D;
        typename LayoutA::Stride::LongIndex lda;
        typename LayoutB::Stride::LongIndex ldb;
        typename LayoutB::Stride::LongIndex ldc;
        typename LayoutC::Stride::LongIndex ldd;

        CUTLASS_HOST_DEVICE Arguments(GemmCoord problem_size, int threadblock_count,
                                    typename EpilogueOutputOp::Params output_op,
                                    typename PersistentTileVisitor::Params & visitor_param,
                                    ElementA *ptr_A, ElementB *ptr_B, ElementC *ptr_C, ElementC *ptr_D,
                                    typename LayoutA::Stride::LongIndex lda, typename LayoutB::Stride::LongIndex ldb,
                                    typename LayoutB::Stride::LongIndex ldc, typename LayoutB::Stride::LongIndex ldd):
                                    problem_size(problem_size), threadblock_count(threadblock_count),
                                    output_op(output_op), visitor_param(visitor_param),
                                    ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D),
                                    lda(lda), ldb(ldb), ldc(ldc), ldd(ldd)
        { }
    };

    struct Params {
        typename PersistentTileVisitor::Params visitor_param;
        int threadblock_count;
        typename EpilogueOutputOp::Params output_op;
        ElementA *ptr_A;
        ElementB *ptr_B;
        ElementC *ptr_C, *ptr_D;
        typename LayoutA::Stride::LongIndex lda, ldb, ldc, ldd;
        void *workspace;

        CUTLASS_HOST_DEVICE Params(): ptr_A(nullptr), ptr_B(nullptr), ptr_C(nullptr), ptr_D(nullptr),
                                    lda(0), ldb(0), ldc(0), ldd(0), workspace(nullptr)
        { }

        CUTLASS_HOST_DEVICE Params(Arguments const &args, void *workspace = nullptr):
                                visitor_param(args.visitor_param),
                                threadblock_count(args.threadblock_count),
                                output_op(args.output_op),
                                ptr_A(args.ptr_A), ptr_B(args.ptr_B), ptr_C(args.ptr_C), ptr_D(args.ptr_D),
                                lda(args.lda), ldb(args.ldb), ldc(args.ldc), ldd(args.ldd),
                                workspace(workspace)
        { }
    };

    struct SharedStorage {
        union {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
        } kernel;
        typename PersistentTileVisitor::SharedStorage visitor;
    };


public:
    CUTLASS_DEVICE GemmPersistent() { }

    static Status can_implement(Arguments const &args) {
        // TODO: Not all problem sizes can execute Atomic GEMM.
        return Status::kSuccess;
    }

    CUTLASS_DEVICE void operator()(Params const &params, SharedStorage &shared_storage) {
        using ElementA = typename Mma::IteratorA::Element;
        using LayoutA =  typename Mma::IteratorA::Layout;
        using ElementB = typename Mma::IteratorB::Element;
        using LayoutB =  typename Mma::IteratorB::Layout;
        using ElementC = typename Epilogue::OutputTileIterator::Element;
        using LayoutC =  typename Epilogue::OutputTileIterator::Layout;

        typename LayoutA::LongIndex ldm_A = kTransposed? params.ldb: params.lda;
        typename LayoutA::LongIndex ldm_B = kTransposed? params.lda: params.ldb;
        typename LayoutA::LongIndex ldm_C = params.ldc;
        typename LayoutA::LongIndex ldm_D = params.ldd;

        int thread_idx = threadIdx.x;
        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
        int lane_idx = threadIdx.x % 32;

        PersistentTileVisitor tile_visitor(params.visitor_param, shared_storage.visitor, params.workspace, blockIdx.x);
        const cutlass::gemm::GemmCoord & problem_size = tile_visitor.problem_size;

        ElementA *ptr_A = reinterpret_cast<ElementA *>(kTransposed? params.ptr_B: params.ptr_A);
        ElementB *ptr_B = reinterpret_cast<ElementB *>(kTransposed? params.ptr_A: params.ptr_B);
        ElementC *ptr_C = reinterpret_cast<ElementC *>(params.ptr_C);
        ElementC *ptr_D = reinterpret_cast<ElementC *>(params.ptr_D);

        tile_visitor.reset_counter();

        while (tile_visitor.next_tile()) {
            GemmCoord grid_shape = PersistentTileVisitor::grid_shape(problem_size);
            int32_t threadblock_idx = int32_t(tile_visitor.threadblock_idx());

            GemmCoord threadblock_offset(0, 0, 0);
            if constexpr ((kDirection == AtomicDirection::kRow) != kTransposed) {
                // schedule tiles along N dimension
                threadblock_offset.m() = int(threadblock_idx / grid_shape.n()) * Mma::Shape::kM;
                threadblock_offset.n() = int(threadblock_idx % grid_shape.n()) * Mma::Shape::kN;
            } else {
                // schedule tiles along M dimension
                threadblock_offset.m() = int(threadblock_idx % grid_shape.m()) * Mma::Shape::kM;
                threadblock_offset.n() = int(threadblock_idx / grid_shape.m()) * Mma::Shape::kN;
            }

            cutlass::MatrixCoord tb_offset_A{threadblock_offset.m(), 0};
            cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.n()};

            typename Mma::IteratorA iterator_A(LayoutA(ldm_A), ptr_A, {problem_size.m(), problem_size.k()}, thread_idx, tb_offset_A);
            typename Mma::IteratorB iterator_B(LayoutB(ldm_B), ptr_B, {problem_size.k(), problem_size.n()}, thread_idx, tb_offset_B);
            typename Mma::FragmentC accumulators;
            accumulators.clear();

            Mma mma(shared_storage.kernel.main_loop, thread_idx, warp_idx, lane_idx);
            int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
            __syncthreads();
            mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

            EpilogueOutputOp output_op(params.output_op);

            LayoutC layout_C(ldm_C);
            LayoutC layout_D(ldm_D);
            typename Epilogue::OutputTileIterator::Params params_C(layout_C);
            typename Epilogue::OutputTileIterator::Params params_D(layout_D);
            typename Epilogue::OutputTileIterator iterator_C(params_C, ptr_C, problem_size.mn(), thread_idx, threadblock_offset.mn());
            typename Epilogue::OutputTileIterator iterator_D(params_D, ptr_D, problem_size.mn(), thread_idx, threadblock_offset.mn());

            Epilogue epilogue(shared_storage.kernel.epilogue, thread_idx, warp_idx, lane_idx);
            epilogue(output_op, iterator_D, accumulators, iterator_C);

            tile_visitor.advance();
        }
    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
