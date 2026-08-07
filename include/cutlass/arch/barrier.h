/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 **************************************************************************************************/

/*! \file
    \brief Barrier Operations on PPU0015+
*/

#pragma once

#include <cutlass/arch/memory_ppu.h>
#if (defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 100 && (__HGGCCC_VER_MAJOR__ >= 12)) || defined(__HGGCCC__)
#define HGGC_BARRIER_ENABLED 1
#else
#define HGGC_BARRIER_ENABLED 0
#endif

namespace cutlass {
/// @brief
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts
// This enum class specifies the NamedBarriers reserved by CUTLASS.
enum class ReservedNamedBarriers { 
  EpilogueBarrier = 0,
  TransposeBarrier = 1,
  TransformBarrier = 2,
  StreamkBarrier0 = 3,
  StreamkBarrier1 = 4
  , FirstUserBarrier = StreamkBarrier1 + 1
};


class NamedBarrier {

  // Data Members:

  // Range = [1 , NUM_THREADS_PER_CTA]
  // Range % warp-size (i.e 32) == 0
  uint32_t const num_threads_;

  // Range : [0, 15]
  uint32_t const id_;

 public:

  // Constructor for CUTLASS developers:
  // effective barrier ID starts from 0
  CUTLASS_DEVICE
  NamedBarrier(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers)
      : num_threads_(num_threads), id_(static_cast<uint32_t>(reserved_named_barriers)) {}

  // Constructor for CUTLASS users:
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  NamedBarrier(uint32_t num_threads, uint32_t id = 0)
      : num_threads_(num_threads), id_(id + ReservedNamedBarrierCount) {
    CUTLASS_ASSERT(id + ReservedNamedBarrierCount <= HardwareMaxNumNamedBarriers && "Effective barrier_id should not exceed 16.");
  }

  CUTLASS_DEVICE
  void arrive_and_wait() const {
    NamedBarrier::arrive_and_wait(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void arrive() const {
    NamedBarrier::arrive(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void sync() const {
    NamedBarrier::arrive_and_wait();
  }

  //  Static variants

  // Calling interface for CUTLASS users: 
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  static void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id) {
    arrive_and_wait_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for CUTLASS developers: 
  // effective barrier ID starts from 0
  CUTLASS_DEVICE
  static void arrive_and_wait(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers) {
    arrive_and_wait_internal(num_threads, static_cast<int>(reserved_named_barriers));
  }

  // Calling interface for CUTLASS users: 
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  static void arrive(uint32_t num_threads, uint32_t barrier_id) {
    arrive_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for CUTLASS developers: 
  // effective barrier ID starts from 0
  CUTLASS_DEVICE
  static void arrive(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers) {
    arrive_internal(num_threads, static_cast<int>(reserved_named_barriers));
  }

  // Calling interface for CUTLASS users: 
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  static void sync(uint32_t num_threads, uint32_t barrier_id) {
    sync_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for CUTLASS developers: 
  // effective barrier ID starts from 0
  CUTLASS_DEVICE
  static void sync(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers) {
    sync_internal(num_threads, static_cast<int>(reserved_named_barriers));
  }

 private:
  CUTLASS_DEVICE
  static void arrive_and_wait_internal(uint32_t num_threads, uint32_t barrier_id) {
#if HGGC_BARRIER_ENABLED
    asm volatile("ppu.bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#endif
  }

  CUTLASS_DEVICE
  static void arrive_internal(uint32_t num_threads, uint32_t barrier_id) {
#if HGGC_BARRIER_ENABLED
    asm volatile("ppu.bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#endif
  }

  CUTLASS_DEVICE
  static void sync_internal(uint32_t num_threads, uint32_t barrier_id) {
    NamedBarrier::arrive_and_wait_internal(num_threads, barrier_id);
  }

 public:
  // Currently we reserve 8 NamedBarriers for CUTLASS' own use cases, 
  // while leaving the renaming for general users.
  static const uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(ReservedNamedBarriers::FirstUserBarrier);
  static const uint32_t HardwareMaxNumNamedBarriers = 16;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ClusterBarrier {

  using ValueType = uint64_t;

protected:
  // Can never be initialized - can only be aliased to smem
  ValueType barrier_;

public:

  CUTLASS_DEVICE
  ClusterBarrier() = delete;

  CUTLASS_DEVICE
  void init(uint32_t arrive_count) const {
    ClusterBarrier::init(&this->barrier_, arrive_count);
  }

  CUTLASS_DEVICE
  uint32_t test_wait(uint32_t phase, uint32_t pred=true) const {
    return ClusterBarrier::test_wait(&this->barrier_, phase, pred);
  }

  CUTLASS_DEVICE
  uint32_t try_wait(uint32_t phase) const {
    return ClusterBarrier::try_wait(&this->barrier_, phase);
  }

  CUTLASS_DEVICE
  void wait(uint32_t phase) const {
    ClusterBarrier::wait(&this->barrier_, phase);
  }

  // Barrier arrive on local smem
  CUTLASS_DEVICE
  void arrive() const {
    ClusterBarrier::arrive(&this->barrier_);
  }

  // Remote SMEM arrive with a perdicate (usually done to pick the thread doing the arrive)
  CUTLASS_DEVICE
  void arrive(uint32_t cta_id, uint32_t pred = true ) const {
    ClusterBarrier::arrive(&this->barrier_, cta_id, pred);
  }
  //
  //  Static Versions
  //
  CUTLASS_DEVICE
  static void init(ValueType const* smem_ptr, uint32_t arrive_count) {
#if HGGC_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "ppu.awbar.init.shared::cta.b64 [%1], %0; \n"
        "}"
        :
        : "r"(arrive_count), "r"(smem_addr));
#endif
  }

  // Static version of wait - in case we don't want to burn a register
  CUTLASS_DEVICE
  static void wait(ValueType const* smem_ptr, uint32_t phase) {
#if HGGC_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    // Arbitrarily large timer value after which try-wait expires and re-tries.
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "ppu.awbar.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 ppu.bra DONE; \n\t"
        "ppu.bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(smem_addr), "r"(phase), "r"(ticks));

#endif
  }

  CUTLASS_DEVICE
  static uint32_t test_wait(ValueType const* smem_ptr, uint32_t phase, uint32_t pred) {
#if HGGC_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t waitComplete;

    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        ".reg .pred P2; \n\t"
        "ppu.cmpp.eq.u32 P2, %3, 1;\n\t"
        "@P2 ppu.awbar.test_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
        "ppu.selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase), "r"(pred));

    return waitComplete;
#endif
    return 0;
  }

  CUTLASS_DEVICE
  static uint32_t try_wait(ValueType const* smem_ptr, uint32_t phase) {
#if HGGC_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t waitComplete;

    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        "ppu.awbar.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
        "ppu.selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase));

    return waitComplete;
#endif
    return 0;
  }

  // Static Predicated version of the above - in case we know the address.
  CUTLASS_DEVICE
  static void arrive(ValueType const* smem_ptr, uint32_t cta_id, uint32_t pred) {
#if HGGC_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 remAddr32;\n\t"
        "ppu.cmpp.eq.u32 p, %2, 1;\n\t"
        "@p ppu.awbar.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(pred));
#elif defined(__HGGC_ARCH__)
    __brkpt();
#endif
  }

  // Barrier arrive on local smem
  CUTLASS_DEVICE
  static void arrive(ValueType const* smem_ptr) {
#if HGGC_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "ppu.awbar.arrive.shared::cta.b64 _, [%0];\n\t"
        "}"
        :
        : "r"(smem_addr));
#endif
  }

  CUTLASS_DEVICE
  static void invalidate(ValueType const* smem_ptr) {
#if HGGC_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "ppu.awbar.ival.shared::cta.b64 [%0]; \n\t"
        "}"
        :
        : "r"(smem_addr));
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// PPU0015 also introduces a new type of cluster-barrier which supports sync.
// not just based on Arrive Count, but also transaction count (in bytes)
struct ClusterTransactionBarrier : public ClusterBarrier {

  CUTLASS_DEVICE
  ClusterTransactionBarrier() = delete;

  // Performs an arrive operation + expected transaction bytes increment
  CUTLASS_DEVICE
  void arrive_and_expect_tx(uint32_t transaction_bytes) const {
    ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes);
  }

  // Performs an arrive operation + expected transaction bytes increment
  CUTLASS_DEVICE
  void arrive_and_expect_tx(uint32_t transaction_bytes, uint32_t cta_id) const {
    ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes , cta_id, true);
  }

  // Performs an expected transaction bytes increment without doing an arrive operation
  CUTLASS_DEVICE
  void expect_transaction(uint32_t transaction_bytes) const {
    ClusterTransactionBarrier::expect_transaction(&this->barrier_, transaction_bytes);
  }

  // Performs an expected transaction bytes decrement without doing an arrive operation
  CUTLASS_DEVICE
  void complete_transaction(uint32_t transaction_bytes, uint32_t pred = 1) const {
    uint32_t cta_rank = 0;  // cute::block_rank_in_cluster();
    ClusterTransactionBarrier::complete_transaction(&this->barrier_, cta_rank, transaction_bytes, pred);
  }

  // Performs an expected transaction bytes decrement without doing an arrive operation
  CUTLASS_DEVICE
  void complete_transaction(uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
    ClusterTransactionBarrier::complete_transaction(&this->barrier_, dst_cta_id, transaction_bytes, pred);
  }

  //
  //  Static Versions
  //

  // Performs an arrive operation + expected transaction bytes increment
  CUTLASS_DEVICE
  static void arrive_and_expect_tx(ValueType const* smem_ptr, uint32_t transaction_bytes) {
#if HGGC_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "ppu.awbar.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
#endif
  }

  // Performs an arrive operation + expected transaction bytes increment for a remote cta_id in a Cluster
  CUTLASS_DEVICE
  static void arrive_and_expect_tx(
      ValueType const* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
#if HGGC_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 remAddr32;\n\t"
        "ppu.cmpp.eq.u32 p, %2, 1;\n\t"
        "@p ppu.awbar.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %3;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(pred), "r"(transaction_bytes));
#elif defined(__HGGC_ARCH__)
    __brkpt();
#endif
  }

  // Performs an expected transaction bytes increment without doing an arrive operation
  CUTLASS_DEVICE
  static void expect_transaction(ValueType const* smem_ptr, uint32_t transaction_bytes) {
#if HGGC_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "ppu.awbar.expect_tx.shared::cta.b64 [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
#endif
  }

  // Performs an expected transaction bytes decrement without doing an arrive operation
  CUTLASS_DEVICE
  static void complete_transaction(
      ValueType const* smem_ptr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred = 1) {
#if HGGC_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    // smem_addr = cute::set_block_rank(smem_addr, dst_cta_id);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "ppu.cmpp.eq.u32 p, %2, 1;\n\t"
        "@p ppu.awbar.complete_tx.shared::cluster.relaxed.cluster.b64   [%1], %0;"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr), "r"(pred));
#endif
  }

  //
  // DEPRECATED APIs
  //
  [[deprecated("Use arrive_and_expect_tx instead")]] CUTLASS_DEVICE
  void arrive_and_reset_bytes(uint32_t transaction_bytes) const {
    arrive_and_expect_tx(transaction_bytes);
  }
  [[deprecated("Use arrive_and_expect_tx instead")]] CUTLASS_DEVICE
  void arrive_and_reset_bytes(uint32_t transaction_bytes, uint32_t cta_id) const {
    arrive_and_expect_tx(transaction_bytes, cta_id);
  }
  [[deprecated("Use expect_transaction instead")]] CUTLASS_DEVICE
  void reset_bytes(uint32_t transaction_bytes) const {
    expect_transaction(transaction_bytes);
  }
  [[deprecated("Use complete_transaction instead")]] CUTLASS_DEVICE
  void commit(uint32_t transaction_bytes, uint32_t pred = 1) const {
    complete_transaction(transaction_bytes, pred);
  }
  [[deprecated("Use complete_transaction instead")]] CUTLASS_DEVICE
  void commit(uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
    complete_transaction(dst_cta_id, transaction_bytes, pred);
  }
  [[deprecated("Use arrive_and_expect_tx instead")]] CUTLASS_DEVICE
  static void arrive_and_reset_bytes(ValueType const* smem_ptr, uint32_t transaction_bytes) {
    arrive_and_expect_tx(smem_ptr, transaction_bytes);
  }
  [[deprecated("Use arrive_and_expect_tx instead")]] CUTLASS_DEVICE
  static void arrive_and_reset_bytes(ValueType const* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
    arrive_and_expect_tx(smem_ptr, transaction_bytes, cta_id, pred);
  }
  [[deprecated("Use expect_transaction instead")]] CUTLASS_DEVICE
  static void reset_bytes(ValueType const* smem_ptr, uint32_t transaction_bytes) {
    expect_transaction(smem_ptr, transaction_bytes);
  }
  [[deprecated("Use complete_transaction instead")]] CUTLASS_DEVICE
  static void commit(ValueType const* smem_ptr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred = 1) {
    complete_transaction(smem_ptr, dst_cta_id, transaction_bytes, pred);
  }
};

// Helps with visibility of barrier init operations across warps / cta / cluster
// Available as a separate function so as to batch inits across barriers and fence once
// Note : It must be composed with an appropriate sync instruction with the right scope
// to ensure visibility eg. __syncthreads() or a cluster_arrive() + cluster_wait()
CUTLASS_DEVICE
void fence_barrier_init() {
#if HGGC_BARRIER_ENABLED
  asm volatile(
      "{\n\t"
      "ppu.fence.mbarrier_init.release.cluster; \n"
      "}"
      ::);
#endif
}

// Issue a shared memory fence for async operations
CUTLASS_DEVICE
void fence_view_async_shared() {
#if HGGC_BARRIER_ENABLED
    asm volatile (
        "{\n\t"
        "ppu.fence.proxy.async.shared::cta; \n"
        "}"
        ::);
#endif
}

// Arrive on completion of in-flight cp.async operations issued by the calling thread
CUTLASS_DEVICE
void cpasync_barrier_arrive(uint64_t const* smem_ptr) {
#if HGGC_BARRIER_ENABLED
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
    "{\n\t"
    "ppu.cp.async.awbar.arrive.shared::cta.b64 [%0];\n\t"
    "}"
    :
    : "r"(smem_addr));
#endif
}

// Arrive on completion of in-flight cp.async operations issued by the calling thread
CUTLASS_DEVICE
void cpasync_barrier_arrive_noinc(uint64_t const* smem_ptr) {
#if HGGC_BARRIER_ENABLED
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
    "{\n\t"
    "ppu.cp.async.awbar.arrive.noinc.shared::cta.b64 [%0];\n\t"
    "}"
    :
    : "r"(smem_addr));
#endif
}


CUTLASS_DEVICE
static uint32_t test_wait(uint64_t const* smem_ptr, uint32_t phase, uint32_t pred) {
#if HGGC_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t waitComplete;

    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        ".reg .pred P2; \n\t"
        "ppu.cmpp.eq.u32 P2, %3, 1;\n\t"
        "@P2 ppu.awbar.test_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
        "ppu.selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase), "r"(pred));

    return waitComplete;
#endif
    return 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // end namespace arch
}  // end namespace cutlass
