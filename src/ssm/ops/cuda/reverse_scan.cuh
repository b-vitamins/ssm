#pragma once

#ifndef USE_ROCM
    #include <cub/config.cuh>

    #include <cub/util_ptx.cuh>
    #include <cub/util_type.cuh>
    #include <cub/block/block_raking_layout.cuh>
    // #include <cub/detail/uninitialized_copy.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif
#include "uninitialized_copy.cuh"

/**
 * Perform a reverse sequential reduction over \p LENGTH elements of the \p input array.  The aggregate is returned.
 */
template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReverseReduce(const T (&input)[LENGTH], ReductionOp reduction_op) {
    static_assert(LENGTH > 0);
    T retval = input[LENGTH - 1];
    #pragma unroll
    for (int i = LENGTH - 2; i >= 0; --i) { retval = reduction_op(retval, input[i]); }
    return retval;
}

/**
 * Perform a sequential inclusive postfix reverse scan over the statically-sized \p input array, seeded with the specified \p postfix.  The aggregate is returned.
 */
template <
    int         LENGTH,
    typename    T,
    typename    ScanOp>
__device__ __forceinline__ T ThreadReverseScanInclusive(
    const T (&input)[LENGTH],
    T (&output)[LENGTH],
    ScanOp scan_op,
    const T postfix)
{
    T inclusive = postfix;
    #pragma unroll
    for (int i = LENGTH - 1; i >= 0; --i) {
        inclusive = scan_op(inclusive, input[i]);
        output[i] = inclusive;
    }
    return inclusive;
}

/**
 * Perform a sequential exclusive postfix reverse scan over the statically-sized \p input array, seeded with the specified \p postfix.  The aggregate is returned.
 */
template <
    int         LENGTH,
    typename    T,
    typename    ScanOp>
__device__ __forceinline__ T ThreadReverseScanExclusive(
    const T (&input)[LENGTH],
    T (&output)[LENGTH],
    ScanOp scan_op,
    const T postfix)
{
    // Careful, output maybe be aliased to input
    T exclusive = postfix;
    T inclusive;
    #pragma unroll
    for (int i = LENGTH - 1; i >= 0; --i) {
        inclusive = scan_op(exclusive, input[i]);
        output[i] = exclusive;
        exclusive = inclusive;
    }
    return inclusive;
}


/**
 * \brief WarpReverseScan provides SHFL-based variants of parallel postfix scan of items partitioned across a CUDA thread warp.
 *
 * LOGICAL_WARP_THREADS must be a power-of-two
 */
template <
    typename    T,                      ///< Data type being scanned
    int         LOGICAL_WARP_THREADS    ///< Number of threads per logical warp
    >
struct WarpReverseScan {
    //---------------------------------------------------------------------
    // Constants and type definitions
    //---------------------------------------------------------------------

    /// Whether the logical warp size and the PTX warp size coincide

    // In hipcub, warp_threads is defined as HIPCUB_WARP_THREADS ::rocprim::warp_size()
    // While in cub, it's defined as a macro that takes a redundant unused argument.
    #ifndef USE_ROCM
        #define WARP_THREADS CUB_WARP_THREADS(0)
    #else
        #define WARP_THREADS HIPCUB_WARP_THREADS
    #endif
    static constexpr bool IS_ARCH_WARP = (LOGICAL_WARP_THREADS == WARP_THREADS);
    /// The number of warp scan steps
    static constexpr int STEPS = cub::Log2<LOGICAL_WARP_THREADS>::VALUE;
    static_assert(LOGICAL_WARP_THREADS == 1 << STEPS);


    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    /// Lane index in logical warp
    unsigned int lane_id;

    /// Logical warp index in 32-thread physical warp
    unsigned int warp_id;

    /// 32-thread physical warp member mask of logical warp
    unsigned int member_mask;

    //---------------------------------------------------------------------
    // Construction
    //---------------------------------------------------------------------

    /// Constructor
    explicit __device__ __forceinline__
    WarpReverseScan()
        : lane_id(threadIdx.x & 0x1f)
        , warp_id(IS_ARCH_WARP ? 0 : (lane_id / LOGICAL_WARP_THREADS))
        , member_mask(cub::WarpMask<LOGICAL_WARP_THREADS>(warp_id))
    {
        if (!IS_ARCH_WARP) {
            lane_id = lane_id % LOGICAL_WARP_THREADS;
        }
    }


    /// Broadcast
    __device__ __forceinline__ T Broadcast(
        T               input,              ///< [in] The value to broadcast
        int             src_lane)           ///< [in] Which warp lane is to do the broadcasting
    {
        return cub::ShuffleIndex<LOGICAL_WARP_THREADS>(input, src_lane, member_mask);
    }


    /// Inclusive scan
    template <typename ScanOpT>
    __device__ __forceinline__ void InclusiveReverseScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &inclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOpT         scan_op)            ///< [in] Binary scan operator
    {
        inclusive_output = input;
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++) {
            int offset = 1 << STEP;
            T temp = cub::ShuffleDown<LOGICAL_WARP_THREADS>(
                inclusive_output, offset, LOGICAL_WARP_THREADS - 1, member_mask
            );
            // Perform scan op if from a valid peer
            inclusive_output = static_cast<int>(lane_id) >= LOGICAL_WARP_THREADS - offset
                ? inclusive_output : scan_op(temp, inclusive_output);
        }
    }

    /// Exclusive scan
    // Get exclusive from inclusive
    template <typename ScanOpT>
    __device__ __forceinline__ void ExclusiveReverseScan(
        T              input,              ///< [in] Calling thread's input item.
        T              &exclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
        ScanOpT        scan_op,            ///< [in] Binary scan operator
        T              &warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        T inclusive_output;
        InclusiveReverseScan(input, inclusive_output, scan_op);
        warp_aggregate = cub::ShuffleIndex<LOGICAL_WARP_THREADS>(inclusive_output, 0, member_mask);
        // initial value unknown
        exclusive_output = cub::ShuffleDown<LOGICAL_WARP_THREADS>(
            inclusive_output, 1, LOGICAL_WARP_THREADS - 1, member_mask
        );
    }

    /**
     * \brief Computes both inclusive and exclusive reverse scans using the specified binary scan functor across the calling warp.  Because no initial value is supplied, the \p exclusive_output computed for the last <em>warp-lane</em> is undefined.
     */
    template <typename ScanOpT>
    __device__ __forceinline__ void ReverseScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &inclusive_output,  ///< [out] Calling thread's inclusive-scan output item.
        T               &exclusive_output,  ///< [out] Calling thread's exclusive-scan output item.
        ScanOpT         scan_op)            ///< [in] Binary scan operator
    {
        InclusiveReverseScan(input, inclusive_output, scan_op);
        // initial value unknown
        exclusive_output = cub::ShuffleDown<LOGICAL_WARP_THREADS>(
            inclusive_output, 1, LOGICAL_WARP_THREADS - 1, member_mask
        );
    }

    //---------------------------------------------------------------------
    // Three-scan variant for single-pass multi-block scan
    //---------------------------------------------------------------------

    template <typename ScanOpT>
    __device__ __forceinline__ void ReverseScan(
        T               input,              ///< [in] Calling thread's input item.
        T               &exclusive_output,  ///< [out] Calling thread's exclusive-scan output item.
        T               &inclusive_output,  ///< [out] Calling thread's inclusive-scan output item.
        ScanOpT         scan_op,            ///< [in] Binary scan operator
        T               &warp_aggregate,    ///< [out] Warp-wide aggregate reduction of input items.
        T               identity)           ///< [in] Identity value for scan operator
    {
        // Have the first thread initialize inclusive_output
        if (lane_id == 0) { detail::uninitialized_copy(&inclusive_output, identity); }
        __syncwarp();
        ReverseScan(input, inclusive_output, exclusive_output, scan_op);
        // exclusive_output is invalid for the last warp lane, so correct it for when
        // it is used by the last warp lane
        if (lane_id == LOGICAL_WARP_THREADS - 1) { detail::uninitialized_copy(&exclusive_output, identity); }
        warp_aggregate = inclusive_output;
    }

};
#undef WARP_THREADS

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int LOGICAL_WARP_THREADS = 32>
struct BlockReverseScan
{
    //---------------------------------------------------------------------
    // Constants and type definitions
    //---------------------------------------------------------------------

    /// Constants for logical warp occupancy of threadblock tiles
    static constexpr int LOGICAL_WARPS = 32 / LOGICAL_WARP_THREADS;
    static constexpr int TILE_ITEMS = 128;
    static constexpr int WARPS_CAN_USE_SHMEM = LOGICAL_WARPS + 1;

    /// Shared memory storage layout type
    union TempStorage
    {
        T smem[WARPS_CAN_USE_SHMEM][LOGICAL_WARP_THREADS];
        cub::TileExchange<typename cub::UnitWord<T>::Type, LOGICAL_WARPS, LOGICAL_WARP_THREADS> exchange;
    };

    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    /// Local reference to temp_storage.per_warp
    T (&smem)[WARPS_CAN_USE_SHMEM][LOGICAL_WARP_THREADS];

    /// BlockScan utility class for handling partial warp-tiles
    cub::BlockScan<T, LOGICAL_WARPS> block_scan;

    /// Local scan utility class for per-thread prefix-sum results (executes in registers)
    WarpReverseScan<T, LOGICAL_WARP_THREADS> warp_scan;

    //---------------------------------------------------------------------
    // Construction
    //---------------------------------------------------------------------

    /// Constructor
    explicit __device__ __forceinline__ BlockReverseScan(void *smem)
        : smem(*reinterpret_cast<T (*)[WARPS_CAN_USE_SHMEM][LOGICAL_WARP_THREADS]>(smem))
        , block_scan(reinterpret_cast<typename decltype(block_scan)::TempStorage &>(smem[WARPS_CAN_USE_SHMEM - 1]))
        , warp_scan()
    {
    }

    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * \\brief Computes an exclusive reverse prefix sum of the input elements using the specified binary scan functor.
     */
    template <typename ScanOpT>
    __device__ __forceinline__ void ExclusiveReverseSum(
        T (&input)[LOGICAL_WARP_THREADS],
        T (&exclusive_output)[LOGICAL_WARP_THREADS],
        ScanOpT scan_op,
        T block_prefix,
        T &block_aggregate)
    {
        // Compute exclusive warp scan of the inputs
        T warp_exclusive, warp_inclusive;
        warp_scan.ReverseScan(input[threadIdx.x % LOGICAL_WARP_THREADS], warp_inclusive, warp_exclusive, scan_op);
        detail::uninitialized_copy(&exclusive_output[threadIdx.x % LOGICAL_WARP_THREADS], warp_exclusive);

        // Cache the inclusive warp-scan result in shared memory
        detail::uninitialized_copy(&smem[threadIdx.x % LOGICAL_WARPS][threadIdx.x / LOGICAL_WARP_THREADS], warp_inclusive);
        __syncthreads();

        // Grab the block-wide prefix from smem into raking_grid_space
        if (threadIdx.x < LOGICAL_WARPS)
        {
            detail::uninitialized_copy(&smem[LOGICAL_WARPS][threadIdx.x],
                                       (threadIdx.x == LOGICAL_WARPS - 1) ? block_prefix : smem[threadIdx.x][LOGICAL_WARP_THREADS - 1]);
        }
        __syncthreads();

        T prefix = smem[LOGICAL_WARPS][threadIdx.x % LOGICAL_WARP_THREADS];
        // Prefix the exclusive warp-scan result with the exclusive prefix in registers
        detail::uninitialized_copy(&exclusive_output[threadIdx.x % LOGICAL_WARP_THREADS], scan_op(prefix, exclusive_output[threadIdx.x % LOGICAL_WARP_THREADS]));

        if (threadIdx.x == 0)
        {
            block_aggregate = smem[LOGICAL_WARPS][0];
        }
    }

};

