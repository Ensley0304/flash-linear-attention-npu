#ifndef CHUNK_KDA_BWD_C_INTRA_COMMON_H
#define CHUNK_KDA_BWD_C_INTRA_COMMON_H

#include "chunk_kda_bwd_c_common.h"

namespace KDA {

constexpr uint32_t kCIntraChunkSize = 64;
constexpr uint32_t kCIntraHeadsPerWindow = 2;
constexpr uint32_t kCIntraWorkspaceSlots = 4;
constexpr uint32_t kCIntraVecReadyFlag = 2;
constexpr uint32_t kCIntraCubeReadyFlag = 4;
// Keep the mature Intra implementation's local names inside this private
// header; they are not exported outside Kernel C.
constexpr uint32_t kRowBlock = 16;
constexpr uint32_t kChunkSize = kCIntraChunkSize;
constexpr uint32_t kHeadsPerWindow = kCIntraHeadsPerWindow;
constexpr uint32_t kWorkspaceSlots = kCIntraWorkspaceSlots;
constexpr uint32_t kVecToCubeReadyFlag = kCIntraVecReadyFlag;
constexpr uint32_t kCubeToVecReadyFlag = kCIntraCubeReadyFlag;
constexpr float kLn2 = 0.69314718055994530942f;

template <uint32_t K_DIM, bool VARLEN_TND>
struct ProcessRowBlock {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // A5 dense K=128 combines two legacy 16-row tiles.  Varlen retains the
    // mature 16-row tail path.
    static constexpr uint32_t value =
        !VARLEN_TND && K_DIM == 128 ? 32 : kRowBlock;
#else
    static constexpr uint32_t value = kRowBlock;
#endif
};

struct CIntraTask {
    uint32_t batchIdx;
    uint32_t chunkIdx;
    uint32_t begin;
    uint32_t end;
};

__aicore__ inline uint32_t CIntraWorkspaceSlot(
    uint64_t windowIdx, uint32_t headInWindow)
{
    return static_cast<uint32_t>(
        ((windowIdx & 1U) << 1U) + headInWindow);
}

__aicore__ inline CIntraTask GetCIntraTask(
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkKdaBwdCTilingData &tiling, uint32_t taskIdx)
{
    const WyChunkTask task = GetWyChunkTask(
        cuSeqlens, chunkIndices, tiling, taskIdx);
    return {task.batchIdx, task.chunkIdx, task.begin, task.end};
}

__aicore__ inline uint64_t CIntraTensorOffset(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx, uint32_t col = 0)
{
    if (tiling.isVarLen != 0) {
        return (static_cast<uint64_t>(headIdx) * tiling.seqlen + tokenIdx) *
                   tiling.keyDim +
               col;
    }
    return ((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
                tiling.seqlen +
            tokenIdx) *
               tiling.keyDim +
           col;
}

__aicore__ inline uint64_t CIntraMatrixOffset(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx, uint32_t col = 0)
{
    if (tiling.isVarLen != 0) {
        return (static_cast<uint64_t>(headIdx) * tiling.seqlen + tokenIdx) *
                   tiling.chunkSize +
               col;
    }
    return ((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
                tiling.seqlen +
            tokenIdx) *
               tiling.chunkSize +
           col;
}

__aicore__ inline uint64_t CIntraScalarOffset(
    const ChunkKdaBwdCTilingData &tiling, uint32_t batchIdx,
    uint32_t headIdx, uint32_t tokenIdx)
{
    if (tiling.isVarLen != 0) {
        return static_cast<uint64_t>(headIdx) * tiling.seqlen + tokenIdx;
    }
    return (static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
               tiling.seqlen +
           tokenIdx;
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_C_INTRA_COMMON_H
