/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH22_POST_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH22_POST_H

#include <type_traits>

#include "kernel_operator.h"
#include "../chunk_kda_bwd_finalize_struct.h"

namespace KDA {

constexpr uint32_t KDA_FINALIZE_A2_CAST_TILE = 4096;

template <typename BetaT>
class ChunkKdaBwdFinalizeA2Post {
public:
    __aicore__ inline void Init(
        GM_ADDR dqFp32, GM_ADDR dkFp32, GM_ADDR dBetaFp32,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dBeta,
        const ChunkKdaBwdFinalizeTilingData *tiling, AscendC::TPipe *pipe)
    {
        batch_ = static_cast<uint32_t>(tiling->B);
        headNum_ = static_cast<uint32_t>(tiling->NV);
        seqlen_ = static_cast<uint32_t>(tiling->T);
        keyDim_ = static_cast<uint32_t>(tiling->K);
        coreIdx_ = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        coreNum_ = static_cast<uint32_t>(tiling->a2.usedCoreNum);

        dqFp32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqFp32));
        dkFp32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dkFp32));
        dBetaFp32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dBetaFp32));
        dq_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dq));
        dk_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dk));
        dBeta_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(dBeta));

        pipe->InitBuffer(inputQueue_, 1, KDA_FINALIZE_A2_CAST_TILE * sizeof(float));
        pipe->InitBuffer(bf16OutputQueue_, 1,
                         KDA_FINALIZE_A2_CAST_TILE * sizeof(bfloat16_t));
        pipe->InitBuffer(betaOutputQueue_, 1,
                         KDA_FINALIZE_A2_CAST_TILE * sizeof(BetaT));
    }

    __aicore__ inline void Process()
    {
        const uint32_t headWindows = (headNum_ + 1U) / 2U;
        for (uint32_t window = coreIdx_; window < headWindows;
             window += coreNum_) {
            const uint32_t head = window * 2U + subBlockIdx_;
            if (head >= headNum_) {
                continue;
            }
            for (uint32_t batch = 0; batch < batch_; ++batch) {
                const uint64_t vectorOffset =
                    (static_cast<uint64_t>(batch) * headNum_ + head) *
                    seqlen_ * keyDim_;
                const uint64_t scalarOffset =
                    (static_cast<uint64_t>(batch) * headNum_ + head) *
                    seqlen_;
                CastVector(dqFp32_, dq_, vectorOffset,
                           static_cast<uint64_t>(seqlen_) * keyDim_);
                CastVector(dkFp32_, dk_, vectorOffset,
                           static_cast<uint64_t>(seqlen_) * keyDim_);
                CastBeta(scalarOffset, seqlen_);
            }
        }
    }

    __aicore__ inline void ProcessBeta()
    {
        const uint32_t headWindows = (headNum_ + 1U) / 2U;
        for (uint32_t window = coreIdx_; window < headWindows;
             window += coreNum_) {
            const uint32_t head = window * 2U + subBlockIdx_;
            if (head >= headNum_) {
                continue;
            }
            for (uint32_t batch = 0; batch < batch_; ++batch) {
                const uint64_t scalarOffset =
                    (static_cast<uint64_t>(batch) * headNum_ + head) *
                    seqlen_;
                CastBeta(scalarOffset, seqlen_);
            }
        }
    }

private:
    __aicore__ inline void CastVector(
        AscendC::GlobalTensor<float> input,
        AscendC::GlobalTensor<bfloat16_t> output,
        uint64_t begin, uint64_t count)
    {
        for (uint64_t localOffset = 0; localOffset < count;
             localOffset += KDA_FINALIZE_A2_CAST_TILE) {
            const uint32_t current = static_cast<uint32_t>(
                count - localOffset < KDA_FINALIZE_A2_CAST_TILE ?
                    count - localOffset : KDA_FINALIZE_A2_CAST_TILE);
            const uint64_t offset = begin + localOffset;
            auto in = inputQueue_.AllocTensor<float>();
            AscendC::DataCopyPad(
                in, input[offset],
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(current * sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
            inputQueue_.EnQue(in);
            auto ready = inputQueue_.DeQue<float>();
            auto out = bf16OutputQueue_.AllocTensor<bfloat16_t>();
            AscendC::Cast(out, ready, AscendC::RoundMode::CAST_RINT, current);
            bf16OutputQueue_.EnQue(out);
            auto packed = bf16OutputQueue_.DeQue<bfloat16_t>();
            AscendC::DataCopyPad(
                output[offset], packed,
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(current * sizeof(bfloat16_t)), 0, 0, 0});
            bf16OutputQueue_.FreeTensor(packed);
            inputQueue_.FreeTensor(ready);
        }
    }

    __aicore__ inline void CastBeta(uint64_t begin, uint64_t count)
    {
        for (uint64_t localOffset = 0; localOffset < count;
             localOffset += KDA_FINALIZE_A2_CAST_TILE) {
            const uint32_t current = static_cast<uint32_t>(
                count - localOffset < KDA_FINALIZE_A2_CAST_TILE ?
                    count - localOffset : KDA_FINALIZE_A2_CAST_TILE);
            const uint64_t offset = begin + localOffset;
            auto in = inputQueue_.AllocTensor<float>();
            AscendC::DataCopyPad(
                in, dBetaFp32_[offset],
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(current * sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
            inputQueue_.EnQue(in);
            auto ready = inputQueue_.DeQue<float>();
            auto out = betaOutputQueue_.AllocTensor<BetaT>();
            if constexpr (std::is_same<BetaT, float>::value) {
                AscendC::Adds(out, ready, 0.0f, current);
            } else {
                AscendC::Cast(out, ready, AscendC::RoundMode::CAST_RINT, current);
            }
            betaOutputQueue_.EnQue(out);
            auto packed = betaOutputQueue_.DeQue<BetaT>();
            AscendC::DataCopyPad(
                dBeta_[offset], packed,
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(current * sizeof(BetaT)), 0, 0, 0});
            betaOutputQueue_.FreeTensor(packed);
            inputQueue_.FreeTensor(ready);
        }
    }

    uint32_t batch_ = 1;
    uint32_t headNum_ = 1;
    uint32_t seqlen_ = 1;
    uint32_t keyDim_ = 1;
    uint32_t coreIdx_ = 0;
    uint32_t subBlockIdx_ = 0;
    uint32_t coreNum_ = 1;
    AscendC::GlobalTensor<float> dqFp32_;
    AscendC::GlobalTensor<float> dkFp32_;
    AscendC::GlobalTensor<float> dBetaFp32_;
    AscendC::GlobalTensor<bfloat16_t> dq_;
    AscendC::GlobalTensor<bfloat16_t> dk_;
    AscendC::GlobalTensor<BetaT> dBeta_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> bf16OutputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> betaOutputQueue_;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH22_POST_H
