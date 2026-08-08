#include "kernel_operator.h"
#include "chunk_kda_bwd_gate_post_struct.h"

namespace KDA {
constexpr uint32_t kGatePostChunkSize = 64;
constexpr uint32_t kGatePostKeyDim = 128;

class ChunkKdaBwdGatePostKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR dgHv, GM_ADDR dgOut, const ChunkKdaBwdGatePostTilingData &tiling)
    {
        tiling_ = tiling;
        dgHv_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dgHv));
        dgOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dgOut));
        constexpr uint32_t chunkBytes =
            kGatePostChunkSize * kGatePostKeyDim * sizeof(float);
        pipe_.InitBuffer(chunkQueue_, 1, chunkBytes);
        pipe_.InitBuffer(scanBuf_, chunkBytes);
        vToMte3Event_ = pipe_.AllocEventID<AscendC::HardEvent::V_MTE3>();
        mte3ToVEvent_ = pipe_.AllocEventID<AscendC::HardEvent::MTE3_V>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        for (uint64_t task = coreIdx; task < tiling_.taskNum; task += coreNum) {
            const uint64_t chunkIdx = task % tiling_.chunkNumPerBatch;
            const uint64_t owner = task / tiling_.chunkNumPerBatch;
            const uint64_t tokenBegin = chunkIdx * tiling_.chunkSize;
            const uint64_t validLen =
                tokenBegin + tiling_.chunkSize <= tiling_.seqlen
                    ? tiling_.chunkSize
                    : tiling_.seqlen - tokenBegin;
            const uint64_t gmOffset =
                (owner * tiling_.seqlen + tokenBegin) * tiling_.keyDim;
            const uint32_t elementCount = static_cast<uint32_t>(validLen * tiling_.keyDim);

            auto chunk = chunkQueue_.AllocTensor<float>();
            AscendC::DataCopy(chunk, dgHv_[gmOffset], elementCount);
            chunkQueue_.EnQue(chunk);
            chunk = chunkQueue_.DeQue<float>();

            auto scan = scanBuf_.Get<float>();
            bool nextSrcIsChunk = true;
            for (uint32_t step = 1; step < validLen; step <<= 1U) {
                auto src = nextSrcIsChunk ? chunk : scan;
                auto dst = nextSrcIsChunk ? scan : chunk;
                const uint32_t activeRows = static_cast<uint32_t>(validLen) - step;
                AscendC::Add(
                    dst, src, src[step * tiling_.keyDim],
                    activeRows * static_cast<uint32_t>(tiling_.keyDim));
                AscendC::Adds(
                    dst[activeRows * tiling_.keyDim],
                    src[activeRows * tiling_.keyDim], 0.0f,
                    step * static_cast<uint32_t>(tiling_.keyDim));
                AscendC::PipeBarrier<PIPE_V>();
                nextSrcIsChunk = !nextSrcIsChunk;
            }
            auto output = nextSrcIsChunk ? chunk : scan;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_);
            AscendC::DataCopy(dgOut_[gmOffset], output, elementCount);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_);
            chunkQueue_.FreeTensor(chunk);
        }
        pipe_.ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3Event_);
        pipe_.ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_);
    }

private:
    ChunkKdaBwdGatePostTilingData tiling_{};
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<float> dgHv_;
    AscendC::GlobalTensor<float> dgOut_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> chunkQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scanBuf_;
    AscendC::TEventID vToMte3Event_;
    AscendC::TEventID mte3ToVEvent_;
};
} // namespace KDA

extern "C" __global__ __aicore__ void chunk_kda_bwd_gate_post(
    GM_ADDR dg_hv, GM_ADDR dg, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(KDA::ChunkKdaBwdGatePostTilingData);
    GET_TILING_DATA_WITH_STRUCT(KDA::ChunkKdaBwdGatePostTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KDA::ChunkKdaBwdGatePostKernel kernel;
    kernel.Init(dg_hv, dg, tilingData);
    kernel.Process();
}
