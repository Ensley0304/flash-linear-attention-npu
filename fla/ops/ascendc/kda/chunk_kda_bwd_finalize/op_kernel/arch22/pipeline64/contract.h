// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <limits>

#ifdef __aicore__
#define KDA_C64_DEVICE __aicore__
#else
#define KDA_C64_DEVICE
#endif

namespace KDA::Arch22::Pipeline64 {
constexpr uint32_t C = 64;
constexpr uint64_t ALIGN = 256;
constexpr uint64_t INPUT_BANK_BYTES = 64 * 1024;
constexpr uint64_t RESULT_BANK_BYTES = 32 * 1024;
enum class Score : uint32_t { Da, Middle, Dl, Rkk, Query, Key, Count };
enum class Key : uint32_t { Rk, Dw, Drk, Dkg, Dq, Dk, Dg, Band, Count };
enum class Value : uint32_t { Rv, Drv, Do, Count };

struct Work {
    uint64_t chunk;
    uint32_t firstHead, heads, generation;
};
struct Layout {
    uint32_t k, v, kp, vp, cg, depth, blocks;
    uint64_t workCount, scoreBytes, keyBytes, valueBytes;
    uint64_t keys, values, recordBytes, inputs, results, exchange, groupBytes;
    uint64_t partial, partialStride, bytes;
    KDA_C64_DEVICE constexpr uint64_t Record(uint32_t generation, uint32_t head) const {
        return (uint64_t(generation) * cg + head) * recordBytes;
    }
    KDA_C64_DEVICE constexpr uint64_t Offset(Score x) const { return uint32_t(x)*scoreBytes; }
    KDA_C64_DEVICE constexpr uint64_t Offset(Key x) const { return keys+uint32_t(x)*keyBytes; }
    KDA_C64_DEVICE constexpr uint64_t Offset(Value x) const { return values+uint32_t(x)*valueBytes; }
};
inline bool Add(uint64_t a, uint64_t b, uint64_t &out) {
    if (a > UINT64_MAX-b) return false;
    out=a+b; return true;
}
inline bool Mul(uint64_t a, uint64_t b, uint64_t &out) {
    if (b && a > UINT64_MAX/b) return false;
    out=a*b; return true;
}
inline bool Align(uint64_t a, uint64_t &out) {
    if (!Add(a,ALIGN-1,out)) return false;
    out &= ~(ALIGN-1); return true;
}

// Pure descriptor builder. Public descriptor validation remains in ACLNN.
// CG is explicit: the host selects a measured candidate, not an implicit CG4.
inline bool MakeLayout(uint32_t k, uint32_t v, uint64_t chunks, uint32_t heads,
                       uint32_t cg, uint32_t aic, uint64_t systemBytes, Layout &out) {
    const bool square = k==v && (k==60 || k==64 || k==100 || k==128 || k==256);
    if ((!square && !(k==64 && v==32)) || !heads || !aic ||
        (cg!=1 && cg!=2 && cg!=4)) return false;
    Layout l{};
    l.k=k; l.v=v; l.kp=(k+15)&~15U; l.vp=(v+15)&~15U;
    l.cg=cg; l.depth=cg==4 ? 1 : 2;
    const uint64_t groups=(uint64_t(heads)+cg-1)/cg;
    if (!Mul(chunks,groups,l.workCount)) return false;
    // An empty packed input still launches the parameter-zero writer. No
    // fictitious chunk is assigned; all participants reach the final barrier.
    const uint64_t workOrOne=l.workCount ? l.workCount : 1;
    l.blocks=uint32_t(workOrOne<aic ? workOrOne : aic);
    l.scoreBytes=4ULL*C*C; l.keyBytes=4ULL*C*l.kp; l.valueBytes=4ULL*C*l.vp;
    l.keys=uint32_t(Score::Count)*l.scoreBytes;
    l.values=l.keys+uint32_t(Key::Count)*l.keyBytes;
    if (!Align(l.values+uint32_t(Value::Count)*l.valueBytes,l.recordBytes)) return false;
    l.inputs=l.recordBytes*l.cg*l.depth;
    l.results=l.inputs+2*INPUT_BANK_BYTES;
    l.exchange=l.results+2*RESULT_BANK_BYTES;
    if (!Align(l.exchange+32ULL*C*l.cg*l.depth,l.groupBytes)) return false;
    if (!Mul(l.groupBytes,l.blocks,l.partial)) return false;
    if (!Align(4ULL*(l.kp+1),l.partialStride)) return false;
    uint64_t partialCount,partialBytes;
    if (!Mul(chunks,heads,partialCount) || !Mul(partialCount,l.partialStride,partialBytes) ||
        !Add(l.partial,partialBytes,l.bytes)) return false;
    uint64_t total;
    if (!Add(l.bytes,systemBytes,total) || total>uint64_t(INT64_MAX)) return false;
    out=l; return true;
}

KDA_C64_DEVICE constexpr Work ResolveWork(uint64_t work, uint64_t iteration,
                                         uint32_t heads, const Layout &l) {
    const uint64_t groups=(uint64_t(heads)+l.cg-1)/l.cg;
    const uint32_t first=uint32_t(work%groups)*l.cg;
    const uint32_t remaining=heads-first;
    return {work/groups,first,remaining<l.cg ? remaining:l.cg,uint32_t(iteration%l.depth)};
}

// Physical on-chip arenas: live slots never alias the two transfer banks.
constexpr uint64_t L1_RESIDENT=256*1024, L1_INPUT=128*1024, L1_RELOAD=64*1024;
constexpr uint64_t UB_RESULTS=64*1024, UB_INPUTS=64*1024, UB_TEMP=32*1024;
constexpr uint64_t UB_SMALL=16*1024, UB_META=8*1024;
static_assert(L1_RESIDENT+L1_INPUT+L1_RELOAD<=512*1024);
static_assert(UB_RESULTS+UB_INPUTS+UB_TEMP+UB_SMALL+UB_META==184*1024);
} // namespace KDA::Arch22::Pipeline64
#undef KDA_C64_DEVICE
