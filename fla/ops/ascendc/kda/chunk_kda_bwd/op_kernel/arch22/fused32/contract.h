// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

namespace KDA::Arch22::Fused32 {
constexpr uint32_t CHUNK = 32;
constexpr uint32_t ARENA_BYTES = 112640;

// Host supplies one record per original sequence, including empty sequences.
// start is the flattened token offset; firstChunk excludes empty sequences.
struct Sequence {
    int64_t start, length, firstChunk;
};
struct Config {
    uint32_t batch, heads, tokens, keys, values, sequences, chunks;
    uint32_t packed, timeMajor, stateVFirst, normalize, betaMode, gateMode;
    uint32_t useExp2, recompute;
    float scale, epsilon, lowerBound;
    uint32_t gateTimeMajor=1, betaTimeMajor=1;
};
// All fields are byte offsets in a single block-private workspace slot.
struct Workspace {
    uint64_t state[2], tape;
    uint64_t qhat, khat, g, w, qg, kg, qgs, u, vn, aqk, akk;
    uint64_t ds, dqr, dvl, dvn, sw, sq, sk, sg, sb, qr, kr, be;
    uint64_t cube,dhuTemp;
    uint64_t bytes;
};
inline uint64_t Align32(uint64_t n) { return (n + 31) & ~uint64_t(31); }
// Called only after ACLNN has checked dimensions and 64-bit size overflow.
inline Workspace MakeWorkspace(const Config &c, bool backward) {
    Workspace w{};
    uint64_t n=0;
    auto reserve=[&](uint64_t count) { uint64_t p=n; n+=Align32(count*4); return p; };
    const uint64_t k=CHUNK*uint64_t(c.keys),v=CHUNK*uint64_t(c.values);
    w.state[0]=reserve(uint64_t(c.keys)*c.values);
    w.state[1]=reserve(uint64_t(c.keys)*c.values);
    w.tape=reserve(backward && c.recompute ? uint64_t(c.chunks)*c.keys*c.values : 0);
    w.qhat=reserve(k); w.khat=reserve(k); w.g=reserve(k); w.w=reserve(k);
    w.qg=reserve(k); w.kg=reserve(k); w.qgs=reserve(k);
    w.u=reserve(v); w.vn=reserve(v); w.aqk=reserve(CHUNK*CHUNK); w.akk=reserve(CHUNK*CHUNK);
    w.ds=reserve(CHUNK*CHUNK); w.dqr=reserve(k); w.dvl=reserve(v); w.dvn=reserve(v);
    w.sw=reserve(k); w.sq=reserve(k); w.sk=reserve(k); w.sg=reserve(k); w.sb=reserve(CHUNK);
    w.qr=reserve(CHUNK); w.kr=reserve(CHUNK); w.be=reserve(CHUNK);
    w.dhuTemp=reserve(backward?uint64_t(c.keys)*c.values:0);
    w.cube=n; if(backward)n+=32+3*128*128*4;
    w.bytes=n;
    return w;
}
} // namespace KDA::Arch22::Fused32
