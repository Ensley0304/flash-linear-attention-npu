// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

#ifdef __aicore__
#define KDA_SAVED_DEVICE __aicore__
#else
#define KDA_SAVED_DEVICE
#endif

namespace KDA::Arch22::Fused32::Saved {
constexpr uint32_t CHUNK_ROWS = 32;
constexpr uint32_t RECORDS = 3;
constexpr uint32_t BANKS = 2;
constexpr uint64_t ALIGNMENT = 256;
constexpr uint64_t INPUT_BANK_BYTES = 64 * 1024;
constexpr uint64_t RESULT_BANK_BYTES = 32 * 1024;
constexpr uint32_t INVALID_SCAN = UINT32_MAX;

enum class Score : uint32_t { Aqk, Akk, Ds, Dl, Da, Middle, Count };
enum class Key : uint32_t { Qhat, Khat, G, Qg, Kg, W, Dw, Drk, Dq, Dk, Dg, Rk, Dkg, Count };
enum class Value : uint32_t { V, Vnew, Do, Du, Drv, Count };
enum class State : uint32_t { Hentry, Dnext, TermQ, Count };

// Byte offsets, shared by Host and device. No pointer aliases between records.
struct RecordLayout {
    uint64_t score, key, value, state, small, bytes;
    uint64_t scoreBytes, keyBytes, valueBytes, stateBytes;
    uint32_t paddedK, paddedV;
    KDA_SAVED_DEVICE constexpr uint64_t Offset(Score x) const { return score + uint32_t(x) * scoreBytes; }
    KDA_SAVED_DEVICE constexpr uint64_t Offset(Key x) const { return key + uint32_t(x) * keyBytes; }
    KDA_SAVED_DEVICE constexpr uint64_t Offset(Value x) const { return value + uint32_t(x) * valueBytes; }
    KDA_SAVED_DEVICE constexpr uint64_t Offset(State x) const { return state + uint32_t(x) * stateBytes; }
};
struct GroupLayout {
    RecordLayout record;
    uint64_t inputs, results, exchange, bytes;
    KDA_SAVED_DEVICE constexpr uint64_t Record(uint32_t scan) const { return (scan % RECORDS) * record.bytes; }
    KDA_SAVED_DEVICE constexpr uint64_t Input(uint64_t packet) const { return inputs + (packet % BANKS) * INPUT_BANK_BYTES; }
    KDA_SAVED_DEVICE constexpr uint64_t Result(uint64_t output) const { return results + (output % BANKS) * RESULT_BANK_BYTES; }
};
struct Layout {
    GroupLayout group;
    uint64_t workCount, partial, partialStride, bytes;
    uint32_t blocks;
};

inline bool CheckedAdd(uint64_t a, uint64_t b, uint64_t &out) {
    if (a > UINT64_MAX - b) return false;
    out = a + b;
    return true;
}
inline bool CheckedMul(uint64_t a, uint64_t b, uint64_t &out) {
    if (b && a > UINT64_MAX / b) return false;
    out = a * b;
    return true;
}
inline bool Align(uint64_t n, uint64_t &out) {
    if (!CheckedAdd(n, ALIGNMENT - 1, out)) return false;
    out &= ~(ALIGNMENT - 1);
    return true;
}

// Not a public validator: ACLNN owns the public descriptor checks. Failure leaves
// the caller's layout untouched, including on overflow in partial allocation.
inline bool MakeLayout(uint32_t k, uint32_t v, uint64_t sequences,
                       uint64_t heads, uint32_t aic, Layout &out) {
    const bool square = k == v && (k == 60 || k == 64 || k == 100 || k == 128 || k == 256);
    if ((!square && !(k == 64 && v == 32)) || !sequences || !heads || !aic) return false;
    Layout l{};
    auto &g = l.group;
    auto &r = g.record;
    r.paddedK = (k + 15U) & ~15U;
    r.paddedV = (v + 15U) & ~15U;
    r.scoreBytes = uint64_t(CHUNK_ROWS) * CHUNK_ROWS * sizeof(float);
    r.keyBytes = uint64_t(CHUNK_ROWS) * r.paddedK * sizeof(float);
    r.valueBytes = uint64_t(CHUNK_ROWS) * r.paddedV * sizeof(float);
    r.stateBytes = uint64_t(r.paddedK) * r.paddedV * sizeof(float);
    r.score = 0;
    r.key = uint32_t(Score::Count) * r.scoreBytes;
    r.value = r.key + uint32_t(Key::Count) * r.keyBytes;
    r.state = r.value + uint32_t(Value::Count) * r.valueBytes;
    r.small = r.state + uint32_t(State::Count) * r.stateBytes;
    // Four scalar rows plus a separate 256-byte control record.
    if (!Align(r.small + 16 * CHUNK_ROWS + 256, r.bytes)) return false;
    g.inputs = RECORDS * r.bytes;
    g.results = g.inputs + BANKS * INPUT_BANK_BYTES;
    g.exchange = g.results + BANKS * RESULT_BANK_BYTES;
    if (!Align(g.exchange + 32 * CHUNK_ROWS, g.bytes)) return false;
    if (!CheckedMul(sequences, heads, l.workCount)) return false;
    l.blocks = l.workCount < aic ? uint32_t(l.workCount) : aic;
    if (!CheckedMul(l.blocks, g.bytes, l.partial)) return false;
    if (!Align(4ULL * (r.paddedK + 1), l.partialStride)) return false;
    uint64_t partialBytes;
    if (!CheckedMul(l.workCount, l.partialStride, partialBytes) ||
        !CheckedAdd(l.partial, partialBytes, l.bytes)) return false;
    out = l;
    return true;
}

struct Chunk {
    uint64_t token, globalChunk;
    uint32_t scan, physical, rows, record;
};
// length and firstChunk come from validated Host sequence metadata. The device
// never searches cu_seqlens or infers a different task order on either AIV.
KDA_SAVED_DEVICE constexpr uint32_t ChunkCount(uint32_t length) {
    return length / CHUNK_ROWS + uint32_t(length % CHUNK_ROWS != 0);
}
KDA_SAVED_DEVICE constexpr Chunk Resolve(uint64_t start, uint32_t length, uint64_t firstChunk, uint32_t scan) {
    const uint32_t n = ChunkCount(length);
    if (scan >= n) return {0, 0, INVALID_SCAN, 0, 0, 0};
    const uint32_t physical = n - 1 - scan;
    const uint64_t local = uint64_t(physical) * CHUNK_ROWS;
    const uint32_t remaining = uint32_t(uint64_t(length) - local);
    return {start + local, firstChunk + physical, scan, physical,
            remaining < CHUNK_ROWS ? remaining : CHUNK_ROWS, scan % RECORDS};
}
struct Step {
    uint32_t scan, prepare, finalize;
};
KDA_SAVED_DEVICE constexpr Step ResolveStep(uint32_t scan, uint32_t chunks) {
    return {scan, scan + 1 < chunks ? scan + 1 : INVALID_SCAN,
            scan ? scan - 1 : INVALID_SCAN};
}
// Input packet and output tile counters are deliberately distinct. A K=256
// contraction has two 128-wide packets but only one completed output tile.
KDA_SAVED_DEVICE constexpr uint32_t ReductionPackets(uint32_t width) {
    return width / 128 + uint32_t(width % 128 != 0);
}
KDA_SAVED_DEVICE constexpr uint16_t InputReady(uint64_t packet) { return uint16_t(packet % BANKS); }
KDA_SAVED_DEVICE constexpr uint16_t InputFree(uint64_t packet) { return uint16_t(2 + packet % BANKS); }
KDA_SAVED_DEVICE constexpr uint16_t ResultReady(uint64_t output) { return uint16_t(4 + output % BANKS); }
KDA_SAVED_DEVICE constexpr uint16_t ResultFree(uint64_t output) { return uint16_t(6 + output % BANKS); }
constexpr uint16_t RETIRE_READY = 8;
constexpr uint16_t RETIRE_ACK = 9;
static_assert(uint32_t(Key::Count) == 13 && uint32_t(Score::Count) == 6);
static_assert(RETIRE_ACK < 14, "SDK flags 14/15 must remain reserved");
} // namespace KDA::Arch22::Fused32::Saved

#undef KDA_SAVED_DEVICE
