// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "saved_contract.h"
#ifdef __aicore__
#define KDA_SAVED_FN __aicore__ inline
#else
#define KDA_SAVED_FN inline
#endif

namespace KDA::Arch22::Fused32::Saved {
// Offsets are bytes; strides are FP32 elements. Transpose is a view and never
// requests an external transpose operator, including physical V-first states.
struct Matrix {
    uint64_t offset;
    uint32_t rowStride, colStride;
    KDA_SAVED_FN Matrix Transpose() const { return {offset, colStride, rowStride}; }
};
enum class Operation : uint32_t {
    PrepareV, PrepareScore, PrepareQ,
    DhuV, DhuQ, DhuW,
    FinalizeW, FinalizeKg, FinalizeAk, FinalizeAv,
    FinalizeMiddle, FinalizeL, FinalizeRk, FinalizeRv,
};
enum class Store : uint32_t {
    Assign, Add, Negate, ScaleLower, NegateStrictLower, UpdateState,
};
struct Product {
    Matrix a, b, out;
    uint32_t m, n, k;
    Store store;
};

KDA_SAVED_FN Product Describe(Operation op, const GroupLayout &g, uint32_t scan,
                             uint32_t rows, uint32_t k, uint32_t v, bool vFirst,
                             Matrix stateOutput) {
    const uint64_t base = g.Record(scan);
    const auto &r = g.record;
    const Matrix aqk{base + r.Offset(Score::Aqk), CHUNK_ROWS, 1};
    const Matrix akk{base + r.Offset(Score::Akk), CHUNK_ROWS, 1};
    const Matrix ds{base + r.Offset(Score::Ds), CHUNK_ROWS, 1};
    const Matrix dl{base + r.Offset(Score::Dl), CHUNK_ROWS, 1};
    const Matrix da{base + r.Offset(Score::Da), CHUNK_ROWS, 1};
    const Matrix middle{base + r.Offset(Score::Middle), CHUNK_ROWS, 1};
    const Matrix vn{base + r.Offset(Value::Vnew), r.paddedV, 1};
    const Matrix dout{base + r.Offset(Value::Do), r.paddedV, 1};
    const Matrix du{base + r.Offset(Value::Du), r.paddedV, 1};
    // Value::Drv contains Rv until FinalizeAv has consumed it.
    const Matrix rv{base + r.Offset(Value::Drv), r.paddedV, 1};
    const Matrix qg{base + r.Offset(Key::Qg), r.paddedK, 1};
    const Matrix kg{base + r.Offset(Key::Kg), r.paddedK, 1};
    const Matrix w{base + r.Offset(Key::W), r.paddedK, 1};
    const Matrix rk{base + r.Offset(Key::Rk), r.paddedK, 1};
    const Matrix dw{base + r.Offset(Key::Dw), r.paddedK, 1};
    const Matrix dkg{base + r.Offset(Key::Dkg), r.paddedK, 1};
    const Matrix drk{base + r.Offset(Key::Drk), r.paddedK, 1};
    const Matrix dqr{base + r.Offset(Key::Dq), r.paddedK, 1};
    const uint32_t sr = vFirst ? 1 : r.paddedV;
    const uint32_t sc = vFirst ? r.paddedK : 1;
    const Matrix h{base + r.Offset(State::Hentry), sr, sc};
    const Matrix dh{base + r.Offset(State::Dnext), sr, sc};
    const Matrix tq{base + r.Offset(State::TermQ), sr, sc};
    switch (op) {
    case Operation::PrepareV: return {aqk.Transpose(), dout, du, rows, v, rows, Store::Assign};
    case Operation::PrepareScore: return {dout, vn.Transpose(), ds, rows, rows, v, Store::ScaleLower};
    case Operation::PrepareQ: return {dout, h.Transpose(), dqr, rows, k, v, Store::Assign};
    case Operation::DhuV: return {kg, dh, du, rows, v, k, Store::Add};
    case Operation::DhuQ: return {qg.Transpose(), dout, tq, k, v, rows, Store::Assign};
    // Explicit destination: the terminal step must never infer a fourth record
    // in the three-record ring. The caller supplies dh0 or a disposable state.
    case Operation::DhuW: return {w.Transpose(), du, stateOutput, k, v, rows, Store::UpdateState};
    case Operation::FinalizeW: return {du, h.Transpose(), dw, rows, k, v, Store::Negate};
    case Operation::FinalizeKg: return {vn, dh.Transpose(), dkg, rows, k, v, Store::Assign};
    case Operation::FinalizeAk: return {dw, rk.Transpose(), da, rows, rows, k, Store::Assign};
    case Operation::FinalizeAv: return {du, rv.Transpose(), da, rows, rows, v, Store::Add};
    case Operation::FinalizeMiddle: return {akk.Transpose(), da, middle, rows, rows, rows, Store::Assign};
    case Operation::FinalizeL: return {middle, akk.Transpose(), dl, rows, rows, rows, Store::NegateStrictLower};
    case Operation::FinalizeRk: return {akk.Transpose(), dw, drk, rows, k, rows, Store::Assign};
    case Operation::FinalizeRv: return {akk.Transpose(), du, rv, rows, v, rows, Store::Assign};
    }
    return {};
}

// The final scan step does not own scan+1. Its state result must be directed to
// the requested dh0 output, or discarded, rather than clobbering a live F record.
KDA_SAVED_FN bool HasNextState(uint32_t scan, uint32_t chunks) { return scan + 1 < chunks; }

struct Tile {
    uint32_t row, col, rows, cols, paddedRows, paddedCols;
};
KDA_SAVED_FN Tile OutputTile(const Product &p, uint32_t index) {
    const uint32_t columns = (p.n + 63) / 64;
    const uint32_t row = (index / columns) * 64;
    const uint32_t col = (index % columns) * 64;
    const uint32_t rows = p.m - row < 64 ? p.m - row : 64;
    const uint32_t cols = p.n - col < 64 ? p.n - col : 64;
    return {row, col, rows, cols, (rows + 15) & ~15U, (cols + 15) & ~15U};
}
KDA_SAVED_FN uint32_t OutputTiles(const Product &p) {
    return ((p.m + 63) / 64) * ((p.n + 63) / 64);
}

struct PacketSlice {
    uint64_t offset;
    uint32_t sourceStride, rows, cols, paddedRows, paddedCols;
};
// Each AIV owns half of the *physical* rows. Transposition happens on Cube;
// AIV never scatters scalar elements or launches an external transpose.
KDA_SAVED_FN PacketSlice Slice(Matrix view, uint32_t row, uint32_t col,
                              uint32_t rows, uint32_t cols,
                              uint32_t paddedRows, uint32_t paddedCols,
                              bool transpose, uint32_t lane) {
    if (transpose) {
        const uint32_t t = row; row = col; col = t;
        const uint32_t s = rows; rows = cols; cols = s;
        const uint32_t p = paddedRows; paddedRows = paddedCols; paddedCols = p;
    }
    const uint32_t half = paddedRows / 2;
    const uint32_t begin = lane * half;
    const uint32_t valid = rows > begin ? (rows - begin < half ? rows - begin : half) : 0;
    const uint32_t stride = transpose ? view.colStride : view.rowStride;
    return {view.offset + 4 * (uint64_t(row + begin) * stride + col),
            stride, valid, cols, half, paddedCols};
}
} // namespace KDA::Arch22::Fused32::Saved
#undef KDA_SAVED_FN
