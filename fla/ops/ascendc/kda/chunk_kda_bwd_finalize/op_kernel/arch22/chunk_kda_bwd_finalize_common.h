/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH22_COMMON_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH22_COMMON_H

#include <cstdint>

namespace KDA::Arch22 {

constexpr uint32_t CHUNK_SIZE = 64;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t VECTOR_ROWS = 16;
constexpr uint32_t HEADS_PER_WORK = 2;
constexpr uint32_t FRACTAL_ELEMENTS = 16 * 16;

// Byte offsets from design/L1内存.md. Both heads keep their own L1 lane.
constexpr uint32_t L1_HEAD_BYTES = 192 * 1024;
constexpr uint32_t L1_BYTES = 512 * 1024;
constexpr uint32_t L1_AKK = 0;
constexpr uint32_t L1_DW = 8 * 1024;
constexpr uint32_t L1_KE = 24 * 1024;
constexpr uint32_t L1_ZB = 40 * 1024;
constexpr uint32_t L1_TZA = 48 * 1024;
constexpr uint32_t L1_AKK_BANK_STRIDE = 56 * 1024;
constexpr uint32_t L1_TZA_BANK_STRIDE = 128 * 1024;
constexpr uint32_t L1_H = 64 * 1024;
constexpr uint32_t L1_DH = 96 * 1024;
constexpr uint32_t L1_V_NEW = 128 * 1024;
constexpr uint32_t L1_DV_SCAN = 144 * 1024;
constexpr uint32_t L1_V = 160 * 1024;
constexpr uint32_t L0_AB_SLOT_BYTES = 32 * 1024;
constexpr uint32_t L0_C_SLOT_BYTES = 64 * 1024;

constexpr uint32_t UB_BYTES = 192 * 1024;
constexpr uint32_t UB_K = 16 * 1024;
constexpr uint32_t UB_EXP2_GK = 32 * 1024;
constexpr uint32_t UB_BETA = 80 * 1024;
constexpr uint32_t UB_DB_V = 80 * 1024 + 256;
constexpr uint32_t UB_RH = 81 * 1024 + 256;
constexpr uint32_t UB_GK_LAST = 81 * 1024 + 768;
constexpr uint32_t UB_WORK = 88 * 1024;
constexpr uint32_t UB_WORK_SLOT_BYTES = 8 * 1024;

// Two generations are used only by the Stage 0-7 development scheduler.
constexpr uint32_t WS_BANK_BYTES = 320 * 1024;
constexpr uint32_t WS_HEAD_BYTES = 2 * WS_BANK_BYTES;
constexpr uint32_t WS_CORE_BYTES = HEADS_PER_WORK * WS_HEAD_BYTES;
constexpr uint32_t WS_DK_STATE_RAW = 0;
constexpr uint32_t WS_DV_B = 32 * 1024;
constexpr uint32_t WS_ZV = 64 * 1024;
constexpr uint32_t WS_DKGB_RAW = 80 * 1024;
constexpr uint32_t WS_ZW = 112 * 1024;
constexpr uint32_t WS_KE = 144 * 1024;
constexpr uint32_t WS_ZB = 160 * 1024;
// Stage 5 replaces consumed raw slots; both generations still fit 256 KiB.
constexpr uint32_t WS_DK_BASE = 0;
constexpr uint32_t WS_DQ_BASE = 32 * 1024;
constexpr uint32_t WS_DG_BASE = 168 * 1024;
constexpr uint32_t WS_DB_BASE = 200 * 1024;
constexpr uint32_t WS_RH = 208 * 1024;
constexpr uint32_t WS_DAKK_RAW = 216 * 1024;

// Stage 7 reuses dead raw slots after Stage 5 and Stage 6. Bases remain intact.
constexpr uint32_t WS_DAQK_LOWER = 64 * 1024;
constexpr uint32_t WS_DAKK_LOWER = 80 * 1024;
constexpr uint32_t WS_KNEG = 96 * 1024;
constexpr uint32_t WS_QPOS = 128 * 1024;
constexpr uint32_t WS_BKPOS = 224 * 1024;

// Stage 8 has independent L1 lanes, leaving both generations of akk intact.
constexpr uint32_t L1_STAGE8 = 384 * 1024;
constexpr uint32_t L1_STAGE8_HEAD_BYTES = 64 * 1024;
constexpr uint32_t WS_DQ_LOCAL = 256 * 1024;
constexpr uint32_t WS_LEFT = 288 * 1024;
constexpr uint16_t BAND_READY = 1;
constexpr uint16_t BAND_LOADED = 2;
constexpr uint16_t S8_READY = 3;
constexpr uint16_t S8_SECOND_READY = 14;

constexpr uint16_t KE_READY = 0;
constexpr uint16_t WORK_OPEN = 8;

// Host-prepared metadata: no sequence search or tail inference in the hot loop.
struct ChunkInfo {
    int64_t sequence;
    int64_t localChunk;
    int64_t tokenStart;
    int64_t validRows;
};

// Private development entry; the public A5 tiling/ABI stays unchanged.
struct Stage01TilingData {
    int64_t batch;
    int64_t heads;
    int64_t tokens;
    int64_t chunks;
    uint32_t usedCoreNum;
    uint32_t isVariable;
};

// Optional test snapshots, separate from the reusable production workspace.
// Each (head, chunk) owns a complete record, including padding and tail rows.
constexpr uint32_t CAPTURE_BYTES = 816 * 1024;
constexpr uint32_t CAPTURE_DK_STATE_RAW = 0;
constexpr uint32_t CAPTURE_DV_B = 32 * 1024;
constexpr uint32_t CAPTURE_ZV = 64 * 1024;
constexpr uint32_t CAPTURE_DW = 80 * 1024;
constexpr uint32_t CAPTURE_EXP2_GK = 96 * 1024;
constexpr uint32_t CAPTURE_KE = 128 * 1024;
constexpr uint32_t CAPTURE_RH = 144 * 1024;
constexpr uint32_t CAPTURE_GK_LAST = 144 * 1024 + 512;

constexpr uint32_t CAPTURE_DKGB_RAW = 160 * 1024;
constexpr uint32_t CAPTURE_ZW = 192 * 1024;
constexpr uint32_t CAPTURE_ZB = 208 * 1024;
constexpr uint32_t CAPTURE_DB_V = 216 * 1024;
constexpr uint32_t CAPTURE_TZA = 224 * 1024;
constexpr uint32_t CAPTURE_TZA_CHECK = 232 * 1024;
constexpr uint32_t CAPTURE_DQ_BASE = 256 * 1024;
constexpr uint32_t CAPTURE_DK_BASE = 288 * 1024;
constexpr uint32_t CAPTURE_DG_BASE = 320 * 1024;
constexpr uint32_t CAPTURE_DB_BASE = 352 * 1024;
constexpr uint32_t CAPTURE_DAKK_RAW = 360 * 1024;

constexpr uint32_t CAPTURE_DAQK_LOWER = 384 * 1024;
constexpr uint32_t CAPTURE_DAKK_LOWER = 400 * 1024;
constexpr uint32_t CAPTURE_DAQK_DIAG = 416 * 1024;
constexpr uint32_t CAPTURE_BANDS = 424 * 1024;
constexpr uint32_t CAPTURE_BAND_BYTES = 116 * 1024;
constexpr uint32_t CAPTURE_DQ_LOCAL = 656 * 1024;
constexpr uint32_t CAPTURE_LEFT = 688 * 1024;
constexpr uint32_t CAPTURE_DQ = 720 * 1024;
constexpr uint32_t CAPTURE_DG = 752 * 1024;
constexpr uint32_t CAPTURE_RIGHT = 784 * 1024;
constexpr uint32_t WS_RIGHT = WS_KNEG; // Stage 8 has consumed kNeg before Stage 10 writes.
static_assert(CAPTURE_RIGHT + 32 * 1024 == CAPTURE_BYTES);
static_assert(CAPTURE_BANDS + 2 * CAPTURE_BAND_BYTES == CAPTURE_DQ_LOCAL);
static_assert(WS_LEFT + 32 * 1024 == WS_BANK_BYTES);

static_assert(WS_RH + HEAD_DIM * 4 <= WS_DAKK_RAW);
static_assert(WS_DAKK_RAW + CHUNK_SIZE * CHUNK_SIZE * 4 <= WS_BANK_BYTES);
static_assert(CAPTURE_DAKK_RAW + CHUNK_SIZE * CHUNK_SIZE * 4 <= CAPTURE_BYTES);
static_assert(sizeof(ChunkInfo) == 32);
static_assert(sizeof(Stage01TilingData) == 40);
static_assert(L1_V + CHUNK_SIZE * HEAD_DIM * 2 <= L1_HEAD_BYTES);
static_assert(L1_BYTES <= 512 * 1024);
static_assert(2 * L0_AB_SLOT_BYTES == 64 * 1024);
static_assert(2 * L0_C_SLOT_BYTES == 128 * 1024);
static_assert(UB_WORK + 13 * UB_WORK_SLOT_BYTES == UB_BYTES);
static_assert(UB_RH + HEAD_DIM * 4 == UB_GK_LAST);
static_assert(WS_DKGB_RAW + CHUNK_SIZE * HEAD_DIM * 4 == WS_ZW);
static_assert(WS_ZW + CHUNK_SIZE * CHUNK_SIZE * 4 <= WS_KE);
static_assert(CAPTURE_ZW + CHUNK_SIZE * CHUNK_SIZE * 4 <= CAPTURE_BYTES);
static_assert(2 * WS_BANK_BYTES == WS_HEAD_BYTES);
static_assert(WS_ZB + CHUNK_SIZE * CHUNK_SIZE * 2 <= WS_BANK_BYTES);
static_assert(CAPTURE_DB_V + CHUNK_SIZE * 4 <= CAPTURE_BYTES);
static_assert(WS_KE + CHUNK_SIZE * HEAD_DIM * 2 <= WS_HEAD_BYTES);
static_assert(CAPTURE_GK_LAST + HEAD_DIM * 4 <= CAPTURE_BYTES);

} // namespace KDA::Arch22

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH22_COMMON_H
