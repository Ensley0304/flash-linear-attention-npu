// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
#include "saved_operations.h"

namespace KDA::Arch22::Fused32::Saved {
// Caller owns the UB bank until MTE3 has completed. Each slice is <= 16 KiB.
// Only valid bytes are read; tail rows and the complete physical padding are 0.
__aicore__ inline void LoadPacketSlice(AscendC::LocalTensor<float> dst,
                                      AscendC::GlobalTensor<float> source,
                                      const PacketSlice &s) {
    using namespace AscendC;
    Duplicate(dst, 0.0f, s.paddedRows * s.paddedCols);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    if (s.rows && s.cols) {
        const uint32_t alignedCols = (s.cols + 7) & ~7U;
        DataCopyExtParams cp{uint16_t(s.rows), s.cols * 4,
                             (s.sourceStride - s.cols) * 4,
                             (s.paddedCols - alignedCols) / 8, 0};
        DataCopyPadExtParams<float> pad{true, 0, uint8_t(alignedCols - s.cols), 0.0f};
        DataCopyPad(dst, source[s.offset / 4], cp, pad);
    }
}
} // namespace KDA::Arch22::Fused32::Saved
