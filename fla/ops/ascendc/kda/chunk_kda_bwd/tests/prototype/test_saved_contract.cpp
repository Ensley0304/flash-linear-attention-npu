// Host checks for the production allocation and reverse-scan descriptors.
#include <cassert>
#include <cstdint>
#include <iostream>
#include "../../op_kernel/arch22/fused32/saved_contract.h"
using namespace KDA::Arch22::Fused32::Saved;

int main() {
    uint32_t cases = 0;
    for (uint32_t k : {60U, 64U, 100U, 128U, 256U}) {
        for (uint32_t v : {k, 32U}) {
            if (v != k && k != 64) continue;
            Layout l{};
            assert(MakeLayout(k, v, 5, 65, 24, l));
            assert(l.blocks == 24 && l.workCount == 325);
            const auto &g = l.group;
            const auto &r = g.record;
            assert(r.Offset(Score::Middle) + r.scoreBytes == r.key);
            assert(r.Offset(Key::Dkg) + r.keyBytes == r.value);
            assert(r.Offset(Value::Drv) + r.valueBytes == r.state);
            assert(r.Offset(State::TermQ) + r.stateBytes == r.small);
            assert(g.Record(2) + r.bytes == g.inputs);
            assert(g.Input(1) + INPUT_BANK_BYTES == g.results);
            assert(g.Result(1) + RESULT_BANK_BYTES == g.exchange);
            assert(g.exchange + 32 * CHUNK_ROWS <= g.bytes);
            assert(l.partial == g.bytes * l.blocks);
            assert(l.partial + l.workCount * l.partialStride == l.bytes);
            for (uint32_t seq = 0; seq < 5; ++seq)
                for (uint32_t head = 0; head < 65; ++head) {
                    const uint64_t offset = l.partial + (uint64_t(seq) * 65 + head) * l.partialStride;
                    assert(offset % ALIGNMENT == 0 && offset + 4 * (k + 1) <= l.bytes);
                }
            ++cases;
        }
    }
    Layout untouched{}; untouched.bytes = 17;
    assert(!MakeLayout(256, 256, UINT64_MAX, 2, 24, untouched));
    assert(!MakeLayout(256, 256, UINT64_MAX / 2, 1, 24, untouched));
    assert(!MakeLayout(64, 128, 1, 1, 24, untouched));
    assert(!MakeLayout(64, 64, 1, 1, 0, untouched));
    assert(untouched.bytes == 17);
    for (uint32_t length = 0; length <= 4097; ++length) {
        const uint32_t count = ChunkCount(length);
        int64_t owner[3] = {-1, -1, -1};
        uint32_t visited = 0;
        if (count) owner[0] = 0;
        for (uint32_t scan = 0; scan < count; ++scan) {
            const Chunk c = Resolve(113, length, 71, scan);
            const Step s = ResolveStep(scan, count);
            assert(c.scan == scan && c.record == scan % 3);
            assert(c.physical == count - scan - 1);
            assert(c.globalChunk == 71 + c.physical);
            assert(c.token == 113 + uint64_t(c.physical) * 32);
            assert(c.rows && c.rows <= 32 && c.token + c.rows <= 113 + length);
            assert(owner[c.record] == scan);
            if (s.prepare != INVALID_SCAN) {
                assert(owner[s.prepare % 3] == -1);
                owner[s.prepare % 3] = s.prepare;
            }
            if (s.finalize != INVALID_SCAN) {
                assert(owner[s.finalize % 3] == s.finalize);
                owner[s.finalize % 3] = -1;
            }
            visited += c.rows;
        }
        if (count) owner[(count - 1) % 3] = -1;
        assert(visited == length && owner[0] == -1 && owner[1] == -1 && owner[2] == -1);
        assert(Resolve(0, length, 0, count).scan == INVALID_SCAN);
    }
    assert(ChunkCount(UINT32_MAX) == 134217728);
    assert(Resolve(0, UINT32_MAX, 0, 0).rows == 31);
    assert(ReductionPackets(256) == 2);
    assert(InputReady(2) == 0 && ResultReady(1) == 5);
    std::cout << "PASS: " << cases << " layouts, overflow, 4098 reverse schedules\n";
}
