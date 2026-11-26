// Checks actual device descriptors against independent dense products. This is
// a host test of addressing and formulas, not device precision acceptance.
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "../../op_kernel/arch22/fused32/saved_operations.h"
using namespace KDA::Arch22::Fused32::Saved;

static size_t At(Matrix m, uint32_t i, uint32_t j) {
    return m.offset / 4 + size_t(i) * m.rowStride + size_t(j) * m.colStride;
}
static void CheckBounds(Matrix m, uint32_t rows, uint32_t cols, size_t size) {
    assert(m.offset % 4 == 0);
    assert(rows && cols && At(m, rows - 1, cols - 1) < size);
}
int main() {
    uint32_t cases = 0;
    for (uint32_t k : {60U, 64U, 100U, 128U, 256U}) {
      for (uint32_t v : {k, 32U}) {
        if (v != k && k != 64) continue;
        Layout layout{};
        assert(MakeLayout(k, v, 1, 1, 1, layout));
        const auto &g = layout.group;
        for (bool vf : {false, true}) {
          for (uint32_t rows : {1U, 15U, 31U, 32U}) {
            for (uint32_t scan : {0U, 1U, 2U, 3U}) {
              // Terminal result deliberately outside every record. Dhu must
              // preserve the still-live Finalize record at (scan+1)%3.
              Matrix dest{g.bytes, vf ? 1U : g.record.paddedV,
                          vf ? g.record.paddedK : 1U};
              std::vector<float> memory((g.bytes + g.record.stateBytes) / 4, 0);
              for (uint32_t oi = 0; oi < 14; ++oi) {
                const auto op = static_cast<Operation>(oi);
                const auto p = Describe(op, g, scan, rows, k, v, vf, dest);
                CheckBounds(p.a, p.m, p.k, memory.size());
                CheckBounds(p.b, p.k, p.n, memory.size());
                CheckBounds(p.out, p.m, p.n, memory.size());
                if (op == Operation::DhuW) assert(p.out.offset == g.bytes);
                // Every output element belongs to exactly one tile, including
                // non-aligned dimensions and the 1-row packed tail.
                std::vector<unsigned> seen(size_t(p.m) * p.n, 0);
                for (uint32_t ti = 0; ti < OutputTiles(p); ++ti) {
                  const auto t = OutputTile(p, ti);
                  assert(t.paddedRows <= 64 && t.paddedCols <= 64);
                  assert(t.paddedRows % 16 == 0 && t.paddedCols % 16 == 0);
                  for (uint32_t i = 0; i < t.rows; ++i)
                    for (uint32_t j = 0; j < t.cols; ++j)
                      ++seen[size_t(t.row+i) * p.n + t.col+j];
                }
                for (auto n : seen) assert(n == 1);
              }
              // Physical K-first/V-first addressing must yield the same
              // PrepareQ = dO @ H^T and DhuV = kg @ Dnext.
              const auto pq = Describe(Operation::PrepareQ,g,scan,rows,k,v,vf,dest);
              const auto dv = Describe(Operation::DhuV,g,scan,rows,k,v,vf,dest);
              for (uint32_t i=0;i<rows;++i)
                for (uint32_t j=0;j<v;++j) memory[At(pq.a,i,j)] = float(i+1)*0.01f+float(j)*0.001f;
              for (uint32_t i=0;i<k;++i)
                for (uint32_t j=0;j<v;++j) {
                  memory[At(pq.b,j,i)] = float(i+1)*0.003f-float(j)*0.002f;
                  memory[At(dv.b,i,j)] = float(i+1)*0.004f+float(j)*0.001f;
                }
              for (uint32_t i=0;i<rows;++i)
                for (uint32_t j=0;j<k;++j) memory[At(dv.a,i,j)] = float(i+1)*0.002f+float(j)*0.003f;
              for (uint32_t i=0;i<rows;++i) {
                for (uint32_t j=0;j<k;++j) {
                  double actual=0,expected=0;
                  for (uint32_t z=0;z<v;++z) {
                    actual += double(memory[At(pq.a,i,z)])*memory[At(pq.b,z,j)];
                    expected += double(float(i+1)*0.01f+float(z)*0.001f)*(float(j+1)*0.003f-float(z)*0.002f);
                  }
                  assert(std::abs(actual-expected)<1e-6 * (1 + std::abs(expected)));
                }
                for (uint32_t j=0;j<v;++j) {
                  double actual=0,expected=0;
                  for (uint32_t z=0;z<k;++z) {
                    actual += double(memory[At(dv.a,i,z)])*memory[At(dv.b,z,j)];
                    expected += double(float(i+1)*0.002f+float(z)*0.003f)*(float(z+1)*0.004f+float(j)*0.001f);
                  }
                  assert(std::abs(actual-expected)<1e-6 * (1 + std::abs(expected)));
                }
              }
              ++cases;
            }
          }
        }
      }
    }
    std::cout << "PASS: " << cases << " shape/layout/record cases, 14 products, terminal state, tile coverage\n";
}
