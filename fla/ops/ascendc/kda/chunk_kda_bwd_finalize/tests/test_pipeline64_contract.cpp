#include <cassert>
#include <iostream>
#include <set>
#include "../op_kernel/arch22/pipeline64/products.h"
using namespace KDA::Arch22::Pipeline64;
int main() {
    unsigned cases=0;
    for (uint32_t k:{60U,64U,100U,128U,256U}) for(uint32_t v:{k,32U}) {
      if(v!=k && k!=64) continue;
      for(uint32_t cg:{1U,2U,4U}) for(uint32_t heads:{1U,2U,5U,65U})
       for(uint64_t chunks:{0ULL,1ULL,2ULL,7ULL}) {
        Layout l{}; assert(MakeLayout(k,v,chunks,heads,cg,24,1024,l));
        assert(l.cg*l.depth<=4 && l.blocks<=24 && l.blocks>0);
        assert(l.Offset(Score::Key)+l.scoreBytes==l.keys);
        assert(l.Offset(Key::Band)+l.keyBytes==l.values);
        assert(l.Offset(Value::Do)+l.valueBytes==l.recordBytes);
        assert(l.Record(l.depth-1,cg-1)+l.recordBytes==l.inputs);
        assert(l.inputs+2*INPUT_BANK_BYTES==l.results);
        assert(l.results+2*RESULT_BANK_BYTES==l.exchange);
        assert(l.partial==l.groupBytes*l.blocks);
        std::set<uint64_t> visited;
        for(uint32_t core=0;core<l.blocks;++core) {
          uint64_t iteration=0;
          for(uint64_t w=core;w<l.workCount;w+=l.blocks,++iteration) {
            auto task=ResolveWork(w,iteration,heads,l);
            assert(task.chunk<chunks && task.generation==iteration%l.depth);
            for(uint32_t h=0;h<task.heads;++h) {
              assert(visited.insert(task.chunk*heads+task.firstHead+h).second);
              auto r=l.Record(task.generation,h);
              for(bool vf:{false,true}) for(uint32_t rows:{1U,63U,64U}) {
                Inputs x{{l.bytes, v,1},{l.bytes+4096,vf?1U:v,vf?k:1U},
                         {l.bytes+8192,vf?1U:v,vf?k:1U},
                         {l.bytes+12288,v,1},{l.bytes+16384,64,1}};
                for(uint32_t i=0;i<8;++i) {
                  auto p=Describe(static_cast<ProductKind>(i),l,r,rows,x);
                  assert(p.out.address>=r && p.out.address+4ULL*((p.m-1)*p.out.rowStride+(p.n-1)*p.out.colStride+1)<=r+l.recordBytes);
                }
                auto dw=Describe(ProductKind::Dw,l,r,rows,x);
                assert(dw.b.rowStride==(vf?k:1) && dw.b.colStride==(vf?1:v));
                auto dl=Describe(ProductKind::Dl,l,r,rows,x);
                for(uint32_t i=0;i<rows;++i) for(uint32_t j=0;j<rows;++j)
                  assert(ValidOutput(dl.epilogue,i,j)==(j<i));
                assert(Describe(ProductKind::Drv,l,r,rows,x).out.address!=r+l.Offset(Value::Rv));
              }
            }
          }
        }
        assert(visited.size()==chunks*heads);
        ++cases;
       }
    }
    Layout unchanged{}; unchanged.bytes=17;
    assert(!MakeLayout(256,256,UINT64_MAX,2,1,24,0,unchanged));
    assert(!MakeLayout(256,256,1,2,1,24,UINT64_MAX,unchanged));
    assert(!MakeLayout(64,128,1,2,1,24,0,unchanged));
    assert(!MakeLayout(64,64,1,2,3,24,0,unchanged));
    assert(unchanged.bytes==17);
    std::cout<<"PASS "<<cases<<" layouts, work coverage, FP32 matrix views, strict dL mask, overflow\n";
}
