#include <iostream>
#ifdef LEGACY_KEYS
#include "legacy_tiling_key.h"
#define KEY(d,g,e,s,f,o,t,a,n) GET_TPL_TILING_KEY(d,128,g,e,s,f,o,t,a)
#else
#include "../../op_kernel/chunk_fwd_h_tiling_key.h"
#define KEY(d,g,e,s,f,o,t,a,n) GET_TPL_TILING_KEY(d,128,g,e,s,f,o,t,a,n)
#endif
int main() {
    using namespace GDN;
    std::cout << "family,dtype,mask,key\n";
    for (uint64_t dtype : {10ULL,30ULL}) for(uint64_t bits=0;bits<64;++bits)
        std::cout << "legacy," << dtype << ',' << bits << ','
                  << KEY(dtype,bits&1,(bits>>1)&1,(bits>>2)&1,(bits>>3)&1,
                         (bits>>4)&1,(bits>>5)&1,0,0) << '\n';
    for(uint64_t bits=0;bits<16;++bits)
        std::cout << "general,30," << bits << ','
                  << KEY(30,1,bits&1,(bits>>1)&1,(bits>>2)&1,(bits>>3)&1,0,1,0) << '\n';
#ifndef LEGACY_KEYS
    for(uint64_t bits=0;bits<4;++bits)
        std::cout << "p32_256,30," << bits << ','
                  << KEY(30,1,bits&1,1,0,(bits>>1)&1,0,0,1) << '\n';
#endif
}
