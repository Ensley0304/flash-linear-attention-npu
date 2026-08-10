#ifndef CHUNK_KDA_BWD_A_ARCH35_REGBASE_H
#define CHUNK_KDA_BWD_A_ARCH35_REGBASE_H

#include "kernel_utils/vector/regbase.hpp"

namespace KDA {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t kKdaBwdARegElements =
    AscendC::VECTOR_REG_WIDTH / sizeof(float);

static __simd_vf__ inline void KdaBwdARegbaseScale(
    __ubuf__ float *data, float scale, uint16_t count)
{
    RegTensor<float> value;
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count;
         offset += kKdaBwdARegElements) {
        MaskReg mask = UpdateMask<float>(remaining);
        DataCopy(value, data + offset);
        Muls(value, value, scale, mask);
        DataCopy(data + offset, value, mask);
    }
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_A_ARCH35_REGBASE_H
