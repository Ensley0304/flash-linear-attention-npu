// Query installed CANN capabilities; this program does not launch a kernel.
#include <acl/acl.h>
#include <tiling/platform/platform_ascendc.h>

#include <iostream>

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS) {
        return 1;
    }
    const char *soc = aclrtGetSocName();
    if (soc == nullptr) {
        aclFinalize();
        return 2;
    }
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    if (platform == nullptr) {
        aclFinalize();
        return 3;
    }
    std::cout << "soc=" << soc << "\nAIC=" << platform->GetCoreNumAic()
              << "\nAIV=" << platform->GetCoreNumAiv() << '\n';
    using Mem = platform_ascendc::CoreMemType;
    const Mem memories[] = {Mem::UB, Mem::L1, Mem::L0_A, Mem::L0_B, Mem::L0_C, Mem::FB};
    const char *names[] = {"UB", "L1", "L0A", "L0B", "L0C", "FixBuf"};
    for (unsigned i = 0; i < 6; ++i) {
        uint64_t bytes = 0;
        platform->GetCoreMemSize(memories[i], bytes);
        std::cout << names[i] << '=' << bytes << '\n';
    }
    return aclFinalize() == ACL_SUCCESS ? 0 : 4;
}
