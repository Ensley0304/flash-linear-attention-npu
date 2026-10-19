/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include <acl/acl.h>
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "stage0.h"
#include "stage1.h"

extern "C" int rtGetC2cCtrlAddr(uint64_t *, uint32_t *);

// Prefix development entry only. The complete operator will use MIX_AIC_1_2.
__global__ __aicore__ void FinalizeStage0(GM_ADDR v, GM_ADDR akk, GM_ADDR vn, GM_ADDR h, GM_ADDR dh,
                                       GM_ADDR ds, GM_ADDR workspace, GM_ADDR dw, uint32_t rows)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;
    KdaFinalizeCatlass::CubeStage0 cube;
    cube.Init(pipe);
    cube.Run(v, akk, vn, h, dh, ds, workspace, dw, rows);
    cube.End();
}

__global__ __aicore__ void FinalizeStage01(GM_ADDR v, GM_ADDR akk, GM_ADDR vn, GM_ADDR h, GM_ADDR dh,
                                        GM_ADDR ds, GM_ADDR workspace, GM_ADDR dw, GM_ADDR vectorCapture,
                                        KdaFinalizeCatlass::VectorInputs input, uint32_t rows, float lowerBound,
                                        uint64_t syncBase, uint32_t prefix, float scale, bool norm, uint32_t betaMode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSyncBaseAddr(syncBase);
    if ASCEND_IS_AIC {
        AscendC::TPipe pipe;
        KdaFinalizeCatlass::CubeStage0 cube;
        cube.Init(pipe);
        cube.Run(v, akk, vn, h, dh, ds, workspace, dw, rows);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(0);
        AscendC::CrossCoreWaitFlag(1);
        if (prefix >= 2) {
            cube.Stage2(workspace);
        }
        if (prefix >= 3) {
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(2);
            AscendC::CrossCoreWaitFlag(3);
        }
        if (prefix >= 4) {
            cube.Stage4(workspace, vectorCapture + 115968);
        }
        if (prefix >= 6) {
            cube.Stage6(workspace);
        }
        if (prefix >= 7) {
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(4);
            AscendC::CrossCoreWaitFlag(5);
            if (prefix >= 8) {
                cube.Stage8(workspace, 1);
                if (prefix >= 9) {
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(7);
                }
            }
            AscendC::CrossCoreWaitFlag(6);
            if (prefix >= 8) {
                cube.Stage8(workspace, 0);
                if (prefix >= 9) {
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(8);
                }
            }
        }
        cube.End();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        KdaFinalizeCatlass::VectorStage1 vector;
        vector.Init(input, pipe);
        vector.Run(rows, lowerBound, workspace, vectorCapture);
        AscendC::CrossCoreWaitFlag(0);
        if (prefix >= 3) {
            AscendC::CrossCoreWaitFlag(2);
            vector.Stage3(rows, workspace, vectorCapture);
        }
        if (prefix >= 5) {
            vector.Stage5(scale, workspace, vectorCapture);
        }
        if (prefix >= 7) {
            AscendC::CrossCoreWaitFlag(4);
            vector.Stage7(rows, workspace, vectorCapture);
        }
        if (prefix >= 9) {
            AscendC::CrossCoreWaitFlag(7);
            if (AscendC::GetSubBlockIdx() == 0) {
                AscendC::CrossCoreWaitFlag(8);
            }
            vector.Stage9(rows, norm, betaMode, workspace, vectorCapture);
            if (AscendC::GetSubBlockIdx() == 1) {
                AscendC::CrossCoreWaitFlag(8);
            }
        }
        vector.End();
    }
}

void Check(aclError error)
{
    if (error != ACL_SUCCESS) {
        throw std::runtime_error("ACL error " + std::to_string(error));
    }
}

std::vector<uint8_t> Read(const std::string &path, size_t bytes)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(bytes)) {
        throw std::runtime_error("Unexpected input size: " + path);
    }
    input.seekg(0);
    std::vector<uint8_t> data(bytes);
    if (!input.read(reinterpret_cast<char *>(data.data()), bytes)) {
        throw std::runtime_error("Cannot read " + path);
    }
    return data;
}

int main(int argc, char **argv)
{
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: finalize_stage0 CASE_DIR ROWS DEVICE [PREFIX=0..9]\n";
        return 2;
    }
    const std::string directory = argv[1];
    const uint32_t rows = std::stoul(argv[2]);
    const int device = std::stoi(argv[3]);
    const unsigned prefix = argc == 5 ? std::stoul(argv[4]) : 0;
    if (rows < 1 || rows > 64 || prefix > 9) {
        return 2;
    }
    aclrtStream stream = nullptr;
    enum Buffer { V, AKK, VN, H, DH, DS, Q, K, G, RAW, ALOG, BIAS, BETA, QR, KR, DQR, DAQ, WORKSPACE, DW, VCAP, COUNT };
    std::array<void *, COUNT> buffers{};
    bool initialized = false;
    bool selected = false;
    int status = 0;
    try {
        Check(aclInit(nullptr));
        initialized = true;
        Check(aclrtSetDevice(device));
        selected = true;
        Check(aclrtCreateStream(&stream));
        const char *names[] = {"v", "akk", "v_new", "h", "dh", "dv_scan", "q", "k", "gk", "raw_g",
                               "a_log", "dt_bias", "beta", "q_rstd", "k_rstd", "dq_raw", "dAqk"};
        const size_t bytes[] = {16384, 8192, 16384, 32768, 32768, 16384, 16384, 16384, 32768,
                               32768, 4, 512, 256, 256, 256, 32768, 16384, 524288, 16384, 322848};
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (prefix == 0 && ((i >= Q && i <= DAQ) || i == VCAP)) {
                continue;
            }
            Check(aclrtMalloc(&buffers[i], bytes[i], ACL_MEM_MALLOC_HUGE_FIRST));
            if (i < WORKSPACE) {
                auto data = Read(directory + "/" + names[i] + ".bin", bytes[i]);
                Check(aclrtMemcpy(buffers[i], bytes[i], data.data(), bytes[i], ACL_MEMCPY_HOST_TO_DEVICE));
            } else {
                Check(aclrtMemset(buffers[i], bytes[i], 0xA5, bytes[i]));
            }
        }
        auto address = [&](Buffer id) { return static_cast<uint8_t *>(buffers[id]); };
        if (prefix == 0) {
            FinalizeStage0<<<1, nullptr, stream>>>(address(V), address(AKK), address(VN), address(H), address(DH),
                                                 address(DS), address(WORKSPACE), address(DW), rows);
        } else {
            uint64_t syncBase = 0;
            uint32_t syncBytes = 0;
            if (rtGetC2cCtrlAddr(&syncBase, &syncBytes) != 0 || syncBase == 0) {
                throw std::runtime_error("Cannot query cross-core synchronization address");
            }
            KdaFinalizeCatlass::VectorInputs input{
                reinterpret_cast<uint64_t>(buffers[Q]), reinterpret_cast<uint64_t>(buffers[K]),
                reinterpret_cast<uint64_t>(buffers[G]), reinterpret_cast<uint64_t>(buffers[RAW]),
                reinterpret_cast<uint64_t>(buffers[ALOG]), reinterpret_cast<uint64_t>(buffers[BIAS]),
                reinterpret_cast<uint64_t>(buffers[BETA]), reinterpret_cast<uint64_t>(buffers[QR]),
                reinterpret_cast<uint64_t>(buffers[KR]), reinterpret_cast<uint64_t>(buffers[H]),
                reinterpret_cast<uint64_t>(buffers[DH]), reinterpret_cast<uint64_t>(buffers[V]),
                reinterpret_cast<uint64_t>(buffers[DQR]), reinterpret_cast<uint64_t>(buffers[DAQ])};
            struct Attributes { float scale, lowerBound; uint32_t norm, betaMode; } attributes{};
            auto attrBytes = Read(directory + "/attributes.bin", sizeof(attributes));
            std::copy(attrBytes.begin(), attrBytes.end(), reinterpret_cast<uint8_t *>(&attributes));
            FinalizeStage01<<<1, nullptr, stream>>>(address(V), address(AKK), address(VN), address(H), address(DH),
                                                  address(DS), address(WORKSPACE), address(DW), address(VCAP),
                                                  input, rows, attributes.lowerBound, syncBase, prefix,
                                                  attributes.scale, attributes.norm != 0, attributes.betaMode);
        }
        Check(aclrtSynchronizeStream(stream));
        const char *outputs[] = {"s0_state_raw", "s0_dv_b", "s0_zv", "s0_dw", "s1_ke", "s1_exp", "s1_rh",
                                 "s1_g_last", "s1_gate_factor", "s1_raw_x", "s2_dkg", "s2_zw",
                                 "s3_zb", "s3_dv", "s3_db_v", "s4_tza", "s5_dq_base", "s5_dk_base",
                                 "s5_dg_base", "s5_db_base", "s6_dakk_raw", "s7_aq", "s7_ak",
                                 "s7_kn1", "s7_qp1", "s7_bp1", "s7_kn0", "s7_qp0", "s7_bp0"};
        const size_t offsets[] = {0, 32768, 65536, 0, 81920, 0, 32768, 33280, 33792, 66560,
                                  98304, 131072, 147456, 99328, 115712, 115968,
                                  124160, 156928, 189696, 222464, 163840, 65536, 163840,
                                  212992, 245760, 278528, 311296, 344064, 376832};
        const size_t sizes[] = {32768, 32768, 16384, 16384, 16384, 32768, 512, 512, 32768, 32768,
                                32768, 16384, 8192, 16384, 256, 8192, 32768, 32768, 32768, 256, 16384,
                                16384, 16384, 32768, 32768, 32768, 32768, 32768, 32768};
        const unsigned outputCount[] = {4, 10, 12, 15, 16, 20, 21, 29, 29, 29};
        for (unsigned i = 0; i < outputCount[prefix]; ++i) {
            std::vector<uint8_t> data(sizes[i]);
            const bool overwritten = prefix >= 7 && (i == 2 || i == 20);
            const Buffer sourceId = i == 3 ? DW : (overwritten || (i >= 5 && i < 10) || (i >= 13 && i < 20) ? VCAP : WORKSPACE);
            const size_t offset = overwritten ? (i == 2 ? 240128 : 223744) : offsets[i];
            auto *source = address(sourceId) + offset;
            Check(aclrtMemcpy(data.data(), data.size(), source, data.size(), ACL_MEMCPY_DEVICE_TO_HOST));
            std::ofstream output(directory + "/" + outputs[i] + ".bin", std::ios::binary);
            output.write(reinterpret_cast<const char *>(data.data()), data.size());
            if (!output) {
                throw std::runtime_error("Cannot write output");
            }
        }
        if (prefix >= 8) {
            const char *localNames[] = {"s8_local_q", "s8_left", "s8_right"};
            for (uint32_t item = 0; item < 3; ++item) {
                std::vector<uint8_t> data(32768);
                for (uint32_t band = 0; band < 2; ++band) {
                    const size_t offset = (400 + (1 - band) * 48 + item * 16) * 1024;
                    Check(aclrtMemcpy(data.data() + band * 16384, 16384, address(WORKSPACE) + offset,
                                      16384, ACL_MEMCPY_DEVICE_TO_HOST));
                }
                std::ofstream output(directory + "/" + localNames[item] + ".bin", std::ios::binary);
                output.write(reinterpret_cast<const char *>(data.data()), data.size());
                if (!output) {
                    throw std::runtime_error("Cannot write local GEMM output");
                }
            }
        }
        if (prefix >= 9) {
            const char *finalNames[] = {"dq", "dk", "dv", "dg", "db", "da", "dbias"};
            const size_t finalOffsets[] = {256512, 272896, 99328, 289280, 322048, 322304, 322336};
            const size_t finalSizes[] = {16384, 16384, 16384, 32768, 256, 4, 512};
            for (uint32_t i = 0; i < 7; ++i) {
                std::vector<uint8_t> data(finalSizes[i]);
                Check(aclrtMemcpy(data.data(), data.size(), address(VCAP) + finalOffsets[i], data.size(),
                                  ACL_MEMCPY_DEVICE_TO_HOST));
                std::ofstream output(directory + "/final_" + finalNames[i] + ".bin", std::ios::binary);
                output.write(reinterpret_cast<const char *>(data.data()), data.size());
                if (!output) {
                    throw std::runtime_error("Cannot write final output");
                }
            }
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    // Release every allocation, including partially initialized failure paths.
    for (void *buffer : buffers) {
        if (buffer && aclrtFree(buffer) != ACL_SUCCESS) {
            status = 1;
        }
    }
    if (stream && aclrtDestroyStream(stream) != ACL_SUCCESS) {
        status = 1;
    }
    if (selected && aclrtResetDevice(device) != ACL_SUCCESS) {
        status = 1;
    }
    if (initialized && aclFinalize() != ACL_SUCCESS) {
        status = 1;
    }
    return status;
}
