/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "finalize.h"

extern "C" int rtGetC2cCtrlAddr(uint64_t *, uint32_t *);

void Check(aclError status)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error("ACL error " + std::to_string(status));
    }
}

uint64_t Multiply(uint64_t a, uint64_t b)
{
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) {
        throw std::runtime_error("Tensor size overflow");
    }
    return a * b;
}

std::vector<uint8_t> Read(const std::string &path, uint64_t bytes)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(bytes)) {
        throw std::runtime_error("Unexpected input size: " + path);
    }
    std::vector<uint8_t> data(bytes);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(data.data()), bytes)) {
        throw std::runtime_error("Cannot read " + path);
    }
    return data;
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: finalize CASE_DIR DEVICE [CORES=20] [WARMUP=0] [REPEATS=1]\n";
        return 2;
    }
    enum Buffer { Q, K, V, VN, DS, H, DH, AKK, G, RAW, DQR, DAQ, BETA, ALOG, BIAS, QR, KR,
                  META, RELAY, PARTIAL, DQ, DK, DV, DB, DG, DA, DBIAS, COUNT };
    std::array<void *, COUNT> buffers{};
    aclrtStream stream = nullptr;
    aclrtEvent startEvent = nullptr, stopEvent = nullptr;
    bool initialized = false, selected = false;
    int device = 0, status = 0;
    try {
        const std::string directory = argv[1];
        device = std::stoi(argv[2]);
        const uint32_t requestedCores = argc > 3 ? std::stoul(argv[3]) : 20;
        const uint32_t warmup = argc > 4 ? std::stoul(argv[4]) : 0;
        const uint32_t repeats = argc > 5 ? std::stoul(argv[5]) : 1;
        KdaFinalizeCatlass::Configuration cfg{};
        auto configBytes = Read(directory + "/configuration.bin", sizeof(cfg));
        std::memcpy(&cfg, configBytes.data(), sizeof(cfg));
        if (cfg.batches == 0 || cfg.heads == 0 || cfg.tokens == 0 || cfg.chunks == 0 || cfg.variable > 1 ||
            cfg.tokenMajor > 1 || cfg.norm > 1 || cfg.betaMode > 2 || !std::isfinite(cfg.scale) ||
            !std::isfinite(cfg.lowerBound) || cfg.lowerBound < -5 || cfg.lowerBound >= 0 ||
            cfg.heads > std::numeric_limits<uint32_t>::max() / 544 || requestedCores == 0 ||
            requestedCores > 20 || repeats == 0) {
            throw std::runtime_error("Invalid configuration");
        }
        auto metaBytes = Read(directory + "/metadata.bin", Multiply(cfg.chunks, 32));
        std::vector<uint64_t> metadata(cfg.chunks * 4);
        std::memcpy(metadata.data(), metaBytes.data(), metaBytes.size());
        std::vector<uint64_t> boundaries(cfg.batches + 1);
        if (cfg.variable) {
            auto cu = Read(directory + "/cu_seqlens.bin", Multiply(cfg.batches + 1, 8));
            std::memcpy(boundaries.data(), cu.data(), cu.size());
            if (boundaries.front() != 0 || boundaries.back() != cfg.tokens ||
                !std::is_sorted(boundaries.begin(), boundaries.end())) {
                throw std::runtime_error("Invalid cu_seqlens");
            }
        }
        uint64_t chunk = 0;
        bool fullChunks = true;
        for (uint64_t seq = 0; seq < cfg.batches; ++seq) {
            const uint64_t begin = cfg.variable ? boundaries[seq] : 0;
            const uint64_t end = cfg.variable ? boundaries[seq + 1] : cfg.tokens;
            for (uint64_t first = begin; first < end; first += 64, ++chunk) {
                if (chunk >= cfg.chunks || metadata[chunk * 4] != seq || metadata[chunk * 4 + 1] != first ||
                    metadata[chunk * 4 + 2] != std::min(uint64_t(64), end - first) || metadata[chunk * 4 + 3] != chunk) {
                    throw std::runtime_error("Invalid chunk metadata");
                }
                fullChunks = fullChunks && metadata[chunk * 4 + 2] == 64;
            }
        }
        if (chunk != cfg.chunks) {
            throw std::runtime_error("Chunk count mismatch");
        }
        const uint64_t tokens = Multiply(Multiply(cfg.variable ? 1 : cfg.batches, cfg.heads), cfg.tokens);
        const uint64_t works = Multiply(cfg.chunks, cfg.heads);
        const uint32_t cores = std::min(uint64_t(requestedCores), works);
        const uint64_t vectorBytes = Multiply(tokens, 256), stateBytes = Multiply(works, 32768);
        const uint64_t floatBytes = Multiply(tokens, 512), scalarBytes = Multiply(tokens, 4);
        const std::array<uint64_t, COUNT> sizes{
            vectorBytes, vectorBytes, vectorBytes, vectorBytes, vectorBytes, stateBytes, stateBytes,
            Multiply(tokens, 128), floatBytes, floatBytes, floatBytes, vectorBytes, scalarBytes,
            Multiply(cfg.heads, 4), Multiply(cfg.heads, 512), scalarBytes, scalarBytes,
            Multiply(cfg.chunks, 32), uint64_t(cores) * 512 * 1024, Multiply(works, 544),
            vectorBytes, vectorBytes, vectorBytes, scalarBytes, floatBytes, Multiply(cfg.heads, 4), Multiply(cfg.heads, 512)};
        const char *names[] = {"q", "k", "v", "v_new", "dv_scan", "h", "dh", "akk", "gk", "raw_g",
                               "dq_raw", "dAqk", "beta", "a_log", "dt_bias", "q_rstd", "k_rstd", "metadata"};
        Check(aclInit(nullptr));
        initialized = true;
        Check(aclrtSetDevice(device));
        selected = true;
        Check(aclrtCreateStream(&stream));
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (!cfg.norm && (i == QR || i == KR)) {
                continue;
            }
            Check(aclrtMalloc(&buffers[i], sizes[i], ACL_MEM_MALLOC_HUGE_FIRST));
            if (i <= META) {
                auto data = i == META ? metaBytes : Read(directory + "/" + names[i] + ".bin", sizes[i]);
                Check(aclrtMemcpy(buffers[i], sizes[i], data.data(), data.size(), ACL_MEMCPY_HOST_TO_DEVICE));
            } else {
                Check(aclrtMemset(buffers[i], sizes[i], 0xA5, sizes[i]));
            }
        }
        auto address = [&](Buffer id) { return reinterpret_cast<uint64_t>(buffers[id]); };
        KdaFinalizeCatlass::Arguments args{};
        args.input = {address(Q), address(K), address(G), address(RAW), address(ALOG), address(BIAS),
                      address(BETA), address(QR), address(KR), address(H), address(DH), address(V), address(DQR), address(DAQ)};
        args.output = {address(DQ), address(DK), address(DV), address(DG), address(DB), address(PARTIAL), 0, 0};
        args.akk = address(AKK);
        args.vNew = address(VN);
        args.dvScan = address(DS);
        args.metadata = address(META);
        args.relay = address(RELAY);
        args.partial = address(PARTIAL);
        args.config = cfg;
        uint32_t syncBytes = 0;
        if (rtGetC2cCtrlAddr(&args.syncBase, &syncBytes) != 0 || args.syncBase == 0) {
            throw std::runtime_error("Cannot query cross-core synchronization address");
        }
        Check(aclrtCreateEvent(&startEvent));
        Check(aclrtCreateEvent(&stopEvent));
        for (uint64_t iteration = 0; iteration < uint64_t(warmup) + repeats; ++iteration) {
            if (iteration == warmup) {
                Check(aclrtRecordEvent(startEvent, stream));
            }
            if (fullChunks) {
                FinalizeCatlass<true><<<cores, nullptr, stream>>>(args);
            } else {
                FinalizeCatlass<false><<<cores, nullptr, stream>>>(args);
            }
            FinalizeCatlassReduce<<<cfg.heads, nullptr, stream>>>(
                static_cast<uint8_t *>(buffers[PARTIAL]), static_cast<uint8_t *>(buffers[DA]),
                static_cast<uint8_t *>(buffers[DBIAS]), cfg.heads, cfg.chunks);
        }
        Check(aclrtRecordEvent(stopEvent, stream));
        Check(aclrtSynchronizeStream(stream));
        float milliseconds = 0;
        Check(aclrtEventElapsedTime(&milliseconds, startEvent, stopEvent));
        std::cout << "combined_event_ms=" << milliseconds / repeats << ", cores=" << cores
                  << ", full_chunks=" << fullChunks << '\n';
        const char *outputs[] = {"dq", "dk", "dv", "db", "dg", "da_log", "ddt_bias"};
        for (size_t i = DQ; i < COUNT; ++i) {
            std::vector<uint8_t> data(sizes[i]);
            Check(aclrtMemcpy(data.data(), data.size(), buffers[i], sizes[i], ACL_MEMCPY_DEVICE_TO_HOST));
            std::ofstream output(directory + "/output_" + outputs[i - DQ] + ".bin", std::ios::binary);
            if (!output.write(reinterpret_cast<const char *>(data.data()), data.size())) {
                throw std::runtime_error("Cannot write output");
            }
        }
        if (std::getenv("KDA_DUMP_RELAY") != nullptr) {
            std::vector<uint8_t> data(sizes[RELAY]);
            Check(aclrtMemcpy(data.data(), data.size(), buffers[RELAY], data.size(), ACL_MEMCPY_DEVICE_TO_HOST));
            std::ofstream output(directory + "/debug_relay.bin", std::ios::binary);
            if (!output.write(reinterpret_cast<const char *>(data.data()), data.size())) {
                throw std::runtime_error("Cannot write diagnostic relay");
            }
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    if (startEvent) aclrtDestroyEvent(startEvent);
    if (stopEvent) aclrtDestroyEvent(stopEvent);
    for (void *buffer : buffers) {
        if (buffer && aclrtFree(buffer) != ACL_SUCCESS) status = 1;
    }
    if (stream && aclrtDestroyStream(stream) != ACL_SUCCESS) status = 1;
    if (selected && aclrtResetDevice(device) != ACL_SUCCESS) status = 1;
    if (initialized && aclFinalize() != ACL_SUCCESS) status = 1;
    return status;
}
