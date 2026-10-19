# A2 环境预检

2026-10-19，服务器 `a2-server`：hostname=ubuntu，用户=root。独立工程位于 `/data/wys/kda-finalize-a2-cannbot-20261019`。计划使用空闲设备3，实际发射前再检查占用。

| 项目 | 固定值 / 实测结果 |
| --- | --- |
| 工作流 | CANNBot `61518bc950fac84c953abe19332e43c24d4c1985` |
| 仓库流程 | PR #833，`c53f7811c0330fbf947a7ea14b4785c6c6b401ab` |
| CATLASS | `0d78a73c192fc04741a4fd57c94080209424775c` |
| CANN 环境 | `/data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh` |
| 编译器 | 上述环境中的 bisheng；CMake 4.3.1 |
| SoC / 分支 | Ascend910B3；`CATLASS_ARCH=2201`，`Arch::AtlasA2` |
| 设备拓扑 | 20 AIC、40 AIV |
| Vector 模型 | MemBase |
| Cube→Vector / Vector→Cube | 经 GM relay；不采用 A5 直连接口 |

通过本工程 `test/query_platform.cpp` 调用已安装 CANN 平台接口，得到以下容量。规划使用实测值和 CATLASS 名义上限中的较小值，再扣除实际组件占用。

| 空间 | 平台返回 bytes | CATLASS 名义 bytes | 设计上限 bytes |
| --- | ---: | ---: | ---: |
| UB | 196352 | 196608 | 196352 |
| L1 | 524032 | 524288 | 524032 |
| L0A | 65536 | 65536 | 65536 |
| L0B | 65536 | 65536 | 65536 |
| L0C | 131072 | 131072 | 131072 |
| FixBuf | 2048 | 7168 | 2048 |

不能直接把 UB 192 KiB、L1 512 KiB 全部分给用户数据。新方案显式保留容量余量；FixBuf 不自行分配。双缓冲、跨阶段常驻、预取、临时数组和组件内部空间均计入峰值。

已核验 CATLASS `gemm/tile/tile_mmad.hpp` 的 `TileMmad`：支持显式传入 L0A/B/C 和 m/n/k，可由本算子管理驻留与同步。FP32 路径还需按实际 A/B 布局核对 `kDirectionAlign`，不得启用 HF32 代替原精度目标。

CPU 标杆运行关闭 `TORCH_DEVICE_BACKEND_AUTOLOAD`，避免纯 CPU 检查因自动加载 torch_npu 而依赖 CANN 动态库。设备测试则显式加载上述 CANN 环境。没有安装或修改系统组件。
