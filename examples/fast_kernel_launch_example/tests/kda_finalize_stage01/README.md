# KDA Finalize A2/A3：Stage 0～10 开发验证

仅验证 Stage 0～10，不代表完整 Finalize 已支持 A2/A3。公开 A5 入口不变，开发入口保留原名。

## 构建和运行

先进入已有的 torch/torch_npu 环境并加载对应 CANN。仓库根目录执行：

```bash
export NPU_ARCH=ascend910b
export FAST_KERNEL_OP_NAME=kda_finalize_stage01
cd examples/fast_kernel_launch_example
python setup.py cmake_build

python tests/kda_finalize_stage01/check_stage01.py \
  --library build/_C.abi3.so --stage 0 --device 1 --cores 1 --suite tails
python tests/kda_finalize_stage01/check_stage01.py \
  --library build/_C.abi3.so --stage 1 --device 1 --cores 1 --suite tails
python tests/kda_finalize_stage01/check_stage01.py \
  --library build/_C.abi3.so --stage 1 --device 1 --cores 0 --suite tails
python tests/kda_finalize_stage01/check_stage01.py \
  --library build/_C.abi3.so --stage 2 --device 1 --cores 0 --suite model --repeat 1
python tests/kda_finalize_stage01/check_stage01.py \
  --library build/_C.abi3.so --stage 4 --device 1 --cores 0 --suite model --repeat 1
```

测试直接加载本次构建的动态库，无需安装 wheel 或覆盖 OPP。`--cores 1` 强制同核复用 workspace；`--cores 0` 使用平台可用 AIC。测试每次进入设备调用后设置 60 秒超时，超时进程以 124 退出。

Host 元数据和大小检查（仓库根目录，无需 NPU）：

```bash
g++ -std=c++17 -Wall -Wextra -Werror -I. \
  examples/fast_kernel_launch_example/tests/kda_finalize_stage01/test_tiling.cpp \
  -o /tmp/kda_finalize_stage01_tiling
/tmp/kda_finalize_stage01_tiling
```

## 范围

- Stage 0：dk_state_raw、dv_b、BF16 dW、zV。
- Stage 1：rH、FP32 exp2_gk、BF16 kE、gk_last。
- Stage 2：dKgb_raw=akkᵀ@dW、zW=dW@kEᵀ，BF16 输入、FP32 累加和输出。
- Stage 3：`Zb=tril((zV-zW)*beta[None,:],-1)`、`dv=beta[:,None]*dv_b`、`db_v=sum(v*dv_b,-1)`。Zb/dv 转 BF16，db_v 保持 FP32。
- 定长 B=1/2；H=3/96；尾长 1/16/31/32/33/63/64；变长含空序列；默认每例重复两次。
- 逐项对照 FP64/FP32 CPU 标杆，检查 NaN/Inf、capture 哨兵及真实 workspace 最后一代数据。
- trace 记录实际 work/core/owner/head/chunk、槽位使用次数、block 数、子核数、GM 偏移及握手进度；奇数 head 的空 owner 仍参与同步。

开发入口返回 `(capture, trace, workspace)`。当前 capture 每个 `(head,chunk)` 占 816 KiB，Stage 2 结果位于 160/192 KiB；真实 workspace 的对应偏移为 80/112 KiB。额外采数不用于性能评估。Stage 2 实际消费驻留 L1 的 dW，验证 L0C→L1 链路。

Stage 0/1 的容差不变。Stage 2 同时检查两层：以设备捕获的 BF16 dW/kE 做 FP64 矩阵乘，使用 `rtol=3e-4, atol=2e-6`；再与独立 CPU 全路径结果比较，传播既定 BF16 门槛 `E(x)=2e-5+8e-3|x|`：

```text
dKgb 误差预算 = |akk|ᵀ @ E(dW)
zW 误差预算   = E(dW) @ |kE|ᵀ + |dW| @ E(kE)ᵀ + E(dW) @ E(kE)ᵀ
```

预算外仍只允许上述 FP32 累加容差，所有结果必须有限。原因是 FP32 累加的微小差异可能使 BF16 舍入跨过相邻值，不能将该传播误差误判为后续 MMAD 错误。`--gate-mode flat` 将 gk 分布设为 [-2,0]，使所有 kE 行保持显著，补充默认累计 gate 场景。

原始输入保持 BNSD/NTD，h/dh 为 chunk-major。变长测试入口接收 CPU `cu_seqlens` 列表并生成完整 chunk 元数据；只接受连续 BF16 主数据、FP32 gk、K=V=128。

## Stage 3 入口与精度门禁

使用 `--stage 3` 执行四个阶段。私有入口末尾新增可选参数 `beta` 和 `dv_out`；Stage 3 必须提供，前三阶段的旧调用方式保持有效。beta 为前向有效值，当前入口要求连续 FP32 `[B,H,T]` / `[H,T]`；dv_out 为连续 BF16 BNSD/NTD，原地填入结果。测试包含 beta=0/1/2 及 sigmoid 饱和附近值。

capture 中 Zb/db_v 分别位于 208/216 KiB。逐元素检查所有 dv、Zb 和 db_v；Zb 必须严格下三角，尾行/列为零。局部标杆使用捕获的 FP32 zV/zW/dv_b；独立 CPU 全路径同时传播既定 Stage 2 误差预算，未修改前三阶段容差。Zb/dv 使用 BF16 门槛 `rtol=8e-3, atol=2e-5`，db_v 使用 FP32 门槛 `rtol=3e-4, atol=2e-6`。

四阶段开发调度将每 head 原有 512 KiB workspace 分为两个 256 KiB 槽，交替保存相邻 work；槽内偏移不变，Zb 位于 160 KiB。检查两槽的最后有效代、capture 保护区与 trace。`--workspace-only` 还检查性能库的全部 dv 输出及两槽 Zb，不依赖额外采数。

该跨 work 调度仅用于当前阶段入口。Stage 4 已增加两代 akk；后续 Stage 5/6 必须继续扩展驻留数据生命周期及握手。

## Stage 4 入口与精度门禁

`--stage 4` 新增 `Tza=Zb@akk.T`，BF16 输入、FP32 累加，结果直接转 BF16 NZ 驻留 L1。capture 的 Tza 位于 224 KiB；232 KiB 保存仅在采数版执行的 `Tza_L1@akk.T`，用于核对实际 L1 布局和同步。性能版不执行该验证乘法，也不导出 Tza。

Tza 同时检查实际捕获 Zb 的局部 FP64 标杆及独立 CPU 全路径；使用既定 BF16 门槛，传播 Stage 3 误差预算与 `|akk|.T` 的乘积。L1 消费结果使用 FP32 门槛 `rtol=3e-4, atol=2e-6`。全部 head/chunk/元素及尾行列均检查。`--workspace-only --stage 4` 检查性能版全部 dv 和保留的 GM 中间量；Tza 本身不在 GM，不能据此宣称性能版 Tza 已直接采数验证。

## Stage 5 入口与精度门禁

`--stage 5` 新增 FP32 `dq_base/dk_base/dg_base/db_base`。私有入口末尾增加可选 `q`（BF16）、`dq_raw`（FP32）和 `scale`；Stage 5 要求前两项连续且与 k 同 shape/device，scale 转 FP32 后须有限。此前阶段的调用保持兼容。

capture 中四个结果分别位于 256/288/320/352 KiB。所有 head/chunk/元素对照局部 FP64 标杆及独立 CPU 全路径；局部使用 `rtol=3e-4, atol=2e-6`，全路径仅增加既定上游误差经门控、beta 和归约传播的预算。检查有限性、尾行零值、末行 gate 补偿和保护区，不改变 Stage 0～4 的门槛。

生产 workspace 中 dq/dk 复用已消费的 dv_b/dk_state_raw；dg/db 位于 168/200 KiB，rH 按代保存在 208 KiB。`--workspace-only --stage 5` 检查全部 dv 和各核最后两代四个 Stage 5 结果，不能据此宣称未保留的中间代均已直接采数。`--scale 0` 与 `--scale -0.5` 补充缩放边界检查。

## Stage 6 入口与精度门禁

`--stage 6` 增加 `dAkk_raw=akk.T@Tza`，不增加调用参数。实际消费 Stage 4 的 BF16 NZ L1 数据，FP32 累加并输出。capture `[360,376) KiB` 保存每个 head/chunk 的完整结果，workspace `[216,232) KiB` 保存每代结果；原有偏移和 workspace 总量不变。

局部 FP64 GEMM 使用捕获的 BF16 Tza，门槛仍为 `rtol=3e-4, atol=2e-6`。独立 CPU 全路径将既定 Stage 4 Tza 误差预算左乘 `|akk|.T`，不放宽原阶段容差；另检查有限值、尾行/列为零、capture/workspace 一致性及保护区。`--workspace-only --stage 6` 直接检查性能库各核最后两代 dAkk_raw，并保留前序阶段检查。

## Stage 7 入口与精度门禁

`--stage 7` 要求末尾参数 `dAqk`：与 akk 同 shape/device 的连续 FP32 输入。两个严格下三角矩阵乘以 ±65536 后拆为 BF16 高位和残差；对角线保留原始 FP32。随后按 32 行带逆序，从原始 gk 计算居中指数，生成 kNeg/qPos/bkPos。无效行不进入 Exp，直接写零。

Stage 7 写入 capture 的前 656 KiB：384/400 KiB 保存两个 strict-lower 高低位，416 KiB 保存对角线；两带从 424 KiB 起各占 116 KiB，依次保存三个操作数、32 行指数和 center。严格下三角拆分与实际 Stage 6 输入逐位一致；居中操作数对照 FP64 公式，检查高低位重建误差、残差幅度、有限值及零填充。新增 `--gate-mode extreme` 覆盖每 token 自然对数 gate=-5。

当前 workspace 为每 AIC 1280 KiB、每 head 两个 320 KiB 槽。Stage 7 消费 raw 后复用其地址，保留四个 Stage 5 base；性能库检查最后两代 band 0 的 Cube 操作数。对角线、center、带内指数保留 UB，仅采数版导出。单独截止 Stage 7 时没有 Cube 消费者；执行 Stage 8/10 时逐带握手后才能覆盖 GM。此处不代表完整 Finalize 精度验收。

## 双缓冲回归与性能

`--suite model --repeat 1` 完整检查 B=1/H=96/T=8192 的定长及变长输入。变长长度为 `[0,1,31,64,65,4096,3935,0]`；全部中间结果对照 CPU 标杆，Stage 0/1 容差不变，Stage 2 使用上述两层门禁。

性能版使用编译开关关闭 capture/trace、L1 验证乘法，并缩小对应返回占位张量。以下命令测 Stage 0～8 开发入口，不代表完整 Finalize 性能，在 `examples/fast_kernel_launch_example` 下运行：

```bash
KDA_FINALIZE_STAGE01_CAPTURE=0 python setup.py cmake_build
msprof --output=./profile_stage01 --ai-core=on --aic-metrics=PipeUtilization \
  --task-time=on --ascendcl=off \
  python tests/kda_finalize_stage01/bench_stage01.py --library build/_C.abi3.so --stage 8
```

默认定长 H=96/T=8192，每阶段预热 20 次、采样 10 次；加 `--varlen` 测同规模变长输入。只统计 `Stage01Kernel` 最后 10 次记录；初始化、Host wall time 和其它 kernel 不计入。精度版恢复时执行 `KDA_FINALIZE_STAGE01_CAPTURE=1 python setup.py cmake_build`。

性能库也可用 `check_stage01.py --workspace-only --stage 8 --cores 0 --suite model --repeat 1 --library build/_C.abi3.so` 检查各核真实 workspace 最后两代结果，避免只验证带 capture 的同步路径。

小 case 模拟器入口支持 `--heads 2 --tokens 192 --cores 1 --stage 8 --warmup 0 --repeat 1 --device 0`。三个 work 用于检查双槽回绕及跨 work 重叠。模拟器使用 CPU 初始化，不读取输出；需要完整的一个 AIC 和两个 AIV 时间线，不能以模拟器耗时替代硬件性能。

## Stage 8

使用同一采数/性能构建的 `--stage 8`。`check_stage8.py` 检查两带的 FP64 公式、实际 BF16 三项乘加、零填充和保留 workspace。capture 的 656/688 KiB 保存 `dq_local_raw`/`left_raw`；每代 320 KiB workspace 的对应偏移为 256/288 KiB。短 chunk 未计算的第二带保持 capture guard。

分带公式检查传播 Stage 7 已有绝对误差：`6e-5 * (abs(A) @ abs(kNeg)) + sum(abs(A), -1) * 2e-38 + 2e-38`。实际三项 MMAD 的局部门槛为 `2e-6 * sum(abs(Ai) @ abs(Bi)) + 2e-38`，前序 Stage 门槛不变。该检查仍不代表完整 Finalize 或 ATK 验收。

## Stage 9

`--stage 9` 增加 dq 合成、dg 更新及可选 Q L2Norm backward。私有入口末尾增加 BF16 `dq_out` 和可选 FP32 `q_rstd`；shape 分别与 q、q 的 token 维一致，均须连续且在同一设备。Stage 0～8 的旧调用保持兼容。

```bash
python tests/kda_finalize_stage01/check_stage01.py \
  --library build/_C.abi3.so --stage 9 --device 3 --cores 1 --suite tails --q-norm
```

每带 32 行、内层 16 行双缓冲。q/k、对角线和带内指数复用 Stage 7 的 UB 数据；读完 base 后在输入槽内原位计算，写回结束前不得再次搬入。FP32 dq/dg 分别复用 workspace 的 32/168 KiB，BF16 dq 写实际输出；workspace 总量不增加。capture 的 720/752 KiB 保存两份 FP32 结果。

Cube 每个 head 写完后独立发布 ready，两 AIV 按相同顺序消费两次通知；h0 的 Stage 9 可与 h1 的 Stage 8 重叠。奇数 head 的空 owner 仍闭合同样的握手。

`check_stage9.py` 对照捕获输入的局部 FP64 公式和原始 gate 差构造的因果公式，检查实际 BF16 dq、FP32 dq/dg、尾行、保护区及最后两代 workspace。局部沿用 FP32 门槛 `rtol=3e-4, atol=2e-6`；最终 BF16 沿用 `rtol=8e-3, atol=2e-5`，上游误差按 Stage 5/8 既有预算传播。`--q-norm` 覆盖独立的编译期归一化分支；性能库使用 `--workspace-only` 验证。

性能命令沿用前文，将 `--stage 8` 改为 `--stage 9`；需要测归一化路径时加 `--q-norm`。

## Stage 10

`--stage 10` 不增加参数。计算两项转置 GEMM，每项保留 `Ah@Bh + Ah@Bl + Al@Bh`，六项在 FP32 L0C 累加后写回。两个 head 使用独立的 L0A/B/C 槽，qPos/bkPos 依次复用 Stage 8 的 L1 kNeg 区；与 Stage 9 重叠。

capture 每 head/chunk 为 816 KiB，784 KiB 起保存两带 right_raw。workspace 大小不变，96 KiB 起的旧 kNeg 区保存最终 band 0 的 FP32 right_raw。两轮带间握手保证输入写完才读、结果写完才复用；Stage 11 尚未接入。

`check_stage10.py` 检查实际六项乘积、独立 FP64 居中公式、尾行零值、未计算带保护区及保留 workspace。局部门槛为 `3e-6 * Σ(abs(Ai) @ abs(Bi)) + 2e-38`；分带门槛沿用 Stage 8 的 `6e-5` 操作数传播预算。Stage 0～9 门槛不变。命令使用 `--stage 10`，覆盖 `--q-norm`、三种 gate 和 `--workspace-only`；此入口仍缺 Stage 11～12，不代表完整算子精度验收。
