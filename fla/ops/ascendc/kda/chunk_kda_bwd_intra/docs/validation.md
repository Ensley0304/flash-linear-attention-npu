# ChunkKdaBwdIntra 验证说明

## 当前状态

本地开发机没有 CANN、PyTorch/NPU 运行时和昇腾设备，因此当前只能完成源码级检查；不能据此声明 AscendC 已编译、上板精度已通过或性能已达标。

| 项目 | 状态 | 说明 |
|---|---|---|
| Python 语法 | 已通过 | wrapper、导出、CPU reference、NPU pytest 均通过 `py_compile`；key15 新增八路 contraction canary 和独立源码契约后应收集 37 项 |
| ABI/layout 静态 smoke | 已通过 | 校验 BNSD 转换、BF16 gate 提升、19 个 aclnn 参数和输出布局恢复 |
| safe-gate 代数 smoke | 已通过 | 首/中/尾参考点分解在多种 chunk/tail 长度下与直接 causal 公式一致 |
| 稳定 C++ 结构 smoke | 已通过 | AIV 基线 6 个 key 分支、Alloc/Release 配对与 packed metadata 路径不变 |
| full-Cube 源码契约 | 本地已通过 | key15 为单 tile `TileMmadTla`、MIX 1:2、六次 FP32 contraction、MMAD/Fixpipe `unitFlag=0b11`、HF32 off、600 KiB/逻辑 AIC；key13/key14 和通用 key7 均保留 |
| patch 卫生 | 已通过 | `git diff --check` 无错误 |
| CANN host/kernel 编译 | key15 待执行 | key13/key14 已生成并上板；新增 key15 必须重新单算子快速编译 |
| AscendC NPU 精度 | key15 待执行 | 历史回退路径已完成回归；key15 必须重新完整通过，不能沿用旧 wheel 结论 |
| Profiling/性能优化 | key14 已采集 | key7 48.660 ms、BK64 key12 32.477 ms、key13 31.034 ms、key14 31.036 ms；key15 待采集，要求每次调用只有一条 MIX KDA 记录 |

## A2 精度与性能基线（2026-07-21）

EVENT4 legacy 版本完整 pytest 为 `22 passed in 30.53s`。随后 `75535cd` AIV block-wise
clean wheel 完成真实 kernel launch，原完整 pytest 为 `22 passed in 12.77s`。覆盖
safe/unsafe、FP16/BF16、dense/varlen、GVA、四种 layout、BT=64/128、K=16/48/96/256、
重复 launch、零 dA 和 one-hot dA 路径。当前额外保留 1 项 endpoint reassociation 极值用例，
因此方向端点修正版原预期收集 23 项。本轮再增加 `dAqk/dAkk` 两路 rowBlock3 off-left BF16
canary、dense random、Cube repeated-launch、八路 full-Cube path canary 和 2 项源码契约，实验分支完整收集数应为 37 项；尚未上板前不声明新增路径已通过。

目标性能 shape 的 legacy AscendC kernel duration 为 477.937 ms，AIV time 为 455.734 ms；
同 shape Triton kernel 为 19.272 ms。`75535cd` AIV block-wise kernel 为 48.660 ms，端到端
中位数为 51.107 ms。当前仅增加方向端点数值修正，必须重新完成编译、23 项精度与 repeated
launch 后再复测；不能把旧二进制的 48.660 ms 自动当作新 wheel 的实测结论。

key13 的目标 shape profiling 已确认设备二进制来自目标 wheel，6 次 kernel duration 为
`31.030～31.044 ms`，中位数 31.034 ms。key14 只在该路径上增加 row post-scale gate/Mul/Add
repeat；该结果不能作为 key14 的编译、精度或性能结论。

## 构建与安装

以下命令在仓库根目录执行，SOC 按机器替换为 `ascend910b`、`ascend910_93` 或 `ascend950`：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/check_npu_env.py

# 单算子 run 包；用于快速暴露 op_host/op_kernel/op_api 编译问题
bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu --ops=chunk_kda_bwd_intra

# Python wrapper wheel
cd torch_custom/fla_npu
python3 setup.py bdist_wheel
cd ../../..

./build_out/fla-npu-*.run --install
python -m pip install --force-reinstall --no-deps \
  torch_custom/fla_npu/dist/flash_linear_attention_npu-*.whl
```

安装 run 包后必须启动新的 Python 进程，避免复用已经 `dlopen` 的旧 `libcust_opapi.so`。

## 精度回归

```bash
# 版本门禁：加上八路 key15 contraction canary 和独立源码契约后，应收集 37 条。
python -m pytest --collect-only -q \
  torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py

# 不上卡也应先通过源码架构契约；实现前该用例按 TDD 预期为红灯。
python -m pytest -q \
  torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py \
  -k "rowblock3_cube_source_contract"

# Cube 实验上卡后的首个数值门禁：BF16/safe、满 BT64、K128 的 dAqk/dAkk 两路。
python -m pytest -q -vv -p no:cacheprovider \
  torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py \
  -k "rowblock3_off_left_cube_canary or rowblock3_cube_dense_random or rowblock3_cube_repeated_launch" -s

# 先跑 5 条路径定位用例，分别隔离零 dA、dAqk/dAkk 和 safe/unsafe。
python -m pytest -q -vv -p no:cacheprovider \
  torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py \
  -k "zero_da or one_hot_da_paths" -s

# 定向用例通过后再跑完整单算子回归。
bash torch_custom/fla_npu/test/test.sh --device 0 --op chunk_kda_bwd_intra
# 或直接运行：
python -m pytest -q torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py -s
```

用例覆盖：

- `safe_gate=true`：BF16 `q/k/g/beta`、大负累积 gate、tail token；
- `safe_gate=false`：FP16 兼容分支；
- GVA：`H=1,HV=4` 与 `H=2,HV=4`；
- dense：`B>1` 且 `T>chunk_size`；
- varlen：显式 canonical `chunk_indices`，并覆盖单序列跨多个 chunk；
- `BSND`、`BNSD`、`TND`、`NTD` 和 `chunk_size=64/128`。
- `K=16/48/96/256` 的下界、16-feature 尾块、非 2 次幂和上界。
- 零 dA 时四个输入梯度累积量原样写回。
- 单点 `dAqk[18,2]` 与 `dAkk[18,2]` 的跨 16-token block 行/列路径，分别覆盖 safe/unsafe。
- BF16 safe `B=1,T=64,H=HV=1,K=128` 下，单点 dA 位于
  `target=55`、`source=9`，分别隔离 rowBlock3 off-left 的 `dAqk` 和 `dAkk` Cube 输出；q 与
  target k 置零，避免 right、`db/dg` 项掩盖 left contraction 错误。

## key15 full-Cube 实验门禁

实验 fastpath 仅面向 `safe_gate=true`、BF16、dense、`B=1`、`H=HV`、`BT=64`、`K=128`
和满 chunk。源码契约要求 public fastpath 只调用一次 L0，key15 为
`KERNEL_TYPE_MIX_AIC_1_2`，六组 A/B/C 为 FP32，使用单 tile `TileMmadTla`，HF32 显式关闭，
且每次 MMAD/Fixpipe 都以 `unitFlag=0b11` 和 `M_FIX/FIX_M` 事件完成闭环；
L1A/L1B、L0A/L0B 分别使用独立的 operand event，禁止合并释放事件。AIC/AIV 使用相同 logical core/slot
映射；两个 AIV 子核每 slot 各 set 一次 ready、各 wait 一次 done，AIC 各 wait/set 一次。

稳定 key13 是目标 shape 的立即 fallback，key7 是通用 fallback。key15 workspace 为
`usedCoreNum * 600 KiB`，20 个逻辑 AIC 约 12 MiB；当前不启用 double buffer。建议按以下顺序放行：

1. 源码契约和 Python collect 通过；
2. 单算子快速 build，确认 key13 回退和 key15 MIX 实例均进入 wheel；
3. 八路 full-Cube path canary、zero dA、endpoint guard 通过；
4. 连续 launch 至少 100 次，无超时或设备复位；
5. 完整 37 项通过，原容差不删除、不放宽；
6. msprof 只出现一条 `ChunkKdaBwdIntra`，该行同时具有 AIC/AIV 时间，再与 key7 比较。

key13 保留为已知可运行的 BK64 回退；当前 key15 改动必须重新执行第 2～6 项。未满足第 3～6 项前，
不扩大实验 fastpath eligibility，不移除 key7/key13，也不引入 double buffer、persistent MMAD
或 workspace ring。

板端通过门槛：四个输出均 finite，且逐输出通过测试中的 CPU FP64 golden 容差。若 safe case 失败，应先分别对比 `dq_local`、`dk_left_pre`、`dk_right`，重点检查 16-token 首/中/尾参考点的内外指数方向。

## 性能采集

先固定常用形状分别测试 safe/unsafe，再采集 kernel duration、AIV utilization、MTE2 bandwidth、Vector utilization、流水 stall 和 task tail。至少覆盖：

```text
(BT,K) = (64,128), (128,128)
dtype  = FP16, BF16
mode   = dense, varlen
HV/H   = 1, 2, 4
```

只有 profiling 证明 AIV 基线受重复 source 搬运或短向量指令限制后，再评估文档中的 AIC/AIV 融合候选，避免未经数据引入跨核同步和固定 workspace。
