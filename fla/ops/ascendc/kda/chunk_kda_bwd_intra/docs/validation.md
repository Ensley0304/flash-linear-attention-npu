# ChunkKdaBwdIntra 验证说明

## 当前状态

本地开发机没有 CANN、PyTorch/NPU 运行时和昇腾设备，因此当前只能完成源码级检查；不能据此声明 AscendC 已编译、上板精度已通过或性能已达标。

| 项目 | 状态 | 说明 |
|---|---|---|
| Python 语法 | 已通过 | wrapper、导出、CPU reference、NPU pytest 均通过 `py_compile` |
| ABI/layout 静态 smoke | 已通过 | 校验 BNSD 转换、BF16 gate 提升、19 个 aclnn 参数和输出布局恢复 |
| safe-gate 代数 smoke | 已通过 | 首/中/尾参考点分解在多种 chunk/tail 长度下与直接 causal 公式一致 |
| C++ 结构 smoke | 已通过 | 花括号、11 个 tiling 字段、4 个 tiling key、packed metadata 路径一致 |
| patch 卫生 | 已通过 | `git diff --check` 无错误 |
| CANN host/kernel 编译 | 部分通过 | A2/CANN 9.1 上旧逐行 dA 读取版本已编译；当前批量 dA slab 与 Scalar→Vector 同步修正待重编 |
| AscendC NPU 精度 | 定位中 | A2 已成功 launch；旧版本 5 个接口/异常用例通过，11 个数值用例均先在 `dk` 失败，当前搬运/同步修正待回归 |
| Profiling/性能优化 | 待 NPU 环境 | 当前仅完成静态流水与搬运优化 |

## A2 定位基线（2026-07-21）

旧逐输出行 dA 搬运版本在 A2/CANN 9.1 上完成构建、wheel 安装和真实 kernel launch。
完整 pytest 为 `5 passed, 11 failed`：所有 11 个数值用例的 `dq` 已通过 CPU FP64
golden，断言随后在 `dk` 失败；失败覆盖 safe/unsafe、FP16/BF16、dense/varlen、GVA、
四种 layout、BT=64/128 和 K=16/48/96/256，因此不能归因于单一公开 layout 或 K tail。

当前修正把每个 16-token task 的 dA 行/列改为一次批量 slab 搬入，并为关闭 auto-sync 的
UB 标量读取补齐显式 `S_V` 依赖；此前仅有 Vector→Scalar 的 `V_S`，不能保证后续 Muls/Vector
指令已经看到 Scalar 流水物化的 dA/beta 系数。测试同时增加一个零 dA
累积量恒等用例及四个单点路径用例：`dAqk/dAkk * safe/unsafe`。它们用于区分基础输入输出、
dA 行向 left 和列向 right 贡献；完整断言也改为
一次报告 `dq/dk/db/dg`，避免在 `dk` 后丢失 `db/dg` 证据。该修正尚未获得板端通过结果，
不得将静态检查视为精度通过。

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
# 版本门禁：当前测试文件应收集 21 条；若仍是 16 条，说明服务器未拉到本修正。
python -m pytest --collect-only -q \
  torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py

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
