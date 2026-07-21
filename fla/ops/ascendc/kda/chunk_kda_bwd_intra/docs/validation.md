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
| CANN host/kernel 编译 | 待 NPU 环境 | 需要 CANN toolchain |
| AscendC NPU 精度 | 待 NPU 环境 | 需要运行新增 pytest，主验 `safe_gate=true` |
| Profiling/性能优化 | 待 NPU 环境 | 当前仅完成静态流水与搬运优化 |

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
