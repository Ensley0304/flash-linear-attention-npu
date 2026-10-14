# GDN / KDA CP 示例调用说明

本目录包含两个独立示例：[GDN](chunk_gdn_cp.py) 和 [KDA](chunk_kda_cp.py)。脚本自行生成输入，执行连续等长 CP 切分的正反向，并可与完整序列 CPU 参考对比。所有命令均在仓库根目录执行。

## 1. 环境准备

在 Linux 昇腾服务器的仓库根目录执行。提前加载本机 CANN 环境，激活已安装兼容 `torch`、`torch_npu` 和 `fla_npu` 的 Python 环境。`fla_npu` 必须包含对应 A2/A3/A5 架构的 GDN/KDA 小算子和 CP 算子 OPP；仅复制 example 文件不能替代算子包安装。构建安装见[开发者指南](../../docs/开发者指南.md)。

```bash
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export FLA_NPU_STABLE_ABI=ctypes
export HCCL_EXEC_TIMEOUT=120
python -c "import fla_npu; import torch_npu; print(fla_npu.__file__)"
```

本版依赖的 CP Python 入口是 `pre_process_fwd_kernel_merged`、`chunk_delta_h_bwd_preprocess`、`merge_fwd_bwd_kernel`。第二个是反向预处理的旧名；请使用与示例匹配的包，不要混用只导出新名字的版本。

A5 KDA 还需要独立的 aclnnChunkKdaBwdPrepare、aclnnChunkKdaBwdFinalize 接口，仅有完整大融合接口不够。

## 2. GDN

### 运行命令

下面假设设备 0、1 可用；请替换为自己分配到的设备。一个进程对应一张卡，两进程组成一个 CP 组。脚本自动识别平台，不需要传 A2/A5 参数。

```bash
# 双卡：全局 T=256，每卡 T=128，Hk=Hv=2
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  examples/cp/chunk_gdn_cp.py --tokens 256 --heads 2 --accuracy-check

# 双卡 GVA：Hk=2、Hv=6，每三个 value head 共用一个 Q/K head
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  examples/cp/chunk_gdn_cp.py --tokens 256 --heads 2 --value-heads 6 --accuracy-check

# 单卡：检查算子链路，不能替代跨卡验证
ASCEND_RT_VISIBLE_DEVICES=0 python examples/cp/chunk_gdn_cp.py \
  --device 0 --tokens 128 --heads 2 --value-heads 6 --accuracy-check

# 仅 CPU 参考，不需要 NPU；不要通过 torchrun 执行
python examples/cp/chunk_gdn_cp.py --tokens 128 --heads 2 --value-heads 6 --cpu-reference-only
```

成功时打印 `gdn CP world=2 accuracy: PASS`，退出码为 0。每个 rank 分别检查 `o/dq/dk/dv/dbeta/dg/h0/dht`。不加 `--accuracy-check` 只检查结果有限，不表示精度通过。

### 命令行参数

| 参数 | 默认值 | 含义与限制 |
| --- | --- | --- |
| `--tokens` | `256` | 切分前的全局 T；必须为 `64 × WORLD_SIZE` 的正整数倍 |
| `--heads` | `2` | Q/K head 数 Hk，范围 `[1,128]` |
| `--value-heads` | 等于 Hk | V、gate、beta head 数 Hv；范围 `[1,128]`，Hv/Hk 必须为 `1/2/3/4` |
| `--seed` | `20261010` | 输入生成随机种子；各 rank 使用同一份全局输入的不同切片 |
| `--device` | 单进程设备 0 | 仅用于单进程；多进程由 `LOCAL_RANK` 选设备，不能同时传此参数 |
| `--accuracy-check` | 关闭 | 启用完整序列 CPU 精度参考；建议转测时始终开启 |
| `--cpu-reference-only` | 关闭 | 只计算 CPU 参考，不运行 NPU 或 CP 通信 |
| `--output-tol` | `0.005` | 输出的逐元素 atol/rtol |
| `--grad-tol` | `0.008` | dq/dk/dv 的逐元素 atol/rtol |
| `--gate-tol` | `0.02` | dbeta/dg 的逐元素 atol/rtol |
| `--boundary-tol` | `0.02` | h0/dht 的逐元素 atol/rtol |

精度检查还要求结果有限，以及余弦相似度：普通量至少 `0.999`，门控梯度至少 `0.99`。日志 `rel_l2` 是辅助指标，不单独决定 PASS。不要通过放宽容差解决接入错误。

### 接模型时的输入与返回值

脚本的 `run_gdn(x, *, scale, rank, world, architecture="a5")` 是示例内部入口，同时执行前向和显式反向。`x` 是字典，内容如下，`L=T/world` 是本 rank 的长度。

| 字典键 | dtype | 本地 shape | 语义 |
| --- | --- | --- | --- |
| `q`、`k` | BF16 | `[1,Hk,L,128]` | Q/K；示例生成时已做 L2 归一化 |
| `v` | BF16 | `[1,Hv,L,128]` | value |
| `g` | FP32 | `[1,Hv,L]` | 自然对数域逐 token 衰减量，不是 chunk 累积 gate；示例生成负值 |
| `beta` | FP32 | `[1,Hv,L]` | 已激活的更新系数，示例用 sigmoid 生成；不是原始 logits |
| `do` | BF16 | `[1,Hv,L,128]` | 上游传入的输出梯度 dO；示例用随机值模拟 |

四维 token 张量均为 **BNSD（`[B,H,T,D]`）连续布局**。模型使用 BSND 时先执行 `permute(0,2,1,3).contiguous()`，返回的 token 输出/梯度按需转回。各 rank 保持相同 head 数与 dtype，按序切分 T，不能直接把不同样本当作 CP rank。

其他参数：`scale=128**-0.5`；`rank/world` 必须与已初始化的默认 HCCL 进程组一致；`architecture` 为 `"a5"` 或 `"a2_a3"`。直接调用时需先加载 `fla_npu`、初始化 NPU、把全部输入放到本 rank 设备，并在多卡时初始化 HCCL；运行脚本时这些步骤由 `main()` 完成。模型已有其他并行组时，需要先改为显式使用 CP 通信组，不能直接复用默认全局组。

返回字典：`o/dv` 按 Hv，`dq/dk` 按 Hk，`dbeta/dg` 与对应输入同 shape；`h0/dht` 为 `[1,Hv,128,128]`。GVA 的共享 Q/K 梯度是组内贡献的和，不是平均。这里的 dq/dk 是对传入 Q/K 的梯度，模型若需要归一化前梯度，还需接自己的 L2Norm backward。

### 范围与已验证配置

- 固定 `B=1`、`K=V=128`、BF16 主输入、`chunk_size=64`；保存中间量，不重计算。
- A5 使用融合阶段小算子；A2/A3 使用独立小算子链。已在 A2 和 A5 实测，A3 共用分支由 A2 验证，未声明 A3 实机结果。
- A2/A5：双卡全局 T=256、Hk=2、Hv=2/4/6/8，以及单卡 T=128、Hk=2、Hv=6，前后向与边界精度通过。范围不是所有 shape 的穷举保证。
- 不覆盖 packed/varlen、zigzag、不等长切分、外部非零初始状态、末状态损失及多机部署验证。
- 这是接入与正确性示例，不是可直接替换模型层的 autograd 接口。实际训练需拆开前后向、保存中间量并接入模型反传；不能直接对返回的 `o` 调用 backward 就期待整个示例自动求导。
- 已验证版本的 Dhu 忽略 `dht`，示例追加两个虚拟 chunk 注入边界梯度后裁剪，有额外开销；未做性能验收。

`--tokens` 增大时，完整序列 CPU 自动求导参考会增加时间和内存消耗；首次请先跑上面的 T=256 命令。

## 3. KDA

### 运行命令

下面假设设备 0、1 可用；请替换为自己分配到的设备。一进程对应一卡，脚本自动识别 A2/A3/A5，无需指定架构参数。

```bash
# 双卡：全局 T=256，每卡 T=128，Hk=Hv=2
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  examples/cp/chunk_kda_cp.py --tokens 256 --heads 2 --accuracy-check

# 双卡：每卡只有一个 64-token chunk
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  examples/cp/chunk_kda_cp.py --tokens 128 --heads 2 --accuracy-check

# 单卡：检查算子链路，不能替代跨卡验证
ASCEND_RT_VISIBLE_DEVICES=0 python examples/cp/chunk_kda_cp.py \
  --device 0 --tokens 128 --heads 2 --accuracy-check

# 仅 CPU 参考，不需要 NPU；不要通过 torchrun 执行
python examples/cp/chunk_kda_cp.py --tokens 128 --heads 2 --cpu-reference-only
```

成功时打印 `kda CP world=2 accuracy: PASS`，退出码为 0。检查 `o/dq/dk/dv/dbeta/dg/dA_log/ddt_bias/h0/dht`。不加 `--accuracy-check` 只检查结果有限，不表示精度通过。

### 命令行参数

| 参数 | 默认值 | 含义与限制 |
| --- | --- | --- |
| `--tokens` | `256` | 切分前全局 T；必须为 `64 × WORLD_SIZE` 的正整数倍 |
| `--heads` | `2` | Hk=Hv=H，范围 `[1,128]`；KDA 示例没有 `--value-heads` 参数 |
| `--seed` | `20261010` | 输入随机种子，各 rank 使用同一全局输入的不同切片 |
| `--device` | 单进程设备 0 | 仅单进程使用；多进程由 `LOCAL_RANK` 选卡，不能同时传此参数 |
| `--accuracy-check` | 关闭 | 启用完整序列 CPU 精度参考，建议转测开启 |
| `--cpu-reference-only` | 关闭 | 只计算 CPU 参考，不运行 NPU 或 CP 通信 |
| `--output-tol` | `0.005` | 输出的逐元素 atol/rtol |
| `--grad-tol` | `0.008` | dq/dk/dv 的逐元素 atol/rtol |
| `--gate-tol` | `0.02` | dbeta/dg/dA_log/ddt_bias 的逐元素 atol/rtol |
| `--boundary-tol` | `0.02` | h0/dht 的逐元素 atol/rtol |

还要求结果有限，普通量余弦相似度至少 `0.999`，门控及其参数梯度至少 `0.99`。日志 `rel_l2` 仅为辅助指标。不要通过放宽容差解决接入错误。

### 接模型时的输入与返回值

`run_kda(x, *, scale, rank, world, architecture="a5")` 是示例内部入口，同时执行前向和显式反向。`x` 为字典，`L=T/world` 是本 rank 的 token 长度。

| 字典键 | dtype | 本地 shape | 语义 |
| --- | --- | --- | --- |
| `q`、`k`、`v` | BF16 | `[1,H,L,128]` | Q/K/V；示例的 Q/K 已做 L2 归一化 |
| `beta` | FP32 | `[1,H,L]` | 已激活的更新系数，示例用 sigmoid 生成；不是原始 logits |
| `g` | FP32 | `[1,H,L,128]` | 原始逐通道 gate 输入，不是激活后或累积后的 gk |
| `A_log` | FP32 | `[H]` | 各 CP rank 共享的门控参数；示例初始化为零 |
| `dt_bias` | FP32 | `[H,128]` | 各 CP rank 共享的门控偏置；示例初始化为零 |
| `do` | BF16 | `[1,H,L,128]` | 上游输出梯度 dO；示例用随机值模拟 |

四维 token 张量均为 **BNSD（`[B,H,T,D]`）连续布局**。模型若使用 BSND，先 `permute(0,2,1,3).contiguous()`，输出和梯度按需转回。只切 token 维，`A_log/dt_bias` 不切分，所有 rank 使用相同参数值。

本示例固定 `safe_gate=True`、`lower_bound=-5`，原始 gate 的激活关系为：

```text
gate = -5 × sigmoid(exp(A_log) × (g + dt_bias))
```

前向 Prepare 在内部计算分块累计 log2 gate，反向 Finalize 返回对原始 `g/A_log/dt_bias` 的梯度。不要给 `x["g"]` 预先执行一次激活或 cumsum。

其他参数：`scale=128**-0.5`；`rank/world` 与默认 HCCL 进程组一致；`architecture` 为 `"a5"` 或 `"a2_a3"`。直接调用前需加载 `fla_npu`、初始化 NPU、把所有输入放到本 rank 设备，多卡时初始化 HCCL；运行脚本由 `main()` 完成。模型已有其他并行组时，需要改成显式使用 CP 组，不能直接复用默认全局组。

返回字典包含 `o/dq/dk/dv/dbeta/dg/dA_log/ddt_bias/h0/dht`。token 输出/梯度与对应输入同布局，`h0/dht` 为 `[1,H,128,128]`。**`run_kda()` 返回的 dA_log/ddt_bias 是本 rank 的贡献，脚本 main 中另行执行 AllReduce SUM**；直接接模型时需保证共享参数梯度按训练框架约定归约一次，避免遗漏或重复求和。dq/dk 是对传入 Q/K 的梯度，不包含归一化前输入的 L2Norm backward。

### 范围与已验证配置

- 固定 `B=1`、`Hk=Hv`、`K=V=128`、BF16 主输入、`chunk_size=64`，保存中间量、不开重计算；KDA 示例尚未接入 GVA。
- A5 反向使用 Prepare → CP 反向预处理 → Merge → Dhu → Finalize。A2/A3 的 Prepare/WY/门控回代使用 PyTorch NPU 运算，并调用 Dhu 和 Ascend C Intra。
- A2 实测：单卡 T=128/H=2；双卡 T=256/H=2、T=256/H=3、T=128/H=2。A5 实测双卡 T=256/H=2。前后向、边界和共享参数梯度通过。A3 共用分支按约定由 A2 验证，未声明 A3 实机结果。
- 不覆盖 packed/varlen、zigzag、不等长切分、外部非零初始状态、末状态损失及多机部署验证。
- 这是显式前后向接入示例，不是完整 autograd 模型层；实际训练需要拆开前后向、保存中间量并接入模型反传。
- 已验证版本的 Dhu 忽略 `dht`，示例追加两个虚拟 chunk 注入末端梯度再裁剪，有额外计算及舍入开销；未做性能验收。

完整序列 CPU 参考随 T 增大而增加时间、内存消耗，首次请从 T=256 双卡配置开始。
