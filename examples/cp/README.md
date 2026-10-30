# GDN CP 示例

KDA 的新接口、AllGather/PTP、packed/zigzag 和 A5 适配请看 [KDA CP 说明](README.kda.md)。本页以下内容仅适用于原有 GDN 示例。

## 环境

在 Linux 昇腾服务器上加载 CANN，激活已安装 `torch`、`torch_npu`、`fla_npu` 的环境。算子包需包含对应平台的 GDN 小算子和 CP 算子，安装见[开发者指南](../../docs/开发者指南.md)。

```bash
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export FLA_NPU_STABLE_ABI=ctypes
export HCCL_EXEC_TIMEOUT=120
```

包需提供 `pre_process_fwd_kernel_merged`、`chunk_delta_h_bwd_preprocess`（反向预处理旧名）、`merge_fwd_bwd_kernel`。

GDN 均直接向 Dhu 传入 Merge 得到的 `dht`，算子包还需包含 [PR #933](https://github.com/flashserve/flash-linear-attention-npu/pull/933) 的修复；旧 Dhu 会忽略该参数，不能用于本示例。

## 运行

在仓库根目录执行，将设备 `0,1` 替换为分配到的卡。脚本自动识别平台；双卡时全局 T=256，每卡处理 128 个 token。

```bash
# GDN 双卡
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  examples/cp/chunk_gdn_cp.py --tokens 256 --heads 2 --accuracy-check

# GDN 双卡 GVA：Hk=2，Hv=6
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  examples/cp/chunk_gdn_cp.py --tokens 256 --heads 2 --value-heads 6 --accuracy-check

# GDN single NPU
ASCEND_RT_VISIBLE_DEVICES=0 python examples/cp/chunk_gdn_cp.py \
  --device 0 --tokens 128 --heads 2 --accuracy-check

# GDN CPU reference only
python examples/cp/chunk_gdn_cp.py --tokens 128 --heads 2 --cpu-reference-only
```

双卡成功时打印 `gdn CP world=2 accuracy: PASS`，退出码为 0。`--accuracy-check` 检查输出、梯度及 h0/dht；不加该参数只检查结果有限。

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--tokens` | `256` | 全局长度，须为 `64 × WORLD_SIZE` 的正整数倍 |
| `--heads` | `2` | Q/K head 数，范围 `[1,128]` |
| `--value-heads` | 等于 heads | 仅 GDN；Hv 在 `[1,128]`，Hv/Hk 为 1、2、3 或 4 |
| `--device` | `0` | 仅单进程使用；多进程由 LOCAL_RANK 选卡 |
| `--seed` | `20261010` | 随机种子 |
| `--accuracy-check` | 关闭 | 与完整序列 CPU 参考比较 |
| `--cpu-reference-only` | 关闭 | 只运行 CPU 参考，不使用 torchrun |
| `--output-tol` | `0.005` | 输出 atol/rtol |
| `--grad-tol` | `0.008` | dq/dk/dv atol/rtol |
| `--gate-tol` | `0.02` | 门控及其参数梯度 atol/rtol |
| `--boundary-tol` | `0.02` | h0/dht atol/rtol |

精度检查同时要求结果有限、普通量余弦相似度 ≥0.999、门控梯度 ≥0.99；`rel_l2` 仅供参考。

## 模型调用入参

内部入口为 `run_gdn(x, *, scale, rank, world, architecture="a5")`，同时执行前向和显式反向。`x` 为下表字典，`L=T/world`。

四维输入使用连续 **BNSD `[B,H,T,D]`** 布局；BSND 输入先做 `permute(0,2,1,3).contiguous()`。固定 `B=1`、`K=V=128`，Q/K 已做 L2 归一化。

### GDN

| 键 | dtype | 本地 shape | 说明 |
| --- | --- | --- | --- |
| `q`、`k` | BF16 | `[1,Hk,L,128]` | Q/K |
| `v`、`do` | BF16 | `[1,Hv,L,128]` | value、上游输出梯度 |
| `g` | FP32 | `[1,Hv,L]` | 自然对数域逐 token gate，未做 cumsum |
| `beta` | FP32 | `[1,Hv,L]` | 已激活的更新系数，非 logits |

返回 `o/dq/dk/dv/dbeta/dg/h0/dht`。`o/dv` 使用 Hv，`dq/dk` 使用 Hk；GVA 的共享 Q/K 梯度求和。`h0/dht` 为 `[1,Hv,128,128]`。

### 调用约定

- `scale=128**-0.5`；`architecture` 为 `"a5"` 或 `"a2_a3"`。
- 直接调用前加载 `fla_npu`、初始化 NPU，并将输入放到本 rank 设备；多卡先初始化 HCCL，`rank/world` 与默认进程组一致。有其他并行组时，需改为使用模型的 CP 组。
- 仅沿 token 维连续等长切分。
- 示例未封装 autograd。训练接入需拆开前后向并保存中间量；`dq/dk` 不含 L2Norm backward。

## 支持范围

- `chunk_size=64`，保存中间量、不开重计算。GDN 支持 GVA。
- GDN 直接传入 dht 的版本此前已在 A2 验证单卡、双卡和 GVA；本次 KDA 更新不修改 GDN 代码，也未重新做 GDN 的 NPU 验证。
- 不覆盖 varlen/packed、zigzag、不等长切分、外部非零初始状态、末状态损失和多机验证。
- Dhu 直接使用 `dht`，不追加虚拟 chunk；未做性能验收。
