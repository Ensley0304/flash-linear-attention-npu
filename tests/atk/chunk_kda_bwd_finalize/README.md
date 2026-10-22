# A2 KDA backward Finalize 精度验证

2026-10-22 增量 S1：Stage 11 前带无 norm 16 行布局，九个边界及原 20 case 的 140 输出均通过原 FP32/FP64 CT L1，参考 hash 和种子未变。设备 4 两轮 A/B 的无 norm 四配置均稳定改善，norm 指令与 R2b 相同。目标 B1/H96/T8192/K=V128/BF16 无附加反向 10.4700→10.3655 ms；仍未达 9 ms。本轮不是完整 200×3 或公开入口验收。库、原始证据及适用条件见[实验记录](../../../fla/ops/ascendc/kda/chunk_kda_bwd_finalize/docs/compact-stage11.md)。

测试入口为 `torch.ops.ascend_ops.kda_finalize_stage01(last_stage=12)`，使用关闭
capture 的实际设备动态库，验证完整七个输出。当前是 A2 开发入口验收，不代表公开
ACLNN/OPP 接口、A3、GPU 双标杆或整模型训练验收。

## 标杆

`scripts/reference.py` 独立计算设计稿 Stage 0～12 的数学公式，不读取 NPU
workspace、capture 或设备输出。高精度路径使用 CPU FP64；同精度路径使用 CPU
FP32，并在 dW/kE/Zb/Tza 四处模拟设计中明确的 BF16 交接。两路使用同一份量化后的
输入，最终转换为公开输出 dtype。指数差按数学等式因式分解：FP64 整 chunk 计算，
FP32 每 8 行计算以限制正指数，不模拟设备的 32 行流水或 BF16 残差矩阵乘。
FP64 golden 按 BF16 步长显式执行最近偶数舍入（含次正规数），避免先过 FP32 的二次舍入；
FP32 对照不受此修正影响，24 组跨分支用例已确认逐位一致。
因此当前对照路径的局部矩阵乘精度高于设备三项残差路径；它不是完整模拟全部
BF16 数值交接的对照。用户已确认以纯 FP32 局部矩阵乘为精度目标，保持当前
标杆、输入和判据，继续修复设备精度；不得把当前失败改记为通过。

七个输出顺序为 `dq, dk, dv, db, dg, da_log, ddt_bias`。比较采用用户提供的 Prepare
工程同款 `cv_fused_double_benchmark`：max/avg/RMSE ratio 为 5/1.5/1.5。
所有输出必须有限。此 CPU 同精度标杆不是 GPU 实测数据。

公式依据为已确认的 [Stage 设计](../../../fla/ops/ascendc/kda/chunk_kda_bwd_finalize/docs/design.md)
和 [接口契约](../../../fla/ops/ascendc/kda/chunk_kda_bwd_finalize/docs/api.md)：

| 语义 | CPU 实现 |
| --- | --- |
| Stage 0～6：WY 与状态反向 | 直接矩阵乘；低精度路径保留四处 BF16 交接 |
| Stage 7～10：严格下三角局部项 | 因式分解指数差，单独补 dAqk 对角线 |
| Stage 11：gate 反向 | chunk 内逆向 cumsum，稳定 sigmoid 导数 |
| beta sigmoid / allow_neg_eigval | 乘 `beta*(1-beta)` / `beta*(1-beta/2)` |
| Q/K L2Norm 反向 | 使用保存的归一化 Q/K 和 FP32 rstd |
| Stage 12：参数梯度 | 跨 chunk 累加 dA_log/dDt_bias |
| 输出布局 | CPU 按逻辑轴排列；DUT 直接检查 kernel 原生写回 |

`scripts/calibrate.py` 用独立 autograd 标量目标验证全部七个梯度，覆盖 1/7/33/64 行、
L2Norm 开关与三种 beta 模式，24 组均通过；gate/a_log/bias 同样由 autograd 求导。
负向校准的 7 个正常输出均接受，21 个全零/取反/减半输出均拒绝。
尚未与 GPU/Triton 实测输出对齐，因此此标杆不代表仓库阶段 2 的完整外部参考验收。

## 200 个场景

case 0～191 按 `(变长, L2Norm, beta_mode, token_major)` 字典序枚举，每组 8 个边界。
每个场景固定运行三个 seed：`20261013 + case_id + 10000 * seed_round`。

| 覆盖维度 | 场景 |
| --- | --- |
| beta 反向 | 关闭、普通 sigmoid、allow_neg_eigval |
| Q/K L2Norm | 成对关闭或开启，rstd 来自输入归一化 |
| 输出布局 | 定长 BNSD/BSND；变长 NTD/TND，均由 kernel 原生写回 |
| 固定输出 | dG/dBeta 始终 BSND/TND，不随 output_layout 切换 |
| 尾块 | 1/16/31/32/33/63/64/129，变长额外含 65 和空序列 |
| 分核/复用 | H=1/2/3/4/5/7/16，奇数 head；强制单核、多 work、两代槽复用 |
| 数值 | scale 为零/负值/正常值，lower_bound=-5/-1/-.5/-.01，饱和 gate、beta logits ±8/±16 |
| 模型 | case 192～199：B=1,H=96,T=8192，定长/变长 × norm × output_layout |

精确参数见 `case_matrix.json`。输入为 BF16，beta/a_log/gk/raw_g/dt_bias/dAqk/dq_raw
使用 FP32；K=V=128、chunk=64。当前入口不支持的 dtype、state_v_first、独立 Q/K norm
开关不伪装成正常精度用例。

## 运行

先准备 CANN、torch/torch_npu 和 ATK，设置 `KDA_FINALIZE_LIBRARY` 为本轮构建的性能库。
仓库根目录构建：

```bash
cd examples/fast_kernel_launch_example
NPU_ARCH=ascend910b FAST_KERNEL_OP_NAME=kda_finalize_stage01 \
  KDA_FINALIZE_STAGE01_CAPTURE=0 python setup.py cmake_build
export KDA_FINALIZE_LIBRARY="$PWD/build/_C.abi3.so"
cd ../..
bash tests/atk/chunk_kda_bwd_finalize/scripts/run_accuracy.sh 2 /path/to/fresh-output
```

`scripts/run_accuracy.sh <device> <output_root>` 依次执行三个 seed round；不截取 case 范围。
输出目录必须尚不存在，避免混入旧报告或覆盖旧日志。
使用 `-sp` 避免共享服务器上 ATK 多进程扩张，并保留 `-cp` 参数。
当前开发入口没有 ACLNN 签名，`-cp` 不构成公开 ACLNN ABI 验证。
脚本最后读取 Excel 报告，要求 600 次执行分别具有七个输出和完整三路计算记录；
ATK 命令退出码为零不能代替精度通过。

性能和 MSS 不在此次精度任务的完成声明中。仓库完整转测还要求满足新版 ATK、公开入口、
性能、确定性和内存检测门禁，不能用这 200 个场景的精度结果代替。

## 已定位的精度问题

2026-10-13，A2、ATK 26.7.8、CT 0.9.1：第一轮完整 200 case 的 Excel 已确认
192 通过、8 未达标，失败 case 为 192～199，仅 dq 小值计数项失败，其余六个输出
全部通过。另两轮在最后一例期间中断，连接恢复后确认进程已退出且最终报告未生成，
不能宣称 600 次全部完成。12 组原生布局对照、默认布局兼容和非法布局检查均通过。

case 192、seed 20261205 的 dq 在 BF16 小值计数项失败：100,663,296 个元素中，
10 个点从 `±0.00390625` 变为相邻的 `±0.003936767578125`，相差一个 ULP。
其绝对误差 `3.0517578125e-5` 超过小值门槛 `1.52587890625e-5`；CPU FP32
标杆没有同类超限点。最大/平均相对误差和 RMSE 项通过，不能据此覆盖小值项失败。

十个点在单 head、单 chunk、单核下均复现。CPU 三项残差算术也复现 10/10，
补齐低位乘低位后仍有 7/10，FP32 矩阵乘在这十个点上为 0/10。
这指向 Stage 7～9 残差近似跨越 BF16 舍入边界；后一项仅为 CPU 定位结果，
尚不能说明改成设备 FP32 路径就能通过全量测试。本轮不修改判据或输入来消除失败。

后续设备实验已将 dq 局部矩阵乘改为 FP32、关闭 HF32。修正 FP64 golden 的二次
舍入后重新执行三轮，前 192 个边界 case 均通过，8K 场景仍有未达标项。
新增 3 个失败点已定位到 Stage 7 的指数舍入；整数/小数指数分解在设备上修复了
这 3 个点，13 个冻结 chunk 的 dq 通过原 CT L1 判据。完整结果仍待验证，不能
将局部修复或旧库报告当作当前源码的 200 case 通过证明。

## 2026-10-22 局部驻留增量回归

在 h24（`1f1321bf`）上增加完整 chunk、无 norm 的 4 KiB dq_base 驻留。
R2b 私有 fast-launch 库 SHA256 为
`1ce88f73f066b3115e873e97af25a56a7de11be7311b5d5ab30f7b894b9fc588`。
原九个边界 case 和原 20 case 均七输出通过；20 case 共 140 输出，种子仍为
20261013+case_id，使用原纯 FP32 局部 GEMM 标杆、独立 FP64 golden 和 CT L1。
两轮性能中无 norm 四配置改善 1.01%～1.33%，带 norm 指令未变且耗时持平。
详细样本、门禁和报告位置见算子 docs/resident-transfer.md。
本记录仅为单项优化的增量回归，不是新版本 200×3、公开入口或完整转测验收。
