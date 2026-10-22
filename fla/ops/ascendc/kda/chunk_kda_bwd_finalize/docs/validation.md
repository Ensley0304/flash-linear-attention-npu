# chunk_kda_bwd_finalize 验证记录

## 1. 验证口径

| 项目 | 值 |
| --- | --- |
| 目标产品 | Atlas A2系列产品 |
| 主 case | `B=1,H=32,T=11264,K=V=128,chunk=64` |
| 正式字段 | `BasicInfo Task Duration(us)` |
| 目标 | `<4000 us` |
| 当前实现 | `CG=1`、2 AIV 各 32 token 行、每 logical core 两个 `180 KiB` relay slot；跨 work 提前执行下一 S1a/S1b、`KE` 和 BaseEarly；`Kneg` 两个 32 行分片使用互不重叠的 relay；集合通知不重复增加同语义 AIV 屏障；常驻区与 scratch/IO 间保留 `256 B` bank-group padding |
| 精度约束 | Cube 输入为单份 BF16；`Zv/Zw/Right` 以 FP32 relay 交接 |
| 采样 | 每个可采用版本先完成正确性检查，再连续采集 30 个 `BasicInfo Task Duration(us)` 并重新采集流水；历史行保留原采样数 |

## 2. A/B 性能

| 方案 | 平均值 | 中位数 | 最小值 | 最大值 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| 合并生产/消费 | 6840.08 us | 6848.36 us | 6800.44 us | 6865.46 us | 淘汰 |
| 拆分生产/消费初版 | 6538.23 us | 6535.90 us | 6510.74 us | 6569.44 us | 历史基线 |
| 拆分优化版（降精度 relay） | 4658.06 us | 4655.84 us | 4553.70 us | 4802.64 us | 恢复精度前基线 |
| 拆分优化版（FP32 relay，历史采样） | 4729.44 us | 4734.83 us | 4635.58 us | 4836.14 us | 稳定实现，未达标 |
| 跳过 Gate/参数归约诊断版 | 4112.29 us | 4110.28 us | 4010.72 us | 4216.94 us | 仅用于测当前结构下界 |
| 跨 work S1a/S1b + Gate 导数重叠 | 4044.15 us | 4046.36 us | 3977.38 us | 4114.36 us | 正确，但距目标约 46 us |
| 同步精简前，第 1 轮 | 3979.23 us | 3983.90 us | 3919.26 us | 4040.70 us | 达标；七个输出回归通过 |
| 同步精简前，第 2 轮 | 3984.67 us | 3980.21 us | 3910.78 us | 4068.34 us | 达标；独立复测 |
| 同步精简前，两轮合并 | **3981.95 us** | **3983.90 us** | **3910.78 us** | **4068.34 us** | **按中位数口径达成 `<4000 us`** |
| 同步精简前，最终恢复复核 | 3987.40 us | 3988.20 us | 3919.98 us | 4056.24 us | 达标；失败实验回退后重新编译部署 |
| 同步精简前，三轮合并 | **3983.76 us** | **3985.50 us** | **3910.78 us** | **4068.34 us** | **30 次正式样本中位数达标** |
| 下一 work S1a 前移 | 4006.58 us | 4014.91 us | 3961.02 us | 4050.30 us | 输出完全一致，但占用当前 Intra 关键路径，回退 |
| KE 写出与 Gate scan 重叠 | 4063.12 us | 4064.76 us | 3982.30 us | 4135.40 us | 输出完全一致，但片上通路争用，回退 |
| TOP5 优化前新鲜基线，30 次 | 3974.834 us | 3971.959 us | 3896.978 us | 4049.861 us | 本轮对照 |
| 删除 4 处重复 AIV 屏障，30 次 | 3609.989 us | 3605.292 us | 3568.431 us | 3670.673 us | 普通回归通过；后续 racecheck 证明旧地址布局无效 |
| 同步精简实现，两轮合并 60 次 | 3607.605 us | 3604.552 us | 3552.871 us | 3683.094 us | 历史性能记录，不作为当前结论 |
| bank padding 前冻结基线，30 次 | 3604.789 us | 3599.412 us | 3563.671 us | 3650.753 us | 历史对照，不作为当前结论 |
| scratch/IO 整体增加 `256 B` padding，第 1 轮 | 3465.967 us | 3466.849 us | 3394.988 us | 3535.611 us | 普通回归通过；后续 racecheck 证明旧地址布局无效 |
| scratch/IO 整体增加 `256 B` padding，第 2 轮 | 3480.290 us | 3474.329 us | 3410.768 us | 3566.091 us | 历史复测，不作为当前结论 |
| `Kneg` 仍覆盖 `[32,48) KiB` 的历史版本，两轮合并 60 次 | 3473.128 us | 3470.659 us | 3394.988 us | 3566.091 us | 后续有效 racecheck 发现跨 AIV `DVb/Kneg` WAR；结果无效，不作为性能结论 |
| 当前无竞态实现，30 次 | **3724.532 us** | **3720.244 us** | **3668.913 us** | **3774.015 us** | **racecheck 无错误；独立冷进程结果稳定；达到目标** |
| Stage 交接候选一：`A/Kneg -> Zb -> DQ`，仅 Vector 提前准备 | 3844.413 us | 3839.747 us | 3796.056 us | 3903.778 us | 七项逐 bit 一致；Cube 仍等待 `ZB_READY`，关键路径变长，回退 |
| Stage 交接候选二：`A/Kneg -> DQ -> Zb -> Stage3`，`Zb` 使用 `[180,188) KiB` | 3896.268 us | 3893.588 us | 3856.697 us | 3947.019 us | 七项逐 bit 一致；增加 relay 和等待重排仍无收益，回退 |
| 恢复稳定 Stage 交接，30 次 | **3727.993 us** | **3728.085 us** | **3665.993 us** | **3780.676 us** | **采用；180 KiB relay、`Zb -> Stage3 -> DQ`** |

### 2.1 L2 Cache 策略 A/B

本轮只改变 GlobalTensor 的 L2 cache hint，算子结构、输入、构建、设备频率和采样口径保持一致；
两侧均合并 60 个 `BasicInfo Task Duration(us)` 正式样本：

| 方案 | 中位数 | 相对变化 | 精度结果 | 判定 |
| --- | ---: | ---: | --- | --- |
| 全部使用原默认 cache | `4041.2509 us` | 基线 | 七个输出作为对照 | 基线 |
| `dAqk/dqRaw/rawG` 与 `dq/dk/dv/dBeta` 使用 bypass | `4020.1903 us` | `-21.0605 us`，`-0.52%` | 主 case 七个输出逐 bit 一致 | 采用 |
| 在上一方案基础上继续对 `dG` 使用 bypass | 不采纳 | 不作为性能结论 | `dG/dALog/dDtBias` 出现非有限值 | 回退 `dG` 为默认 cache |

本组 A/B 用于选择 L2 策略，不替代上表历史三轮正式性能结果。最终实现只对已验证的单次流式
输入和只写输出使用 bypass；relay、Gate scratch、PA/PB 以及 `dG` 保持默认 cache。

同步精简前的三轮 30 次正式样本为：

```text
3929.660156, 4017.879883, 3952.959961, 4040.700195, 3950.780273,
4023.660156, 3920.560059, 4014.840088, 3919.260254, 4021.980225,
3929.560059, 4067.840088, 3936.280029, 4024.140137, 3922.979736,
4029.100098, 3910.780273, 4030.719971, 3926.940186, 4068.339844,
4027.660156, 3939.159912, 4056.239990, 3956.159912, 4029.080078,
3945.639893, 4037.399902, 3919.980225, 4020.239990, 3942.399902 us
```

`3470.659 us` 来自存在跨 AIV WAR 的历史版本，只用于追溯优化过程，不能作为当前实现性能。
当前无竞态实现 30 个样本全部低于 `4 ms`。

### 2.2 本轮证据驱动候选

| 候选 | 精度 | `BasicInfo Task Duration(us)` | 判定 |
| --- | --- | --- | --- |
| 将 Q 搬运从 `BuildKe` 后移到 Base | 七项逐 bit 一致 | 均值/中位数 `3599.220/3601.012` | 未超过波动，回退 |
| 删除 Zb 写出前最终 Vector barrier | 七项逐 bit 一致 | `3603.450/3601.762` | 未超过波动，回退 |
| 删除 Zb 内两条独立计算之间的 barrier | 七项逐 bit 一致 | `3601.345/3598.422` | 未超过波动，回退 |
| 将 `rh` 计算前移到 `BASE0_READY` 等待前 | 七项逐 bit 一致 | `3606.015/3601.442` | 无收益，回退 |
| 删除 DK 或 Right input ready 前的 AIV 汇合 | `dk/db/dg/dA/dBias` 错误 | 不采集 | 正确性必需，回退 |
| scratch/IO 相对常驻区增加 `256 B` padding | 普通回归逐 bit 一致 | 历史旧地址布局 `3473.128/3470.659` | 只保留布局选择；该组性能不作为最终结论 |
| `Kneg` 继续复用 slot `[32,48) KiB` | 普通回归可重复，但 racecheck 报跨 AIV WAR | 历史中位数 `3470.659` | 无效布局，淘汰 |
| 将两个 `Kneg` 分片移至 `[136,144)`、`[156,164) KiB` | racecheck 无错误；独立冷进程稳定 | `3724.532/3720.244` | 采用；workspace 不变 |
| 删除 DK input ready 前汇合 | racecheck 报 UB 写后读冲突 | 表观约 `3.57 ms` | 无效结果，回退 |
| 删除 Right input ready 前汇合 | 独立进程结果不稳定 | 不作为性能结论 | 回退 |
| 删除 `KE/Zb` 发布前的 MTE3 排空 | 独立进程结果不稳定 | 不作为性能结论 | 回退 |
| 将 scratch padding 改为 `128 B / 512 B` | `128 B` 正确、`512 B` 正确 | `128 B` 无收益；`512 B` 中位数约 `3850 us` | 保留 `256 B` |
| 删除 mask 写入前 `V->S` 或 PackMatrix 最终排空 | racecheck 稳定复现 RAW | 不作为性能结论 | 回退 |
| `A/Kneg -> DQ` 先于 `Zb/Stage3` | 七项逐 bit 一致 | 候选一/二中位数分别为 `3839.747/3893.588 us` | Cube 等待链虽改变，但 AIV MTE2/MTE3 等待和关键路径变长；回退 |

## 3. 结果与内存检查

| 检查 | 结果 | 说明 |
| --- | --- | --- |
| 当前实现 vs 本轮优化前正确版本 | 通过 | 主 case 七个输出逐元素完全一致，且全部为有限值 |
| Gate 导数与 Right MTE2 重叠 | 通过 | `dq/dk/dv/db/dALog/dDtBias` 一致；`dG` 仅有 FP32 运算结合顺序导致的 `9.31e-10` 最大差异 |
| DQ/DK/Right 独立 ready 调整前后 | 通过 | 主 case 七个输出逐 bit 一致 |
| 降精度 relay vs FP32 relay | 符合预期地发生变化 | `dq/dk/dv/db` 不变；Gate 链的 `dg/dA/dBias` 变化 |
| CPU 高精度 golden | 未完成 | 仍需验证 Cube BF16 输入下的完整输出误差 |
| 独立进程首次冷启动 | 通过 | 6 个 `H=1` 与 2 个主 case 冷进程的七个输出均逐 bit 一致且为有限值 |
| sanitizer racecheck | 通过 | 当前代码重新以 sanitizer 编译，目标 MIX kernel 无错误 |
| sanitizer synccheck | 通过 | 无同步错误；仅报告不影响正确性的 redundant wait 警告 |
| sanitizer initcheck | 通过 | 目标 MIX kernel 明确无错误；其后无源码归属的 GM 诊断不计入目标 kernel 结论 |
| 正式包构建与加载核对 | 通过 | opapi 与目标 kernel 均和本轮构建产物一致 |
| S10 去除输出 store 等待实验 | 失败，已回退 | 小样例 `dk` 最大误差 `0.49736`，确认两个等待承担输出缓冲保护 |
| 双 work UB 流水 | 失败，已回退 | 主 case Gate 输出不正确，Python 整调用约 `7.60 ms`，额外分片与重搬入抵消并行收益 |
| S1b 尾部提前执行初版 | 失败，已修正 | 初版过早复用 `BASE1_READY/RIGHT_READY` 事件；当前版在 `WORK_DONE` 后发布下一代 `BASE1_READY` |
| Gate sigmoid 提前计算初版 | 失败，已修正 | 初版破坏 UB 生命周期；当前版只在 Right MTE2 期间计算导数并保留 FP32 |
| Gate 前提前发布下一 work `Zb` | 失败，已回退 | 多 work 复用同一 CrossCore 事件代际，实测死锁 |
| `dG` 写回与下一 BaseEarly 重叠初版 | 失败，已修正 | 初版缺少 `V->MTE2` 依赖导致 Gate 输入被覆盖；当前版补齐依赖后七项逐元素一致 |
| 下一 work `dqRaw` 提前搬入 | 失败，已回退 | 手工 MTE2 进入既有 V/MTE2 事件链后输出变化，且整调用变慢 |
| `expG/centerExp` 改为倒数乘法 | 失败，已回退 | FP32 舍入差异传播到多个输出，快速采样无性能收益 |
| 下一 work S1a 移入当前等待窗 | 失败，已回退 | 七项逐元素一致，但正式中位数为 `4014.91 us` |
| 下一 work 整段 `BuildKe` 提前到 Gate scan 前 | 失败，已回退 | 覆盖当前 Gate 仍存活的 UB 数据，Gate 相关输出出现非有限值 |
| KE 仅提前写出、scan 后发布 | 失败，已回退 | 七项逐元素一致，但正式中位数为 `4064.76 us` |
| 删除 S3 `PIPE_FIX` 屏障 | 失败，已回退 | 出现非有限输出，确认该屏障参与 Fixpipe 结果可见性保护 |
| 删除 `ZB_READY/DQ_INPUT_READY/KE_READY/WORK_DONE` 前的重复 AIV 屏障 | 有条件采用 | 集合通知负责汇合；`Zb` 后续写入需配合 `Kneg` 分片地址隔离，不能只删除屏障而保留旧地址 |
| 删除 DK 或 Right input ready 前的 AIV 屏障 | 失败，已回退 | 两处分别单测都会破坏 `dk/Gate` 结果，属于必要的数据保护 |
| 删除 `KE_READY/ZB_READY` 内部 `MTE3->MTE2` 排空 | 失败，已回退 | relay/carry 尚未稳定，多个输出变化 |
| Right MTE2 与前序 MTE3 强制并行 | 无收益，已回退 | 无等待版 60 次中位数 `3622.762 us`，保留等待版为 `3604.552 us` |
| Gate UB 热区置换与 `512 B` 地址平移 | 无收益，已回退 | 三种局部布局中位数为 `3613.952/3615.142/3604.812 us`；不同于当前将全部 scratch/IO 与常驻区错开的全局 `256 B` padding |
| UB 容量 | 通过 | scratch 起点 `88.25 KiB`，最高有效对象结束地址低于 `192 KiB` |
| L1/L0 容量 | 通过 | 每 AIC L1 最高地址 `448 KiB`；L0A/L0B/L0C 分别不超过 `16/32/32 KiB` |
| relay 地址与生命周期 | 通过 | 两个 `180 KiB` 槽按 `WORK_DONE` 回绕，各 Stage 地址与释放条件闭合 |
| 用户额外 workspace | 通过 | 主 case 为 `8.23296 MB`，不随 `T` 线性增长 |

## 4. 实现与设计校准

| 实现事实 | 设计处理 | 状态 |
| --- | --- | --- |
| `CG=1`，每 head 内连续 chunk 分段并按 head 轮转余数 | 已写入任务映射公式 | 已校准 |
| 两个 `180 KiB` relay slot | 已写入地址、生命周期和 workspace | 已校准 |
| DQ、DK、Right 独立生产消费 | 已拆为 S5/S6、S7/S8、S9/S10 | A/B 性能支持保留 |
| Intra Cube 输入采用单份 BF16 | 已确认为计算精度约束 | 保留 |
| `Zv/Zw` 以 FP32 写 relay | 已恢复；`Zb` 移至后续 DQ 区以消除跨 AIV 覆盖风险 | 主 case 稳定性通过 |
| `Right` 以 FP32 写 relay | 已恢复，slot 大小不变 | 主 case 稳定性通过 |
| `Kneg` 两个 AIV 分片 | 分别写 slot `[136,144)` 与 `[156,164) KiB`，AIC 搬入连续 L1 | 消除与 peer AIV `DVb` 的跨 AIV WAR；racecheck 通过 |
| Gate 按物理 AIV/head 保存 PA/PB | 已增加 S11 和持久区计算 | 历史 sanitizer 通过，待数值验收 |
| Gate `one=1` 常量 | 固定驻留 UB，避免每 chunk 重复初始化 | 小样例七个输出逐 bit 一致 |
| 下一 work S1a/S1b 提前执行 | `BASE0_READY` 可立即发布；复用事件的 `BASE1_READY` 延迟到当前 `WORK_DONE` 后 | 主 case 七项回归通过 |
| 下一 work BaseEarly 提前执行 | 当前 Gate 建立 `V->MTE2` 依赖后启动，与当前 `dG` MTE3 重叠 | 三轮正式中位数均低于 4 ms |
| L2 cache hint | `dAqk/dqRaw/rawG` 与 `dq/dk/dv/dBeta` 使用 bypass；relay、持久记录和 `dG` 保持默认 cache | 同条件 A/B 中位数降低 `21.0605 us`，七个输出逐 bit 一致 |
| 集合通知前的重复 AIV 屏障 | `ZB_READY/DQ_INPUT_READY/KE_READY/WORK_DONE` 直接使用 `0x2` 集合通知；DK/Right input ready 保留必要屏障 | 配合 `Kneg` 地址隔离后 racecheck 与冷进程回归通过 |
| UB bank-group padding | 常驻区 `[0,88) KiB` 与 scratch/IO `[88.25,192) KiB` 间留 `256 B` 空洞 | `128 B/512 B` 替代布局均未获益，保留 `256 B` |

## 5. 用户额外 workspace

只计算算子执行期间额外占用的 GM，不包含 API 输入、最终输出、片上空间和固定系统
workspace。`Nphys=2*Nc`：

```text
relay_bytes = Nc * 2 * 180 KiB
gate_scratch_bytes = Nc * 8 KiB
PA_bytes    = Nphys * H * 32 B
PB_bytes    = hasDtBias ? Nphys * H * 512 B : 0

user_workspace_size = Align512(
    relay_bytes + gate_scratch_bytes + PA_bytes + PB_bytes)
```

主 case `Nc=20、H=32、hasDtBias=true`：

| 区域 | 字节 | MB（十进制） |
| --- | ---: | ---: |
| relay | 7,372,800 B | 7.37280 MB |
| Gate scratch | 163,840 B | 0.16384 MB |
| PA | 40,960 B | 0.04096 MB |
| PB | 655,360 B | 0.65536 MB |
| 合计 | **8,232,960 B** | **8.23296 MB** |

该容量不随 `T` 或总 chunk 数线性增长；slot 按并发 logical core 复用，PA/PB 只按
物理 AIV 和 head 持久化。

## 6. 当前结论

当前无竞态版本保留拆分式 DQ/DK/Right 生产消费、FP32 relay、跨 work S1/BaseEarly 预取、
`256 B` UB padding，以及必要的 DK/Right 汇合。`Kneg` 两个 32 行分片改用互不重叠的 GM
地址，避免 peer AIV 尚未完成 `DVb` 读取时发生覆盖；relay 槽仍为 `180 KiB`，workspace
不增加。

本轮重新设计并实测了两种 Stage 交接：仅提前 Vector 的 `A/Kneg` 准备没有改变 Cube
等待点；进一步让 Cube 先算 DQ 并把 `Zb` 移到 `[180,188) KiB` 虽然功能逐 bit 一致，
但正式中位数分别为 `3.839747 ms` 和 `3.893588 ms`，均慢于稳定版。本轮不改变已验证
的 `Zb -> Stage3 -> DQ` 交接结构。

恢复稳定版本后的最新 30 次采样中位数为 `3.728085 ms`，均值为 `3.727993 ms`，
范围为 `3.665993~3.780676 ms`。流水仍为 pipeline bound：Cube 计算约 `9%`，
AIV `MTE2` 等待约 `75%`、`MTE3` 等待约 `65%`；继续提升需要新的片上数据通路设计，
当前已验证的 Stage 交接没有可重复收益的安全改动。

主 case 30 次 `BasicInfo Task Duration(us)` 的中位数为 `3.720244 ms`，平均值为
`3.724532 ms`，范围为 `3.668913~3.774015 ms`。对应流水仍为 pipeline bound：Cube
占用 `8.99%`、Vector 占用 `48.70%`；Cube wait ratio `63.69%`，AIV MTE2 wait ratio
`74.40%`、MTE3 wait ratio `64.63%`；AIC 主要等待约 `1293.886 us/1481.898 us`。
GM 各方向带宽利用率仅约 `3%~9%`，因此不是带宽瓶颈。

五类候选已按流水证据逐项验证：压缩 KE/Zb 等待链、减少 Stage 通知、增加 MTE2/V/MTE3
并行、调整 UB bank-group、继续优化 L2/GM/GEMM。可删除的冗余同步已删除；其余同步删除
会产生 racecheck 冲突或冷进程不稳定；`128 B/512 B` padding 均不优于 `256 B`；GM 带宽
和 Cube 计算本身不是限制。当前 Stage 结构下已没有通过正确性检查且有可重复收益的候选，
继续提升需要重新设计 Stage 交接，而不是继续删除数据保护。

当前实现没有新增非 Cube 降精度：Cube 输入为 BF16，Vector 与
`DkState/DVb/DKgb/Zv/Zw/DQ/DK/Right` relay 保持 FP32。racecheck 无错误，独立冷进程
回归通过；尚未完成的是 CPU 高精度 golden。
