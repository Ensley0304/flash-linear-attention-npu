# A2 Stage 11/12 开发验证

基线为 `869f1237`（Stage 0～10）。本轮完成 A2 私有开发入口的 Stage 11/12，公开 A5 ABI 不变；尚未接入公开 A2 入口或完成 A3/ATK 验收。

## 实现

- Stage 11：合成 dk/dBeta，反向扫描 dg，再计算 gate 与参数梯度。输入、输出分别双缓冲，q/k、beta、对角线和带内指数沿用驻留数据。
- 完整 8 行用三步并行反向扫描，复用已失效的 left 槽；尾 tile 保留逐行扫描。suffix 跨 tile/带保留，每个 chunk 清零。
- gate 导数用 `exp(-abs(x))/(1+exp(-abs(x)))²`；beta sigmoid 因子用前向保存的 FP32 有效值计算。支持 Q/K norm 成对开启，以及 beta 关闭、普通 sigmoid、allow_neg 三种模式。
- Stage 12：同 stream 独立 AIV kernel，固定顺序归约全部 chunk；16 chunk 输入双缓冲，不用原子加。每个 head 的累加器清零后显式执行 Vector 屏障。
- dk 保持输入布局；dG/dBeta 固定 BSND/TND。主 kernel 的 L1/L0、192 KiB UB、每 AIC 1280 KiB workspace 不增加；另有 `[H,Nc,136]` FP32 partial（目标定长为 6.375 MiB）。Stage 12 使用 18 KiB UB。

## 精度与同步

r4 通过 48 次 Stage 11/12 smoke、268 次采数版扩展检查、132 次性能版检查，共 448 次。覆盖尾长 1/16/31/32/33/63/64、奇数 head、96 head、空序列、B=2、三种 gate、Q/K norm、三种 beta 模式和不同 lower_bound。性能版全部实际输出与采数版逐位一致。

Stage 11 对照已验证中间量的 FP64 公式；Stage 12 对照全部 FP32 partial 的 FP64 和。此前 Stage 0～10 门槛不变。校准通过 autograd 检查扫描/gate/beta 链式法则，并拒绝零输出、符号、扫描方向、漏扫描、漏 lower_bound 和饱和 sigmoid 错误。+30 用例中稳定公式保留约 `-3.34e-12`，朴素 FP32 sigmoid 导数为零。

完整 8K 定长/变长 × Q/K norm 四组模型在采数版、性能版全部通过。新增 Stage 11/12 覆盖全部 head/chunk 的 FP64 检查；前序 816 KiB 采数及 dq/dv 与同输入 Stage 10 截断执行逐位一致。性能版七个最终输出及 partial 与采数版逐位一致。结果见 [采数版](evidence/r4-model-capture.json) 和 [性能版](evidence/r4-model-perf.json)。

主 kernel 模拟器 H=2/T=192/单核三个 work 完整结束，包含一个 AIC、两个 AIV，192 次 MMAD 与 Stage 10 一致。归约 kernel 单独得到完整 AIV 时间线。17/34 chunk 与 96 head 的归约回绕另有硬件检查；不以模拟器时间替代实测性能。

## 性能

使用关闭 capture/trace 的动态库，B=1/H=96/T=8192/K=V=128，归一化关闭。变长切分 `[0,1,31,64,65,4096,3935,0]`。20 次预热、10 次采样，按每次两个 kernel 的 Task Duration 之和取中位数；流水占比分 kernel 报告。

Ascend910B3 设备 2，同环境、同输入连续采集；两版每个 case 均得到 30 条主 kernel 和 30 条归约 kernel 记录。

| case | Stage 0～10 | Stage 0～12 逐行扫描 r3 | Stage 0～12 并行扫描 r4 | r4 相对 r3 |
| --- | ---: | ---: | ---: | ---: |
| 定长 | 8497.680 μs | 12486.179 μs | 12243.415 μs | −1.94% |
| 变长 | 8619.153 μs | 12726.684 μs | 12433.819 μs | −2.30% |

r4 主 kernel 最大流水均为 **Vector 62.9%**；Stage 12 约 17～18 μs，最大流水为 MTE2 56.8% / 58.65%。仍未达到 80% bound。扫描优化降低了总耗时，没有通过增加计算抬高占比；Stage 10 与完整 Stage 12 工作量不同，前缀差值不能当作独立 Stage 11/12 耗时。

原始记录：[Stage 10 基线](evidence/d2-baseline-clean-profile.json)、[r3](evidence/d2-r3-profile.json)、[r4](evidence/d2-r4-profile.json)。各组 min/max、两个 kernel 的独立指标和命令均在 JSON 中。

## 复现与边界

服务器目录 `/data/wys/kda-finalize-a2-stage9-20261012/stage12/`。r3 为修正同步后的逐行扫描基线，r4 只将完整 8 行改为并行扫描；改动见 [r4-scan.patch](evidence/r4-scan.patch)。使用 [build-r4.sh](evidence/build-r4.sh)、[matrix.py](evidence/matrix.py)、[model.py](evidence/model.py)、[run.py](evidence/run.py) 复现。

日志、op_summary、源文件/动态库哈希及模拟器摘要保存在 [evidence](evidence/)。大体积动态库、golden 和原始时间线留在服务器。源码按 LF 归一化核对，clang-format 与 host tiling 检查通过。

开发中已修复尾块非对齐补零，以及归约核处理下一个 head 前缺少清零屏障的问题。设备 3 后续两次在 `set_device` 阶段报 507033，均未执行目标 kernel；失败日志保留，最终完整验证与成对性能采集转至空闲设备 2。旧 Stage 10 对照有一次在导出后异常退出，最终表格使用正常退出的重采记录。长序列归约模拟器任务主动停止并缩小为 T=192，未将不完整记录计入通过结果。

这是分阶段开发验证。完整模型新增阶段通过局部 FP64 检查、前序阶段通过逐位回归；不等同于真实训练前向 dump 的整链路双标杆验收。
