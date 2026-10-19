# 重新设计的推导记录

这是候选推导，不是设备验收结果。最终选定方案另写 `design.md`，通过设计检查后才能进入实现。

## 固定条件

- 独立输入的语义不变，不能用前向恒等式消去 dW、Zb、Tza。
- 保留原 FP32 局部矩阵乘目标及 FP64 golden，七输出必须通过原 CT L1。
- A2 使用 GM 进行 C/V 交接。目标是完整调用耗时下降；Vector 80% 只是诊断项。
- 不复用旧版的调度结论。旧版逐 head owner、CG=2、跨 work 预取作为测量对照。

## 任务映射候选

| 候选 | 分工 | 有利点 | 必须解决的问题 |
| --- | --- | --- | --- |
| 完整 head owner | AIV0/1 各负责一个 head，CG=2 | 无跨 AIV 归约，旧版已有精度证据 | 每 AIV 常驻完整64行，bases写回GM；晚段受UB容量限制 |
| 按 K/V 各64列 | 两个 AIV 处理同 head 的全部64行 | gate reverse scan不跨片 | db、L2Norm、若干行归约均需跨片；全图通知成本高 |
| 按 token 各32行 | 两个 AIV 处理同 head，保留完整128列 | L2Norm和db行归约本地完成，三个base可驻留UB | reverse scan需要一份128维suffix；末行state修正需要另一片的行和 |

优先实现按 token 分片候选。AIV1负责后32行，先处理；AIV0负责前32行，随后处理。尾块分片仍以32为界，空片跳过数据操作但参加协议。CG先取1，再比较CG=2；不把减少单任务工作量直接解释为吞吐翻倍。

## 完整计算图的调整

原版局部部分按“准备 → dq/left Cube → dq Vector → right Cube → gate Vector”执行。新候选先完成同一32行带的四项局部GEMM，再由该带的owner一次完成dq/dk/db/dg、reverse scan、gate和norm。

```text
全 head 前半段：C0 / V1 → C2 → V3 → C4 / V5 → C6
局部部分：
  V7 准备后32行带的操作数 → C8 后带四项 GEMM → AIV1 V9 收尾
  V7 准备前32行带的操作数 → C8 前带四项 GEMM → AIV0 V9 收尾
最终：独立 V10 归并各 chunk 的 da_log / ddt_bias
```

V7的两片都参与完整操作数准备；V9每片只消费自己32行。V7后带发布后，两个AIV继续准备前带，与C8后带重叠。前带发布完后，AIV1进入后带V9，AIV0等待前带结果。AIV1先完成suffix供AIV0使用，避免再引入一个对称全局归约。

V5的 `sum_rows(k * dk_state)` 两份partial提前写入独立GM记录，AIV1在V9修正chunk末行。其后只需单向传递reverse-scan suffix。这个交接与切开每次128列归约不同。

suffix使用模式1的双参与方协议：AIV0在完成V7前带写出后提前贡献一次set；AIV1写完suffix后贡献一次PIPE_MTE3 set。两片均消费一次wait。AIV0的set不代表suffix就绪，只有两份set聚合后才能读取。AIV1可以先消费并推进，AIV0在需要suffix时消费；同一物理槽的下一代仍由free保护。不能把模式1当成只需AIV1发送的单向通知。

依据：[CANN CrossCoreWaitFlag](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E5%90%8C%E6%AD%A5%E6%8E%A7%E5%88%B6/%E6%A0%B8%E9%97%B4%E5%90%8C%E6%AD%A5/CrossCoreWaitFlag_ISASI.md)，以及固定CATLASS源码 `arch/cross_core_sync.hpp` 的模式1 subblock barrier。实际流水仍需短case验证通知平衡。

## 常驻候选与容量

每个AIV一份32行状态，单位KiB，地址为UB字节偏移除1024：

| 地址 | 对象 | 大小 | 生命周期 |
| --- | --- | ---: | --- |
| 0～8 | q BF16 | 8 | V1～V9 |
| 8～16 | k BF16 | 8 | V1～V9 |
| 16～32 | 原始gk FP32 | 16 | V1～V7前带 |
| 32～48 | exp2(gk) FP32 | 16 | V1～V5 |
| 48～64 | dq_base FP32 | 16 | V5～V9 |
| 64～80 | dk_base FP32 | 16 | V5～V9 |
| 80～96 | dg_base FP32 | 16 | V5～V9 |
| 96～112 | gate导数因子 FP32 | 16 | V1～V9 |
| 112～128 | raw_g+dt_bias FP32 | 16 | V1～V9 |
| 128～132 | beta、db、rstd、diag、rH、g_last、suffix等 | 4 | 分别列生命周期后检查是否足够 |
| 132～188 | 输入/输出双buffer及运算scratch | 56 | 逐阶段复用，不能同时隐式占用 |

这是188KiB上界候选，距平台可用191.75KiB剩3.75KiB。小量区和scratch必须在最终详设中逐项展开，不能以此表直接宣告容量通过。V5以16行tile处理，三路FP32输入×双buffer共48KiB，余8KiB用于cast/运算；输出写常驻bases。其它阶段另核算。

三个base合计每head96KiB常驻，理论上省去192KiB/head的GM写读。主case共12,288个head-chunk，减少2.25GiB流量；这是字节数估计，不是耗时承诺。

gate导数和raw+bias提前准备会增加32KiB/片常驻；需要与晚计算候选独立比较。若堵塞V7或下一work，回退晚计算，不能只凭最早就绪就提前执行。

## 精度候选

早段四处BF16交接保留。局部GEMM先比较纯FP32与BF16高低位残差：纯FP32可省去Vector拆分、cast和残差数据，但Cube执行时间可能增加。必须看完整流水，不能按MMAD次数判断。

以32行带中心 `c=(g[first]+g[last])/2` 构造指数。kNeg仅生成band末行以前的有效行；qPos/betaKPos仅生成band首行以后的有效行。先替换无效指数再exp，避免masked Inf。当前lower_bound≥−5时，带内半跨度最多约111.82（log2单位），再结合输入幅度核查中间乘积。

不使用已下溢的exp2(gk)重新计算带中心。不用HF32代替FP32目标。最终norm、beta反向、gate导数保持FP32，并按原CT检查归约次序变化。

## 比较顺序

1. 逐Stage建立直调正确性版本，覆盖tail、空片、奇数head、变长和slot回绕。
2. 固定分片和精度路径，比较W=1 / W=2；W=2优先只提前下一work的Cube，UB仍一份完整状态。
3. 固定调度，比较局部纯FP32 / 残差；稳定收益再保留。
4. 固定上述路径，比较gate提前 / 晚计算、CG=1 / 2。
5. 八配置两轮交替A/B，收益成立后原20case七输出；最终完整200×3和非法输入。

没有通过精度和性能的候选不替换原基线。与A5的比例注明历史基准条件，仅用同设备新基线A/B判断本轮收益。
