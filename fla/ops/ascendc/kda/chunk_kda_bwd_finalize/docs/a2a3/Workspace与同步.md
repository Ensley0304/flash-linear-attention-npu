# A2/A3 Finalize Workspace 与同步

首版使用两个 kernel：主 MIX kernel 执行 Stage 0～11，独立 Vector kernel 执行 Stage 12；调用层在同一 stream 顺序提交。以下是落码方案，尚未经过设备编译、精度或性能验证。

## 任务映射

设 C 为本次调用的有效 chunk 总数。定长时 C=B×ceil(T/64)；变长时 C=Σseq ceil(length_seq/64)，空序列不生成任务。每个 chunk 保存 sequence_id、序列内 chunk_id、起始 token、valid_rows；每项用 int64，记录为 32 B。

```text
Nwork = ceil(H/2) × C
P     = min(可用 AIC 核数, Nwork)
w     = core_id, core_id+P, ...
c     = w % C
head0 = 2 × floor(w/C)
head1 = head0 + 1
```

一个 AIC 依次完成两 head 的 Cube；AIV0/1 各处理一个 head，整个工作项共享 chunk 边界。奇数 H 的无效 head 跳过数据访问，但仍参与全部核间 set/wait。每核仅保留一个工作项，完成后复用槽位。

目标 case：B=1、H=96、T=8192，C=128、Nwork=6144；变长采用相同调度，不跨序列读尾块。

## GM 固定地址

每个 AIC 预留 1 MiB scratch，两 head 各 512 KiB：

```text
head_scratch = scratch_base + (2×core_id + aiv_id) × 512 KiB
```

下表偏移相对 head_scratch，单位 KiB；均为 ND 连续布局，区间左闭右开。高低位分别成平面，按表中顺序连续存放。

| 区间 | 数据 | dtype / 形状 | 生产→消费 |
| --- | --- | --- | --- |
| [0,32) | dk_state_raw | FP32，64×128 | 0→5 |
| [32,64) | dv_b | FP32，64×128 | 0→3 |
| [64,80) | zV | FP32，64×64 | 0→3 |
| [80,112) | dKgb_raw | FP32，64×128 | 2→5 |
| [112,128) | zW | FP32，64×64 | 2→3 |
| [128,144) | dAkk_raw | FP32，64×64 | 6→7 |
| [144,160) | kE | BF16，64×128 | 1→2 |
| [160,168) | Zb | BF16，64×64 | 3→4 |
| [168,200) | dq_base | FP32，64×128 | 5→9 |
| [200,232) | dk_base | FP32，64×128 | 5→11 |
| [232,264) | dg_base | FP32，64×128 | 5→9 |
| [264,280) | dAqk_strict_lower 高/低位 | 各 BF16，64×64 | 7→8/10 |
| [280,296) | dAkk_strict_lower 高/低位 | 各 BF16，64×64 | 7→8/10 |
| [296,328) | kNeg 高/低位 | 各 BF16，64×128 | 每带 7→8 |
| [328,360) | qPos 高/低位 | 各 BF16，64×128 | 每带 7→10 |
| [360,392) | bkPos 高/低位 | 各 BF16，64×128 | 每带 7→10 |
| [392,408) | dq_local_raw | FP32，32×128 | 每带 8→9 |
| [408,424) | left_raw | FP32，32×128 | 每带 8→11 |
| [424,440) | right_raw | FP32，32×128 | 每带 10→11 |
| [440,456) | dg | FP32，32×128 | 每带 9→11 |
| [456,512) | 预留 | 56 KiB | — |

dW/Tza 只在 L1 中交接，不分配 GM 副本。每带输出的第 0 行对应逻辑 token s；输入/base 仍按完整 chunk 行号索引。

chunk partial 独立于可复用 scratch，采用 `[H,C,136]` FP32：前 128 个元素为 dt_bias partial，后 8 个元素占 32 B，仅第一个保存 a_log partial，其余置零。记录大小 544 B。

```text
scratch_bytes = P × 1 MiB
partial_base  = workspace + scratch_bytes
partial(h,c)  = partial_base + (h×C+c) × 544 B
meta_base     = align_up(partial_base + H×C×544 B, 512 B)
workspace_bytes = scratch_bytes + align_up(H×C×544 B,512 B)
                  + align_up(C×32 B,512 B)
```

Host 用 64 位整数计算大小并检查溢出；以上为算子自有 workspace，框架要求的系统 workspace 另计。chunk 元数据由调用层按 cu_seqlens/chunk_indices 准备，并在主 kernel 前就绪；各 launch 使用同一份映射。

目标 case 的 partial 为 6.375 MiB、元数据为 4 KiB；scratch 随 P 分配，不随 T 增长。每个有效 (head,chunk) 只写一次 partial，无需预清空整个 workspace。

## 主 kernel 核间同步

使用基本 MMAD 和显式事件。模式 2 中，一次 AIC 通知由两个 AIV 分别等待；AIC 等待的通知必须由两个 AIV 都发出。首版固定以下 flag，9/10 保留；不混用占用相同 flag 的高阶 Matmul。依据：[官方 CrossCoreSetFlag](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0273.html)。

| flag | 名称 | 方向 / 发出流水 | 条件 |
| --- | --- | --- | --- |
| 0 | KE_READY | 两 AIV→AIC，MTE3 | Stage 1 的 kE 全部写完 |
| 1 | ZB_READY | 两 AIV→AIC，MTE3 | Stage 3 的 Zb 全部写完 |
| 2 | LOCAL_READY | 两 AIV→AIC，MTE3 | 当前带 Stage 7 全部操作数写完 |
| 3 | BAND_DONE | 两 AIV→AIC，MTE3 | 当前带 Stage 11 写完；末带包括 partial |
| 4 | S02_READY | AIC→两 AIV，FIX | 两 head 的 Stage 0/2 GM 结果写完 |
| 5 | DAKK_READY | AIC→两 AIV，FIX | 两 head 的 Stage 6 结果写完 |
| 6 | LEFT_READY | AIC→两 AIV，FIX | 当前带两个 Stage 8 结果写完 |
| 7 | RIGHT_READY | AIC→两 AIV，FIX | 当前带 Stage 10 结果写完 |
| 8 | WORK_OPEN | AIC→两 AIV，FIX | 当前工作项槽位可用；上一工作项已结束 |

```text
AIC（每个工作项）：
  set WORK_OPEN
  Stage 0 × 两 head；每 head 输入末次读取后预取下一 work 同 owner 的五个输入
  wait KE_READY → Stage 2 × 两 head → set S02_READY
  wait ZB_READY → Stage 4/6 × 两 head → set DAKK_READY
  对每个逆序带：
    wait LOCAL_READY → Stage 8 × 两 head → set LEFT_READY
    Stage 10 × 两 head → set RIGHT_READY
    wait BAND_DONE

每个 AIV（每个工作项）：
  wait WORK_OPEN
  Stage 1 → set KE_READY
  wait S02_READY → Stage 3 → set ZB_READY
  Stage 5
  wait DAKK_READY
  对每个逆序带：
    Stage 7 → set LOCAL_READY
    wait LEFT_READY → Stage 9
    wait RIGHT_READY → Stage 11 → set BAND_DONE
```

以上 set 必须排在表中生产流水的写出之后，wait 在消费搬入之前。每轮通知均一一消费，不累积跨工作项通知。最后一带 BAND_DONE 同时释放 scratch；只有收到两 AIV 的该通知，AIC 才进入下一工作项。AIV 还须等下一次 WORK_OPEN，不能提前覆盖。

## 核内 ready/free

| 区域 | ready | free / 可覆盖条件 |
| --- | --- | --- |
| GM→L1 | MTE2→MTE1 | 末次 L1 读取完成，MTE1→MTE2 |
| dW/Tza L1 | FIX→MTE1 | 两次/一次消费的最后 L1 读取完成；再复用对应地址 |
| L0A/B slot | MTE1→M | MMAD 读完，M→MTE1 |
| L0C slot | M→FIX | Fixpipe 读完，FIX→M |
| GM→UB 工作槽 | MTE2→V | Vector 读完，V→MTE2 |
| UB→GM 输出槽 | V→MTE3 | MTE3 读完，MTE3→V |
| Stage 9 dg GM→Stage 11 | MTE3→MTE2 | 本 AIV 写回完成后再读，不能只等 RIGHT_READY |

L0 的两个 slot 各使用一组本地事件；free 事件进入循环前初始化，退出时回收。dW/Tza 直传事件按两 head 分开并在消费后复用。事件实际 ID 统一申请，不与队列/库内部事件重叠；所有事件 set/wait 配对。PipeBarrier 不能替代上述跨流水、跨核依赖。

Stage 0/1 已实现的本地事件：AIC 为 L1 ready/free、L0A/B ready/free、L0C ready/free、dW ready 各两份，共 14 个 event（每种 HardEvent 两个 ID）；AIV 为输入 ready/free、输出 ready/free 各两份，共 8 个 event。Stage 1 结束消费两个输出 free 后，额外经 `V_MTE3` 确认已消费，MTE3 才能重新置 free；禁止跨 pipe 无确认地连续两次置同一 flag。驻留数据仍按完整 work 生命周期保留，不将 Stage 0 输入搬运完成视作后续 Stage 的释放许可。

Stage 0 五个输入区末次 L1→L0 读取后即可预取下一 work（每 owner 深度为 1），akk/dW 与 workspace 生命周期不变。下一 work 的 input-ready 在补搬 akk 后发布；结束前排空最后一代 free。Stage 7 首带前仅释放当前 work 驻留区，qPos/bkPos 改用 [384,512)，避开下一 work 预取区；后续带只更新三个居中操作数。下一带 LOCAL_READY 到来前，AIC 已完成上一带 Stage 10，并收到 BAND_DONE，读写不会交叉。

## Stage 12 分工与验证

Stage 12 是纯 Vector kernel，按两类 job 做 grid-stride：

- dt_bias：每个 job 处理一个 head 的连续 32 个 K 通道，共 H×4 个 job。
- a_log：每个 job 处理连续 8 个 head，共 ceil(H/8) 个 job，集中写出一组 FP32，尾组只写有效值。

两类 job 各自按 chunk 顺序做 FP32 分批归约，不用原子操作。C=0 时跳过主 kernel，归约 kernel 将两个参数梯度置零。dq/dk/dv/dG/dBeta 已由主 kernel 写出。

首轮验证覆盖：1/16/31/32/33/63/64 行尾块、跨 32 行带的后缀和、奇数 H、含空序列的变长、两个输出布局、beta sigmoid 和 Q/K L2Norm 开关，以及连续多次调用。重点检查无死锁、无越界、无旧 workspace 数据泄漏及七个输出精度。

性能按同一次调用的主 kernel 与归约 kernel 的 Task Duration 合计，另记录调用总耗时。完成设备验证后，再评估跨工作项流水、残差减项及计算图备选；当前容量和同步闭环不代表性能最优。
