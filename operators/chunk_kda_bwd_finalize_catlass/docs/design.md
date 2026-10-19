# KDA Finalize Atlas A2系列产品设计方案

## 1. 目标

主模型 case 为B=1、H=96、T=8192/16384、K=V=128、BF16。覆盖已冻结的定长/变长、尾块、空序列、奇数head、输出layout、L2Norm及beta sigmoid反向。

先建立逐Stage正确性版本，再比较W=1/2及局部矩阵精度实现。性能目标：8K无附加反向约9.60ms；八配置均按完整主核加归约核计时，最终以同设备交替A/B验证，不以Vector占比单独判定收益。当前为待运行验证的设计。

### 1.1 数据类型

| 范围 | 数据 | 数据类型 |
| --- | --- | --- |
| API输入 | q/k/v/v_new/dv_scan/akk/h/dh | BF16 |
| API输入 | gk/dq_raw/dAqk/raw_g/beta/a_log/dt_bias/rstd | FP32 |
| API输出 | dq/dk/dv；db/dg/da_log/ddt_bias | BF16；FP32 |
| Vector计算 | bases、归约、指数、gate与norm | FP32 |
| Cube输入与累加 | S0/S2/S4/S6：BF16×BF16；S8：FP32×FP32 | 均FP32累加，HF32关闭 |
| 跨Stage交接 | dW/kE/Zb/Tza | BF16，其余矩阵结果FP32 |

所有七输出同时满足原CT L1和冻结的CANNBot分区检查。保持FP32局部矩阵乘标杆，不用BF16残差模拟替代它。

## 2. 变量与任务映射

| 符号 | 含义 |
| --- | --- |
| C、D | chunk=64，K=V=128 |
| Nc、H | 全部非空chunk数量、head数量 |
| CG | 第一版为1；两个AIV协作一个head |
| s | AIV子核号0/1；负责行[32s, min(32s+32, validRows)) |
| W、g | relay窗口深度1或2、当前核的work代数 |
| slot | g mod W，与两个带的编号分开 |
| b | 带号1然后0；即先后32行、再前32行 |
| P | min(20, Nc×H)，实际值取平台核数 |

```text
for work = core; work < Nc*H; work += P:
    chunk = work / H; head = work % H
    generation = (work - core) / P
    slot = generation % W
    relay = relay_base + (uint64(core)*W + slot)*524288

fixed token_index  = ((uint64(sequence)*H + head)*T + token_start + row)
varlen token_index = uint64(head)*T + token_start + row
fixed state_index  = ((uint64(sequence)*ceil(T/64)+local_chunk)*H+head)*128*128
varlen state_index = (uint64(chunk)*H+head)*128*128
```

所有GM乘积和加法从第一项起使用uint64_t。token张量最后再乘D和sizeof(dtype)。输出dq/dk/dv按output_layout计算；dg/db固定token-major。尾块pad到64行，Cube输入无效区填零，Vector只写有效行。空片执行通知但不读写无效数据。

UB地址以KiB表示：q=0、k=8、gk=16、e=32、dq_base=48、dk_base=64、dg_base=80、gate_factor=96、raw_x=112、小量区=128、scratch=132。小量区布局：beta128B、db128B、q_rstd128B、k_rstd128B、diag128B、rH512B、g_last512B、center512B、suffix512B、state_sum512B、partial544B，共3744B；剩352B保留。q/k为32×128 BF16，其余大常驻量为32×128 FP32。

scratch共56KiB。S5采用8行tile：三路FP32输入双槽24KiB，其余用于q/k转换、state、乘积与归约。S7采用8行tile：三路FP32输出双槽24KiB，其余用于指数和输入转换。S9采用8行tile：三路FP32输入双槽24KiB，收尾阶段复用为BF16输出双槽8KiB；FP32梯度直接从常驻区写出。两个子阶段不同时占用同一地址。跨Stage复用同时保护MTE2搬入与Vector清零，不能只等待MTE3→MTE2。

UB峰值188KiB，平台上限191.75KiB；剩余连续空间3.75KiB。第一版使用显式本地buffer，不引入未计入的高阶Vector临时空间。

性能候选P1：在scratch末端186～187KiB保留32行beta广播，每行8个FP32副本。S1用Brcb生成，S3/S5/S7/S9按行repeat乘法读取，替代逐行标量GetValue；到S9最后一次beta乘法后释放。该区与已有临时量不重叠，UB峰值仍188KiB，数学运算和舍入位置不变。先做边界精度和同条件性能对照，未过门禁不作为已验收方案。

候选P2：小量区float偏移936处保存32个对角Gather字节索引（额外128B，初始化一次）。S7从已补零的Aq提取32个对角元素，广播结果驻留187～188KiB到S9对角项合并结束。S9 norm用182KiB处256B临时广播区，dBeta在144KiB处生成32B/行打包；保持原归约顺序和乘法顺序。所有区域均在现有188KiB内，不增加事件或workspace。

L1地址以KiB表示：akk[0/1]=0/8、dW=16、kE=32、Zb=48、Tza=56、h=64、dh=96、v_new=128、dv_scan=144、v=160。局部Aq/Ak共享区位于176/192，各16KiB；两个带操作数槽位于208和304，各96KiB：kNeg32、qPos32、betaKPos32。Aq/Ak从GM仅搬入一次并保留到两个带全部消费。两个akk代际最多16KiB，其余早段驻留仅一份；下一work仅在这些区域旧消费者结束后进入。L1峰值400KiB，剩余连续111.75KiB。

L0A/L0B各两槽，每槽32KiB；L0C两槽，每槽32KiB。分别不超过64/64/128KiB。无自定义FixBuf分配，按平台返回2KiB限制使用普通Fixpipe。使用CATLASS `TileMmad<Arch::AtlasA2,...>`及对应Copy组件；地址、双buffer和事件由本算子显式管理。FP32转置路径按layout选择LoadData与kDirectionAlign，不沿用BF16的分形宽度。

## 3. Stage 总览

| Stage | 执行单元 | 主要动作 | 结束条件 |
| --- | --- | --- | --- |
| S0 | Cube | dW、dk_state_raw、dv_b、Zv | 四个结果写入完成 |
| S1 | Vector | kE、exp、rH、gate因子、基础输入 | 两片kE就绪 |
| S2 | Cube | dKgb_raw、Zw | 两结果写入GM |
| S3 | Vector | Zb、dv、db_v | 两片Zb就绪 |
| S4 | Cube | Tza | BF16驻留L1完成 |
| S5 | Vector | 三个base、db_base、state_sum | 常驻结果就绪，partial写完 |
| S6 | Cube | dAkk_raw | 写入GM完成 |
| S7 | Vector | 严格下三角、两个带的指数操作数 | 逐带两片均写完 |
| S8 | Cube | 每带dq_local、left、right | 逐带三项结果写完 |
| S9 | Vector | 局部梯度合并、scan、gate、norm、输出 | 七输出中的局部结果及partial写完 |
| S10 | Vector | 跨chunk归并da_log/ddt_bias | 独立归约kernel结束 |

## 4. Stage 详细设计

### S0 · Cube：早段矩阵乘

```text
dW = BF16(dv_scan @ h.T)
dk_state_raw = v_new @ dh.T
dv_b = akk.T @ dv_scan
Zv = dv_scan @ v.T
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| dW | L1 | 16KiB | 64×128 BF16，一份，L0C经Fixpipe转换；S2最后消费后释放 |
| dk_state_raw、dv_b、Zv | GM | relay+0/32/64KiB | FP32，分别32/32/16KiB；消费者S5/S3/S3 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| akk/h/dh/v_new/dv_scan/v | L1 | 0或8/64/96/128/144/160KiB | BF16来自API；akk保护至S6，dv_scan至S2，h/dh仅当前Cube使用；与S1各自独立搬入 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| A/B tile | L0A、L0B | 各0/32KiB | 两槽，最大BF16 128×128；MTE1/M事件保护 |
| 累加结果 | L0C | 0/32KiB | 两槽FP32，最大64×128；Fixpipe完成后复用 |

### S1 · Vector：独立准备

```text
e = exp2(gk); kE = BF16(k*e)
rH[k] = sum_v(h[k,v]*dh[k,v])
x = raw_g + dt_bias; ea = exp(a_log)
z = exp(-abs(x*ea)); gate_factor = lower_bound*ea*z/(1+z)^2
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| q/k/gk/e/gate_factor/x | UB | 0/8/16/32/96/112KiB | 每片32行；e至S5，gk至S9恢复factor，其余至S9；下一work不能覆盖 |
| beta/rstd/g_last/rH | UB | 小量区128KiB内 | FP32；rH由chunk末行owner计算（validRows>32为AIV1，否则AIV0），按16个K行分块搬h/dh；diag留到S7读取dAqk时提取，避免重复搬矩阵 |
| kE | GM | relay+80KiB | 64×128 BF16，两片各写32行；S2读后释放 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| 原始输入 | GM | API指针+任务地址 | q/k/gk/raw_g/beta/rstd等；当前Vector路径只搬一次常驻范围 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| 输入/输出双buffer、cast临时量 | UB | scratch132～188KiB | 8或16行tile；kE写回完成后才能复用输出槽；rH的h/dh临时量不与gate临时量同时存活 |

S0/S1并行。无效行先替换为安全指数0，最终kE填零。S1在kE写出后发布ready；独立gate准备不延迟kE通知。

### S2 · Cube：dKgb与Zw

```text
dKgb_raw = akk.T @ dW
Zw = dW @ kE.T
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| dKgb_raw、Zw | GM | relay+96/128KiB | FP32，32/16KiB，分别由S5/S3消费 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| akk、dW、kE | L1 | 0或8/16/32KiB | akk来自S0，dW至本Stage最后消费；kE等待S1两片完成后搬入，使用BF16×BF16 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| A/B双槽、C双槽 | L0A、L0B、L0C | 各0/32KiB | 同S0，操作数各自消费完成后交换槽位 |

### S3 · Vector：Zb与dv

```text
Zb = BF16(tril((Zv-Zw)*beta[None,:], -1))
dv = BF16(beta[:,None]*dv_b)
db_v = sum_v(v*dv_b)
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| db_v | UB | 小量区db | 每片32个FP32，S5原位更新db_base，保留到S9 |
| Zb | GM | relay+144KiB | 64×64 BF16，两片写互不重叠行；S4读取 |
| dv | GM | 输出地址 | 原生目标layout，两个输出槽各自等待MTE3完成 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| S1常驻量 | UB | 0～48、96～132KiB | 继续保护，不覆盖gate/x和原始gk |
| Zv、Zw、dv_b、v | GM | relay+64/128/32KiB及API地址 | FP32/FP32/FP32/BF16；每片仅取负责行，Zv最后读后可供S7改义 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| 三路输入和输出双槽 | UB | scratch132～188KiB | 8行tile，各阶段共享的scratch在本地MTE3完成后改义 |

Zb按列乘beta，需要完整64个beta值；在scratch中额外读取完整beta，不能误用仅32行的常驻beta。该256B读入与运算临时区复用，不延长到S5。

### S4 · Cube：Tza

```text
Tza = BF16(Zb @ akk.T)
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| Tza | L1 | 56KiB | 64×64 BF16，Fixpipe转换写入，保护至S6消费 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| Zb、akk | L1 | 48KiB、0或8KiB | Zb由S3两片写完后搬入；akk不得提前用于下一代 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| A/B/C双槽 | L0A、L0B、L0C | 各0/32KiB | BF16矩阵乘、FP32累加、最终转BF16 |

### S5 · Vector：base常驻

```text
state = dk_state_raw * exp2(g_last-gk)
q0 = scale*dq_raw*e; gb = beta[:,None]*e*dKgb_raw
dq_base = q0
dk_base = state - gb
dg_base = q*q0 - k*state - k*gb
db_base = db_v - sum_k(k*e*dKgb_raw)
state_sum = sum_rows(k*state)
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| dq/dk/dg_base | UB | 48/64/80KiB | 各32×128 FP32，一份，S9直接消费，不写GM |
| db_base | UB | 小量区db | 覆盖已消费的db_v，保留至S9 |
| state_sum | GM | relay+152KiB+s×512B | 每片128个FP32，空片写零；S9由末行owner合并，不能以单片ready代替两片数据 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| q/k/gk/e、beta/g_last、gate/x | UB | 固定区 | e最后消费，其他固定数据继续保护；state指数非正 |
| dk_state_raw、dKgb_raw、dq_raw | GM | relay+0/96KiB及API地址 | 8行tile，三路双buffer；尾行搬入前按有效范围补零 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| 三路输入双槽、cast | UB | scratch132～188KiB | 输入24KiB，其余用于cast和计算；结果写常驻base |

S4与S5可以重叠。S5不等待Tza；rH与state_sum的chunk末行修正留到S9。

### S6 · Cube：dAkk

```text
dAkk_raw = akk.T @ Tza
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| dAkk_raw | GM | relay+160KiB | 64×64 FP32；S7取负并保留严格下三角 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| akk、Tza | L1 | 0或8/56KiB | 都在本Stage最后消费；后续释放早段L1给下一work预取 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| A/B/C双槽 | L0A、L0B、L0C | 各0/32KiB | BF16输入，FP32输出；发布ready绑定PIPE_FIX |

### S7 · Vector：逐带局部操作数

```text
Aq = tril(dAqk, -1); Ak = -tril(dAkk_raw, -1)
for b in [1,0]:
    first=32*b; last=min(first+31,validRows-1)
    center=(gk[first]+gk[last])/2
    kNeg[row] = k[row]*exp2(center-gk[row])       # row <= last
    qPos[row] = q[row]*exp2(gk[row]-center)       # row >= first
    betaKPos[row] = beta[row]*k[row]*exp2(gk[row]-center)
```

无效带不计算center，操作数填零且维持通知闭环。无效行先置安全指数再exp。原始gk用于居中，不能由e反推。

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| center | UB | 小量区center | 本片所属带128个FP32，S9消费；另一带中心只用scratch |
| diag(dAqk) | UB | 小量区diag | 从本Stage首次读取的dAqk提取32个对角元，保留到S9 |
| Aq、Ak | GM | relay+64/160KiB | 两个64×64 FP32，分别覆盖已消费Zv和raw；每片仅覆盖自己的32行 |
| kNeg/qPos/betaKPos | GM | relay+208KiB+(1-b)×96KiB+0/32/64KiB | 每带三份64×128 FP32，两片各准备32行；不同带使用不同物理槽 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| q/k/gk、bases、gate/x | UB | 固定区 | bases不参与本Stage但仍驻留；gk保留至S9恢复带内factor |
| dAkk_raw/dAqk、带边界gk | GM | relay+160KiB及API地址 | dAkk仅读本片；center读取两条完整128维边界行，额外流量明确计入 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| 输入、三路输出双槽及exp临时量 | UB | scratch132～188KiB | 8行tile，24KiB输出双槽+32KiB临时；原32行状态不复制 |

S7先发布后带，再准备前带，与S8后带重叠。AIV0完成前带写出后提前贡献suffix模式1的空payload通知。

### S8 · Cube：每带四项局部矩阵乘

```text
local_q = Aq[band,:] @ kNeg
left = Ak[band,:] @ kNeg
right = Aq[:,band].T @ qPos + Ak[:,band].T @ betaKPos
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| local_q/left/right | GM | relay+400KiB+(1-b)×48KiB+0/16/32KiB | 每项32×128 FP32；当前带owner在S9消费；其他片只消费同步token |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| Aq/Ak/kNeg/qPos/betaKPos | L1 | Aq/Ak共享176/192KiB；带槽208或304KiB | FP32共享32KiB+每带96KiB；Aq/Ak仅搬一次，保护至前带结束；每带kNeg复用两次，最后MMAD读取后释放 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| FP32 A/B双槽 | L0A、L0B | 各0/32KiB | tile=(32,128,64)；转置通过类型与layout显式区分 |
| local_q/left/right累加 | L0C | 0/32KiB | right两次MMAD累加，第二次initC=false；Fixpipe完成前不复用 |

先算后带再前带。两片都消费每带ready，以保持通知代际一致；非owner不读该带输出。不因为FP32少了残差乘法就假定更快，完整测量后再选择精度实现。

### S9 · Vector：合并与反向收尾

```text
factor=exp2(gk_owned-center)
lq=local_q*factor; l=left*factor; r=right/factor
dq=dq_base+lq+diag(dAqk)*k
dk=dk_base+beta*l+r+diag(dAqk)*q
db=db_base+sum_k(k*l)
dg=dg_base+q*lq+k*(beta*l-r)
last_row(dg) += state_sum0+state_sum1+exp2(g_last)*rH
scan=reverse_cumsum_rows(dg)
if s==0: scan += suffix_from_AIV1
gate=scan*gate_factor
partial_a=sum(gate*raw_x); partial_bias=sum_rows(gate)
apply optional q/k L2Norm backward to dq/dk
apply optional beta*(1-beta/multiplier) to db
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| suffix/partial | UB | 小量区suffix/partial | FP32，按后32行到前32行顺序；suffix是未乘gate因子的扫描和 |
| suffix与后半partial | GM | relay+496KiB起 | AIV1唯一写者，AIV0消费；suffix512B、partial544B，分别32B对齐 |
| dq/dk/db/dg | GM | 原生输出地址 | 每片独占行；dq/dk转BF16，db/dg保持FP32 |
| 完整head-chunk partial | GM | partial_base+(uint64(chunk)*H+head)×544B | AIV0合并两片后唯一写入，S10消费；136个FP32，padding置零 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| bases/q/k/gate/x/小量 | UB | 固定区 | 最后消费后分段释放；下一work不得覆盖尚未消费区域 |
| 三项局部结果、state_sum | GM | relay+400KiB带槽、152KiB | 8行tile倒序读取；仅末行owner执行chunk末行修正；空后片仍提供零suffix和零partial |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| 三路输入双槽、输出双槽、norm临时 | UB | scratch132～188KiB | 24KiB输入+8KiB BF16输出+8KiB FP32输出+16KiB临时；FP32输出槽轮流写dg/db |

前带owner等待suffix模式1聚合，后带owner写完后贡献set；两片各wait一次。AIV0读取后半partial前另确认其写出完成，不能以早发布的suffix替代后半partial就绪。第一版把suffix和后半partial一起发布，保证一个ready覆盖完整记录。

### S10 · Vector：跨chunk归并

```text
for head = vector_core; head < H; head += reduce_block_dim:
    da_log[head] = sum_chunks(partial_a[chunk,head])
    ddt_bias[head,:] = sum_chunks(partial_bias[chunk,head,:])
```

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| da_log/ddt_bias | GM | 最终输出地址 | FP32，每head唯一写者；所有chunk参与，不包含空序列 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| 完整partial | GM | partial_base | 主kernel完成后由同一stream启动归约；此前禁止释放workspace |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | 功能 |
| --- | --- | --- | --- |
| 输入双槽、FP32累加及输出 | UB | 0/544/1088B起 | 每槽136个FP32，按chunk升序累加；输出写回完成后退出 |

## 5. 用户额外 workspace

```text
metadata_bytes = AlignUp(uint64(Nc)*32,64)
relay_bytes = uint64(P)*W*524288
partial_bytes = uint64(Nc)*H*544
user_workspace_size = AlignUp(metadata_bytes+relay_bytes+partial_bytes,64)
```

| 区域 | 数量与单份大小 | 主case大小（8K） |
| --- | --- | ---: |
| chunk metadata | 128×32B | 4KiB |
| relay | P×W×512KiB；首版W=1 | 10MiB |
| chunk partial | 128×96×544B | 6.375MiB |
| 用户额外workspace | 64B对齐 | **16.37890625MiB** |

首版总量为17.174528 MB（十进制）。W=2是后续独立候选，对应26.37890625MiB。中间结果按物理槽复用，不按Nwork分配；partial跨work保留。API输入、最终输出、片上空间、固定系统workspace不计入。metadata和partial对所有变长chunk使用同一64位公式。

## 6. 跨 Stage 同步契约

模式2用于一个AIC与两个AIV的聚合/广播。每次AIC发布由两片各消费一次；每次AIV发布由两片各贡献一次，AIC等待聚合。模式1仅用于同组两片的suffix记录。不同模式使用独立flag ID，不混用未消费代际。

直调host通过 `rtGetC2cCtrlAddr` 取得同步地址，kernel入口调用 `SetSyncBaseAddr`；依据固定CATLASS示例03及kernel_adapter。只声明MIX任务类型不能替代同步地址初始化。

| 事件ID/资源 | 第一次语义 | 复用语义 | 被保护的数据 |
| --- | --- | --- | --- |
| 0 S0 ready | Fixpipe结果写完 | 每work两片各wait一次 | state/dv_b/Zv |
| 1 kE ready | 两片MTE3写完kE | 每workAIC wait一次 | kE |
| 2 S2 ready | Fixpipe写完gb/Zw | 两片各wait一次 | gb/Zw |
| 3 Zb ready | 两片MTE3写完 | AIC wait一次 | Zb |
| 4 S6 ready | Fixpipe写完dAkk | 两片各wait一次 | dAkk_raw |
| 5/6 带操作数ready | 后/前带全部写完 | AIC按后再前wait | Aq/Ak及指数操作数 |
| 7/8 带结果ready | 后/前带Fixpipe写完 | 两片各按后再前wait | 三项局部结果 |
| 9 slot free | 首版W=1，初始隐式空闲 | AIC每work末等待两片S9结束；ID10保留给W=2候选 | 整代relay和下代常驻状态 |
| 11 suffix记录ready | AIV0提前空贡献；AIV1完整记录MTE3写完再贡献 | 两片均wait，释放前不允许同slot覆盖 | suffix和后半partial |
| 本地HardEvent | 初始化可用槽位 | MTE2/V/MTE3、MTE2/MTE1/M/FIX成对set/wait | UB/L1/L0各自双buffer |

执行顺序：

```text
AIC: slot free → S0 → wait kE → S2 → wait Zb → S4 → S6
     → wait band1 → S8(band1) → wait band0 → S8(band0)
AIV: S1 → wait S0/S2 → S3 → S5 → wait S6 → S7(band1,band0)
     → 两片消费band1 ready，AIV1执行S9
     → 两片消费band0 ready，AIV0执行S9
     → 各自MTE3结束后贡献slot free
```

实现不得把“AIV0等待前带结果”写成阻止AIV1后带S9推进的全局barrier。S9内部suffix等待按对应数据依赖执行。W=2允许下一work的S0使用另一relay槽，但其L1目标必须已经结束旧代消费；每个AIV只有一份完整UB状态。退出时AIC消费尚未回收的slot free，核内双buffer事件排空后释放。空片仍参加全部set/wait；无效带写安全零并闭环，不能跳过一侧通知。

P3已验收：S9逆序scan直接写回dG。最后有效行先加远端suffix，前面的行依次加已完成的下一行；结束后仅复制首行到suffix。保持串行加法顺序，不改跨AIV通知、UB地址和gate计算。空片保留原suffix，不访问dG首行。
