# compact 布局的 Stage 9→11 dG 直用

候选 S4，只在 S1b 验收后测试。附件的一项优势是 dG 不往返 GM；原先 H20/H21 采用额外复制的驻留方案曾回退。本次利用新 compact 16 行布局，直接保留 Stage 9 的两个 dg 输入/输出平面，Stage 11 改变其它平面地址来避开它们，**不增加 UB 复制**。

仅完整 chunk、非 capture、无 norm 的前一带 start=32 启用。Stage 9 仍读取 Stage 5 的 dG base 并执行原运算，但不将更新后的 dG 写回 workspace；Stage 11 COMPACT 路径直接读取这些 UB 值。其余带、norm、尾块和 capture 保持原实现。

每 AIV 的地址如下，单位 KiB，每个 FP32 平面都是 16×128×4=8 KiB：

| 数据 | 行 48:64，先计算 | 行 32:48，后计算 |
| --- | --- | --- |
| dg，从 Stage 9 保留 | [128,136) | [104,112) |
| left，之后复用 raw_g | [88,96) | [120,128) |
| right，之后复用 q/导数/BF16 dk | [96,104) | [136,144) |
| dk，FP32 直到最后 Cast | [112,120) | [144,152) |

两组平面互不重叠。key/扫描区[152,160)、band 指数[160,176)、beta/小量[176,184)、product/分母[184,192) 均沿用 S1b；q/k/gk[0,64) 和 mask[64,80) 不动。仍为双输入/输出组，总 UB 192 KiB，无新增 L1/L0/workspace。

Stage 9 原入口、出口的事件不变；Stage 11 仍先等 inputFree/outputFree，再 V→MTE2 桥接。新的 left/dk/right 搬运必须避开两个 dg 平面；LoadStage11Tile 在 COMPACT 下跳过 dg 搬运，right、dk、dg 地址均显式选槽，不再假定四平面连续。raw_g 提前搬运、扫描、Gate 和写回顺序保持 S1b；每带末尾仍等两组 MTE3 读完才允许下一带覆盖。

每 chunk/head 少 16 KiB 写＋16 KiB 读，原始模型共少 384 MiB GM，无额外复制。矩阵乘、扫描加法树、参数归约及舍入不变。收益须实测；不能沿用附件的整算子收益比例。

九个边界七输出→八配置两轮交替 A/B。无 norm 四配置每轮 >0.3%，norm 主核指令不变且每轮回退不超过 0.3%→原 20 case、140 输出全过才保留。原 FP32/FP64 标杆、种子、CT L1 不变。失败归档并恢复验收版本。

## 实测：不保留

九边界七输出通过。设备 4 八配置两轮 A/B 未通过专项门禁；目标 case 两轮分别 -0.119%/+0.052%，同轮 S1b 10.1230 ms，S4 10.1264 ms。norm 主核指令不变。目标 MTE3 忙时约 2.370→2.327 ms，但总耗时没有稳定下降；减少中间流量并不等于缩短当前关键路径。

未跑原 20 case、未提交实现；已恢复 S1b 后再开展其它候选。库 SHA256 `f2a5778ebf0cc41b89174b90b246f3cad833cc8138f11eede1111fefe26b385c`。摘要见 `hybrid-evidence/s4-*.json`，原始 CSV、源码和日志为工作区 `validation/compact-stage11-20261022/s4-evidence.tar.gz`，已按清单核对哈希。
