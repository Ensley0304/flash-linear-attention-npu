# Stage 11 Gate 与扫描交错实验

候选 S6，独立基于已验收的原扫描结构，不叠加未验收的 S5。只改 COMPACT 前带（完整 chunk、非 capture、无 norm）。原 raw_g 已在 dk 计算中提前搬入，Gate 导数也不依赖 dG；尝试把其独立运算放进原扫描步骤之间，共用现有 PipeBarrier。

| 原扫描步骤 | 同时发出的 Gate 运算 |
| --- | --- |
| 高八行 step=1 | raw += bias |
| 高八行 step=2 | temp = raw × expA |
| 高八行 step=4 | temp = abs(temp) |
| 高八行加 suffix | temp *= -1 |
| 更新高八行 suffix | temp = exp(temp) |
| 低八行 step=1 | denom = temp + 1 |
| 低八行 step=2 | denom *= denom |
| 低八行 step=4 | temp /= denom |

每行两类运算后仍执行原扫描 barrier。Gate 链内部顺序和每个扫描组的加法树、suffix 顺序不变；最终 Gate 乘法与参数归约位置不变。原独立 Gate 链的七个 barrier 被合并，每 chunk 前带两个 tile 共少 14 次；不减少算术量。

raw=MTE2→left[88,96)/[120,128)，导数=temp=right[96,104)/[128,136)，扫描临时区=key[152,160)，denom=product[184,192)，互不重叠。这些地址均在 dk 形成后结束旧用途。MTE2_V 的等待前移至扫描开始，只消费同一个通知一次；其它事件、双槽、UB 192 KiB、L1/L0/workspace 不变。代价是可能过早等待 raw_g，以及更复杂的指令排程，必须实测。

九边界七输出→八配置两轮 A/B；无 norm 四配置每轮 >0.3%，norm 主核指令不变且每轮回退不超过 0.3%→原 20 case、140 输出全过才保留。标杆、种子、CT L1 不变，不采用近似 Gate 公式。失败恢复原验收版。

## 实测：不保留

基于 S3，九边界七输出通过。设备 4 两轮 A/B 的无 norm 收益只有 0.069%～0.219%，低于 0.3% 门槛；norm 主核指令与 S3 相同，耗时波动不超过 0.1%。没有进入原 20 case，也未提交实现。源码和两处运行库均已恢复 S3，并校验库哈希。

库 SHA256 `9d8d06610897ba7e622f7f848776f24b95cf4418ee7bbbcca6d92d3974ea4aac`。摘要见 `hybrid-evidence/s6-*.json`，原始 CSV/源码/日志为工作区 `validation/compact-stage11-20261022/s6-evidence.tar.gz`，已逐文件校验哈希。
