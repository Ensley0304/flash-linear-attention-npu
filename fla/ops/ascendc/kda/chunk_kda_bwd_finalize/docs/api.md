# S24 私有验证接口

本PR新增 `torch.ops.ascend_ops.kda_finalize_stage01`，未接入或替换主分支已有的公开A2/A5接口。完整调用使用 `last_stage=12`；所有tensor连续并位于同一NPU。

| 参数 | dtype / shape |
|---|---|
| q/k/v/v_new/dv_scan | BF16，定长 `[B,H,T,128]`，变长 `[H,T,128]` |
| gk/raw_g/dq_raw | FP32，与k同shape |
| akk / dAqk | BF16 / FP32，与k相同token布局，最后一维64 |
| h/dh | BF16，定长 `[B,Nc,H,128,128]`，变长 `[Nc,H,128,128]` |
| beta | FP32前向有效值，`[B,H,T]` / `[H,T]` |
| a_log / dt_bias | FP32 `[H]` / `[H,128]` |
| q_rstd/k_rstd | 可选FP32，与q/k去掉末维后的shape相同；完整入口成对提供 |
| cu_seqlens | host int数组；定长为空，变长从0到T非递减，允许空序列 |

K=V=128，chunk=64；Nc为每序列chunk数，变长状态使用总chunk数。`scale`须有限，`lower_bound`属于[-5,0)，`core_limit=0`使用设备可用核数。

beta_mode=0直接输出有效beta梯度；1乘 `beta*(1-beta)`；2乘 `beta*(1-beta/2)`。融合反向依赖保存的FP32有效beta，不能用已舍入为BF16的饱和值替代。

输出由调用方分配并原地写入：

| 输出 | dtype / layout |
|---|---|
| dq_out/dk_out/dv_out | BF16，output_layout指定BNSD/BSND或NTD/TND；空字符串随输入 |
| dg_out/db_out | FP32，固定 `[B,T,H,128]` / `[T,H,128]`；db去掉末维 |
| partial_out | FP32 `[H,total_chunks,136]` |
| da_log_out/ddt_bias_out | FP32 `[H]` / `[H,128]` |

Python返回值是 `(capture, trace, workspace)` 三个开发辅助tensor，七个梯度从上述输出参数读取。Stage0～11前缀保留开发采数用途；Stage12完整生产路径采用单次launch。测试工程按 `dq, dk, dv, db, dg, da_log, ddt_bias` 比较七输出。

私有schema及参数校验以 `examples/fast_kernel_launch_example/csrc/kda_finalize_stage01/ascend910b/kda_finalize_stage01.cpp` 为准。完整200case、A3/A5、公开接口迁移及端到端验收均不在本次已完成声明中。
