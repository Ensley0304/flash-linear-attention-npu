# 当前检查点：P2已验收

P2 对角、norm和db批量处理：八配置两轮交替A/B均有收益，降时6.55%～9.87%；8K无附加反向19459.909→18184.937us。原20case七输出全部通过CT L1和冻结附加策略，保留P2。case192首轮host进程60秒超时，原因未确定；同一输入独立重试18.21秒正常退出，随后192/199完整精度通过。失败日志保留，没有放宽超时或精度阈值。证据见evidence/p2-*。

/data空间紧张，后续工作在/tmp/wys/kda-finalize-a2-cannbot-20261019；bin/finalize-p2为已验收二进制，SHA256=d28889fdd6be7122c3a2bae6f1447a3643293e6cd886d29d4c643405515719af。原始输入、CPU缓存、CATLASS仍在/data/wys原目录。下一项P3原位scan。

以下为此前记录，路径与版本以本段为准。

# 恢复位置：2026-10-19

最新：P1 beta批量广播已通过原20case七输出、八配置两轮交替A/B。8K无附加反向21061.379→19452.631us，八配置平均降时7.33%，目标9604.066us尚未达到。报告见`evidence/p1-precision20.json`和`p1-eight-*.json`。A2根目录仍为`/data/wys/kda-finalize-a2-cannbot-20261019`，当前`build-r1/finalize`为P1，初始完整版本保存为`build-r1/finalize-initial`。下一项从P2批量对角/norm/db处理继续，不能叠加未验收的候选。

用户已授权并完成删除两处旧性能目录中的35个`output_*.bin`，释放约4.7GiB；其他文件未删除。CPU标杆缓存为`reference-cache`，测试复用`precision-work`，性能复用`perf-inputs`；新的删除仍需按服务器规则确认。最后一组A/B曾在设备初始化报507033，重试成功，失败日志保留。

基线仍在原分支`kda-finalize-a2`，标签`baseline/kda-finalize-a2-20261019`。本候选在`kda-finalize-a2-cannbot`，不替换原入口。CANNBot仍处于`implementation / precision_debug`，没有进入complete。

## 已确认

- S0～S6：7个行边界case、21项中间量，原CT L1和附加检查通过，代码检查点`cfae5960`。
- S0～S9单head、单chunk：1/7/31/32/33/63/64行的七个最终输出全部通过。包含norm与beta反向。见`evidence/full-progress.json`。
- 完整主核加S10归约：强制单核6个冒烟case通过，包含多work回绕、四种输出布局、奇数head、变长空序列。远端完整记录为`reports/full-r2/report.json`，断线前尚未下载。
- 多核重复3次：前4个case通过，变长case4失败。第二轮20次重复：case4通过、case5失败。不能把单核通过外推为多核通过。

## 当前问题与修复

`reports/full-r5`抓取证明首个偏差在S3 Zb：work6/7/8的第48～62行被覆盖；S0、kE、dKgb、Zw均正常，后续Ak随之出错。

Zb的两个输出槽是UB152～156KiB。后续dv尾块输入的Vector清零复用这些地址。已有MTE3→MTE2等待只约束DMA，不能保护Vector写入。最新本地修复在S3切换到dv前，额外等待两个`MTE3_V outputFree`，再恢复free token。

**A2连接恢复后，此修复已上传、编译并通过复测。** 本地与远端`stage1.h`的SHA256一致。`full-r6`的变长case4/5通过；`full-r7`全部6个冒烟case通过。两轮均请求20核、每case连续执行20次，检查最后一次运行的七个输出，原CT及附加检查均通过；这不等同于逐次检查或全量验收。报告已下载至本地`validation/workflow-review-20261019/full-r6.json`和`full-r7.json`。

另有S7/S8比较问题未关闭：缩放后的内部量可达大数量级，沿用最终FP32输出绝对误差上限时失败，部分local_q/指数操作数CT比值也失败。原始报告保留，不能据最终小case通过将这些记录改成PASS。下一步应按工作流检查内部观察量的值域与策略适用范围；最终七输出的原CT及既有策略保持不变。

## 服务器与命令

仅使用`a2-server`、设备3；先确认设备仍空闲。根目录：

```text
/data/wys/kda-finalize-a2-cannbot-20261019
source-r1/operators/chunk_kda_bwd_finalize_catlass  当前部署源
build-r1/finalize                               无抓取完整入口
build-r1/finalize_stage0                        阶段观察入口
reports/full-r5/case-5/debug_relay.bin           失败现场
```

加载CANN：`/data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh`。
Python：`/data/wys/kda-finalize-a2-stage9-20261012/atk-finalize/venv/bin/python`。
CPU标杆需设`TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。

修复上传并重新构建后，先执行`check_full.py --cores 20 --repeat 20 --smoke-ids 4,5`。保留`KDA_DUMP_RELAY=1`可抓取每核relay，`inspect_relay.py`只用于这个固定case5的定位。通过后再跑强制单核和多核完整冒烟、原20case；之后才扩大到模型性能和全量验收。

剩余工作：clang-format及源码审查；观察点增补后的CPU一致性；host非法输入；模型精度；八配置两轮A/B及msprof；W=1/W=2等候选独立比较；最终原200case×3轮；工作流complete后仓库适配与ATK。尚无新性能结论。
