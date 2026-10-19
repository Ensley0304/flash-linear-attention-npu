# 当前检查点：P4已验收

P4 suffix/partial分开发布已保留：六个边界冒烟及原20case七输出通过；两轮八配置平均降时1.90%，单轮最小1.66%。8K无附加反向17666.728→17311.589us。证据见evidence/p4-*，未改变浮点运算顺序和精度阈值。

- 分支：kda-finalize-a2-cannbot；原基线kda-finalize-a2及baseline/kda-finalize-a2-20261019保留。
- A2工作目录：/tmp/wys/kda-finalize-a2-cannbot-20261019。/data空间紧张，不向原目录添加大文件。
- bin/finalize-p4：SHA256=69e94546cc88c5cf9c8a2245d83c93630515f1945e0ef71e0ddb7338ed426836；bin/finalize-p1/p2/p3及initial也保留。
- 构建：source-r1/operators/chunk_kda_bwd_finalize_catlass，build-r1，Release/dav2201；设备3。
- CANN：/data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh。
- Python：/data/wys/kda-finalize-a2-stage9-20261012/atk-finalize/venv/bin/python；禁用torch后端自动加载，OMP/MKL/OPENBLAS均设1。
- 原始性能输入、CPU缓存、CATLASS及固定CANNBot仍在/data/wys/kda-finalize-a2-cannbot-20261019。
- 每项先六个冒烟，再八配置两轮交替A/B；稳定收益后原20case、七输出全部通过才保留。最终原200case×三seed及功能门禁仍未执行。

下一项：定长chunk元数据由编号推导，去掉热循环GM逐项读取；变长维持原路径。之后评估三角mask批量选择、无效区指数裁剪、S7复用e扩大tile及参数归约。

工作流仍为implementation/precision_debug。S7/S8原始观察量比较存在失败，最终输出通过不能将这些记录改为PASS。最终9.604ms目标及full验收未完成，不接入原公开入口。模拟器在设备初始化报507033，未生成有效指令时间线；性能结论仅基于硬件msprof。

本轮仅删除已授权两处性能目录的35个output_*.bin，释放5049141120字节。其他删除仍需确认。P2 case192首轮host60秒超时原因未确定，随后独立重试和完整精度通过，失败记录保留。历史细节见validation.md及Git记录。
