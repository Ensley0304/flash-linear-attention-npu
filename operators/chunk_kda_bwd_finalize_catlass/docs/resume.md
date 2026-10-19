# 当前检查点：P5b已验收

P5b S7采用16行tile已验收：六个冒烟每例20次及原20case七输出全部通过；设备4八配置两轮交替A/B平均降时1.03%，单轮最小0.35%；8K无附加反向15557.913→15469.834us。复用e区，输出双buffer，UB总量188KiB不变；原FP32标杆和精度阈值不变。证据见evidence/p5b-*。

基线bin/finalize-p5b，SHA256=2d23d4541fd94f326f2987f641621ea135fd24ae6d2d79704b751ad7cd27eacb。下一项P5d rH两片分担，见rh-split-design.md；需先验证S1中间量及单核多work，然后原八配置/20case门禁。尚未修改rH代码。

以下为历史检查点，以上述为准。

P5已提交4431b553并推送。当前P5b候选已上传并启动run_candidate.py p5b finalize-p5 --device 4；日志reports/p5b-driver.log。修改stage1.h中S7的tile和UB别名，设计见p5b-design.md；尚未通过构建/验收。设备4启动前空闲，根分区16GiB。不要重复启动。

# 当前检查点：P5已验收

P5有效区指数裁剪已验收：设备4六个冒烟每例20次通过，八配置两轮交替A/B平均降时1.29%，单轮最小0.21%；8K无附加反向15654.978→15534.413us。原20case七输出全部通过，FP32标杆和精度阈值不变。设备3首次冒烟case1曾host60秒超时，原因未确定；后续设备3被其他任务占用，改为设备4对两份相同hash二进制验收。采集中根分区满导致一条报告缺失，保留失败日志后续跑，仅统计32条完整记录；没有删除文件。证据见evidence/p5-*。

下一项P5b：按p5b-design.md将S7的tile扩大到16行，保持双buffer及浮点次序。基线bin/finalize-p5，hash=93c37581548c6aa7799451f00f782625541bb1946421c6dc1f17b19b967c32a5。设备4用于本轮验证，启动前仍需检查占用。

以下为历史检查点，当前状态以上述为准。

# 当前检查点：P4d已验收，P5测试已启动

P4d Stage0提前一个work已验收：常规六例、单核多work六例均连续20次通过；八配置两轮交替A/B平均降时4.93%，8K无附加反向16388.544→15713.235us；原20case七输出全部通过，精度阈值不变。证据见evidence/p4d-*。
候选已保存为bin/finalize-p4d，SHA256=967163b0be323e8740647d7c52f0573cff78a0c00f5922ba1f8f799945c9123d。

P4c批量三角mask已保留：六个边界冒烟及原20case七输出通过；两轮八配置平均降时4.98%。8K无附加反向17314.034→16424.155us。证据见evidence/p4c-*；只改变掩码执行方式，原精度条件不变。

- 分支：kda-finalize-a2-cannbot；原基线kda-finalize-a2及baseline/kda-finalize-a2-20261019保留。
- A2工作目录：/tmp/wys/kda-finalize-a2-cannbot-20261019。/data空间紧张，不向原目录添加大文件。
- bin/finalize-p4c：SHA256=ef5826c23f8397a24cbcfecdb8fd9af0f11ded3834b275591aadd1462f675365；bin/finalize-p1/p2/p3及initial也保留。
- 构建：source-r1/operators/chunk_kda_bwd_finalize_catlass，build-r1，Release/dav2201；设备3。
- CANN：/data/wys/Ascend/cann_9.1.0_910b_0605/cann/set_env.sh。
- Python：/data/wys/kda-finalize-a2-stage9-20261012/atk-finalize/venv/bin/python；禁用torch后端自动加载，OMP/MKL/OPENBLAS均设1。
- 原始性能输入、CPU缓存、CATLASS及固定CANNBot仍在/data/wys/kda-finalize-a2-cannbot-20261019。
- 每项先六个冒烟，再八配置两轮交替A/B；稳定收益后原20case、七输出全部通过才保留。最终原200case×三seed及功能门禁仍未执行。

下一项P5：S7仅计算有效行的指数，保持有效行运算顺序。P4b定长元数据已因平均慢0.44%撤销。后续还有无效区指数裁剪、S7复用e扩大tile、rH两片分担和参数归约。

工作流仍为implementation/precision_debug。S7/S8原始观察量比较存在失败，最终输出通过不能将这些记录改为PASS。最终9.604ms目标及full验收未完成，不接入原公开入口。模拟器在设备初始化报507033，未生成有效指令时间线；性能结论仅基于硬件msprof。

本轮仅删除已授权两处性能目录的35个output_*.bin，释放5049141120字节。其他删除仍需确认。P2 case192首轮host60秒超时原因未确定，随后独立重试和完整精度通过，失败记录保留。历史细节见validation.md及Git记录。

P5候选已上传stage1.h及design.md，并通过nohup启动run_candidate.py p5 finalize-p4d。启动时确认PID1667985存活；后续SSH连续超时，尚未确认构建/测试结果，不应重新启动。查看/tmp/wys/kda-finalize-a2-cannbot-20261019/reports/p5-driver.log及对应进程。离线穷举rows=1..64、sub/band/tile，新的有效前缀与旧逐行mask完全一致；这不替代设备精度验收。P5尚未提交，若无稳定收益应恢复61324ef2的stage1.h。

P5恢复检查：原driver已退出，设备3首例通过，第二例host60秒超时；原因未确定。连接恢复时设备3有其他ATK，因此同一二进制改在空闲设备4跑六个冒烟，每例20次全部通过，hash=93c37581548c6aa7799451f00f782625541bb1946421c6dc1f17b19b967c32a5。已启动run_p5_device4.py，顺序八配置A/B（P4d对P5，两者均设备4）及性能门禁通过后的原20case。日志reports/p5-driver-device4.log，结果p5-eight-device4、p5-precision20-device4；不要重复启动。profile_eight和gate_and_check20新增--device并校验resume设备一致，默认仍3。旧失败日志保留。

P5设备4采集已终止：根分区/tmp所在卷剩余0，r0-t16384-n1-b1-A的msprof未导出完整CSV。summary含14条有效记录，已下载p5-eight-device4-progress.json；原driver/profile PID1776803/1776808已消失。尚无完整A/B结论。已请求用户授权删除/tmp/wys/kda-finalize-a2-cannbot-20261019/precision-work约4.1GiB（只临时数据，报告保留），未获准前不删。恢复空间后profile_eight.py使用原label p5-eight-device4加--resume、--device 4，继续缺失记录；失败目录保留，自动使用retry新目录。之后同label性能门禁及20case，禁止从头覆盖报告。

后续复查：根分区自行恢复16GiB可用，未删除任何目录，无需等待删除授权即可继续。已核对P5/P4d二进制hash不变、设备4空闲，启动resume_p5_device4.py；读取原14条记录，profile_eight使用--resume继续缺失项。新日志p5-driver-device4-resume.log、p5-eight-device4-resume.log，之后precision20同样使用设备4。原失败日志未覆盖。
