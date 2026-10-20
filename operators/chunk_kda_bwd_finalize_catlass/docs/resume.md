# 当前检查点：P5f已验收

P5f S9采用16行tile已验收：普通及单核多work各六例20次、原20case七输出全部通过。设备4八配置两轮交替A/B平均降时5.07%，单轮最小3.60%；8K无附加反向14725.871→14164.177us。输入和输出均保留双buffer，缩放因子全部消费后复用g/e，原运算次序及精度阈值不变。证据见evidence/p5f-*。运行目录已迁至/data/wys/kda-finalize-a2-cannbot-20261019/run-r2；此前/tmp/wys消失，原因未知，已从c55a2fb7重建基线，hash与原验收完全一致且六例冒烟通过，所有本项性能重新采集。

当前远端根目录/data/wys/kda-finalize-a2-cannbot-20261019/run-r2，基线bin/finalize-p5f，hash862e8c6177c64f194203a6981126a6db6d8546f5cedc816929c1dc2bf380fec8。下一项P5e完整64行chunk模板路径，见full-chunk-design.md；尚未实施。原DATA参考/输入仍在，旧/tmp已不存在。最终目标9.604ms及全量验收均未完成。

以下为历史，以以上为准。

run-r2恢复成功：重建P5d六例20次冒烟通过，hash与原验收34131cf...完全一致。P5f普通及单核六例20次冒烟均通过，报告已下载；正在run-r2/reports/p5f-eight采集，profile PID787382。恢复驱动765179、candidate驱动776620，日志recovery-driver.log。门禁通过后自动20case；未提交，不重复启动。

重要恢复状态：2026-10-20 /tmp/wys整个目录在本轮操作期间消失，原因未知；此前P5f启动未发生。已迁移到/data/wys/kda-finalize-a2-cannbot-20261019/run-r2持久目录。source-baseline为git c55a2fb7导出的kernel/host/CMake；source-r1在此基础上替换P5f stage1.h。recover_p5f.py已nohup启动：配置两套Release/dav2201构建→重建P5d→六例20次基线冒烟→run_candidate.py p5f finalize-p5d --single-core-smoke --device 4。新日志run-r2/reports/recovery-driver.log和recovery-build.log；不要找旧/tmp进程，也不要重复启动。新run_candidate/profile/gate脚本根路径均为run-r2。原DATA输入/reference-cache和本地Git报告保留；旧/tmp的二进制及原始profiling不再可用，性能必须重采。P5f未提交。

当前P5f候选已实现并上传：S9 16行tile，设计见s9-tile-design.md。启动run_candidate.py p5f finalize-p5d --single-core-smoke --device 4；日志reports/p5f-driver.log。按构建、普通六例、单核六例、八配置A/B、原20case顺序验证；未提交。q/k输出双槽各8KiB位于132和148KiB，避免packedBeta144KiB重叠；beta/diag仍186/187KiB。启动前设备4空闲、根分区26GiB。不要重复启动。

# 当前检查点：P5d已验收

P5d rH两片各64行已验收：S1七个边界中间量、普通及单核多work各六个冒烟（每例20次）、原20case七输出均通过。设备4八配置两轮交替A/B平均降时3.41%，单轮最小1.84%；8K无附加反向15361.780→14753.984us。每行FP32归约次序不变，workspace总量不变；原FP32标杆、CT和附加精度策略不变。证据见evidence/p5d-*。

基线bin/finalize-p5d，hash=34131cf653bd9e208cd50b1ee2d606e5d0d4a2a7275c4477b6c9dd96fb5243fe。下一项P5f扩大S9 tile，见s9-tile-design.md，尚未实施。服务器设备4此前空闲，启动前仍需复查。最终目标9.604ms、200case三seed及工作流完整验收均未完成。

以下为历史检查点，以以上为准。

2026-10-20设备4恢复空闲，已启动run_p5d.py。S1七个边界case、六个普通冒烟及六个单核冒烟（各20次）均通过，报告已下载validation。候选hash34131cf653bd9e208cd50b1ee2d606e5d0d4a2a7275c4477b6c9dd96fb5243fe；正在p5d-eight八配置两轮A/B，驱动PID430407，profile PID452914。门禁通过后自动原20case；未提交。当前状态以此为准。

P5b已提交c123f8af。当前P5d候选（rH双AIV分担）已修改stage1.h并上传，build-r1的finalize_stage0和finalize都构建通过，日志reports/p5d-build.log；尚未复制到bin/finalize，也未设备验证。所有卡存在其他ATK PID2163022/2163023，未启动设备工作。恢复后先检查空闲卡，跑新build-r1/finalize_stage0的check_stage0.py --prefix 1（device按实际空闲卡），再run_candidate.py p5d finalize-p5b --single-core-smoke --device D，按八配置/20case门禁。不要使用DATA下旧kernel构建结果。S1测试用DATA/source-r1的原check_stage0.py和固定reference，适用新snapshot布局。

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
