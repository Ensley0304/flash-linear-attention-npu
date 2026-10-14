# BF16 raw_g 开发验证

- 基线：`7e39f9668eed285fc91cde0502f46cd48686a681`。
- 范围：A5 V2 raw_g 的独立 Cast；A_log 转换保留。公开 V2 签名及输出不变，
  Finalize 注册增加 raw_g BF16 类型组合；CPU 数学语义与输入量化值不变。
- 工程参考：现有 Finalize Stage11 与 Recompute GateType 分支，
  `CHUNK_INDEPENDENT V1` 仅用于核对原有 Stage11 资源与事件闭环。
- 比较：同一份 BF16 raw_g 与其精确 FP32 展开，逐输出检查有限值及逐位一致。
  覆盖 saved/recompute、dense/packed、rstd 有/无、beta BF16/FP32，各 3 个种子。
  前向缓存由仓库既有 `_chunked_reference` CPU 标杆生成，h 转回 K-first；
  测试不依赖前向 NPU 包。另有 4 组直接 Finalize dtype 配对测试。
- 模型 case：B=1、H=96、T=8192、K=V=128、chunk=64、raw_g BF16。
  用户 profile 中 Cast=191.593 us；该数字不是本次环境的可比基线。
- 目标：raw_g 不再产生独立 Cast kernel；不把整个 Cast 时长当作净收益。
  同环境性能与正式 ATK 验收尚待完成，回归脚本不替代独立 CPU 标杆验收。

| 设计项 | 实现 | 验证状态 |
| --- | --- | --- |
| 8 组 raw_g/beta/A_log dtype 组合 | Finalize op def | A5 构建通过，4 组 dtype 配对测试覆盖全部 8 个组合 |
| Stage11 BF16 加载、寄存器扩展 | Finalize gate/vector | 4 组独立 Finalize 与 48 组 V2 NPU 配对测试全部输出逐元素一致 |
| raw_g 直传 | BwdV2 L2 | 8K 同 shape trace：两次调用各 5 个 kernel，无 V2 Cast |
| FP32 兼容 | 保留 FP32 编译分支 | 独立 Finalize 与 V2 配对测试通过 |

2026-10-14：私人 A5，`wys_gdn`，显式 source 已安装的 CANN 9.1.0-beta.1
（全局 cann 链接指向缺少 libopapi.so 的 9.2 beta）。
`bash build.sh --pkg --soc=ascend950 --ops=chunk_kda_bwd --vendor_name=fla_npu`
退出码 0，安装后 opapi MD5 一致。`python -m pytest -xq
torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_raw_g.py -k finalize`：4 passed。
产物与日志保存在 A5 `/home/Ensley/workspace/kda-bf16-rawg-20261014/`。

完整回归：52 passed（120.66 s），覆盖 Finalize 8 个 dtype 组合和 V2 的
saved/recompute、dense/packed、rstd、beta dtype 与 3 个种子。
已安装 wheel 的旧 `_kda_policy.py` 仍禁止重算尾块，因此测试进程通过
`run_current_kda_policy.py` 加载本次基线仓库的同名模块，再执行 pytest；
未修改环境中的旧校验文件。测试直接使用当前主干已支持的尾块场景。
初次反向包的 opapi MD5 为 `02a299c02960fbdec89e08a640365f9b`，
安装后与原构建一致；8 个 Finalize `.o` 的 SHA256 均与构建目录一致。
额外发起的未修改前向依赖构建已停止，未安装该未完成产物。

模型同 shape 检查：`run_current_kda_policy.py --profile` 调用回归 helper，
H=96、T=8192、beta FP32、raw_g BF16/同值 FP32，seed=7；
CPU 标杆准备缓存，NPU 执行完整 recompute 反向，两次所有输出逐元素完全一致。
这是合成输入的同 shape 回归，不是用户模型输入 dump。
`profile-rawg-model/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中恰有
10 条 `aclnnChunkKdaBwdV2_` 记录，两次调用均为 Recompute、FwdH、Prepare、Dhu、Finalize。
没有 V2 Cast；唯一 Cast 为两次调用之间测试显式 `raw.float()` 产生的
`aclnnInplaceCopy_CastAiCore_Cast`。分析记录在 `profile-rawg-summary.json`。
profiler 提示 RECORD 状态停止；已核对两次完整反向的 10 个预期 kernel 均在导出结果中。
此次采集仅验证任务组成，没有预热/重复测量，不作为性能达标或净收益证据。

回退：恢复 V2 的 raw_g ToFloat32 及 Finalize 原 FP32 注册/读取实现。
测试代码：`torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_raw_g.py`。
