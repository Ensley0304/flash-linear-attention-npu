# BF16 raw_g 开发验证

- 基线：`7e39f9668eed285fc91cde0502f46cd48686a681`。
- 范围：A5 V2 raw_g 的独立 Cast；A_log 转换保留。公开 V2 签名及输出不变，
  Finalize 注册增加 raw_g BF16 类型组合；CPU 数学语义与输入量化值不变。
- 工程参考：现有 Finalize Stage11 与 Recompute GateType 分支，
  `CHUNK_INDEPENDENT V1` 仅用于核对原有 Stage11 资源与事件闭环。
- 比较：同一份 BF16 raw_g 与其精确 FP32 展开，逐输出检查有限值及逐位一致。
  覆盖 saved/recompute、dense/packed、rstd 有/无、beta BF16/FP32，各 3 个种子。
- 模型 case：B=1、H=96、T=8192、K=V=128、chunk=64、raw_g BF16。
  用户 profile 中 Cast=191.593 us；该数字不是本次环境的可比基线。
- 目标：raw_g 不再产生独立 Cast kernel；不把整个 Cast 时长当作净收益。
  同环境性能与正式 ATK 验收尚待完成，回归脚本不替代独立 CPU 标杆验收。

| 设计项 | 实现 | 验证状态 |
| --- | --- | --- |
| 8 组 raw_g/beta/A_log dtype 组合 | Finalize op def | 待 A5 构建 |
| Stage11 BF16 加载、寄存器扩展 | Finalize gate/vector | 待同值 NPU 回归 |
| raw_g 直传 | BwdV2 L2 | 待 NPU profile |
| FP32 兼容 | 保留 FP32 编译分支 | 待同值 NPU 回归 |

回退：恢复 V2 的 raw_g ToFloat32 及 Finalize 原 FP32 注册/读取实现。
测试代码：`torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_raw_g.py`。
