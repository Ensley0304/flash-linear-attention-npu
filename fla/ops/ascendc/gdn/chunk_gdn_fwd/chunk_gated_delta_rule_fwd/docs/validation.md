# 9.1 megakernel 移植验证记录

状态：24份必要实现已完成移植，A2/A3目标编译与A2 public合同回归通过；补充精度、性能和MSS正在执行。
完整门禁尚未齐全，保持Draft；A3编译不代替实机验证。

- 基线：`36e67d2decea70745554f7aebf9a6a29e9168a91`。
- 固定来源：`95004a10a0176113348c50e33ce0a98106eb5b49`。
- 主线来源与本发布分支结果严格分开；未获得发布分支 A3 原始运行数据。
- 原 CPU recurrence、500条冻结矩阵、输入值域和精度阈值不变。
- 必需依赖：算子私有 arch22 kernel18、arch22 tiling2、public host/私有 L0/opdef4文件。
  公共组件 blob 与来源相同，不移植 arch35/internal operators 或其他算子。
- 单独的 gate FP32 依赖与 PR #887 对应，保留其 A5 原低精度 Prepare 行为。

| 验证项 | 入口 | 当前状态 |
| --- | --- | --- |
| public ABI 与既有 A5 调用合同 | aclnn_abi_contract.py | 29参数及源码映射检查通过，非A5实机结论 |
| A2/A3/A5 目标编译 | build.sh 单算子含原依赖 | A2/A3 exit0；A5未执行 |
| 既有精度矩阵 | 原 ATK generalized500 与 smoke | 正式ATK最低版本门禁未满足；兼容补充待执行 |
| native layout、offset、非连续 view、低精度 gate | public API 同输入回归 | A2三种子108等价+24空序列明确拒绝通过，15参考合同通过 |
| 五 shape 同输入 kernel profiling | msprof，ABBA | 正在执行；失败记录不补wall时间 |
| 局部内存检测 | CANN memcheck包裹public回归 | 正在执行 |
| A5 原 Prepare/H/O 扩展 | 原有测试与 CODEOWNER/CI | 待执行 |
| 全量 wheel/OPP、Example/ST、NPU CI | 仓库 Admin CI | 未执行 |

完整原始日志、输入、版本收据、device 快照和 manifest 存于独立旁车，公开 PR 仅附去除私有路径的摘要。
只有门禁实际齐全后才归档完整验收；本记录不能当作通过证据。

目标包构建源码为0dbf1834，后续仅测试/docs变化，全部host/kernel/Python wrapper的Git diff为空。
A2/A3包SHA256分别为be37881afd9259910c960145ed16a0188a25c6287d41a780976a7fbcbd4ba670和
0c79008cba9cbe1ecf0dfb6d00907f771b097a7c04583b9c558274f439250ea6。
功能回归使用隔离Python/OPP和公开ctypes入口，未宣称完整wheel/Stable-ABI launcher验收。

空序列首次被误作支持输入，实机WU tiling拒绝，原失败报告保留。发布基线与冻结来源均不支持
空序列；仅放宽guard会破坏原序列状态索引，因此保留原范围并补明确拒绝签名的负向合同测试。
