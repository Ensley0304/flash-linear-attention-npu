# 9.1 megakernel 移植验证记录

状态：依赖和方案已固定，尚未开始本分支硬件验收。用户已明确授权补齐 A2/A3 megakernel。

- 基线：`36e67d2decea70745554f7aebf9a6a29e9168a91`。
- 固定来源：`95004a10a0176113348c50e33ce0a98106eb5b49`。
- 主线来源与本发布分支结果严格分开；未获得发布分支 A3 原始运行数据。
- 原 CPU recurrence、500条冻结矩阵、输入值域和精度阈值不变。
- 必需依赖：算子私有 arch22 kernel18、arch22 tiling2、public host/私有 L0/opdef4文件。
  公共组件 blob 与来源相同，不移植 arch35/internal operators 或其他算子。
- 单独的 gate FP32 依赖与 PR #887 对应，保留其 A5 原低精度 Prepare 行为。

| 验证项 | 入口 | 当前状态 |
| --- | --- | --- |
| public ABI 与既有 A5 调用合同 | aclnn_abi_contract.py | 待执行 |
| A2/A3/A5 目标编译 | build.sh 单算子含原依赖 | 待执行 |
| 既有精度矩阵 | 原 ATK generalized500 与 smoke | 待执行 |
| native layout、offset、非连续 view、低精度 gate | public API 同输入回归 | 待实现/执行 |
| 陈曦五 shape 同输入 kernel profiling | msprof，ABBA | 待执行 |
| A5 原 Prepare/H/O 扩展 | 原有测试与 CODEOWNER/CI | 待执行 |
| 全量 wheel/OPP、Example/ST、NPU CI | 仓库 Admin CI | 未执行 |

完整原始日志、输入、版本收据、device 快照和 manifest 存于独立旁车，公开 PR 仅附去除私有路径的摘要。
只有门禁实际齐全后才归档完整验收；本记录不能当作通过证据。
