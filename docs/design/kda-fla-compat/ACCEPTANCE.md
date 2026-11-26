# 集成验收执行记录

当前总状态：未通过，持续开发。不得用本文件或局部通过结果宣称所有性能/精度目标完成。

## 固定测试来源

- fla-org 本地快照提交：`c55dc553edf44701bf703715ed052d811f8a6376`。
- `tests/ops/test_kda.py` SHA256：`81c3a30f7859e66e933179eb338c5e5d38e8236327f65be88a97be597c71d81f`。
- A3 远端为展开的源码目录，没有 `.git`；其测试文件哈希与本地相同，不能据此推断整个远端树完全一致。
- 当前环境 pytest 收集 277 项，来源包含 37 个函数，节点清单见 `upstream-inventory.json`，收集日志见 `evidence/integration-upstream-collection.log`。
- 当前 public_backward_plugin 只接管 10 个 chunk 函数。其他 recurrent、gate、分派等测试即使通过，也不能证明新 ACLNN kernel 已覆盖。
- 原生 Ascend skip、范围外规格、未接管节点必须分别报告。跳过不算通过；新增 skip 或缩减参数不能用于达标。
- 已有 96 项专项通过只是子集。最终还须在集成构建上运行完整相关脚本，并对 C32/C64、保存/重算显式路由验证。

## 当前执行

在独立于窗口1–4设备的 A3 物理卡0上运行当前已安装基线：

```sh
ASCEND_RT_VISIBLE_DEVICES=0 FLA_NPU_OVERLAY_VENDOR= OMP_NUM_THREADS=4 \
PYTHONPATH=/root/wys/fla-compat-d22/upstream-tests \
python -m pytest tests/ops/test_kda.py -p public_backward_plugin -q --maxfail=5
```

此为缺口定位轮，最多5个失败后停止，不等价于最终完整验收。外层900秒超时；任何超时必须记录为未完成。当前轮未通过、结果待收集。

## 性能证据检查

`fla/ops/ascendc/kda/chunk_kda_bwd/tests/upstream/check_performance_gate.py` 对提供的配对样本检查元数据、3轮各至少10次采样、各轮倍率。条件不同或基线角色错误将失败；KV256上限5倍，C32严格低于已验收优化C64，原快路径回退超过5%失败。输入格式见脚本说明。

该脚本只验证输入样本，不生成测量，不证明baseline_role标签真实，不证明case矩阵完整；集成窗口仍需对照源提交、加载包、原始profiling及覆盖清单核实。C32接近噪声时必须增加样本/重复采集，不能凭微小差值宣布胜出。

## 完成条件

1. 四窗口全部实现接入同一集成构建，共享缓存和同步契约审查通过。
2. 完整相关上游脚本及新增泛化覆盖通过；失败、跳过、范围外项没有被隐藏。
3. 原快路径无确认退化；KV256不超过原KV128的5倍；C32两模式前反向与训练整链快于优化后C64。
4. 三轮配对性能证据、实际源码/包身份、设备任务范围和测试日志完整。
5. 任一窗口仅组件通过、仅原型、仅保存模式完成或仍用慢C64基线，都不能结束整体验收。
