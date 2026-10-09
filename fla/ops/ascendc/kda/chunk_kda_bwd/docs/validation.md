# Issue #615 开发期验证

## 范围与方案

- 基线：`203148ff96bc6d3342d87314dafa075cff271006`。
- 原 ACLNN 入口统一非空张量的 OriginalShape 与 ViewShape；Kernel A 仅扩充失败日志。
- 不修改 ABI、kernel、布局转换、CPU 标杆或数学语义。
- 原失败场景：逻辑视图为 dense BNSD，创建 aclTensor 时 storageDims 为一维，导致 origin rank 为 1。
- 正常多维 descriptor 与展平 descriptor 使用相同输入、dtype、stride 和属性进行对照。

## A2 验证记录

- 日期：2026-10-09；平台：a2-server，910B3，CANN 9.1.0。
- 工作目录：`/data/wys/issue615-fix-20261009`。
- 使用上次复现保留的构建对象与 kernel，仅重编译修改后的 ACLNN 和 tiling 翻译单元并重新链接 Host 库。
- 产物放在本次任务的独立目录，未覆盖原环境。此验证不等同于当前主线的完整 wheel 构建或正式 ATK 验收。
- 编译：`python3 build_fix615.py`，日志 `build.log`。
- 回归：`bash test_fix615.sh`，日志 `test.log` 及 `logs/`。
- 用例：dense T=64/65、varlen T=65、raw gate 与 bias、非法 Aqk rank；各正常场景对照多维与展平 descriptor。
- 设备对照：dense T=64 的确定性非零输入，比较正常与展平 descriptor 的五个梯度输出。

### 结果

- 两个修改涉及的翻译单元编译及 Host 库链接通过；`git diff --check` 通过。
- 八个正常 Host 用例均返回 0；每组正常/展平 descriptor 的 workspace 相同：T=64 为 44,237,312 字节，T=65 为 45,386,752 字节。
- 真正的一维 Aqk 仍返回 561103，失败日志包含 `AqkRank=1, vNewRank=4, dORank=4, hRank=5, expectedTokenRank=4, expectedStateRank=5`。
- dense T=64 两组均完成设备执行；dq/dv/dg 逐位一致，dk 最大绝对差为 2.2737367544323206e-13，db 为 2.5011104298755527e-12，所有输出均为有限值。
- 正常 descriptor 再运行一次也出现上述 dk/db 差异，因此未将逐位一致作为本次一致性门禁。
- `compare_fix615.py` 使用 `abs(a-b) <= 1e-10 + 1e-5*abs(a)` 比较，同步检查有限值；正常/展平与正常重复两组均通过。该门禁仅为 descriptor 一致性检查，不代表独立 CPU 标杆精度通过。
- 首轮 `test.log` 在严格字节比较 dk 时退出；后续完整数值比较见 `consistency.log`，重复运行比较见 `repeat-consistency.log`，产物与修改源码的 SHA256 见 `hashes.txt`。

## 尚未完成

- MindSpore bridge 端到端调用。
- 独立 CPU 标杆精度、完整 ATK 验收以及其他 CANN/SoC 回归。
- 当前 `tests/atk` 尚无原 `ChunkKdaBwd` 的正式验收用例包；本记录只归档缺陷修复的开发期验证，不声明阶段 5 正式验收完成。
