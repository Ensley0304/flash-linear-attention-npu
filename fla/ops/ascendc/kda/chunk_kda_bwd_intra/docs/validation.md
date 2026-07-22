# ChunkKdaBwdIntra 验证说明

> 2026-07-22 correction: DAV_2201 validation accepts only the GM stage bridge. The retained `--stage-io tscm` source experiment is rejected for `ascend910b`, because A2 emulates AIV `UB -> TSCM` through GM/Matmul KFC and this direct CATLASS kernel has no KFC client. Stage I/O remains part of the wheel identity so an unsupported artifact cannot be mistaken for the GM baseline. The required gate is still a clean CANN 9.1 build, complete 37-case NPU suite, repeated launches, and same-card msprof.

> 2026-07-22 runtime diagnosis: key 23 clean compilation and wheel packaging pass, while the former persistent and nominally scoped variants both blocked in the first grouped BF16 case. Handshake, stage0-right, stage0-left, and stage0-both all returned on physical device 2, excluding the outer C/V handshake and either diagonal MMAD shape. Source audit then found that the old scoped path still wrapped 2--4 right-side `operator()` calls in one flag envelope starting at stage 1; this usage was unique in the repository and matched the exact pass/hang boundary. The source now defaults to one complete event envelope per logical GEMM and retains persistent mode only as an unvalidated compile-time experiment. Grouped precision and profiling remain unaccepted until this corrected default completes a clean device run.

## 当前状态

本地开发机没有 CANN、PyTorch/NPU 运行时和昇腾设备，因此当前只能完成源码级检查；不能据此声明 AscendC 已编译、上板精度已通过或性能已达标。

| 项目 | 状态 | 说明 |
|---|---|---|
| Python 语法 | 已通过 | wrapper、导出、CPU reference、NPU pytest 均通过 `py_compile`；当前测试矩阵应收集 37 个 case |
| ABI/layout 静态 smoke | 已通过 | 校验 BNSD 转换、BF16 gate 提升、19 个 aclnn 参数和输出布局恢复 |
| safe-gate 代数 smoke | 已通过 | 首/中/尾参考点分解在多种 chunk/tail 长度下与直接 causal 公式一致 |
| 块对角设计模型 | 已通过 | 证明零填充候选不重关联 gate reference、保留有效 K 顺序；4-GEMM 为 5.45x FMA，布局可实现的 11-GEMM 为 1.85x FMA，A2 L1/L0 容量检查通过；这不是设备性能结论 |
| C++ 结构 smoke | 已通过 | key 23 dispatch 与 key 15 重新编包回退、26,624-FP32 GM 双槽、split/packed stage-A 两种等价 workspace 布局、stage-0/shared-setup overlap、每核一次 task 坐标除法与热循环进位递推、每 AIV 一次 causal mask/zero 初始化、每 task 一次常驻 beta `Brcb`、AIV 六类同步事件一次分配/尾释放且 `MTE3_V` 最多同时占用三个 ID、179,936-B 单 scratch 与 192,256-B 局部 pair-scratch ping-pong、color-8 first/middle 与 color-12 right reference cache、color-12/color-4 左右 accumulator、repeat-stride `Sub` 直广播、64-lane FP32 外部 mask 复用及逐调用回退、四尾块 scalar/batch 两条路径、六个 color-4 K-wide pair bridge 及 repeat-stride-0 `Mul` 直广播、17 个逻辑 GEMM、scoped 每-task 11 包络与 persistent 每-Process 双包络、IEEE/HF32 Cube 两条编译分支、K32+K16/M16-strided、6 次 Exp 与 8 次 Select、db K=128 合并/逐行两条归约路径、whole-tail 与 8-row stage-epilogue 两条源码分支、跨 task serial/overlap 写回分支、完整 row scratch UB 边界及显式事件生命周期均已静态检查 |
| patch 卫生 | 已通过 | `git diff --check` 无错误 |
| CANN host/kernel 编译 | grouped 待验证 | AIV phase1 已在 A2/CANN 9.1 clean build；新增 key 23 尚未编译 |
| AscendC NPU 精度 | AIV phase1 已通过 | clean wheel 旧版完整 22 项通过；当前新增 9 个 grouped NPU case、1 个目标域 unsafe fallback、4 个 CPU 数值 guard 和 1 个 source-dispatch contract，尚待完整跑 37 项 |
| Profiling/性能优化 | phase1 已采集 | 目标 shape AIV phase1 48.660 ms、端到端 51.107 ms；grouped key 23 待采集，4 ms 未达成 |

## A2 精度与性能基线（2026-07-21）

EVENT4 legacy 版本完整 pytest 为 `22 passed in 30.53s`。随后 AIV phase1 clean wheel 也完成
真实 kernel launch，完整 pytest 为 `22 passed in 12.77s`。覆盖 safe/unsafe、FP16/BF16、dense/varlen、GVA、四种 layout、
BT=64/128、K=16/48/96/256、重复 launch、零 dA 和 one-hot dA 路径。

目标性能 shape 的 legacy AscendC kernel duration 为 477.937 ms，同 shape Triton kernel 为
19.272 ms；AIV phase1 kernel 为 48.660 ms，端到端中位数为 51.107 ms。当前 grouped 修改使用
新 tiling key 23，必须重新完成 key23-only 编译探针、clean wheel、9 个 grouped NPU 定向、
1 个目标域 unsafe fallback、4 个 CPU 数值 guard、1 个 source-dispatch contract、完整 37 项与 repeated launch 后才能
评价性能；新的默认四尾块 batch、scoped MMAD、Vector-mask 复用与 db 合并归约路径，以及
stage-epilogue、pair-scratch ping-pong、跨 task 输出重叠、packed stage-A 和 HF32 Cube 候选还必须各自单独完成
同样的 clean build、37 项和 profile。phase1、key 15 与 key 23 `tail/single/batch/scoped` 的结论都不能
自动外推到其他候选。

## 构建与安装

推荐使用仓内分阶段脚本；它把产物放到持久目录，不会修改或安装到全局 CANN：

```bash
CANN_ENV=/data/wys/Ascend/cann_9.1.0_910b_0605/ascend-toolkit/set_env.sh

# 输出最后会给出本轮 /var/tmp/.../state.env。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode build --cann-env "$CANN_ENV" --physical-device 2 \
  --pair-gates source --shared-setup source --stage-epilogue source \
  --pair-scratch source --tail-blocks source --task-store source \
  --mmad-engines source \
  --vector-mask source --db-reduce source --stage-a source --cube-mode source \
  --stage-io source

STATE=/var/tmp/wys_kda_grouped_YYYYMMDD_HHMMSS/state.env
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode test --state "$STATE" --cann-env "$CANN_ENV" --physical-device 2

# 首轮先采 PipeUtilization；精度通过后加 --full-metrics 采集全部七组指标。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode profile --state "$STATE" --cann-env "$CANN_ENV" \
  --physical-device 2 --full-metrics
```

脚本要求同一 wheel 的完整 37 项先通过才允许 profile；JUnit 门禁严格要求
`tests=37, failures=0, errors=0, skipped=0`。build/test/profile 会绑定同一个 commit、wheel
SHA256、KDA object digest、验证源码 manifest、runner SHA256、clean CATLASS commit/tree、SOC、CANN
realpath、Python、torch/torch_npu 和物理卡，并在 test/profile 前分别创建新的隔离安装目录。若 build
后当前仓的 runner 已更新，split test/profile 会拒绝旧 state；此时应使用该 state 对应
`$WHEEL_SRC/scripts/run_chunk_kda_bwd_intra_grouped_validation.sh`。`--final-gate` 只用于最终验收并且
必须与 `--full-metrics` 同时使用；
首轮 profile 不应加该参数，否则未达到 4 ms 时会按设计返回非零。

### 同提交 A/B 归因

runner 可以只改 clean archive 中的十二个编译期常量，不修改调用者工作区，用于在同一提交上隔离
pair-bridge 因式分解、shared-setup、stage-epilogue、局部 pair-scratch、四尾块 batch/scalar 和
persistent MMAD 调度、Vector mask 复用、db 跨行归约、跨 task 输出写回与 stage-A 打包的收益。
`--stage-epilogue` 接受 `source|overlap|tail`，`--pair-scratch` 接受
`source|pingpong|single`，`--tail-blocks` 接受 `source|batch|scalar`，`--mmad-engines` 接受
`source|persistent|scoped`；`--vector-mask` 接受 `source|reuse|per-call`；`--db-reduce` 接受
`source|coalesced|per-row`；`--task-store` 接受 `source|overlap|serial`；`--stage-a` 接受
`source|packed|split`；`--cube-mode` 接受 `source|ieee|hf32`；`--stage-io` 接受
`source|gm|tscm`，但 A2/`ascend910b` 会拒绝 `tscm`；`source` 表示沿用提交中的常量。
当前默认源码对应 `pair-factor_setup-overlap_epilogue-tail_scratch-single_tail-batch_mmad-scoped_vmask-reuse_dbr-coalesced_store-serial_stagea-split_cube-ieee_io-gm`，
pair-scratch 候选为 `pair-factor_setup-overlap_epilogue-tail_scratch-pingpong_tail-batch_mmad-scoped_vmask-reuse_dbr-coalesced_store-serial_stagea-split_cube-ieee_io-gm`，
尾块 scalar 回退为 `pair-factor_setup-overlap_epilogue-tail_scratch-single_tail-scalar_mmad-scoped_vmask-reuse_dbr-coalesced_store-serial_stagea-split_cube-ieee_io-gm`，
跨 task 候选为 `pair-factor_setup-overlap_epilogue-tail_scratch-single_tail-batch_mmad-scoped_vmask-reuse_dbr-coalesced_store-overlap_stagea-split_cube-ieee_io-gm`，
persistent-MMAD 实验为 `pair-factor_setup-overlap_epilogue-tail_scratch-single_tail-batch_mmad-persistent_vmask-reuse_dbr-coalesced_store-serial_stagea-split_cube-ieee_io-gm`；
逐调用 mask 回退为 `pair-factor_setup-overlap_epilogue-tail_scratch-single_tail-batch_mmad-persistent_vmask-per-call_dbr-coalesced_store-serial_stagea-split_cube-ieee_io-gm`；
逐行 db 回退为 `pair-factor_setup-overlap_epilogue-tail_scratch-single_tail-batch_mmad-persistent_vmask-reuse_dbr-per-row_store-serial_stagea-split_cube-ieee_io-gm`；
packed-A 候选为 `pair-factor_setup-overlap_epilogue-tail_scratch-single_tail-batch_mmad-persistent_vmask-reuse_dbr-coalesced_store-serial_stagea-packed_cube-ieee_io-gm`；
HF32 候选为 `pair-factor_setup-overlap_epilogue-tail_scratch-single_tail-batch_mmad-persistent_vmask-reuse_dbr-coalesced_store-serial_stagea-split_cube-hf32_io-gm`。
每个命令都会创建独立的 run root、wheel、37 项精度
记录和 profile manifest，状态文件也会绑定
`variant/pair_gates/shared_setup/stage_epilogue/pair_scratch/tail_blocks/mmad_engines/vector_mask/db_reduce/task_store/stage_a/cube_mode/stage_io`，不能混用不同变体的测试或性能证据。
build run root 与 profile group 均通过 `mktemp` 原子创建，允许顺序或并发运行多个变体而不覆盖
产物。每个完整 profile group 还会生成 `profile_evidence.pass`，记录 commit、wheel/KDA digest、
CANN、CATLASS、37/37、变体身份、主 CSV/manifest 哈希和当前 4 ms 判定；文件名中的 `pass`
表示该轮请求的采集证据完整，不等于 `performance_target=PASS`，最终仍以其中字段和 runner 退出码为准。

```bash
SRC=/data/wys/gdn/kda_single/flash-linear-attention-npu
CANN_ENV=/data/wys/Ascend/cann_9.1.0_910b_0605/ascend-toolkit/set_env.sh

# 源码默认基线：pair bridge 因式分解 + shared-setup overlap + whole-tail epilogue + single scratch + batch tail + serial store + scoped MMAD + mask 复用 + db 合并归约。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates source --shared-setup source \
  --stage-epilogue source --pair-scratch source --tail-blocks source \
  --task-store source --mmad-engines source --vector-mask source \
  --db-reduce source --stage-a source --cube-mode source --stage-io source

# 只切换 stage-A workspace 打包；数学矩阵、safe-gate reference 和 17 个 GEMM 不变。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates factor --shared-setup overlap \
  --stage-epilogue tail --pair-scratch single --tail-blocks batch \
  --task-store serial --mmad-engines scoped --vector-mask reuse \
  --db-reduce coalesced --stage-a packed --cube-mode ieee --stage-io gm

# 只切换 Cube 输入模式；HF32 必须先通过完整精度门禁，不能放宽阈值。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates factor --shared-setup overlap \
  --stage-epilogue tail --pair-scratch single --tail-blocks batch \
  --task-store serial --mmad-engines scoped --vector-mask reuse \
  --db-reduce coalesced --stage-a split --cube-mode hf32 --stage-io gm

# 默认 scoped 源码现在关闭 UnitFlag，由公共 BlockMmad non-UnitFlag 分支为每个逻辑 GEMM
# 建立完整的 M_FIX/FIX_M 事务。先用 clean wheel 跑 grouped BF16 单测确认可退出，再进入
# 37 项精度和性能验收；逐调用 UnitFlag 包络和外加 FIX_M 的旧结果均为 timeout。
# runner 的 test/all 模式会先执行 120 秒 grouped BF16 预检；超时会显式打印 exit=124/137，
# 并停止进入 directed/full37，避免只返回 shell prompt 而无法判断退出原因。

# 仅构建 persistent MMAD 实验；当前协议已知 timeout，不用于精度或性能验收。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates factor --shared-setup overlap \
  --stage-epilogue tail --pair-scratch single --tail-blocks batch \
  --task-store serial --mmad-engines persistent --vector-mask reuse \
  --db-reduce coalesced --stage-a split --cube-mode ieee --stage-io gm

# 只打开局部 pair scratch ping-pong，其余十一维固定为源码默认基线。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates factor --shared-setup overlap \
  --stage-epilogue tail --pair-scratch pingpong --tail-blocks batch \
  --task-store serial --mmad-engines scoped --vector-mask reuse \
  --db-reduce coalesced --stage-a split --cube-mode ieee --stage-io gm

# 只回退为逐块 scalar tail，其余十一维固定为源码默认基线。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates factor --shared-setup overlap \
  --stage-epilogue tail --pair-scratch single --tail-blocks scalar \
  --task-store serial --mmad-engines scoped --vector-mask reuse \
  --db-reduce coalesced --stage-a split --cube-mode ieee --stage-io gm

# 跨 task 写回最小 A/B 基线：固定 batch tail，只保留串行输出写回。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates factor --shared-setup overlap \
  --stage-epilogue tail --pair-scratch single --tail-blocks batch \
  --task-store serial --mmad-engines scoped --vector-mask reuse \
  --db-reduce coalesced --stage-a split --cube-mode ieee --stage-io gm

# 跨 task 写回候选：唯一变化是 store-overlap；它与 scalar tail/stage-local epilogue 不兼容。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates factor --shared-setup overlap \
  --stage-epilogue tail --pair-scratch single --tail-blocks batch \
  --task-store overlap --mmad-engines scoped --vector-mask reuse \
  --db-reduce coalesced --stage-a split --cube-mode ieee --stage-io gm

# 只前移 stage epilogue，其余十一维固定：每个 8-row block 在 ConsumeStage 后完成 db reduce 和 dkLeft*beta。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates factor --shared-setup overlap \
  --stage-epilogue overlap --pair-scratch single --tail-blocks batch \
  --task-store serial --mmad-engines scoped --vector-mask reuse \
  --db-reduce coalesced --stage-a split --cube-mode ieee --stage-io gm

# 只关闭 pair bridge 因式分解，其余十一维固定为默认基线。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates direct --shared-setup overlap \
  --stage-epilogue tail --pair-scratch single --tail-blocks batch \
  --task-store serial --mmad-engines scoped --vector-mask reuse \
  --db-reduce coalesced --stage-a split --cube-mode ieee --stage-io gm

# 只把共享初始化移回串行 prologue，其余十一维固定为默认基线。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates factor --shared-setup prologue \
  --stage-epilogue tail --pair-scratch single --tail-blocks batch \
  --task-store serial --mmad-engines scoped --vector-mask reuse \
  --db-reduce coalesced --stage-a split --cube-mode ieee --stage-io gm

# 只关闭外部 Vector mask 复用，其余十一维固定；回退到每个 Level-2 API 自行配置 mask/counter。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates factor --shared-setup overlap \
  --stage-epilogue tail --pair-scratch single --tail-blocks batch \
  --task-store serial --mmad-engines scoped --vector-mask per-call \
  --db-reduce coalesced --stage-a split --cube-mode ieee --stage-io gm

# 只关闭 db 跨行归约，其余十一维固定；回退到每行一次第一层 WholeReduceSum。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode all --source "$SRC" --cann-env "$CANN_ENV" \
  --physical-device 2 --pair-gates factor --shared-setup overlap \
  --stage-epilogue tail --pair-scratch single --tail-blocks batch \
  --task-store serial --mmad-engines scoped --vector-mask reuse \
  --db-reduce per-row --stage-a split --cube-mode ieee --stage-io gm
```

各轮必须使用相同物理卡、CANN、commit、shape、warmup/repeat 和 profiler 指标。先分别确认完整
37/37 精度，再比较各自 PipeUtilization 的 10 次测量中位数；首轮 A/B 不加 `--final-gate`，也不把
缺少完整精度记录或 manifest 不匹配的 CSV 用作优化收益证据。

grouped key 23 的 build/test/profile 流程仅适用于 DAV_2201 的 A2/A3 SoC，例如
`ascend910b` 或对应的 `ascend910_93` 配置；A5/`ascend950` 只验证通用 AIV fallback，不能使用
本脚本的 `--require-mixed` 性能门禁。key 23 探针、key 15
重编回退探针和最终 wheel 必须分别从同一个提交创建独立 clean 源码快照，不能复用前一个探针留下的
`build/`、`build_out/` 或 tiling-key 过滤状态：

```bash
CANN_ENV=/data/wys/Ascend/cann_9.1.0_910b_0605/ascend-toolkit/set_env.sh
set +u
source "$CANN_ENV"
set -u
python scripts/check_npu_env.py --build-only

# 先只编译 key 23，快速暴露 grouped 模板/设备代码问题；该产物不用于交付。
bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu \
  --ops=chunk_kda_bwd_intra --tiling_key=23

# 在第二份 clean 快照中只编译 key 15，证明关闭 grouped 后可重新编包回退。
bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu \
  --ops=chunk_kda_bwd_intra --tiling_key=15

# 在第三份 clean 快照的仓库根目录构建一体化验证 wheel。只过滤算子，不过滤
# tiling key，因而 key 15/23 会一起进入该算子的设备编译范围。
export FLA_NPU_SOC=ascend910b
export FLA_NPU_OPS=chunk_kda_bwd_intra
unset FLA_NPU_SKIP_RUN_BUILD FLA_NPU_SKIP_RUN_INSTALL \
  FLA_NPU_INCREMENTAL_BUILD FLA_NPU_BUILD_LEGACY_EXTENSION TILING_KEY
mkdir -p dist
python3 -m pip wheel -v --no-build-isolation --no-deps . -w dist

# 隔离安装并让后续测试只解析当前 wheel，避免加载环境中已有的 fla_npu。
WHEEL=$(find dist -maxdepth 1 -type f \
  -name 'flash_linear_attention_npu-*.whl' -print -quit)
WHEEL_INSTALL=$(mktemp -d /tmp/fla_npu_kda_install.XXXXXX)
python3 -m pip install --no-deps --target "$WHEEL_INSTALL" "$WHEEL"
export PYTHONPATH="$WHEEL_INSTALL${PYTHONPATH:+:$PYTHONPATH}"

# 定向确认 wheel 内嵌 KDA OPP；单算子验证 wheel 不运行全仓 API 配置检查。
find "$WHEEL_INSTALL/fla_npu/opp" -type f \
  \( -name 'chunk_kda_bwd_intra.json' -o -path '*chunk_kda_bwd_intra/*.o' \) -print
python3 -c 'import inspect, fla_npu; from fla_npu.ops import ascendc; print(inspect.getfile(fla_npu)); assert callable(ascendc.chunk_kda_bwd_intra)'

# 完整 release wheel 另行在 clean 快照中 unset FLA_NPU_OPS 后构建；只有完整 wheel
# 才运行 scripts/check_packaged_wheel_api.py。
```

以下旧式拆分构建不能作为本算子的 wheel 验证证据，因为 run 包和 Python wrapper 可能来自不同源码状态：

```bash
# 不用于本轮验证：
bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu \
  --ops=chunk_kda_bwd_intra
cd torch_custom/fla_npu
python3 setup.py bdist_wheel
```

安装或切换 wheel 后必须启动新的 Python 进程，避免复用已经 `dlopen` 的旧 `libcust_opapi.so`。

## 精度回归

```bash
# 版本门禁：grouped 数值与 dispatch 保护加入后，当前测试文件应收集 37 条。
python -m pytest --collect-only -q \
  torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py

# 先跑 grouped 随机、128-task/B2 slot canary、back-to-back launch/大负 gate、
# 跨 AIV one-hot、目标域 unsafe fallback，以及 FTZ、overflow、cancellation 和 source contract。
python -m pytest -q -vv -p no:cacheprovider \
  torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py \
  -k "grouped_fastpath or grouped_dispatch_source_contract or unsafe_target_shape or reference_right_diag_ftz_guard or reference_left_diag_overflow_guard or reference_off_right_cross_block_ftz_guard or reference_grouped_cancellation_guard" -s

# 再跑原有 5 条路径定位用例，隔离零 dA、dAqk/dAkk 和 safe/unsafe。
python -m pytest -q -vv -p no:cacheprovider \
  torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py \
  -k "zero_da or one_hot_da_paths" -s

# 定向用例通过后再跑完整单算子回归。
bash torch_custom/fla_npu/test/test.sh --device 0 --op chunk_kda_bwd_intra
# 或直接运行：
python -m pytest -q torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py -s
```

用例覆盖：

- `safe_gate=true`：BF16 `q/k/g/beta`、大负累积 gate、tail token；
- `safe_gate=false`：FP16 兼容分支，以及会与 grouped safe 目标域重合的 BF16/T64/K128 回退；
- GVA：`H=1,HV=4` 与 `H=2,HV=4`；
- dense：`B>1` 且 `T>chunk_size`；
- varlen：显式 canonical `chunk_indices`，并覆盖单序列跨多个 chunk；
- `BSND`、`BNSD`、`TND`、`NTD` 和 `chunk_size=64/128`。
- `K=16/48/96/256` 的下界、16-feature 尾块、非 2 次幂和上界。
- 零 dA 时四个输入梯度累积量原样写回。
- 单点 `dAqk[18,2]` 与 `dAkk[18,2]` 的跨 16-token block 行/列路径，分别覆盖 safe/unsafe。
- grouped key 23 的 BF16/T64/K128 随机精度、B2/T128/H32 四 stage 唯一 marker slot canary、
  back-to-back launch、大负 gate，以及 `dA[24,3]` / `dA[55,43]` 跨两个 AIV 半区的
  dAqk/dAkk 拼接路径；后一个点同时覆盖 stage-3 grouped 前缀和 pair-2 stride-48 的
  M16 ColumnMajor 右 GEMM。
- grouped `dAkk` 定向项还在同一个 pytest case 内追加四 stage、两个 AIV 半区的精确 diagonal
  marker，覆盖非零输入 `db`、`beta=0` 与非单位 beta，专门检查 stage epilogue 必须先执行
  `db += reduce(dk_left_pre*k)`，再执行 `dk_left_pre *= beta`；该增强不改变 37 项 provenance。
- 合法 `-5/ln(2)` gate 步长下的 right-diagonal FTZ 与 left-diagonal 中间溢出保护，
  off-right 跨块公共 reference FTZ 反例、左右 inner 同时使用非单位 outer/bridge 的 FP32
  normal-floor 边界，以及 stage-3 K32+K16 对三个 off-diagonal block 的 FP32 cancellation
  顺序保护。

板端通过门槛：四个输出均 finite，且逐输出通过测试中的 CPU FP64 golden 容差。若 safe case 失败，应先分别对比 `dq_local`、`dk_left_pre`、`dk_right`，重点检查 16-token 首/中/尾参考点的内外指数方向。

## 性能采集

先固定常用形状分别测试 safe/unsafe，再采集 kernel duration、AIC/AIV time、Cube/Vector utilization、MTE2/FIXPIPE、跨核 wait、流水 stall 和 task tail。至少覆盖：

```text
(BT,K) = (64,128), (128,128)
dtype  = FP16, BF16
mode   = dense, varlen
HV/H   = 1, 2, 4
```

msprof 中必须确认目标 launch 生成 mixed task、AIC/AIV time 均非零、`Mix Block Num>0`，并将
Cube 精度模式作为独立证据核对：默认 `cube_mode=ieee` 必须为 `HF32 Eligible=NO`，显式
`cube_mode=hf32` 候选必须为 `YES`。还要与同一 clean wheel 的普通计时交叉核对。msprof 的 mixed 行本身不能区分
key 15 与 key 23；运行时 key 23 由“同提交 key23-only 编译探针 + clean wheel 无 tiling-key 过滤 +
host/source dispatch contract + `BF16/BT64/K128/safe_gate=true` launch manifest + mixed AIC/AIV profile”
联合证明。只有完整 37 项通过后才进行性能判断；若 key 23 精度失败则关闭 grouped 开关并重新编包
回退 key 15/既有 key，若性能未接近目标则按 AIC/AIV wait、Cube/FIX、Exp/Scalar 占比继续迭代，
不能用 phase1 数据代替 grouped 结果。

正式 PipeUtilization 在同一个 msprof 进程中执行 3 次 warmup 和 10 次测量。CSV 会包含 13 个目标
算子 task，分析时先按 start time 排序并丢弃前三行，再对剩余 10 行的
`max(Task Duration, AIC time, AIV time)` 取中位数。采集后使用仓内分析脚本把“精确 10 输入
shape/dtype”“mixed AIC/AIV 证据”和“4 ms 达标”分开判定。launch manifest 还会绑定
`gate_scale=0.2`、进程逻辑卡 0、唯一可见物理卡，以及与 `discard-first/expected-rows` 一致的
warmup/repeat，避免不同测量合同被误归为同一轮证据：

```bash
PIPE_DIR="$PROF_GROUP/PROF_PipeUtilization"
OP_CSV=$(find "$PIPE_DIR" -type f -name 'op_summary_*.csv' -print | sort | tail -1)
LAUNCH_MANIFEST="$PIPE_DIR/launch_manifest.json"  # 由实际 benchmark 参数生成

# 以下示例校验源码默认 tail/single/batch/serial/scoped 变体；候选只改对应期望值。
# 首轮性能诊断：精确签名与 mixed AIC/AIV 必须成立，但允许性能暂未达标。
python3 scripts/analyze_chunk_kda_bwd_intra_profile.py "$OP_CSV" \
  --target-ms 4.0 --baseline-ms 48.660472 \
  --discard-first 3 --min-rows 10 --expected-rows 10 \
  --expected-device-id 2 --launch-manifest "$LAUNCH_MANIFEST" \
  --expected-pair-gates factor --expected-shared-setup overlap \
  --expected-stage-epilogue tail --expected-pair-scratch single \
  --expected-tail-blocks batch --expected-task-store serial \
  --expected-mmad-engines scoped \
  --expected-vector-mask reuse --expected-db-reduce coalesced \
  --expected-cube-mode ieee \
  --require-target-shape --require-mixed

# 该命令只判断这份 CSV 是否达到 4 ms，不替代 wheel/37项/full-metrics 证据链。
python3 scripts/analyze_chunk_kda_bwd_intra_profile.py "$OP_CSV" \
  --target-ms 4.0 --baseline-ms 48.660472 \
  --discard-first 3 --min-rows 10 --expected-rows 10 \
  --expected-device-id 2 --launch-manifest "$LAUNCH_MANIFEST" \
  --expected-pair-gates factor --expected-shared-setup overlap \
  --expected-stage-epilogue tail --expected-pair-scratch single \
  --expected-tail-blocks batch --expected-task-store serial \
  --expected-mmad-engines scoped \
  --expected-vector-mask reuse --expected-db-reduce coalesced \
  --expected-cube-mode ieee \
  --require-target-shape --require-mixed \
  --require-under-target

# 最终交付门禁必须从 build 阶段保存的 state 运行；只有完整 runner 返回 0 才能声明达标。
bash scripts/run_chunk_kda_bwd_intra_grouped_validation.sh \
  --mode profile --state "$STATE" --cann-env "$CANN_ENV" \
  --physical-device 2 --full-metrics --final-gate

# single/pingpong 两轮 --mode all 完成后，严格校验哈希和除 pair_scratch 外的身份，
# 再比较十次测量中位数。默认至少快 1% 才推荐 pingpong，否则判为 inconclusive。
python3 scripts/compare_chunk_kda_bwd_intra_profiles.py \
  "$SINGLE_PROFILE_GROUP" "$PINGPONG_PROFILE_GROUP" \
  --require-pingpong-faster

# 只有候选自身也达到 4 ms 时才可用于最终目标门禁。
python3 scripts/compare_chunk_kda_bwd_intra_profiles.py \
  "$SINGLE_PROFILE_GROUP" "$PINGPONG_PROFILE_GROUP" \
  --require-pingpong-faster --require-under-target
```

其余六组 PMU 指标分别采集、分别分析，不能把不同 `--aic-metrics` 的行混在一起计算一个性能
中位数。runner 同时生成 `kda_profile_summary.json`，其中保留 AIC/AIV 总时间以及 Cube、Vector、
Scalar、MTE1/2/3、Fixpipe 的逐样本时间/占比；pair-scratch comparer 使用这些字段定位收益是否
确实来自 AIV MTE3/Vector 重叠。最终验收还要求 sample-based 采集成功并实际生成非空
`aicore.db`，脚本会归档其 SHA256。
