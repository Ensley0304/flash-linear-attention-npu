# A2 Finalize：CANNBot 独立实现

## 基本信息

- **算子名称**：chunk_kda_bwd_finalize_catlass
- **创建时间**：2026-10-19 09:48:56
- **开发状态**：S0～S10候选已实现；P4已通过原20case七输出和八配置两轮A/B，8K无附加反向17.312ms；200case全量及9.604ms目标未验收
- **基线标签**：`baseline/kda-finalize-a2-20261019`，原入口保留

## 目录结构

```
chunk_kda_bwd_finalize_catlass/
├── docs/           # 设计、计划、环境、审查、串讲文档
├── build/          # 编译输出
├── test/           # 测试数据
├── CMakeLists.txt  # A2直调构建
├── op_host/        # 独立C++ host
├── op_kernel/      # CATLASS Cube与AscendC Vector
└── reference/      # 唯一CPU标杆入口与冻结精度策略
```

## Catlass 编译要求

编译时需注入 catlass 头文件与架构号：
- `-I$\{CMAKE_SOURCE_DIR\}/../../catlass/include`
- `-DCATLASS_ARCH=<架构号>`（910b/910_93=2201；950=3510）

```bash
cmake -S operators/chunk_kda_bwd_finalize_catlass -B build-finalize -DCATLASS_ROOT=/path/to/catlass
cmake --build build-finalize -j4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python operators/chunk_kda_bwd_finalize_catlass/test/check_stage0.py \
  --executable /absolute/path/build-finalize/finalize_stage0 \
  --skill-root /path/to/catlass-cpp-generator \
  --artifacts /path/to/report --device 3 --prefix 6
```

测试需使用已有PyTorch、CT与NumPy环境。`finalize_stage0`是开发期前缀入口，PREFIX范围0～9。完整无抓取入口为`finalize`，包含主核和独立归约核；测试使用`test/check_full.py`，默认6个冒烟case。原200case指定`--cases all`，三个seed基数为20261013、20271013、20281013；大用例可用`--work-dir`复用输入目录并保留各轮报告。

唯一数学标杆为`tests/atk/chunk_kda_bwd_finalize/reference.py`。阶段抓取包含大尺度中间量，仍有单独比较失败记录；最终七输出验收不放宽阈值。恢复位置见`docs/resume.md`。
