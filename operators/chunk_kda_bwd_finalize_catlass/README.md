# A2 Finalize：CANNBot 独立实现

## 基本信息

- **算子名称**：chunk_kda_bwd_finalize_catlass
- **创建时间**：2026-10-19 09:48:56
- **开发状态**：S0～S6单head、单chunk通过7个边界case；整算子及性能未验收
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

测试需使用已有PyTorch、CT与NumPy环境。`finalize_stage0`是开发期前缀入口，PREFIX范围0～6。中间结果抓取仅用于验收，性能版需关闭抓取。唯一数学标杆为`tests/atk/chunk_kda_bwd_finalize/reference.py`，本目录没有另写公式。
