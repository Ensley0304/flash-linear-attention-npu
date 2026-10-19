# chunk_kda_bwd_finalize_catlass 算子

## 基本信息

- **算子名称**：chunk_kda_bwd_finalize_catlass
- **创建时间**：2026-10-19 09:48:56
- **开发状态**：开发中

## 目录结构

```
chunk_kda_bwd_finalize_catlass/
├── docs/           # 设计、计划、环境、审查、串讲文档
├── build/          # 编译输出
├── test/           # 测试数据
├── CMakeLists.txt  # 构建脚本（待创建）
├── run.sh          # 运行脚本（待创建）
├── gen_data.py     # 测试数据生成（待创建）
└── golden.py       # Golden 数据计算（待创建）
```

## Catlass 编译要求

编译时需注入 catlass 头文件与架构号：
- `-I$\{CMAKE_SOURCE_DIR\}/../../catlass/include`
- `-DCATLASS_ARCH=<架构号>`（910b/910_93=2201；950=3510）
