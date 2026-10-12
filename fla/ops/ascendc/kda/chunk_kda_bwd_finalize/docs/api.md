# Finalize 接口

A2/A3 扩展为已确认的设计契约；当前开发提交实现 Stage 0～6，尚未接入完整公开接口。现有 A5 调用保持不变。

## 现有 A5 L0 接口

```cpp
namespace l0op {

const std::array<const aclTensor *, 7> ChunkKdaBwdFinalize(
    // 输入
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *gk, const aclTensor *rawG, const aclTensor *beta,
    const aclTensor *aLog, const aclTensor *dtBias, const aclTensor *akk,
    const aclTensor *vNew, const aclTensor *h, const aclTensor *dh,
    const aclTensor *dvScan, const aclTensor *dAqk, const aclTensor *dqRaw,

    // 可选输入
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,

    // 属性
    double scale, double lowerBound, int64_t chunkSize,
    bool safeGate, bool useGateInKernel, bool useExp2, bool stateVFirst,

    // 输出：调用方提供 Tensor
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dBetaOut, const aclTensor *dGOut,
    const aclTensor *dALogOut, const aclTensor *dDtBiasOut,

    aclOpExecutor *executor);

}
```

返回顺序：

```cpp
{dqOut, dkOut, dvOut, dBetaOut, dGOut, dALogOut, dDtBiasOut}
```

| 类型             | 输入 / 输出                                                  |
| ---------------- | ------------------------------------------------------------ |
| BF16 输入        | q、k、v、akk、vNew、h、dh、dvScan                            |
| FP32 输入        | gk、rawG、dtBias、dAqk、dqRaw                                |
| BF16 / FP32 输入 | beta、aLog                                                   |
| 可选 FP32 输入   | qRstdOptional、kRstdOptional                                 |
| 可选整数数组     | cuSeqlensOptional、chunkIndicesOptional；内部转 INT64 Tensor |
| BF16 输出        | dqOut、dkOut、dvOut                                          |
| 与 beta 同类型   | dBetaOut                                                     |
| FP32 输出        | dGOut、dALogOut、dDtBiasOut                                  |

## 当前 A5 限制

- 仅注册 `ascend950`。
- `K = V = 128`，`Hq = Hv`，`chunkSize = 64`，输入非空。
- Tensor 使用连续 ND 布局。
- 定长 Token 布局为 `[B,H,T,D]`，变长为 `[H,T,D]`。
- 定长状态 h/dh 为 `[B,Nc,H,K,V]`，变长为 `[Nc,H,K,V]`。
- qRstdOptional、kRstdOptional 必须同时提供或同时为空；提供时融合 Q/K L2Norm 反向。
- cuSeqlensOptional、chunkIndicesOptional 必须同时提供或同时为空；提供时走变长路径。
- `scale` 与前向一致；`lowerBound` 默认 `-5`，文档约定范围 `[-5,0)`。
- `safeGate = true`、`useGateInKernel = true`、`useExp2 = true`。
- 当前 dBetaOut 是对有效 beta 的梯度，未融合 beta sigmoid 反向。

## beta sigmoid 反向扩展方案

### 新增属性

在现有属性区末尾、输出参数之前增加：

```cpp
bool useBetaSigmoidInKernel,
bool allowNegEigval,
```

| L0 参数                | 算子属性 / Python 参数     | 默认值 |
| ---------------------- | -------------------------- | ------ |
| useBetaSigmoidInKernel | use_beta_sigmoid_in_kernel | false  |
| allowNegEigval         | allow_neg_eigval           | false  |

- `allowNegEigval = true` 要求 `useBetaSigmoidInKernel = true`。

### 输入与输出语义

- beta 仍传前向使用的有效值，无需增加 beta_raw 输入。
- 普通 sigmoid：`beta = sigmoid(beta_raw)`。
- allowNegEigval 模式：`beta = 2 × sigmoid(beta_raw)`。
- 关闭融合：dBetaOut 输出对有效 beta 的梯度，类型与输入 beta 一致。
- 开启融合：dBetaOut 输出对 beta_raw 的梯度，类型与原始 beta_raw 一致（BF16 / FP32），由调用方提供的输出 Tensor 确定，不随 beta_eff 的存储类型改变。

### 精度约定

- 开启融合时，beta 输入使用前向以 FP32 计算并保存的 beta_eff，无需新增 beta_raw 输入；不能先保存为 BF16，再转回 FP32。
- sigmoid 反向在 FP32 中计算：普通模式乘 `beta × (1 − beta)`；allowNegEigval 模式乘 `beta × (1 − beta/2)`，最后转换为 dBetaOut 的类型。
- 例如 `sigmoid(8) ≈ 0.99966466`，转 BF16 后为 1，导数会从约 `0.00033522` 变为 0；保存 FP32 beta_eff 可避免这次提前舍入。

## 输出可选BSND

### 新增属性

新增 `const char *outputLayout`，对应算子属性 / Python 参数 `output_layout`。
与 beta sigmoid 扩展合并后，L0 属性区为：

- 定长支持 `BNSD / BSND`；变长支持 `NTD / TND`。
- 上层默认：定长 `BNSD`，变长 `NTD`，保持现有输出布局；L0 显式传参。
- outputLayout 仅控制 dqOut / dkOut / dvOut 的输出布局。
- 本项不扩展输入布局：Token 输入及内部中间量继续使用 BNSD / NTD。

### 输出形状

N 表示 head 数，S 表示定长序列长度，T 表示变长打包后的总 token 数。

| outputLayout | dqOut / dkOut | dvOut |
| ------------ | ------------- | ----- |
| BNSD         | `[B,N,S,K]` | `[B,N,S,V]` |
| BSND         | `[B,S,N,K]` | `[B,S,N,V]` |
| NTD          | `[N,T,K]` | `[N,T,V]` |
| TND          | `[T,N,K]` | `[T,N,V]` |

dGOut 固定为定长 BSND `[B,S,N,K]`、变长 TND `[T,N,K]`；dBetaOut 对应为 `[B,S,N]`、`[T,N]`，不受 outputLayout 控制。

dALogOut、dDtBiasOut 的形状及布局不变；dBetaOut 的 dtype 遵循上述融合规则，其余输出 dtype、返回顺序不变。

## 接口设计

```cpp
namespace l0op {

const std::array<const aclTensor *, 7> ChunkKdaBwdFinalize(
    // 输入：Token 张量使用 BNSD / NTD；beta 为前向有效值
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *gk,
    const aclTensor *rawG,
    const aclTensor *beta,
    const aclTensor *aLog,
    const aclTensor *dtBias,
    const aclTensor *akk,
    const aclTensor *vNew,
    const aclTensor *h,
    const aclTensor *dh,
    const aclTensor *dvScan,
    const aclTensor *dAqk,
    const aclTensor *dqRaw,

    // 可选：Q/K L2Norm 反向，成对提供
    const aclTensor *qRstdOptional,
    const aclTensor *kRstdOptional,

    // 可选：变长元数据，成对提供
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,

    // 原有属性
    double scale,
    double lowerBound,
    int64_t chunkSize,
    bool safeGate,
    bool useGateInKernel,
    bool useExp2,
    bool stateVFirst,

    // 新增：beta sigmoid 反向
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,

    // 新增：定长 BNSD / BSND，变长 NTD / TND
    const char *outputLayout,

    // dq/dk/dv：布局由 outputLayout 控制
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dvOut,

    // 固定布局：dBeta 为 BSN / TN，dG 为 BSND / TND
    const aclTensor *dBetaOut,
    const aclTensor *dGOut,

    // 参数梯度输出：布局固定
    const aclTensor *dALogOut,
    const aclTensor *dDtBiasOut,

    aclOpExecutor *executor);

} // namespace l0op
```

返回顺序：

```cpp
{dqOut, dkOut, dvOut, dBetaOut, dGOut, dALogOut, dDtBiasOut}
```
