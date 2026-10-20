# 完整chunk专用路径（待试验）

前提：先完成P5f验收。只新增完整64行的编译期路径，不叠加norm/beta模板化。

Host完成原metadata合法性校验后，检查所有chunk的rows是否都是64。满足时调用FinalizeCatlass<true>，否则调用<false>。定长与变长均按实际metadata判断，不从T整除64推断变长chunk完整。

Kernel中rows为FULL_CHUNK ? 64 : metadata.GetValue(chunk*4+2)。seq、first和输出地址算法不变，GM偏移仍uint64_t。将常量64传播到各Stage，裁剪尾行清零、active、lastOwner等分支。不会复用已撤销P4b的元数据地址推导。

两条路径共享同一实现，workspace、布局、同步和FP32运算顺序不变。编译后确认MIX AIC/AIV任务类型和模板核名称，profiling仍须恰好捕获主核、归约核各40条记录。

定向用例必须包含：单chunk64行、多个完整chunk且单核回绕、全64行变长、混合完整/尾块变长；普通六例不一定覆盖完整路径，不能代替这些用例。性能仍用八配置两轮交替A/B，有稳定收益后原20case七输出验收。
