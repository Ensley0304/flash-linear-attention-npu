# rH分担（待试验）

先结束P5b验收，再单独试此项。当前S1只有最后有效行所在AIV计算128行rH；改成两片各计算64行。每行仍按原16行tile、两次64元素WholeReduceSum再Add，保持FP32加法顺序。

| 时点 | 数据与动作 |
| --- | --- |
| S1 | AIV0读取h/dh的0～63行，AIV1读取64～127行；各4个16行tile，输入双buffer不变 |
| S1结束 | 各自64个FP32结果经V→MTE3同步后写relay的153KiB+sub×256B |
| S3 | 已有flag3由MTE3发布，覆盖此前rH写入；不增加跨核事件 |
| S6结束 | AIC等待两片flag3后，沿原依赖发布flag4；此时完整rH可读 |
| S9 | 最后有效行owner从relay152KiB读取384个FP32：前256为state_sum，后128为rH；沿用原状态修正乘加次序 |

relay153～153.5KiB原来未使用，workspace总量不变；Stage0提前执行也不会写此区。

跨work复用：S9状态修正在flag11/12之前。AIV1离开S9前等待AIV0的flag11/12；AIV0离开前等待AIV1的flag12，因此下一work的S1不能覆盖尚未消费的rH。空片仍须参与rH及原事件，不按active跳过。

S1观察点改为两片各写capture的8192+sub×64处64个FP32；拼接后仍是原128行rH。大GM偏移使用uint64_t。

验收除常规门禁外，需重跑S1观察点、尾块和单核多work回绕；保留同一FP32/FP64标杆、原七输出精度阈值。
