# 64k下1–5个选定层的attention/KV卸载

2026-10-13历史集成研究摘要。这里“whole attention”指选定层完整上下文的**attention/KV计算**，不是把这些decoder层的norm、投影、MLP和权重一起迁走；后者见[五层段实验](segment-validation-five-layer.zh.md)。

## 后续hi/lo数值修复

服务v10集成了MMA attention候选。非零query的FP64资格发现早期单FP16-P版本的误差问题，随后把PV权重改为FP16 hi/lo双MMA。

原v10-04历史资格中，六组长history sweep的`max_abs_normalized`最坏约**1.8031492309e-8**，这里指归一化输出`n/l`的**最大绝对误差**，不是NRMS，也不是整套资格的最坏值。小池poison项另有约**6.1946786722e-6**最大绝对误差，按原`rtol=5e-5 / atol=3e-6`混合规则通过。不能用一个长history数字概括全部输入域。

原DLL身份为`df3d21fc16b9c59cae9c0d810bbef3395aa405ce5843650aea3869ee7725c12c`。它属于原服务构建，不能证明抽取后的独立包已构建或通过。[脱敏v10 oracle摘要](../data/v10-oracle-summary.json)保留原失败、通过指标、公式与来源SHA；不含原输出张量，数据范围见[来源说明](provenance.md)。

## 实测速度与输出

64k历史、512输出tokens、实际DFlash2 B8。以下均使用每组运行自己的基线；原本完整反向第二扫未完成。

| 远端attention层数 | decode tok/s | 同组基线 | 保留基线速度 |
| ---: | ---: | ---: | ---: |
| 1 | 97.64 | 102.02 | 95.7% |
| 2 | 87.82 | 102.02 | 86.1% |
| 3 | 73.05 | 102.02 | 71.6% |
| 4 | 63.17 | 98.36 | 64.2% |
| 5 | 53.17 | 98.36 | 54.1% |

来源：[matrix-02-summary.json](../data/matrix-02-summary.json)的完成行与[matrix-06-summary.json](../data/matrix-06-summary.json)。matrix-02含整体失败字段，不能称整轮矩阵完整成功；matrix-06是后续4/5层继续轮。各完成档记录内容与同pass基线相同，但单一输出任务不能代替广泛模型质量验证。

原服务资格报告另记录64k `kernel_event_ms`中位约2.413ms/op；此数字不是当前独立示例的计时。引擎RPC还包含服务、复制、网络和等待，不能把RPC减kernel全部称为网络。

## 内存与状态边界

摘要明确`local_backing_preserved=true`、`physical_backing_released_bytes=0`。这轮证明的是计算接管和匹配输出，没有证明Mac实体KV容量已经释放。实际DFlash2 B8也不能写成纯AR或draft关闭。

不同组Mac基线与热状态有漂移，因此不能把跨组差异全部归因层数。串行层依赖意味着增加远端层会增加交接成本，但每层开销并非已证明固定常数。

## 失败保留与下一步

原研究包含编译修复、首次数值失败、资源准入拒绝、超时和释放阶段重试。这些过程由内部保护/审计处理；完整工具和证据链未随包提供。随包摘要保留成功行和失败状态，不以完成行覆盖原失败。

下一步需要匹配条件的正反扫、协议路径定位、长上下文及实际backing迁出。KV sidecar或RPC批处理只是假设，收益和新增内存都待测。当前独立包首先按[复现说明](reproducibility.md)验证自己的构建与数值。
