# Attention MMA 优化研究：历史版本与证据边界

研究日期2026-10-13；公开摘要整理于2026-10-14。v2-03至v2-07是历史合成微基准系列。**当前包包含后续hi/lo PV修改，不是v2-07的原样可执行版本；尚无当前独立包性能成绩。**

## 历史优化轨迹

下表为原研究报告摘录，64k为正扫p50。除最后一行的64k记录外，完整各轮原始数据未随包提供。

| 版本 | 主要改动 | 64k MMA |
| --- | --- | ---: |
| v2-03 | 首个BF16/FP16 MMA，标量staging、手工fragment | 4.614ms |
| v2-04 | 向量staging、padding、ldmatrix、PV fragment复用 | 3.347ms |
| v2-05 | 512线程、每warp两个PV输出tile | 2.412ms |
| v2-06 | 寄存器预取、QK双累加链 | 2.406ms |
| v2-07 | softmax使用`__expf` | 2.395ms |

后两次改动没有显示明显的额外收益。v2-06同时改变了预取和累加链，因此不能把这一次时间变化解释为两个独立的因果实验；也不能由“打平”证明已经找到全部硬件瓶颈。

## 可公开复核的64k记录

[att-bench-65536-v2-07.json](../data/att-bench-65536-v2-07.json)保存原32次/扫、8 warmup、正反扫数据。输入标记为`synthetic_patterned_kv_layout_identity_pages_no_model`。

| 指标 | 原v2-07记录 |
| --- | ---: |
| SIMT A正扫p50 | 7.8141ms |
| MMA正扫p50 | 2.3951ms |
| 两者时间比 | 约3.26倍 |
| MMA相对A的normalized RMS | 约1.283e-5 |
| MMA相对A的max abs | 约0.1588 |

**后两项作用于拼接的`n/m/l`，不是归一化输出`n/l`。** 同一组容差常数用于不同输出对象时，含义不同，不能跨对象继承资格。此fixture也不能覆盖非零query下所有softmax分布。

## 后续精度修复不能与早期速度拼接

服务v10后来使用非零query和FP64 oracle发现：单FP16-P候选未通过指定数值门，改成FP16 hi/lo双MMA后通过该历史服务资格。当前包承接了这一后续源码。历史六组长history sweep的`n/l`最大绝对误差约1.80315e-8，**不是NRMS，也不是整套资格的最坏误差**；小池poison另有约6.19468e-6最大绝对误差，并按当时相对/绝对混合规则通过。

因此不能把“v2-07的2.3951ms”“v10长history的1.80315e-8”合写成当前独立源码一次实测结果。原失败与通过指标见[脱敏v10 oracle摘要](../data/v10-oracle-summary.json)；完整张量和原工具未打包，版本关系见[来源说明](provenance.md)。

## 平台期之后的研究方向

同布局纯流读probe与attention做的工作不同；缓存命中还可能使codes/time高于标称DRAM带宽。它适合提示优化空间，不能直接给出attention理论可达时间。

可继续研究profiler指导的指令/占用分析、生产者/消费者调度或KV预转换sidecar。sidecar若同时保留原INT8和更宽镜像，需累计两份存储及维护成本；不是无成本的“直接逼近probe”。这些路线均需新实验，本包不附成功承诺。

下一步是当前源的独立[复现与验证](reproducibility.md)，再开展受控性能比较。
