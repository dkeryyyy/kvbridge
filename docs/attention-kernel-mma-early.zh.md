# 早期 MMA 优化：布局、向量读与 fragment 装配

本页是2026-10-13历史v2-03/v2-04实验的公开摘要，不是当前独立包运行报告。原构建日志与这些轮次的完整原始数据未随包提供；可公开复核的后续64k记录见[数据索引](provenance.md)。

## 研究问题与改动

原SIMT注意力在RTX 4060 Laptop、8 query rows、Page32 INT8 KV、64k历史下约7.8ms。早期MMA候选保留原KV字节与scale，采用BF16 QK、FP16 PV、FP32累加，希望减少逐元素算术和装配成本。

| v2-04改动 | 作用 |
| --- | --- |
| u32读取原INT8，打包BF16/FP16写shared | 减少标量加载和转换指令 |
| Q/K、S、V的shared行距加padding | 改善特定访问方式的bank映射 |
| `ldmatrix.x4/x2`装配 | 替换部分手工fragment标量读取 |
| V改为dim-major暂存、PV A复用 | 配合PV的fragment访问 |

构建前审查曾发现K fragment误用`.trans`，修正后才进入设备资格。这个失败说明“字节类型可精确转换”不足以证明MMA布局正确；必须独立核对lane、矩阵索引和尾边界。

## 历史结果（报告摘录）

| tokens | SIMT A正扫p50 | v2-03 MMA | v2-04 MMA |
| ---: | ---: | ---: | ---: |
| 8,192 | 1.0895ms | 0.5406ms | 0.3666ms |
| 32,768 | 3.9248ms | 2.3419ms | 1.7060ms |
| 65,536 | 7.8149ms | 4.6141ms | 3.3471ms |

当时使用合成patterned INT8 KV与8 warmup、每扫32次的正反扫。报告中的MMA容差比较属于原fixture和部分统计量口径，不能移作当前hi/lo源码的`n/l`验证。后续非零query资格还发现了早期FP16-P路径的精度缺口，见[1–5层集成专题](whole-layer-1to5-64k.zh.md)。

## 保留的经验与未知

布局修正、宽读与fragment装配在该轮有实测收益。至于剩余瓶颈各占多少，需要硬件计数器；寄存器数、无spill或纯流读速度不能单独证明算力利用率。

较长上下文按split数线性外推只是估算；流读probe省略softmax、转换、MMA与merge，不能当注意力可直接达到的时间。当前独立包首先需要[构建、oracle与sanitizer验证](reproducibility.md)，历史命令因依赖未打包的内部工具而不在本页提供。
