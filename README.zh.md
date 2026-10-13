# KVBridge Offload Study — v0.1 研究版

[English](README.md) · [复现边界](docs/reproducibility.md) · [源码与数据来源](docs/provenance.md) · [验证状态](VALIDATION.md) · [变更记录](CHANGELOG.md) · [贡献说明](CONTRIBUTING.md) · [MIT 许可](LICENSE)

本项目整理了把 KV 注意力从 Apple Silicon Mac 迁至 RTX 4060 Laptop 的研究，附带一个**固定几何的实验性 CUDA 注意力库**。包括 SIMT 参考核、张量核候选、合成示例，以及成功、失败和负结果的部分历史数据。

**验证状态：原集成研究系统已经构建并实测；本独立包经过抽取与 v0.1 修订，当前版本的 CUDA 库构建、GPU 数值测试及 sanitizer 检查仍未验证。下列历史成绩不等于当前独立源码的成绩。**

**本地macOS/AppleClang 21的host-only CMake/CTest已通过：98项形状、15项CLI、6,149项oracle辅助逻辑检查；host AddressSanitizer/UndefinedBehaviorSanitizer亦通过。** 精确范围见[验证状态](VALIDATION.md)。

## 包含什么

- 分页 INT8 GQA 注意力、split-K 部分结果与 merge，CUDA 内存由调用方管理。
- SIMT 参考核与 BF16 QK / FP16 PV、FP32 累加的 MMA 候选；当前源码包含后续 PV 权重 FP16 hi/lo 修改。
- 有界合成示例和小型 FP64 oracle 自测入口，不加载模型、不启动远端服务。
- 精选 JSON/CSV 证据及专题总结。**不含**原推理引擎、模型权重、远端服务、保护监督器或完整历史实验框架。

这是研究版，不是通用注意力后端，也不是可直接启动的 Mac/Windows 卸载产品。

## 历史成果与边界

| 范围 | 结果 | 证据边界 |
| --- | --- | --- |
| 早期 attention v2-07 基准 | 65,536 tokens：SIMT 7.8141ms，MMA 2.3951ms，约3.26倍 | RTX 4060 Laptop、合成输入、正扫中位数；早于当前 hi/lo 源码。[原记录](data/att-bench-65536-v2-07.json)。 |
| service v10 注意力卸载 | 64k 下，1个选定注意力层保留95.7%基线decode速度，5层保留54.1% | DFlash2 B8、512输出tokens，与同轮基线比较；本地backing仍保留。[专题](docs/whole-layer-1to5-64k.zh.md)。 |
| 协议客户端测量 | 客户端RTT中位4.098ms，服务端窗口3.227ms，成对“传输＋客户端残余”0.867ms | 0.867ms不是单独测出的网线时间；与另一客户端相差约6ms是定位线索，因果未证。[数据](data/blackbox-session-05-analysis.json)。 |
| 含权重的5个完整decoder层 | CUDA 40.271872ms，Mac 4.081718ms | 同短历史组件快照，仅2次CUDA重放，不是整模型生成速度；带宽推论受硬件、B8与权重流量假设限制。[专题](docs/segment-validation-five-layer.zh.md)。 |
| KV容量算术 | 每个注意力层每token 2,080B | 4 KV heads、dim256的INT8 K/V＋FP32 scales，不含scratch、页表、运行时和储备。[容量表](data/kv-memory-table.csv)。 |

研究经验是分别测量计算、协议开销和实际物理内存释放。不能据此断言权重卸载普遍不可能，或KV卸载必然提速。

## 构建与验证

历史集成实验使用 **Windows、RTX 4060 Laptop（sm_89）、CUDA 13.3.1**。本包默认目标为sm_89；其他GPU、系统与CUDA版本均未验证，指令存在不等于兼容性验收。还需CMake 3.20+与兼容的C++17主机编译器；工具链不随包提供。

无需CUDA也可运行[host-only检查](docs/reproducibility.md#host-only-checks)，覆盖形状/尺寸、CLI和参考计算辅助逻辑；它们不验证GPU实现。发布完整性检查需要Python 3.9+及标准库：

```sh
python3 tools/validate_release.py
```

它检查本地链接、JSON、文件哈希和部分历史算术，不执行CUDA，也不修改manifest。

以下是拟定的CUDA验证流程，**不是已成功执行的命令记录**。从包根目录运行，先读[复现与资源说明](docs/reproducibility.md)。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release --parallel 2
```

单配置生成器：

```sh
./build/attention_bench --self-test
./build/attention_bench --tokens 8192 --warmup 8 --iters 32
```

Windows多配置生成器的可执行文件通常为`build/Release/attention_bench.exe`。先通过小型数值和内存检查，再增加到`--tokens 65536`。CLI提供`--tokens`（256的倍数，256–65,536）、`--warmup`（0–32）、`--iters`（1–256）、`--self-test`，精确校验以示例源码为准。新结果须记录实际工具链、设备、命令、退出状态和源哈希；数值失败的运行不能进入性能胜者表。

## API与内存合同

接口见[attention_common.h](include/kvbridge/attention_common.h)。示例负责buffer与event生命周期，库启动器不代调用方分配。

对已验证的`Shape s`，至少分配：

| 输出 | float元素数 |
| --- | ---: |
| `m`、`l`各自 | `s.rows * s.q_heads` |
| `n` | `s.rows * s.q_heads * s.dim` |
| `scratch` | `kvbridge::scratch_elements(s)` |

分母为正时，归一化输出为`n[(row*q_heads+head)*dim+d] / l[row*q_heads+head]`；空范围或没有可见key时原始约定为`m=-inf, l=0, n=0`，不能除以零；示例把归一化向量视为零。页表条目、容量、位置及device指针必须符合头文件布局。

MMA仅面向INT8 KV、8 query rows、每KV head对应6 query heads、dim256、Page32和页对齐的`begin`。Q以FP32传入，但预期数值域要求其值能用BF16精确表示。完整形状和启动限制以源码为准。

**`launch_attention_mma`没有自动回退。** 不支持的形状或shared-memory opt-in失败会抛异常。`attention_mma_supported`只是几何检查，不证明设备能力、分配成功或数值资格；调用方可在验证参考路径限制后显式选择`launch_attention`。

## 数值口径

全部signed INT8码值都能在K的BF16、V的FP16暂存中精确表示。QK的scale在dot后FP32应用；PV则先在FP32中计算softmax权重乘V scale，再分成FP16 hi/lo。hi/lo减少误差，不是精确运算，也不提供覆盖极小值/溢出等情况的统一相对误差保证。

须区分三类证据：

1. 随包v2-07 JSON检查的是拼接的 **`n/m/l`**，normalized RMS约`1.283e-5`，不是当前hi/lo核的`n/l`验证。
2. 后来的v10集成报告记录了非零query的FP64资格：单FP16-P版本失败，hi/lo通过；随包有[脱敏指标摘要](data/v10-oracle-summary.json)，不含原输出张量和完整资格工具。
3. 当前独立示例检查归一化 **`n/l`**，另设小型oracle自测，**当前GPU结果尚未验证**，且新非零query合成生成器不同于历史基准。未来通过也只覆盖该fixture，不等于完整服务或模型资格。

验收规则与记录要求见[内核设计](docs/kernel-design.md)、[复现说明](docs/reproducibility.md)。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [reproducibility](docs/reproducibility.md) | 可复核范围、验证流程和缺失工件 |
| [provenance](docs/provenance.md) | 源码沿革、公开数据索引、许可边界 |
| [kernel-design](docs/kernel-design.md) | 布局、形状、算术、所有权 |
| [MMA优化](docs/attention-kernel-mma.zh.md) | v2-03至v2-07历史优化与平台期 |
| [早期MMA](docs/attention-kernel-mma-early.zh.md) | 布局错误、向量读和fragment装配 |
| [1–5层注意力卸载](docs/whole-layer-1to5-64k.zh.md) | 集成服务曲线与本地backing保留 |
| [协议计时](docs/blackbox-timing-split.zh.md) | 客户端/服务端测量与因果边界 |
| [完整五层组件](docs/segment-validation-five-layer.zh.md) | 连续decoder段实测与失败 |
| [整段可行性](docs/segment-feasibility-review.zh.md) | 条件带宽估算与容量/速度取舍 |
| [Q8线性核](docs/linear-kernels-q8.zh.md) | 独立投影实验，不属于本包库实现 |

## 局限与许可

本库不含模型加载、网络、页归属、资源准入、训练反向或生产集成。历史保护器未打包；调用方须提供分配上限、所有权、同步、取消和清理。长上下文容量、物理内存收益及其他硬件性能仍需新证据。

许可为[MIT](LICENSE)；CUDA、工具链及另行获取的模型遵守各自条款。[来源说明](docs/provenance.md)列出已查范围；本包许可不宣称拥有第三方工具或研究成果。
