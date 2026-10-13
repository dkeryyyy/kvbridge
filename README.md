# KVBridge

English · [中文](README.zh.md)

**Can a second computer's GPU help a Mac handle the memory and computation needed to run a large language model?**

That was the question behind KVBridge. I tried moving part of a model's attention computation from a Mac to an NVIDIA GPU on a Windows computer, then measured whether the tradeoff was worthwhile.

This repository shares the CUDA code, selected measurements, and lessons from both successful and failed attempts. The research stops here for now, published as a **v0.1 research edition** for others to explore.

## Why try this?

When a language model processes a long conversation or document, it keeps intermediate data called a **KV cache** to reuse while generating more text. That cache generally grows with the length of the input.

The idea was to have another computer store and process part of it, reducing the Mac's workload and memory use while keeping generation reasonably fast.

```text
Mac: runs the language model
  |
  +-- Sends part of the attention work to the Windows GPU
  |
  +-- Receives the results and continues generating
```

The experiments used an Apple Silicon Mac and a Windows computer with an RTX 4060 Laptop GPU.

## What did I learn?

**The GPU computation can get faster without making the combined system faster.**

- **Optimizing the GPU code helped.** In an early isolated attention test, computation fell from about 7.81 ms to 2.40 ms, a 3.26× speedup. This was a particular kernel version with synthetic input, not a speedup for the entire model. [Test details, in Chinese](docs/attention-kernel-mma.zh.md)
- **Moving more work could cost more overall speed.** At roughly 65,000 tokens of context, moving one attention layer retained about 96% of the original generation speed; moving five retained about 54%. Communication and software overhead matter too. [Results, in Chinese](docs/whole-layer-1to5-64k.zh.md)
- **This implementation did not yet free the corresponding memory on the Mac.** The original cache allocation stayed in place, so these experiments do not demonstrate a working solution to the Mac's memory limit.

These findings come from the original experimental system. They describe what worked and what did not in this setup, and offer leads for further research. They are not conclusions about every model or GPU.

## What is in this repository?

| Content | What it is useful for |
| --- | --- |
| [CUDA code](src/) and [interfaces](include/kvbridge/) | Studying the GPU attention implementations |
| [Example program](examples/attention_bench.cu) | Seeing how to compare two implementations using generated data |
| [Measurement data](data/) | Inspecting selected historical results and checking their statistics |
| [Research notes](docs/technical-reference.md) | Understanding implementation details, failed approaches, and limitations |

Model weights, the complete inference engine, and the service connecting the two computers are not included. Downloading this repository will not give you a complete offload system to launch.

## Where should I start?

For the findings and their context, see the [results and documentation guide](docs/technical-reference.md). The detailed experiment narratives are mostly in Chinese, including the [one-to-five-layer study](docs/whole-layer-1to5-64k.zh.md) and [where the time went](docs/blackbox-timing-split.zh.md).

To explore the code or try it yourself, read the [technical reference](docs/technical-reference.md) and [build and test guide](docs/reproducibility.md). Inspecting the data and running the CPU checks requires neither model weights nor a GPU.

For a fixed source snapshot, download the package from the [v0.1.0 release page](https://github.com/dkeryyyy/kvbridge/releases/tag/v0.1.0).

## Project status

**This is research source code, not ready-to-use software.** The original experimental system was tested, but the extracted standalone CUDA program has not been rebuilt or validated on a GPU. CPU tests and data checks passed; see [validation status](VALIDATION.md) for their scope.

I am not continuing the experiments for now, and ongoing maintenance is not promised. You are welcome to study, reproduce, or extend the work. See [contributing](CONTRIBUTING.md) if you want to propose a change.

Released under the [MIT license](LICENSE). The [provenance notes](docs/provenance.md) explain where the code and data came from.
