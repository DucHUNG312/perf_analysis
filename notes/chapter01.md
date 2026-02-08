# Chapter 01. Introduction and AI System Overview

## I. AI System Performance Engineer Role

An AI systems performance engineer has a clear impact on the bottom line. We blend expertise across hardware, software, and algorithms. We must understand low-level OS considerations, memory hierarchies, networking fundamentals, and multiple languages like Python and C++, as well as different AI frameworks and libraries such as PyTorch, OpenAI’s Triton, and NVIDIA’s Compute Unified Device Architecture (CUDA).

### 1. Benchmarking and Profiling

Benchmarking and profiling involve measuring latency, throughput, memory usage, and other performance metrics for AI models under various workloads, including training and inference. To identify bottlenecks, we must iteratively use **NVIDIA Nsight Systems** and **NVIDIA Nsight Compute** together with the **PyTorch profiler**. Combined, these tools help pinpoint bottlenecks and track performance over time at different levels of the stack as we continue to improve overall performance of our AI system.

### 2. Scaling Distributed Training and Inference

Scaling small research workloads to larger  roduction workloads on ultrascale clusters will ensure that as we move from 8 GPUs to 80 000 GPUs, the system will scale with minimal overhead and loss of efficiency. This requires optimizing communication using **NVIDIA Collective Communications Library** (NCCL, pronounced “nickel”) for distributed collectives like all-reduce commonly seen in training runs. In addition, the **NVIDIA Inference Xfer Library** (NIXL) provides high throughput, low latency, and point-to-point data movement across GPU memory and storage tiers for distributed inference. 

### 3. Managing Resources Efficiently

It’s important to optimize how models utilize resources like CPU cores, GPU memory, interconnect bandwidth, and storage I/O. This can involve many efforts such as ensuring GPUs are fed with data at full throttle, pinning threads on specific CPU cores, reducing context-switch overhead, orchestrating memory usage, and avoiding out-of-memory (OOM) errors on GPUs when training and inferencing with large models.

### 4. Cross-Team Collaboration

Improving performance might require modifying model code, which involves coordination with researchers. Or you may want to deploy a new GPU driver to improve efficiency, which requires the infrastructure team.

### 5. Transparency and Reproducibility

In performance engineering, it’s vital to measure everything and trust data, not assumptions. By publishing your work, others can learn, reproduce, and build upon your findings.

### 6. Mechanical Sympathy: Hardware-Software Codesign

Real-world experience has shown that even minor tweaks in GPU kernels or memory access patterns can produce outsized gains. A classic example is **FlashAttention**, a novel algorithm that reimplements the transformer attention mechanism in a hardware-aware way. Since FlashAttention, many new attention algorithms have emerged, including **DeepSeek’s Multi-Head Latent Attention** (MLA). This kind of change reduces what used to be a major bottleneck (attention) down to a fraction of overall runtime. 

### 7. Measuring “Goodput” Useful Throughput

Traditional throughput metrics like FLOPS and device utilization are misleadingly high, as much of the time is likely spent on stalled communication, idling computation, or failed job restarts. This is where the concept of “goodput” comes in. 

In simple terms, goodput measures the throughput of useful work completed (number of tokens processed or inference requests completed) per unit time—discounting everything that doesn’t directly contribute to model training or inference. For example, suppose a node with 8 GPUs can process 100,000 tokens in 10 seconds. In this case, its goodput is 10,000 tokens per second. If each GPU in the node can achieve a peak theoretical throughput of 1,500 tokens per second, or 12,000 tokens per second across all 8 GPUs, the node’s efficiency is 83.3% (0.833 = 10,000 achieved throughput/12,000 peak throughput).

Improving goodput requires a deep understanding of the interactions between the hardware (e.g., CPUs, GPUs, network topologies, memory hierarchies, storage layouts), software (e.g., operating system configurations, paged memory, I/O utilization), and algorithms (e.g., transformer architecture variants, attention mechanism alternatives, and different caching and batching strategies).