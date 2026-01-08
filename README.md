
<div align="center">

# Daily Arxiv Papers (LMSys)

![Static Badge](https://img.shields.io/badge/total_papers-1826-blue?logo=gitbook)
![Static Badge](https://img.shields.io/badge/update-2026.01.08-red?logo=fireship)
[![Static Badge](https://img.shields.io/badge/arXiv-cs.DC-green)](https://arxiv.org/list/cs.DC/recent)
[![Static Badge](https://img.shields.io/badge/arXiv-cs.OS-green)](https://arxiv.org/list/cs.OS/recent)
[![Static Badge](https://img.shields.io/badge/arXiv-cs.LG-green)](https://arxiv.org/list/cs.LG/recent)

`Fetch from arxiv` → `LLM Filter` → `GitHub workflow update`

## [🔥Daily Arxiv: LLM Systems 👉 paper.tju.chat 👈](https://paper.tju.chat)

</div>

**👍Conference Papers on LMSys**: [conference.md](conference.md)

**⚠️NOTE**: Update papers up to last day every morning (8:00 UTC+8) automatically.

**🙋WANT**: Keyword subscription (email); Functional web page.

**🔖TAGS**:`serving` `training` `offline` `thinking` `RL` `MoE` `RAG` `video` `multi-modal` `sparse` `quantization` `offloading` `hardware` `storage` `kernel` `diffusion` `agentic` `edge` `networking`

---
### 2026-01-08
* `serving` `scaling` [Hummingbird: SLO-Oriented GPU Preemption at Microsecond-scale](http://arxiv.org/abs/2601.04071v1)
  > **TL;DR**: Addresses GPU sharing inefficiencies for SLO adherence in inference serving. Proposes Hummingbird with microsecond-scale preemption on closed-source GPUs to harvest idle slices. Achieves 9.7x better SLO attainment for high-priority tasks over spatial sharing.
* `MoE` `serving` `edge` [A Scheduling Framework for Efficient MoE Inference on Edge GPU-NDP Systems](http://arxiv.org/abs/2601.03992v1)
  > **TL;DR**: Proposes a scheduling framework for efficient MoE inference on edge GPU-NDP systems. Key optimizations include tensor parallelism for expert partitioning, load-balancing scheduling, and dataset-free pre-fetching. Achieves 2.41x average and up to 2.56x speedup in end-to-end latency versus state-of-the-art.

### 2026-01-07
* `serving` `agentic` [Software-Defined Agentic Serving](http://arxiv.org/abs/2601.03197v1)
  > **TL;DR**: Addresses inflexibility in multi-agent LLM serving pipelines. Proposes a software-defined networking inspired framework for dynamically controlling communication based on runtime state. Achieves efficient and responsive agent systems.
* `MoE` `serving` `offloading` [Making MoE-based LLM Inference Resilient with Tarragon](http://arxiv.org/abs/2601.01310v2)
  > **TL;DR**: Addresses poor failure resilience in MoE-based LLM inference. Proposes Tarragon with reconfigurable datapath, asynchronous KV cache checkpointing, and shadow experts to confine failures. Reduces failure-induced stalls from ~64s to 0.3-0.4s (160-213× improvement) compared to MegaScale-Infer.
* `training` `LoRA` `kernel` [Chronicals: A High-Performance Framework for LLM Fine-Tuning with 3.51x Speedup over Unsloth](http://arxiv.org/abs/2601.02609v1)
  > **TL;DR**: Addresses memory and compute inefficiencies in LLM fine-tuning with Chronicals, a framework featuring fused Triton kernels, Cut Cross-Entropy, LoRA+, and sequence packing. Achieves 3.51x speedup over Unsloth on full fine-tuning of Qwen2.5-0.5B at 41,184 tokens/sec.
* `training` [First Provably Optimal Asynchronous SGD for Homogeneous and Heterogeneous Data](http://arxiv.org/abs/2601.02523v1)
  > **TL;DR**: Develops provably optimal asynchronous SGD methods for distributed learning with heterogeneous worker speeds. Introduces Ringmaster ASGD (handling homogeneous data) and Ringleader ASGD (handling heterogeneous data) based on a framework to manage staleness, plus ATA for adaptive task allocation. Achieves optimal time complexity matching synchronous methods while reducing waiting time.

### 2026-01-06
* `training` `scaling` `memory` [Placement Semantics for Distributed Deep Learning: A Systematic Framework for Analyzing Parallelism Strategies](http://arxiv.org/abs/2601.02311v1)
  > **TL;DR**: Introduces placement semantics to systematically analyze parallelism strategies by predicting memory and communication for distributed training. Derives conditions for correctness and composition rules. Matches exact results: e.g., ZeRO-3 uses 8× less memory with 1.5× communication cost vs data parallelism.
* `serving` `offloading` `storage` [RelayGR: Scaling Long-Sequence Generative Recommendation via Cross-Stage Relay-Race Inference](http://arxiv.org/abs/2601.01712v1)
  > **TL;DR**: Addresses serving latency for long-sequence generative recommendation models under strict SLOs. Proposes RelayGR, which pre-infers reusable prefixes and maintains KV caches in HBM across pipeline stages via selective triggering, affinity routing, and memory expansion. Achieves 1.5× longer sequences and 3.6× higher throughput under fixed P99 SLO.
* `training` `hardware` `scaling` [DiT-HC: Enabling Efficient Training of Visual Generation Model DiT on HPC-oriented CPU Cluster](http://arxiv.org/abs/2601.01500v1)
  > **TL;DR**: Proposes DiT-HC, a system for efficient training of DiT visual generation model on HPC CPU clusters. Introduces communication-free tensor parallelism, optimized kernels for vector/matrix units, and customized MPI backend. Achieves 90.6% weak scaling efficiency on 256 nodes and up to 87.7x speedups.
* `MoE` `serving` [Making MoE based LLM inference resilient with Tarragon](http://arxiv.org/abs/2601.01310v1)
  > **TL;DR**: Addresses resilience in MoE-based LLM serving by confining worker failures. Proposes Tarragon with reconfigurable datapath, asynchronous KV cache checkpointing, and shadow experts for low-overhead recovery. Reduces failure-induced stalls by 160-213x compared to MegaScale-Infer.
* `serving` `offloading` `sparse` [Warp-Cortex: An Asynchronous, Memory-Efficient Architecture for Million-Agent Cognitive Scaling on Consumer Hardware](http://arxiv.org/abs/2601.01298v1)
  > **TL;DR**: Proposes Warp-Cortex for memory-efficient large-scale multi-agent language model inference. Uses weight sharing and topological synapse-based context sparsification to reduce KV-cache memory from O(N*L) to O(N*k). Achieves 100 concurrent agents at 2.2 GB VRAM on a single GPU.
* `RL` `training` `networking` [OrchestrRL: Dynamic Compute and Network Orchestration for Disaggregated RL](http://arxiv.org/abs/2601.01209v1)
  > **TL;DR**: Addresses compute imbalance and network bottlenecks in disaggregated RL for LLMs. Proposes OrchestrRL with adaptive compute scheduler and RFabric reconfigurable network fabric. Achieves 1.40x higher throughput on 48 H800 GPUs.

### 2026-01-05
* `edge` `serving` `networking` [FlexSpec: Frozen Drafts Meet Evolving Targets in Edge-Cloud Collaborative LLM Speculative Decoding](http://arxiv.org/abs/2601.00644v1)
  > **TL;DR**: Addresses high communication overhead in edge-cloud LLM speculative decoding for evolving models. Proposes FlexSpec with a shared-backbone architecture and channel-aware adaptive speculation. Reduces latency by 25-60% compared to conventional SD under varying conditions.
* `serving` `offloading` [Revati: Transparent GPU-Free Time-Warp Emulation for LLM Serving](http://arxiv.org/abs/2601.00397v1)
  > **TL;DR**: Proposes Revati, a time-warp emulator for transparent GPU-free performance modeling of LLM serving systems. Executes real serving code by virtualizing CUDA calls and fast-forwarding time via predicted kernel durations. Achieves <5% error in predictions and runs 5-17x faster than GPU execution.
* `RL` `agentic` `RAG` [Bio-inspired Agentic Self-healing Framework for Resilient Distributed Computing Continuum Systems](http://arxiv.org/abs/2601.00339v1)
  > **TL;DR**: Proposes ReCiSt, a bio-inspired agentic self-healing framework for distributed systems using LM-powered agents to autonomously diagnose faults and reconfigure resources. Achieves self-healing within tens of seconds with agent CPU usage as low as 10%.
* `offloading` `scaling` `serving` [Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling](http://arxiv.org/abs/2512.24637v2)
  > **TL;DR**: Targets GPU memory bottleneck for hosting large-scale tasks via demand paging; proposes MSched, an OS-level scheduler enabling proactive working set migration via kernel argument prediction. Achieves up to 57.88× speedup over conventional demand paging for LLMs under memory oversubscription.

### 2026-01-01
* `serving` `kernel` `RAG` [Vulcan: Instance-Optimal Systems Heuristics Through LLM-Driven Search](http://arxiv.org/abs/2512.25065v1)
  > **TL;DR**: Proposes Vulcan, which synthesizes instance-optimal heuristics for resource-management tasks using LLM-driven evolutionary search. Vulcan provides task-agnostic interfaces to enable LLMs to generate executable policies for specific workloads/hardware. Demonstrates up to 69% performance improvement over state-of-the-art algorithms for cache eviction.
* `training` `serving` `networking` [Reliable and Resilient Collective Communication Library for LLM Training and Serving](http://arxiv.org/abs/2512.25059v1)
  > **TL;DR**: Addresses network failures during large-scale LLM training and serving. Designs R^2CCL, a fault-tolerant communication library with rapid connection migration and resilient collective algorithms. Achieves >12x speedup over baselines with 1% training and 3% inference overheads.
* `scaling` `networking` [AI-Driven Cloud Resource Optimization for Multi-Cluster Environments](http://arxiv.org/abs/2512.24914v1)
  > **TL;DR**: Addresses inefficient resource management in multi-cluster cloud environments. Introduces an AI-driven framework with predictive learning and cross-cluster coordination for dynamic resource allocation. Achieves up to 25% higher resource efficiency and faster stabilization during workload fluctuations.
* `serving` `networking` `storage` [Adaptive Resource Orchestration for Distributed Quantum Computing Systems](http://arxiv.org/abs/2512.24902v1)
  > **TL;DR**: Proposes ModEn-Hub architecture for orchestrating distributed quantum computing resources. Integrates a photonic interconnect with a quantum network orchestrator that schedules non-local gates and manages shared quantum memory and ebit caching. Achieves 90% teleportation success versus 30% in baseline while maintaining higher entanglement attempts per node.
* `training` `edge` `offloading` [Distributed Bilevel Optimization with Dual Pruning for Resource-limited Clients](http://arxiv.org/abs/2512.24667v1)
  > **TL;DR**: Proposes a resource-adaptive distributed bilevel optimization framework for low-resource clients, using a second-order free hypergradient estimator. Achieves asymptotically optimal convergence rate of O(1/√(Cₓ*Q)), validated on two tasks.
* `training` `storage` [Understanding LLM Checkpoint/Restore I/O Strategies and Patterns](http://arxiv.org/abs/2512.24511v1)
  > **TL;DR**: Investigates efficient I/O strategies for LLM checkpoint/restore operations during training. Proposes microbenchmarks and techniques including file system-aware aggregation and coalescing with liburing. Achieves up to 7.6× higher write throughput compared to existing systems.
* `offloading` `sparse` `quantization` [PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy Compression](http://arxiv.org/abs/2512.24449v1)
  > **TL;DR**: Proposes PackKV, a lossy compression framework for KV cache in long-context LLM inference, reducing memory footprint. Combines tailored compression algorithms with system co-design to minimize decompression overhead. Achieves 153.2% higher K cache and 179.6% V cache memory reduction vs. SOTA quantization at same accuracy drop.
* `offline` `video` `quantization` [RedunCut: Measurement-Driven Sampling and Accuracy Performance Modeling for Low-Cost Live Video Analytics](http://arxiv.org/abs/2512.24386v1)
  > **TL;DR**: Proposes RedunCut for dynamic model size selection in live video analytics. Uses measurement-driven sampling and data-driven accuracy prediction. Reduces compute cost by 14-62% at fixed accuracy for road-vehicle, drone, and surveillance videos.
* `edge` `serving` `kernel` [Squeezing Edge Performance: A Sensitivity-Aware Container Management for Heterogeneous Tasks](http://arxiv.org/abs/2512.23952v1)
  > **TL;DR**: Optimizes container-based resource management for heterogeneous tasks on edge servers. Proposes a sensitivity-aware framework using profiling-based latency modeling and MINLP optimization decomposed into convex subproblems. Achieves over 14% latency reduction and higher energy efficiency versus baselines.
* `edge` `training` `RL` [Vulcan: Instance-Optimal Systems Heuristics Through LLM-Driven Search](http://arxiv.org/abs/2512.25065v1)
  > **TL;DR**: Proposes Vulcan, a system using LLM-driven evolutionary search to synthesize instance-optimal heuristics for resource management. Separates policy and mechanism via LLM-friendly interfaces for synthesizing OS/distributed system policies. Achieves up to 69% higher performance over state-of-the-art in cache eviction and memory tiering.
* `serving` `offloading` [MSched: GPU Multitasking via Proactive Memory Scheduling](http://arxiv.org/abs/2512.24637v1)
  > **TL;DR**: Addresses high overheads in GPU multitasking due to HBM capacity limits and demand paging. Proposes MSched, an OS-level scheduler with proactive memory migration using template-based working set prediction and global page placement. Improves performance by up to 11.05x for DL workloads and 57.88x for LLMs.

### 2025-12-30
* `networking` `scaling` `storage` [Local Rendezvous Hashing: Bounded Loads and Minimal Churn via Cache-Local Candidates](http://arxiv.org/abs/2512.23434v1)
  > **TL;DR**: Proposes Local Rendezvous Hashing to reduce load imbalance and churn in distributed systems. Restricts hash selection to cache-local physical nodes to improve locality. Achieves a max/avg load of 1.0947 vs. 1.2785 and 6.8× faster throughput than multi-probe hashing for K=50M keys.
* `scaling` `serving` [An SLO Driven and Cost-Aware Autoscaling Framework for Kubernetes](http://arxiv.org/abs/2512.23415v1)
  > **TL;DR**: Addresses SLO violations and cost inefficiencies in Kubernetes autoscaling for cloud-native applications. Proposes an SLO-driven, cost-aware framework with AIOps principles and lightweight forecasting. Reduces SLO violation duration by 31%, lowers cost by 18%, and improves response time by 24%.
* `serving` `edge` `scaling` [Splitwise: Collaborative Edge-Cloud Inference for LLMs via Lyapunov-Assisted DRL](http://arxiv.org/abs/2512.23310v1)
  > **TL;DR**: Proposes Splitwise, a Lyapunov-assisted DRL framework for adapting LLM inference partitions between edge devices and cloud to reduce latency and energy. Divides layers into sub-blocks for fine-grained decisions. Achieves 1.4x-2.8x lower latency and up to 41% energy savings versus baselines.
* `serving` `quantization` `offloading` [Viability and Performance of a Private LLM Server for SMBs: A Benchmark Analysis of Qwen3-30B on Consumer-Grade Hardware](http://arxiv.org/abs/2512.23029v1)
  > **TL;DR**: Investigates deploying private LLM servers for small businesses using quantized MoE model on consumer hardware. Benchmarks quantized Qwen3-30B performance under load, focusing on latency, tokens/second and scalability. Achieves comparable performance to cloud services while reducing cost and preserving privacy.
* `serving` `offloading` `edge` [Argus: Token Aware Distributed LLM Inference Optimization](http://arxiv.org/abs/2512.22925v1)
  > **TL;DR**: Addresses efficiency challenges in distributed LLM inference across edge-cloud systems with variable token lengths. Proposes Argus with token-length prediction and Lyapunov-based offloading optimization. Achieves robust performance in dynamic environments with 30-50% latency reduction.
* `training` `networking` `scaling` [OptiNIC: A Resilient and Tail-Optimal RDMA NIC for Distributed ML Workloads](http://arxiv.org/abs/2512.22743v1)
  > **TL;DR**: Addresses tail latency in collective communication for distributed ML workloads. Proposes OptiNIC, an RDMA transport eliminating retransmissions and in-order delivery, using adaptive timeouts and shifting loss recovery to the ML pipeline. Achieves 3.5x lower 99th-percentile latency and 2x higher time-to-accuracy.
* `multi-modal` `serving` [Modality Inflation: Energy Characterization and Optimization Opportunities for MLLM Inference](http://arxiv.org/abs/2512.22695v1)
  > **TL;DR**: Analyzes energy inefficiency in multimodal LLM inference due to modality inflation. Breaks down stages (vision encoding, prefill, decoding) and identifies GPU underutilization. Proposes stage-wise DVFS optimization, achieving up to 94% overhead reduction with minimal performance impact.
* `RL` `MoE` `disaggregation` [RollArt: Scaling Agentic RL Training via Disaggregated Infrastructure](http://arxiv.org/abs/2512.22560v1)
  > **TL;DR**: Addresses inefficiencies in agentic RL training workloads on disaggregated infrastructure. Proposes RollArc with hardware-affinity workload mapping, fine-grained asynchrony, and statefulness-aware computation. Achieves 1.35-2.05× end-to-end training time reduction on large-scale MoE models.
* `training` `RL` [Role-Based Fault Tolerance System for LLM RL Post-Training](http://arxiv.org/abs/2512.22492v1)
  > **TL;DR**: Addresses fault tolerance for RL post-training of LLMs by proposing RobustRL with role-based isolation for trainer, rollout, and management roles. Introduces role-aware failure detection, non-disruptive recovery, warm standbys, isolated replacement, and dynamic UCX-based reconnection. Achieves 80% ETTR (vs 60%) and 8.4%-17.4% faster training time under failure.
* `serving` `quantization` [Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving](http://arxiv.org/abs/2512.22420v1)
  > **TL;DR**: Addresses the trade-off of fixed-length speculative decoding in LLM inference under varying loads. Proposes Nightjar, a learning-based algorithm that adaptively adjusts speculative length or disables SD. Achieves 14.8% higher throughput and 20.2% lower latency compared to standard SD.
* `serving` `scaling` `storage` [Efficient Multi-Model Orchestration for Self-Hosted Large Language Models](http://arxiv.org/abs/2512.22402v1)
  > **TL;DR**: Proposes Pick and Spin, a Kubernetes-based orchestration framework for self-hosted LLMs with Helm deployment, scale-to-zero automation, and hybrid routing. Achieves 30% lower latency and 33% lower GPU cost per query versus static deployments.
* `storage` `kernel` [Valori: A Deterministic Memory Substrate for AI Systems](http://arxiv.org/abs/2512.22280v1)
  > **TL;DR**: Addresses floating-point non-determinism in AI vector embedding storage. Proposes Valori, a deterministic memory substrate using fixed-point arithmetic (Q16.16) and replayable state machine modeling. Guarantees bit-identical memory states and search results across hardware platforms.
* `edge` `serving` `scaling` [Scalable Cloud-Native Architectures for Intelligent PMU Data Processing](http://arxiv.org/abs/2512.22231v1)
  > **TL;DR**: Proposes a cloud-native framework for low-latency PMU stream processing using distributed edge-cloud orchestration and containerized services. Achieves sub-second response times at scale for real-time power grid analytics.

### 2025-12-29
* `RL` `RAG` `serving` [Agentic Structured Graph Traversal for Root Cause Analysis of Code-related Incidents in Cloud Applications](http://arxiv.org/abs/2512.22113v1)
  > **TL;DR**: Proposes PRAXIS, an LLM-driven orchestrator for root cause analysis using structured graph traversal over service and code dependency graphs. Agents traverse graphs as policies to localize failures. Improves accuracy by 3.1x and reduces tokens by 3.8x versus ReAct baselines.
* `MoE` `training` `networking` [FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion](http://arxiv.org/abs/2512.22036v1)
  > **TL;DR**: Addresses inefficiency in distributed data shuffling for MoE model parallelism. Develops FUSCO, a communication library that fuses transformation and communication with lightweight load balancing. Achieves up to 3.84× speedup over NCCL and reduces training latency by 1.17-1.39×.
* `serving` `offloading` `networking` [Optimizing Resource Allocation for Geographically-Distributed Inference by Large Language Models](http://arxiv.org/abs/2512.21884v1)
  > **TL;DR**: Addresses optimal resource allocation for geographically-distributed LLM inference. Proposes an offline MILP formulation and online algorithm for block placement and request routing. Reduces inference time by up to 50% compared to PETALS in diverse geographical settings.
* `edge` `offloading` `serving` [LIME:Accelerating Collaborative Lossless LLM Inference on Memory-Constrained Edge Devices](http://arxiv.org/abs/2512.21835v1)
  > **TL;DR**: Proposes LIME for collaborative lossless LLM inference on memory-constrained edge devices. Uses interleaved pipeline parallelism, model offloading, and adaptive scheduling. Achieves 1.7–3.7x speedup over state-of-the-art baselines on Jetson devices for LLaMA3.3-70B inference.
* `video` `multi-modal` `offloading` [Hyperion: Low-Latency Ultra-HD Video Analytics via Collaborative Vision Transformer Inference](http://arxiv.org/abs/2512.21730v1)
  > **TL;DR**: Proposes Hyperion, a cloud-device collaborative framework for low-latency Ultra-HD video analytics using vision transformers. Integrates importance scoring, dynamic scheduling, and weighted ensembling to optimize transmission and computation. Achieves 1.61x higher frame rate and 20.2% accuracy gain.
* `offline` `training` `networking` [Embedding Samples Dispatching for Recommendation Model Training in Edge Environments](http://arxiv.org/abs/2512.21615v1)
  > **TL;DR**: Reduces embedding transmission cost in distributed edge training for deep learning recommendation models (DLRMs). Proposes ESD with HybridDis, a hybrid dispatch algorithm optimizing sample-to-worker assignment. Achieves up to 36.76% lower transmission cost and 1.74× training speedup.
* `kernel` `storage` `hardware` [nncase: An End-to-End Compiler for Efficient LLM Deployment on Heterogeneous Storage Architectures](http://arxiv.org/abs/2512.21571v1)
  > **TL;DR**: Introduces nncase, an end-to-end compiler that employs e-graph-based term rewriting to optimize LLM deployment across heterogeneous targets. It integrates Auto Vectorize, Auto Distribution, and Auto Schedule modules for joint computation-data movement optimization. Outperforms MLC LLM and Intel IPEX, matching hand-optimized llama.cpp on CPUs.
* `MoE` `serving` `offloading` [Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism](http://arxiv.org/abs/2512.21487v1)
  > **TL;DR**: Addresses memory-intensive MoE inference by proposing FinDEP, a fine-grained task scheduling algorithm for disaggregated expert parallelism. Optimizes computation/communication task overlap, variable granularity, and scheduling, achieving up to 1.61x throughput improvement over baselines.
* `training` `kernel` `hardware` [Demystifying ARM SME to Optimize General Matrix Multiplications](http://arxiv.org/abs/2512.21473v1)
  > **TL;DR**: Optimizes GEMM performance for deep learning on ARM SME architectures. Proposes MpGEMM with cache-aware partitioning, data packing, and SME-specific micro-kernels. Achieves 1.23x average speedup over vendor library on Apple M4 Pro with DeepSeek/LLaMA workloads.

### [🔥Daily Arxiv: LLM Systems 👉 paper.tju.chat 👈](https://paper.tju.chat)

</div>

**👍Conference Papers on LMSys**: [conference.md](conference.md)

**⚠️NOTE**: Update papers up to last day every morning (8:00 UTC+8) automatically.

**🙋WANT**: Keyword subscription (email); Functional web page.

**🔖TAGS**:`serving` `training` `offline` `thinking` `RL` `MoE` `RAG` `video` `multi-modal` `sparse` `quantization` `offloading` `hardware` `storage` `kernel` `diffusion` `agentic` `edge` `networking`

---
### 2025-12-27
* `training` `offline` `storage` [Optimizing Frequent Checkpointing via Low-Cost Differential for Distributed Training Systems](http://arxiv.org/abs/2509.04084v3)
  > **TL;DR**: Proposes \sysname and \sysnameplus to reduce checkpointing overhead in distributed training. Uses compressed gradients as differential checkpoints and batched writes for efficient persistence, with dynamic tuning. Reduces training time by 89.2% with per-iteration checkpointing.
* `training` `MoE` `quantization` [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](http://arxiv.org/abs/2505.09343v2)
  > **TL;DR**: Addresses hardware scalability challenges in LLM training via co-design with DeepSeek-V3. Proposes MLA attention, MoE optimization, FP8 precision, and network topology innovations. Achieves cost-efficient training on 2048 H800 GPUs with 20% higher throughput versus FP16.
* `kernel` `sparse` `hardware` [Parallel GPU-Enabled Algorithms for SpGEMM on Arbitrary Semirings with Hybrid Communication](http://arxiv.org/abs/2504.06408v2)
  > **TL;DR**: Proposes GPU-accelerated distributed SpGEMM with hybrid communication for efficiency. Implements on CombBLAS with dynamic switching between host/device communication paths. Achieves 2-3x speedup over CPU-only CombBLAS and PETSc for large sparse matrices.
* `kernel` `sparse` `hardware` [Libra: Unleashing GPU Heterogeneity for High-Performance Sparse Matrix Multiplication](http://arxiv.org/abs/2506.22714v2)
  > **TL;DR**: Proposes Libra, a framework to accelerate sparse matrix multiplication on heterogeneous GPUs by optimal task distribution balancing Tensor Cores and CUDA cores. Achieves up to 2.9x speedup in GNN applications.
* `kernel` `training` [LEGO: A Layout Expression Language for Code Generation of Hierarchical Mapping](http://arxiv.org/abs/2505.08091v2)
  > **TL;DR**: Presents LEGO, a layout expression language for optimizing data movement in GPU computations. Generates complex indexing expressions for hierarchical mappings, integrated with Triton, MLIR, and CUDA. Achieves performance competitive with Triton for data mapping optimizations and thread layouts.
* `offloading` `kernel` `storage` [The Impact of Partial Computations on the Red-Blue Pebble Game](http://arxiv.org/abs/2506.10854v2)
  > **TL;DR**: Studies how partial computation steps reduce I/O cost in red-blue pebble game models. Extends game to allow step-by-step input aggregation, enabling smaller I/O costs. Shows I/O cost can decrease by up to a linear factor in certain settings, but remains NP-hard to optimize.
* `training` `hardware` `scaling` [Story of Two GPUs: Characterizing the Resilience of Hopper H100 and Ampere A100 GPUs](http://arxiv.org/abs/2503.11901v4)
  > **TL;DR**: Characterizes resilience of A100 and H100 GPUs in large-scale AI training systems using operational data. Analyzes error rates, recovery mechanisms, and failure impacts across 1,056 GPUs. Projects 5% overprovisioning needed to handle failures at scale, with H100 showing higher memory error rates but critical hardware improvements.
* `training` `storage` `offline` [Redox: Improving I/O Efficiency of Model Training Through File Redirection](http://arxiv.org/abs/2505.16280v2)
  > **TL;DR**: Addresses slow I/O in model training by redirecting file reads. Proposes Redox with batch reading and opportunistic prefetch to minimize wasted disk reads. Achieves up to 4.57x faster training than PyTorch.
* `training` `multi-modal` `scaling` [Efficient Distributed MLLM Training with Cornstarch](http://arxiv.org/abs/2503.11367v3)
  > **TL;DR**: Addresses inefficiency in training multimodal LLMs due to heterogeneity. Proposes Cornstarch framework with frozen-aware pipeline parallelism and token workload-balanced context parallelism. Achieves 2.26x higher training throughput versus state-of-the-art.
* `serving` `scheduling` `kernel` [Optimal Scheduling Algorithms for LLM Inference: Theory and Practice](http://arxiv.org/abs/2508.01002v2)
  > **TL;DR**: Proposes SLO-Aware LLM Inference (SLAI) scheduler optimizing prefill-decode phases with dynamic resource allocation and deadline prioritization. Reduces median TTFT by 53% and increases serving capacity by 26% under TBT constraints.
* `kernel` `training` `RL` [CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning](http://arxiv.org/abs/2507.14111v9)
  > **TL;DR**: Proposes CUDA-L1, a contrastive RL framework for automated CUDA kernel optimization. Uses reinforcement learning with speedup-based rewards to train an LLM-based optimizer, discovering and combining optimization techniques. Achieves average 3.12x speedup on KernelBench kernels over baselines.
* `serving` `RAG` `offloading` [Understanding and Optimizing Multi-Stage AI Inference Pipelines](http://arxiv.org/abs/2504.09775v4)
  > **TL;DR**: Proposes HERMES, a multi-stage LLM inference simulator for complex pipelines (RAG, KV retrieval, reasoning). Models heterogeneous hardware and advanced batching to analyze latency, memory contention, and communication. Achieves insights on optimal batching and architecture for 53% lower latency in hybrid deployments.
* `training` `recommendation` `scaling` [MTGenRec: An Efficient Distributed Training System for Generative Recommendation Models in Meituan](http://arxiv.org/abs/2505.12663v2)
  > **TL;DR**: Addresses inefficient training of generative recommendation models in industry. Proposes MTGenRec with dynamic hash tables, sequence balancing, and feature deduplication for embedding management. Achieves 1.6×–2.4× higher training throughput and scales to over 100 GPUs.
* `serving` `edge` `scaling` [FailLite: Failure-Resilient Model Serving for Resource-Constrained Edge Environments](http://arxiv.org/abs/2504.15856v2)
  > **TL;DR**: Proposes FailLite, a failure-resilient model serving system for edge environments using heterogeneous replica variants, warm/cold replicas, and progressive failover. Achieves 175.5ms mean time to recovery with 0.6% accuracy drop for 27 models.
* `training` `offline` `LoRA` [From Legacy Fortran to Portable Kokkos: An Autonomous Agentic AI Workflow](http://arxiv.org/abs/2509.12443v3)
  > **TL;DR**: Proposes an autonomous agentic AI workflow for translating and optimizing Fortran to Kokkos C++. Specialized LLM agents collaborate to generate performance-portable code across hardware. Achieves optimized codes surpassing Fortran baselines at a cost of a few U.S. dollars with paid models.
* `training` [Maya: Optimizing Deep Learning Training Workloads using GPU Runtime Emulation](http://arxiv.org/abs/2503.20191v2)
  > **TL;DR**: Presents Maya, a GPU runtime emulation system for optimizing deep learning training workloads. It intercepts device API calls from unmodified training code to model performance without semantic gaps. Achieves <5% prediction error and reduces training costs by up to 56%.
* `offloading` `edge` `serving` [To Offload or Not To Offload: Model-driven Comparison of Edge-native and On-device Processing In the Era of Accelerators](http://arxiv.org/abs/2504.15162v3)
  > **TL;DR**: Addresses when to offload ML inference tasks to edge servers versus local device processing. Develops queuing theory-based analytical models for latency prediction and adaptive offloading. Achieves 2.2% mean absolute error in latency modeling across varying workloads and network conditions.
* `training` `scaling` `kernel` [Mapple: A Domain-Specific Language for Mapping Distributed Programs](http://arxiv.org/abs/2507.17087v2)
  > **TL;DR**: Proposes Mapple, a declarative programming interface for mapping distributed programs onto high-performance systems. It resolves dimensionality mismatches via decompose primitives to minimize communication, reducing mapper code by 14× while improving performance up to 1.34× over expert implementations.
* `offloading` `edge` `serving` [Inference Offloading for Cost-Sensitive Binary Classification at the Edge](http://arxiv.org/abs/2509.15674v2)
  > **TL;DR**: Optimizes edge-based binary classification by dynamically offloading uncertain samples to a remote model. Proposes H2T2, an online learning policy adapting confidence thresholds for offloading decisions. Achieves up to 1.7% accuracy gain with 45% cost reduction compared to baselines under distribution shifts.
* `RL` `training` `LoRA` [EcoLoRA: Communication-Efficient Federated Fine-Tuning of Large Language Models](http://arxiv.org/abs/2506.02001v2)
  > **TL;DR**: Proposes EcoLoRA, a communication-efficient FL framework for LLM fine-tuning using round-robin LoRA segment sharing, adaptive sparsification, and encoding. Reduces communication time by up to 79% and total training time by 65% without performance loss.
* `scaling` `storage` `networking` [GPUnion: Autonomous GPU Sharing on Campus](http://arxiv.org/abs/2507.18928v2)
  > **TL;DR**: Addresses campus-wide GPU underutilization by designing GPUnion, an autonomous sharing platform with container-based dispatching, provider-first architecture, and resilient execution via checkpointing/migration. Achieves 30% higher GPU utilization and 94% successful workload migration.
* `training` `serving` `quantization` [AIMeter: Measuring, Analyzing, and Visualizing Energy and Carbon Footprint of AI Workloads](http://arxiv.org/abs/2506.20535v2)
  > **TL;DR**: Presents AIMeter, a toolkit for measuring energy, power, hardware metrics, and carbon emissions in AI workloads. Integrates with AI frameworks to provide fine-grained time-series data and correlation analysis for bottleneck identification. Achieves lightweight overhead and enables sustainability benchmarking.
* `training` [MinatoLoader: Accelerating Machine Learning Training Through Efficient Data Preprocessing](http://arxiv.org/abs/2509.10712v2)
  > **TL;DR**: Addresses inefficient data preprocessing causing GPU idleness in ML training. Proposes MinatoLoader, which prioritizes fast-to-preprocess samples and parallelizes slow samples. Achieves up to 7.5× training acceleration and 90.45% GPU utilization vs. PyTorch's 46.4%.
* `training` `storage` `scaling` [FalconFS: Distributed File System for Large-Scale Deep Learning Pipeline](http://arxiv.org/abs/2507.10367v3)
  > **TL;DR**: Investigates inefficiency of client-side metadata caching in distributed file systems for DL pipelines. Proposes FalconFS with stateless-client architecture using server-side path resolution, hybrid indexing, and concurrent request merging. Achieves up to 12.81× throughput in model training and supports 10,000 NPUs in production.
* `sparse` `inference` `training` [BLaST: High Performance Inference and Pretraining using BLock Sparse Transformers](http://arxiv.org/abs/2507.03117v2)
  > **TL;DR**: Proposes BLaST, a method for block sparsity in Transformers to reduce data movement costs. Achieves 95% weight sparsity with <2.25% accuracy loss, demonstrating 2.2x inference speedup for Llama and 4.45x memory reduction.
* `serving` `scaling` `offline` [Minos: Exploiting Cloud Performance Variation with Function-as-a-Service Instance Selection](http://arxiv.org/abs/2505.12928v3)
  > **TL;DR**: Presents Minos, a system that improves FaaS performance by terminating slow instances after a benchmark test and reusing fast instances. It uses instance selection to reduce execution time and cost, achieving up to 13% speedup in workflow-intensive parts and 4% overall cost reduction.
* `training` `sparse` `scaling` [Intelligent Sampling of Extreme-Scale Turbulence Datasets for Accurate and Efficient Spatiotemporal Model Training](http://arxiv.org/abs/2508.03872v3)
  > **TL;DR**: Proposes SICKLE framework for intelligent sparse subsampling to reduce data volume in extreme-scale turbulence model training. Features maximum entropy sampling and scalable training. Achieves up to 38x lower energy consumption while improving accuracy.
* `serving` `multi-modal` `scaling` [ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving](http://arxiv.org/abs/2502.00937v3)
  > **TL;DR**: Addresses high-cost, low-throughput serving of large multimodal models (LMMs). Proposes ModServe, a modular system with stage disaggreation, modality-aware scheduling, and autoscaling. Achieves 3.3-5.5× higher throughput (25-41.3% cost saving) while meeting tail latency SLOs.
* `offline` `training` [The Streaming Batch Model for Efficient and Fault-Tolerant Heterogeneous Execution](http://arxiv.org/abs/2501.12407v5)
  > **TL;DR**: Addresses inefficiency in heterogeneous GPU/CPU resource utilization for data processing. Proposes the streaming batch model with dynamic partitioning for pipelining in Ray Data. Improves batch inference throughput by 2.5-12x and multimodal training by 31%.
* `kernel` `training` `hardware` [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using $\mathbb{F}_2$](http://arxiv.org/abs/2505.23819v3)
  > **TL;DR**: Proposes Linear Layouts, a tensor layout modeling approach using linear algebra over $\mathbb{F}_2$ for flexible and performant tensor computation. Integrates with Triton compiler to optimize operators/kernels, reducing engineering effort and fixing bugs in existing layout systems.
* `serving` `scaling` [Hierarchical Prediction-based Management for LMaaS Systems](http://arxiv.org/abs/2504.03702v2)
  > **TL;DR**: Proposes PreServe, a hierarchical prediction-based management framework for LMaaS. It uses workload and load predictors to anticipate demand, enabling proactive resource scaling and request routing. Reduces tail latency by 41.3% and resource consumption by 49.38% with minimal overhead.
* `training` `RL` `scaling` [Scaling Multi Agent Reinforcement Learning for Underwater Acoustic Tracking via Autonomous Vehicles](http://arxiv.org/abs/2505.08222v2)
  > **TL;DR**: Proposes GPU-accelerated iterative distillation for scaling multi-agent RL training by transferring high-fidelity simulations to simplified environments. Achieves a 30,000x speedup over Gazebo, enabling large-scale curriculum training with Transformer-based policies invariant to agent count.
* `training` `MoE` `scaling` [SYMI: Efficient Mixture-of-Experts Training via Model and Optimizer State Decoupling](http://arxiv.org/abs/2504.19925v2)
  > **TL;DR**: Addresses inefficiency in MoE training due to token imbalance and expert load variability. Decouples expert parameter placement from optimizer state, statically partitioning optimizer across nodes and dynamically adjusting parameters without migration overhead. Achieves 30.5% and 25.9% faster convergence versus DeepSpeed and FlexMoE.
* `RAG` `storage` `networking` [Exploring Distributed Vector Databases Performance on HPC Platforms: A Study with Qdrant](http://arxiv.org/abs/2509.12384v2)
  > **TL;DR**: Investigates vector database performance for retrieval-augmented generation (RAG) in HPC systems. Measures Qdrant's insertion, indexing, and query latency scaling on 32 workers with biological text embeddings. Reports query latency reduction and throughput scaling on Polaris supercomputer.
* `serving` `disaggregation` `networking` [MoLink: Distributed and Efficient Serving Framework for Large Models](http://arxiv.org/abs/2507.05043v2)
  > **TL;DR**: Addresses efficient distributed serving of large models on heterogeneous consumer-grade GPUs with limited network conditions. Introduces MoLink, incorporating techniques like model chaining and pipelined execution over weak links. Achieves up to 458% throughput improvement and 151% cost-profit margin gain versus state-of-the-art systems.
* `kernel` `video` [TC-GS: A Faster Gaussian Splatting Module Utilizing Tensor Cores](http://arxiv.org/abs/2505.24796v2)
  > **TL;DR**: Proposes TC-GS for accelerating Gaussian Splatting rendering via mapping alpha computation to Tensor Core matrix multiplication. Introduces coordinate transformation to reduce precision errors. Achieves 2.18x speedup over existing methods, totaling 5.6x faster than baselines.
* `MoE` `training` `networking` [FAST: An Efficient Scheduler for All-to-All GPU Communication](http://arxiv.org/abs/2505.09764v2)
  > **TL;DR**: Addresses slow scheduling for skewed All-to-All communications in MoE training. Proposes FAST, an efficient scheduler using intra-server rebalancing and one-to-one scale-out transfers to avoid incast. Reduces synthesis time by orders of magnitude versus prior work in NVIDIA H200/AMD MI300X clusters.
* `training` `MoE` `offloading` [Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices](http://arxiv.org/abs/2508.19078v2)
  > **TL;DR**: Proposes FLUX for federated fine-tuning of MoE LLMs on resource-constrained devices. Features quantization-based profiling, adaptive expert merging, and dynamic expert assignment. Achieves up to 4.75x speedup in time-to-accuracy.
* `training` `quantization` `scaling` [Phantora: Maximizing Code Reuse in Simulation-based Machine Learning System Performance Estimation](http://arxiv.org/abs/2505.01616v3)
  > **TL;DR**: Proposes Phantora, a hybrid GPU cluster simulator for ML training performance estimation. Executes unmodified ML frameworks in containerized emulated environments, intercepting GPU and comms ops. Reduces estimation overhead by eliminating trace capture, supporting modern frameworks on a single GPU.
* `networking` `training` `scaling` [Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering](http://arxiv.org/abs/2504.19519v2)
  > **TL;DR**: Addresses communication bottleneck in multi-GPU training of generative models. Proposes FlashOverlap, using signaling and reordering for tile-wise overlapping of computation and communication without interference. Achieves up to 1.65x speedup over existing methods.
* `training` `quantization` `kernel` [Tempo: Compiled Dynamic Deep Learning with Symbolic Dependence Graphs](http://arxiv.org/abs/2501.05408v3)
  > **TL;DR**: Addresses the challenge of optimizing dynamic dependencies in DL programs with temporal relationships. Introduces Tempo, a system with symbolic dependence graphs enabling compiler optimizations like tiling and fusion. Achieves 7x speedup and 16x lower peak memory usage for models like Llama-3.2-3B.
* `serving` `diffusion` `kernel` [PATCHEDSERVE: A Patch Management Framework for SLO-Optimized Hybrid Resolution Diffusion Serving](http://arxiv.org/abs/2501.09253v2)
  > **TL;DR**: Addresses inefficient diffusion model serving under mixed-resolution workloads. Proposes patch-based workflow with specialized caching and SLO scheduling using lightweight latency prediction. Achieves 30.1% higher SLO satisfaction than state-of-the-art.
* `RL` `LoRA` `scaling` [Fed-SB: A Silver Bullet for Extreme Communication Efficiency and Performance in (Private) Federated LoRA Fine-Tuning](http://arxiv.org/abs/2502.15436v2)
  > **TL;DR**: Proposes Fed-SB, a federated fine-tuning method using LoRA-SB adapter updates with a shared small matrix. Reduces communication costs by 230x while maintaining performance via exact adapter averaging, and enhances privacy by lowering noise requirements.
* `training` `offloading` `scaling` [NotebookOS: A Replicated Notebook Platform for Interactive Training with On-Demand GPUs](http://arxiv.org/abs/2503.20591v2)
  > **TL;DR**: Proposes NotebookOS, a replicated notebook platform for interactive deep learning training by oversubscribing server resources and allocating GPUs only during active cell execution. Achieves 1,187 GPU hours saved in 17.5 hours, improving utilization and interactivity.
* `serving` `kernel` [Lobster: A GPU-Accelerated Framework for Neurosymbolic Programming](http://arxiv.org/abs/2503.21937v2)
  > **TL;DR**: Proposes Lobster, a GPU-accelerated framework for neurosymbolic programming via compilation to an intermediate language with optimization passes. Achieves 3.9x speedup over Scallop in 9 applications.
* `training` `scaling` `networking` [Sailor: Automating Distributed Training over Dynamic, Heterogeneous, and Geo-distributed Clusters](http://arxiv.org/abs/2504.17096v2)
  > **TL;DR**: Addresses efficient distributed training over heterogeneous, geo-distributed clusters. Proposes Sailor with exploration algorithm, simulator, and framework supporting heterogeneity. Optimizes throughput and cost via dynamic resource utilization.
* `kernel` [Segmented Operations using Matrix Multiplications](http://arxiv.org/abs/2506.23906v2)
  > **TL;DR**: Proposes MMV-RAM, a computational model integrating matrix multiplication units for efficient parallel operations. Designs algorithms like segmented scan/sum using MMUs for speculative blocked computations. Achieves speed-ups on Ascend 910B, demonstrated with LLM-relevant kernels like sparse matrix products.
* `serving` `scaling` `training` [Scheduler-Driven Job Atomization](http://arxiv.org/abs/2509.19086v1)
  > **TL;DR**: Addresses GPU cluster inefficiency by atomizing jobs into subjobs for dynamic scheduling. Proposes scheduler-driven job atomization with bidirectional gap advertising and job signaling to fit time-capacity windows. Reduces job rejection rates and improves utilization by minimizing fragmentation.
* `serving` `quantization` [Speculative Decoding via Hybrid Drafting and Rollback-Aware Branch Parallelism](http://arxiv.org/abs/2506.01979v2)
  > **TL;DR**: Proposes SpecBranch, a speculative decoding framework with branch parallelism and hybrid drafting to reduce mutual waiting bubbles. Introduces adaptive draft lengths and rollback-aware branches. Achieves 1.8×–4.5× speedups over auto-regressive decoding and 50% fewer rollback tokens.
* `networking` `training` `inference` [Whack-a-Mole: Deterministic Packet Spraying Across Multiple Network Paths](http://arxiv.org/abs/2509.18519v1)
  > **TL;DR**: Proposes Whack-a-Mole, a deterministic packet spraying algorithm for multipath transport to reduce tail latency and imbalance in distributed AI training/inference. Uses bit-reversal counter and dynamic allocation to achieve O(log m) discrepancy bound, improving GPU utilization and collective completion time.
* `kernel` [XaaS Containers: Performance-Portable Representation With Source and IR Containers](http://arxiv.org/abs/2509.17914v1)
  > **TL;DR**: Proposes Source and IR Containers for performance-portable HPC deployment, delaying compilation until runtime to incorporate hardware-specific optimizations. Uses LLM-assisted discovery of specializations and compile-time optimization to achieve peak performance. Prototype shows performance comparable to system-specialized builds.
* `scaling` `storage` `training` [Enhancing Cluster Scheduling in HPC: A Continuous Transfer Learning for Real-Time Optimization](http://arxiv.org/abs/2509.22701v1)
  > **TL;DR**: Proposes continuous transfer learning for real-time cluster task scheduling optimization. Dynamically adapts scheduling policy during operations to reduce retraining. Achieves over 99% accuracy with reduced overhead and latency on HPC workload traces.
* `MoE` `training` `serving` [LongCat-Flash Technical Report](http://arxiv.org/abs/2509.01322v2)
  > **TL;DR**: Presents LongCat-Flash, a 560B MoE model with novel architectures (Zero-computation Experts and Shortcut-connected MoE) for efficient training and inference. Achieves training on 20T tokens in 30 days and inference at 100+ TPS costing $0.70 per million output tokens.
* `edge` `scheduling` `networking` [KubeDSM: A Kubernetes-based Dynamic Scheduling and Migration Framework for Cloud-Assisted Edge Clusters](http://arxiv.org/abs/2501.07130v2)
  > **TL;DR**: Aims to improve resource management in cloud-assisted edge clusters for latency-sensitive apps. Proposes KubeDSM, a Kubernetes-based framework with batch scheduling and live migration to minimize fragmentation. Achieves higher average edge ratio and lower standard deviation in edge ratios.
* `diffusion` `edge` `inference` [Conditional Prior-based Non-stationary Channel Estimation Using Accelerated Diffusion Models](http://arxiv.org/abs/2509.15182v1)
  > **TL;DR**: Proposes a diffusion-based method for non-stationary wireless channel estimation, using conditional priors and accelerated sampling to improve accuracy. Achieves lower NMSE than baselines across all SNRs on 3GPP benchmarks.
* `training` `scaling` [Cost-Performance Analysis: A Comparative Study of CPU-Based Serverless and GPU-Based Training Architectures](http://arxiv.org/abs/2509.14920v1)
  > **TL;DR**: Compares serverless (SPIRT) and GPU-based training architectures for distributed ML. SPIRT leverages parallel batch processing and in-database operations via RedisAI to reduce costs and training times. Achieves significant reductions in training time and communication overhead versus traditional systems.
* `scaling` [Precomputed Dominant Resource Fairness](http://arxiv.org/abs/2507.08846v3)
  > **TL;DR**: Addresses efficient computation of fair multi-resource allocation in distributed systems. Proposes Precomputed Dominant Resource Fairness, an approximation algorithm reducing steps to compute Dominant Resource Fairness allocations. Achieves comparable fairness with fewer computational iterations.
* `MoE` `training` `inference` [When MoE Meets Blockchain: A Trustworthy Distributed Framework of Large Models](http://arxiv.org/abs/2509.12141v2)
  > **TL;DR**: Proposes a blockchain-based trustworthy distributed MoE framework for LMs to enhance security against data manipulation attacks during training and inference. Integrates edge computing with decentralized verification, improving robustness while maintaining learning performance.
* `scaling` `training` [Cloud abstractions for AI workloads](http://arxiv.org/abs/2501.09562v2)
  > **TL;DR**: Proposes HarmonAIze, a framework for cloud abstraction to coordinate AI workload optimization between tenants and providers. Focuses on partitioning, scheduling, and fault tolerance for improved resource efficiency and resiliency. Achieves enhanced performance in multi-tenant environments.
* `training` `scaling` `scheduling` [GFS: A Preemption-aware Scheduling Framework for GPU Clusters with Predictive Spot Instance Management](http://arxiv.org/abs/2509.11134v1)
  > **TL;DR**: Addresses inefficiencies in GPU scheduling for LLM workloads with preemptible spot instances. Proposes GFS framework with demand forecasting, dynamic spot quota allocation, and preemption-aware scheduling. Reduces evictions by 33.0% and queuing delays by 44.1% for LP tasks.
* `training` `offloading` `sparse` [Chameleon: Taming Dynamic Operator Sequences for Memory-Intensive LLM Training](http://arxiv.org/abs/2509.11076v1)
  > **TL;DR**: Addresses memory-intensive LLM training under dynamic operator sequences in Eager Mode. Proposes Chameleon with online profiler, swap policy generation, and optimized execution. Reduces profiling overhead by 84.25%, enables 4x larger models, improves performance by up to 38.94% over recomputation.
* `scaling` `offloading` `networking` [Learning In Chaos: Efficient Autoscaling and Self-Healing for Multi-Party Distributed Training](http://arxiv.org/abs/2505.12815v2)
  > **TL;DR**: Addresses slow autoscaling and recovery in multi-party distributed training under churn. Proposes Chaos with multi-neighbor state replication, model sharding, and greedy shard assignment that reduces MINLP to MILP. Achieves scale-out delay reduction vs. Pollux/Elan and handles events within 20ms with lowest idle time.
* `serving` `edge` `scaling` [SynergAI: Edge-to-Cloud Synergy for Architecture-Driven High-Performance Orchestration for AI Inference](http://arxiv.org/abs/2509.12252v1)
  > **TL;DR**: Proposes SynergAI, a framework for architecture-aware scheduling of inference workloads across heterogeneous edge-to-cloud infrastructures. Integrates offline and online policies to dynamically allocate tasks, minimizing QoS violations. Reduces violations by 2.4x compared to state-of-the-art solutions.
* `training` `networking` `sparse` [Towards Communication-Efficient Decentralized Federated Graph Learning over Non-IID Data](http://arxiv.org/abs/2509.08409v1)
  > **TL;DR**: Addresses high communication costs in decentralized federated graph learning. Proposes Duplex, a framework that jointly optimizes network topology sparsification and graph neighbor sampling. Reduces training time by 20.1-48.8% and communication costs by 16.7-37.6% while improving accuracy by 3.3-7.9%.
* `video` `multi-modal` `kernel` [Déjà Vu: Efficient Video-Language Query Engine with Learning-based Inter-Frame Computation Reuse](http://arxiv.org/abs/2506.14107v2)
  > **TL;DR**: Proposes Déjà Vu to accelerate ViT-based Video-Language Models via inter-frame computation reuse. Introduces ReuseViT to detect reuse opportunities and employs memory-compute joint compaction to convert FLOP savings. Achieves up to 2.64× embedding speedup within 2% accuracy error.
* `kernel` `hardware` [Dynamic Memory Management on GPUs with SYCL](http://arxiv.org/abs/2504.18211v2)
  > **TL;DR**: Addresses efficient dynamic memory allocation in GPU kernels for open standards. Ports Ouroboros from CUDA to SYCL for cross-platform support. Achieves performance comparable to CUDA implementation on multiple backends.
* `MoE` `training` `hardware` [Accelerating Frontier MoE Training with 3D Integrated Optics](http://arxiv.org/abs/2510.15893v1)
  > **TL;DR**: Explores using 3D-stacked photonics to overcome interconnect limitations for large-scale MoE training. Proposes GPUs with 3D co-packaged optics for high-bandwidth multi-rack scaling. Achieves 8× scale-up capability and 2.7× reduction in training time for trillion-parameter models.
* `hardware` `kernel` `training` [ELK: Exploring the Efficiency of Inter-core Connected AI Chips with Deep Learning Compiler Techniques](http://arxiv.org/abs/2507.11506v2)
  > **TL;DR**: Proposes ELK, a DL compiler framework to optimize compute, communication, and I/O for inter-core connected AI chips. Uses inductive operator scheduling and cost-aware on-chip memory allocation to generate globally optimized execution plans. Achieves 94% of ideal roofline performance.
* `training` `scaling` `MoE` [Scaling Intelligence: Designing Data Centers for Next-Gen Language Models](http://arxiv.org/abs/2506.15006v3)
  > **TL;DR**: Proposes a co-design framework for data centers to scale training of trillion-parameter LLMs. Evaluates network topologies, parallelism strategies, and hardware specs to optimize Model FLOPS Utilization. Achieves accurate performance predictions (within 10% error) and improves throughput with FullFlat networks for both dense and sparse (MoE) models.
* `training` `networking` `storage` [Hiding Latencies in Network-Based Image Loading for Deep Learning](http://arxiv.org/abs/2503.22643v2)
  > **TL;DR**: Addresses data loading bottlenecks in deep learning caused by network/storage latencies. Proposes storing data in NoSQL databases and using out-of-order incremental prefetching for high-latency networks. Achieves high-throughput image loading even over intercontinental networks.
* `edge` `networking` `storage` [Ratio1 -- AI meta-OS](http://arxiv.org/abs/2509.12223v1)
  > **TL;DR**: Proposes Ratio1, a decentralized meta-OS for distributed AI execution across edge devices. Integrates blockchain, federated learning, and novel storage/homomorphic encryption to pool idle resources into a global supercomputer. Lowers deployment barriers and improves cost-efficiency compared to centralized systems.
* `kernel` `hardware` `training` [Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis](http://arxiv.org/abs/2501.12084v2)
  > **TL;DR**: Analyzes performance of NVIDIA Hopper GPU for compute-intensive workloads. Microbenchmarks memory subsystem, tensor cores (FP8, async wgmma), TMA for async data movement, and DPX instructions. TMA achieves 1.5x matmul speedup; FP8 doubles FP16 performance; DPX accelerates bio-algorithm 4.75x.
* `training` `RL` `edge` [Semi-decentralized Federated Time Series Prediction with Client Availability Budgets](http://arxiv.org/abs/2509.03660v1)
  > **TL;DR**: Proposes FedDeCAB, a semi-decentralized client selection method for federated time-series prediction with client availability budgets. Uses probabilistic client ranking and neighbor model parameter sharing to handle disconnections, reducing communication overhead. Achieves effective convergence under data heterogeneity, with 30% lower communication costs in experiments on real-world taxi and vessel trajectory datasets.
* `training` `scaling` `kernel` [ORBIT-2: Scaling Exascale Vision Foundation Models for Weather and Climate Downscaling](http://arxiv.org/abs/2505.04802v2)
  > **TL;DR**: Scales vision foundation models for climate downscaling via Reslim architecture and TILES algorithm, reducing self-attention complexity to near-linear. Achieves 4.1 exaFLOPS on 65,536 GPUs with 74-98% strong scaling efficiency for 0.9 km resolution.
* `hardware` `storage` `kernel` [Disaggregated Design for GPU-Based Volumetric Data Structures](http://arxiv.org/abs/2503.07898v2)
  > **TL;DR**: Proposes a disaggregated design for GPU-based volumetric data structures to balance locality with occupancy, communication, and fusion. The design reduces communication overhead on multi-GPU systems, mitigates register pressure, and enables kernel fusion. Achieves up to 3× speedup over state-of-the-art LBM solvers.
* `training` `kernel` `hardware` [COMET: A Framework for Modeling Compound Operation Dataflows with Explicit Collectives](http://arxiv.org/abs/2509.00599v1)
  > **TL;DR**: Proposes COMET, a framework for modeling and optimizing dataflows of compound operations (e.g., GEMM-Softmax) on ML accelerators, with explicit collective communication modeling. Achieves up to 3.46× speedup for GEMM-LayerNorm compared to unfused baselines.
* `networking` `training` [FlexLink: Boosting your NVLink Bandwidth by 27% without accuracy concern](http://arxiv.org/abs/2510.15882v1)
  > **TL;DR**: Addresses heterogeneous inter-GPU communication bottleneck in LLM training. Proposes FlexLink, a collective communication framework that aggregates NVLink, PCIe, and RDMA NICs with two-stage adaptive load balancing. Boosts AllGather bandwidth by 27% via offloading 2-22% traffic to idle links.
* `serving` `MoE` `networking` [Learning to Shard: RL for Co-optimizing the Parallelism Degrees and Per-operator Sharding Dimensions in Distributed LLM Inference](http://arxiv.org/abs/2509.00217v1)
  > **TL;DR**: Co-optimizes parallelism degrees and sharding dimensions for distributed LLM inference via RL with attention-based policy. Achieves up to 3.5x throughput gains over baselines on MoE models with 1.6T parameters on H100 clusters.
* `serving` `kernel` `sparse` [TinyServe: Query-Aware Cache Selection for Efficient LLM Serving](http://arxiv.org/abs/2509.12211v1)
  > **TL;DR**: Reduces LLM serving KV cache overhead via query-aware page selection. Introduces metadata-driven KV sparsity and integrated CUDA kernel for selective loading. Achieves 3.4x speedup and 2x memory savings with minimal accuracy loss.
* `scheduling` `kernel` `hardware` [IsoSched: Preemptive Tile Cascaded Scheduling of Multi-DNN via Subgraph Isomorphism](http://arxiv.org/abs/2509.12208v1)
  > **TL;DR**: Develops IsoSched for preemptive multi-DNN scheduling on tile spatial architectures. Combines ILP-based formulation, layer concatenate/split, and accelerated subgraph isomorphism with CSR encoding. Achieves higher latency-bound throughput and critical task satisfaction over baselines, with speedups in energy efficiency.
* `training` `storage` `recommendation` [AGILE: Lightweight and Efficient Asynchronous GPU-SSD Integration](http://arxiv.org/abs/2504.19365v3)
  > **TL;DR**: Proposes AGILE, an asynchronous GPU-centric I/O library with HBM cache to overlap computation and I/O for memory-intensive workflows like recommendation systems. Achieves up to 1.88× speedup by eliminating thread stalls and integrating software caching.
* `networking` `kernel` `training` [LCI: a Lightweight Communication Interface for Efficient Asynchronous Multithreaded Communication](http://arxiv.org/abs/2505.01864v2)
  > **TL;DR**: Proposes LCI, a lightweight communication library for efficient asynchronous multithreaded communication in distributed systems. Features atomic data structures, fine-grained locks, and network optimizations. Outperforms existing libraries on Infiniband and Slingshot-11, exceeding multi-process performance in multithreaded scenarios.
* `scheduling` `scaling` `edge` [A User-centric Kubernetes-based Architecture for Green Cloud Computing](http://arxiv.org/abs/2509.13325v1)
  > **TL;DR**: Proposes a Kubernetes-based system to reduce cloud computing emissions by scheduling workloads based on green energy availability. Employs a carbon intensity forecaster to exploit regional and temporal energy variations. Achieved 13% reduction in emissions compared to round-robin scheduler.
* `training` `scaling` `networking` [BandPilot: Towards Performance-Aware GPU Dispatching in AI Clusters](http://arxiv.org/abs/2506.15595v3)
  > **TL;DR**: Proposes BandPilot, a performance-aware GPU dispatching system for AI clusters. Uses a Transformer to predict NCCL bandwidth and a heuristic search for allocations. Achieves 12-31% higher bandwidth efficiency over topology-aware dispatchers.
* `training` `kernel` `networking` [MSCCL++: Rethinking GPU Communication Abstractions for Cutting-edge AI Applications](http://arxiv.org/abs/2504.09014v3)
  > **TL;DR**: Proposes MSCCL++, a GPU communication library separation to improve portability and performance. Introduces primitive and higher-level interfaces enabling custom optimizations. Achieves up to 5.4× speedup in collective communication and 15% improvement in AI inference workloads.
* `agentic` `RAG` [ASIC-Agent: An Autonomous Multi-Agent System for ASIC Design with Benchmark Evaluation](http://arxiv.org/abs/2508.15940v1)
  > **TL;DR**: Proposes ASIC-Agent, a multi-agent system for ASIC design that enhances LLMs with specialized sub-agents and a vector database. Integrates RTL generation, verification, and chip hardening tools within a sandbox. When powered by Claude 4 Sonnet, it successfully automates ASIC design tasks, accelerating workflows.
* `edge` `networking` `serving` [Edge-Cloud Collaborative Computing on Distributed Intelligence and Model Optimization: A Survey](http://arxiv.org/abs/2505.01821v4)
  > **TL;DR**: Surveys edge-cloud collaborative computing for AI/LLMs, focusing on model optimization (compression, adaptation) and AI-driven resource management to reduce latency and improve energy efficiency in applications like autonomous driving and healthcare.
* `kernel` `quantization` [SGEMM-cube: Emulating FP32 GEMM on Ascend NPUs Using FP16 Cube Units with Precision Recovery](http://arxiv.org/abs/2507.23387v3)
  > **TL;DR**: Proposes SGEMM-cube, an algorithm using FP16 units to emulate FP32 GEMM via operand decomposition and tunable scaling. Includes cache-aware blocking and double-buffering to achieve 77% of theoretical FP32 peak performance on Ascend NPU lacking native support.
* `edge` `offloading` `RL` [Oranits: Mission Assignment and Task Offloading in Open RAN-based ITS using Metaheuristic and Deep Reinforcement Learning](http://arxiv.org/abs/2507.19712v2)
  > **TL;DR**: Addresses mission assignment and task offloading for edge-based autonomous vehicles. Proposes Oranits with metaheuristic (CGG-ARO) and DRL (MA-DDQN) approaches for cooperative offloading. MA-DDQN improves mission completions by 11.0% and overall benefit by 12.5%.
* `RAG` `serving` [From Data Center IoT Telemetry to Data Analytics Chatbots -- Virtual Knowledge Graph is All You Need](http://arxiv.org/abs/2506.22267v2)
  > **TL;DR**: Designs a chatbot for IoT data analytics using LLMs with dynamically generated Virtual Knowledge Graphs (VKGs) to translate natural language to SPARQL queries. Achieves 92.5% accuracy and reduces latency by 85% to 3.03s compared to direct LLM-to-NoSQL.
* `serving` `training` `scaling` [An Empirical Study of Production Incidents in Generative AI Cloud Services](http://arxiv.org/abs/2504.08865v2)
  > **TL;DR**: Analyzes production incidents in generative AI cloud services to understand reliability challenges. Examines symptoms, root causes, and resolutions from incident data. Identifies research gaps in incident detection and mitigation for scalable GenAI systems.
* `RL` `offloading` `training` [Flexible Personalized Split Federated Learning for On-Device Fine-Tuning of Foundation Models](http://arxiv.org/abs/2508.10349v1)
  > **TL;DR**: Proposes FlexP-SFL for personalized federated fine-tuning of foundation models via split learning. Allows clients to offload model portions to server based on resource constraints, with alignment strategy for personalized models. Achieves higher accuracy and efficiency than baselines.
* `training` `networking` [Characterization of GPU TEE Overheads in Distributed Data Parallel ML Training](http://arxiv.org/abs/2501.11771v3)
  > **TL;DR**: Characterizes performance overheads of GPU TEEs in distributed data parallel ML training. Focuses on encryption/decryption and MAC costs during ring-all-reduce operations. Shows a runtime increase of up to 41.6x per training iteration with 4 GPUs versus non-TEE training.
* `networking` `scaling` `storage` [Faster Multi-Source Reachability and Approximate Distances via Shortcuts, Hopsets and Matrix Multiplication](http://arxiv.org/abs/2507.13470v2)
  > **TL;DR**: Introduces centralized and parallel algorithms for multi-source reachability and approximate distances. Leverages shortcut constructions, matrices, and small-separator graphs for efficient computation. Achieves time complexity $̂ O(n^{1 + √{2}{3} ω(σ)})$ improving over prior bounds in key $σ$ ranges.
* `training` `offloading` `kernel` [Performant Automatic BLAS Offloading on Unified Memory Architecture with OpenMP First-Touch Style Data Movement](http://arxiv.org/abs/2501.00279v4)
  > **TL;DR**: Develops SCILIB-Accel, an automatic tool to offload BLAS operations to GPUs on unified memory architectures using a first-touch policy and dynamic binary instrumentation. Minimizes data transfers and requires no code modifications. Achieves up to 3x speedup on quantum physics benchmarks.
* `training` `sparse` `kernel` [AMPED: Accelerating MTTKRP for Billion-Scale Sparse Tensor Decomposition on Multiple GPUs](http://arxiv.org/abs/2507.15121v2)
  > **TL;DR**: Accelerates MTTKRP for billion-scale sparse tensor decomposition on multi-GPU systems. Introduces a partitioning strategy with dynamic load balancing to distribute computation and minimize GPU idle time. Achieves 5.1x geometric mean speedup on real-world tensors over state-of-the-art baselines using 4 GPUs.
* `serving` `MoE` `disaggregation` [xDeepServe: Model-as-a-Service on Huawei CloudMatrix384](http://arxiv.org/abs/2508.02520v5)
  > **TL;DR**: Proposes xDeepServe, a Transformerless disaggregated architecture for serving MoE LLMs on SuperPod-scale hardware. Decomposes models into attention, feedforward, and MoE units executed independently with XCCL for efficient communication. Achieves scalable inference across hundreds of NPUs without performance sacrifice.
* `storage` `quantization` [Floating-Point Data Transformation for Lossless Compression](http://arxiv.org/abs/2506.18062v2)
  > **TL;DR**: Proposes Typed Data Transformation (TDT) for lossless compression of floating-point data by grouping correlated bytes. TDT improves geometric mean compression ratio by 1.16× and throughput by up to 3.79× over zstd, with applications to language model weights storage.
* `kernel` `hardware` `quantization` [High-Performance and Power-Efficient Emulation of Matrix Multiplication using INT8 Matrix Engines](http://arxiv.org/abs/2508.03984v2)
  > **TL;DR**: Proposes high-performance emulation methods for SGEMM/DGEMM using INT8 matrix engines. Leverages low-precision engines to emulate higher precision matrix multiplications. Achieves up to 3.0x speedup and 154% power efficiency improvement over native implementations.
* `training` `MoE` `multi-modal` [VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo](http://arxiv.org/abs/2508.02317v3)
  > **TL;DR**: Addresses inefficiency in training omni-modal LLMs with heterogeneous architectures. Proposes VeOmni, a modular framework using model-centric distributed recipes decoupling communication from computation for 3D parallelism. Achieves 2,800 tokens/sec/GPU throughput on a 30B MoE model at 160K context length on 128 GPUs.
* `training` `scaling` [Tesserae: Scalable Placement Policies for Deep Learning Workloads](http://arxiv.org/abs/2508.04953v1)
  > **TL;DR**: Proposes novel placement policies for DL training jobs using graph matching to minimize migration overhead and improve packing. Integrates into Tesserae scheduler, achieving up to 1.62× lower average job completion time and 1.15× lower makespan.
* `recommendation` `training` `sparse` [Two-dimensional Sparse Parallelism for Large Scale Deep Learning Recommendation Model Training](http://arxiv.org/abs/2508.03854v1)
  > **TL;DR**: Proposes a 2D sparse parallelism for large-scale deep learning recommendation model training to address embedding table scalability. Combines model and data parallelism with a momentum-scaled optimizer to reduce communication and memory use. Achieves near-linear scaling up to 4000 GPUs.
* `offloading` `training` `quantization` [ZenFlow: Enabling Stall-Free Offloading Training via Asynchronous Updates](http://arxiv.org/abs/2505.12242v3)
  > **TL;DR**: Addresses GPU stalls during offloaded LLM fine-tuning by prioritizing parameter updates. Proposes ZenFlow with asynchronous, decoupled gradient processing exploiting spatial-temporal locality for low-overhead selection. Achieves 5x speedup, 85% stall reduction, and 2x lower PCIe traffic.
* `training` `scaling` `quantization` [DHO$_2$: Accelerating Distributed Hybrid Order Optimization via Model Parallelism and ADMM](http://arxiv.org/abs/2505.00982v2)
  > **TL;DR**: Proposes DHO$_2$, a distributed hybrid order optimizer for DNN training acceleration. Utilizes model parallelism and ADMM for efficient curvature information calculation and updates, reducing memory burden per device. Achieves near-linear memory reduction and $1.4\times\sim2.1\times$ training time speedup compared to conventional optimizers.
* `hardware` `kernel` [TeraNoC: A Multi-Channel 32-bit Fine-Grained, Hybrid Mesh-Crossbar NoC for Efficient Scale-up of 1000+ Core Shared-L1-Memory Clusters](http://arxiv.org/abs/2508.02446v1)
  > **TL;DR**: Proposes TeraNoC, a hybrid mesh-crossbar NoC for scaling shared-L1-memory clusters to 1000+ cores. Balances traffic via router remapper and multi-channel design to reduce area and power. Achieves 37.8% area reduction and up to 98.7% higher area efficiency versus crossbar-only.
* `quantization` `networking` `training` [FlashCommunication V2: Bit Splitting and Spike Reserving for Any Bit Communication](http://arxiv.org/abs/2508.03760v1)
  > **TL;DR**: Addresses communication bottlenecks in distributed training of LLMs. Proposes FlashCommunication V2 with bit splitting for arbitrary bit-width transmission and spike reserving for 2-bit quantization. Achieves up to 3.2× speedup in AllReduce and 2× in All2All communication.
* `scaling` `networking` `storage` [Self-assessment approach for resource management protocols in heterogeneous computational systems](http://arxiv.org/abs/2508.02202v1)
  > **TL;DR**: Proposes a heuristics-based self-assessment approach for resource management in heterogeneous systems. Dynamically weights requirements, computes node capacity for requests, and supports extensible resource types. Achieves straightforward resource estimation with scalability and extensibility.
* `serving` `scaling` [Learning Unified System Representations for Microservice Tail Latency Prediction](http://arxiv.org/abs/2508.01635v1)
  > **TL;DR**: Proposes USRFNet, a deep learning model using GNNs and gMLPs to unify traffic and resource features for microservice P95 latency prediction. Achieves higher accuracy in predicting tail latency during large-scale stress tests than SOTA baselines.
* `serving` `diffusion` `sparse` [MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models](http://arxiv.org/abs/2503.11972v2)
  > **TL;DR**: Addresses the trade-off between latency and quality in diffusion model serving. Proposes MoDM, a caching system that uses smaller models for cache-hit requests and larger models for cache-misses. Achieves 2.5x reduction in average serving time while preserving image quality.
* `training` `scaling` [CarbonScaling: Extending Neural Scaling Laws for Carbon Footprint in Large Language Models](http://arxiv.org/abs/2508.06524v1)
  > **TL;DR**: Extends neural scaling laws to include carbon footprint in LLM training. Integrates scaling, hardware evolution, parallelism optimization, and carbon models. Finds hardware scaling has diminishing returns for large LLMs and that critical batch size optimization reduces carbon by up to 40% for billion-parameter models.
* `training` `quantization` `offloading` [Compression-Induced Communication-Efficient Large Model Training and Inferencing](http://arxiv.org/abs/2508.00960v1)
  > **TL;DR**: Proposes phantom parallelism to reduce energy consumption in large model training. Introduces new forward/backward propagation operators as custom autograd operations, minimizing bandwidth and FLOP counts. Achieves ~50% energy reduction vs. conventional tensor parallelism on FFNs across 256 GPUs.
* `serving` `scaling` `offloading` [Tetris: Efficient Intra-Datacenter Calls Packing for Large Conferencing Services](http://arxiv.org/abs/2508.00426v1)
  > **TL;DR**: Addresses inefficient call packing in datacenter conferencing services leading to hot Media Processors. Proposes Tetris, which optimizes initial call assignments using historical data and migrates calls via linear optimization to balance CPU. Reduces participants on hot MPs by 2.5X using real trace of 10M+ calls.
* `networking` `edge` [Service Discovery-Based Hybrid Network Middleware for Efficient Communication in Distributed Robotic Systems](http://arxiv.org/abs/2508.00947v1)
  > **TL;DR**: Addresses communication inefficiency in distributed robotic systems. Proposes RIMAOS2C, a service discovery-based middleware with Message Bridge for optimized data flow and shared memory. Improves large-data transmission efficiency by 36-40% and reduces latency variation by 42-906%.
* `RL` `edge` `networking` [Satellite Federated Fine-Tuning for Foundation Models in Space Computing Power Networks](http://arxiv.org/abs/2504.10403v3)
  > **TL;DR**: Proposes a satellite-ground collaborative federated fine-tuning framework for large remote sensing foundation models. Decomposes and allocates model components to satellites and ground stations with tailored communication strategies. Achieves 33% reduction in training time.
* `training` `RL` `edge` [Learning Like Humans: Resource-Efficient Federated Fine-Tuning through Cognitive Developmental Stages](http://arxiv.org/abs/2508.00041v1)
  > **TL;DR**: Proposes Developmental Federated Tuning (DevFT) for resource-efficient LLM fine-tuning on edge devices. Stages submodel growth with knowledge transfer and introduces layer grouping/fusion for efficiency. Achieves 4.59× faster convergence and 10.67× lower communication overhead.
* `training` `storage` `scaling` [Data Readiness for Scientific AI at Scale](http://arxiv.org/abs/2507.23018v1)
  > **TL;DR**: Proposes a scientific data readiness framework for AI training at scale on HPC. Introduces Data Readiness Levels and Processing Stages to standardize workflows supporting transformer-based foundation models. Enables scalable and reproducible preprocessing for scientific domains.
* `offloading` `serving` `inference` [A Survey on Large Language Model Acceleration based on KV Cache Management](http://arxiv.org/abs/2412.19442v3)
  > **TL;DR**: Survey on Key-Value cache management techniques for accelerating LLM inference. Covers token-level, model-level, and system-level optimizations (e.g., KV cache quantization, offloading, scheduling). Highlights reduced computational and memory demands for real-world LLM deployments with long-context and real-time constraints.
* `training` `scaling` `networking` [The Performance of Low-Synchronization Variants of Reorthogonalized Block Classical Gram--Schmidt](http://arxiv.org/abs/2507.21791v1)
  > **TL;DR**: Addresses the communication bottleneck in distributed QR factorization for Krylov solvers. Proposes low-synchronization variants BCGSI+P-1S and BCGSI+P-2S of Gram-Schmidt orthogonalization. Achieves 4× speedup for BCGSI+P-1S and 2× for BCGSI+P-2S compared to classical BCGSI+.
* `MoE` `serving` `disaggregation` [MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism](http://arxiv.org/abs/2504.02263v4)
  > **TL;DR**: Proposes MegaScale-Infer for efficient MoE model serving by disaggregating attention/FFN modules and using ping-pong pipeline parallelism. Achieves up to 1.90x higher per-GPU throughput than state-of-art systems.
* `kernel` [MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation](http://arxiv.org/abs/2507.17773v2)
  > **TL;DR**: Presents MultiKernelBench, a multi-platform benchmark for LLM-based deep learning kernel generation. Introduces a modular backend abstraction for hardware extensibility and category-aware prompting for improved generation. Evaluates seven LLMs across 285 tasks on GPU, NPU, TPU platforms.
* `RAG` `networking` `storage` [CleANN: Efficient Full Dynamism in Graph-based Approximate Nearest Neighbor Search](http://arxiv.org/abs/2507.19802v1)
  > **TL;DR**: Proposes CleANN, a dynamic graph-based ANNS index with semi-lazy memory cleaning and adaptive consolidation for efficient concurrent updates and searches. Achieves 7-1200x throughput improvement while maintaining query quality on million-scale in-memory datasets.
* `serving` `agentic` `multi-modal` [Efficient and Scalable Agentic AI with Heterogeneous Systems](http://arxiv.org/abs/2507.19635v1)
  > **TL;DR**: Addresses efficient deployment of AI agent workloads on heterogeneous hardware. Proposes a framework for optimizing execution graphs with cost models, an MLIR-based compilation system, and a dynamic orchestration system. Achieves significant TCO benefits by leveraging older GPUs with newer accelerators.
* `edge` `offloading` `serving` [Deadline-Aware Joint Task Scheduling and Offloading in Mobile Edge Computing Systems](http://arxiv.org/abs/2507.18864v1)
  > **TL;DR**: Proposes deadline-aware joint scheduling and offloading for mobile edge computing to meet task deadlines efficiently. Designs an optimal scheduling algorithm with O(n log n) complexity and an online variant for dynamic arrivals. Achieves improvements in service ratio and scheduling cost.
* `serving` `offline` `hardware` [PPipe: Efficient Video Analytics Serving on Heterogeneous GPU Clusters via Pool-Based Pipeline Parallelism](http://arxiv.org/abs/2507.18748v1)
  > **TL;DR**: Investigates how to efficiently serve video analytics inference on heterogeneous GPU clusters using pipeline parallelism. Proposes PPipe, a system with MILP-based control plane and adaptive batching for pipeline parallelism across GPU tiers. Achieves 32.2%-75.1% higher throughput than baselines.
* `kernel` `training` `serving` [CUTHERMO: Understanding GPU Memory Inefficiencies with Heat Map Profiling](http://arxiv.org/abs/2507.18729v1)
  > **TL;DR**: Introduces cuThermo, a profiling tool for GPU memory inefficiencies. Uses heat maps based on warp counts to identify access patterns. Achieves up to 721.79% performance improvement on optimized applications.
* `serving` `edge` `networking` [AI Flow: Perspectives, Scenarios, and Approaches](http://arxiv.org/abs/2506.12479v3)
  > **TL;DR**: Proposes AI Flow, a device-edge-cloud framework for efficient large model inference. Introduces familial models with aligned features for collaborative execution and emergent intelligence via enhanced networking. Reduces latency and improves scalability for ubiquitous AI services.
* `edge` `serving` `RL` [FCPO: Federated Continual Policy Optimization for Real-Time High-Throughput Edge Video Analytics](http://arxiv.org/abs/2507.18047v1)
  > **TL;DR**: Proposes a federated continual RL approach (FCPO) for edge video analytics to optimize batch sizes, resolutions, and processing threads in dynamic environments. Achieves 5× higher throughput and 60% lower latency with 10× less memory than SOTA RL schedulers.
* `scaling` `inference` `networking` [C-Koordinator: Interference-aware Management for Large-scale and Co-located Microservice Clusters](http://arxiv.org/abs/2507.18005v1)
  > **TL;DR**: Addresses resource competition and interference in large-scale co-located microservice clusters. Proposes C-Koordinator with CPI-based interference prediction and mitigation strategies. Reduces application latency by up to 36.1% across all response time percentiles.
* `training` `scaling` `networking` [PowerTrip: Exploiting Federated Heterogeneous Datacenter Power for Distributed ML Training](http://arxiv.org/abs/2507.17904v1)
  > **TL;DR**: Examines optimizing geo-distributed LLM training under power constraints and network latency. Proposes PowerTrip, a dynamic site selection using power-to-cost heuristic and marginal gain. Reduces time-to-accuracy by up to 50% compared to baselines using real power traces.
* `serving` `offloading` `scaling` [Multiprocessor Scheduling with Memory Constraints: Fundamental Properties and Finding Optimal Solutions](http://arxiv.org/abs/2507.17411v1)
  > **TL;DR**: Addresses scheduling computational DAGs on multiprocessors with memory constraints. Proposes an ILP-based holistic scheduling algorithm optimizing workload balancing and data movement. Achieves better solutions than classical scheduling baselines, reducing memory-related overhead.
* `edge` `hardware` `offline` [CHAMP: A Configurable, Hot-Swappable Edge Architecture for Adaptive Biometric Tasks](http://arxiv.org/abs/2507.17793v1)
  > **TL;DR**: Proposes CHAMP, a modular edge system with hot-swappable FPGA accelerators and custom OS for adaptive AI tasks. Achieves near-linear throughput scaling from 1 to 5 accelerators on USB3 bus, optimizing resource use for field-deployable multi-stage biometric pipelines.
* `serving` `offloading` [KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider](http://arxiv.org/abs/2506.02634v4)
  > **TL;DR**: Characterizes KV Cache workload patterns in LLM serving from a cloud provider and develops a workload-aware cache eviction policy. Optimizes serving performance under real-world traces, achieving up to 20% higher throughput with limited cache capacity.
* `kernel` `sparse` `offline` [Efficient Column-Wise N:M Pruning on RISC-V CPU](http://arxiv.org/abs/2507.17301v1)
  > **TL;DR**: Proposes column-wise N:M pruning at tile level for CNNs on RISC-V, fusing im2col with data packing to reduce memory overhead. Uses AITemplate profiling to select optimal convolution implementations. Achieves up to 4.0x ResNet inference throughput increase with <2.1% accuracy drop.
* `training` `kernel` `RAG` [PathWeaver: A High-Throughput Multi-GPU System for Graph-Based Approximate Nearest Neighbor Search](http://arxiv.org/abs/2507.17094v1)
  > **TL;DR**: Develops PathWeaver, a multi-GPU system for graph-based ANNS with pipelining and data selection to minimize redundant computations. Introduces pipelined path extension and direction-guided selection for early filtering. Achieves 3.24× geomean speedup at 95% recall vs. state-of-the-art.
* `kernel` `hardware` `networking` [AcceleratedKernels.jl: Cross-Architecture Parallel Algorithms from a Unified, Transpiled Codebase](http://arxiv.org/abs/2507.16710v1)
  > **TL;DR**: Presents AcceleratedKernels.jl, a unified transpiled codebase for cross-architecture parallel algorithms, enabling efficient CPU-GPU cooperation and specialized MPI. Achieved GPU sorting throughputs of 538-855 GB/s using 200 A100 GPUs with 4.93x speedup via GPU interconnects.
* `edge` `offloading` `quantization` [An Experimental Study of Split-Learning TinyML on Ultra-Low-Power Edge/IoT Nodes](http://arxiv.org/abs/2507.16594v1)
  > **TL;DR**: Explores split-learning for TinyML on ultra-low-power edge nodes. Proposes a testbed with ESP32-S3 boards using wireless offloading (ESP-NOW, BLE, UDP/IP) of quantized MobileNetV2 partitions. Achieves 3.7 s round-trip latency with ESP-NOW compared to 10s+ for BLE.
* `hardware` `quantization` `kernel` [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](http://arxiv.org/abs/2507.10789v2)
  > **TL;DR**: Conducts microarchitectural analysis of NVIDIA Blackwell GPU using microbenchmarks to understand subsystems including tensor cores supporting FP4/FP6. Compares with Hopper architecture for latency, throughput, and power efficiency. Highlights FP4/FP6 support enabling efficient quantized inference for LLMs.
* `training` `offline` `edge` [ACME: Adaptive Customization of Large Models via Distributed Systems](http://arxiv.org/abs/2507.14802v1)
  > **TL;DR**: Proposes ACME, a distributed system for cost-efficient customization of Transformer models on edge devices. Uses bidirectional single-loop collaboration and Pareto-optimal backbone/header generation to handle user/data heterogeneity. Reduces data transmission by 94% and improves accuracy by 10% over centralized systems.
* `edge` `scaling` [Towards a Proactive Autoscaling Framework for Data Stream Processing at the Edge using GRU and Transfer Learning](http://arxiv.org/abs/2507.14597v1)
  > **TL;DR**: Addresses proactive autoscaling for edge data stream processing under workload fluctuations. Combines GRU load forecasting with transfer learning to adapt offline models to online environments and adjust operator parallelism. Achieves 1.3% SMAPE prediction error, outperforming baselines with lower training time.
* `edge` `sparse` `offline` [Edge Intelligence with Spiking Neural Networks](http://arxiv.org/abs/2507.14069v1)
  > **TL;DR**: Surveys Spiking Neural Networks for edge intelligence, focusing on lightweight inference, resource-aware training under non-stationary data, and secure deployment. Discusses neuron models, learning algorithms, and benchmarking for hardware-aware optimization. Highlights low-power event-driven computation.
* `training` `storage` [Checkmate: Zero-Overhead Model Checkpointing via Network Gradient Replication](http://arxiv.org/abs/2507.13522v1)
  > **TL;DR**: Introduces Checkmate, a system for zero-overhead model checkpointing in DNN training by replicating gradients to a shadow cluster. Utilizes multicast to deliver gradients to a CPU-based cluster that maintains checkpoints. Achieves per-iteration checkpointing with throughput comparable to no-checkpoint baseline and reduces repeated work by 80-97.1%.
* `serving` `scaling` `RL` [Autonomous Resource Management in Microservice Systems via Reinforcement Learning](http://arxiv.org/abs/2507.12879v1)
  > **TL;DR**: Proposes reinforcement learning for autonomous microservice resource management to optimize allocation, reduce latency, and improve throughput. Achieves significant improvements in response speed and throughput under varying loads while optimizing resource usage and energy consumption.
* `serving` `scaling` [PolyServe: Efficient Multi-SLO Serving at Scale](http://arxiv.org/abs/2507.17769v1)
  > **TL;DR**: Addresses efficient LLM serving with multiple latency SLO requirements. Introduces PolyServe, a scheduler grouping requests into bins per SLO, using load-based routing and bin sharing. Achieves 1.23x goodput gain versus baselines and 92.5% of optimal goodput.
* `edge` `networking` `scaling` [CRAFT: Latency and Cost-Aware Genetic-Based Framework for Node Placement in Edge-Fog Environments](http://arxiv.org/abs/2507.12445v1)
  > **TL;DR**: Proposes CRAFT, a genetic algorithm-based node placement framework for edge-fog environments to minimize latency and cost in IoT systems. Achieves 2.77% latency and 31.15% cost reduction in simulations.
* `training` `scaling` `networking` [Incentivised Orchestrated Training Architecture (IOTA): A Technical Primer for Release](http://arxiv.org/abs/2507.17766v1)
  > **TL;DR**: Proposes IOTA, a decentralized LLM training architecture using orchestrator-driven layer distribution across miners. Implements activation compression (128x reduction) and butterfly all-reduce for scalability. Achieves linear scaling by partitioning parameters across contributors while ensuring fair incentives.
* `edge` `scaling` `storage` [ARRC: Explainable, Workflow-Integrated Recommender for Sustainable Resource Optimization Across the Edge-Cloud Continuum](http://arxiv.org/abs/2507.12032v1)
  > **TL;DR**: Proposes ARRC, a recommender system for resource optimization in edge-cloud continuum, using explainable agents integrated into operator workflows. Reduces operator workload by over 50% and improves compute utilization by up to 7.7x while maintaining error rates below 5%.
* `edge` `offloading` `networking` [MOFCO: Mobility- and Migration-Aware Task Offloading in Three-Layer Fog Computing Environments](http://arxiv.org/abs/2507.12028v1)
  > **TL;DR**: Proposes MOFCO, a mobility-aware task offloading algorithm for fog computing environments. Uses heuristic-aided evolutionary game theory to solve MINLP formulation for task placement and resource allocation. Reduces system cost (latency + energy) by average 19% and up to 43%.
* `kernel` [NineToothed: A Triton-Based High-Level Domain-Specific Language for Machine Learning](http://arxiv.org/abs/2507.11978v1)
  > **TL;DR**: Introduces NineToothed, a high-level DSL with serial semantics for ML kernels. It automatically transforms serial tensor-oriented code into parallel code using arrange-and-apply paradigm and code generation. Maintains Triton-like performance while significantly simplifying kernel development.
* `serving` `kernel` [BlockBPE: Parallel BPE Tokenization](http://arxiv.org/abs/2507.11941v1)
  > **TL;DR**: Addresses slow CPU-bound BPE tokenization in LLM inference. Proposes BlockBPE, a GPU-optimized parallel byte-pair encoding algorithm that skips regex pre-tokenization, enabling thread-block merges. Achieves up to 2.5x higher throughput versus HuggingFace Tokenizers on batch inference.
* `serving` `offline` `scaling` [Making Serverless Computing Extensible: A Case Study of Serverless Data Analytics](http://arxiv.org/abs/2507.11929v1)
  > **TL;DR**: Proposes Proteus, an extensible serverless platform enabling domain-specific optimizations via decision workflows. Focuses on improving resource sharing and performance for serverless data analytics workloads without compromising generality. Achieves better query execution efficiency in preliminary results.
* `training` [DeInfoReg: A Decoupled Learning Framework for Better Training Throughput](http://arxiv.org/abs/2506.18193v2)
  > **TL;DR**: Proposes DeInfoReg, a decoupled learning framework with information regularization to transform long gradient flows into shorter ones, enabling pipeline parallelism across GPUs for improved throughput. Achieves higher training throughput and better resource utilization compared to standard backpropagation.
* `networking` `scaling` `serving` [Arcturus: A Cloud Overlay Network for Global Accelerator with Enhanced Performance and Stability](http://arxiv.org/abs/2507.10928v1)
  > **TL;DR**: Proposes Arcturus, a cloud-native global accelerator overlay network using heterogeneous cloud resources. Features two-plane design: forwarding plane for adaptive proxy network and scheduling plane for load coordination. Achieves 1.7× speedup over commercial services, 71% cost reduction, and 80% resource efficiency at millions of RPS.
* `edge` `offloading` `serving` [A Model Aware AIGC Task Offloading Algorithm in IIoT Edge Computing](http://arxiv.org/abs/2507.11560v1)
  > **TL;DR**: Proposes MADDPG-MATO, a multi-agent deep reinforcement learning algorithm for AI-generated content task offloading in IIoT edge computing. Minimizes latency and energy by model-aware scheduling, considering model switching overheads. Achieves 6.98% lower latency and 7.12% lower energy consumption versus baselines.
* `disaggregation` `networking` `hardware` [Compute Can't Handle the Truth: Why Communication Tax Prioritizes Memory and Interconnects in Modern AI Infrastructure](http://arxiv.org/abs/2507.07223v2)
  > **TL;DR**: Addresses scaling bottlenecks in AI infrastructure due to inter-GPU communication. Proposes a modular data center architecture with disaggregated memory/compute using CXL and hybrid interconnects (XLink). Demonstrates improved scalability and throughput with efficiency gains in communication overheads.
* `serving` `offloading` `quantization` [MQFQ-Sticky: Fair Queueing For Serverless GPU Functions](http://arxiv.org/abs/2507.08954v1)
  > **TL;DR**: Addresses high latency in GPU-accelerated FaaS due to limited concurrency and cold starts. Proposes MQFQ-Sticky, a fair queueing and memory management system adapting I/O scheduling principles for GPUs. Reduces function latency by 2x-20x compared to existing policies.
* `training` `scaling` `RL` [Efficient Long Context Fine-tuning with Chunk Flow](http://arxiv.org/abs/2503.02356v3)
  > **TL;DR**: Addresses inefficiencies in fine-tuning LLMs on datasets with variable sequence lengths. Proposes ChunkFlow, a method that reorganizes sequences into uniform chunks and uses state-aware scheduling. Achieves up to 4.53x speedup over Megatron-LM.
* `edge` `kernel` `video` [Accelerating Transposed Convolutions on FPGA-based Edge Devices](http://arxiv.org/abs/2507.07683v1)
  > **TL;DR**: Proposes MM2IM, a hardware-software accelerator combining matrix multiplication with col2IM to optimize transposed convolutions in generative AI models on edge devices. Achieves up to 4.2x speedup on TCONV layers from generative models and 2.4x energy reduction on DCGAN and pix2pix.
* `serving` `offline` `networking` [Towards Efficient and Scalable Distributed Vector Search with RDMA](http://arxiv.org/abs/2507.06653v1)
  > **TL;DR**: Addresses scalability bottlenecks in distributed vector search systems. Proposes CoTra with data partitioning, asynchronous execution, and task pushing leveraging RDMA and system optimizations. Achieves 9.8–13.4x throughput scaling on 16 machines with 2.12–3.58x baseline improvement at 0.95 recall.
* `edge` `efficiency` `MoE` [The AI Shadow War: SaaS vs. Edge Computing Architectures](http://arxiv.org/abs/2507.11545v1)
  > **TL;DR**: Investigates the performance and efficiency trade-offs between centralized cloud and decentralized edge architectures for AI inference. Highlights innovations like mixture-of-experts and test-time training on edge devices, achieving a 10,000x efficiency gain on ARM processors compared to cloud processing.
* `training` `diffusion` `sparse` [FedPhD: Federated Pruning with Hierarchical Learning of Diffusion Models](http://arxiv.org/abs/2507.06449v1)
  > **TL;DR**: FedPhD addresses efficient distributed training of diffusion models with federated learning. It uses hierarchical learning with structured pruning and heterogeneity-aware aggregation to reduce costs. Achieves 88% lower comm. costs with 34% better FID using 56% resources.
* `training` `multi-modal` `edge` [Fine-tuning Multimodal Transformers on Edge: A Parallel Split Learning Approach](http://arxiv.org/abs/2502.06355v3)
  > **TL;DR**: Proposes MPSL, a parallel split learning approach for fine-tuning multimodal transformers on edge devices. Uses lightweight client tokenizers and a unified modality-agnostic encoder to reduce computation. Cuts client-side computation by 250× and improves communication scalability versus federated learning.
* `training` `scaling` [CFP: Efficient Optimization of Intra-Operator Parallelism Plans for Large Model Training](http://arxiv.org/abs/2504.00598v2)
  > **TL;DR**: Optimizes intra-operator parallelism plans for large model training by identifying parallel-preserving and repetitive subgraphs to reduce search space. Uses dynamic programming to find optimal strategies, achieving speedups for GPT and LLAMA.
* `training` `sparse` `scaling` [Distributed Equivariant Graph Neural Networks for Large-Scale Electronic Structure Prediction](http://arxiv.org/abs/2507.03840v1)
  > **TL;DR**: Addresses memory and scalability limitations in training equivariant GNNs for large-scale electronic structure prediction. Proposes a distributed implementation with GPU communication and graph partitioning to reduce embedding exchanges. Achieves strong scaling up to 128 GPUs and weak scaling up to 512 GPUs with 87% efficiency.
* `training` `networking` `scaling` [Collective Communication Profiling of Modern-day Machine Learning Workloads](http://arxiv.org/abs/2507.07117v1)
  > **TL;DR**: Analyzes network bottlenecks in distributed ML workloads' collective communications (e.g., AllReduce). Instruments NCCL to profile operations across models (DeepSeek, GPT) under varied configurations. Observes bursty traffic causing congestion, suggesting framework and topology redesigns.
* `training` `networking` `scaling` [Characterizing Compute-Communication Overlap in GPU-Accelerated Distributed Deep Learning: Performance and Power Implications](http://arxiv.org/abs/2507.03114v1)
  > **TL;DR**: Characterizes compute-communication overlap in GPU-accelerated distributed deep learning training. Evaluates impact of overlapping strategies, hardware features like numeric precision and power capping on performance and power. Finds overlapping causes average 18.9% compute slowdown but is still 10.2% faster than sequential execution.
* `edge` `quantization` `hardware` [Red grape detection with accelerated artificial neural networks in the FPGA's programmable logic](http://arxiv.org/abs/2507.02443v1)
  > **TL;DR**: Addresses slow object detection in robotics by deploying quantized ANNs on FPGAs using FINN framework. Achieves 6611 FPS inference with 98% success rate, accelerating attention mechanisms for realtime robotic tasks.
* `training` `storage` `offline` [The Artificial Scientist -- in-transit Machine Learning of Plasma Simulations](http://arxiv.org/abs/2501.03383v3)
  > **TL;DR**: Proposes an in-transit ML workflow for large-scale plasma simulations to bypass file system bottlenecks. Data is streamed directly from simulation to ML framework for asynchronous transformation and training. Achieves efficient streaming on Frontier exascale system at petabyte scale.
* `training` `hardware` `networking` [SAKURAONE: Empowering Transparent and Open AI Platforms through Private-Sector HPC Investment in Japan](http://arxiv.org/abs/2507.02124v1)
  > **TL;DR**: Introduces SAKURAONE HPC cluster for large-scale LLM training. Features 800 GbE with SONiC open networking and RoCEv2 for high-speed interconnect. Achieves 339.86 PFLOP/s on HPL-MxP (FP8), demonstrating suitability for low-precision AI workloads.
* `training` `scaling` `hardware` [Evolving HPC services to enable ML workloads on HPE Cray EX](http://arxiv.org/abs/2507.01880v1)
  > **TL;DR**: Addresses extending HPC services to support ML workloads on Alps infrastructure. Proposes user environments, performance screening tools, observability, storage enhancements, and service plane for multi-workload deployment. Achieves scalable support for 10,752 GPUs to meet dynamic AI/ML community needs.
* `training` `networking` `scaling` [Distributed Training under Packet Loss](http://arxiv.org/abs/2507.07114v1)
  > **TL;DR**: Proposes a distributed training framework that maintains model accuracy and convergence under packet loss. It features unbiased gradient aggregation and bounded-drift parameter broadcasts. Achieves at most 0.8% perplexity change for LLAMA2 7B with 64 GPUs under 10% packet loss.
* `serving` `training` `scaling` [Not All Water Consumption Is Equal: A Water Stress Weighted Metric for Sustainable Computing](http://arxiv.org/abs/2506.22773v2)
  > **TL;DR**: Proposes SCARF, a framework for assessing water impact of computing with water-stress weighted metrics. Introduces Adjusted Water Impact (AWI) considering spatiotemporal water stress variations. Case studies show significant reductions in water impact for LLM serving via optimized scheduling and siting.
* `quantization` `training` [Enabling mixed-precision in spectral element codes](http://arxiv.org/abs/2503.02134v2)
  > **TL;DR**: Proposes a methodology for mixed-precision in spectral element CFD codes using Verificarlo and computer arithmetic. Applied to Nekbone and Neko, achieving up to 1.62× faster time-to-solution and 2.43× lower energy-to-solution while matching double-precision accuracy.
* `storage` [yProv4ML: Effortless Provenance Tracking for Machine Learning Systems](http://arxiv.org/abs/2507.01078v1)
  > **TL;DR**: Addresses the lack of transparency and lineage tracking in ML systems like LLM training. Proposes yProv4ML, a framework that captures provenance data in standard PROV-JSON format with low code changes. Enables effortless hyperparameter and epoch tracking for model selection.
* `training` `storage` `scaling` [Provenance Tracking in Large-Scale Machine Learning Systems](http://arxiv.org/abs/2507.01075v1)
  > **TL;DR**: Addresses efficient scaling of large AI models by introducing yProv4ML, a provenance tracking library for monitoring resource usage and inefficiencies during training. Enables JSON-based data collection compliant with standards. Achieves better resource utilization for energy-efficient distributed training.
* `training` `storage` `hardware` [eACGM: Non-instrumented Performance Tracing and Anomaly Detection towards Machine Learning Systems](http://arxiv.org/abs/2506.02007v2)
  > **TL;DR**: Proposes eACGM, an eBPF-based monitoring framework for AI/ML systems, capturing GPU/network/CUDA metrics without instrumentation. Uses GMM for anomaly detection of latency/failures/inefficiencies. Achieves low-overhead anomaly detection in distributed training, supporting fault diagnosis.
* `serving` `edge` `networking` [Real-Time In-Network Machine Learning on P4-Programmable FPGA SmartNICs with Fixed-Point Arithmetic and Taylor](http://arxiv.org/abs/2507.00428v1)
  > **TL;DR**: Investigates low-latency ML inference on smart network devices at the network edge. Proposes P4-programmable FPGA SmartNICs using fixed-point arithmetic and dynamic table lookups for model weights. Enables real-time deployment with dynamic reconfiguration and efficient edge processing for network tasks.
* `training` `networking` `scaling` [CrossPipe: Towards Optimal Pipeline Schedules for Cross-Datacenter Training](http://arxiv.org/abs/2507.00217v1)
  > **TL;DR**: Proposes CrossPipe, a framework for optimizing LLM training across datacenters by co-optimizing pipeline parallelism and data parallelism overlapping to mitigate network latency. Uses solver-based and greedy scheduling algorithms, reducing training time by up to 33.6% under memory constraints.
* `training` `offloading` `scaling` [PipeOffload: Improving Scalability of Pipeline Parallelism with Memory Optimization](http://arxiv.org/abs/2503.01328v2)
  > **TL;DR**: Reduces activation memory constraints in pipeline parallelism for large model training by selectively offloading activations, enabling better scalability. Combines memory offload with throughput optimization, achieving up to 19% speedup with lower memory consumption.
* `kernel` `storage` `inference` [Verifying Properties of Index Arrays in a Purely-Functional Data-Parallel Language](http://arxiv.org/abs/2506.23058v1)
  > **TL;DR**: Presents a framework for verifying properties of data-parallel programs to enable optimizations. Uses index function transformations and algebraic solving to prove array properties. Eliminating dynamic checks in GPU programs yields significant speedups.
* `RL` `LoRA` `edge` [Adaptive Rank Allocation for Federated Parameter-Efficient Fine-Tuning of Language Models](http://arxiv.org/abs/2501.14406v3)
  > **TL;DR**: Addresses challenges in federated parameter-efficient fine-tuning (FedPEFT) of PLMs: data heterogeneity and communication inefficiency. Proposes FedARA with adaptive rank allocation, dynamic module pruning, and SVD adaptation. Reduces training time by 48.9% and energy by 46.95% on edge devices.
* `kernel` `hardware` `sparse` [TriADA: Massively Parallel Trilinear Matrix-by-Tensor Multiply-Add Algorithm and Device Architecture for the Acceleration of 3D Discrete Transformations](http://arxiv.org/abs/2506.22818v1)
  > **TL;DR**: Proposes TriADA, a hardware architecture with elastic sparse outer-product kernels for massively parallel trilinear transformations. Decouples memory and processing via mesh PEs to achieve hypercubic complexity in linear steps. Achieves energy efficiency for sparse tensors without unnecessary zero operations.
* `kernel` `training` `quantization` [MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators](http://arxiv.org/abs/2506.22169v1)
  > **TL;DR**: Proposes MCFuser, a framework for fusing memory-bound compute-intensive operator chains to improve GPU utilization. Uses high-level tiling, DAG analysis, and heuristic search to optimize kernels. Achieves up to 5.9x speedup and 70x faster tuning time compared to Ansor.
* `scaling` `serving` [Efficient and Reuseable Cloud Configuration Search Using Discovery Spaces](http://arxiv.org/abs/2506.21467v1)
  > **TL;DR**: Proposes Discovery Space, an abstraction for efficient cloud configuration search to minimize cost and meet SLAs for workloads like LLM inference. Enables safe sharing and reuse of optimizer data across executions and similar search spaces, achieving over 90% search speed-up.
* `RAG` `serving` `storage` [BLOCKS: Blockchain-supported Cross-Silo Knowledge Sharing for Efficient LLM Services](http://arxiv.org/abs/2506.21033v1)
  > **TL;DR**: Proposes a blockchain-based framework for secure cross-silo knowledge sharing to augment LLMs. Uses knowledge distillation into prompts with reputation mechanisms and cross-validation. Achieves efficient knowledge sharing, enabling reliable RAG without compromising data privacy.
* `inference` `serving` `scaling` [SuperSONIC: Cloud-Native Infrastructure for ML Inferencing](http://arxiv.org/abs/2506.20657v1)
  > **TL;DR**: Proposes SuperSONIC, a cloud-native server infrastructure for scalable ML inference using remote coprocessors and Kubernetes. Utilizes NVIDIA Triton to decouple clients and servers, optimizing throughput and load balancing. Deployed on scientific experiments, demonstrating enhanced efficiency for accelerator-based inference.
* `training` `networking` [PAT: a new algorithm for all-gather and reduce-scatter operations at scale](http://arxiv.org/abs/2506.20252v1)
  > **TL;DR**: Proposes PAT, a parallel aggregated trees algorithm for efficient all-gather and reduce-scatter operations in distributed training. Features logarithmic network transfers and buffers, minimizing long-distance communication. Improves NCCL performance for small operations at scale, reducing latency where ring algorithms underperform.
* `training` `RAG` `kernel` [MegaFold: System-Level Optimizations for Accelerating Protein Structure Prediction Models](http://arxiv.org/abs/2506.20686v1)
  > **TL;DR**: Accelerates AlphaFold3 training via system optimizations: ahead-of-time caching for retrieval-augmented pipelines, custom Triton kernels for memory-efficient attention, and operator fusion. Achieves up to 1.73× faster iteration times and 1.35× longer sequence lengths on GPUs.
* `networking` `scaling` [MAIZX: A Carbon-Aware Framework for Optimizing Cloud Computing Emissions](http://arxiv.org/abs/2506.19972v1)
  > **TL;DR**: Introduces MAIZX, a carbon-aware framework for cloud computing emissions optimization. Dynamically ranks resources (data centers, edge nodes, multi-cloud) based on carbon intensity, PUE, and energy. Reduces CO2 emissions by 85.68% compared to baseline hypervisor operations.
* `training` `RL` `networking` [GradualDiff-Fed: A Federated Learning Specialized Framework for Large Language Model](http://arxiv.org/abs/2506.19164v1)
  > **TL;DR**: Proposes GradualDiff-Fed, a federated learning framework for efficient LLM fine-tuning that transmits weight differences to reduce communication costs. Achieves performance equivalent to centralized training while cutting communication overhead.
* `quantization` `offloading` `edge` [NestQuant: Post-Training Integer-Nesting Quantization for On-Device DNN](http://arxiv.org/abs/2506.17870v1)
  > **TL;DR**: Proposes NestQuant, a post-training integer-nesting quantization method for on-device DNNs. Uses integer weight decomposition and nesting to enable adaptive bitwidth switching with one stored model. Achieves 78.1% reduction in switching overheads while maintaining high accuracy.
* `edge` `serving` `kernel` [ConsumerBench: Benchmarking Generative AI Applications on End-User Devices](http://arxiv.org/abs/2506.17538v1)
  > **TL;DR**: Presents ConsumerBench, a benchmarking framework for evaluating GenAI application efficiency on end-user devices. Simulates multi-application scenarios on constrained hardware and captures system metrics. Reveals inefficiencies in resource sharing and shows custom kernels improve performance on consumer GPUs.
* `training` `networking` `quantization` [NetSenseML: Network-Adaptive Compression for Efficient Distributed Machine Learning](http://arxiv.org/abs/2506.16235v1)
  > **TL;DR**: Proposes NetSenseML, a network-adaptive framework that dynamically adjusts gradient compression based on real-time network conditions to balance payload reduction and accuracy. Achieves 1.55× to 9.84× higher training throughput in bandwidth-constrained distributed deep learning.
* `hardware` `kernel` `training` [HetGPU: The pursuit of making binary compatibility towards GPUs](http://arxiv.org/abs/2506.15993v1)
  > **TL;DR**: Proposes hetGPU, a system enabling single GPU binary execution across diverse vendor hardware (NVIDIA, AMD, etc.) through an architecture-agnostic IR and runtime translation. Addresses SIMT vs. MIMD divergence and state serialization challenges. Reduces migration overhead, achieving vendor-agnostic compatibility with minimal performance impact.
* `hardware` `training` `scaling` [A System Level Compiler for Massively-Parallel, Spatial, Dataflow Architectures](http://arxiv.org/abs/2506.15875v1)
  > **TL;DR**: Proposes MACH, a compiler for efficient execution on massively-parallel spatial dataflow architectures like wafer-scale engines. It introduces a Virtual Machine model, domain-specific language, and lowering flow to generate optimized code for tensor computations. Demonstrates applicability to dense tensor operations with potential for reduced execution time on specialized hardware.
* `serving` `MoE` `quantization` [Utility-Driven Speculative Decoding for Mixture-of-Experts](http://arxiv.org/abs/2506.20675v1)
  > **TL;DR**: Addresses inefficiency of speculative decoding in MoE LLMs due to increased verification cost. Proposes Cascade, a utility-driven framework that dynamically enables speculation and tunes token count K to maximize token gain per cost. Achieves 7-14% higher throughput and limits slowdown to 5% vs. up to 1.5x slowdown.
* `quantization` `offloading` `training` [ILVES: Accurate and efficient bond length and angle constraints in molecular dynamics](http://arxiv.org/abs/2503.13075v3)
  > **TL;DR**: Introduces ILVES, a parallel algorithm for molecular dynamics simulations that accurately solves bond length and angular constraints with better convergence. Achieves a 1.65x increase in simulated time using the same resources and wall-clock time compared to state-of-the-art methods.
* `video` `serving` `scaling` [DDiT: Dynamic Resource Allocation for Diffusion Transformer Model Serving](http://arxiv.org/abs/2506.13497v1)
  > **TL;DR**: Proposes DDiT for efficient serving of diffusion transformer-based text-to-video models. Integrates inter-phase and intra-phase optimizations with dynamic resource allocation. Achieves up to 1.44x lower p99 latency and 1.43x lower average latency compared to baselines.
* `serving` `edge` `scaling` [QoS-aware Scheduling of Periodic Real-time Task Graphs on Heterogeneous Pre-occupied MECs](http://arxiv.org/abs/2506.12415v1)
  > **TL;DR**: Proposes QoS-aware scheduling for periodic real-time task graphs on heterogeneous MECs. Uses modified HEFT algorithm to exploit residual capacity by dynamically mapping tasks to idle VM intervals. Improves load balancing and resource utilization by 30% compared to baselines while meeting timing constraints.
* `storage` `training` `inference` [Efficient Unified Caching for Accelerating Heterogeneous AI Workloads](http://arxiv.org/abs/2506.12370v1)
  > **TL;DR**: Proposes IGTCache, a unified caching framework for AI clusters that uses hierarchical access abstraction (AccessStreamTree) to detect heterogeneous data access patterns. Tailors prefetching, eviction, and allocation strategies dynamically, improving cache hit ratio by 55.6% and reducing job completion time by 52.2%.
* `hardware` `training` `inference` [Topology-Aware Virtualization over Inter-Core Connected Neural Processing Units](http://arxiv.org/abs/2506.11446v1)
  > **TL;DR**: Proposes vNPU for topology-aware virtualization of inter-core connected NPUs. Techniques include NPU route virtualization, memory virtualization, and best-effort topology mapping to balance resource utilization and performance. Achieves up to 2x performance improvement with 2% hardware overhead.
* `serving` `offloading` `RAG` [WindVE: Collaborative CPU-NPU Vector Embedding](http://arxiv.org/abs/2504.14941v4)
  > **TL;DR**: Optimizes vector embedding concurrency in RAG for LLM inference cost-performance. Designs WindVE: CPU-NPU queue manager with linear regression for dynamic offloading. Achieves 22.3% higher concurrency than state-of-the-art without offloading.
* `hardware` `quantization` `kernel` [Proteus: Enabling High-Performance Processing-Using-DRAM with Dynamic Bit-Precision, Adaptive Data Representation, and Flexible Arithmetic](http://arxiv.org/abs/2501.17466v2)
  > **TL;DR**: Proposes Proteus, a hardware framework to reduce latency in processing-using-DRAM (PUD) operations. Uses dynamic bit-precision reduction, parallel execution across arrays, and adaptive arithmetic representation. Achieves up to 2.2× speedup against existing PUD approaches.
* `edge` `scaling` [Automating Multi-Tenancy Performance Evaluation on Edge Compute Nodes](http://arxiv.org/abs/2506.10461v1)
  > **TL;DR**: Proposes an auto-benchmarking framework for multi-tenancy performance evaluation on edge compute nodes, integrating monitoring and diverse workloads. Achieves streamlined analysis of resource contention impact, providing hardware-specific insights for service co-location optimization.
* `scaling` `edge` [Multi-dimensional Autoscaling of Processing Services: A Comparison of Agent-based Methods](http://arxiv.org/abs/2506.10420v1)
  > **TL;DR**: Introduces an agent-based autoscaling framework for edge computing that adjusts hardware and service configurations. Compares performance of four agents (Active Inference, DQN, ASK, Deep Active Inference) for multi-dimensional scaling. Achieves acceptable SLO performance on YOLOv8 and OpenCV services.
* `training` [PerfTracker: Online Performance Troubleshooting for Large-scale Model Training in Production](http://arxiv.org/abs/2506.08528v3)
  > **TL;DR**: Proposes PerfTracker, an online troubleshooting system using fine-grained profiling to diagnose performance issues in large-scale model training. Analyzes hardware and software interactions for efficiency. Deployed on clusters with 10,000 GPUs.
* `training` `quantization` `offloading` [Low-resource domain adaptation while minimizing energy and hardware resource consumption](http://arxiv.org/abs/2506.08433v2)
  > **TL;DR**: Investigates reducing computation cost for LLM domain adaptation in low-resource settings. Evaluates numerical precision formats and data parallelization to optimize energy efficiency and hardware consumption. Achieves comparable accuracy with FP16 to FP32 at lower resource use.
* `training` `quantization` [TTrace: Lightweight Error Checking and Diagnosis for Distributed Training](http://arxiv.org/abs/2506.09280v1)
  > **TL;DR**: Addresses silent bug detection in distributed LLM training. Proposes TTrace, which compares intermediate tensors against a single-device reference with threshold analysis for floating-point errors. Detects 14 bugs in Megatron-LM with fewer than 10 code changes, including in BF16 and FP8 training.
* `training` `scaling` [A Survey of End-to-End Modeling for Distributed DNN Training: Workloads, Simulators, and TCO](http://arxiv.org/abs/2506.09275v1)
  > **TL;DR**: Surveys distributed DNN training simulators for efficient system co-design. Focuses on workload representation, simulation infrastructure, and TCO/carbon emission modeling. Highlights tools for reducing the cost and complexity of scaling large-scale training systems.
* `scaling` `training` [Balancing Fixed Number of Nodes Among Multiple Fixed Clusters](http://arxiv.org/abs/2506.08715v1)
  > **TL;DR**: Addresses underutilization of fixed nodes across multiple clusters by proposing a Node Balancing Cluster Group (NBCG) system for dynamic rebalancing based on real-time thresholds. Achieves optimized resource utilization without additional costs.
* `serving` `scaling` [DeepServe: Serverless Large Language Model Serving at Scale](http://arxiv.org/abs/2501.14417v3)
  > **TL;DR**: Proposes DeepServe, a serverless platform for scalable LLM serving. Key features include a serverless abstraction, NPU-optimized serving engine, scheduling for disaggregated/colocated instances, and optimizations like pre-warmed pods. Achieves scaling to 64 instances in seconds to handle cold starts.
* `kernel` `sparse` `training` [GPU-Parallelizable Randomized Sketch-and-Precondition for Linear Regression using Sparse Sign Sketches](http://arxiv.org/abs/2506.03070v2)
  > **TL;DR**: Proposes GPU-parallelizable randomized sketch-and-precondition method for large linear regression using sparse sign sketches. Introduces rejection-sampling based sparse sign sketch generation optimized for single/multi-GPU systems. Improves computational efficiency on GPUs, suitable for black-box least-squares solvers.
* `training` `RL` `LoRA` [Mitigating Catastrophic Forgetting with Adaptive Transformer Block Expansion in Federated Fine-Tuning](http://arxiv.org/abs/2506.05977v1)
  > **TL;DR**: Addresses catastrophic forgetting in federated LLM fine-tuning. Proposes FedBE with adaptive transformer block expansion and dynamic block allocation to separate new knowledge from pre-trained representations. Achieves 12-74% higher general-task accuracy retention and 1.9-3.1x faster convergence.
* `serving` `disaggregation` `scaling` [BestServe: Serving Strategies with Optimal Goodput in Collocation and Disaggregation Architectures](http://arxiv.org/abs/2506.05871v1)
  > **TL;DR**: Addresses the labor-intensive process of finding optimal LLM serving strategies. Proposes BestServe, a lightweight inference simulator based on an adapted roofline model to predict goodput. Achieves optimal strategy selection in minutes with predictions within 20% error margin.
* `training` `kernel` `networking` [Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler](http://arxiv.org/abs/2504.19442v3)
  > **TL;DR**: Proposes Triton-distributed compiler extension for efficient distributed AI workloads, supporting overlapping of computation and communication via OpenSHMEM. Demonstrates up to 64 devices utilization, outperforming hand-optimized code with reduced development effort.
* `offloading` `training` `scaling` [Analysis of Server Throughput For Managed Big Data Analytics Frameworks](http://arxiv.org/abs/2506.03854v1)
  > **TL;DR**: Examines how to reduce garbage collection and serialization overhead in big data frameworks to improve server throughput. Proposes TeraHeap, which offloads objects to a secondary heap on fast storage, and studies DRAM distribution. Achieves higher throughput by co-locating memory-bound instances and optimizing memory utilization.
* `edge` `scaling` `offloading` [LRScheduler: A Layer-aware and Resource-adaptive Container Scheduler in Edge Computing](http://arxiv.org/abs/2506.03694v1)
  > **TL;DR**: Proposes LRScheduler, a layer-aware and resource-adaptive container scheduler for edge computing. Utilizes container image layers to reduce deployment costs via node scoring and resource-adaptive mechanisms for load balancing. Achieves up to 30% reduced container deployment cost compared to default Kubernetes scheduler.
* `kernel` `serving` `quantization` [FlashMLA-ETAP: Efficient Transpose Attention Pipeline for Accelerating MLA Inference on NVIDIA H20 GPUs](http://arxiv.org/abs/2506.01969v2)
  > **TL;DR**: Proposes FlashMLA-ETAP to optimize transpose attention for MLA inference by aligning KV context with WGMMA operations, reducing redundancy. Achieves 2.78x speedup over FlashMLA at batch 16 with 64K sequence length while maintaining accuracy.
* `training` `LoRA` `edge` [Memory-Efficient Split Federated Learning for LLM Fine-Tuning on Heterogeneous Mobile Devices](http://arxiv.org/abs/2506.02940v1)
  > **TL;DR**: Proposes a split federated learning framework for memory-efficient LLM fine-tuning on heterogeneous mobile devices. Uses device-specific LoRA on lower layers and server-side scheduling of sequential fine-tuning. Reduces memory by 79% and training time by 6% compared to baselines.
* `training` `scaling` [Ringmaster ASGD: The First Asynchronous SGD with Optimal Time Complexity](http://arxiv.org/abs/2501.16168v3)
  > **TL;DR**: Proposes Ringmaster ASGD, an asynchronous SGD method that achieves optimal time complexity under heterogeneous worker computation times. The design dynamically manages straggler effects. Demonstrated optimal convergence matching theoretical lower bounds in scaling workers.
* `training` `scaling` [Rethinking Dynamic Networks and Heterogeneous Computing with Automatic Parallelization](http://arxiv.org/abs/2506.02787v1)
  > **TL;DR**: Proposes an automatic parallel planning framework for LLM training on heterogeneous nodes with dynamic network changes. Uses simulation-based strategies and pruning for optimal workload allocation. Achieves up to 5× faster search speed and competitive training performance in cloud environments.
* `RL` `training` `quantization` [Reconciling Hessian-Informed Acceleration and Scalar-Only Communication for Efficient Federated Zeroth-Order Fine-Tuning](http://arxiv.org/abs/2506.02370v1)
  > **TL;DR**: Proposes HiSo, a federated zeroth-order fine-tuning method using Hessian-informed optimization and scalar-only communication to accelerate LLM fine-tuning. It transmits only scalars per round with curvature-based acceleration. Achieves up to 4.6× faster convergence than DeComFL on RoBERTa-large.
* `edge` `serving` `kernel` [Scheduling Techniques of AI Models on Modern Heterogeneous Edge GPU -- A Critical Review](http://arxiv.org/abs/2506.01377v1)
  > **TL;DR**: Reviews scheduling techniques for AI models on heterogeneous edge GPUs like NVIDIA Jetson. Analyzes schedulers that optimize resource utilization across CPU, GPU, and accelerators to meet demands of modern AI workloads. Highlights performance improvements in throughput and latency for DNNs on resource-constrained devices.
* `training` `LoRA` `RL` [FedRPCA: Enhancing Federated LoRA Aggregation Using Robust PCA](http://arxiv.org/abs/2506.01194v1)
  > **TL;DR**: FedRPCA enhances federated aggregation for LoRA fine-tuning by decomposing updates with Robust-PCA: averaging common components and scaled averaging sparse client-specific ones. Achieves faster convergence and higher accuracy across vision/language tasks.
* `serving` `kernel` `networking` [SPD: Sync-Point Drop for Efficient Tensor Parallelism of Large Language Models](http://arxiv.org/abs/2502.20727v4)
  > **TL;DR**: Proposes Sync-Point Drop (SPD) to reduce tensor parallelism communication overheads in LLM inference. Selectively drops attention synchronization via block design and sensitivity-based strategies. Achieves 20% latency reduction for LLaMA2-70B on 8 GPUs with <1% accuracy loss.
* `serving` `quantization` `offloading` [TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference](http://arxiv.org/abs/2501.16007v2)
  > **TL;DR**: Proposes TOPLOC, a locality-sensitive hashing scheme for verifiable LLM inference with polynomial encoding. Ensures model integrity by detecting unauthorized changes to model, prompts, or precision with 100% accuracy. Reduces memory overhead by 1000x, requiring 258 bytes instead of 262 KB per 32 tokens.
* `scaling` `training` `RL` [GrapheonRL: A Graph Neural Network and Reinforcement Learning Framework for Constraint and Data-Aware Workflow Mapping and Scheduling in Heterogeneous HPC Systems](http://arxiv.org/abs/2506.00260v1)
  > **TL;DR**: Proposes a GNN and RL framework for constraint and data-aware workflow scheduling in heterogeneous HPC systems. Uses GNN for dependencies and RL for policy-based scheduling to adapt to dynamic workflows. Achieves 76% faster execution than ILP with near-optimal solutions.
* `edge` `serving` `networking` [Smaller, Smarter, Closer: The Edge of Collaborative Generative AI](http://arxiv.org/abs/2505.16499v2)
  > **TL;DR**: Proposes collaborative inference systems for GenAI deploying smaller models at edge integrated with cloud resources to reduce latency, cost, and privacy issues. Designs cooperation strategies across edge-cloud continuum. Achieves reduced end-to-end latency and optimized resource utilization.
* `training` `scaling` `hardware` [DOPPLER: Dual-Policy Learning for Device Assignment in Asynchronous Dataflow Graphs](http://arxiv.org/abs/2505.23131v1)
  > **TL;DR**: Addresses device assignment for minimizing execution time in asynchronous dataflow graphs of ML workloads. Proposes Doppler, a dual-policy network with operation selection and placement stages. Reduces system execution time by 15-40% compared to baselines across tasks.
* `MoE` `serving` `disaggregation` [Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony](http://arxiv.org/abs/2505.08944v2)
  > **TL;DR**: Addresses inefficiencies in serving Mixture-of-Experts models due to load skew and synchronization. Proposes Asynchronous Expert Parallelism (AEP) with μ-queuing and adaptive re-batching. Achieves up to 2.7× higher throughput than baselines.
* `offline` `storage` `kernel` [Speeding up Model Loading with fastsafetensors](http://arxiv.org/abs/2505.23072v1)
  > **TL;DR**: Addresses slow loading of large safetensors models by optimizing deserialization with direct tensor instantiation in device memory. Proposes fastsafetensors library with parallel I/O and GPU offloading. Achieves 4.8x–7.5x speedup for models up to 176B parameters.
* `training` `LoRA` `quantization` [Profiling and optimization of multi-card GPU machine learning jobs](http://arxiv.org/abs/2505.22905v1)
  > **TL;DR**: Investigates optimization techniques for multi-GPU LLM fine-tuning. Analyzes DPO, LoRA, QLoRA, and QAT strategies for memory efficiency and computational cost. Achieves reduced VRAM utilization and iteration times on NVIDIA H100 GPUs.
* `training` `scaling` `RL` [Incentivizing Permissionless Distributed Learning of LLMs](http://arxiv.org/abs/2505.21684v1)
  > **TL;DR**: Proposes Gauntlet, an incentive system for distributed training of LLMs via permissionless contributions. Uses two-stage filtering, loss estimation, OpenSkill rating, and unique computation to reward pseudo-gradient contributions. Achieved competitive 1.2B model with real token payouts in deployed blockchain system.
* `kernel` `compiler` `hardware` [KPerfIR: Towards an Open and Compiler-centric Ecosystem for GPU Kernel Performance Tooling on Modern AI Workloads](http://arxiv.org/abs/2505.21661v1)
  > **TL;DR**: Proposes KPerfIR, a compiler-integrated profiling infrastructure for GPU kernel performance analysis in AI workloads. Enables customizable profiling via compiler passes within Triton, providing fine-grained insights for optimization. Achieves 8.2% overhead and 2% measurement error.
* `serving` `edge` `offloading` [Fast and Cost-effective Speculative Edge-Cloud Decoding with Early Exits](http://arxiv.org/abs/2505.21594v1)
  > **TL;DR**: Proposes an edge-cloud speculative decoding framework with early exits to reduce LLM inference latency and cost. Combines on-device drafting with server verification and preemptive token generation, leveraging idle time for parallelism. Achieves up to 35% latency reduction and 21% speedup in real robot deployment.
* `training` `networking` `scaling` [DeepCEE: Efficient Cross-Region Model Distributed Training System under Heterogeneous GPUs and Networks](http://arxiv.org/abs/2505.15536v2)
  > **TL;DR**: Proposes DeepCEE, a geo-distributed training system for heterogeneous GPUs and unstable networks. Uses device grouping, pipeline parallelism with zero bubble, and dynamic adaptation to network fluctuations. Achieves 1.3-2.8x higher throughput versus state-of-the-art systems.
* `edge` `training` `offloading` [ECC-SNN: Cost-Effective Edge-Cloud Collaboration for Spiking Neural Networks](http://arxiv.org/abs/2505.20835v1)
  > **TL;DR**: Proposes ECC-SNN, an edge-cloud collaboration framework using spiking neural networks to offload compute from cloud to edge. Employs joint training of ANN and SNN for enhanced performance and on-device incremental learning. Reduces energy by 79.4% and latency by 39.1%.
* `serving` `kernel` `quantization` [InstGenIE: Generative Image Editing Made Efficient with Mask-aware Caching and Scheduling](http://arxiv.org/abs/2505.20600v1)
  > **TL;DR**: Proposes InstGenIE for efficient diffusion-based image editing serving. Utilizes mask-aware caching to skip redundant computations, bubble-free pipeline, continuous batching, and load balancing. Achieves up to 3x higher throughput and 14.7x lower latency.
* `training` `sparse` `networking` [PacTrain: Pruning and Adaptive Sparse Gradient Compression for Efficient Collective Communication in Distributed Deep Learning](http://arxiv.org/abs/2505.18563v1)
  > **TL;DR**: Proposes PacTrain, combining pruning and adaptive sparse gradient compression to reduce gradient aggregation overhead in distributed DNN training. Achieves near-optimal compression compatible with all-reduce, improving throughput 1.25-8.72x under bandwidth constraints.
* `training` `scaling` [ATA: Adaptive Task Allocation for Efficient Resource Management in Distributed Machine Learning](http://arxiv.org/abs/2502.00775v2)
  > **TL;DR**: Addresses inefficient resource utilization in distributed ML training due to heterogeneous worker speeds. Proposes ATA, an adaptive task allocation method that dynamically assigns tasks without prior computation time knowledge. Achieves significant cost reduction versus greedy approaches while maintaining speed, without assuming known distributions.
* `training` `scaling` `scheduling` [Resource Heterogeneity-Aware and Utilization-Enhanced Scheduling for Deep Learning Clusters](http://arxiv.org/abs/2503.10918v2)
  > **TL;DR**: Proposes Hadar and HadarE schedulers for resource efficiency in deep learning clusters. Designs task-level heterogeneity-aware optimization and job forking to boost GPU utilization. Reduces training duration by 50-80% and improves utilization by 1.45x versus state-of-the-art schedulers.
* `kernel` `hardware` [Parallel Scan on Ascend AI Accelerators](http://arxiv.org/abs/2505.15112v1)
  > **TL;DR**: Proposes parallel prefix scan algorithms optimized for Ascend AI accelerators by leveraging matrix multiplication units. Implements scan-based operators like sorting and sampling with matrix multiplications and accumulations. Achieves 5×-9.6× speedups over vector-only baselines and up to 37.5% of theoretical memory bandwidth.
* `training` `scaling` [COSMIC: Enabling Full-Stack Co-Design and Optimization of Distributed Machine Learning Systems](http://arxiv.org/abs/2505.15020v1)
  > **TL;DR**: Proposes COSMIC, a full-stack co-design environment for distributed ML systems using end-to-end simulation and abstraction for cross-stack optimization. Introduces Parameter Set Architecture to unify configurations. Achieves up to 48.41x higher performance in transformer models up to 175B parameters.
* `edge` `networking` `serving` [SkyMemory: A LEO Edge Cache for Transformer Inference Optimization and Scale Out](http://arxiv.org/abs/2505.14427v1)
  > **TL;DR**: Proposes SkyMemory, a LEO satellite edge cache system for transformer inference, enhancing speed via distributed key-value caching and inter-satellite links. Demonstrated in a 19x5 testbed, improves cache hits and inference latency by reducing ground data center dependencies.
* `training` `networking` `scaling` [Prime Collective Communications Library -- Technical Report](http://arxiv.org/abs/2505.14065v1)
  > **TL;DR**: Presents PCCL, a fault-tolerant collective communication library for distributed ML training over public internet. Supports dynamic peer joining/failure recovery and efficient all-reduce with techniques like multi-connection dispatch and quantization. Achieves up to 45 Gbit/s bandwidth across Europe, reduces communication frequency and bandwidth.
* `training` `scaling` `quantization` [A Study on Distributed Strategies for Deep Learning Applications in GPU Clusters](http://arxiv.org/abs/2505.12832v1)
  > **TL;DR**: Investigates distributed training strategies for scalable deep learning on GPU clusters. Empirically evaluates DDP, FSDP, and PS models, analyzing memory usage, training time, and accuracy. FSDP reduces GPU memory by over 60% but increases training time up to 6x; PS improves throughput with potential accuracy trade-offs.
* `training` `scaling` `storage` [OVERLORD: Ultimate Scaling of DataLoader for Multi-Source Large Foundation Model Training](http://arxiv.org/abs/2504.09844v2)
  > **TL;DR**: Addresses workload imbalance and redundant memory in multi-source data loading for large foundation models. Proposes Omniload with disaggregated preprocessing, centralized data plane, auto-partitioning, and fault recovery. Achieves 4.5x throughput improvement and 13.5x memory reduction at scale.
* `RL` `training` [SGDPO: Self-Guided Direct Preference Optimization for Language Model Alignment](http://arxiv.org/abs/2505.12435v1)
  > **TL;DR**: Proposes Self-Guided Direct Preference Optimization (SGDPO) for LLM alignment, adding a pilot term to steer gradient flow and control chosen/rejected reward updates. Improves preference response generation by up to 9.19% in benchmark scores.
* `scaling` `networking` [Workflow-Driven Modeling for the Compute Continuum: An Optimization Approach to Automated System and Workload Scheduling](http://arxiv.org/abs/2505.12184v1)
  > **TL;DR**: Proposes a framework for automated scheduling of workloads across HPC-Cloud continuum to minimize latency and communication overhead. Uses MILP for optimal small-scale scheduling and heuristics for large-scale with <10% deviation from optimal. Heuristics achieve 99% faster estimates with 5-10% deviation from optimal makespan.
* `offline` `offloading` `quantization` [Communication-Efficient Hybrid Language Model via Uncertainty-Aware Opportunistic and Compressed Transmission](http://arxiv.org/abs/2505.11788v1)
  > **TL;DR**: Reduces communication in hybrid language models by transmitting truncated vocabulary distributions only when uncertainty is high. Proposes CU-HLM with optimal uncertainty thresholds and truncation strategies. Achieves 206× higher token throughput with 97.4% vocabulary compression while maintaining 97.4% accuracy.
* `serving` `RL` `RAG` [Cloud-Based AI Systems: Leveraging Large Language Models for Intelligent Fault Detection and Autonomous Self-Healing](http://arxiv.org/abs/2505.11743v1)
  > **TL;DR**: Proposes an LLM system for intelligent fault detection and autonomous healing in cloud systems. Integrates LLM natural language processing with ML algorithms and a multi-level architecture for real-time anomaly detection and healing. Achieves higher accuracy and up to 50% downtime reduction over traditional systems.
* `serving` `offloading` `edge` [SpecMemo: Speculative Decoding is in Your Pocket](http://arxiv.org/abs/2506.01986v1)
  > **TL;DR**: Proposes SpecMemo, a device-aware inference engine enabling speculative decoding on memory-constrained devices. It models memory footprint to minimize allocations while retaining throughput, reducing generation-memory by 65% on mobile GPUs. Achieves 96% speculative decoding throughput on MT-Bench and 8x throughput increase with batched decoding on multiple GPUs.
* `training` `hardware` `quantization` [Assessing the Performance of Analog Training for Transfer Learning](http://arxiv.org/abs/2505.11067v1)
  > **TL;DR**: Proposes chopped TTv2 algorithm for analog in-memory computing to overcome device non-linearity and variation challenges in deep learning training. Enables energy-efficient transfer learning on Swin-ViT with CIFAR100. Achieves robustness against weight noise and asymmetry while maintaining accuracy.
* `training` `networking` `scaling` [KAITIAN: A Unified Communication Framework for Enabling Efficient Collaboration Across Heterogeneous Accelerators in Embodied AI Systems](http://arxiv.org/abs/2505.10183v1)
  > **TL;DR**: Proposes KAITIAN, a communication framework enabling efficient collaboration in heterogeneous accelerators for distributed AI training. Features unified abstraction and load-adaptive scheduling. Accelerates training by up to 42% with minimal overhead (2.8--4.3%).
* `inference` `networking` `offline` [AI Greenferencing: Routing AI Inferencing to Green Modular Data Centers with Heron](http://arxiv.org/abs/2505.09989v1)
  > **TL;DR**: Proposes Heron, a cross-site router to dispatch AI inferencing workloads to wind-powered modular data centers during power availability. Optimizes routing using power and workload traces for cost-efficient green computing. Achieves 80% higher aggregate goodput than SOTA.
* `offline` `storage` `scaling` [Efficient Graph Embedding at Scale: Optimizing CPU-GPU-SSD Integration](http://arxiv.org/abs/2505.09258v2)
  > **TL;DR**: Proposes Legend, a graph embedding system optimizing CPU-GPU-SSD integration for billion-scale datasets. Features efficient data placement, direct GPU-SSD access, and parallel execution to minimize I/O overhead. Achieves 4.8x speedup and matches 4-GPU performance using one GPU.
* `serving` `hardware` `quantization` [On the Partitioning of GPU Power among Multi-Instances](http://arxiv.org/abs/2501.17752v2)
  > **TL;DR**: Proposes ML-based models for GPU power estimation in multi-tenant environments using NVIDIA MIG technology. Utilizes partition-level utilization metrics to dynamically predict power consumption per instance. Achieves accurate power estimation across diverse workloads including LLM inference, aiding fair carbon reporting.
* `video` `networking` `offline` [Toward Accessible and Safe Live Streaming Using Distributed Content Filtering with MoQ](http://arxiv.org/abs/2505.08990v1)
  > **TL;DR**: Addresses real-time content moderation for live video streams with strict latency constraints. Extends Media Over QUIC protocol to distribute analysis tasks to clients and remove only objectionable segments. Increases client latency by only one group-of-pictures duration.
* `networking` `scaling` [ATLAHS: An Application-centric Network Simulator Toolchain for AI, HPC, and Distributed Storage](http://arxiv.org/abs/2505.08936v1)
  > **TL;DR**: Proposes ATLAHS, a network simulator toolchain for AI/HPC/storage workloads, using the GOAL format for tracing and simulating multi-job scenarios. Validated with <5% error, outperforming AstraSim in simulation runtime and trace efficiency, and applied to congestion control and job placement case studies.
* `scaling` [Local Constant Approximation for Dominating Set on Graphs Excluding Large Minors](http://arxiv.org/abs/2504.01091v3)
  > **TL;DR**: Investigates constant-factor approximation algorithms for Minimum Dominating Set in graphs excluding K_{2,t} minors. Introduces a distributed algorithm leveraging asymptotic dimension analysis, achieving a 50-approximation in f(t) rounds independent of graph size.
* `edge` `scaling` [Benchmarking of CPU-intensive Stream Data Processing in The Edge Computing Systems](http://arxiv.org/abs/2505.07755v1)
  > **TL;DR**: Investigates optimizing resource utilization in edge computing for CPU-intensive stream data processing. Uses a microbenchmark to profile performance-power relations under varied workloads and CPU frequencies. Achieves optimized edge resource usage by balancing computational efficiency with energy savings.
* `serving` `edge` `offloading` [LA-IMR: Latency-Aware, Predictive In-Memory Routing and Proactive Autoscaling for Tail-Latency-Sensitive Cloud Robotics](http://arxiv.org/abs/2505.07417v1)
  > **TL;DR**: Proposes LA-IMR, a control layer combining predictive edge-to-cloud offloading and proactive autoscaling to reduce tail latency for cloud-edge inference. Uses a utilization-driven latency model to steer QoS-aware routing and replica scaling, achieving 20.7% lower P99 latency under bursty workload.
* `training` `RL` `networking` [INTELLECT-2: A Reasoning Model Trained Through Globally Decentralized Reinforcement Learning](http://arxiv.org/abs/2505.07291v1)
  > **TL;DR**: Proposes globally distributed RL training for a 32B-parameter model. Develops PRIME-RL framework with TOPLOC verifier and SHARDCAST broadcast system for asynchronous training on heterogeneous compute. Achieves state-of-the-art reasoning results over centralized baseline (QwQ-32B).
* `storage` `offline` `serving` [TierBase: A Workload-Driven Cost-Optimized Key-Value Store](http://arxiv.org/abs/2505.06556v1)
  > **TL;DR**: Proposes TierBase, a distributed key-value store with workload-driven cost optimization. Integrates pre-trained data compression, elastic threading, and persistent memory to balance performance and storage costs. Achieves up to 62% cost reduction in real-world online data serving scenarios.
* `training` `networking` `scaling` [On Optimal Batch Size in Coded Computing](http://arxiv.org/abs/2505.06199v1)
  > **TL;DR**: Investigates optimal batch size and redundancy in coded computing for parallel job execution. Proposes joint optimization of redundancy level and batch size to minimize expected job completion time. Simulations show improvements in execution time for two service-time distributions.
* `networking` `serving` `RL` [Efficient Information Updates in Compute-First Networking via Reinforcement Learning with Joint AoI and VoI](http://arxiv.org/abs/2505.06025v1)
  > **TL;DR**: Proposes an Age-and-Value-Aware (AVA) metric for compute-first networking to optimize service information updates. Uses reinforcement learning to selectively transmit updates, reducing update frequency by over 90% on average without degrading task execution accuracy.
* `training` `hardware` `scaling` [Toward Heterogeneous, Distributed, and Energy-Efficient Computing with SYCL](http://arxiv.org/abs/2505.06022v1)
  > **TL;DR**: Proposes SYCL extensions for distributed heterogeneous computing. Introduces Celerity for workload distribution and SYnergy for energy efficiency on accelerators. Achieves scalable performance and up to 30% energy reduction in experiments.
* `training` `MoE` [DawnPiper: A Memory-scablable Pipeline Parallel Training Framework](http://arxiv.org/abs/2505.05856v1)
  > **TL;DR**: Addresses memory imbalance and inefficiency in pipeline parallelism for large model training. Proposes DawnPiper framework with fine-grained computation graph partitioning, a memory optimization algorithm, and automatic code generation. Achieves up to 4x larger batch size and 1.5x speedup over vPipe.
* `kernel` `hardware` [PUDTune: Multi-Level Charging for High-Precision Calibration in Processing-Using-DRAM](http://arxiv.org/abs/2505.05266v1)
  > **TL;DR**: Addresses precision calibration for Processing-Using-DRAM (PUD) to mitigate error-prone columns. Proposes PUDTune, a multi-level charging technique applying column-specific offsets to enhance error-free columns. Increases error-free columns by 1.81×, boosting arithmetic throughput by 1.88–1.89×.
* `training` `networking` `offline` [An Asynchronous Distributed-Memory Parallel Algorithm for k-mer Counting](http://arxiv.org/abs/2505.04431v1)
  > **TL;DR**: Presents an asynchronous distributed algorithm for k-mer counting to reduce global communication. Uses fine-grained messaging and custom aggregation to avoid Many-To-Many collectives. Achieves 9x faster performance than state-of-the-art and scales to 256 nodes.
* `training` `RL` `networking` [Decentralized Nonconvex Optimization under Heavy-Tailed Noise: Normalization and Optimal Convergence](http://arxiv.org/abs/2505.03736v1)
  > **TL;DR**: Proposes GT-NSGDm, a decentralized optimization method using gradient tracking and normalization, to handle heavy-tailed noise in nonconvex stochastic optimization. Achieves optimal convergence rate O(1/T^{(p-1)/(3p-2)}), and shows robustness in training language models.
* `offloading` `hardware` [EPOCH: Enabling Preemption Operation for Context Saving in Heterogeneous FPGA Systems](http://arxiv.org/abs/2501.16205v3)
  > **TL;DR**: Addresses context preservation challenges when preempting FPGA tasks in multi-tenant systems. Proposes EPOCH, a framework for state snapshot capture and restoration at any clock cycle. Achieves context save and restore in 62.2μs and 67.4μs per frame on ZynQ-XC7Z020.
* `thinking` `multi-modal` `RL` [A Hashgraph-Inspired Consensus Mechanism for Reliable Multi-Model Reasoning](http://arxiv.org/abs/2505.03553v1)
  > **TL;DR**: Proposes a Hashgraph-inspired consensus mechanism for multi-model reasoning systems to reduce hallucinations and inconsistencies. Treats each reasoning model as a black-box peer, using gossip-about-gossip communication and virtual voting for iterative convergence. Achieves higher accuracy and confidence by incorporating cross-verification across models.
* `serving` `scaling` ["Two-Stagification": Job Dispatching in Large-Scale Clusters via a Two-Stage Architecture](http://arxiv.org/abs/2505.03032v1)
  > **TL;DR**: Investigates how to optimize job dispatching for improved response times in large-scale clusters. Proposes a two-stage architecture with workload-type separation and classical dispatching policies. Achieves better mean response times than single-stage policies, approaching state-aware methods.
* `serving` `quantization` [PipeSpec: Breaking Stage Dependencies in Hierarchical LLM Decoding](http://arxiv.org/abs/2505.01572v1)
  > **TL;DR**: Proposes PipeSpec, a hierarchical pipeline speculation framework for asynchronous token generation and verification in LLM inference. Breaks sequential dependencies with pipe-staged execution and verifier for rollback. Achieves up to 2.54× speedup over baselines on LLaMA models with increasing gains at pipeline depth.
* `training` `quantization` `sparse` [Nesterov Method for Asynchronous Pipeline Parallel Optimization](http://arxiv.org/abs/2505.01099v1)
  > **TL;DR**: Addresses staleness in asynchronous pipeline training due to delayed gradients. Proposes a modified Nesterov Accelerated Gradient that adjusts the look-ahead step to handle gradient delays. Achieves up to 1.28× faster convergence than synchronous baselines on 1B parameter models.
* `agentic` `RL` `serving` [UserCentrix: An Agentic Memory-augmented AI Framework for Smart Spaces](http://arxiv.org/abs/2505.00472v1)
  > **TL;DR**: Proposes UserCentrix, an agentic memory-augmented framework for smart spaces with personalized LLM agents and hybrid control. Features proactive scaling, VoI-based decisions, and multi-agent coordination. Reduces computational resources by up to 45% while improving response accuracy by 30% in evaluations.
* `RAG` `offline` `edge` [Distributed Retrieval-Augmented Generation](http://arxiv.org/abs/2505.00443v1)
  > **TL;DR**: Proposes Distributed RAG (DRAG) to address privacy and scalability in centralized RAG. Uses Topic-Aware Random Walk for peer-to-peer knowledge retrieval in edge environments. Achieves near-centralized performance with half the messages of flooding.
* `hardware` `offline` [Exploration of Cryptocurrency Mining-Specific GPUs in AI Applications: A Case Study of CMP 170HX](http://arxiv.org/abs/2505.03782v1)
  > **TL;DR**: Explores repurposing mining-specific GPUs for AI inference via instruction set modifications. Tests CUDA modifications on NVIDIA CMP 170HX to restore FP32 performance, achieving >15× FP32 gain and >3× LLM inference speedup. Evaluates energy efficiency for cost-effective edge computing.
* `hardware` `inference` `offline` [Comparative Analysis of FPGA and GPU Performance for Machine Learning-Based Track Reconstruction at LHCb](http://arxiv.org/abs/2502.02304v4)
  > **TL;DR**: Compares FPGA and GPU performance for ML inference in LHCb track reconstruction. Focuses on MLP deployment via HLS4ML for FPGAs against GPU baseline, showing FPGAs achieve higher throughput and lower latency at significantly reduced power consumption.
* `training` `scaling` `kernel` [Galvatron: An Automatic Distributed System for Efficient Foundation Model Training](http://arxiv.org/abs/2504.21411v1)
  > **TL;DR**: Galvatron automates selection of hybrid parallelism strategies for efficient foundation model training. It uses a profiler, decision-tree/dynamic-programming search engine, and runtime to optimize across data, tensor, pipeline, and sequence parallelism. Achieves superior throughput over existing frameworks on various clusters.
* `networking` `scaling` `storage` [Raptr: Prefix Consensus for Robust High-Performance BFT](http://arxiv.org/abs/2504.18649v2)
  > **TL;DR**: Raptr is a BFT SMR protocol using Prefix Consensus to achieve robust low-latency high-throughput distributed consensus. It optimizes data dissemination to maintain performance under attacks, handling 260,000 TPS with sub-second latency and minimal degradation at 1% message drops.
* `networking` `training` [Towards Easy and Realistic Network Infrastructure Testing for Large-scale Machine Learning](http://arxiv.org/abs/2504.20854v1)
  > **TL;DR**: Proposes Genie, a framework for testing network infrastructure impact on ML workloads without GPUs. Uses CPU-emulated GPU communication and adapted ASTRA-sim for network-workload interaction modeling. Achieves accurate emulation of network conditions for large-scale ML systems.
* `offloading` `edge` `networking` [Intelligent Task Offloading in VANETs: A Hybrid AI-Driven Approach for Low-Latency and Energy Efficiency](http://arxiv.org/abs/2504.20735v1)
  > **TL;DR**: Addresses high latency and energy inefficiency in vehicular task offloading. Proposes hybrid AI framework with supervised learning, RL, and PSO for dynamic offloading decisions and resource allocation. Reduces latency by 30% and energy consumption by 25% in simulations.
* `hardware` `scaling` `training` [Good things come in small packages: Should we build AI clusters with Lite-GPUs?](http://arxiv.org/abs/2501.10187v2)
  > **TL;DR**: Proposes building AI clusters with numerous small-scale Lite-GPUs instead of large monolithic GPUs for improved scalability and efficiency. Leverages advancements in co-packaged optics for high-bandwidth communication. Achieves reduced manufacturing costs, improved power efficiency, yield, and reduced blast radius compared to traditional GPU clusters.
* `training` `scaling` [Hetu v2: A General and Scalable Deep Learning System with Hierarchical and Heterogeneous Single Program Multiple Data Annotations](http://arxiv.org/abs/2504.20490v1)
  > **TL;DR**: Addresses inefficient DL training under workload heterogeneity from mixed hardware or uneven data. Proposes HSPMD, extending SPMD with asymmetric sharding, hierarchical communication, and dynamic graph switching for spatial/temporal adaptation. Achieves performance matching or exceeding specialized systems on heterogeneous clusters and elastic training.
* `scaling` `training` `edge` [Electricity Cost Minimization for Multi-Workflow Allocation in Geo-Distributed Data Centers](http://arxiv.org/abs/2504.20105v1)
  > **TL;DR**: Proposes an electricity-cost-aware scheduling algorithm (ECMWS) for multi-workflow allocation in geo-distributed data centers. Uses graph embedding and policy network to solve MDP for deadline-constrained resource scheduling. Achieves over 15% cost reduction compared to state-of-the-art methods.
* `training` `networking` `scaling` [The Big Send-off: High Performance Collectives on GPU-based Supercomputers](http://arxiv.org/abs/2504.18658v1)
  > **TL;DR**: Addresses inefficiencies in collective communication for LLM training on large GPU clusters. Introduces PCCL, an optimized communication library for all-gather/reduce-scatter operations that maximizes network/compute utilization. Delivers 6-33x speedups over RCCL and up to 60% end-to-end training speedup for GPT-3-style models.
* `serving` `edge` `scaling` [Adaptive Heuristics for Scheduling DNN Inferencing on Edge and Cloud for Personalized UAV Fleets](http://arxiv.org/abs/2412.20860v2)
  > **TL;DR**: Proposes deadline-driven heuristics (DEMS-A and GEMS) for scheduling DNN inferencing tasks from UAV fleets on edge and cloud. Uses strategies like task dropping, work stealing, migration, and adaptation to cloud variability to maximize QoS and QoE. Achieves up to 88% task completion rate and 2.7x higher QoS utility.
* `edge` `quantization` `offloading` [EPSILON: Adaptive Fault Mitigation in Approximate Deep Neural Network using Statistical Signatures](http://arxiv.org/abs/2504.20074v1)
  > **TL;DR**: Proposes EPSILON, a lightweight framework for adaptive fault mitigation in approximate deep neural networks using statistical signatures and layer-wise metrics. Achieves 80.05% model accuracy with 22% faster inference and 28% improved energy efficiency on edge devices under faults.
* `training` `networking` `scaling` [Cross-region Model Training with Communication-Computation Overlapping and Delay Compensation](http://arxiv.org/abs/2504.17672v1)
  > **TL;DR**: Addresses inefficiency in cross-region LLM training due to high latency. Proposes CoCoDC with Delay Compensation via Taylor expansion and Adaptive Transmission to optimize synchronization. Reduces training steps by up to 21.0% compared to baselines to reach target perplexity.
* `training` `networking` `sparse` [GRANITE : a Byzantine-Resilient Dynamic Gossip Learning Framework](http://arxiv.org/abs/2504.17471v1)
  > **TL;DR**: Proposes GRANITE, a Byzantine-resilient gossip learning framework using a history-aware peer sampling protocol and adaptive probabilistic threshold for model aggregation. Ensures convergence with 30% Byzantine nodes and operates over 9× sparser graphs than current theory allows.
* `offline` `kernel` `training` [Scalable and Performant Data Loading](http://arxiv.org/abs/2504.20067v1)
  > **TL;DR**: Addresses the data loading bottleneck in AI training. Proposes SPDL, a library using parallel preprocessing with GIL release for efficient array data transfer to GPU. Achieves 74% faster ImageNet iteration and 38% lower CPU usage versus PyTorch DataLoader.
* `edge` `networking` `serving` [6G EdgeAI: Performance Evaluation and Analysis](http://arxiv.org/abs/2504.16529v1)
  > **TL;DR**: Investigates integrated communication-computing for low-latency edge LLM services. Proposes a framework embedding compute in RAN nodes with joint resource management. Achieves 60% higher service capacity vs 5G MEC under transformer workloads.
* `training` `storage` `scaling` [Deep RC: A Scalable Data Engineering and Deep Learning Pipeline](http://arxiv.org/abs/2502.20724v2)
  > **TL;DR**: Addresses scalable data preprocessing and training pipelines for deep learning in HPC environments. Proposes Deep RC, a heterogeneous pipeline integrating data frameworks and deep learning with MPI/GLOO/NCCL support. Reduces pipeline time by 3.28 and 75.9 seconds for different models under identical resources.
* `training` `kernel` `offloading` [Scaling Neural-Network-Based Molecular Dynamics with Long-Range Electrostatic Interactions to 51 Nanoseconds per Day](http://arxiv.org/abs/2504.15508v1)
  > **TL;DR**: Optimizes neural-network-based molecular dynamics with long-range electrostatics for high-efficiency simulations. Introduces hardware-offloaded FFT, computation overlapping, and ring-based load balancing to accelerate DPLR. Achieves 37x speedup, reaching 51 ns/day simulation speed on Fugaku supercomputer.
* `kernel` `serving` `sparse` [ClusterViG: Efficient Globally Aware Vision GNNs via Image Partitioning](http://arxiv.org/abs/2501.10640v2)
  > **TL;DR**: Addresses computational bottlenecks in Vision GNNs via dynamic efficient graph convolution (DEGC), partitioning images for parallel graph construction and global feature learning. ClusterViG architecture achieves 5× inference latency reduction with similar parameter count.
* `training` `MoE` `networking` [Advancing MoE Efficiency: A Collaboration-Constrained Routing (C2R) Strategy for Better Expert Parallelism Design](http://arxiv.org/abs/2504.01337v2)
  > **TL;DR**: Addresses inefficiencies in MoE model parallelism due to imbalanced expert activation and communication overhead. Proposes Collaboration-Constrained Routing (C2R) to specialize expert groups via group-constrained routing. Reduces all2all communication costs by 20%-30% over MegaBlocks while improving accuracy.
* `training` `RL` `RAG` [GENE-FL: Gene-Driven Parameter-Efficient Dynamic Federated Learning](http://arxiv.org/abs/2504.14628v1)
  > **TL;DR**: Addresses communication inefficiency and model initialization in dynamic federated learning. Proposes GENE-FL framework using Learngene to compress models via Fisher-based parameter constraints and sensitivity analysis. Reduces communication costs by 4x with 9.04 MB initialization size.
* `training` `multi-modal` `scaling` [PipeWeaver: Addressing Data Dynamicity in Large Multimodal Model Training with Dynamic Interleaved Pipeline](http://arxiv.org/abs/2504.14145v1)
  > **TL;DR**: Proposes PipeWeaver, a dynamic pipeline scheduler for efficient multimodal model training. Uses adaptive modality-aware partitioning and hierarchical schedule search with SEMU simulator. Improves training efficiency by up to 97.3% over SOTA systems.
* `edge` `training` `RL` [Collaborative Learning of On-Device Small Model and Cloud-Based Large Model: Advances and Future Directions](http://arxiv.org/abs/2504.15300v1)
  > **TL;DR**: Surveys collaborative learning between on-device small models and cloud-based large models to address latency, cost, and privacy. Categorizes collaboration algorithms into data, feature, and parameter-based frameworks. Reviews real-world deployments like recommender systems and mobile livestreaming.
* `edge` `diffusion` `quantization` [Diffusion Models on the Edge: Challenges, Optimizations, and Applications](http://arxiv.org/abs/2504.15298v1)
  > **TL;DR**: Addresses deployment challenges of diffusion models on edge devices. Surveys model compression, efficient sampling, and hardware-software co-design optimizations. Highlights methods achieving up to 10x latency reduction and energy savings while maintaining output quality.
* `edge` `RAG` `offloading` [Efficient Distributed Retrieval-Augmented Generation for Enhancing Language Model Performance](http://arxiv.org/abs/2504.11197v2)
  > **TL;DR**: Proposes DRAGON, a distributed RAG framework for on-device SLMs that decomposes multi-document retrieval into parallel token generation with speculative aggregation and network-aware scheduling. Achieves 1.9x performance gain over standalone SLM and reduces per-token latency substantially with negligible TTFT overhead.
* `kernel` `training` `quantization` [TurboFNO: High-Performance Fourier Neural Operator with Fused FFT-GEMM-iFFT on GPU](http://arxiv.org/abs/2504.11681v1)
  > **TL;DR**: Addresses inefficiency in Fourier Neural Operator (FNO) training by fusing FFT-GEMM-iFFT stages into a single GPU kernel. Proposes TurboFNO with custom FFT/GEMM kernels and shared memory optimizations to reduce global memory traffic. Achieves 150% speedup over cuBLAS/cuFFT on A100.
* `serving` `offline` `scaling` [Transformer-Based Model for Cold Start Mitigation in FaaS Architecture](http://arxiv.org/abs/2504.11338v1)
  > **TL;DR**: Addresses cold start latency in serverless FaaS architectures using Transformer models to predict and prewarm functions. Achieves up to 79% reduction in cold start times compared to baselines.
* `training` `storage` [Morphing-based Compression for Data-centric ML Pipelines](http://arxiv.org/abs/2504.11067v1)
  > **TL;DR**: Proposes BWARE, a workload-aware lossless matrix compression method that leverages data transformations and morphing for efficient compressed operations. Reduces training time for data-centric ML pipelines from days to hours by minimizing I/O and improving parallelism.
* `edge` `offloading` `networking` [High-Efficiency Split Computing for Cooperative Edge Systems: A Novel Compressed Sensing Bottleneck](http://arxiv.org/abs/2504.15295v1)
  > **TL;DR**: Proposes HECS-B, a compressed sensing-based split computing architecture for edge-cloud systems. Integrates a compressed sensing autoencoder at a DNN bottleneck layer to reduce bandwidth usage and accelerate processing. Achieves 50% bandwidth reduction and 60% speed-up while maintaining accuracy.
* `training` `kernel` `storage` [Optimizing Data Distribution and Kernel Performance for Efficient Training of Chemistry Foundation Models: A Case Study with MACE](http://arxiv.org/abs/2504.10700v1)
  > **TL;DR**: Optimizes data distribution and kernel performance for efficient training of chemistry foundation models (CFMs). Addresses load balancing via multi-objective bin packing and optimizes symmetric tensor contraction kernel. Achieves 6× speedup, reducing per-epoch time from 12 to 2 minutes on 740 GPUs.
* `serving` `networking` `scaling` [Load Balancing with Network Latencies via Distributed Gradient Descent](http://arxiv.org/abs/2504.10693v1)
  > **TL;DR**: Proposes Distributed Gradient Descent Load Balancing (DGD-LB) for globally distributed LLM serving to minimize latency. Uses decentralized gradient descent with delayed feedback on backend states. Achieves substantial latency reduction compared to baselines, especially under high network delays.
* `training` `LoRA` `MoE` [DeepCompile: A Compiler-Driven Approach to Optimizing Distributed Deep Learning Training](http://arxiv.org/abs/2504.09983v1)
  > **TL;DR**: Proposes DeepCompile, a compiler-driven framework that optimizes distributed training via dynamic memory-aware scheduling of operations. Applies profiling-guided optimization passes to improve communication-computation overlap and coordinate optimizations like prefetching and offloading. Achieves up to 1.54x speedup over FSDP for Llama 3 70B.
* `kernel` `sparse` [A Nonlinear Hash-based Optimization Method for SpMV on GPUs](http://arxiv.org/abs/2504.08860v1)
  > **TL;DR**: Proposes a hash-based partition (HBP) format for sparse matrix-vector multiplication (SpMV) on GPUs. Introduces competitive load balancing and hash transformation to accelerate pre-processing. Achieves up to 3.5x speedup in pre-processing and 3.3x speedup in SpMV execution compared to baselines.
* `training` `storage` `scaling` [A Hybrid Cloud Management Plane for Data Processing Pipelines](http://arxiv.org/abs/2504.08225v1)
  > **TL;DR**: Proposes Titchener, a hybrid-cloud management plane for data processing pipelines, enabling seamless deployment across cloud environments. Uses Kubernetes-based control planes for global discovery and connectivity. Reduces deployment complexity by 60% in real-world workflows.
* `agentic` `RAG` `serving` [Orchestrating Agents and Data for Enterprise: A Blueprint Architecture for Compound AI](http://arxiv.org/abs/2504.08148v1)
  > **TL;DR**: Proposes a blueprint architecture for compound AI systems using streams to orchestrate agents and data for enterprise applications. Features agent registry, data registry, and planners to optimize for quality of service (cost, accuracy, latency). Demonstrates an implementation for HR with optimized QoS compliance.
* `training` `quantization` `offloading` [GPT Carry-On: Training Foundation Model for Customization Could Be Simple, Scalable and Affordable](http://arxiv.org/abs/2504.07513v1)
  > **TL;DR**: Proposes GPT Carry-On, a lightweight adapter method to customize large language models by training compact modules on pretrained embeddings. Enables outsourcing computations to inference nodes, reducing GPU memory usage to <1GB for a 100M parameter module on 30B LLMs. Achieves faster convergence and efficient task specialization with small computation.
* `scaling` `networking` `serving` [ICPS: Real-Time Resource Configuration for Cloud Serverless Functions Considering Affinity](http://arxiv.org/abs/2504.06512v1)
  > **TL;DR**: Addresses resource allocation inefficiency in serverless workflows with branching structures. Proposes ICPS, which predicts concurrency with LSTM and uses affinity-based function co-location to reduce network delays. Experimental results show consistent performance improvement over existing approaches.
* `quantization` `serving` `kernel` [MILLION: Mastering Long-Context LLM Inference Via Outlier-Immunized KV Product Quantization](http://arxiv.org/abs/2504.03661v2)
  > **TL;DR**: Proposes MILLION, a KV cache quantization framework using outlier-immunized product quantization to enhance inference efficiency. Key techniques include non-uniform quantization via product quantization and optimized GPU kernels with sparse computation. Achieves 4-bit quantization with minimal accuracy loss and 2.09x speedup at 32K context.
* `training` `sparse` `networking` [TAGC: Optimizing Gradient Communication in Distributed Transformer Training](http://arxiv.org/abs/2504.05638v1)
  > **TL;DR**: Addresses gradient synchronization bottleneck in distributed transformer training. Proposes TAGC, a transformer-aware gradient compression algorithm with layer-selective compression and dynamic sparsification. Achieves up to 15% training acceleration compared to standard FSDP with minimal accuracy loss.
* `offloading` `hardware` `storage` [dpBento: Benchmarking DPUs for Data Processing](http://arxiv.org/abs/2504.05536v1)
  > **TL;DR**: Presents dpBento, a benchmark suite to evaluate DPU offloading for data processing tasks. Measures performance of various data operations and systems across DPU resources, highlighting offloading benefits for compute, memory, and storage efficiency.
* `kernel` `hardware` `training` [oneDAL Optimization for ARM Scalable Vector Extension: Maximizing Efficiency for High-Performance Data Science](http://arxiv.org/abs/2504.04241v1)
  > **TL;DR**: Optimizes oneDAL for ARM SVE to enhance ML performance. Ports library to ARM with OpenBLAS, introduces vectorized routines and SVE-optimized SVM. Achieves up to 200x speedup vs. scikit-learn on ARM and cost parity with x86 systems.
* `storage` `quantization` `sparse` [IPComp: Interpolation Based Progressive Lossy Compression for Scientific Applications](http://arxiv.org/abs/2502.04093v3)
  > **TL;DR**: Proposes IPComp, an interpolation-based progressive lossy compression with multi-level bitplane and predictive coding for scientific data. Achieves up to 487% higher compression ratios and 698% faster speed than state-of-the-art progressive compressors.
* `training` [Accurate GPU Memory Prediction for Deep Learning Jobs through Dynamic Analysis](http://arxiv.org/abs/2504.03887v1)
  > **TL;DR**: Addresses GPU memory prediction for deep learning training to prevent OOM errors. Proposes VeritasEst, a CPU-based dynamic analysis tool for offline peak memory estimation. Reduces relative error by 84% and failure probability by 73% compared to baselines.
* `training` `kernel` [TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives](http://arxiv.org/abs/2503.20313v3)
  > **TL;DR**: Proposes TileLink, a framework for generating compute-communication overlapped kernels via tile-centric primitives and backend integration. Achieves 1.17× to 20.76× speedup over non-overlapping baselines.
* `inference` `hardware` [Exploring energy consumption of AI frameworks on a 64-core RV64 Server CPU](http://arxiv.org/abs/2504.03774v1)
  > **TL;DR**: Analyzes energy consumption of ML inference frameworks on RISC-V hardware. Benchmarks PyTorch, ONNX Runtime, and TensorFlow on 64-core RV64 CPU, finding XNNPACK-backed frameworks (ONNX, TF) consume less energy than OpenBLAS-backed PyTorch. Quantifies up to lower energy usage in specific setups.
* `training` `edge` `storage` [GraphGen+: Advancing Distributed Subgraph Generation and Graph Learning On Industrial Graphs](http://arxiv.org/abs/2503.06212v2)
  > **TL;DR**: Addresses the inefficiency of distributed subgraph generation and high storage overhead in graph learning. Proposes GraphGen+, which synchronizes distributed subgraph generation with in-memory learning. Achieves up to 27× subgraph generation speedup and supports 1 million nodes per iteration.
* `quantization` `offloading` `kernel` [A Pilot Study on Tunable Precision Emulation via Automatic BLAS Offloading](http://arxiv.org/abs/2503.22875v2)
  > **TL;DR**: Explores precision emulation for HPC workloads via automatic BLAS offloading and INT8 quantization. Uses low-bitwidth integers and unified memory to emulate double-precision operations without code modifications. Achieves improved accuracy and performance simultaneously in MuST application.
* `edge` `training` `RAG` [Satellite Edge Artificial Intelligence with Large Models: Architectures and Technologies](http://arxiv.org/abs/2504.01676v1)
  > **TL;DR**: Proposes architectures for efficient large AI model deployment on satellites. Designs federated fine-tuning over space-ground networks and microservice-based lightweight partitioning for inference. Addresses resource constraints to enable real-time downstream tasks with reduced latency for applications like extreme weather nowcasting.
* `scaling` `networking` `storage` [Carbon and Reliability-Aware Computing for Heterogeneous Data Centers](http://arxiv.org/abs/2504.00518v1)
  > **TL;DR**: Proposes a carbon and reliability-aware optimization framework for workload migration across distributed data centers. Formulates a mixed-integer problem minimizing operational/embodied carbon while meeting SLAs, with server dispatch and backup allocation. Reduces total carbon emissions by up to 21%.
* `scaling` `hardware` [Rack Position Optimization in Large-Scale Heterogeneous Data Centers](http://arxiv.org/abs/2504.00277v1)
  > **TL;DR**: Addresses rack positioning optimization in large data centers to balance operational efficiency and fault tolerance. Proposes a two-tier framework with DRL-guided heuristic for scalable rack placement. Achieves 30% better objective than MIP solvers and 100% success rate in just 2 minutes for 100k-position scale.
* `networking` `training` [GPU-centric Communication Schemes for HPC and ML Applications](http://arxiv.org/abs/2503.24230v1)
  > **TL;DR**: Surveys GPU-centric communication schemes to move the control path from CPU to GPU for reducing overheads in distributed HPC and ML workloads. Leverages GPU and NIC capabilities for direct data transfer without host staging. Discusses techniques addressing bottlenecks in parallel execution.
* `serving` `offline` [Deep Learning Model Deployment in Multiple Cloud Providers: an Exploratory Study Using Low Computing Power Environments](http://arxiv.org/abs/2503.23988v1)
  > **TL;DR**: Explores cost-effective deployment of deep learning models in low-power cloud environments for inference. Evaluates AWS, Google Cloud, and Azure across 7 execution environments using CPU vs GPU tradeoffs. Achieves 50% cost reduction with CPUs leveraging cache optimization, while GPUs cost 300% more on average.
* `training` `offloading` `networking` [FeedSign: Robust Full-parameter Federated Fine-tuning of Large Models with Extremely Low Communication Overhead of One Bit](http://arxiv.org/abs/2501.17610v2)
  > **TL;DR**: Addresses communication overhead in federated fine-tuning of large models. Proposes FeedSign, a method using zeroth-order optimization and shared PRNGs to reduce upload/download to 1 bit per step. Achieves convergence rate O(e^{-t}) with orders-of-magnitude lower communication compared to baselines.
* `training` `scaling` [Optimizing Distributed Training Approaches for Scaling Neural Networks](http://arxiv.org/abs/2503.23186v1)
  > **TL;DR**: Compares distributed training strategies for neural networks. Proposes adaptive scheduling algorithm to switch between data/model parallelism based on resources. Achieves 3.2x speedup with hybrid parallelism and 18% efficiency gain via adaptive scheduling.
* `hardware` `serving` `RAG` [PilotANN: Memory-Bounded GPU Acceleration for Vector Search](http://arxiv.org/abs/2503.21206v1)
  > **TL;DR**: Proposes PilotANN, a hybrid CPU-GPU system for vector search to accelerate ANNS under GPU memory constraints. Decomposes graph traversal into GPU subgraph processing with SVD and CPU refinement. Achieves 3.9-5.4× throughput speedup on 100M datasets and handles 12× larger datasets.
* `scaling` `training` `offline` [Cloud Resource Allocation with Convex Optimization](http://arxiv.org/abs/2503.21096v1)
  > **TL;DR**: Proposes a convex optimization resource allocation framework for Kubernetes clusters to dynamically select node types minimizing cost and fragmentation. Uses logarithmic approximation and interior-point methods to transform discrete scaling into convex problem. Reduces costs by 20% and improves resource utilization over baseline autoscalers.
* `serving` `edge` `offloading` [Solving AI Foundational Model Latency with Telco Infrastructure](http://arxiv.org/abs/2504.03708v1)
  > **TL;DR**: Proposes using telco infrastructure as hierarchical AI edges for caching and partial inference to reduce LLM latency. Introduces tiered caching strategies and split-inference architectures. Achieves significant latency reduction and compute cost savings for real-time applications.
* `networking` `scaling` `storage` [INDIGO: Page Migration for Hardware Memory Disaggregation Across a Network](http://arxiv.org/abs/2503.18140v1)
  > **TL;DR**: Addresses performance degradation in hardware-memory-disaggregated systems due to remote memory access. Proposes INDIGO, a network-aware page migration framework using page telemetry and learning-based adaptation. Achieves 50-70% application performance improvement and reduces network traffic by 2x.
* `edge` `storage` `serving` [DEEP: Edge-based Dataflow Processing with Hybrid Docker Hub and Regional Registries](http://arxiv.org/abs/2504.08741v1)
  > **TL;DR**: Proposes DEEP, a hybrid Docker registry system using regional registries to optimize edge deployment of ML microservices. Compares deployments from Docker Hub, regional registries, and hybrid approach, showing 0.34% energy reduction for text processing microservices via hybrid deployment.
* `training` `scaling` `networking` [Using a Market Economy to Provision Compute Resources Across Planet-wide Clusters](http://arxiv.org/abs/2503.17691v1)
  > **TL;DR**: Proposes a market-based framework for resource provisioning across heterogeneous clusters to balance supply and demand. Implements simulated clock auctions with dynamic reserve prices to influence user bids. Achieves transition of users to under-utilized resource pools, reducing shortages/surpluses.
* `hardware` `kernel` [Debunking the CUDA Myth Towards GPU-based AI Systems](http://arxiv.org/abs/2501.00210v2)
  > **TL;DR**: Evaluates Intel Gaudi NPUs as alternative to NVIDIA GPUs for AI systems. Benchmarks performance in AI operations, workloads, and software optimization for FBGEMM and vLLM operators. Shows Gaudi-2 achieves comparable energy efficiency to A100 but requires software maturity improvements.
* `edge` `inference` `hardware` [Achieving Dependability of AI Execution with Radiation Hardened Processors](http://arxiv.org/abs/2504.03680v1)
  > **TL;DR**: Proposes using a radiation-hardened co-processor (HPDP) with optimized AI runtime (Klepsydra) for dependable, low-latency AI inference in harsh environments. Achieves high-throughput streaming data processing without increased power consumption. Demonstrates suitability for space, nuclear, and medical applications.
* `RL` `serving` `quantization` [GREEN-CODE: Learning to Optimize Energy Efficiency in LLM-based Code Generation](http://arxiv.org/abs/2501.11006v2)
  > **TL;DR**: Proposes GREEN-CODE, an RL-based framework for energy-efficient LLM inference in code generation via dynamic early exits. Achieves 23-50% energy reduction on JavaCorpus and PY150 datasets without significant accuracy loss.
* `training` `networking` `scaling` [DeFT: Mitigating Data Dependencies for Flexible Communication Scheduling in Distributed Training](http://arxiv.org/abs/2503.16815v1)
  > **TL;DR**: Proposes DeFT, a communication scheduling scheme for distributed training that mitigates data dependencies via delayed updates and reformulates scheduling as knapsack problems. Achieves speedups of 29% to 115% on 16 A100 GPUs with no accuracy loss.
* `networking` [Contemplating a Lightweight Communication Interface for Asynchronous Many-Task Systems](http://arxiv.org/abs/2503.15400v1)
  > **TL;DR**: Proposes a lightweight communication interface (LCI) for efficient networking in Asynchronous Many-Task systems. Introduces objectized flexible functions for expressive primitives and fine-grained resource mapping targeting AMT concurrency needs. Achieves optimized communication throughput compared to existing libraries.
* `edge` `networking` `offloading` [Communication-Efficient Distributed On-Device LLM Inference Over Wireless Networks](http://arxiv.org/abs/2503.14882v1)
  > **TL;DR**: Proposes a distributed on-device LLM inference framework using tensor parallelism and over-the-air computation for communication-efficient collaboration among edge devices. Jointly optimizes model assignment and transceiver to minimize transmission error, achieving 5x inference speed acceleration.
* `kernel` `hardware` `serving` [Fake Runs, Real Fixes -- Analyzing xPU Performance Through Simulation](http://arxiv.org/abs/2503.14781v1)
  > **TL;DR**: Proposes xPU-Shark, a fine-grained ML accelerator performance analysis method using hardware simulation. Captures traces from production LLM deployments and replays in a microarchitecture simulator to identify inefficiencies. Optimizes a collective by 15% and reduces token latency by 4.1%.
* `training` `RL` `RAG` [Towards Resource-Efficient Compound AI Systems](http://arxiv.org/abs/2501.16634v3)
  > **TL;DR**: Proposes a declarative workflow programming model and adaptive runtime system for resource-efficient Compound AI Systems. Murakkab prototype decouples application logic from execution, enabling dynamic scheduling and resource-aware decisions. Achieves ∼3.4× faster workflow completion and ∼4.5× higher energy efficiency.
* `networking` `offloading` [Optimizing Data Transfer Performance and Energy Efficiency with Deep Reinforcement Learning](http://arxiv.org/abs/2503.13662v1)
  > **TL;DR**: Proposes a reinforcement learning framework for efficient large-scale data transfers. The method dynamically adjusts transfer settings and pauses threads during heavy network use to balance throughput and energy efficiency. Achieves 25% higher throughput and 40% less energy usage compared to baselines.
* `MoE` `edge` `networking` [Optimal Expert Selection for Distributed Mixture-of-Experts at the Wireless Edge](http://arxiv.org/abs/2503.13421v1)
  > **TL;DR**: Proposes DMoE protocol for distributed Mixture-of-Experts at wireless edge, addressing joint expert selection and subcarrier allocation. Introduces Dynamic Expert Selection algorithm with linear relaxation and tunable optimization for AI-channel tradeoff. Achieves high performance with reduced communication cost.
* `video` `sparse` `training` [DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training](http://arxiv.org/abs/2502.07590v3)
  > **TL;DR**: Proposes DSV, a dynamic sparsity exploitation method for accelerating video DiT training. Utilizes two-stage low-rank approximation and custom kernels for sparse attention, plus sparsity-aware hybrid parallelism. Achieves 3.02x higher throughput at 128 GPUs with 520k token lengths.
* `edge` `offloading` `scaling` [CCRSat: A Collaborative Computation Reuse Framework for Satellite Edge Computing Networks](http://arxiv.org/abs/2503.11946v1)
  > **TL;DR**: Proposes CCRSat, a collaborative computation reuse framework for satellite edge networks using local reuse assessment and inter-satellite data similarity sharing. Reduces task completion time by up to 62.1% and computational resource consumption by up to 28.8%.
* `serving` `offloading` `training` [Alchemist: Towards the Design of Efficient Online Continual Learning System](http://arxiv.org/abs/2503.01066v2)
  > **TL;DR**: Addresses redundant computation in online continual learning for LLMs by reusing serving activations. Proposes storing prefill-phase activations/KV cache and smart offloading techniques. Achieves up to 1.72x training throughput increase with 47% memory reduction during training.
* `serving` `hardware` `scaling` [ARCAS: Adaptive Runtime System for Chiplet-Aware Scheduling](http://arxiv.org/abs/2503.11460v1)
  > **TL;DR**: Addresses memory contention in chiplet-based CPUs for parallel applications. Proposes ARCAS runtime with chiplet-aware task scheduling, NUMA-aware memory allocation, and performance monitoring. Achieves significant performance improvements in memory-intensive applications through adaptive task migration and optimization.
* `training` `kernel` `scaling` [FastCHGNet: Training one Universal Interatomic Potential to 1.5 Hours with 32 GPUs](http://arxiv.org/abs/2412.20796v2)
  > **TL;DR**: Proposes FastCHGNet to accelerate training of graph neural network interatomic potentials. Introduces Force/Stress readout modules and GPU optimizations (kernel fusion, load balancing) for multi-GPU scaling. Achieves 3.59× memory reduction and 1.53-hour training time on 32 GPUs (vs 8.3 days on single GPU).
* `training` `scaling` `networking` [Power-Aware Scheduling for Multi-Center HPC Electricity Cost Optimization](http://arxiv.org/abs/2503.11011v1)
  > **TL;DR**: Proposes TARDIS, a power-aware scheduler for multi-center HPC systems, using GNN-based job power prediction and spatio-temporal scheduling to minimize electricity costs. Achieves 10-20% cost reduction by shifting workloads across time and locations using variable electricity pricing.
* `training` `scaling` `sparse` [Galvatron: Automatic Distributed Training for Large Transformer Models](http://arxiv.org/abs/2504.03662v1)
  > **TL;DR**: Proposes Galvatron, a framework for automatically combining data, tensor, and pipeline parallelism to optimize large transformer training. It dynamically selects and adjusts strategies based on model and hardware. Achieves higher throughput compared to static frameworks.
* `serving` `training` `edge` [Concurrent Scheduling of High-Level Parallel Programs on Multi-GPU Systems](http://arxiv.org/abs/2503.10516v1)
  > **TL;DR**: Presents a runtime system for concurrent scheduling of GPU-accelerated distributed programs. Introduces an instruction-graph IR to decouple dependency analysis from execution, using lookahead for memory optimization. Achieves strong scaling on 128 GPUs with reduced runtime delays.
* `RL` `training` `edge` [Efficient Federated Fine-Tuning of Large Language Models with Layer Dropout](http://arxiv.org/abs/2503.10217v1)
  > **TL;DR**: Addresses inefficiency in federated fine-tuning of LLMs on resource-constrained devices. Proposes DropPEFT with adaptive layer dropout to reduce computational and memory overhead. Achieves 1.3-6.3× convergence speedup and 40%-67% memory reduction.
* `training` `scaling` [Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo](http://arxiv.org/abs/2503.09799v1)
  > **TL;DR**: Analyzes scaling laws for DiLoCo, a communication-efficient training method for LLMs. Shows that DiLoCo predictably scales better than data-parallel training, improving batch sizes, generalization, and evaluation loss with model size. Achieves superior performance even at small scales under fixed compute.
* `edge` `serving` [Evaluating Multi-Instance DNN Inferencing on Multiple Accelerators of an Edge Device](http://arxiv.org/abs/2503.09546v1)
  > **TL;DR**: Examines multi-instance DNN inferencing on edge accelerators (GPU CUDA, Tensor Cores, DLA). Evaluates throughput and latency with varied batch sizes and hardware combinations. Identifies contention issues and achieves up to 20% throughput gain by balancing GPU resources.
* `training` `scaling` [Automatic Operator-level Parallelism Planning for Distributed Deep Learning -- A Mixed-Integer Programming Approach](http://arxiv.org/abs/2503.09357v1)
  > **TL;DR**: Proposes automatic operator-level parallelism planning using mixed-integer programming to optimize distributed deep learning for complex models. Formulates as a scheduling problem with bi-level optimization to balance optimality and efficiency. Reduces computational bubbles by half compared to expert strategies like DualPipe under same memory constraints.
* `hardware` `training` `storage` [FpgaHub: Fpga-centric Hyper-heterogeneous Computing Platform for Big Data Analytics](http://arxiv.org/abs/2503.09318v1)
  > **TL;DR**: Proposes FpgaHub, an FPGA-centric hyper-heterogeneous computing platform for big data analytics. Uses FPGA as a hub for data movement, scheduling, and pre-processing to exploit hardware heterogeneity. Improves performance by leveraging reconfigurable computing and rich IO interfaces.
* `edge` `scaling` `storage` [GMB-ECC: Guided Measuring and Benchmarking of the Edge Cloud Continuum](http://arxiv.org/abs/2503.07183v2)
  > **TL;DR**: Proposes GMB-ECC to optimize energy efficiency across edge-cloud computing layers. Introduces a precision parameter for adaptable energy measurement in heterogeneous systems. Uses parameter tuning to achieve optimal energy savings without performance loss.
* `hardware` `serving` `scaling` [VersaSlot: Efficient Fine-grained FPGA Sharing with Big.Little Slots and Live Migration in FPGA Cluster](http://arxiv.org/abs/2503.05930v2)
  > **TL;DR**: Proposes VersaSlot, a FPGA sharing system with Big.Little slots and live migration to resolve reconfiguration contention and task blocking in clusters. Achieves up to 13.66x lower average response time and improves resource utilization by 35% (LUT) and 29% (FF).
* `serving` `scaling` `offline` [Dilu: Enabling GPU Resourcing-on-Demand for Serverless DL Serving via Introspective Elasticity](http://arxiv.org/abs/2503.05130v1)
  > **TL;DR**: Addresses GPU resource fragmentation in serverless DL serving. Proposes Dilu, a system with introspective elasticity for fine-grained two-dimensional co-scaling and dynamic GPU allocation. Achieves up to 1.8× inference throughput improvement and 10%-46% GPU defragmentation.
* `RL` `edge` `offloading` [Incentivizing Multi-Tenant Split Federated Learning for Foundation Models at the Network Edge](http://arxiv.org/abs/2503.04971v1)
  > **TL;DR**: Proposes PRINCE, a price-incentive mechanism for multi-tenant split federated learning (SFL) at the network edge to coordinate device participation in FM fine-tuning. Includes bias-resilient aggregation, convergence-bound-based contribution evaluation, and Stackelberg-game-based strategy optimization. Accelerates fine-tuning by up to 3.07x vs baselines.
* `offline` `parallelization` `optimization` [Parallel-in-Time Kalman Smoothing Using Orthogonal Transformations](http://arxiv.org/abs/2502.11686v2)
  > **TL;DR**: Proposes a parallel-in-time linear Kalman smoother using novel QR factorization and selective inversion for state estimation. Implements with TBB, achieving up to 47x speedup on 64 cores despite higher serial arithmetic (1.8-2.5x slower on single core). Outperforms existing parallel smoothers.
* `edge` `offline` `quantization` [Ecomap: Sustainability-Driven Optimization of Multi-Tenant DNN Execution on Edge Servers](http://arxiv.org/abs/2503.04148v1)
  > **TL;DR**: Addresses sustainable edge inference for multi-tenant DNNs, proposing a framework that dynamically adjusts system power threshold and employs mixed-quality models with an estimator. Achieves 30% lower carbon emissions and 25% reduced carbon delay product versus state-of-the-art.
* `edge` `networking` `scaling` [Benchmarking Dynamic SLO Compliance in Distributed Computing Continuum Systems](http://arxiv.org/abs/2503.03274v1)
  > **TL;DR**: Benchmarks Active Inference against RL algorithms for dynamic SLO compliance in edge-based video streaming applications. Continuously adjusts streaming parameters (e.g., resolution, frame rate) under shifting workloads and constraints. Active Inference reduces memory usage by 20% and stabilizes CPU while ensuring SLOs.
* `MoE` `training` `offline` [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](http://arxiv.org/abs/2502.19811v3)
  > **TL;DR**: Addresses communication overhead in distributed training of MoE models. Proposes COMET with fine-grained computation-communication overlapping via dependency analysis and adaptive workload assignment. Achieves 1.96× per-layer and 1.71× end-to-end speedup, saving millions of GPU hours.
* `training` `scaling` [Memory and Bandwidth are All You Need for Fully Sharded Data Parallel](http://arxiv.org/abs/2504.03655v1)
  > **TL;DR**: Investigates how hardware constraints limit training efficiency for large transformers under FSDP. Analyzes computational, memory, and network demands via simulations and tests to find hardware-optimal configurations. Shows interconnection bandwidth and GPU memory thresholds significantly influence training throughput scalability.
* `sparse` `kernel` [NM-SpMM: Accelerating Matrix Multiplication Using N:M Sparsity with GPGPU](http://arxiv.org/abs/2503.01253v2)
  > **TL;DR**: Proposes NM-SpMM, an N:M sparse matrix multiplication kernel for GPUs, using hierarchical blocking, memory access optimization, and pipelining. Achieves 2.1x speedup over prior work and up to 6.3x over dense cuBLAS GEMM, approaching theoretical peak for sparsity levels.
* `edge` `offline` `kernel` [PointSplit: Towards On-device 3D Object Detection with Heterogeneous Low-power Accelerators](http://arxiv.org/abs/2504.03654v1)
  > **TL;DR**: Proposes PointSplit, a 3D object detection framework for heterogeneous edge accelerators (GPU/NPU). Key designs include semantics-aware point sampling, parallel feature extraction, and role-based quantization. Achieves 24.7× speedup versus GPU-only implementation while maintaining accuracy on RGB-D datasets.
* `hardware` `edge` `quantization` [PVU: Design and Implementation of a Posit Vector Arithmetic Unit (PVU) for Enhanced Floating-Point Computing in Edge and AI Applications](http://arxiv.org/abs/2503.01313v1)
  > **TL;DR**: Proposes a Posit Vector Arithmetic Unit (PVU) for efficient floating-point computation in edge/AI. Implements vector operations in Chisel with RISC-V extension, targeting reduced hardware consumption. Achieves 100% accuracy in most operations with only 65,407 LUTs.
* `hardware` `edge` `scaling` [SAF: Scalable Acceleration Framework for dynamic and flexible scaling of FPGAs](http://arxiv.org/abs/2503.00974v1)
  > **TL;DR**: Proposes SAF, an Ethernet-based framework enabling flexible scaling and simultaneous reconfiguration of FPGAs for acceleration. Features hot-plug capability and custom protocols, reducing reconfiguration time by 13X, hardware costs by 38%, and energy consumption by 27% in edge/cloud environments.
* `training` `RL` `LoRA` [HLoRA: Efficient Federated Learning System for LLM Heterogeneous Fine-Tuning](http://arxiv.org/abs/2503.00813v1)
  > **TL;DR**: Proposes HLoRA for efficient federated fine-tuning of LLMs with heterogeneous client resources. Uses modified heterogeneous LoRA to address bias in parameter aggregation. Improves convergence speed by 1.7× and reduces communication overhead by 1.5× compared to vanilla LoRA.
* `serving` `scaling` `offloading` [Echo: Efficient Co-Scheduling of Hybrid Online-Offline Tasks for Large Language Model Serving](http://arxiv.org/abs/2504.03651v1)
  > **TL;DR**: Addresses resource underutilization in hybrid LLM serving by co-scheduling online and offline tasks. Introduces Echo with a scheduler, KV cache manager, and estimators for optimized KV reuse and batch prediction. Achieves 3.3× higher offline throughput while meeting online SLOs.
* `kernel` `edge` `training` [WgPy: GPU-accelerated NumPy-like array library for web browsers](http://arxiv.org/abs/2503.00279v1)
  > **TL;DR**: Addresses slow deep learning in web browsers by introducing WgPy, a GPU-accelerated NumPy-like array library using WebGL/WebGPU for custom kernels and JavaScript-Python synchronization. Achieves 95x speedup in CNN training compared to CPU execution.
* `scaling` `serving` [AARC: Automated Affinity-aware Resource Configuration for Serverless Workflows](http://arxiv.org/abs/2502.20846v1)
  > **TL;DR**: Proposes AARC, an automated framework for serverless workflows that decouples CPU/memory via critical path identification and priority scheduling. Reduces total search time by up to 89.6% and costs by up to 61.7%, while ensuring SLO compliance.
* `kernel` `hardware` `training` [Methodology for GPU Frequency Switching Latency Measurement](http://arxiv.org/abs/2502.20075v1)
  > **TL;DR**: Proposes a methodology to measure GPU frequency switching latency for energy optimization in HPC and AI systems. Uses an artificial iterative workload and statistical analysis across different GPUs. Reveals significant latency variations critical for dynamic tuning.
* `training` `edge` `offloading` [RingAda: Pipelining Large Model Fine-Tuning on Edge Devices with Scheduled Layer Unfreezing](http://arxiv.org/abs/2502.19864v1)
  > **TL;DR**: Proposes RingAda for efficient on-edge fine-tuning of large transformer models via pipeline parallelism and scheduled layer unfreezing. Utilizes ring topology with adapters on devices, enabling early backpropagation termination. Reduces fine-tuning time by up to 2.4× and memory usage by 25% compared to baselines.
* `hardware` `kernel` [Can Tensor Cores Benefit Memory-Bound Kernels? (No!)](http://arxiv.org/abs/2502.16851v2)
  > **TL;DR**: Theoretical and empirical analysis challenges the claim that tensor cores benefit memory-bound kernels, showing maximum speedup of 1.33x over CUDA cores is unsound for STREAM Scale, SpMV, and stencil on V100, A100, and H100 GPUs.
* `serving` `hardware` `kernel` [PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System](http://arxiv.org/abs/2502.15470v2)
  > **TL;DR**: Addresses suboptimal static mapping of dynamic LLM decoding kernels. Proposes PAPI, a PIM-enabled architecture with online kernel characterization and heterogeneous hardware orchestration. Achieves up to 11.1× speedup over state-of-the-art PIM-only accelerators.
* `training` `scaling` `hardware` [Static task mapping for heterogeneous systems based on series-parallel decompositions](http://arxiv.org/abs/2502.19745v1)
  > **TL;DR**: Addresses task mapping for heterogeneous systems (CPUs, GPUs, FPGAs etc.) to optimize performance. Proposes a decomposition-based algorithm using series-parallel DAGs and model-based evaluation. Achieves substantially better makespan (up to orders of magnitude faster than baselines) while handling complex dependencies.
* `training` `MoE` [HDEE: Heterogeneous Domain Expert Ensemble](http://arxiv.org/abs/2502.19385v1)
  > **TL;DR**: Proposes HDEE, a heterogeneous mixture-of-experts approach allowing variable expert sizes/training steps per domain. Trains ensembles with domain-adapted experts under fixed compute budget. Reduces perplexity in 20/21 domains compared to homogeneous baselines.
* `RAG` `multi-modal` `edge` [Efficient Federated Search for Retrieval-Augmented Generation](http://arxiv.org/abs/2502.19280v1)
  > **TL;DR**: Proposes RAGRoute, a federated retrieval mechanism for RAG systems using a lightweight classifier to dynamically select data sources, reducing queries by up to 77.5% and communication volume by 76.2%.
* `edge` `hardware` [A Reliable, Time-Predictable Heterogeneous SoC for AI-Enhanced Mixed-Criticality Edge Applications](http://arxiv.org/abs/2502.18953v1)
  > **TL;DR**: Proposes a heterogeneous SoC with configurable hardware IPs for reliable, time-predictable AI-enhanced workloads on edge devices. Integrates accelerators for mixed-precision AI and floating-point workloads, achieving 304.9 GOPS at 1.6 TOPS/W within a 1.2W power envelope.
* `training` `kernel` [Systems and Algorithms for Convolutional Multi-Hybrid Language Models at Scale](http://arxiv.org/abs/2503.01868v1)
  > **TL;DR**: Proposes convolutional multi-hybrid architectures optimized for token manipulation tasks. Designs hardware-aware algorithms with overlap-add blocked kernels and context parallelism. Trains 1.2-2.9x faster than Transformers and achieves 2x throughput improvement on GPUs.
* `training` `networking` `scaling` [ZCCL: Significantly Improving Collective Communication With Error-Bounded Lossy Compression](http://arxiv.org/abs/2502.18554v1)
  > **TL;DR**: Proposes ZCCL, an error-bounded lossy compression framework for MPI collectives to reduce communication costs in distributed training. Integrates customized fZ-light compressor and optimizes collective operations. Achieves 1.9–8.9x speedup over original MPI collectives.
* `kernel` `training` `inference` [Kitsune: Enabling Dataflow Execution on GPUs](http://arxiv.org/abs/2502.18403v1)
  > **TL;DR**: Proposes Kitsune, a set of primitives and compiler for dataflow execution on GPUs to overcome limitations of bulk-synchronous execution. Achieves 1.3x-2.4x performance improvement and 16%-98% off-chip traffic reduction for both training and inference.
* `recommendation` `offline` `edge` [Collaboration of Large Language Models and Small Recommendation Models for Device-Cloud Recommendation](http://arxiv.org/abs/2501.05647v2)
  > **TL;DR**: Proposes LSC4Rec, a device-cloud collaborative framework combining large language models (LLMs) and small recommendation models for efficient real-time recommendation. Uses candidate list generation, dynamic request strategies, and edge deployment to reduce inference costs. Achieves up to 38% higher effectiveness than baselines with minimal resource overhead.
* `training` `disaggregation` `scaling` [Armada: Memory-Efficient Distributed Training of Large-Scale Graph Neural Networks](http://arxiv.org/abs/2502.17846v1)
  > **TL;DR**: Addresses memory inefficiency in distributed training for billion-scale graph neural networks. Introduces Armada with GREM partitioning for minimal edge cuts and disaggregated architecture for resource optimization. Achieves 8-65x less memory, 8-46x faster partitioning, and up to 4.5x runtime improvement.
* `training` `offloading` `storage` [CRIUgpu: Transparent Checkpointing of GPU-Accelerated Workloads](http://arxiv.org/abs/2502.16631v1)
  > **TL;DR**: Addresses checkpointing overhead in GPU-accelerated training workloads. Proposes CRIUgpu, leveraging driver capabilities for transparent GPU snapshots without steady-state overhead. Reduces recovery time with zero performance penalty during execution compared to API interception methods.
* `serving` `LoRA` `offloading` [AIBrix: Towards Scalable, Cost-Effective Large Language Model Inference Infrastructure](http://arxiv.org/abs/2504.03648v1)
  > **TL;DR**: Proposes AIBrix, a co-designed cloud-native framework for scalable LLM inference. Key components include dynamic LoRA management, distributed KV cache, SLO-driven GPU optimizer, and hybrid orchestration. Achieves 50% higher throughput and 70% lower latency via token reuse.
* `serving` `edge` `scaling` [Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models](http://arxiv.org/abs/2502.15964v1)
  > **TL;DR**: Proposes MinionS, a cost-efficient local-remote LM collaboration system where the cloud LM decomposes tasks into subtasks executed by on-device LMs. Achieves 5.7x cloud cost reduction while recovering 97.9% performance vs remote-only inference.
* `serving` `sparse` [Learning to Keep a Promise: Scaling Language Model Decoding Parallelism with Learned Asynchronous Decoding](http://arxiv.org/abs/2502.11517v2)
  > **TL;DR**: Proposes PASTA, a learning-based system to enable asynchronous parallel decoding for LLMs via semantic independence annotations. Trains LLMs to express parallelizable chunks using PASTA-LANG language, interpreted for on-the-fly parallel decoding. Achieves 1.21x-1.93x decoding speedup with minor quality changes on AlpacaEval.
* `hardware` `networking` [Next-Gen Computing Systems with Compute Express Link: a Comprehensive Survey](http://arxiv.org/abs/2412.20249v2)
  > **TL;DR**: Explores Compute Express Link (CXL) to overcome interconnect bottlenecks in computing systems. Proposes memory expansion and unified memory for single-machine systems, and disaggregated architectures for distributed systems using CXL's low-latency coherence. Achieves scalable interconnection with significant latency reduction and enhanced resource sharing.
* `serving` `scaling` `offloading` [It Takes Two to Tango: Serverless Workflow Serving via Bilaterally Engaged Resource Adaptation](http://arxiv.org/abs/2502.14320v1)
  > **TL;DR**: Addresses resource inefficiency in serverless workflows due to worst-case function sizing. Proposes Janus, a framework enabling dynamic resource adaptation via developer-provided hints and runtime adjustments. Achieves up to 34.7% higher resource efficiency.
* `edge` `quantization` `offloading` [GenAI at the Edge: Comprehensive Survey on Empowering Edge Devices](http://arxiv.org/abs/2502.15816v1)
  > **TL;DR**: Surveys techniques for deploying generative AI models on edge devices. Covers software/hardware optimizations and frameworks to address constraints like model size and computation. Discusses quantization, pruning, offloading, and specialized hardware achieving over 10× speedup/footprint reduction.
* `training` `scaling` `MoE` [Astra: Efficient and Money-saving Automatic Parallel Strategies Search on Heterogeneous GPUs](http://arxiv.org/abs/2502.13480v1)
  > **TL;DR**: Proposes Astra, an automatic parallel strategy search framework for transformer models on heterogeneous GPUs. Uses multi-objective optimization to find efficiency-optimal and cost-saving configurations in GPU types and parallel parameters. Achieves better throughput than expert strategies with search times under 1.35 minutes at 95% accuracy.
* `serving` `offline` `scaling` [Connecting Large Language Model Agent to High Performance Computing Resource](http://arxiv.org/abs/2502.12280v1)
  > **TL;DR**: Addresses how to connect LLM agents to HPC resources for parallel execution of scientific tasks. Integrates Parsl with LangChain/LangGraph to enable concurrent tool execution on HPC systems. Demonstrates efficient resource utilization for molecular dynamics simulations on Polaris/ALCF.
* `kernel` `hardware` [Comparison of Vectorization Capabilities of Different Compilers for X86 and ARM CPUs](http://arxiv.org/abs/2502.11906v1)
  > **TL;DR**: Evaluates compiler auto-vectorization performance on x86 and ARM CPUs using the TSVC2 suite. Analyzes loop vectorization rates and speedups across GCC, ICX, Clang, and ACFL compilers. Measures 46-56% vectorization rates with mixed performance impact.
* `edge` `offloading` [InTec: integrated things-edge computing: a framework for distributing machine learning pipelines in edge AI systems](http://arxiv.org/abs/2502.11644v1)
  > **TL;DR**: Proposes InTec, a framework for strategically distributing machine learning tasks across Things, Edge, and Cloud layers to reduce latency and optimize resources. Achieves 81.56% reduction in response time and 21.86% reduction in edge energy consumption.
* `recommendation` `serving` `hardware` [GPU-accelerated Multi-relational Parallel Graph Retrieval for Web-scale Recommendations](http://arxiv.org/abs/2502.11490v1)
  > **TL;DR**: Proposes a GPU-accelerated multi-relational parallel graph retrieval framework for web-scale recommendations. Integrates multi-relational metric learning with hierarchical parallel graph-based ANNS to optimize retrieval efficiency. Achieves throughput of over 100 million requests per second while improving accuracy.
* `kernel` `training` `edge` [Gensor: A Graph-based Construction Tensor Compilation Method for Deep Learning](http://arxiv.org/abs/2502.11407v1)
  > **TL;DR**: Proposes Gensor, a graph-based tensor compilation method that models construction space exploration as graph traversal with Markov analysis for efficient kernel generation. Achieves average performance improvements of 18% (up to 30%) and reduces optimization time to seconds.
* `edge` `serving` `hardware` [JExplore: Design Space Exploration Tool for Nvidia Jetson Boards](http://arxiv.org/abs/2502.15773v1)
  > **TL;DR**: Addresses the need for optimal hardware configurations on Nvidia Jetson boards for AI workloads at the edge. Introduces JExplore, a design space exploration tool that automates hardware tuning and integrates with search algorithms. Accelerates configuration search for edge deployments, improving efficiency.
* `training` `offline` `kernel` [Image Pre-Processing Framework for Time-Domain Astronomy in the Artificial Intelligence Era](http://arxiv.org/abs/2502.10783v1)
  > **TL;DR**: Addresses slow image pre-processing for AI training in astronomy. Proposes a GPU-accelerated framework with Eager and Pipeline modes for real-time tuning and batch processing. Achieves significantly faster pre-processing speed while matching CPU accuracy.
* `edge` `offloading` `serving` [Janus: Collaborative Vision Transformer Under Dynamic Network Environment](http://arxiv.org/abs/2502.10047v1)
  > **TL;DR**: Proposes Janus for low-latency cloud-device collaborative inference of Vision Transformers. Combines token pruning with dynamic model splitting and pruning policies to balance accuracy and latency. Reduces latency violation ratios by up to 98.7% and increases throughput by 5.15× under dynamic networks.
* `quantization` `edge` `offloading` [EmbBERT-Q: Breaking Memory Barriers in Embedded NLP](http://arxiv.org/abs/2502.10001v1)
  > **TL;DR**: Proposes EmbBERT-Q, a quantized tiny language model for memory-constrained embedded devices. Combines architectural innovations and 8-bit quantization to reduce memory footprint to 781 kB (25x smaller than SotA) while maintaining competitive NLP accuracy on TinyNLP and GLUE benchmarks.
* `edge` `offline` [SmartEdge: Smart Healthcare End-to-End Integrated Edge and Cloud Computing System for Diabetes Prediction Enabled by Ensemble Machine Learning](http://arxiv.org/abs/2502.15762v1)
  > **TL;DR**: Proposes SmartEdge, an edge-cloud system for low-latency diabetes prediction using ensemble ML. Deploys models across edge nodes and cloud servers to process IoMT data. Achieves 15% lower latency at edge versus cloud and 5% higher accuracy than single-model prediction.
* `kernel` `scaling` `serving` [Measuring GPU utilization one level deeper](http://arxiv.org/abs/2501.16909v2)
  > **TL;DR**: Proposes profiling GPU kernel resource interference to enable predictable colocation for AI applications. Develops methodology to measure contention on schedulers, caches, and memory bandwidth. Aims to build GPU schedulers improving utilization with performance guarantees.
* `training` `scaling` `offline` [General Coded Computing: Adversarial Settings](http://arxiv.org/abs/2502.08058v1)
  > **TL;DR**: Proposes a general coded computing scheme for distributed systems to handle adversarial servers during computations. The method enables robustness for diverse computations, achieving optimal adversarial tolerance with error decaying at O(N^{6/5(a-1)}). Validated with DNN inference, tolerating O(N^a) adversarial nodes.
* `serving` `scaling` [Data-aware Dynamic Execution of Irregular Workloads on Heterogeneous Systems](http://arxiv.org/abs/2502.06304v1)
  > **TL;DR**: Proposes DyPe, a scheduling framework for heterogeneous systems that dynamically analyzes input data characteristics to optimize workload partitioning. Achieves 1.53x throughput improvement and 1.09x energy efficiency over static scheduling.
* `hardware` `training` [Exploring Uncore Frequency Scaling for Heterogeneous Computing](http://arxiv.org/abs/2502.03796v1)
  > **TL;DR**: Proposes MAGUS, a runtime for energy-efficient uncore frequency scaling in heterogeneous CPU-GPU systems. It dynamically detects application phases impacting uncore utilization, predicting memory throughput with vendor power management. Achieves up to 27% energy savings and 26% EDP reduction with <5% performance loss.
* `RAG` `serving` `quantization` [Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation](http://arxiv.org/abs/2502.15734v1)
  > **TL;DR**: Addresses redundant KV computations in RAG systems by reusing chunk-caches with selective recomputation. Proposes Cache-Craft for managing reusable chunk caches via identification, fixing, and storage optimization. Reduces redundant computation by 51%, achieving 1.6× throughput improvement and 2× latency reduction.
* `training` `MoE` `scaling` [Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism](http://arxiv.org/abs/2502.02581v1)
  > **TL;DR**: Addresses straggler effects and load imbalance in Mixture-of-Experts (MoE) model training. Proposes Fully Sharded Sparse Data Parallelism (FSSDP) with sparse collectives and materialization techniques. Achieves up to 3.54× speedup over state-of-the-art MoE training systems.
* `scaling` `training` [WaterWise: Co-optimizing Carbon- and Water-Footprint Toward Environmentally Sustainable Cloud Computing](http://arxiv.org/abs/2501.17944v2)
  > **TL;DR**: Addresses conflict between carbon and water sustainability in cloud computing. Proposes WaterWise, a scheduler for parallel workloads that co-optimizes carbon and water footprints in geo-distributed data centers. Achieves reduced environmental impact with explicit trade-offs evaluation.
* `training` [Scaling Large Language Model Training on Frontier with Low-Bandwidth Partitioning](http://arxiv.org/abs/2501.04266v2)
  > **TL;DR**: Addresses high communication overhead in large-scale LLM training on supercomputers. Proposes a 3-level hierarchical partitioning strategy tailored to the Frontier cluster's bandwidth hierarchy. Achieves 1.71x higher TFLOPS per GPU vs ZeRO++ and 0.94 scaling efficiency at 384 GCDs.
* `training` `hardware` [Ilargi: a GPU Compatible Factorized ML Model Training Framework](http://arxiv.org/abs/2502.01985v1)
  > **TL;DR**: Proposes Ilargi, a factorized ML training framework that uses matrix-represented metadata for automatic factorization on CPU/GPU, avoiding materialization costs. Features an ML cost estimator to choose between factorization and materialization. Achieves up to 8.9x GPU speedups and >20% batch training acceleration.
* `scaling` `hardware` [A Multi-Objective Framework for Optimizing GPU-Enabled VM Placement in Cloud Data Centers with Multi-Instance GPU Technology](http://arxiv.org/abs/2502.01909v1)
  > **TL;DR**: Optimizes GPU resource utilization for LLM serving in cloud data centers. Proposes GRMU, a multi-objective framework with migration-based defragmentation and quota baskets for workload partitioning. Increases acceptance by 22% and reduces hardware by 17%.
* `quantization` `offline` `storage` [SQUASH: Serverless and Distributed Quantization-based Attributed Vector Similarity Search](http://arxiv.org/abs/2502.01528v1)
  > **TL;DR**: Proposes SQUASH, a serverless and distributed quantization-based system for efficient hybrid vector similarity search. Uses OSQ quantization, segment-based storage, and multi-level search for pruning, eliminating redundant I/O. Achieves significant performance improvements at lower cost compared to state-of-the-art.
* `training` `sparse` [Enhancing Token Filtering Efficiency in Large Language Model Training with Collider](http://arxiv.org/abs/2502.00340v1)
  > **TL;DR**: Investigates token filtering inefficiency in LLM training. Proposes Collider, which filters inconsequential tokens across all layers and transforms sparse GEMM into dense GEMM for workload reduction. Achieves 35.1% backpropagation time reduction and 22.0% end-to-end training time improvement with 40% token filtering.
* `training` `offloading` `storage` [Uncoded Download in Lagrange-Coded Elastic Computing with Straggler Tolerance](http://arxiv.org/abs/2501.16298v2)
  > **TL;DR**: Proposes Lagrange-Coded Storage with Uncoded Download (LCSUD) to address elasticity and stragglers in cloud computing, reducing storage size, encoding complexity, and upload cost for matrix multiplications. Achieves lower storage and costs compared to prior methods.
* `scaling` `serving` `networking` [STaleX: A Spatiotemporal-Aware Adaptive Auto-scaling Framework for Microservices](http://arxiv.org/abs/2501.18734v1)
  > **TL;DR**: Proposes STaleX, a spatiotemporal-aware adaptive auto-scaling framework for microservices. Integrates PID controllers with dynamic weight adjustments based on spatial dependencies and temporal workload variations. Reduces resource usage by 26.9% compared to Kubernetes HPA.
* `edge` `agentic` `scaling` [Quantifying Energy and Cost Benefits of Hybrid Edge Cloud: Analysis of Traditional and Agentic Workloads](http://arxiv.org/abs/2501.14823v2)
  > **TL;DR**: Analyzes workload distribution inefficiencies in cloud systems for traditional and agentic AI workloads. Proposes Hybrid Edge Cloud (HEC) to mitigate bottlenecks via edge offloading. Achieves 75% energy savings and over 80% cost reductions in resource-intensive scenarios.
* `serving` `scaling` [Prompt-Aware Scheduling for Efficient Text-to-Image Inferencing System](http://arxiv.org/abs/2502.06798v1)
  > **TL;DR**: Addresses high-load inference inefficiency in text-to-image models due to prompt sensitivity. Proposes prompt-aware scheduler matching prompts to model instances at varying approximations. Achieves high image quality under fixed budget with reduced model loading overheads.
* `hardware` `offloading` `storage` [SP-IMPact: A Framework for Static Partitioning Interference Mitigation and Performance Analysis](http://arxiv.org/abs/2501.16245v1)
  > **TL;DR**: Addresses temporal interference mitigation in embedded systems with static partitioning hypervisors. Proposes SP-IMPact framework to analyze and optimize cache coloring and memory bandwidth reservation configurations. Reduces performance unpredictability by evaluating shared resource interactions on real hardware.
* `edge` `offloading` `networking` [Reinforcement Learning Controlled Adaptive PSO for Task Offloading in IIoT Edge Computing](http://arxiv.org/abs/2501.15203v1)
  > **TL;DR**: Proposes a hybrid approach combining adaptive PSO with reinforcement learning to optimize task offloading in IIoT edge computing. Achieves improved resource management and service quality with minimal latency.
* `hardware` `kernel` `scaling` [TCDM Burst Access: Breaking the Bandwidth Barrier in Shared-L1 RVV Clusters Beyond 1000 FPUs](http://arxiv.org/abs/2501.14370v1)
  > **TL;DR**: Addresses performance degradation in hierarchical network topologies for SIMD/vector cores accessing multi-banked L1 memory. Proposes TCDM Burst Access architecture with a Burst Manager for parallel word retirement. Achieves up to 77-226% higher bandwidth and 2.76x performance improvement in large clusters.
* `kernel` `training` `serving` [RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection](http://arxiv.org/abs/2501.14336v1)
  > **TL;DR**: Develops a GPU-optimized radix top-k selection algorithm for improved scalability to large k values. Proposes a memory bandwidth and resource optimization framework with adaptive scaling. Achieves up to 4.8x speedup for batch queries on adversarial inputs over prior methods.
* `recommendation` `hardware` `storage` [Multi-Tenant SmartNICs for In-Network Preprocessing of Recommender Systems](http://arxiv.org/abs/2501.12032v2)
  > **TL;DR**: Proposes Piper, a network-attached FPGA accelerator for streaming data preprocessing in recommender systems. Introduces MiniPipe for runtime-reconfigurable multi-pipeline execution. Achieves 39–105× speedup over CPUs and 3–17× over GPUs, improving energy efficiency.
* `kernel` `hardware` [Compiler Support for Speculation in Decoupled Access/Execute Architectures](http://arxiv.org/abs/2501.13553v1)
  > **TL;DR**: Develops compiler support for speculation in decoupled access/execute architectures to maintain decoupling despite control dependencies. Speculates memory requests and poisons mis-speculations without replays. Enables DAE on previously inapplicable codes, improving efficiency for irregular workloads.
* `multi-modal` `diffusion` `networking` [Resource Allocation Driven by Large Models in Future Semantic-Aware Networks](http://arxiv.org/abs/2501.14832v1)
  > **TL;DR**: Proposes a large-model-driven semantic communication network to reduce data transmission. Uses scene graphs and multimodal pre-trained models for efficient transmission, and a diffusion model-based scheme for power allocation. Achieves improved semantic transmission quality with optimized resource utilization.
* `edge` `networking` `LoRA` [SplitLLM: Hierarchical Split Learning for Large Language Model over Wireless Network](http://arxiv.org/abs/2501.13318v1)
  > **TL;DR**: Proposes hierarchical split learning for LLM fine-tuning over wireless networks to reduce communication and memory overhead. Deploys model parts across users, edge servers, and cloud, updating only LoRA adapters. Reduces peak memory usage by 74% compared to benchmarks.
* `RAG` [Chat3GPP: An Open-Source Retrieval-Augmented Generation Framework for 3GPP Documents](http://arxiv.org/abs/2501.13954v1)
  > **TL;DR**: Proposes Chat3GPP, an open-source RAG framework for querying 3GPP telecom documents using chunking, hybrid retrieval, and efficient indexing. Eliminates domain-specific fine-tuning while achieving superior performance on telecom datasets for protocol generation and code automation tasks.
* `scaling` `serving` `training` [It's the People, Not the Placement: Rethinking Allocations in Post-Moore Clouds](http://arxiv.org/abs/2501.11185v1)
  > **TL;DR**: Proposes dynamic resource allocation with continuous multilateral cost re-negotiation to address inefficiencies in heterogeneous neoclouds. Shifts from static allocations to improve resource utilization and reduce costs. Demonstrates significant improvements in resource efficiency and cost reduction.
* `training` `storage` `scaling` [Scalable Machine Learning Training Infrastructure for Online Ads Recommendation and Auction Scoring Modeling at Google](http://arxiv.org/abs/2501.10546v1)
  > **TL;DR**: Addresses training system bottlenecks in large-scale ads recommendation models. Proposes shared input generation, partitioning/pipelining for embeddings, and preemption handling mechanisms. Achieves 116% performance boost and 18% training cost reduction in production.
* `hardware` `inference` `memory optimization` [Managed-Retention Memory: A New Class of Memory for the AI Era](http://arxiv.org/abs/2501.09605v1)
  > **TL;DR**: Proposes Managed-Retention Memory (MRM) as a HBM alternative for AI inference workloads, optimizing for density/read bandwidth over retention/write performance. Achieves 20% higher bandwidth and 15% lower energy/bit versus HBM in simulations.
* `kernel` `training` `offline` [Boosting Performance of Iterative Applications on GPUs: Kernel Batching with CUDA Graphs](http://arxiv.org/abs/2501.09398v1)
  > **TL;DR**: Proposes kernel batching via CUDA Graphs to reduce launch overhead in iterative GPU applications. Uses iteration batch unrolling to consolidate kernels into static graphs, finding optimal batch size. Achieves up to 1.4x speed-up in skeleton app and benchmarks.
* `kernel` `training` `quantization` [Acc-SpMM: Accelerating General-purpose Sparse Matrix-Matrix Multiplication with GPU Tensor Cores](http://arxiv.org/abs/2501.09251v1)
  > **TL;DR**: Accelerates sparse matrix-matrix multiplication (SpMM) for deep learning using GPU Tensor Cores. Proposes Acc-SpMM with data reordering, memory-efficient formats, pipelining, and load balancing. Achieves up to 5.11x speedup over cuSPARSE on RTX 4090.
* `offloading` `RL` `edge` [Split Fine-Tuning for Large Language Models in Wireless Networks](http://arxiv.org/abs/2501.09237v1)
  > **TL;DR**: Proposes Split Fine-Tuning (SFT) to enable efficient LLM fine-tuning on mobile devices by splitting the model between edge server and device, with data compression to reduce communication overhead. Achieves 80.2% lower delay and 93.6% less overhead while satisfying memory and accuracy constraints.
* `training` `serving` `scaling` [DNN-Powered MLOps Pipeline Optimization for Large Language Models: A Framework for Automated Deployment and Resource Management](http://arxiv.org/abs/2501.14802v1)
  > **TL;DR**: Proposes a DNN-based framework to optimize MLOps pipelines for LLM deployment. Uses multi-stream neural architecture and adaptive resource allocation to automate decisions. Achieves 40% higher resource utilization, 35% lower deployment latency, and 30% reduced costs.
* `serving` `scaling` [Hierarchical Autoscaling for Large Language Model Serving with Chiron](http://arxiv.org/abs/2501.08090v1)
  > **TL;DR**: Explores SLO-aware autoscaling for interactive and batch LLM inference. Proposes Chiron, a hierarchical backpressure-based autoscaler using queue metrics to manage instance count and batch sizes. Achieves up to 90% higher SLO attainment and 70% higher GPU efficiency.
* `training` `scaling` `networking` [Communication-Efficient, 2D Parallel Stochastic Gradient Descent for Distributed-Memory Optimization](http://arxiv.org/abs/2501.07526v1)
  > **TL;DR**: Proposes 2D parallel SGD (HybridSGD) to optimize communication efficiency in distributed training. Combines s-step SGD and Federated SGD with Averaging for a performance trade-off, reducing communication bottlenecks. Achieves up to 121× speedup over FedAvg on a supercomputing system.
* `offline` `storage` `hardware` [COMPASS: A Compiler Framework for Resource-Constrained Crossbar-Array Based In-Memory Deep Learning Accelerators](http://arxiv.org/abs/2501.06780v1)
  > **TL;DR**: Introduces COMPASS, a compiler framework for partitioning large DNNs onto resource-constrained PIM accelerators with crossbar arrays. Optimizes layer partitioning to minimize latency and memory accesses by considering data dependencies and core utilization. Achieves 1.78× higher throughput and 1.28× lower EDP versus baselines.
* `serving` `offloading` `multi-modal` [Mell: Memory-Efficient Large Language Model Serving via Multi-GPU KV Cache Management](http://arxiv.org/abs/2501.06709v1)
  > **TL;DR**: Addresses memory inefficiency in LLM serving due to imbalanced KV cache across GPUs. Proposes MELL, a system using adaptive request migration and multi-GPU scheduling for load balancing. Reduces required GPUs by 31% and increases utilization by 43% compared to existing systems.
* `training` `diffusion` `networking` [Decentralized Diffusion Models](http://arxiv.org/abs/2501.05450v2)
  > **TL;DR**: Addresses high networking costs and infrastructure constraints in diffusion model training. Proposes decentralized expert diffusion models trained in isolation and ensembled at inference. Reduces infrastructure costs and achieves FLOP-for-FLOP superior performance compared to standard diffusion models.
* `training` `scaling` `scheduling` [Prediction-Assisted Online Distributed Deep Learning Workload Scheduling in GPU Clusters](http://arxiv.org/abs/2501.05563v1)
  > **TL;DR**: Proposes A-SRPT, a prediction-assisted online scheduler for distributed deep learning training in GPU clusters. Uses random forest to predict job iterations and maps scheduling to a single-machine SRPT problem, minimizing communication overhead. Achieves competitive scheduling efficiency per theoretical proofs and evaluations.
* `validation` `GPU` `trustless` [Validation of GPU Computation in Decentralized, Trustless Networks](http://arxiv.org/abs/2501.05374v1)
  > **TL;DR**: Proposes GPU computation verification in decentralized networks. Develops probabilistic frameworks using model fingerprinting, semantic similarity, and GPU profiling. Achieves trustless verification without hardware requirements, addressing non-determinism in GPU nodes.
* `edge` `RL` `serving` [Microservice Deployment in Space Computing Power Networks via Robust Reinforcement Learning](http://arxiv.org/abs/2501.06244v1)
  > **TL;DR**: Proposes a robust reinforcement learning-based microservice deployment framework for low-latency remote sensing inference in satellite constellations. Minimizes resource utilization while meeting QoS via decomposed module optimization and handles data uncertainty. Achieves acceptable computational cost with minimized accuracy loss in simulations.
* `offloading` `edge` `networking` [Intelligent Task Offloading: Advanced MEC Task Offloading and Resource Management in 5G Networks](http://arxiv.org/abs/2501.06242v1)
  > **TL;DR**: Proposes an intelligent task offloading framework using Proximal Policy Optimization for MEC in 5G networks. Optimizes communication and computational resource allocation among UEs, reducing URLLC processing time by 4% and mMTC power consumption by 26% compared to baselines.
* `edge` `recommendation` `training` [Forward Once for All: Structural Parameterized Adaptation for Efficient Cloud-coordinated On-device Recommendation](http://arxiv.org/abs/2501.02837v1)
  > **TL;DR**: Proposes Forward-OFA for efficient on-device recommendation by dynamically constructing device-specific networks (structure and parameters) using a structure controller and mapper. Achieves adaptation in one forward pass, improving efficiency by eliminating backpropagation and adapting to varying device capabilities.
* `edge` `RL` `LoRA` [Efficient Deployment of Large Language Models on Resource-constrained Devices](http://arxiv.org/abs/2501.02438v1)
  > **TL;DR**: Proposes FedSpine, a federated learning framework combining PEFT and structured pruning to efficiently deploy LLMs on resource-constrained devices. Uses an adaptive MAB algorithm for heterogeneous pruning ratios and LoRA ranks. Achieves 1.4×-6.9× faster fine-tuning with 0.4%-4.5% higher accuracy.
* `serving` `offline` [SMDP-Based Dynamic Batching for Improving Responsiveness and Energy Efficiency of Batch Services](http://arxiv.org/abs/2501.02181v1)
  > **TL;DR**: Proposes an SMDP-based dynamic batching scheme to balance latency and energy efficiency for batch services. Uses semi-Markov decision process to minimize weighted response time and power consumption, introducing abstract cost to reduce complexity. Reduces time complexity by 98% and space complexity by 63.5%.
* `edge` `disaggregation` `training` [LEO-Split: A Semi-Supervised Split Learning Framework over LEO Satellite Networks](http://arxiv.org/abs/2501.01293v1)
  > **TL;DR**: Addresses training deep learning models on LEO satellites with intermittent connectivity. Proposes LEO-Split, a semi-supervised split learning framework with auxiliary modeling, pseudo-labeling for imbalance, and adaptive activation interpolation. Achieves superior performance over benchmarks in real satellite traces.
* `serving` [Towards Sustainable Large Language Model Serving](http://arxiv.org/abs/2501.01990v1)
  > **TL;DR**: Characterizes operational and embodied carbon emissions for LLM serving across GPU types and models. Develops analytical models for performance, energy, and carbon emissions based on hardware specs and grid regions. Insights enable optimizing sustainable LLM serving by jointly reducing both emission types.
* `edge` `serving` `multi-modal` [Distributed Mixture-of-Agents for Edge Inference with Large Language Models](http://arxiv.org/abs/2412.21200v1)
  > **TL;DR**: Proposes a distributed MoA architecture for collaborative LLM inference on edge devices using gossip algorithms. Analyzes queue stability and validates via AlpacaEval 2.0. Achieves bounded queue sizes while improving response quality over single models.
* `scaling` `training` `kernel` [TokenRing: An Efficient Parallelism Framework for Infinite-Context LLMs via Bidirectional Communication](http://arxiv.org/abs/2412.20501v1)
  > **TL;DR**: Addresses communication bottlenecks in long-sequence LLM parallelism. Proposes TokenRing, a fine-grained framework with bidirectional P2P communication that overlaps computation and data transmission. Achieves improved scalability and throughput compared to existing methods like Ring-Attention.
* `serving` `disaggregation` `offloading` [GreenLLM: Disaggregating Large Language Model Serving on Heterogeneous GPUs for Lower Carbon Emissions](http://arxiv.org/abs/2412.20322v1)
  > **TL;DR**: Addresses high carbon emissions in LLM serving by reusing heterogeneous GPUs. Proposes GreenLLM, an SLO-aware framework disaggregating computations to older GPUs via offloading. Reduces carbon emissions by up to 40.6% while meeting latency SLOs for >90% requests.
* `RL` `edge` `networking` [Federated Unlearning with Gradient Descent and Conflict Mitigation](http://arxiv.org/abs/2412.20200v1)
  > **TL;DR**: Proposes Federated Unlearning with Orthogonal Steepest Descent (FedOSD) to remove client data from FL models while preserving utility. Uses orthogonal gradients to mitigate conflicts during unlearning and maintain unlearning during recovery. Achieves higher unlearning effectiveness and model utility than SOTA methods.
* `LoRA` `training` [Adaptive Parameter-Efficient Federated Fine-Tuning on Heterogeneous Devices](http://arxiv.org/abs/2412.20004v1)
  > **TL;DR**: Addresses resource and heterogeneity challenges in federated fine-tuning. Proposes LEGEND, an adaptive LoRA framework that optimizes LoRA depth and rank distribution via efficient configuration algorithms. Achieves 1.5-2.8x speedup and 42.3% communication savings vs. baselines.
* `inference` `edge` `storage` [Unlocking True Elasticity for the Cloud-Native Era with Dandelion](http://arxiv.org/abs/2505.01603v2)
  > **TL;DR**: Proposes Dandelion, an elastic cloud platform with declarative DAG programming model for cloud-native applications. Uses lightweight sandboxes without guest OS to cold start in hundreds of microseconds. Achieves 96% memory reduction on Azure Functions trace compared to Firecracker.
* `serving` `scaling` [Demystifying Serverless Costs on Public Platforms: Bridging Billing, Architecture, and OS Scheduling](http://arxiv.org/abs/2506.01283v2)
  > **TL;DR**: Analyzes cost inefficiencies in public serverless platforms by characterizing billing models, request architectures, and OS scheduling. Quantifies billing overheads up to 4.35× due to wall-clock allocation, invocation fees, and rounding. Identifies architectural and scheduling factors affecting cost and performance.
* `edge` `serving` [A Task Equalization Allocation Algorithm Incorporating Blocking Estimation and Resource Similarity Analysis for Vehicle Control Real-Time Systems](http://arxiv.org/abs/2509.14086v1)
  > **TL;DR**: Addresses real-time task scheduling inefficiency in vehicle control systems. Proposes the BR-WFD algorithm that estimates blocking time and groups resource-similar tasks. Reduces processor core requirements by 11-28% and improves schedulable ratio by 15-20% under high-load scenarios.
* `edge` `training` `networking` [Ratio1 -- AI meta-OS](http://arxiv.org/abs/2509.12223v1)
  > **TL;DR**: Proposes Ratio1, a decentralized meta-OS for AI pipelines using blockchain to pool heterogeneous edge resources. Key components include dAuth, CSTORE, R1FS, EDIL, Deeploy, and OracleSync for secure execution. Claimed cost-efficiency improvements over centralized systems via Proof-of-Availability and Proof-of-AI consensus.
* `training` `offloading` `kernel` [MaLV-OS: Rethinking the Operating System Architecture for Machine Learning in Virtualized Clouds](http://arxiv.org/abs/2508.03676v1)
  > **TL;DR**: Proposes MaLV-OS, an operating system tailored for ML workloads in virtualized clouds. Features a micro-kernel that enables GPU access in kernel space and offloads system-sensitive model components to the OS. Achieves reduced execution time by optimizing resource allocation and integrating GPU virtualization.
* `edge` `kernel` `serving` [Composable OS Kernel Architectures for Autonomous Intelligence](http://arxiv.org/abs/2508.00604v1)
  > **TL;DR**: Proposes a composable OS kernel architecture for AI systems, integrating AI-oriented LKMs, built-in deep learning inference, and real-time scheduling. Enhances ML workload efficiency by enabling kernel-space processing and floating-point acceleration. Achieves adaptive scheduling for autonomous intelligence applications.
* `edge` `serving` `scaling` [ConsumerBench: Benchmarking Generative AI Applications on End-User Devices](http://arxiv.org/abs/2506.17538v1)
  > **TL;DR**: Proposes ConsumerBench, a framework for benchmarking GenAI application performance on end-user devices under multi-application contention. Evaluates system-level metrics and resource sharing inefficiencies, showing benefits of custom kernels and SLO-aware scheduling for improved latency and SLO attainment.
* `serving` `networking` `edge` [ROS 2 Agnocast: Supporting Unsized Message Types for True Zero-Copy Publish/Subscribe IPC](http://arxiv.org/abs/2506.16882v1)
  > **TL;DR**: Develops Agnocast, a true zero-copy IPC framework for ROS 2 to eliminate serialization overhead in publish/subscribe systems. Enables direct shared-memory communication for unsized message types with minimal code changes. Achieves 16–25% response time improvement in autonomous driving point cloud processing.
* `serving` `offloading` `thinking` [OSWorld-Human: Benchmarking the Efficiency of Computer-Use Agents](http://arxiv.org/abs/2506.16042v1)
  > **TL;DR**: Studies inefficiency in computer-use AI agents, identifying model call latency and step progression delays as key issues. Proposes OSWorld-Human benchmark with human-annotated trajectories for efficiency evaluation. Agents take 1.4-2.7x more steps than necessary, highlighting optimization needs.
* `training` `hardware` `kernel` [PerfTracker: Online Performance Troubleshooting for Large-scale Model Training in Production](http://arxiv.org/abs/2506.08528v3)
  > **TL;DR**: Addresses the challenge of diagnosing performance issues in large-scale model training systems. Proposes PerfTracker, an online troubleshooting system using fine-grained profiling and differential observability. Deployed on clusters up to 10,000 GPUs, diagnosing diverse performance issues with minimal impact.
* `networking` `scaling` [Söze: One Network Telemetry Is All You Need for Per-flow Weighted Bandwidth Allocation at Scale](http://arxiv.org/abs/2506.00834v2)
  > **TL;DR**: Addresses agile weighted bandwidth allocation in large-scale data centers. Proposes a decentralized system using simple network telemetry to compute and enforce flows without per-flow state. Reduces TPC-H job completion time by up to 0.59×.
* `storage` `offloading` `kernel` [Adaptive Migration Decision for Multi-Tenant Memory Systems](http://arxiv.org/abs/2505.09164v1)
  > **TL;DR**: Proposes a migration control framework for multi-tenant tiered memory systems. Introduces detection of migration friendliness using per-page ping-pong status and low-cost pattern change detection. Achieves efficient memory management in single and multi-tenant setups with improved performance.
* `edge` `serving` [Work-in-Progress: Multi-Deadline DAG Scheduling Model for Autonomous Driving Systems](http://arxiv.org/abs/2505.06780v2)
  > **TL;DR**: Proposes a multi-deadline DAG scheduling model for autonomous driving systems to decompose end-to-end latency constraints into local deadlines. Extends Global EDF for sub-DAGs to ensure data freshness without complex flow analysis. Evaluation shows improved timing guarantees for Autoware workloads.
* `hardware` `kernel` `serving` [CaMDN: Enhancing Cache Efficiency for Multi-tenant DNNs on Integrated NPUs](http://arxiv.org/abs/2505.06625v1)
  > **TL;DR**: Proposes CaMDN, an architecture-scheduling co-design to enhance cache efficiency for multi-tenant DNNs on NPUs. Introduces model-exclusive cache regions and a cache-aware scheduler. Reduces memory access by 33.4% and achieves up to 2.56× model speedup.
* `kernel` [Safe and usable kernel extensions with Rex](http://arxiv.org/abs/2502.18832v2)
  > **TL;DR**: Addresses usability issues of kernel extensions via static verifiers like eBPF. Proposes Rex, a framework using safe Rust and lightweight runtime for safety properties without separate verification. Achieves comparable performance to eBPF.
* `agentic` `multi-modal` `RL` [UFO2: The Desktop AgentOS](http://arxiv.org/abs/2504.14603v2)
  > **TL;DR**: Presents UFO2, an AgentOS integrating LLM agents with desktop automation. Features OS-level integration, hybrid GUI parsing, and multi-action planning to reduce overhead. Achieves higher robustness and execution accuracy across 20+ applications over prior methods.
* `networking` `serving` [The NIC should be part of the OS](http://arxiv.org/abs/2501.10138v2)
  > **TL;DR**: Examines the split between OS and NIC for network performance. Proposes integrating NICs with OS kernels via cache-coherent interconnects to optimize RPC response latency. Achieves lower latency than kernel-bypass approaches while maintaining flexibility.
* `serving` `kernel` `training` [LithOS: An Operating System for Efficient Machine Learning on GPUs](http://arxiv.org/abs/2504.15465v1)
  > **TL;DR**: Proposes LithOS, an operating system for GPUs to efficiently manage resources during ML inference and training. Introduces TPC scheduler, kernel atomization, hardware right-sizing, and power management for improved utilization. Reduces inference tail latencies by 13x vs MPS and improves throughput by 1.6x vs SotA while saving GPU capacity and energy.
* `storage` `disaggregation` `networking` [My CXL Pool Obviates Your PCIe Switch](http://arxiv.org/abs/2503.23611v3)
  > **TL;DR**: Proposes software-based PCIe device pooling using CXL memory pools to replace hardware PCIe switches. Enables PCIe devices to use CXL pools as I/O buffers without hardware changes, improving flexibility and deployment. Achieves cost reduction and higher resource utilization compared to hardware switches.
* `kernel` `storage` `quantization` [Futureproof Static Memory Planning](http://arxiv.org/abs/2504.04874v1)
  > **TL;DR**: Addresses memory inefficiency in dynamic storage allocation for static deep neural networks. Proposes idealloc, a low-fragmentation implementation optimized for million-buffer instances. Outperforms production implementations in effectiveness/robustness on hard benchmarks.
* `serving` `offloading` [Efficient Function-as-a-Service for Large Language Models with TIDAL](http://arxiv.org/abs/2503.06421v1)
  > **TL;DR**: Addresses GPU cold start in FaaS serving for LLMs. Proposes TIDAL, which traces execution paths to generate adaptive function templates for efficient state preloading. Reduces cold start latency by 1.79x~2.11x and improves 95%-ile TTFT by 76.0%.
* `offloading` `edge` `serving` [FlexInfer: Breaking Memory Constraint via Flexible and Efficient Offloading for On-Device LLM Inference](http://arxiv.org/abs/2503.03777v1)
  > **TL;DR**: Addresses high memory demands in on-device LLM inference. Proposes FlexInfer with asynchronous prefetching, balanced memory locking, and flexible tensor preservation to enhance efficiency. Achieves 12.5× higher throughput under resource constraints compared to existing methods.
* `training` `edge` `quantization` [Accelerated Training on Low-Power Edge Devices](http://arxiv.org/abs/2502.18323v1)
  > **TL;DR**: Accelerates training on power-constrained edge devices by jointly optimizing GPU frequency and batch size. Uses a cross-layer approach combining batch size efficiency prediction and device profiling. Reduces training time by 2.4× with close to optimal energy efficiency.
* `serving` `offloading` `networking` [Taming and Controlling Performance and Energy Trade-offs Automatically in Network Applications](http://arxiv.org/abs/2502.14987v1)
  > **TL;DR**: Proposes a Bayesian optimizer for network applications to reduce energy consumption while meeting latency SLAs. Dynamically controls packet batching and processing rate (DVFS) without app changes. Achieves up to 60% energy savings across diverse hardware systems.
* `training` `kernel` `hardware` [Phoenix -- A Novel Technique for Performance-Aware Orchestration of Thread and Page Table Placement in NUMA Systems](http://arxiv.org/abs/2502.10923v2)
  > **TL;DR**: Proposes Phoenix, a kernel-level technique for coordinated thread and page table placement in NUMA systems. Integrates CPU scheduler and memory manager with on-demand page table replication and bandwidth management. Reduces CPU cycles by 2.09x and page-walk cycles by 1.58x compared to state-of-the-art.
* `networking` `serving` [Fast Userspace Networking for the Rest of Us](http://arxiv.org/abs/2502.09281v1)
  > **TL;DR**: Proposes Machnet, a userspace network stack using a 'Least Common Denominator' model for public cloud VMs. Optimized for generic NIC features without flow steering or RSS, with microkernel design using IPC for performance. Achieves comparable performance to state-of-the-art stacks despite minimal vNIC features.
* `training` `diagnostics` `scaling` [XPUTimer: Anomaly Diagnostics for Divergent LLM Training in GPU Clusters of Thousand-Plus Scale](http://arxiv.org/abs/2502.05413v1)
  > **TL;DR**: Addresses anomaly diagnosis in large-scale LLM training clusters. Proposes XPUTimer, a real-time framework using lightweight tracing and intra-kernel diagnostics for GPU clusters. Achieved significant improvements across training stack on 6000 GPUs.
* `serving` `RAG` `quantization` [Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation](http://arxiv.org/abs/2502.15734v1)
  > **TL;DR**: Proposes Cache-Craft, a system for reusing precomputed KV caches of text chunks in RAG-based LLM inference to reduce redundant computations. It identifies reusable chunks, performs partial recomputation to maintain quality, and optimizes storage and eviction. Achieves 1.6× throughput increase and 51% less computation.
* `serving` `networking` `offloading` [GPUs, CPUs, and... NICs: Rethinking the Network's Role in Serving Complex AI Pipelines](http://arxiv.org/abs/2502.15712v1)
  > **TL;DR**: Investigates reducing resource overheads in complex AI inference pipelines by offloading data processing tasks to SmartNICs. Proposes leveraging network hardware packet processing pipelines for computational tasks. Explores integration to mitigate network delays and optimize serving efficiency.

### 2025-12-25
* `training` `scaling` `networking` [Mesh-Attention: A New Communication-Efficient Distributed Attention with Improved Data Locality](http://arxiv.org/abs/2512.20968v1)
  > **TL;DR**: Proposes Mesh-Attention, a distributed attention algorithm using 2D tile assignment and a greedy scheduler for communication efficiency. Reduces communication volume by up to 85.4% and achieves 3.4x speedup on 256 GPUs compared to Ring-Attention during LLM training.
* `training` `scaling` [Deadline-Aware Online Scheduling for LLM Fine-Tuning with Spot Market Predictions](http://arxiv.org/abs/2512.20967v1)
  > **TL;DR**: Proposes deadline-aware scheduling for LLM fine-tuning using spot instances with predictive models. Designs mixed online algorithms with prediction and policy selection to handle price and availability uncertainties. Achieves up to 54.8% utility improvement over baselines.
* `training` `scaling` `quantization` [Diving into 3D Parallelism with Heterogeneous Spot Instance GPUs: Design and Implications](http://arxiv.org/abs/2512.20953v1)
  > **TL;DR**: Addresses efficient distributed training of LLMs across heterogeneous GPUs. Presents AutoHet, a system for automating optimal 3D parallelism plan selection and recovery upon spot preemption. Achieves up to 1.79× training throughput speedup and 4.38× faster recovery.
* `video` `training` `networking` [AirGS: Real-Time 4D Gaussian Streaming for Free-Viewpoint Video Experiences](http://arxiv.org/abs/2512.20943v1)
  > **TL;DR**: Proposes AirGS, a streaming-optimized 4D Gaussian Splatting framework for free-viewpoint video. Rearchitects training with multi-channel conversion and keyframe identification, and optimizes delivery via adaptive pruning. Reduces transmission size by ~50% and training time by 6x while maintaining PSNR above 30.
* `serving` `RL` `scaling` [RHAPSODY: Execution of Hybrid AI-HPC Workflows at Scale](http://arxiv.org/abs/2512.20795v1)
  > **TL;DR**: Enables concurrent execution of heterogeneous AI-HPC workflows (simulation, training, inference, agentic control). Proposes RHAPSODY, a multi-runtime middleware coordinating existing runtimes. Achieves near-linear scaling for high-throughput inference and minimal overhead with Dragon/vLLM on HPC systems.
* `serving` `offloading` `kernel` [PHOTON: Hierarchical Autoregressive Modeling for Lightspeed and Memory-Efficient Language Generation](http://arxiv.org/abs/2512.20687v1)
  > **TL;DR**: Addresses memory inefficiency and latency in Transformer decoding by proposing PHOTON, a hierarchical model with compressed contextual states. Reduces KV-cache traffic through multi-resolution context access. Achieves up to 1000× higher throughput per unit memory for long-context tasks.

### 2025-12-24
* `serving` `quantization` `diffusion` [Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs](http://arxiv.org/abs/2512.20573v1)
  > **TL;DR**: Proposes FailFast, a diffusion LLM-based speculative decoding framework that dynamically adapts draft length to accelerate autoregressive LLM inference. Achieves up to 4.9× speedup over vanilla decoding with lossless quality.
* `networking` `storage` [WOC: Dual-Path Weighted Object Consensus Made Efficient](http://arxiv.org/abs/2512.20485v1)
  > **TL;DR**: Addresses consensus protocols' inability to handle node heterogeneity and workload independence. Proposes WOC with dual-fast/slow paths using object-specific weighted quorums and leader coordination. Achieves 4× higher throughput than Cabinet for low-contention workloads.
* `LoRA` `serving` [Predictive-LoRA: A Proactive and Fragmentation-Aware Serverless Inference System for LLMs](http://arxiv.org/abs/2512.20210v1)
  > **TL;DR**: Addresses latency and fragmentation in serverless LLM serving with multiple LoRA adapters. Proposes proactive adapter prefetching and page-based memory management. Achieves 35% reduction in average TTFT and 1.52x throughput improvement over prior system.
* `serving` `thinking` `RL` [Reaching Agreement Among Reasoning LLM Agents](http://arxiv.org/abs/2512.20184v1)
  > **TL;DR**: Proposes Aegean, a consensus protocol for multi-agent reasoning to ensure reliable agreement while reducing latency. Implements a serving engine with incremental quorum detection for early termination. Cuts latency by 1.2-20× compared to baselines while maintaining answer quality within 2.5%.
* `sparse` `training` `kernel` [SHIRO: Near-Optimal Communication Strategies for Distributed Sparse Matrix Multiplication](http://arxiv.org/abs/2512.20178v1)
  > **TL;DR**: Addresses high communication overhead in distributed sparse matrix multiplication. Proposes fine-grained sparsity-aware communication and hierarchical strategies leveraging GPU network topologies. Achieves up to 221.5× speedup over baselines at 128-GPU scale.
* `offline` `scaling` `quantization` [FastMPS: Revisit Data Parallel in Large-scale Matrix Product State Sampling](http://arxiv.org/abs/2512.20064v1)
  > **TL;DR**: Addresses high memory and I/O overhead in large-scale Matrix Product State (MPS) sampling. Proposes Fast-MPS, combining data and tensor parallelism with compression and overlapping techniques. Achieves over 10× speedup and scales to thousands of processes, handling 8,176 sites.
* `training` `storage` `networking` [Scaling Point-based Differentiable Rendering for Large-scale Reconstruction](http://arxiv.org/abs/2512.20017v1)
  > **TL;DR**: Addresses distributed training inefficiencies in Point-based Differentiable Rendering (PBDR) for large-scale 3D reconstruction. Introduces Gaian, a system unifying PBDR APIs and optimizing data locality to reduce communication. Achieves up to 91% communication reduction and 1.50x-3.71x throughput improvement.
* `MoE` `training` `networking` [UCCL-EP: Portable Expert-Parallel Communication](http://arxiv.org/abs/2512.19849v1)
  > **TL;DR**: Addresses poor portability of expert-parallel (EP) communication systems across heterogeneous GPU/NIC platforms. Proposes UCCL-EP with GPU-CPU control channel and RDMA emulation for ordering semantics. Achieves up to 2.1× dispatch/combine throughput and 45% training throughput improvement.

### 2025-12-23
* `serving` `RL` `edge` [Q-IRIS: The Evolution of the IRIS Task-Based Runtime to Enable Classical-Quantum Workflows](http://arxiv.org/abs/2512.13931v1)
  > **TL;DR**: Proposes Q-IRIS, a hybrid runtime for classical-quantum workflows integrating IRIS task-based runtime with XACC quantum framework. Enables asynchronous scheduling of QIR programs across heterogeneous backends (including quantum simulators). Demonstrates circuit cutting to reduce per-task simulation load, improving throughput and reducing queueing behavior.
* `hardware` `training` `kernel` [Design in Tiles: Automating GEMM Deployment on Tile-Based Many-PE Accelerators](http://arxiv.org/abs/2512.13638v1)
  > **TL;DR**: Automates GEMM deployment on tile-based many-PE accelerators via DiT framework, connecting deployment toolchain with configurable executable model. Achieves 1.2-2.0x speedup over NVIDIA GH200 with higher PE utilization for diverse matrix shapes.
* `serving` `MoE` `offloading` [Janus: Disaggregating Attention and Experts for Scalable MoE Inference](http://arxiv.org/abs/2512.13525v2)
  > **TL;DR**: Addresses inefficient resource usage in Mixture-of-Experts inference. Proposes Janus, a system that disaggregates attention and experts onto separate GPU sub-clusters with adaptive communication and lightweight scheduling. Achieves up to 3.9× higher per-GPU throughput while meeting latency targets.
* `training` `MoE` `hardware` [SIGMA: An AI-Empowered Training Stack on Early-Life Hardware](http://arxiv.org/abs/2512.13488v1)
  > **TL;DR**: SIGMA addresses reliability, stability, and efficiency challenges in large-scale training on early-life AI accelerators. It introduces the LUCIA Training Platform and Framework, optimizing clusters for high utilization. Achieved 94.45% accelerator utilization and 21.08% MFU when training a 200B MoE model on 2,048 accelerators.
* `training` `offline` `kernel` [Temporal parallelisation of continuous-time maximum-a-posteriori trajectory estimation](http://arxiv.org/abs/2512.13319v1)
  > **TL;DR**: Proposes a parallel-in-time method for continuous-time MAP trajectory estimation to accelerate computations on parallel architectures like GPUs. Uses a reformulation into an optimal control problem and associative scan algorithms for parallelization. Achieves significant speedup in GPU experiments while maintaining sequential algorithm accuracy.
* `kernel` `training` `quantization` [FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection](http://arxiv.org/abs/2512.12949v1)
  > **TL;DR**: Proposes FlashFuser, a compiler framework for kernel fusion using GPU inter-core Distributed Shared Memory. Enables fusion of large memory-intensive operators like FFN via DSM-based communication, optimized scheduling, and tiling. Achieves 3.3x kernel speedup against libraries and 58% memory access reduction.
* `serving` [PROSERVE: Unified Multi-Priority Request Scheduling for LLM Serving](http://arxiv.org/abs/2512.12928v1)
  > **TL;DR**: Addresses scheduling LLM serving requests with mixed priorities for maximizing service gain. Proposes a two-tier scheduler using SlideBatching for adaptive batching/ordering and GoRouting for distributed dispatching. Increases system gain by up to 35% and SLO attainment by up to 52%.
* `serving` `quantization` `offloading` [Fine-Grained Energy Prediction For Parallellized LLM Inference With PIE-P](http://arxiv.org/abs/2512.12801v1)
  > **TL;DR**: Addresses the challenge of accurately predicting energy consumption for multi-GPU parallelized LLM inference. Proposes PIE-P, a framework with scalable prediction via precise sampling and modeling of communication overheads. Achieves significantly higher accuracy than baselines across tensor, pipeline, and data parallelism.
* `RL` `training` [HetRL: Efficient Reinforcement Learning for LLMs in Heterogeneous Environments](http://arxiv.org/abs/2512.12476v1)
  > **TL;DR**: Addresses inefficiency in RL training for large language models across heterogeneous GPU environments. Introduces HetRL, a distributed system with a multi-level scheduling algorithm that optimizes resource allocation. Achieves up to 9.17x higher throughput versus state-of-the-art systems.
* `training` `RL` [Reputation-Based Leader Election under Partial Synchrony: Towards a Protocol-Independent Abstraction with Enhanced Guarantees](http://arxiv.org/abs/2512.12409v2)
  > **TL;DR**: Proposes Sliding Window Leader Election (SWLE), a reputation-based abstraction for leader-based BFT protocols under partial synchrony. Enhances leader election via Byzantine-cost amplification and consensus-behavior reputation. Achieves 4.2x throughput, 75% lower latency vs state-of-the-art in geo-distributed deployment.
* `inference` `offline` `kernel` [Near-Zero-Overhead Freshness for Recommendation Systems via Inference-Side Model Updates](http://arxiv.org/abs/2512.12295v2)
  > **TL;DR**: Proposes LiveUpdate, a system for near-zero-overhead freshness in recommendation systems by colocating LoRA trainers within underutilized inference nodes, using dynamic rank adaptation and NUMA-aware scheduling. Achieves <2% memory overhead and P99 latency impact <20ms, reducing update costs by 2x versus baselines.
* `edge` `offloading` `hardware` [Fast Online Digital Twinning on FPGA for Mission Critical Applications](http://arxiv.org/abs/2512.17942v1)
  > **TL;DR**: Proposes an FPGA-accelerated digital twinning framework for low-latency mission-critical applications. Offloads GRU and dense layers to reconfigurable hardware for parallel execution. Achieves 5x faster operation than human reaction time, enabling real-time edge deployment.
* `edge` `hardware` `quantization` [Accelerated Digital Twin Learning for Edge AI: A Comparison of FPGA and Mobile GPU](http://arxiv.org/abs/2512.17941v1)
  > **TL;DR**: Proposes a digital twin learning framework accelerated on FPGAs for edge AI healthcare applications. Leverages reconfigurable hardware optimizations for compute/memory efficiency. Achieves 8.8x performance-per-watt gain and 28.5x DRAM reduction over cloud GPU baselines.
* `training` `kernel` `sparse` [BOOST: BOttleneck-Optimized Scalable Training Framework for Low-Rank Large Language Models](http://arxiv.org/abs/2512.12131v1)
  > **TL;DR**: Proposes BOOST, a training framework for low-rank bottleneck LLMs. Combines bottleneck-aware tensor parallelism, online-RMSNorm, and low-rank activation checkpointing to optimize GPU utilization and reduce communication. Achieves 1.46-1.91× speedup over full-rank baselines and up to 2.27× over naive low-rank parallelism.
* `kernel` `sparse` `hardware` [Accelerating Sparse Matrix-Matrix Multiplication on GPUs with Processing Near HBMs](http://arxiv.org/abs/2512.12036v1)
  > **TL;DR**: Optimizes SpGEMM on GPUs for irregular memory access via hardware-software co-design with HBM processing. Presents Hash-based Multi-phase approach and AIA technique, achieving up to 4.18x speedup in GNN training over cuSPARSE on large datasets.
* `video` `storage` `networking` [ECCO: Leveraging Cross-Camera Correlations for Efficient Live Video Continuous Learning](http://arxiv.org/abs/2512.11727v1)
  > **TL;DR**: Proposes ECCO, a video analytics system that reduces retraining costs for continuous learning by grouping cameras with correlated data drift and dynamically allocating GPU resources. Achieves 6.7%-18.1% higher retraining accuracy at same resource cost or supports 3.3× more cameras at same accuracy.
* `edge` `kernel` `serving` [Parallax: Runtime Parallelization for Operator Fallbacks in Heterogeneous Edge Systems](http://arxiv.org/abs/2512.11532v1)
  > **TL;DR**: Introduces Parallax, a framework that optimizes runtime parallelization for operator fallbacks on edge devices. Employs DAG partitioning, branch-aware memory management, and adaptive scheduling to accelerate mobile DNN inference. Achieves up to 46% latency reduction and 30% energy savings on mobile devices.
* `training` `RL` `offloading` [RollMux: Phase-Level Multiplexing for Disaggregated RL Post-Training](http://arxiv.org/abs/2512.11306v2)
  > **TL;DR**: Proposes RollMux, a cluster scheduling framework for disaggregated RL post-training that reclaims dependency bubbles via co-execution groups and two-tier scheduling. Achieves 1.84x cost efficiency improvement over standard disaggregation with 100% SLO attainment.
* `kernel` `training` `hardware` [Theoretical Foundations of GPU-Native Compilation for Rapid Code Iteration](http://arxiv.org/abs/2512.11200v1)
  > **TL;DR**: Addresses latency bottlenecks in AI code generation from CPU-GPU data transfers during compilation. Proposes GPU-native compilation methods: parallel traditional, neural sequence translation with verification, and hybrid. Achieves 10-100x speedups via transfer elimination and massive parallelism.
* `serving` `offloading` `sparse` [ESS: An Offload-Centric Latent-Cache Management Architecture for DeepSeek-V3.2-Exp](http://arxiv.org/abs/2512.10576v1)
  > **TL;DR**: Addresses Decode-stage bottleneck in DeepSeek-V3.2-Exp due to Latent-Cache memory constraints. Proposes ESS, an offload-centric system that selectively offloads Latent-Cache to CPU to free GPU memory. Achieves 69.4% to 123% throughput improvement at 32K-128K context lengths.
* `training` `RL` [Hybrid Learning and Optimization-Based Dynamic Scheduling for DL Workloads on Heterogeneous GPU Clusters](http://arxiv.org/abs/2512.10271v1)
  > **TL;DR**: Proposes RLTune, a reinforcement learning-based scheduler for heterogeneous GPU clusters, integrating RL prioritization with MILP-based mapping to optimize job completion time and utilization. Achieves up to 70% lower job completion time and 20% higher GPU utilization without per-job profiling.
* `offline` `edge` `kernel` [Ariel-ML: Computing Parallelization with Embedded Rust for Neural Networks on Heterogeneous Multi-core Microcontrollers](http://arxiv.org/abs/2512.09800v1)
  > **TL;DR**: Aims to automate parallelization of TinyML inference on multi-core microcontrollers. Introduces Ariel-ML, an embedded Rust toolkit for parallel execution on heterogeneous cores. Achieves lower inference latency than prior work while maintaining comparable memory footprint to C/C++ toolkits.
* `training` `scheduling` [Straggler Tolerant and Resilient DL Training on Homogeneous GPUs](http://arxiv.org/abs/2512.09685v1)
  > **TL;DR**: Addresses stragglers in homogeneous GPU training due to resource imbalances. Proposes STAR, with adaptive synchronization modes and resource-aware scheduling. Achieves 48-84% lower Time-To-Accuracy than prior systems while maintaining accuracy.
* `serving` `offloading` [WarmServe: Enabling One-for-Many GPU Prewarming for Multi-LLM Serving](http://arxiv.org/abs/2512.09472v1)
  > **TL;DR**: Proposes WarmServe for multi-LLM serving with universal GPU workers enabling one-for-many prewarming. Uses evict-aware placement, proactive prewarming, and zero-overhead memory switching to reduce prewarming interference. Improves TTFT by up to 50.8× and serves 2.5× more requests.
* `RAG` `storage` `networking` [Passing the Baton: High Throughput Distributed Disk-Based Vector Search with BatANN](http://arxiv.org/abs/2512.09331v1)
  > **TL;DR**: Presents BatANN, a distributed disk-based vector search system for large datasets, optimizing throughput by sending full query state to remote machines. Achieves 2.5-6.49× higher throughput than scatter-gather baseline on 100M-1B vector datasets with mean latency below 6 ms over standard TCP.
* `offline` `kernel` `offloading` [SHARe-KAN: Holographic Vector Quantization for Memory-Bound Inference](http://arxiv.org/abs/2512.15742v1)
  > **TL;DR**: Addresses memory constraints in Kolmogorov-Arnold Networks (KANs) during inference. Proposes SHARe-KAN with Gain-Shape-Bias Vector Quantization and LUTHAM compiler for static memory planning. Achieves 88x runtime memory reduction (1.13GB to 12.91MB) while maintaining accuracy.
* `serving` `offloading` `quantization` [GoodSpeed: Optimizing Fair Goodput with Adaptive Speculative Decoding in Distributed Edge Inference](http://arxiv.org/abs/2512.09963v2)
  > **TL;DR**: Proposes GOODSPEED, an adaptive speculative decoding framework for distributed edge inference. Uses a gradient scheduler to dynamically allocate token verification across heterogeneous draft servers, optimizing fair goodput. Achieves near-optimal goodput with provable fairness and up to 10x lower latency.
* `MoE` `serving` `offloading` [Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens](http://arxiv.org/abs/2512.09277v1)
  > **TL;DR**: Proposes METRO, a token-routing algorithm for MoE serving that balances activated experts instead of tokens to mitigate memory-bound bottlenecks. Achieves 11-22% decode latency reduction and up to 4.11x higher decode throughput while using novel allGather optimization.
* `serving` `training` `quantization` [Magneton: Optimizing Energy Efficiency of ML Systems via Differential Energy Debugging](http://arxiv.org/abs/2512.08365v1)
  > **TL;DR**: Addresses software energy waste in ML systems via differential energy debugging. Magneton compares energy consumption at operator level to pinpoint inefficiencies in code/configurations. Applied to 9 ML systems, found 16 known and 8 new inefficiencies, with 7 confirmed.
* `training` `hardware` `kernel` [Chopper: A Multi-Level GPU Characterization Tool & Derived Insights Into LLM Training Inefficiency](http://arxiv.org/abs/2512.08242v1)
  > **TL;DR**: Proposes Chopper, a multi-level GPU profiling tool, to analyze inefficiencies in distributed LLM training. Identifies frequency overhead (DVFS) as the primary performance gap contributor during Llama 3 training on AMD MI300X GPUs. Achieves detailed characterization leading to 10% potential performance improvement insights.
* `serving` `training` `edge` [Dora: QoE-Aware Hybrid Parallelism for Distributed Edge AI](http://arxiv.org/abs/2512.10990v1)
  > **TL;DR**: Proposes Dora for QoE-aware hybrid parallelism in distributed edge AI training and inference. It uses heterogeneity-aware partitioning, contention-aware scheduling, and runtime adaptation to balance latency and resource efficiency. Achieves 1.1-6.3× faster execution or 21-82% energy reduction while maintaining QoE.
* `training` `networking` `offloading` [Modeling the Potential of Message-Free Communication via CXL.mem](http://arxiv.org/abs/2512.08005v1)
  > **TL;DR**: Models performance benefits of using CXL.mem technology for MPI data exchange. Develops toolchain with memory trace sampling (Mitos) to identify high-potential MPI calls for replacing traditional messages with direct memory pooling. Example: Applied to HPCG benchmark with predicted communication latency reduction.
* `offline` `scheduling` [Quantifying the Carbon Reduction of DAG Workloads: A Job Shop Scheduling Perspective](http://arxiv.org/abs/2512.07799v1)
  > **TL;DR**: Quantifies carbon reduction potential for DAG-based batch workloads (e.g., offline inference) via flexible job-shop scheduling. Formulates dependency-aware scheduling to exploit carbon intensity variations. Achieves up to 25% lower emissions without increasing makespan under homogeneous servers.
* `training` `networking` [Bandwidth-Aware Network Topology Optimization for Decentralized Learning](http://arxiv.org/abs/2512.07536v1)
  > **TL;DR**: Proposes a bandwidth-aware network topology optimization framework for decentralized learning to maximize consensus speed under bandwidth constraints. Uses Mixed-Integer SDP reformulation and ADMM with conjugate gradient to solve large-scale topologies. Achieves up to 1.21× training speedup on heterogeneous bandwidth.
* `video` `training` `networking` [Communication-Efficient Serving for Video Diffusion Models with Latent Parallelism](http://arxiv.org/abs/2512.07350v1)
  > **TL;DR**: Addresses communication bottlenecks in parallel serving of video diffusion models. Proposes Latent Parallelism (LP), a dimension-rotating partition strategy with patch-aligned overlapping. Reduces communication overhead by up to 97% while maintaining generation quality.
* `edge` `RAG` `video` [Venus: An Efficient Edge Memory-and-Retrieval System for VLM-based Online Video Understanding](http://arxiv.org/abs/2512.07344v1)
  > **TL;DR**: Proposes Venus, an edge-cloud system for efficient online video understanding with VLMs. Uses edge-based memory construction, keyframe retrieval, and progressive sampling to reduce overhead. Achieves 15x-131x speedup in latency while maintaining accuracy.
* `hardware` `kernel` `serving` [DCO: Dynamic Cache Orchestration for LLM Accelerators through Predictive Management](http://arxiv.org/abs/2512.07312v1)
  > **TL;DR**: Proposes DCO, a dynamic cache orchestration for LLM accelerators using predictive management. Utilizes dataflow-driven cache policies (bypass and thrashing mitigation) and dead-block prediction. Achieves up to 1.80x speedup compared to conventional cache architectures.
* `video` `serving` `kernel` [Optimizing video analytics inference pipelines: a case study](http://arxiv.org/abs/2512.07009v1)
  > **TL;DR**: Addresses high computational costs in livestock monitoring video analytics. Introduces multi-level parallelization, GPU acceleration of CPU code, vectorized clustering, and memory-efficient post-processing for a 2x pipeline speedup without accuracy loss, enabling high-throughput, low-latency inference.
* `serving` `edge` `kernel` [ELANA: A Simple Energy and Latency Analyzer for LLMs](http://arxiv.org/abs/2512.09946v1)
  > **TL;DR**: Presents ELANA, a lightweight profiler for evaluating latency metrics (TTFT, TPOT, TTLT) and energy consumption of LLMs on multi-GPU and edge platforms. Integrates with Hugging Face models and supports low-precision models. Demonstrates usage for efficient deployment analysis without specific quantitative results mentioned in abstract.
* `storage` `serving` `offline` [A Chunked-Object Pattern for Multi-Region Large Payload Storage in Managed NoSQL Databases](http://arxiv.org/abs/2512.06852v1)
  > **TL;DR**: Addresses challenges of storing and accessing large payloads exceeding NoSQL database item size limits. Proposes a chunked-object pattern that persists large entities as ordered chunks within the database for consistency. Reduces p99 cross-region time-to-consistency for 1 MB payloads by eliminating replication lag.
* `training` `MoE` `edge` [Stable-MoE: Lyapunov-based Token Routing for Distributed Mixture-of-Experts Training over Edge Networks](http://arxiv.org/abs/2512.06784v1)
  > **TL;DR**: Proposes Stable-MoE, a Lyapunov-based token routing framework for distributed MoE training over heterogeneous edge networks. Optimizes token routing and resource allocation via online Lyapunov optimization to maximize throughput and gating consistency. Achieves 40% higher throughput and 5% accuracy gains on SVHN/CIFAR-100.
* `training` `RL` `kernel` [A-3PO: Accelerating Asynchronous LLM Training with Staleness-aware Proximal Policy Approximation](http://arxiv.org/abs/2512.06547v1)
  > **TL;DR**: Addresses computational bottleneck in asynchronous RL training for LLMs by approximating the proximal policy without an extra forward pass. Proposes A-3PO, a staleness-aware proximal policy approximation via interpolation. Reduces training time by 18% while maintaining performance.
* `edge` `quantization` `kernel` [Vec-LUT: Vector Table Lookup for Parallel Ultra-Low-Bit LLM Inference on Edge Devices](http://arxiv.org/abs/2512.06443v1)
  > **TL;DR**: Proposes Vec-LUT, a vector lookup paradigm for ultra-low-bit LLM inference on edge devices. Introduces unified LUT across tokens and cache-aware techniques to reduce memory bandwidth underutilization during parallel inference. Achieves up to 4.2× speedup on CPUs across 5 edge devices.
* `offline` `serving` [Metronome: Differentiated Delay Scheduling for Serverless Functions](http://arxiv.org/abs/2512.05703v1)
  > **TL;DR**: Addresses inefficient scheduling for serverless functions with heterogeneous execution and locality patterns. Designs Metronome, a predictive delay scheduling framework using online Random Forest Regression to forecast execution times and optimize node selection. Reduces mean execution time by 64.88%-95.83% while maintaining SLA.
* `offloading` `quantization` [Compiler-supported reduced precision and AoS-SoA transformations for heterogeneous hardware](http://arxiv.org/abs/2512.05516v1)
  > **TL;DR**: Explores compiler-supported reduced precision and AoS-SoA layouts for particle simulations on GPUs. Proposes compiler annotations to orchestrate CPU-GPU data offloading. Achieves up to 2.6x speedup on Nvidia GPUs through optimized data transformations.
* `serving` `RAG` `agentic` [Model Gateway: Model Management Platform for Model-Driven Drug Discovery](http://arxiv.org/abs/2512.05462v1)
  > **TL;DR**: Proposes Model Gateway, a management platform for ML models in drug discovery, supporting LLM Agents for model management, execution, and results retrieval. Achieves 0% failure rate with over 10k simultaneous clients scaling.
* `offloading` `hardware` `training` [Offloading to CXL-based Computational Memory](http://arxiv.org/abs/2512.04449v1)
  > **TL;DR**: Addresses data movement costs in disaggregated memory systems using CXL. Proposes 'Asynchronous Back-Streaming' protocol and KAI system for optimized offloading. Achieves 50.4% runtime reduction and 22.11x idle time improvement.
* `training` `serving` `offline` [Reducing Fragmentation and Starvation in GPU Clusters through Dynamic Multi-Objective Scheduling](http://arxiv.org/abs/2512.10980v1)
  > **TL;DR**: Aims to improve GPU utilization and reduce fragmentation/starvation in multi-tenant clusters running heterogeneous AI workloads. Proposes three dynamic schedulers (HPS, PBS, SBS) that optimize for utilization, fairness, and throughput. Achieves up to 78.2% GPU utilization (vs 45-67% in baselines) and 25.8 jobs/hour throughput.
* `training` `kernel` `offloading` [VLCs: Managing Parallelism with Virtualized Libraries](http://arxiv.org/abs/2512.04320v1)
  > **TL;DR**: Addresses resource contention in parallel library composition for high-performance computing. Introduces Virtual Library Contexts (VLCs) to encapsulate libraries and manage resources without code modification. Achieves speedup of up to 2.85x in benchmarks with OpenMP, OpenBLAS, and LibTorch.
* `kernel` `training` `inference` [tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection](http://arxiv.org/abs/2512.04226v1)
  > **TL;DR**: Proposes tritonBLAS, an analytical model for generating optimized GEMM kernels without autotuning. Uses architectural parameters to predict near-optimal configurations via deterministic modeling. Achieves over 95% of autotuned performance while eliminating tuning overhead.
* `MoE` `edge` `offloading` [OD-MoE: On-Demand Expert Loading for Cacheless Edge-Distributed MoE Inference](http://arxiv.org/abs/2512.03927v1)
  > **TL;DR**: Addresses MoE inference on memory-constrained edge devices by eliminating expert caches. Proposes OD-MoE with parallelized expert loading/computation and emulative activation prediction. Achieves 99.94% prediction accuracy and 75% decoding speed of GPU-cached deployment with 1/3 GPU memory usage.
* `training` `networking` `storage` [FFTrainer: Fast Failover in Large-Language Model Training with Almost-Free State Management](http://arxiv.org/abs/2512.03644v1)
  > **TL;DR**: Addresses high recovery overhead in large-scale LLM training. Introduces FFTrainer, using surplus network capacity for fast failover and nearly-free state management. Reduces recovery time by up to 98% and GPU utilization loss by up to 68%.
* `kernel` `training` `inference` [Agentic Operator Generation for ML ASICs](http://arxiv.org/abs/2512.10977v1)
  > **TL;DR**: Proposes TritorX, an agentic AI system for generating correct Triton PyTorch ATen kernels for ML accelerators. Uses LLMs with linting, JIT compilation, and PyTorch OpInfo testing for broad coverage. Generates 481 verified operators passing 20,000+ tests.
* `serving` `offloading` `networking` [TokenScale: Timely and Accurate Autoscaling for Disaggregated LLM Serving with Token Velocity](http://arxiv.org/abs/2512.03416v1)
  > **TL;DR**: Proposes TokenScale, an autoscaler for disaggregated LLM serving using Token Velocity (leading indicator of backpressure) and Convertible Decoders (dynamic resource switching). Achieves 80-96% SLO attainment (vs. 50-88% baselines) and reduces costs by 4-14%.
* `serving` `quantization` `offloading` [TokenPowerBench: Benchmarking the Power Consumption of LLM Inference](http://arxiv.org/abs/2512.03024v1)
  > **TL;DR**: Introduces TokenPowerBench, a benchmark for measuring LLM inference power consumption. It features declarative configuration, multi-level power measurement without specialized meters, and phase-aligned metrics. The benchmark quantifies efficiency, e.g., joules per token when varying batch sizes and quantization on models up to Llama3-405B.
* `offloading` `storage` `training` [Offloading Artificial Intelligence Workloads across the Computing Continuum by means of Active Storage Systems](http://arxiv.org/abs/2512.02646v1)
  > **TL;DR**: Proposes active storage for offloading AI workloads across the computing continuum to reduce data transfer. Integrates computation into storage using dataClay to optimize memory, storage, and training times. Achieves significant memory efficiency and training speed improvements while maintaining accuracy.
* `serving` `storage` `offloading` [Fantasy: Efficient Large-scale Vector Search on GPU Clusters with GPUDirect Async](http://arxiv.org/abs/2512.02278v1)
  > **TL;DR**: Addresses how to efficiently perform large-scale vector similarity search on GPU clusters. Introduces Fantasy, which pipelines search and data transfer with GPUDirect Async for overlapping computation and network communication. Achieves significantly improved throughput for large graphs and large query batches.
* `serving` `offloading` [Tangram: Accelerating Serverless LLM Loading through GPU Memory Reuse and Affinity](http://arxiv.org/abs/2512.01357v1)
  > **TL;DR**: Addresses model loading latency in serverless LLMs. Proposes Tangram using GPU memory reuse via unified pool, on-demand KV cache, and affinity-aware scheduling. Achieves 6.2× faster loading and 23–55% lower TTFT.
* `edge` `serving` `offloading` [Joint Partitioning and Placement of Foundation Models for Real-Time Edge AI](http://arxiv.org/abs/2512.01039v1)
  > **TL;DR**: Proposes dynamic partitioning and placement of foundation models for real-time edge inference. Introduces a runtime optimization framework for layer-wise assignments under latency and resource constraints. Achieves adaptive orchestration responsive to infrastructure volatility in 6G edge networks.
* `training` `RL` `MoE` [Elastic Mixture of Rank-Wise Experts for Knowledge Reuse in Federated Fine-Tuning](http://arxiv.org/abs/2512.00902v1)
  > **TL;DR**: Proposes SmartFed for efficient federated fine-tuning of LLMs via Mixture of Rank-Wise Experts (MoRE) and Elastic Expert Quota Allocation (EEQA). Leverages knowledge reuse in LoRA modules to reduce computation. Achieves higher accuracy with 2.3× faster training than baselines.
* `serving` `kernel` [SIMPLE: Disaggregating Sampling from GPU Inference into a Decision Plane for Faster Distributed LLM Serving](http://arxiv.org/abs/2512.00719v1)
  > **TL;DR**: Addresses sampling bottleneck in distributed LLM serving by disaggregating to CPU with sequence-parallel sampling and speculative hot-vocab sampling. Achieves up to 96% throughput increase and 20-65% latency reduction.
* `edge` `serving` `offloading` [IslandRun: Privacy-Aware Multi-Objective Orchestration for Distributed AI Inference](http://arxiv.org/abs/2512.00595v1)
  > **TL;DR**: Proposes IslandRun, a multi-objective orchestration system for distributed AI inference across personal devices, edge servers, and cloud. Uses agent-based routing, tiered island groups, and reversible anonymization to optimize performance, privacy, and cost. Achieves up to 40% lower cost while maintaining privacy compared to federated baselines.
* `training` `offline` `kernel` [Heimdall++: Optimizing GPU Utilization and Pipeline Parallelism for Efficient Single-Pulse Detection](http://arxiv.org/abs/2512.00398v1)
  > **TL;DR**: Proposes Heimdall++, an optimized GPU parallelization framework for radio astronomy single-pulse detection pipelines. Introduces fine-grained GPU parallelization, enhanced memory management, and multi-threading to decouple CPU/GPU stages, reducing GPU stalls. Achieves up to 2.66x speedup in single-file processing.
* `diffusion` `serving` `offloading` [InvarDiff: Cross-Scale Invariance Caching for Accelerated Diffusion Models](http://arxiv.org/abs/2512.05134v1)
  > **TL;DR**: Proposes Invariance Caching (InvarDiff), a technique to accelerate diffusion model inference by exploiting temporal and layer-wise feature invariance. Implements per-module binary cache plans with re-sampling correction. Achieves 2-3× end-to-end speedups with minimal fidelity impact on DiT and FLUX models.
* `training` `edge` `offloading` [Communication-Computation Pipeline Parallel Split Learning over Wireless Edge Networks](http://arxiv.org/abs/2511.23167v1)
  > **TL;DR**: Proposes C$^2$P$^2$SL, a pipeline parallel split learning framework for wireless edge networks. Overlaps communication and computation by splitting batches and optimizing task split/resource allocation. Achieves 38% reduction in training time while maintaining accuracy.
* `serving` `quantization` `offloading` [Serving Heterogeneous LoRA Adapters in Distributed LLM Inference Systems](http://arxiv.org/abs/2511.22880v1)
  > **TL;DR**: Addresses performance skew in serving heterogeneous LoRA adapters due to rank variability. Proposes LoRAServe, a dynamic adapter placement and routing framework with remote access via GPU Direct RDMA. Achieves up to 2$	imes$ higher throughput and 9$	imes$ lower TTFT with 50% fewer GPUs under SLOs.
* `edge` `storage` `serving` [DisCEdge: Distributed Context Management for Large Language Models at the Edge](http://arxiv.org/abs/2511.22599v1)
  > **TL;DR**: Proposes DisCEdge for distributed context management in edge-deployed LLMs. Stores user context as token sequences for efficient replication and lower bandwidth. Reduces client request sizes by 90% and improves response times by up to 14.46% compared to alternatives.
* `MoE` `serving` `kernel` [OmniInfer: System-Wide Acceleration Techniques for Optimizing LLM Serving Throughput and Latency](http://arxiv.org/abs/2511.22481v1)
  > **TL;DR**: Proposes OmniInfer, a system-level framework optimizing LLM serving via expert placement, sparse attention, and request scheduling. Integrates load-aware MoE scheduling, attention acceleration, and disaggregation-aware coordination. Achieves 616 QPM and reduces TPOT by 36%, TTFT by 38%.
* `serving` `kernel` `inference` [PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel](http://arxiv.org/abs/2511.22333v2)
  > **TL;DR**: Accelerates LLM decoding by exploiting shared prefixes across requests. Proposes PAT, a prefix-aware attention kernel with pack-forward-merge paradigm and multi-tile execution. Achieves 53.5% average attention latency reduction compared to SOTA kernels.
* `serving` `offline` `offloading` [OOCO: Latency-disaggregated Architecture for Online-Offline Co-locate LLM Serving](http://arxiv.org/abs/2511.21862v1)
  > **TL;DR**: Addresses load imbalance in co-located online-offline LLM serving. Proposes latency-disaggregated architecture with separate pools, bottleneck-based scheduler, and preemption mechanism. Achieves 3x higher offline throughput while maintaining online SLOs.
* `offloading` `hardware` `networking` [A Sustainable and Reward Incentivized High-Performance Cluster Computing for Artificial Intelligence: A Novel Bayesian-Time-Decay Trust Mechanism in Blockchain](http://arxiv.org/abs/2511.21844v1)
  > **TL;DR**: Proposes a blockchain-based incentive mechanism for distributed AI cluster computing. Integrates dynamic trust ratings and a statistical draw system to efficiently utilize diverse hardware. Achieves up to 30% improvement in resource utilization while enabling broader participation.
* `serving` `edge` `offloading` [DSD: A Distributed Speculative Decoding Solution for Edge-Cloud Agile Large Model Serving](http://arxiv.org/abs/2511.21669v2)
  > **TL;DR**: Proposes DSD, a distributed speculative decoding framework for edge-cloud LLM serving. Uses DSD-Sim simulator and an Adaptive Window Control policy to coordinate draft-target execution. Achieves up to 1.1x speedup and 9.7% higher throughput over baselines.
* `training` `MoE` `sparse` [MemFine: Memory-Aware Fine-Grained Scheduling for MoE Training](http://arxiv.org/abs/2511.21431v1)
  > **TL;DR**: Addresses memory bottleneck in MoE training due to token imbalanc‎e. Proposes MemFine, using chunk‎ed decomposition and recomputation with a memory model for optimizer. Reduces activation memory by 48.03% and inc‎reases throughput by 4.42%.
* `serving` [Automated Dynamic AI Inference Scaling on HPC-Infrastructure: Integrating Kubernetes, Slurm and vLLM](http://arxiv.org/abs/2511.21413v1)
  > **TL;DR**: Proposes integrating Kubernetes, Slurm, and vLLM to dynamically scale LLM serving workloads on HPC infrastructure. Addresses challenges of synchronous user-facing AI applications on classical HPC systems. Benchmarks show ~500ms overhead scaling to 1000 concurrent requests.
* `training` [GPU Memory Prediction for Multimodal Model Training](http://arxiv.org/abs/2512.07853v1)
  > **TL;DR**: Introduces a framework for predicting peak GPU memory usage during multimodal model training to prevent OOM errors. It decomposes models into layers and applies factorization for memory estimation. Achieves ~8.7% MAPE accuracy.
* `serving` `offloading` `kernel` [LLaMCAT: Optimizing Large Language Model Inference with Cache Arbitration and Throttling](http://arxiv.org/abs/2512.00083v1)
  > **TL;DR**: Proposes LLaMCAT for reducing KV cache stalls in LLM inference via LLC optimization. Combines MSHR-aware cache arbitration, load balancing, and thread throttling. Achieves up to 1.58x speedup over unoptimized baseline under limited cache scenarios.
* `networking` `hardware` [Handling of Memory Page Faults during Virtual-Address RDMA](http://arxiv.org/abs/2511.21018v1)
  > **TL;DR**: Proposes a hardware-software mechanism for handling memory page faults during RDMA communication. Integrates fault detection via ARM SMMU and resolution through retransmission, modifying Linux drivers and DMA engine. Reduces overhead and improves efficiency compared to pinning.
* `serving` `RAG` `offloading` [Efficient Multi-Adapter LLM Serving via Cross-Model KV-Cache Reuse with Activated LoRA](http://arxiv.org/abs/2512.17910v1)
  > **TL;DR**: Addresses inefficiency in multi-adapter LLM serving pipelines via Activated LoRA for cross-model KV-cache reuse. Proposes base-aligned block hashing and activation-aware masking in vLLM. Achieves up to 58x latency reduction and 100x TTFT improvement over standard LoRA baselines.
* `serving` `offloading` `quantization` [DOPO: A Dynamic PD-Disaggregation Architecture for Maximizing Goodput in LLM Inference Serving](http://arxiv.org/abs/2511.20982v2)
  > **TL;DR**: Proposes DOPD, a dynamic LLM inference system that optimizes prefill-to-decoding instance ratio via real-time load monitoring and request scheduling. Achieves 1.5x higher goodput, 67.5% lower P90 TTFT, and 22.8% lower P90 TPOT vs vLLM and DistServe.
* `serving` `RL` `agentic` [Aragog: Just-in-Time Model Routing for Scalable Serving of Agentic Workflows](http://arxiv.org/abs/2511.20975v2)
  > **TL;DR**: Addresses cost and latency in serving multi-stage agentic workflows by dynamically routing agents to LLMs. Proposes a two-step approach: offline accuracy-preserving configuration identification and runtime scheduler using system load. Increases throughput by 50-217% and reduces latency by 32.5-78.9%.
* `offloading` `serving` `hardware` [Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management](http://arxiv.org/abs/2511.20172v2)
  > **TL;DR**: Proposes Beluga, a CXL-based memory architecture for scalable KVCache management in LLM inference. Enables GPU direct access to a shared memory pool via CXL, reducing latency and complexity. Achieves 89.6% lower TTFT and 7.35x throughput gains over RDMA.
* `kernel` `training` [QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation](http://arxiv.org/abs/2511.20100v1)
  > **TL;DR**: Addresses efficient generation of optimized GPU kernels using LLMs. Proposes a hierarchical framework (Macro Thinking Micro Coding) decoupling optimization strategies (via RL-guided LLMs) from implementation details (via general LLMs). Achieves up to 7.3x speedup over LLMs and 2.2x over PyTorch kernels on KernelBench.
* `training` `RL` `networking` [ParaBlock: Communication-Computation Parallel Block Coordinate Federated Learning for Large Language Models](http://arxiv.org/abs/2511.19959v1)
  > **TL;DR**: Proposes ParaBlock, a communication-computation parallel federated learning method for efficient LLM fine-tuning. Uses dual threads to overlap local block updates and communication. Achieves 50% communication reduction while maintaining convergence rate and performance on instruction following tasks.
* `batch_denoising` [Batch Denoising for AIGC Service Provisioning in Wireless Edge Networks](http://arxiv.org/abs/2511.19847v1)
  > **TL;DR**: Proposes a batch denoising framework for AI-generated image services in edge networks. Develops STACKING algorithm to optimize batch processing and transmission, reducing per-step delay and improving parallelism. Achieves up to 40% lower latency and higher quality within service delay constraints.
* `storage` `edge` `RAG` [AME: An Efficient Heterogeneous Agentic Memory Engine for Smartphones](http://arxiv.org/abs/2511.19192v1)
  > **TL;DR**: Proposes AME, an on-device agentic memory engine for smartphones, optimizing vector database storage with hardware-aware matrix pipelines and workload-aware scheduling. Achieves up to 1.4x higher query throughput at matched recall and 7x faster index construction.
* `training` `serving` `hardware` [An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds](http://arxiv.org/abs/2511.18906v1)
  > **TL;DR**: Addresses GPU fragmentation in multi-tenant MIG clouds for AI workloads. Proposes an online fragmentation-aware scheduler using a fragmentation metric and greedy algorithm. Achieves 10% higher workload acceptance rate under heavy load.
* `kernel` `quantization` `sparse` [Low-Rank GEMM: Efficient Matrix Multiplication via Low-Rank Approximation with FP8 Acceleration](http://arxiv.org/abs/2511.18674v1)
  > **TL;DR**: Proposes Low-Rank GEMM for matrix multiplication via low-rank approximations and FP8. Leverages hardware-aware decomposition selection and FP8 quantization to reduce computation complexity and memory usage. Achieves 7.8× speedup over PyTorch FP32 and 75% memory savings for large matrices (N=20480).
* `RL` `training` [ADF-LoRA: Alternating Low-Rank Aggregation for Decentralized Federated Fine-Tuning](http://arxiv.org/abs/2511.18291v1)
  > **TL;DR**: Proposes ADF-LoRA for decentralized federated fine-tuning, alternating updates of one low-rank matrix per round while mixing both to stabilize decentralized propagation. Achieves faster convergence and higher average accuracy on GLUE tasks vs. existing LoRA variants in decentralized FL.
* `edge` `offloading` `multi-modal` [AVERY: Adaptive VLM Split Computing through Embodied Self-Awareness for Efficient Disaster Response Systems](http://arxiv.org/abs/2511.18151v1)
  > **TL;DR**: Proposes AVERY, an adaptive split computing framework for Vision-Language Models on UAVs. Features dual-stream separation and dynamic compression selection based on network conditions. Achieves 93.98% lower energy consumption vs full-edge execution and 11.2% higher accuracy than raw compression.
* `scheduling` `training` `storage` [Simulating Dynamic Cloud Marketspaces: Modeling Spot Instance Behavior and Scheduling with CloudSim Plus](http://arxiv.org/abs/2511.18137v1)
  > **TL;DR**: Extends CloudSim Plus to model spot instance lifecycle for cost-effective workload scheduling in cloud environments. Proposes HLEM-VMP algorithm to reduce interruptions under volatility. Achieves fewer interruptions and shorter max interruption duration in simulations using Google Cluster Trace.
* `training` [Pier: Efficient Large Language Model pretraining with Relaxed Global Communication](http://arxiv.org/abs/2511.17849v2)
  > **TL;DR**: Proposes Pier, an optimizer reducing global communication bottlenecks in LLM pretraining via relaxed communication, momentum warmup/decay, and parallelization. Achieves 2.7x-3.7x GPT-2 XL speedup on 256 A100 GPUs with no loss degradation.
* `training` `hardware` `networking` [Towards a future space-based, highly scalable AI infrastructure system design](http://arxiv.org/abs/2511.19468v1)
  > **TL;DR**: Investigates space-based AI infrastructure for scalable ML compute using solar power. Proposes satellite fleets with TPUs, optical interlinks, and formation flight for low-latency communication. Radiation tests show Trillium TPUs withstand 5-year mission doses; projected launch costs drop below $200/kg by 2030s.
* `MoE` `training` `hardware` [Training Foundation Models on a Full-Stack AMD Platform: Compute, Networking, and System Design](http://arxiv.org/abs/2511.17127v2)
  > **TL;DR**: Presents large-scale MoE pretraining on AMD hardware (MI300X GPUs, Pollara networking), with cluster characterization, kernel/memory microbenchmarks, and transformer/MoE sizing rules. Optimizes for training throughput and inference latency. Achieves competitive model performance (ZAYA1-base) vs Qwen3-4B/Gemma3-12B.
* `edge` `sparse` `kernel` [SparOA: Sparse and Operator-aware Hybrid Scheduling for Edge DNN Inference](http://arxiv.org/abs/2511.19457v1)
  > **TL;DR**: Proposes SparOA, a CPU-GPU hybrid DNN inference framework for edge devices using sparsity and operator-aware scheduling. It includes a RL-based scheduler and asynchronous execution with batch optimization. Achieves 1.22-1.31x speedup and 7-16% lower energy consumption than baselines.
* `serving` `kernel` [Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2511.16964v1)
  > **TL;DR**: Investigates how to optimize PyTorch inference using LLM-based multi-agent systems. Proposes a framework comparing strategies with exploit-heavy approaches and error-fixing agents. Achieves 2.88x speedup on H100 GPU across tasks in KernelBench.
* `training` `MoE` `sparse` [MicroMoE: Fine-Grained Load Balancing for Mixture-of-Experts with Token Scheduling](http://arxiv.org/abs/2511.16947v1)
  > **TL;DR**: Addresses load imbalance in Mixture-of-Experts training. Proposes MicroEP, a fine-grained parallelization strategy with token scheduling, and implements it in MicroMoE system. Achieves 47.6% higher training throughput over state-of-the-art.
* `training` `RL` `sparse` [Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter](http://arxiv.org/abs/2511.16665v1)
  > **TL;DR**: Addresses long-tail execution inefficiency in RL-based LLM reasoning training. Proposes TLT with Adaptive Drafter (idle GPU-trained speculative model) and Adaptive Rollout Engine (CUDAGraph strategies). Achieves 1.7x end-to-end RL training speedup without accuracy loss.
* `training` `RL` `quantization` [Fast LLM Post-training via Decoupled and Best-of-N Speculation](http://arxiv.org/abs/2511.16193v2)
  > **TL;DR**: Accelerates LLM post-training rollout via dynamic decoupled and Best-of-N speculative decoding. Maximizes GPU efficiency for large batches and adapts drafting methods. Achieves 1.3–1.5× faster rollout than speculative decoding baselines.
* `storage` `networking` [Mitigating Shared Storage Congestion Using Control Theory](http://arxiv.org/abs/2511.16177v1)
  > **TL;DR**: Proposes a control-theoretic approach to regulate client I/O rates dynamically in HPC shared storage systems. Uses runtime load metrics to mitigate congestion, reducing total runtime by up to 20% and tail latency while ensuring stability.
* `kernel` `training` [Can Asymmetric Tile Buffering Be Beneficial?](http://arxiv.org/abs/2511.16041v1)
  > **TL;DR**: Examines tiling strategies for GEMM operations in AI workloads. Proposes asymmetric tile buffering (ATB) that decouples buffered tile dimensions of inputs and outputs. Achieves up to 4.54x speedup (24.6 TFLOPS vs 4.8 TFLOPS) on mixed-precision GEMM for AMD XDNA2 AI Engine.
* `serving` `hardware` `kernel` [A Tensor Compiler for Processing-In-Memory Architectures](http://arxiv.org/abs/2511.15503v1)
  > **TL;DR**: Proposes DCC, a data-centric compiler for PIM systems that co-optimizes data rearrangements and compute code for ML kernels on diverse PIM backends. Achieves up to 7.71x speedup in LLM inference over GPU on AttAcc PIM.
* `MoE` `training` `networking` [GPU-Initiated Networking for NCCL](http://arxiv.org/abs/2511.15076v2)
  > **TL;DR**: Proposes GPU-Initiated Networking (GIN) in NCCL to enable device-initiated RDMA communication for modern AI workloads like MoE. Integrates three operation modes and leverages DOCA GPUNetIO or proxy queues to reduce CPU overhead. Benchmarks show up to 8-16x lower latency in MoE communication.
* `kernel` `training` `serving` [PolyKAN: Efficient Fused GPU Operators for Polynomial Kolmogorov-Arnold Network Variants](http://arxiv.org/abs/2511.14852v1)
  > **TL;DR**: Proposes PolyKAN, an optimized CUDA kernel library for polynomial KAN variants to overcome low GPU utilization. Fuses forward/backward passes using lookup tables, 2D tiling, two-stage reduction, and coefficient reordering. Achieves up to 10× faster inference and 12× faster training over Triton+cuBLAS baseline.
* `training` `RL` `serving` [Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning](http://arxiv.org/abs/2511.14617v2)
  > **TL;DR**: Addresses synchronous RL rollout bottlenecks for LLMs via online context learning. Proposals include divided rollout load balancing, context-aware scheduling, and adaptive grouped speculative decoding. Seer improves rollout throughput by up to 97% and reduces long-tail latency by up to 93%.
* `serving` `offloading` `edge` [Hyperion: Hierarchical Scheduling for Parallel LLM Acceleration in Multi-tier Networks](http://arxiv.org/abs/2511.14450v2)
  > **TL;DR**: Proposes Hyperion, a hierarchical framework for efficient LLM inference in multi-tier edge networks by jointly optimizing model partitioning and request scheduling. Combines offline inter-tier partitioning (HypSplit-DP) with online intra-tier scheduling (HypSched-RT). Achieves 52.1% latency reduction vs. GPipe.
* `training` `offloading` [10Cache: Heterogeneous Resource-Aware Tensor Caching and Migration for LLM Training](http://arxiv.org/abs/2511.14124v1)
  > **TL;DR**: Proposes 10Cache, a tensor caching and migration system for LLM training that optimizes memory across GPU/CPU/NVMe tiers via prefetching and buffer reuse. Achieves up to 2x training speedup and 86.6x higher GPU cache hit rate over existing offloading methods.
* `serving` `training` [FailSafe: High-performance Resilient Serving](http://arxiv.org/abs/2511.14116v1)
  > **TL;DR**: Addresses fragility in tensor-parallel LLM serving under GPU failures. Proposes FailSafe with cyclic KVCache placement, hybrid attention, and dynamic routing to balance computation and memory, plus proactive backup for fault recovery. Achieves 2× higher throughput and 100× lower recovery latency amidst multiple GPU failures.
* `MoE` `offloading` `quantization` [MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts](http://arxiv.org/abs/2511.14102v1)
  > **TL;DR**: Addresses I/O bottlenecks during resource-limited MoE inference due to synchronous expert offloading. Proposes speculative expert sequence prediction via a draft model and prefetching, dynamically tuned by an amortization roofline model. Achieves up to 2.34x speedup over state-of-the-art offloading frameworks.
* `training` `kernel` `networking` [ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels](http://arxiv.org/abs/2511.13940v1)
  > **TL;DR**: Proposes ParallelKittens, a minimal CUDA framework simplifying multi-GPU kernel design for communication-efficient AI workloads. Embodies principles via reusable primitives and unified template across architectures. Achieves up to 4.08× speedup for sequence-parallel workloads on Hopper/Blackwell GPUs.
* `serving` `offloading` `kernel` [Comparative Analysis of Large Language Model Inference Serving Systems: A Performance Study of vLLM and HuggingFace TGI](http://arxiv.org/abs/2511.17593v1)
  > **TL;DR**: Empirically evaluates LLM serving systems vLLM and TGI. Compares throughput, latency, memory, and scalability using LLaMA-2 models. vLLM achieves 24× higher throughput than TGI under high concurrency via PagedAttention, while TGI offers lower tail latency for interactive use.
* `edge` `training` `offloading` [Distributed Hierarchical Machine Learning for Joint Resource Allocation and Slice Selection in In-Network Edge Systems](http://arxiv.org/abs/2511.13313v2)
  > **TL;DR**: Proposes a distributed hierarchical model (DeepSets-S) for joint resource allocation and slice selection in edge systems. Uses a DeepSets architecture with slack-aware normalization and task-specific decoders for efficient online inference. Reduces execution time by 86.1% while maintaining within 6.1% of optimal cost.
* `edge` `serving` [Pico-Cloud: Cloud Infrastructure for Tiny Edge Devices](http://arxiv.org/abs/2511.13253v1)
  > **TL;DR**: Proposes Pico-Cloud, a micro-edge cloud architecture for ultra-minimal hardware like Raspberry Pi Zero. Features container virtualization and lightweight orchestration for local low-latency workloads, including edge AI inference. Achieves cost-effective and decentralized operation for lightweight distributed applications.
* `training` `storage` `offloading` [Learning Process Energy Profiles from Node-Level Power Data](http://arxiv.org/abs/2511.13155v1)
  > **TL;DR**: Proposes a method to model per-process energy consumption using synchronized node-level power data and process-level resource metrics via regression. Enables fine-grained energy prediction, improving efficiency monitoring for AI workloads over coarse tools like RAPL.
* `sparse` `kernel` `inference` [MACKO: Sparse Matrix-Vector Multiplication for Low Sparsity](http://arxiv.org/abs/2511.13061v1)
  > **TL;DR**: Proposes MACKO-SpMV, a GPU-optimized sparse matrix-vector multiplication format and kernel for efficient inference in pruned sparse LLMs. Achieves 1.5x memory reduction and 1.5x speedup over dense at 50% sparsity with unstructured pruning.
* `training` `kernel` `networking` [Iris: First-Class Multi-GPU Programming Experience in Triton](http://arxiv.org/abs/2511.12500v1)
  > **TL;DR**: Addresses complexity in multi-GPU programming by introducing Iris, a Python/Triton-based communication library with tile-based symmetric memory abstractions. Enables computation-communication overlap with minimal code. Achieves 1.79x speedup over PyTorch/RCCL for GEMM+All-Scatter.
* `edge` `networking` `multi-modal` [Semantic Multiplexing](http://arxiv.org/abs/2511.13779v1)
  > **TL;DR**: Proposes Semantic Multiplexing to offload parallel tasks by merging compressed task representations for edge devices. Shifts multiplexing from bits to tasks, enabling more concurrent tasks than physical channels. Reduces latency by 8× and energy consumption by 25× while maintaining accuracy within 4% drop for 8 tasks over 4×4 channel.
* `kernel` `quantization` [Guaranteed DGEMM Accuracy While Using Reduced Precision Tensor Cores Through Extensions of the Ozaki Scheme](http://arxiv.org/abs/2511.13778v1)
  > **TL;DR**: Proposes Automatic Dynamic Precision (ADP) for emulating FP64 accuracy using low-precision Tensor Cores. ESC estimator determines decomposition parameters, with exception handling and fallback to FP64. Achieves up to 13.2x speedup over native FP64 GEMM while preserving accuracy with <10% overhead.
* `serverless` `storage` `networking` [Combining Serverless and High-Performance Computing Paradigms to support ML Data-Intensive Applications](http://arxiv.org/abs/2511.12185v2)
  > **TL;DR**: Proposes Cylon, a high-performance distributed data frame solution using TCP hole punching for serverless environments. Enables direct communication between functions to speed up machine learning data pipelines. Achieves performance close to EC2/HPC clusters.
* `video` `training` `offloading` [PipeDiT: Accelerating Diffusion Transformers in Video Generation with Task Pipelining and Model Decoupling](http://arxiv.org/abs/2511.12056v1)
  > **TL;DR**: Addresses slow inference and high memory in diffusion transformer video generation. Proposes PipeDiT with pipelined sequence parallelism, decoupled VAE execution, and attention co-processing. Achieves up to 4.02x speedup over state-of-the-art frameworks on 8-GPU systems.
* `serving` `offline` `offloading` [Striking the Right Balance between Compute and Copy: Improving LLM Inferencing Under Speculative Decoding](http://arxiv.org/abs/2511.12031v1)
  > **TL;DR**: Proposes BMC, a KV cache allocation method balancing memory copy overhead and redundant compute for LLM inference. BMC allocates tensors with redundant rows periodically to avoid copies, using extra rows for speculative decoding. Achieves up to 3.2x throughput over HuggingFace baseline and 1.36x over vLLM server.
* `kernel` [Modular GPU Programming with Typed Perspectives](http://arxiv.org/abs/2511.11939v1)
  > **TL;DR**: Addresses modular GPU programming challenges for collective operations like Tensor Core instructions. Introduces Prism, a GPU language with typed perspectives tracking thread granularity control. Maintains performance parity without safety compromises for state-of-the-art kernels.
* `offloading` `edge` `kernel` [KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference](http://arxiv.org/abs/2511.11907v2)
  > **TL;DR**: Addresses on-device KV cache memory bottleneck for long-context inference. Proposes KVSwap, a disk-aware offloading framework with predictive preloading, compute-disk overlap, and I/O-optimized access patterns. Achieves higher throughput under memory constraints with no quality loss.
* `edge` `serving` `RAG` [Flash-Fusion: Enabling Expressive, Low-Latency Queries on IoT Sensor Streams with LLMs](http://arxiv.org/abs/2511.11885v1)
  > **TL;DR**: Proposes Flash-Fusion, an edge-cloud system for low-latency LLM queries on IoT sensor streams. Uses edge summarization for data reduction and cloud-based query planning with clustered data for prompt assembly. Achieves 95% latency reduction and 98% cost decrease compared to raw data feeding.
* `agentic` `edge` `networking` [UFO$^3$: Weaving the Digital Agent Galaxy](http://arxiv.org/abs/2511.11332v1)
  > **TL;DR**: Proposes UFO$^3$, a system for orchestrating LLM-powered agents across heterogeneous devices. Models user requests as mutable TaskConstellations with dynamic DAGs, asynchronous execution, and reliable Agent Interaction Protocol. Reduces end-to-end latency by 31% on a benchmark of 55 cross-device tasks.
* `training` `networking` [What happens when nanochat meets DiLoCo?](http://arxiv.org/abs/2511.13761v1)
  > **TL;DR**: Studies performance impact of DiLoCo's communication-efficient decentralized training on LLMs. Implements DiLoCo with inner-outer optimizer into nanochat baseline, reducing communication by orders of magnitude vs data-parallel training. Shows irreversible representation drift causing 5-10% accuracy drops on downstream tasks despite pretraining convergence.
* `offloading` `edge` `offline` [SemanticNN: Compressive and Error-Resilient Semantic Offloading for Extremely Weak Devices](http://arxiv.org/abs/2511.11038v1)
  > **TL;DR**: Proposes SemanticNN, an error-resilient semantic offloading system for weak IoT devices featuring BER-aware decoder and Soft Quantization encoder. Includes Feature-augmentation Learning and XAI-based Asymmetry Compensation. Reduces feature transmission volume by 56.82-344.83x while maintaining accuracy on STM32.
* `agentic` `RL` `offline` [HPCAgentTester: A Multi-Agent LLM Approach for Enhanced HPC Unit Test Generation](http://arxiv.org/abs/2511.10860v1)
  > **TL;DR**: Proposes a multi-agent LLM framework for automated HPC unit test generation. Uses specialized agents in iterative critique loops to generate tests targeting parallelism and communication patterns. Increases test compilation rates by 30% and correctness by 25% compared to standalone LLMs.
* `edge` `serving` `multi-modal` [EarthSight: A Distributed Framework for Low-Latency Satellite Intelligence](http://arxiv.org/abs/2511.10834v1)
  > **TL;DR**: Addresses high latency in satellite image analysis by introducing EarthSight, a distributed runtime for multi-task inference with shared backbones, query scheduling, and dynamic filtering. Achieves 1.9× faster compute time and reduces 90th percentile latency from 51 to 21 minutes.
* `inference` `offloading` `hardware` [FengHuang: Next-Generation Memory Orchestration for AI Inferencing](http://arxiv.org/abs/2511.10753v1)
  > **TL;DR**: Proposes FengHuang, a disaggregated infrastructure with multi-tier memory and active tensor paging to overcome GPU memory and scaling limits for LLM inference. Achieves 93% local memory reduction, 50% GPU savings, and 16x-70x faster inter-GPU communication while maintaining performance.
* `training` `RL` `storage` [STAGE: A Symbolic Tensor grAph GEnerator for distributed AI system co-design](http://arxiv.org/abs/2511.10480v2)
  > **TL;DR**: Introduces STAGE, a framework for synthesizing high-fidelity execution traces to model LLM workloads. Enables scalable exploration of parallelization strategies and system configurations for large-scale training. Generates traces spanning 32K GPUs with tensor-level accuracy in compute, memory, and communication.
* `kernel` `offloading` [On The Performance of Prefix-Sum Parallel Kalman Filters and Smoothers on GPUs](http://arxiv.org/abs/2511.10363v1)
  > **TL;DR**: Evaluates parallel scan algorithms for GPU-based Kalman filters and smoothers. Implements all-prefix-sum algorithms for temporal parallelization and proposes a novel two-filter smoother. Achieves reduced run times on CUDA/Metal GPUs through optimized kernel operations and offloading techniques.
* `edge` `serving` `RL` [Dynamic Edge Server Selection in Time-Varying Environments: A Reliability-Aware Predictive Approach](http://arxiv.org/abs/2511.10146v1)
  > **TL;DR**: Proposes MO-HAN, a lightweight edge server selection method for embedded applications. It fuses latency prediction with adaptive reliability and hysteresis-based handover to balance predicted latency and reliability. Reduces handovers by 50% and lowers mean/tail latencies compared to baselines.
* `serving` `offloading` `networking` [Speculative Decoding in Decentralized LLM Inference: Turning Communication Latency into Computation Throughput](http://arxiv.org/abs/2511.11733v1)
  > **TL;DR**: Proposes Decentralized Speculative Decoding (DSD) to turn network latency into computation throughput for distributed LLM inference. Uses parallel verification of candidate tokens across nodes and adaptive thresholding by token importance. Achieves 2.59x speedup on GSM8K with preserved accuracy.
* `serving` `offline` `RL` [Harli: SLO-Aware Co-location of LLM Inference and PEFT-based Finetuning on Model-as-a-Service Platforms](http://arxiv.org/abs/2511.11729v2)
  > **TL;DR**: Proposes Harli, a system for co-locating PEFT-based finetuning with LLM inference decode. Integrates memory reuse, latency prediction, and QoS-aware scheduling to maximize throughput. Achieves 46.2% average finetune throughput gain while maintaining QoS.
* `training` `hardware` `offloading` [Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs](http://arxiv.org/abs/2511.09861v2)
  > **TL;DR**: Investigates thermal imbalance-induced performance variation in multi-GPU LLM training systems. Proposes analytical models and mitigation techniques via power management optimization. Achieves up to 6% performance gain and 4% power reduction on AMD MI300X systems.
* `training` `quantization` [MoFa: A Unified Performance Modeling Framework for LLM Pretraining](http://arxiv.org/abs/2511.09837v2)
  > **TL;DR**: Proposes MoFa, a unified performance modeling framework for LLM pretraining that integrates optimization features and fault tolerance. Uses an enhanced cost model and reliability-based fault model to optimize distributed training strategies. Achieves high prediction accuracy and reveals performance bottlenecks for pretraining configurations.
* `edge` `offloading` `serving` [ECCENTRIC: Edge-Cloud Collaboration Framework for Distributed Inference Using Knowledge Adaptation](http://arxiv.org/abs/2511.11719v1)
  > **TL;DR**: Proposes ECCENTRIC, an edge-cloud collaboration framework for distributed inference that uses knowledge adaptation to balance computation, communication, and performance. Achieves reduced costs while maintaining accuracy for classification and object detection tasks.
* `training` `MoE` `sparse` [TawPipe: Topology-Aware Weight Pipeline Parallelism for Accelerating Long-Context Large Models Training](http://arxiv.org/abs/2511.09741v1)
  > **TL;DR**: Addresses high communication overhead in long-context LLM training and intra-node bandwidth underutilization. Proposes topology-aware weight pipeline parallelism that groups devices hierarchically to reduce redundant transfers and overlaps communication with computation. Achieves up to 1.5x higher throughput and scales to 24 GPUs without saturation.
* `serving` `networking` `offline` [LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations with Fast All-Reduce Communication](http://arxiv.org/abs/2511.09557v3)
  > **TL;DR**: Analyzes bottlenecks in multi-node distributed LLM inference and introduces NVRAR, a fast hierarchical all-reduce communication algorithm using NVSHMEM. Achieves up to 1.9×-3.6× lower latency for key operations and 1.72× lower batch latency for Llama 405B in decode workloads.
* `serving` `storage` `hardware` [Flex-MIG: Enabling Distributed Execution on MIG](http://arxiv.org/abs/2511.09143v2)
  > **TL;DR**: Addresses GPU underutilization in multi-tenant clusters with NVIDIA MIG. Proposes Flex-MIG, a software framework enabling distributed execution across MIG instances via host-shared-memory collectives. Achieves up to 17% makespan improvement by reducing fragmentation and eliminating drain-required reconfiguration.
* `edge` `training` `RL` [A Structure-Agnostic Co-Tuning Framework for LLMs and SLMs in Cloud-Edge Systems](http://arxiv.org/abs/2511.11678v1)
  > **TL;DR**: Proposes Co-PLMs, a co-tuning framework for collaborative training of cloud LLMs and edge SLMs via distilled proxy models, enabling structure-agnostic knowledge exchange. Achieves 5.38% Rouge-L and 4.88% EM gains over SOTA.
* `training` `edge` `hardware` [OSGym: Super-Scalable Distributed Data Engine for Generalizable Computer Agents](http://arxiv.org/abs/2511.11672v2)
  > **TL;DR**: Presents OSGym, a scalable and affordable distributed engine for training computer agents. Utilizes parallelized OS replicas for dynamic environments, optimizing resource use and cost. Generates up to 1420 trajectories/min at $0.2-0.3/day per instance.
* `offloading` `hardware` `kernel` [An MLIR pipeline for offloading Fortran to FPGAs via OpenMP](http://arxiv.org/abs/2511.08713v1)
  > **TL;DR**: Proposes an MLIR-based pipeline for offloading Fortran code to FPGAs via OpenMP directives. Combines OpenMP and HLS dialects to enable portable compilation and kernel optimization. Achieves reduced development effort while supporting manual kernel tuning for FPGA acceleration.
* `edge` `offloading` `storage` [Range Asymmetric Numeral Systems-Based Lightweight Intermediate Feature Compression for Split Computing of Deep Neural Networks](http://arxiv.org/abs/2511.11664v1)
  > **TL;DR**: Addresses communication overhead in split DNN inference by proposing a lightweight compression framework using rANS encoding, asymmetric integer quantization, and sparse tensor representation. Achieves near-baseline accuracy on vision and language models (e.g., ResNet, Llama2) with sub-millisecond encoding latency.
* `hardware` `kernel` `sparse` [LOw-cOst yet High-Performant Sparse Matrix-Matrix Multiplication on Arm SME Architectures](http://arxiv.org/abs/2511.08158v2)
  > **TL;DR**: Proposes LOOPS, a hybrid framework for sparse matrix multiplication exploiting Arm SME and SIMD resources. Combines row-wise and vector-wise layouts with adaptive parallelization, achieving up to 14.4x speedup and superior energy efficiency over baselines on Apple M4Pro CPU compared to GPUs.
* `training` `kernel` [UniFormer: Unified and Efficient Transformer for Reasoning Across General and Custom Computing](http://arxiv.org/abs/2511.08135v1)
  > **TL;DR**: Proposes UniFormer, a unified Transformer architecture for efficient deployment on both general-purpose GPUs and custom hardware. Unifies parallelism and compute-storage fusion to optimize performance. Achieves SOTA accuracy and latency on GPUs with improved FPGA adaptability.
* `edge` `serving` `offloading` [Intelligence per Watt: Measuring Intelligence Efficiency of Local AI](http://arxiv.org/abs/2511.07885v2)
  > **TL;DR**: Proposes intelligence per watt (IPW) to measure efficiency of local LLM inference on edge devices. Evaluates 20+ models across 8 accelerators using 1M queries, finding local accelerators achieve 1.4x better IPW than cloud counterparts, enabling feasible redistribution of inference demand.
* `serving` `training` `quantization` [Parallel Sampling via Autospeculation](http://arxiv.org/abs/2511.07869v1)
  > **TL;DR**: Proposes parallel sampling via autospeculation to reduce sampling time for autoregressive and diffusion models. Uses speculative rejection sampling with sequence-level speculation from the same oracle. Achieves expected sampling time of ̃O(n^{1/2}), improving over sequential O(n).
* `training` `hardware` `kernel` [HeteroSTA: A CPU-GPU Heterogeneous Static Timing Analysis Engine with Holistic Industrial Design Support](http://arxiv.org/abs/2511.11660v1)
  > **TL;DR**: Proposes HeteroSTA, a CPU-GPU heterogeneous timing analysis engine supporting various delay models and industry formats. Features end-to-end GPU acceleration for graph-based and path-based timing queries. Achieves remarkable runtime speed-up in industrial use cases.
* `thinking` `kernel` `RAG` [SemanticForge: Repository-Level Code Generation through Semantic Knowledge Graphs and Constraint Satisfaction](http://arxiv.org/abs/2511.07584v1)
  > **TL;DR**: Proposes SemanticForge for repository-level code generation by integrating semantic knowledge graphs and constraint solving. Combines graph query learning, beam search with SMT solving, and incremental graph maintenance to address logical/schematic hallucination. Achieves 73% query precision versus 51% baseline.
* `serving` `hardware` `offloading` [LLMServingSim2.0: A Unified Simulator for Heterogeneous Hardware and Serving Techniques in LLM Infrastructure](http://arxiv.org/abs/2511.07229v1)
  > **TL;DR**: Presents LLMServingSim2.0, a trace-driven simulator for heterogeneous hardware integration in LLM serving systems. Features operator-level latency profiling and flexible interfaces for serving techniques like request routing and cache management. Achieves 18.5x fewer LoC in TPU integration with 1.9% error in GPU serving simulation.
* `diffusion` `serving` `offloading` [Argus: Quality-Aware High-Throughput Text-to-Image Inference Serving System](http://arxiv.org/abs/2511.06724v1)
  > **TL;DR**: Focuses on high-throughput serving for text-to-image diffusion models. Proposes Argus, a system that dynamically selects approximation levels per prompt to balance quality and throughput. Achieves 40% higher throughput with 10% higher average quality.
* `training` `networking` `offloading` [DMA Collectives for Efficient ML Communication Offloads](http://arxiv.org/abs/2511.06605v1)
  > **TL;DR**: Analyses DMA offloading for ML communication collectives to improve training efficiency. Optimizes DMA collective implementations on AMD GPUs by reducing synchronization costs. Achieves up to 16% better performance and 32% lower power at large sizes, improving small-size performance by 20-30%.
* `networking` [Towards Optimal Constellation Design for Digital Over-the-Air Computation](http://arxiv.org/abs/2511.06372v1)
  > **TL;DR**: Addresses optimal constellation design for digital over-the-air computation to minimize mean-squared error. Proposes a modulation framework with non-linear equations for unique solutions and high-SNR closed-form analysis. Achieves reduced MSE under power constraints for efficient wireless function aggregation.
* `kernel` `training` [PRAGMA: A Profiling-Reasoned Multi-Agent Framework for Automatic Kernel Optimization](http://arxiv.org/abs/2511.06345v2)
  > **TL;DR**: PRAGMA proposes a profiling-reasoned multi-agent framework for automatic GPU/CPU kernel generation. It integrates hardware profiling into the LLM reasoning loop to identify bottlenecks and iteratively refine kernels. Achieves 2.81× and 2.30× speedups on CPU and GPU platforms compared to Torch.
* `serving` `sparse` `offloading` [Optimizing Long-context LLM Serving via Fine-grained Sequence Parallelism](http://arxiv.org/abs/2511.06247v2)
  > **TL;DR**: Optimizes long-context LLM serving with fine-grained sequence parallelism. Proposes Tetris system that dynamically allocates parallelism per token segments in intra-request chunks. Achieves 4.35× lower TTFT, 40.1% median TBT reduction, and 45% higher request capacity.
* `training` `hardware` `kernel` [Exploring Parallelism in FPGA-Based Accelerators for Machine Learning Applications](http://arxiv.org/abs/2511.11640v1)
  > **TL;DR**: Proposes speculative backpropagation for neural network training acceleration by overlapping forward and backward passes. Implemented using OpenMP multi-threading on CPUs and targeted for FPGA synthesis. Achieves up to 35% speedup in step execution time while maintaining comparable accuracy on MNIST.
* `networking` `offline` `storage` [Elastic Data Transfer Optimization with Hybrid Reinforcement Learning](http://arxiv.org/abs/2511.06159v2)
  > **TL;DR**: Optimizes large-scale data transfer for scientific datasets using hybrid heuristics and deep reinforcement learning. Combines parallelism, pipelining, and a lightweight simulator for fast concurrency tuning. Achieves 9.5x higher throughput compared to state-of-the-art solutions.
* `serving` `offloading` `kernel` [MoSKA: Mixture of Shared KV Attention for Efficient Long-Sequence LLM Inference](http://arxiv.org/abs/2511.06010v1)
  > **TL;DR**: Addresses KV cache bottlenecks in long-sequence LLM inference. Proposes MoSKA with Shared KV Attention to batch shared data processing (GEMM instead of GEMV), sparse attention pruning, and disaggregated hardware. Achieves up to 538.7x higher throughput in high-sharing scenarios.
* `serving` `kernel` [Kunlun Anomaly Troubleshooter: Enabling Kernel-Level Anomaly Detection and Causal Reasoning for Large Model Distributed Inference](http://arxiv.org/abs/2511.05978v1)
  > **TL;DR**: Presents Kunlun Anomaly Troubleshooter (KAT) for kernel-level anomaly detection in distributed LLM inference. Uses GPU trace data and domain-adapted LLM for causal reasoning. Achieves 0.884 precision and 0.936 recall in anomaly detection.
* `offloading` `RL` `networking` [DWM-RO: Decentralized World Models with Reasoning Offloading for SWIPT-enabled Satellite-Terrestrial HetNets](http://arxiv.org/abs/2511.05972v1)
  > **TL;DR**: Proposes DWM-RO framework for decentralized MARL in satellite-terrestrial hetnets with SWIPT. Uses world models for predictive representations and offloading gate for selective edge coordination. Achieves 5x faster convergence, 34.7% higher spectral efficiency, and 40% lower violations.
* `hardware` `training` `kernel` [MT4G: A Tool for Reliable Auto-Discovery of NVIDIA and AMD GPU Compute and Memory Topologies](http://arxiv.org/abs/2511.05958v1)
  > **TL;DR**: Proposes MT4G, a vendor-agnostic tool for auto-discovering GPU compute/memory topologies via microbenchmarks and statistical methods. Enables performance modeling, bottleneck analysis, and resource partitioning, demonstrated across NVIDIA/AMD GPUs to optimize HPC/AI system utilization.
* `edge` `RAG` `serving` [CoEdge-RAG: Optimizing Hierarchical Scheduling for Retrieval-Augmented LLMs in Collaborative Edge Computing](http://arxiv.org/abs/2511.05915v1)
  > **TL;DR**: Proposes CoEdge-RAG, a hierarchical scheduling framework for RAG-enhanced LLMs in collaborative edge environments. Uses online PPO-based query identification, dynamic inter-node workload balancing, and intra-node resource optimization to handle privacy constraints and heterogeneity. Achieves 4.23% to 91.39% performance gains over baselines.
* `training` `networking` [An Efficient Gradient-Aware Error-Bounded Lossy Compressor for Federated Learning](http://arxiv.org/abs/2511.05770v1)
  > **TL;DR**: Proposes an error-bounded lossy compressor for federated learning gradients to reduce communication overhead. The method exploits temporal correlations and convolutional kernel structures via two predictors: magnitude and sign predictors. Achieves up to 1.53x higher compression ratio than SZ3 and reduces communication time by 76.1%-96.2%.
* `edge` `offline` `hardware` [Optimal Multi-Constrained Workflow Scheduling for Cyber-Physical Systems in the Edge-Cloud Continuum](http://arxiv.org/abs/2511.07466v1)
  > **TL;DR**: Proposes an optimal multi-constrained workflow scheduler for edge-cloud continuum CPS to minimize latency. Uses continuous-time MILP formulation with heterogeneous multicore and capability constraints. Achieves 13.54% average latency reduction vs. enhanced heuristic in real-world use case.
* `kernel` `hardware` [GPU Under Pressure: Estimating Application's Stress via Telemetry and Performance Counters](http://arxiv.org/abs/2511.05067v1)
  > **TL;DR**: Studies estimating GPU stress under sustained workloads using telemetry and performance counters. Combines telemetry data with specific counters (throughput, instructions issued, stalls) to assess stress. Demonstrates effectiveness for CNN workloads, aiming to predict reliability and aging effects.
* `video` `edge` `hardware` [Accelerating HDC-CNN Hybrid Models Using Custom Instructions on RISC-V GPUs](http://arxiv.org/abs/2511.05053v1)
  > **TL;DR**: Explores optimizing hybrid HDC-CNN models for energy-efficient ML. Designs custom instructions on RISC-V GPUs for accelerated processing. Achieves 56.2x speedup in microbenchmarks.
* `edge` `serving` `quantization` [Characterizing and Understanding Energy Footprint and Efficiency of Small Language Model on Edges](http://arxiv.org/abs/2511.11624v1)
  > **TL;DR**: This paper analyzes the power efficiency of small language models on edge devices, comparing CPU and GPU configurations. It evaluates energy-to-performance ratios across Raspberry Pi, Jetson Nano, and Jetson Orin platforms. Jetson Orin Nano with GPU achieves highest efficiency, with Llama 3.2 offering best accuracy-power balance.
* `kernel` `hardware` `networking` [Marionette: Data Structure Description and Management for Heterogeneous Computing](http://arxiv.org/abs/2511.04853v1)
  > **TL;DR**: Proposes Marionette, a C++17 library for flexible and efficient data structure management in heterogeneous computing. It decouples data layout from interface description, supports multiple memory strategies, and enables efficient data transfers across devices. Achieves minimal runtime overhead, demonstrated in a CUDA case study.
* `sparse` `quantization` `kernel` [Enabling Dynamic Sparsity in Quantized LLM Inference](http://arxiv.org/abs/2511.04477v1)
  > **TL;DR**: Explores enabling dynamic sparsity in quantized LLM inference. Proposes a zigzag-patterned quantization layout, specialized GEMV kernel, and sparse index gathering. Achieves up to 1.55x faster decoding throughput on commodity GPUs without accuracy loss.
* `serving` `offloading` `kernel` [AIvailable: A Software-Defined Architecture for LLM-as-a-Service on Heterogeneous and Legacy GPUs](http://arxiv.org/abs/2511.11621v1)
  > **TL;DR**: Proposes AIvailable, a software-defined LLM-as-a-Service platform for heterogeneous GPUs. Uses VRAM-aware dynamic allocation to run GPU-accelerated inference without CPU fallbacks on mixed NVIDIA/AMD nodes. Achieves efficient resource utilization by repurposing legacy GPUs for scalable serving.
* `training` `sparse` `networking` [Parallel Spawning Strategies for Dynamic-Aware MPI Applications](http://arxiv.org/abs/2511.04268v1)
  > **TL;DR**: Addresses the high cost of reconfiguration for malleable MPI applications. Proposes a cooperative parallel spawning strategy that reuses original processes and fully releases unneeded nodes during shrinkage. Reduces shrink operation cost by at least 20× while maintaining expansion overhead below 1.25×.
* `edge` `scaling` [Stochastic Modeling for Energy-Efficient Edge Infrastructure](http://arxiv.org/abs/2511.03941v1)
  > **TL;DR**: Addresses energy efficiency in edge computing via AI-driven predictive power scaling. Uses Markov Chains for modeling power state transitions and optimizes workload distribution to minimize unnecessary transitions. Reduces energy consumption disparities by 30% and improves system responsiveness.
* `offloading` `serving` `MoE` [AnchorTP: Resilient LLM Inference with State-Preserving Elastic Tensor Parallelism](http://arxiv.org/abs/2511.11617v1)
  > **TL;DR**: Proposes AnchorTP, a state-preserving elastic tensor parallelism framework for resilient LLM inference. Preserves model parameters and KV caches via a daemon and minimizes recovery time with a bandwidth-aware planner and pipelined transfers. Reduces Time to First Success by 11x and Time to Peak by 59%.
* `edge` `serving` `offloading` [UMDAM: A Unified Data Layout and DRAM Address Mapping for Heterogenous NPU-PIM](http://arxiv.org/abs/2511.03293v2)
  > **TL;DR**: Addresses memory bottlenecks in edge LLM inference with NPU-PIM co-execution. Proposes UMDAM, a unified data layout and DRAM mapping scheme optimized for compute and PIM efficiency. Achieves 3.0× TTFT reduction and 2.18× TTLT improvement for OPT models.
* `serving` `offloading` `kernel` [SnapStream: Efficient Long Sequence Decoding on Dataflow Accelerators](http://arxiv.org/abs/2511.03092v5)
  > **TL;DR**: Proposes SnapStream, a KV cache compression method for efficient long-sequence LLM inference on static-graph accelerators. Integrates sparse attention with continuous batching to reduce on-chip memory by 4× while maintaining accuracy, achieving up to 1832 tokens/sec with 128k context on real hardware deployment.
* `training` `hardware` `kernel` [Implementing Multi-GPU Scientific Computing Miniapps Across Performance Portable Frameworks](http://arxiv.org/abs/2511.02655v1)
  > **TL;DR**: Evaluates performance portability frameworks (Kokkos, OpenMP, RAJA, OCCA) for multi-GPU scientific computing miniapps. Measures time-to-solution on NVIDIA A100 GPUs for N-body and structured grid simulations. Shows OCCA achieves faster execution for small problems but lacks optimized reductions.
* `edge` `serving` `networking` [Federated Attention: A Distributed Paradigm for Collaborative LLM Inference over Edge Networks](http://arxiv.org/abs/2511.02647v1)
  > **TL;DR**: Proposes Federated Attention (FedAttn), a distributed LLM inference framework for edge networks that exchanges aggregated KV matrices across Transformer blocks with privacy. Reduces communication by 3.2× and latency by 1.7× while maintaining model quality via sparse attention and adaptive aggregation.
* `serving` `kernel` `offloading` [From Models to Operators: Rethinking Autoscaling Granularity for Large Generative Models](http://arxiv.org/abs/2511.02248v1)
  > **TL;DR**: Proposes operator-level autoscaling for generative models by treating models as graphs of heterogeneous operators. Optimizes scaling, batching, and placement per operator. Achieves 40% fewer GPUs and 1.6x higher throughput under fixed resources while preserving SLOs.
* `hardware` `inference` `edge` [Beyond the GPU: The Strategic Role of FPGAs in the Next Wave of AI](http://arxiv.org/abs/2511.11614v1)
  > **TL;DR**: Explores FPGA-based acceleration for AI to address GPU limitations in latency and energy efficiency. Advocates for direct algorithm-to-logic mapping with parallel pipelines, deterministic timing, and local inference near sensors. Achieves up to 40% lower power consumption compared to GPUs.
* `training` `kernel` `serving` [Eliminating Multi-GPU Performance Taxes: A Systems Approach to Efficient Distributed LLMs](http://arxiv.org/abs/2511.02168v1)
  > **TL;DR**: Addresses inefficiencies in distributed GPU execution for LLM workloads. Moves beyond bulk synchronous model with fine-grained programming patterns using in-kernel communication to create tile-level pipelines. Achieves 10-20% end-to-end latency speedup.
* `training` `kernel` `hardware` [Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects](http://arxiv.org/abs/2511.02132v1)
  > **TL;DR**: Investigates how NUMA effects degrade attention performance on modern GPUs. Proposes Swizzled Head-first Mapping, a scheduling strategy that aligns attention heads with NUMA domains to enhance cache reuse. Achieves up to 50% higher performance and 80-97% L2 cache hit rates on AMD MI300X.
* `serving` `offloading` `edge` [Why Should the Server Do It All?: A Scalable, Versatile, and Model-Agnostic Framework for Server-Light DNN Inference over Massively Distributed Clients via Training-Free Intermediate Feature Compression](http://arxiv.org/abs/2511.11608v1)
  > **TL;DR**: Proposes SLICER, a training-free framework to compress intermediate features for distributed DNN inference, reducing communication and server load. Uses asymmetric top-K filtering, magnitude-splitting, and adaptive quantization to achieve up to 10x lower uplink volume and 4.4x server GPU time reduction.
* `edge` [Boosting performance of computer vision applications through embedded GPUs on the edge](http://arxiv.org/abs/2511.01129v1)
  > **TL;DR**: Proposes using embedded GPUs on edge devices to boost computer vision application performance. Designs a system offloading intensive tasks to embedded GPUs for efficient execution. Achieves performance gains over CPU-only, improving user experience through faster processing.
* `offloading` `edge` `sparse` [Neuro-Inspired Task Offloading in Edge-IoT Networks Using Spiking Neural Networks](http://arxiv.org/abs/2511.01127v1)
  > **TL;DR**: Explores energy-efficient task offloading in edge-IoT networks. Uses spiking neural networks (SNNs) for real-time, adaptive task orchestration. Achieves 32% lower energy consumption and 25% higher success rate versus baselines under high load.
* `serving` `energy` [FREESH: Fair, Resource- and Energy-Efficient Scheduling for LLM Serving on Heterogeneous GPUs](http://arxiv.org/abs/2511.00807v2)
  > **TL;DR**: Proposes FREESH for energy-efficient LLM serving on heterogeneous GPUs by jointly optimizing query routing, scheduling, and GPU frequency scaling. Matches GPU power-throughput characteristics with query workloads to reduce carbon and energy. Achieves 28.6% energy reduction and 45.45% emissions reduction while improving SLO attainment.
* `RL` `training` `offloading` [AReaL-Hex: Accommodating Asynchronous RL Training over Heterogeneous GPUs](http://arxiv.org/abs/2511.00796v1)
  > **TL;DR**: Aims to maximize training throughput and cost efficiency for reinforcement learning (RL) with LLMs over heterogeneous GPUs. Proposes AReaL-Hex, a heterogeneity-aware scheduler with two-phase optimization (MILP-based assignment and graph partitioning) to balance stages. Achieves up to 1.50x higher throughput or 1.46x cost reduction.
* `edge` `serving` `offloading` [Toward Sustainability-Aware LLM Inference on Edge Clusters](http://arxiv.org/abs/2512.04088v1)
  > **TL;DR**: Explores carbon- and latency-aware routing for LLM inference on edge clusters. Proposes strategies that balance latency and carbon footprint by routing prompts to specific hardware based on benchmarking. Achieves optimal batch size of 4 for throughput and energy efficiency without memory saturation.
* `serving` `edge` `scheduling` [EPARA: Parallelizing Categorized AI Inference in Edge Clouds](http://arxiv.org/abs/2511.00603v1)
  > **TL;DR**: Proposes EPARA, an end-to-edge parallel inference framework for edge clouds. It categorizes tasks by latency/GPU sensitivity, using parallelism allocator, request handler, and state-aware scheduler. Achieves 2.1× higher goodput in production workloads versus prior frameworks.
* `offloading` `edge` `serving` [Gaia: Hybrid Hardware Acceleration for Serverless AI in the 3D Compute Continuum](http://arxiv.org/abs/2511.13728v1)
  > **TL;DR**: Proposes Gaia, a GPU-as-a-service architecture for serverless AI. Dynamically switches between CPU/GPU backends based on workload and SLOs via execution modes and runtime adjustments. Achieves up to 95% reduction in end-to-end latency.
* `training` `MoE` `multi-modal` [LongCat-Flash-Omni Technical Report](http://arxiv.org/abs/2511.00279v2)
  > **TL;DR**: Addresses efficient large-scale omni-modal model training. Proposes modality-decoupled parallelism and efficient MoE architecture with progressive curriculum. Achieves over 90% of text-only training throughput for a 560B parameter model.
* `networking` `serving` [Fix: externalizing network I/O in serverless computing](http://arxiv.org/abs/2511.00205v1)
  > **TL;DR**: Proposes a serverless computing system that externalizes network I/O, enabling the platform to manage data transfers. Uses deterministic procedures with precise data dependencies to optimize scheduling and reduce starvation. Shifts billing model to pay-for-results with improved efficiency.
* `networking` `training` `offloading` [RDMA Point-to-Point Communication for LLM Systems](http://arxiv.org/abs/2510.27656v1)
  > **TL;DR**: Addresses inflexible point-to-point communication in LLM systems requiring support for disaggregated inference, MoE, and RL fine-tuning. Introduces TransferEngine, a uniform NIC interface using one-sided WriteImm operations. Achieves 400 Gbps throughput and 1.3-second RL updates for trillion-parameter models.
* `kernel` `training` [ML-Based Optimum Sub-system Size Heuristic for the GPU Implementation of the Tridiagonal Partition Method](http://arxiv.org/abs/2510.27351v1)
  > **TL;DR**: Proposes a ML-based heuristic to find optimal sub-system size for CUDA implementation of parallel partition algorithms. Uses kNN to predict optimal sizes for solving systems of linear equations, reducing empirical tuning. Achieves acceptably good predictions validated against actual data.
* `training` `kernel` [Synergistic Tensor and Pipeline Parallelism](http://arxiv.org/abs/2510.27257v1)
  > **TL;DR**: Proposes a synergistic tensor-pipeline parallel schedule to reduce communication and synchronization overheads in LLM/MLLM training. Decouples forward/backward passes into fine-grained units braided into a composite sequence. Achieves 12-16% training throughput improvement over baselines.
* `serving` `offloading` [SERFLOW: A Cross-Service Cost Optimization Framework for SLO-Aware Dynamic ML Inference](http://arxiv.org/abs/2510.27182v1)
  > **TL;DR**: Addresses how to optimize cloud costs for dynamic ML inference requests with variable early exits. Proposes SERFLOW, which leverages FaaS and stage-specific resource provisioning with adaptive load balancing. Reduces cloud costs by over 23% while adapting to dynamic workloads.
* `serving` `training` [Glia: A Human-Inspired AI for Automated Systems Design and Optimization](http://arxiv.org/abs/2510.27176v3)
  > **TL;DR**: Proposes Glia, an AI system using multi-agent LLMs to autonomously design computer systems. It generates interpretable algorithms for distributed LLM inference, including routing, scheduling, and autoscaling. Achieves human-expert performance levels with significant time reduction.
* `hardware` `inference` `training` [Mind the Gap: Revealing Inconsistencies Across Heterogeneous AI Accelerators](http://arxiv.org/abs/2511.11601v1)
  > **TL;DR**: Investigates inconsistencies in ML model execution across heterogeneous AI accelerators. Uses automated pipeline to test 100,000+ model variants on five accelerators. Finds newer platforms support 17% fewer operators and exhibit >5% output discrepancies due to implementation differences.
* `training` `offline` `RL` [FlowMesh: A Service Fabric for Composable LLM Workflows](http://arxiv.org/abs/2510.26913v1)
  > **TL;DR**: Proposes FlowMesh, a service fabric for composable LLM workflows like RLHF and agentic pipelines. Uses fine-grained operators with lineage for deduplication and batching on heterogeneous GPUs via a global scheduler. Achieves 3.8x cost reduction and 2.0x lower energy usage.
* `MoE` `serving` `offloading` [ExpertFlow: Adaptive Expert Scheduling and Memory Coordination for Efficient MoE Inference](http://arxiv.org/abs/2510.26730v1)
  > **TL;DR**: Proposes ExpertFlow, a runtime system for MoE inference that adaptively prefetches experts using runtime statistics and hybrid cross-layer prediction. Reduces model stall time to less than 0.1% of baseline by decreasing cache misses via coordinated memory management and expert scheduling.
* `sparse` `training` `networking` [An All-Reduce Compatible Top-K Compressor for Communication-Efficient Distributed Learning](http://arxiv.org/abs/2510.26709v3)
  > **TL;DR**: Addresses communication bottlenecks in distributed training via an All-Reduce compatible Top-K compressor that aligns sparsity patterns using gradient sketches. Combines contraction property with efficient index-free All-Reduce. Achieves up to 60.7% reduction in wall-clock training time.
* `RL` `training` `offline` [ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems](http://arxiv.org/abs/2510.26475v1)
  > **TL;DR**: Proposes ReSpec to address three gaps limiting speculative decoding in RL-based LLM training: diminishing speedup at scale, drafter staleness, and policy degradation. Combines dynamic SD tuning, drafter evolution via distillation, and reward-weighted updates. Achieves up to 4.5x speedup on Qwen models while preserving reward convergence.
* `training` `networking` `storage` [Detecting Anomalies in Machine Learning Infrastructure via Hardware Telemetry](http://arxiv.org/abs/2510.26008v2)
  > **TL;DR**: Proposes Reveal, an unsupervised pipeline that uses hardware telemetry to detect anomalies in ML infrastructure without workload knowledge. Achieves 5.97% acceleration for DeepSeek model by identifying network and system configuration issues.
* `quantization` `serving` `kernel` [Opt4GPTQ: Co-Optimizing Memory and Computation for 4-bit GPTQ Quantized LLM Inference on Heterogeneous Platforms](http://arxiv.org/abs/2511.19438v1)
  > **TL;DR**: Optimizes memory and computation for 4-bit GPTQ quantized LLM inference on heterogeneous platforms. Proposes SMB-Opt, VML-Opt, and ILA-Opt in vLLM. Achieves 84.42% higher throughput and 51.35% latency reduction.
* `MoE` `hardware` `offlining` [MoEntwine: Unleashing the Potential of Wafer-scale Chips for Large-scale Expert Parallel Inference](http://arxiv.org/abs/2510.25258v1)
  > **TL;DR**: Addresses imbalanced communication and expert migration overhead in wafer-scale chips for mixture-of-expert parallelism. Proposes entwined ring mapping and non-invasive balancer to optimize layer mapping and split migration steps. Achieves up to 62% communication reduction and 54% faster MoE computation.
* `training` `networking` `quantization` [Machine Learning and CPU (Central Processing Unit) Scheduling Co-Optimization over a Network of Computing Centers](http://arxiv.org/abs/2510.25176v1)
  > **TL;DR**: Co-optimizes CPU resource allocation and distributed ML training over networked computing nodes. Combines local training with consensus-based resource scheduling and log-quantized communication. Achieves over 50% better cost optimality than existing schedulers.
* `training` `kernel` `sparse` [Multi-Resolution Model Fusion for Accelerating the Convolutional Neural Network Training](http://arxiv.org/abs/2510.25170v1)
  > **TL;DR**: Introduces Multi-Resolution Model Fusion (MRMF) to accelerate CNN training time. Trains initial models on reduced-resolution data, then fuses and refines them with original-resolution data. Achieves up to 47% training time reduction while maintaining accuracy on CosmoFlow and Neuron Inverter benchmarks.
* `edge` `training` `serving` [AeroResQ: Edge-Accelerated UAV Framework for Scalable, Resilient and Collaborative Escape Route Planning in Wildfire Scenarios](http://arxiv.org/abs/2511.00038v1)
  > **TL;DR**: Proposes AeroResQ, an edge-accelerated UAV framework for real-time escape route planning in wildfires. Uses multi-layer orchestration with service drones running DNNs on edge accelerators and coordinator drones for path planning. Achieves end-to-end latency <=500 ms and over 98% task reassignment success.
* `edge` `serving` `networking` [Bayes-Split-Edge: Bayesian Optimization for Constrained Collaborative Inference in Wireless Edge Systems](http://arxiv.org/abs/2510.23503v1)
  > **TL;DR**: Proposes Bayes-Split-Edge, a Bayesian optimization framework for constrained collaborative inference between edge devices and servers. Jointly optimizes transmission power and neural network split point to minimize energy/delay costs. Achieves up to 2.4x cost reduction vs standard BO and near-linear convergence with max 20 evaluations.
* `edge` `offloading` [Rethinking Inference Placement for Deep Learning across Edge and Cloud Platforms: A Multi-Objective Optimization Perspective and Future Directions](http://arxiv.org/abs/2510.22909v1)
  > **TL;DR**: Surveys optimal placement of DL model partitions across edge and cloud for multi-objective optimization. Analyzes offloading techniques, model adaptations, and compression to balance latency, cost, and privacy. Highlights trade-offs between ≤40ms edge latency vs. cloud cost savings.
* `networking` `scheduling` `storage` [Learning to Schedule: A Supervised Learning Framework for Network-Aware Scheduling of Data-Intensive Workloads](http://arxiv.org/abs/2510.21419v1)
  > **TL;DR**: Addresses network-aware job scheduling for data-intensive workloads via supervised learning. Collects real-time telemetry to predict job completion times per node and ranks placements. Achieves 34-54% higher accuracy in optimal node selection versus default Kubernetes scheduler.
* `edge` `kernel` `serving` [Accelerating Mobile Inference through Fine-Grained CPU-GPU Co-Execution](http://arxiv.org/abs/2510.21081v1)
  > **TL;DR**: Reduces mobile inference latency via CPU-GPU co-execution. Introduces lightweight SVM-based synchronization and ML-based execution time prediction for task assignment. Achieves up to 1.89x speedup for linear layers on Pixel 5 smartphone.
* `training` `offloading` `storage` [xMem: A CPU-Based Approach for Accurate Estimation of GPU Memory in Deep Learning Training Workloads](http://arxiv.org/abs/2510.21048v1)
  > **TL;DR**: Proposes xMem, a CPU-only dynamic analysis framework for precise a priori estimation of peak GPU memory in deep learning training. Targets resource allocation optimizations without runtime GPU overhead. Achieves 91% lower median relative error and 75% reduction in OOM probability.
* `hardware` `training` `inference` [Lincoln AI Computing Survey (LAICS) and Trends](http://arxiv.org/abs/2510.20931v1)
  > **TL;DR**: Surveys AI accelerators for GenAI training and inference, plotting peak performance vs. power. Updates previous data with new market segment analysis and architectural categorizations. Highlights trends and provides performance/power metrics for commercial systems.
* `serving` `networking` [Morpheus: Lightweight RTT Prediction for Performance-Aware Load Balancing](http://arxiv.org/abs/2510.20506v1)
  > **TL;DR**: Proposes lightweight RTT predictors using time-series cluster metrics for performance-aware load balancing in distributed systems. Achieves up to 95% prediction accuracy, reducing tail latency and resource waste in Kubernetes-managed GPU clusters via proactive routing optimization.
* `serving` `offloading` [FLAS: a combination of proactive and reactive auto-scaling architecture for distributed services](http://arxiv.org/abs/2510.20388v1)
  > **TL;DR**: Proposes FLAS, an auto-scaler combining proactive forecasting of SLA trends with reactive estimation from resource metrics, enabling adaptive resource scaling. Ensures SLA compliance for distributed services with >99% adherence during worst-case scenarios.
* `training` `networking` `kernel` [Collective Communication for 100k+ GPUs](http://arxiv.org/abs/2510.20171v3)
  > **TL;DR**: Addresses collective communication bottlenecks in LLM training at 100k+ GPU scale. Proposes NCCLX framework optimizing throughput and latency for synchronous training and inference workloads. Achieves substantial communication efficiency gains on Llama4 model.
* `training` `MoE` `networking` [AsyncHZP: Hierarchical ZeRO Parallelism with Asynchronous Scheduling for Scalable LLM Training](http://arxiv.org/abs/2510.20111v1)
  > **TL;DR**: Proposes AsyncHZP, an asynchronous hierarchical ZeRO parallelism method for scalable LLM training, featuring adaptive resharding and multi-stream scheduling to reduce communication overhead. Achieves state-of-the-art performance without complex tuning, validated on Dense and MoE models.
* `offline` `serving` `quantization` [Serverless GPU Architecture for Enterprise HR Analytics: A Production-Scale BDaaS Implementation](http://arxiv.org/abs/2510.19689v1)
  > **TL;DR**: Proposes a serverless GPU architecture for low-latency, cost-efficient TabNet inference in regulated HR analytics. Integrates serverless GPU runtime with interpretability for compliance. Achieves up to 4.5× higher throughput and 98× lower latency vs. Spark at 90% lower cost per 1K inferences.
* `MoE` `training` `networking` [HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid Expert/Data Transmission](http://arxiv.org/abs/2510.19470v1)
  > **TL;DR**: Proposes HybridEP, a framework optimizing Mixture-of-Experts training for cross-datacenter scalability under constrained bandwidth. Dynamically transforms expert placement via domain-based partition and parameter-efficient migration, guided by stream-based modeling. Achieves up to 5.6x speedup over existing systems.
* `networking` `training` [Enabling Reconfiguration-Communication Overlap for Collective Communication in Optical Networks](http://arxiv.org/abs/2510.19322v1)
  > **TL;DR**: Addresses inefficiency in optical networks for distributed ML training collectives. Proposes SWOT framework with intra-collective reconfiguration, overlapping switch reconfigurations with transmissions. Achieves performance improvements in simulations.
* `training` `MoE` `networking` [RailS: Load Balancing for All-to-All Communication in Distributed Mixture-of-Experts Training](http://arxiv.org/abs/2510.19262v2)
  > **TL;DR**: Addresses imbalanced all-to-all communication overhead in MoE training. Proposes RailS, a load-balancing framework leveraging rail topology symmetry with local LPT-based scheduling and multipath transmission. Improves bus bandwidth by 20%–78% and reduces iteration time by 18%–40% for Mixtral workloads.
* `RL` `training` `offloading` [RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs](http://arxiv.org/abs/2510.19225v2)
  > **TL;DR**: Explores cost-efficient RL training for LLMs by harvesting preemptible GPU resources. Proposes RLBoost, a hybrid architecture with adaptive offload, pull-based weight transfer, and token-level response migration. Achieves 1.51x-1.97x training throughput and 28%-49% cost savings over on-demand resources.
* `hardware` `training` [Learned Cost Model for Placement on Reconfigurable Dataflow Hardware](http://arxiv.org/abs/2511.01872v1)
  > **TL;DR**: Proposes a learned cost model to predict throughput for mapping ML dataflow graphs onto reconfigurable hardware, improving accuracy by 31-52% and achieving 5.6% faster compiled graphs compared to analytical models.
* `training` `sparse` [MTraining: Distributed Dynamic Sparse Attention for Efficient Ultra-Long Context Training](http://arxiv.org/abs/2510.18830v1)
  > **TL;DR**: Addresses computational imbalance in distributed training of LLMs with ultra-long contexts via dynamic sparse attention. Introduces MTraining with balanced and hierarchical sparse ring attention to reduce overhead. Achieves 6x higher training throughput while maintaining accuracy on Qwen2.5-3B.
* `serving` `offloading` `RAG` [Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications](http://arxiv.org/abs/2510.18586v2)
  > **TL;DR**: Addresses KV cache inefficiency in multi-agent LLM serving with external tool calls. Proposes Tokencake framework with agent-aware dynamic memory partitioning and proactive offload/upload mechanisms. Achieves 47.06% end-to-end latency reduction and 16.9% better GPU memory utilization vs vLLM.
* `serving` `edge` `offline` [SLICE: SLO-Driven Scheduling for LLM Inference on Edge Computing Devices](http://arxiv.org/abs/2510.18544v3)
  > **TL;DR**: Proposes SLICE, a scheduling system for LLM inference on edge devices with diverse SLO constraints. Combines utility-based scheduling and dynamic generation rate control to improve SLO attainment. Achieves 35x higher SLO attainment and 3.4x faster task completion vs. Orca/FastServe.
* `edge` `serving` `thinking` [EdgeReasoning: Characterizing Reasoning LLM Deployment on Edge GPUs](http://arxiv.org/abs/2511.01866v1)
  > **TL;DR**: Characterizes deployment of reasoning LLMs on edge GPUs under latency constraints. Evaluates model sizes, token reduction techniques, and parallelism strategies to balance accuracy-latency. Achieves optimized configurations mapping Pareto frontiers, improving edge deployment efficiency by up to 4x.
* `training` `kernel` [Efficient Long-context Language Model Training by Core Attention Disaggregation](http://arxiv.org/abs/2510.18121v1)
  > **TL;DR**: Addresses load imbalance in long-context LLM training by disaggregating core attention computation. Proposes Core Attention Disaggregation (CAD) that schedules stateless attention tasks on dedicated devices using dynamic batching. Achieves up to 1.35x throughput improvement on 512 H200 GPUs with 512k context lengths.
* `serving` `storage` `training` [AI for Distributed Systems Design: Scalable Cloud Optimization Through Repeated LLMs Sampling And Simulators](http://arxiv.org/abs/2510.18897v1)
  > **TL;DR**: Develops AI-driven distributed scheduler design via LLM-generated policies verified in a domain-specific simulator. Proposes iterative generate-verify loop using simulator (Eudoxia) and feedback to optimize policies. Achieves throughput improvements across models in Function-as-a-Service runtime.
* `offloading` `edge` `serving` [DynaKV: Enabling Accurate and Efficient Long-Sequence LLM Decoding on Smartphones](http://arxiv.org/abs/2511.07427v1)
  > **TL;DR**: Proposes DynaKV, an adaptive key-value cache management system for efficient long-sequence LLM decoding on smartphones. Uses migration-free cluster adaptation, continuity-centric flash layout, and virtualized cache to address KVCache distribution shifts. Achieves 1.47x speedups and 1.38x accuracy gains over state-of-the-art.
* `agentic` `RL` `offline` [Network and Systems Performance Characterization of MCP-Enabled LLM Agents](http://arxiv.org/abs/2511.07426v1)
  > **TL;DR**: Characterizes performance and cost of MCP-enabled LLM agents. Measures token efficiency, cost, time and success rate across different models and MCP configurations. Achieves optimizations via parallel tool calls and task abort for cost-effective workflows.
* `training` `kernel` `thinking` [Integrating Performance Tools in Model Reasoning for GPU Kernel Optimization](http://arxiv.org/abs/2510.17158v1)
  > **TL;DR**: Proposes training LLMs to interact with performance tools during reasoning for GPU kernel optimization. Introduces a method integrating hardware/environment data via chain-of-thought reasoning. Achieves state-of-the-art results in optimizing GPU kernels, though specific metrics not detailed in abstract.
* `edge` `quantization` `offloading` [An Evaluation of LLMs Inference on Popular Single-board Computers](http://arxiv.org/abs/2511.07425v1)
  > **TL;DR**: Evaluates LLM inference performance on single-board computers (SBCs) for edge deployment. Benchmarks 25 quantized LLMs using Ollama/Llamafile runtimes on three SBCs, measuring throughput, memory, and power. Achieves 4x higher throughput and 30-40% lower power usage with Llamafile.
* `serving` `RL` [Justitia: Fair and Efficient Scheduling for LLM Applications](http://arxiv.org/abs/2510.17015v1)
  > **TL;DR**: Proposes Justitia, a fair and efficient scheduler for LLM applications to reduce head-of-line blocking and resource allocation issues. Uses memory-centric cost modeling, neural demand prediction, and virtual-time fair queuing. Achieves improved scheduling efficiency while preserving fairness.
* `networking` [Host-Side Telemetry for Performance Diagnosis in Cloud and HPC GPU Infrastructure](http://arxiv.org/abs/2510.16946v1)
  > **TL;DR**: Presents an eBPF-based telemetry system for diagnosing GPU tail latency in cloud/HPC. Correlates host-side metrics with GPU events for unified observability, achieving 81-88% diagnostic accuracy and detecting spikes in under 5 seconds with 1.21% CPU overhead at 100Hz.
* `serving` `edge` `hardware` [Exact Nearest-Neighbor Search on Energy-Efficient FPGA Devices](http://arxiv.org/abs/2510.16736v1)
  > **TL;DR**: Investigates energy-efficient exact kNN search for high-dimensional latent spaces using FPGA configurations. Proposes two FPGA solutions: one maximizing throughput via batched stream processing, another minimizing latency with in-memory processing. Achieves 16.6× higher throughput and 11.9× energy savings versus CPUs.
* `serving` `offline` [CodeCRDT: Observation-Driven Coordination for Multi-Agent LLM Code Generation](http://arxiv.org/abs/2510.18893v1)
  > **TL;DR**: Proposes CodeCRDT, an observation-driven coordination using CRDTs for multi-agent LLM code generation to avoid explicit messaging. Enables lock-free concurrent code generation with 100% convergence, achieving up to 21.1% speedup but also up to 39.4% slowdown depending on task structure.
* `training` `networking` `sparse` [Reimagining RDMA Through the Lens of ML](http://arxiv.org/abs/2510.16606v1)
  > **TL;DR**: Addresses tail latency in collective communication for distributed ML training. Proposes Celeris, a domain-specific RDMA transport that removes retransmissions and in-order delivery, exploiting ML's tolerance for partial data. Achieves up to 2.3x lower 99th-percentile latency and doubles NIC resilience.
* `edge` `serving` `offloading` [Edge-Based Speech Transcription and Synthesis for Kinyarwanda and Swahili Languages](http://arxiv.org/abs/2510.16497v1)
  > **TL;DR**: Proposes an edge-cloud cascading framework to optimize speech transcription/synthesis for under-resourced languages using Whisper and SpeechT5. Distributes inference workload between edge and cloud to reduce latency and memory usage. Achieves 9.5-14% model compression on edge devices with memory usage capped at 149MB.
* `serving` `offloading` `edge` [FourierCompress: Layer-Aware Spectral Activation Compression for Efficient and Accurate Collaborative LLM Inference](http://arxiv.org/abs/2510.16418v1)
  > **TL;DR**: Proposes FourierCompress, a layer-aware spectral activation compression method for collaborative LLM inference between edge devices and servers. Uses FFT to retain low-frequency coefficients, reducing activation size by 7.6x with under 0.3% accuracy loss and 32x faster compression than Top-k.
* `training` `offloading` `edge` [MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization](http://arxiv.org/abs/2510.16415v1)
  > **TL;DR**: Proposes MeCeFO, a fault-tolerant optimization for distributed LLM training with minimal overhead during node failures. It uses skip-connection, recomputation, and low-rank gradient approximation to offload tasks to neighboring nodes. Achieves only 4.18% throughput drop under high failure rates, 5.0–6.7× better resilience than SOTA.
* `serving` `offline` `networking` [Enhancing reliability in AI inference services: An empirical study on real production incidents](http://arxiv.org/abs/2511.07424v1)
  > **TL;DR**: Presents empirical analysis of production incidents in hyperscale LLM inference services. Develops taxonomy and methodology using real incident data, identifying failure modes and mitigation strategies (e.g., GPU-aware routing). Achieves high labeling consistency (Cohen's K ~0.89) and reduces incident impact.
* `hardware` `serving` `storage` [Funky: Cloud-Native FPGA Virtualization and Orchestration](http://arxiv.org/abs/2510.15755v1)
  > **TL;DR**: Proposes Funky, a cloud-native FPGA orchestration engine with virtualization, state management, and FPGA-aware components to enhance performance and utilization. Achieves only 7.4% performance overhead vs. native execution while enabling strong isolation and efficient scheduling in distributed FPGAs.
* `serving` `training` `hardware` [GOGH: Correlation-Guided Orchestration of GPUs in Heterogeneous Clusters](http://arxiv.org/abs/2510.15652v1)
  > **TL;DR**: Proposes GOGH, a learning-based resource allocator for ML workloads in heterogeneous GPU clusters. Uses two NNs: one for initial performance/co-location prediction and one for online refinement. Minimizes energy while meeting SLOs, achieving 30% lower energy use and improved SLO compliance.
* `training` `kernel` `sparse` [PRISM: Probabilistic Runtime Insights and Scalable Performance Modeling for Large-Scale Distributed Training](http://arxiv.org/abs/2510.15596v1)
  > **TL;DR**: Proposes PRISM, a probabilistic performance modeling framework for large-scale distributed training that accounts for runtime variations. It uses statistical methods to provide guarantees on training time, analyzes sensitivity to hardware variation, and identifies critical kernels (e.g., AllGather, ReduceScatter) for optimization. Achieves 1.26x potential performance improvement.
* `serving` `offloading` [BeLLMan: Controlling LLM Congestion](http://arxiv.org/abs/2510.15330v1)
  > **TL;DR**: Addresses how to control LLM serving congestion to avoid latency inflation. Proposes beLLMan, a controller that signals applications to adjust output length based on system load. Achieves 8× lower latency and 25% energy reduction while serving 19% more requests.
* `serving` `offloading` `edge` [Synera: Synergistic LLM Serving across Device and Cloud at Scale](http://arxiv.org/abs/2511.07423v1)
  > **TL;DR**: Proposes Synera, a device-cloud synergistic LLM serving system with selective offloading, parallel inference, and scalable batching to balance quality and latency. Achieves 1.20-5.47x better generation quality than baselines with on-par latency and 8.2-16.5% lower cloud costs.
* `kernel` `training` `offline` [Hive Hash Table: A Warp-Cooperative, Dynamically Resizable Hash Table for GPUs](http://arxiv.org/abs/2510.15095v1)
  > **TL;DR**: Introduces Hive, a high-performance GPU hash table with warp-cooperative protocols and dynamic resizing for concurrent updates. Proposes cache-aligned buckets, warp-synchronous concurrency, and load-aware resizing. Achieves 3.5 billion updates/s and 2x higher throughput than state-of-the-art GPU hash tables.
* `video` `multi-modal` `edge` [Multi-modal video data-pipelines for machine learning with minimal human supervision](http://arxiv.org/abs/2510.14862v1)
  > **TL;DR**: Proposes a multi-modal video data-pipeline requiring minimal human supervision, using pre-trained experts and PHG-MAE for efficient modeling. Deploys a distilled model (<1M params) for real-time semantic segmentation on commodity hardware, achieving competitive results against larger models.
* `serving` `multi-modal` `offline` [xLLM Technical Report](http://arxiv.org/abs/2510.14686v1)
  > **TL;DR**: Introduces xLLM, a decoupled inference framework with service and engine layers for efficient serving. Features intelligent scheduling for multimodal inputs, workload-adaptive disaggregation policies, and engine optimizations including execution pipelines and memory management. Achieves up to 2.2x higher throughput than baselines.
* `hardware` `training` `networking` [MPI-over-CXL: Enhancing Communication Efficiency in Distributed HPC Systems](http://arxiv.org/abs/2510.14622v1)
  > **TL;DR**: Proposes MPI-over-CXL, a communication paradigm using CXL's shared memory for direct pointer-based access in MPI, eliminating data copying. Reduces latency and bandwidth usage in distributed HPC systems. Achieves substantial performance gains over traditional MPI implementations.
* `training` `networking` `hardware` [ScalePool: Hybrid XLink-CXL Fabric for Composable Resource Disaggregation in Unified Scale-up Domains](http://arxiv.org/abs/2510.14580v1)
  > **TL;DR**: Proposes ScalePool, a hybrid XLink-CXL fabric for composable resource disaggregation in accelerator clusters. Integrates low-latency XLink for intra-cluster communication and hierarchical CXL for memory sharing and tiering. Accelerates LLM training by up to 1.84× and reduces memory latency by 4.5×.
* `serving` `offline` `quantization` [FairBatching: Fairness-Aware Batch Formation for LLM Inference](http://arxiv.org/abs/2510.14392v1)
  > **TL;DR**: Addresses unfair resource allocation in LLM inference batching. Introduces FairBatching scheduler with adaptive batch capacity and dynamic resource reclamation between prefill and decode tasks. Reduces TTFT tail latency by up to 2.29x and improves cluster capacity by 54.3%.
* `serving` `offloading` `networking` [From Attention to Disaggregation: Tracing the Evolution of LLM Inference](http://arxiv.org/abs/2511.07422v1)
  > **TL;DR**: Proposes disaggregated inference architecture for LLMs, decoupling prefill and decode phases as independently scalable components to overcome GPU cluster limitations. Mitigates resource contention, optimizing Time to First Token and Inter Token Latency while reducing cost. Achieves significant improvements in latency and throughput.
* `serving` `agentic` [Cortex: Workflow-Aware Resource Pooling and Scheduling for Agentic Serving](http://arxiv.org/abs/2510.14126v1)
  > **TL;DR**: Investigates resource pooling and scheduling for agentic workflows. Proposes Cortex with stage isolation, dedicating resource pools per workflow stage to mitigate interference and improve KV cache utilization. Increases throughput by 4.2× and reduces SLO violations by 48% compared to baseline.
* `training` `RL` `edge` [FedHFT: Efficient Federated Finetuning with Heterogeneous Edge Clients](http://arxiv.org/abs/2510.14054v1)
  > **TL;DR**: Proposes FedHFT for efficient federated fine-tuning of LLMs across heterogeneous edge clients. Uses masked adapters for resource heterogeneity and bi-level optimization with client clustering for non-iid data. Achieves up to 15% accuracy gain with 60% less communication overhead.
* `serving` `offloading` `edge` [Efficiently Executing High-throughput Lightweight LLM Inference Applications on Heterogeneous Opportunistic GPU Clusters with Pervasive Context Management](http://arxiv.org/abs/2510.14024v1)
  > **TL;DR**: Addresses high LLM startup costs and queue delays in HPC clusters with lightweight LLM inference. Proposes pervasive context management to decouple initialization from inference, retaining context on GPUs opportunistically. Achieves 72.1% execution time reduction (from 3h to 48min) and further scales to 13min using idle GPUs.
* `serving` `offline` `RAG` [FIRST: Federated Inference Resource Scheduling Toolkit for Scientific AI Model Access](http://arxiv.org/abs/2510.13724v1)
  > **TL;DR**: Proposes FIRST, a federated framework for scalable and private LLM inference on HPC clusters. Utilizes an OpenAI-compliant API with auto-scaling, parallel backend support, and hot nodes to balance batch/interactive modes. Enables billions of tokens daily on-premises.
* `serving` `offloading` [Adaptive Rescheduling in Prefill-Decode Disaggregated LLM Inference](http://arxiv.org/abs/2510.13668v1)
  > **TL;DR**: Addresses dynamic workload imbalance during LLM inference caused by output length variation. Introduces ARES with lightweight LLM-native length prediction to anticipate workloads and enable adaptive rescheduling, reducing P99 TPOT by 74.77% and achieving 2.24x higher goodput.
* `quantization` `edge` `kernel` [F-BFQ: Flexible Block Floating-Point Quantization Accelerator for LLMs](http://arxiv.org/abs/2510.13401v1)
  > **TL;DR**: Proposes F-BFQ, a flexible FPGA accelerator supporting dynamic switching between BFP quantization variants for LLM inference. Achieves 1.4x faster inference and 5.2 tokens/sec on AMD Kria compared to CPU baseline.
* `edge` `serving` `offline` [ACE-GNN: Adaptive GNN Co-Inference with System-Aware Scheduling in Dynamic Edge Environments](http://arxiv.org/abs/2511.11586v1)
  > **TL;DR**: Proposes ACE-GNN, an adaptive device-edge co-inference framework for GNNs in dynamic edge environments. Uses system-level abstraction and prediction for runtime optimization, combining pipeline and data parallelism. Achieves up to 12.7x speedup and 82.3% energy savings over prior methods.
* `serving` `offloading` `networking` [BanaServe: Unified KV Cache and Dynamic Module Migration for Balancing Disaggregated LLM Serving in AI Infrastructure](http://arxiv.org/abs/2510.13223v1)
  > **TL;DR**: Proposes BanaServe, a dynamic orchestration framework for disaggregated LLM serving. It uses layer/attention-level migration and global KV cache store to balance prefill/decode stages and eliminate cache-induced hotspots. Achieves 1.2x-3.9x higher throughput and 3.9%-78.4% lower processing time than vLLM.
* `kernel` `training` `hardware` [A GPU-resident Memory-Aware Algorithm for Accelerating Bidiagonalization of Banded Matrices](http://arxiv.org/abs/2510.12705v1)
  > **TL;DR**: Presents a GPU-optimized algorithm for bidiagonal reduction of banded matrices, crucial for SVD in scientific computing. Utilizes hardware-agnostic implementation via Julia's Array abstractions and KernelAbstractions. Achieves up to 100x speedup over CPU libraries for 32k x 32k matrices and linear performance scaling with bandwidth size.
* `training` `RL` `kernel` [Laminar: A Scalable Asynchronous RL Post-Training Framework](http://arxiv.org/abs/2510.12633v1)
  > **TL;DR**: Scales RL post-training for LLMs by decoupling weight synchronization via relay workers and dynamic trajectory repack. Achieves 5.48× training speedup and reduces convergence time on a 1024-GPU cluster.
* `training` `networking` [Metronome: Efficient Scheduling for Periodic Traffic Jobs with Network and Priority Awareness](http://arxiv.org/abs/2510.12274v1)
  > **TL;DR**: Addresses scheduling inefficiencies for distributed training jobs with periodic traffic. Proposes Metronome, a network-aware scheduler with time-division multiplexing and priority-based optimization. Reduces job completion time by up to 19.50% and improves bandwidth utilization by 23.20%.
* `training` `hardware` `quantization` [Deploying Atmospheric and Oceanic AI Models on Chinese Hardware and Framework: Migration Strategies, Performance Optimization and Analysis](http://arxiv.org/abs/2510.17852v1)
  > **TL;DR**: Develops migration framework for porting atmospheric/oceanic AI models (e.g., FourCastNet) to Chinese chips/MindSpore. Optimizes via software-hardware adaptation, memory optimization, and parallelism. Achieves preserved accuracy with improved operational efficiency and reduced dependencies.
* `serving` `offloading` [FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters](http://arxiv.org/abs/2510.11938v1)
  > **TL;DR**: Addresses efficient LLM serving in serverless clusters with dynamic pipeline adaptations. Introduces FlexPipe, with fine-grained model partitioning, live pipeline refactoring, and topology-aware resource placement. Achieves 8.5× better resource efficiency and 38.3% lower latency while reducing GPU reservations to 30%.
* `training` `serving` `offloading` [An Explorative Study on Distributed Computing Techniques in Training and Inference of Large Language Models](http://arxiv.org/abs/2510.11211v1)
  > **TL;DR**: Explores distributed computing for democratizing LLMs on consumer hardware and comparing serving techniques. Proposes metaheuristic-based modification for resource optimization and evaluates three serving systems. Achieves efficient resource utilization enabling LLMs on limited hardware.
* `edge` `serving` `networking` [A Decentralized Microservice Scheduling Approach Using Service Mesh in Cloud-Edge Systems](http://arxiv.org/abs/2510.11189v1)
  > **TL;DR**: Proposes a decentralized microservice scheduling approach using service mesh sidecar proxies for cloud-edge systems. Embeds lightweight autonomous schedulers in sidecars to avoid centralized control, leveraging service mesh for distributed traffic management. Initial results show improved scalability in response time and latency under varying request rates.
* `training` `hardware` `storage` [Improving AI Efficiency in Data Centres by Power Dynamic Response](http://arxiv.org/abs/2510.11119v1)
  > **TL;DR**: Proposes dynamic power management for AI data centres to improve efficiency. Introduces active/passive solutions where input power adapts to computing demand. Quantifies 12-15% energy savings and capital expenditure reduction in global hyperscalers.
* `training` [DCP: Addressing Input Dynamism In Long-Context Training via Dynamic Context Parallelism](http://arxiv.org/abs/2510.10620v1)
  > **TL;DR**: Addresses inefficiency of static context parallelism for long-context training with variable sequence lengths. Proposes DCP, a dynamic framework using block-wise partitioning for adaptive computation mapping. Accelerates attention layers by up to 2.45x for causal masks and 3.77x for sparse patterns.
* `networking` [A Verified High-Performance Composable Object Library for Remote Direct Memory Access (Extended Version)](http://arxiv.org/abs/2510.10531v1)
  > **TL;DR**: Introduces LOCO, a formally verified composable object library for RDMA to simplify distributed programming. Achieves performance comparable to custom RDMA systems with verified correctness via the Mowgli verification framework. Reduces complexity while matching custom system performance.
* `MoE` `offloading` `serving` [SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference](http://arxiv.org/abs/2510.10302v2)
  > **TL;DR**: Addresses GPU memory and bandwidth bottlenecks in speculative decoding for MoE model inference. Proposes SP-MoE with speculative expert prefetching, cutoff-layer policy, and pipelined runtime. Achieves 1.07-3.5× TPOT speedup over SOTA methods.
* `training` `edge` `quantization` [Parameter-Efficient and Personalized Federated Training of Generative Models at the Edge](http://arxiv.org/abs/2511.11585v1)
  > **TL;DR**: Proposes FedGen-Edge, a federated learning framework using LoRA adapters to train generative models efficiently at the edge. Reduces uplink traffic by >99% vs full FedAvg, achieves lower perplexity/FID on PTB/CIFAR-10 while personalizing client models.
* `edge` `serving` `RL` [Efficient Onboard Vision-Language Inference in UAV-Enabled Low-Altitude Economy Networks via LLM-Enhanced Optimization](http://arxiv.org/abs/2510.10028v1)
  > **TL;DR**: Addresses efficient vision-language inference on resource-constrained UAVs. Proposes combined resource allocation and LLM-enhanced reinforcement learning for trajectory optimization. Achieves reduced task latency and power consumption under accuracy constraints in dynamic networks.
* `hardware` `training` `edge` [Co-designing a Programmable RISC-V Accelerator for MPC-based Energy and Thermal Management of Many-Core HPC Processors](http://arxiv.org/abs/2510.09163v1)
  > **TL;DR**: Proposes a hardware-software codesign for an MPC-based energy/thermal management accelerator for many-core processors. Uses a pruned operator-splitting quadratic programming solver and scheduled parallel execution on RISC-V cores. Achieves sub-millisecond latency controlling 144 PEs with 33× lower latency than baseline.
* `RAG` `multi-modal` `sparse` [Hierarchical Scheduling for Multi-Vector Image Retrieval](http://arxiv.org/abs/2510.08976v1)
  > **TL;DR**: Proposes HiMIR, a hierarchical scheduling framework for multi-vector image retrieval to improve accuracy and efficiency in MLLM RAG systems. Uses multiple granularities and cross-hierarchy similarity with sparsity to reduce redundant matching. Reduces computation by 3.5× while improving accuracy.
* `training` `storage` `networking` [Slicing Is All You Need: Towards A Universal One-Sided Algorithm for Distributed Matrix Multiplication](http://arxiv.org/abs/2510.08874v1)
  > **TL;DR**: Proposes a universal one-sided algorithm for distributed matrix multiplication to avoid operand redistribution across partitionings. Uses slicing to compute overlapping tiles and lowers to optimized IR. Achieves competitive performance with PyTorch DTensor for varied partitionings and replication factors.
* `training` `offline` `kernel` [Maple: A Multi-agent System for Portable Deep Learning across Clusters](http://arxiv.org/abs/2510.08842v2)
  > **TL;DR**: Proposes Maple, a multi-agent system for generating correct command lines for distributed DL training across heterogeneous GPU clusters using natural language. Agents handle extraction, template retrieval, verification, and correction, achieving 92.0% accuracy across 567 test cases.
* `hardware` `serving` `offline` [SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inference](http://arxiv.org/abs/2510.08544v1)
  > **TL;DR**: Proposes SPAD, specialized hardware for disaggregated LLM inference with Prefill and Decode Chips optimized for each phase. Achieves 19%-41% lower hardware cost and 2%-17% lower TDP compared to H100 clusters while maintaining performance.
* `training` `RL` [DYNAMIX: RL-based Adaptive Batch Size Optimization in Distributed Machine Learning Systems](http://arxiv.org/abs/2510.08522v1)
  > **TL;DR**: Proposes DYNAMIX, an RL-based framework for adaptive batch size optimization in distributed ML training. Uses PPO with multi-dimensional system state to dynamically adjust batch sizes without explicit modeling. Achieves 46% training time reduction and 6.3% higher accuracy.
* `edge` `training` `serving` [Distributed Resource Selection for Self-Organising Cloud-Edge Systems](http://arxiv.org/abs/2510.08228v1)
  > **TL;DR**: Addresses efficient resource allocation in cloud-edge systems for distributed applications. Proposes a distributed consensus-based mechanism using local knowledge and inter-agent collaboration for dynamic resource selection. Achieves up to 30 times faster allocation than centralized heuristics without compromising optimality.
* `hardware` `serving` `edge` [Towards Energy-Efficient Serverless Computing with Hardware Isolation](http://arxiv.org/abs/2510.08180v1)
  > **TL;DR**: Proposes hardware isolation for serverless computing by assigning individual processors per function to avoid software overhead and idle servers. Achieves 90.63% reduction in energy consumption overheads in preliminary evaluation.
* `networking` `training` [When Light Bends to the Collective Will: A Theory and Vision for Adaptive Photonic Scale-up Domains](http://arxiv.org/abs/2510.08072v1)
  > **TL;DR**: Proposes adaptive photonic interconnects to optimize collective communications in scale-up systems. Uses dynamic reconfiguration guided by operation structure and theoretical framework balancing reconfiguration delay and performance gains. Improves maximum concurrent flow for collectives (e.g., AllReduce) via BvN decomposition.
* `serving` `MoE` `offloading` [From Tokens to Layers: Redefining Stall-Free Scheduling for LLM Serving with Layered Prefill](http://arxiv.org/abs/2510.08055v1)
  > **TL;DR**: Proposes layered prefill scheduling for LLM serving, replacing token-based chunking with layer-group interleaving to reduce redundant MoE weight reloads. Achieves 70% lower TTFT, 41% lower end-to-end latency, and 22% lower per-token energy versus stall-free chunked prefill.
* `serving` `networking` `hardware` [Evaluating Rapid Makespan Predictions for Heterogeneous Systems with Programmable Logic](http://arxiv.org/abs/2510.06998v1)
  > **TL;DR**: Addresses rapid makespan prediction for task scheduling on heterogeneous accelerators (CPUs, GPUs, FPGAs). Proposes an evaluation framework using task graphs to validate analytical models against real-world data transfers and congestion. Achieves practical prediction methods for mapping efficiency.
* `edge` `serving` `offloading` [Multi-Dimensional Autoscaling of Stream Processing Services on Edge Devices](http://arxiv.org/abs/2510.06882v1)
  > **TL;DR**: Proposes MUDAP for multi-dimensional autoscaling of stream processing on resource-constrained edge devices. Combines service-level (e.g., data quality, model size) and resource-level vertical scaling with a RASK agent for regression-based action optimization. Achieves 28% fewer SLO violations than Kubernetes VPA and RL baselines.
* `kernel` `serving` `quantization` [Vectorized FlashAttention with Low-cost Exponential Computation in RISC-V Vector Processors](http://arxiv.org/abs/2510.06834v1)
  > **TL;DR**: Optimizes FlashAttention computation by vectorizing operations and introducing low-cost exponential approximation for softmax on RISC-V processors. Combines tiling strategies for memory efficiency. Achieves significant speedups over baseline.
* `hardware` `serving` `training` [On-Package Memory with Universal Chiplet Interconnect Express (UCIe): A Low Power, High Bandwidth, Low Latency and Low Cost Approach](http://arxiv.org/abs/2510.06513v1)
  > **TL;DR**: Proposes enhancing UCIe with memory semantics for on-package memory to address AI memory walls. Approaches include reusing LPDDR6/HBM via logic die and native UCIe DRAM, yielding up to 10x bandwidth density, 3x lower latency/power, and reduced cost vs. HBM4/LPDDR.
* `training` `offline` [Adaptive Protein Design Protocols and Middleware](http://arxiv.org/abs/2510.06396v1)
  > **TL;DR**: Proposes IMPRESS, an adaptive protein design system with middleware for coupling AI/ML to HPC. Implements dynamic resource allocation and asynchronous workload execution to improve throughput. Achieves increased consistency and enhanced throughput in computational protein design tasks.
* `training` `RL` `offloading` [EARL: Efficient Agentic Reinforcement Learning Systems for Large Language Models](http://arxiv.org/abs/2510.05943v1)
  > **TL;DR**: Addresses memory inflation and OOM failures in agentic RL for LLMs. Proposes EARL with dynamic parallelism selector and layout-aware data dispatcher for adaptive model parallelism and decentralized data exchange. Reduces long-context failures and achieves stable large-scale training at scale.
* `serving` `kernel` `hardware` [The Anatomy of a Triton Attention Kernel](http://arxiv.org/abs/2511.11581v1)
  > **TL;DR**: Develops a portable paged attention kernel for LLM inference using Triton to eliminate manual tuning. The kernel achieves 105.9% of state-of-the-art performance across NVIDIA and AMD GPUs through JIT compilation and auto-tuning, improving portability and efficiency.
* `agentic` `offloading` `serving` [Toward Systems Foundations for Agentic Exploration](http://arxiv.org/abs/2510.05556v1)
  > **TL;DR**: Addresses systems challenges in supporting agentic exploration with LLMs. Identifies key issues in fork semantics, external side-effects, and native forking. Benchmarks show existing snapshot/restore mechanisms are insufficient for real deployment speeds, requiring microsecond-scale cloning without bulk copying.
* `serving` `MoE` `hardware` [Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting](http://arxiv.org/abs/2510.05497v3)
  > **TL;DR**: Addresses data movement overhead in MoE LLM serving from random expert selection. Proposes profiling-driven insights for hardware redesign, with wafer-scale GPU modifications achieving up to 5.3× speedup on DeepSeek V3.
* `networking` [cMPI: Using CXL Memory Sharing for MPI One-Sided and Two-Sided Inter-Node Communications](http://arxiv.org/abs/2510.05476v2)
  > **TL;DR**: Optimizes MPI inter-node communication using CXL memory sharing to bypass network protocols. Proposes cMPI, transforming cross-node communication into CXL memory transactions. Achieves up to 72x higher bandwidth and 49x lower latency versus TCP over Ethernet and SmartNIC for small messages.
* `training` `offloading` `kernel` [OptPipe: Memory- and Scheduling-Optimized Pipeline Parallelism for LLM Training](http://arxiv.org/abs/2510.05186v1)
  > **TL;DR**: Optimizes pipeline parallelism scheduling for LLM training via constrained optimization modeling memory, activation reuse, and bubble minimization. Dynamically adjusts offloading tradeoffs by hardware/model structure. Reduces idle pipeline time by up to 50% under same memory limits.
* `networking` `serving` `offline` [Next-Generation Event-Driven Architectures: Performance, Scalability, and Intelligent Orchestration Across Messaging Frameworks](http://arxiv.org/abs/2510.04404v2)
  > **TL;DR**: Studies performance and scalability of event-driven architectures for AI inference pipelines. Introduces AIEO with ML-driven predictive scaling and RL-based resource allocation. Achieves 34% latency reduction, 28% resource utilization improvement across messaging systems.
* `agentic` `serving` [Speculative Actions: A Lossless Framework for Faster Agentic Systems](http://arxiv.org/abs/2510.04371v1)
  > **TL;DR**: Addresses high-latency bottlenecks in agentic systems via speculative parallel execution of likely actions. Introduces a framework with faster models to predict multiple steps ahead, verified for losslessness. Achieves up to 55% action prediction accuracy and significant end-to-end latency reduction.
* `serving` `RAG` [SATER: A Self-Aware and Token-Efficient Approach to Routing and Cascading](http://arxiv.org/abs/2510.05164v1)
  > **TL;DR**: Addresses LLM serving cost vs. accuracy tradeoffs via SATER, a dual-mode routing framework applying response optimization and confidence-aware rejection. Reduces computational costs by over 50% and cascade latency by 80%.
* `training` `networking` `sparse` [Toward Co-adapting Machine Learning Job Shape and Cluster Topology](http://arxiv.org/abs/2510.03891v1)
  > **TL;DR**: Addresses resource allocation for distributed ML jobs in torus-topology clusters. Proposes RFold, adapting job shapes and cluster topology via homomorphic shape identification and optical circuit switch reconfiguration. Achieves 57% higher utilization and up to 11× lower job completion time.
* `training` `hardware` `serving` [Datacenter Energy Optimized Power Profiles](http://arxiv.org/abs/2510.03872v2)
  > **TL;DR**: Introduces datacenter power profiles for NVIDIA Blackwell GPUs to optimize energy efficiency in AI/HPC workloads. Employs hardware-software innovations and application-aware optimization recipes. Achieves up to 15% energy savings and 13% throughput increase within power constraints.
* `training` `networking` `kernel` [Short-circuiting Rings for Low-Latency AllReduce](http://arxiv.org/abs/2510.03491v1)
  > **TL;DR**: Challenges the assumption that Recursive Doubling is superior to Ring for small AllReduce messages in GPU clusters. Proposes a heuristic for circuit-switching photonic interconnects to dynamically balance reconfiguration delays and congestion. Achieves faster completion times than static Ring AllReduce in evaluations with realistic delays.
* `diffusion` `training` `sparse` [Paris: A Decentralized Trained Open-Weight Diffusion Model](http://arxiv.org/abs/2510.03434v1)
  > **TL;DR**: Proposes Paris, a decentralized trained diffusion model with expert partitioning for text-to-image generation. Distributes training across isolated experts with data clustering and dynamic routing, avoiding synchronization. Achieves comparable quality with 14× less data and 16× less compute than prior decentralized baseline.
* `training` `quantization` `offloading` [Energy Efficiency in Cloud-Based Big Data Processing for Earth Observation: Gap Analysis and Future Directions](http://arxiv.org/abs/2510.02882v1)
  > **TL;DR**: Addresses energy inefficiency in cloud-based big data processing for Earth Observation with foundation models. Proposes energy-aware monitoring, optimization techniques, and task scheduling for distributed frameworks. Aims to reduce power consumption and carbon footprint while maintaining processing performance.
* `serving` `edge` `offloading` [Action Deviation-Aware Inference for Low-Latency Wireless Robots](http://arxiv.org/abs/2510.02851v2)
  > **TL;DR**: Proposes Action Deviation-Aware Hybrid Inference (ADAHI) to reduce latency for robot policy inference by selectively offloading drafts to a server based on action deviation. Combines lightweight on-device model with remote verification, cutting latency by 39.2% and transmission by 40%, preserving 97.2% task-success.
* `video` `serving` `offloading` [TridentServe: A Stage-level Serving System for Diffusion Pipelines](http://arxiv.org/abs/2510.02838v1)
  > **TL;DR**: Proposes TridentServe, a dynamic stage-level serving system for diffusion pipelines that co-optimizes resource allocation across model stages and requests. It automates placement and dispatch plans, achieving up to 2.5x lower average latency and 4.1x lower P95 latency over existing systems.
* `training` `networking` `sparse` [Distributed Low-Communication Training with Decoupled Momentum Optimization](http://arxiv.org/abs/2510.03371v1)
  > **TL;DR**: Reduces communication overhead in distributed LLM training by decomposing Nesterov momentum into high/low-frequency components via DCT, synchronizing only high-frequency components every H steps. Achieves 16x less communication than DiLoCo baseline while maintaining generalization across architectures.
* `hardware` `kernel` `quantization` [UPMEM Unleashed: Software Secrets for Speed](http://arxiv.org/abs/2510.15927v1)
  > **TL;DR**: Examines software inefficiencies in UPMEM PIM platforms for low-precision kernels. Proposes assembly-level optimizations, bit-serial processing INT4/INT8, and NUMA-aware allocations. Achieves 1.4-5.9x arithmetic speedups and over 3x faster INT8 GEMV vs CPU after optimizations.
* `serving` `MoE` `offloading` [ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models](http://arxiv.org/abs/2510.02613v1)
  > **TL;DR**: Addresses elastic scaling challenges for MoE model serving. Proposes ElasticMoE with decoupled execution-memory operations and zero-copy KV cache reuse for concurrent scaling. Achieves up to 9x lower scale-up latency and 2x higher throughput during scaling.
* `edge` `offloading` `serving` [Accuracy vs Performance: An abstraction model for deadline constrained offloading at the mobile-edge](http://arxiv.org/abs/2510.01885v1)
  > **TL;DR**: Proposes a deadline-aware scheduling algorithm for DNN offloading to mobile edge devices, using lightweight network state representation and dynamic bandwidth estimation. Improves task throughput under resource scarcity and reduces latency in high-volume workloads.
* `edge` `serving` `RL` [Percepta: High Performance Stream Processing at the Edge](http://arxiv.org/abs/2510.05149v1)
  > **TL;DR**: Proposes Percepta, a lightweight edge stream processing system for AI workloads, focusing on reinforcement learning. It includes reward computation, real-time data prep, and handles normalization/harmonization. Achieves reduced latency and efficient data handling for continuous decision-making at the edge.
* `training` `serving` `storage` [Semantic-Aware Scheduling for GPU Clusters with Large Language Models](http://arxiv.org/abs/2510.03334v1)
  > **TL;DR**: Proposes SchedMate, a semantic-aware GPU cluster scheduler using LLMs to analyze unstructured data (source code, logs, history). Integrates with existing schedulers to reduce profiling overhead and improve estimates. Achieves up to 1.91× reduction in average job completion time.
* `serving` `diffusion` `edge?` [TetriServe: Efficient DiT Serving for Heterogeneous Image Generation](http://arxiv.org/abs/2510.01565v2)
  > **TL;DR**: Proposes TetriServe, a DiT serving system using step-level sequence parallelism to adapt parallelism per request based on deadlines. Implements round-based scheduling to minimize GPU hours and late completions. Achieves 32% higher SLO attainment.
* `agentic` `thinking` `serving` [FlashResearch: Real-time Agent Orchestration for Efficient Deep Research](http://arxiv.org/abs/2510.05145v1)
  > **TL;DR**: Addresses inefficiency in deep research agents via runtime parallelization. Introduces adaptive query breakdown, orchestration pruning, and multidimension concurrency. Achieves 5x speedup while maintaining report quality under fixed time budget.
* `training` `networking` `kernel` [An Efficient, Reliable and Observable Collective Communication Library in Large-scale GPU Training Clusters](http://arxiv.org/abs/2510.00991v1)
  > **TL;DR**: Addresses inefficiency, reliability, and observability in collective communication for large-scale LLM training. Proposes ICCL, a library that offloads P2P communication to CPU, adds NIC failure tolerance, and introduces fine-grained monitoring. Achieves 23.4%/28.5% better P2P throughput/latency and 6.02% higher training throughput versus NCCL.
* `training` `offloading` `kernel` [ElasWave: An Elastic-Native System for Scalable Hybrid-Parallel Training](http://arxiv.org/abs/2510.00606v3)
  > **TL;DR**: Addresses fault tolerance and elasticity in large-scale LLM training. Proposes ElasWave with multi-dimensional scheduling, online pipeline resharding, and DVFS to maintain consistency and reduce recovery time. Achieves up to 1.60x higher throughput and 51% lower migration MTTR.
* `edge` `serving` `RAG` [PolyLink: A Blockchain Based Decentralized Edge AI Platform for LLM Inference](http://arxiv.org/abs/2510.02395v1)
  > **TL;DR**: Proposes PolyLink, a blockchain-based decentralized platform for LLM inference at the edge. Uses crowdsourced devices, TIQE protocol for integrity, and token incentives. Achieves low verification latency and resists security attacks in geo-distributed deployment.
* `MoE` `training` `offloading` [FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training](http://arxiv.org/abs/2510.00207v2)
  > **TL;DR**: Addresses inefficiencies in distributed MoE training pipelines. Proposes FlowMoE, a unified scheduling framework integrating MHA, gating, expert computing, and communications via tensor-chunk prioritization. Reduces training time by 13%-57%, energy by 10%-39%, and memory by 7%-32%.
* `training` `kernel` `quantization` [LoRAFusion: Efficient LoRA Fine-Tuning for LLMs](http://arxiv.org/abs/2510.00206v1)
  > **TL;DR**: Presents LoRAFusion, optimizing LoRA fine-tuning via fused kernels and adaptive batching. Addresses redundant memory access and concurrent multi-job scheduling. Achieves up to 1.96x end-to-end speedup over Megatron-LM.
* `networking` `training` `edge` [Lattica: A Decentralized Cross-NAT Communication Framework for Scalable AI Inference and Training](http://arxiv.org/abs/2510.00183v2)
  > **TL;DR**: Proposes Lattica, a decentralized cross-NAT communication framework using P2P mesh, CRDTs, and DHTs for scalable AI inference and training in heterogeneous environments. Achieves sovereign operation and efficient model synchronization.
* `training` `sparse` `kernel` [TASP: Topology-aware Sequence Parallelism](http://arxiv.org/abs/2509.26541v2)
  > **TL;DR**: Proposes topology-aware sequence parallelism for long-context LLMs to improve communication efficiency. Utilizes Hamiltonian decomposition to map Ring AllGather to AlltoAll topology via concurrent ring datapaths. Achieves 3.58x speedup over Ring Attention.
* `offline` `hardware` `training` [Rearchitecting Datacenter Lifecycle for AI: A TCO-Driven Framework](http://arxiv.org/abs/2509.26534v1)
  > **TL;DR**: Proposes a holistic TCO-driven framework for AI datacenter lifecycle management across building, hardware refresh, and operation stages. Coordinates design choices in power, cooling, networking, refresh strategies, and software optimizations. Achieves up to 40% reduction in total cost of ownership.
* `serving` `offloading` `networking` [Parallax: Efficient LLM Inference Service over Decentralized Environment](http://arxiv.org/abs/2509.26182v1)
  > **TL;DR**: Addresses efficient LLM inference in decentralized environments with heterogeneous GPU resources. Proposes Parallax, a two-phase scheduler that optimizes model allocation and request-time GPU pipeline selection. Reduces latency and increases throughput over decentralized baselines in real volunteer node deployments.
* `serving` `offline` `edge` [Accelerating LLM Inference with Precomputed Query Storage](http://arxiv.org/abs/2509.25919v1)
  > **TL;DR**: Proposes StorInfer, which precomputes and stores query-response pairs to bypass GPU inference for matched queries. Uses adaptive LLM-driven query generation and disk-backed vector DB for fast retrieval. Achieves 17.3% latency reduction with 150K pairs using 830 MB storage.
* `agentic` `edge` `offloading` [Adaptive and Resource-efficient Agentic AI Systems for Mobile and Embedded Devices: A Survey](http://arxiv.org/abs/2510.00078v1)
  > **TL;DR**: Surveys adaptive and resource-efficient agentic AI systems for mobile/embedded devices, covering techniques like elastic inference and test-time adaptation. Focuses on balancing model complexity with constraints like memory, energy, and latency in edge deployments.
* `kernel` `hardware` `sparse` [LAPIS: A Performance Portable, High Productivity Compiler Framework](http://arxiv.org/abs/2509.25605v1)
  > **TL;DR**: Proposes LAPIS, an MLIR-based compiler framework for high performance portability across architectures. Focuses on automatic lowering of sparse/dense linear algebra kernels from scientific and AI use cases, facilitating integration of PyTorch and Kokkos. Achieves comparable kernel performance to MLIR on diverse architectures.
* `training` `security` `edge` [Enhancing Split Learning with Sharded and Blockchain-Enabled SplitFed Approaches](http://arxiv.org/abs/2509.25555v1)
  > **TL;DR**: Proposes Sharded SplitFed Learning (SSFL) and Blockchain-enabled SplitFed (BSFL) to address scalability, performance, and security in federated settings. SSFL distributes server workload across shards; BSFL adds blockchain consensus for integrity. Achieves 85.2% scalability improvement and 62.7% higher attack resilience.
* `edge` `serving` `kernel` [Context-Driven Performance Modeling for Causal Inference Operators on Neural Processing Units](http://arxiv.org/abs/2509.25155v2)
  > **TL;DR**: Analyzes performance bottlenecks of causal inference operators on NPUs for edge LLM deployment. Benchmarks quadratic attention vs. sub-quadratic alternatives under architectural constraints. Finds attention becomes memory-bound with cache inefficiency; achieves profiling insights for 4-10× optimization in co-design approaches.
* `hardware` `kernel` `offloading` [Accelerating Dynamic Image Graph Construction on FPGA for Vision GNNs](http://arxiv.org/abs/2509.25121v1)
  > **TL;DR**: Addresses the bottleneck of dynamic image graph construction in Vision GNNs. Proposes a streaming FPGA accelerator with on-chip buffers and parallel sorting to minimize memory traffic. Achieves up to 16.6x speedup over CPU baselines.
* `training` `kernel` `storage` [A Scalable Distributed Framework for Multimodal GigaVoxel Image Registration](http://arxiv.org/abs/2509.25044v1)
  > **TL;DR**: Proposes FFDP with fused kernels and distributed framework for large-scale image registration. Optimizes non-GEMM bottlenecks with IO-aware kernels and convolution-aware sharding. Achieves 7x speedup and 59% peak memory reduction in multimodal brain registration.
* `MoE` `serving` `offloading` [GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference](http://arxiv.org/abs/2509.25041v2)
  > **TL;DR**: Addresses communication overhead and computational load imbalance in distributed MoE inference. Proposes GRACE-MoE: expert grouping/replication to reduce cross-device transfers and locality-aware routing with load prediction. Achieves up to 3.79x inference speedup over SOTA systems.
* `MoE` `serving` `networking` [From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing](http://arxiv.org/abs/2510.03293v1)
  > **TL;DR**: Proposes LASER, a plug-and-play routing algorithm for MoE inference that adapts routing based on gate score distributions to balance expert load. Achieves better load balancing, reducing latency by 1.2-2.1x and increasing throughput by 1.6-2.3x while maintaining accuracy across datasets.
* `training` `MoE` [HAPT: Heterogeneity-Aware Automated Parallel Training on Heterogeneous Clusters](http://arxiv.org/abs/2509.24859v1)
  > **TL;DR**: Presents Hapt, an automated fine-grained inter-operator parallel training framework for heterogeneous clusters. It includes a planner for load balancing and a scheduler for optimized computation-communication overlap. Achieves 1.3x-1.6x higher performance vs. state-of-the-art frameworks.
* `RAG` `storage` `serving` [Intent-Driven Storage Systems: From Low-Level Tuning to High-Level Understanding](http://arxiv.org/abs/2510.15917v1)
  > **TL;DR**: Proposes Intent-Driven Storage Systems (IDSS) using LLMs to infer workload intent for adaptive storage parameter reconfiguration. Integrates LLMs into control loops with policy guardrails, generating configurations for components like caching. Achieves up to 2.45x IOPS improvement on FileBench workloads.
* `serving` `offloading` `sparse` [SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving](http://arxiv.org/abs/2509.24626v1)
  > **TL;DR**: Reduces costs of serving long-context LLMs by optimizing dynamic sparse attention and hierarchical KV cache management. Introduces fragmentation-aware transfers, working-set batch control, and layer-segmented prefill to minimize HBM constraints and avoid thrashing. Achieves 9.26x lower TTFT latency and 3.14x higher throughput compared to state-of-the-art systems.
* `multi-modal` `serving` [RServe: Overlapping Encoding and Prefill for Efficient LMM Inference](http://arxiv.org/abs/2509.24381v1)
  > **TL;DR**: Addresses inefficiency in serving large multimodal models (LMMs) by proposing REDServe, which overlaps multimodal encoding with prefill computations and balances loads via token budgeting. Achieves up to 66% lower latency and 109% higher throughput than existing systems.
* `training` `RL` `networking` [Asynchronous Policy Gradient Aggregation for Efficient Distributed Reinforcement Learning](http://arxiv.org/abs/2509.24305v1)
  > **TL;DR**: Proposes asynchronous policy gradient aggregation algorithms (Rennala/Malenia NIGT) for efficient distributed RL. Addresses heterogeneous computations and communication bottlenecks, with improved theoretical complexity and experiments showing significant speedup over prior methods.
* `training` `RL` `offloading` [RL in the Wild: Characterizing RLVR Training in LLM Deployment](http://arxiv.org/abs/2509.25279v2)
  > **TL;DR**: Characterizes system challenges in RLVR training for LLMs, identifying GPU idling from skewed sequence lengths, inefficient parallelism, and load imbalance. Proposes the PolyTrace benchmark suite for realistic evaluation, achieving 94.7% accuracy in validation.
* `edge` `serving` `training` [MACE: A Hybrid LLM Serving System with Colocated SLO-aware Continuous Retraining Alignment](http://arxiv.org/abs/2510.03283v1)
  > **TL;DR**: Proposes MACE, an edge LLM serving system with colocated SLO-aware continuous retraining. Uses iteration-level fine-tuning scheduling and memory management to balance inference latency and model accuracy. Reduces inference latency by up to 63% while maintaining throughput and high GPU utilization.
* `training` `MoE` [AdaPtis: Reducing Pipeline Bubbles with Adaptive Pipeline Parallelism on Heterogeneous Models](http://arxiv.org/abs/2509.23722v1)
  > **TL;DR**: Addresses pipeline bubbles in LLM training on heterogeneous models. Proposes AdaPtis, which jointly optimizes model partition, placement and scheduling using a performance model. Achieves an average 1.42x speedup over Megatron-LM.
* `serving` `offline` `edge` [A Predictive and Synergistic Two-Layer Scheduling Framework for LLM Serving](http://arxiv.org/abs/2509.23384v3)
  > **TL;DR**: Proposes NexusSched, a two-layer scheduling framework for LLM inference serving. Uses a performance model for per-step latency prediction to enable adaptive engine scheduling and predictive cluster routing. Achieves 43% higher SLO attainment and up to 3x throughput speedup in heterogeneous scenarios.
* `edge` `quantization` `serving` [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](http://arxiv.org/abs/2509.23324v1)
  > **TL;DR**: Addresses inefficient LLM inference on mobile NPUs by proposing a hardware-aware tile quantization and LUT-based operators for test-time scaling. Achieves up to 19.0x speedup for GEMM and enables smaller models to match larger model accuracy with lower cost.
* `training` `sparse` [A Flexible Programmable Pipeline Parallelism Framework for Efficient DNN Training](http://arxiv.org/abs/2510.05112v2)
  > **TL;DR**: Proposes FlexPipe, a programmable pipeline parallelism framework with a DSL and automated scheduler for efficient DNN training. Enables automated exploration of diverse pipeline schedules and customizations. Achieves up to 2.28X speedup over Megatron-LM.
* `MoE` `training` `offloading` [Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](http://arxiv.org/abs/2510.02345v1)
  > **TL;DR**: Addresses load imbalance, redundancy, and communication in MoE LLMs via dynamic expert clustering and structured compression. Introduces online clustering with routing-based regrouping and low-rank residual adapters, enabling hierarchical routing and heterogeneous precision. Achieves 80% parameter reduction, 10-20% higher throughput, and 3x lower load variance.
* `training` [Memory Efficient and Staleness Free Pipeline Parallel DNN Training Framework with Improved Convergence Speed](http://arxiv.org/abs/2509.23241v1)
  > **TL;DR**: Proposes V-TiMePReSt and I-TiMePReSt frameworks for pipeline-parallel DNN training to eliminate weight staleness and improve memory efficiency. V-TiMePReSt ensures staleness-free training with latest weights, while I-TiMePReSt uses intermediate weights to balance memory and convergence. Achieves improved GPU memory efficiency and optimal convergence speed.
* `training` `kernel` `hardware` [Efficient Fine-Grained GPU Performance Modeling for Distributed Deep Learning of LLM](http://arxiv.org/abs/2509.22832v1)
  > **TL;DR**: Proposes a fine-grained GPU performance modeling system for distributed LLM training. Combines operator-level decomposition, lightweight hardware-aware prediction, and end-to-end integration for parallelism. Achieves 4.98% prediction error on A100 and 9.38% on GH200 at 20B/128GPU scale without cluster testing.
* `serving` `storage` `hardware` [Agora: Bridging the GPU Cloud Resource-Price Disconnect](http://arxiv.org/abs/2510.05111v1)
  > **TL;DR**: Addresses market inefficiency in cloud GPU pricing for bandwidth-bound workloads. Proposes Agora, a feature-based pricing framework linked to resource consumption like memory bandwidth. Implementation with 10us sampling shows only 2.4% revenue loss compared to ideal sampling.
* `offloading` `training` `hardware` [The AI_INFN Platform: Artificial Intelligence Development in the Cloud](http://arxiv.org/abs/2509.22117v2)
  > **TL;DR**: Presents AI_INFN, a Kubernetes-based platform for efficient GPU resource sharing in AI workflows. Integrates offloading via Virtual Kubelet and InterLink API to span resources across distributed providers. Demonstrated scalable execution on heterogeneous resources, including WLCG sites and HPC centers.
* `training` `networking` `sparse` [Zeppelin: Balancing Variable-length Workloads in Data Parallel Large Model Training](http://arxiv.org/abs/2509.21841v2)
  > **TL;DR**: Addresses load imbalance in large-scale data-parallel training of LLMs with variable sequence lengths. Proposes Zeppelin with hierarchical sequence partitioning, dynamic routing for NIC utilization, and layout remapping. Achieves 2.80x average speedup over SOTA methods.
* `multi-modal` `edge` `offloading` [Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices](http://arxiv.org/abs/2510.05109v3)
  > **TL;DR**: Addresses high latency and inefficiency in monolithic execution of LMMs on heterogeneous battery-powered devices. Proposes NanoMind: hardware-software co-design with modular offloading to accelerators and token-aware memory management. Achieves 42.3% lower energy consumption and 20.8-hour runtime on one charge.
* `training` `sparse` `offloading` [Data-Centric Elastic Pipeline Parallelism for Efficient Long-Context LLM Training](http://arxiv.org/abs/2509.21275v2)
  > **TL;DR**: Proposes Elastic Pipeline Parallelism (EPP) and InfiniPipe system to optimize pipeline parallelism granularity for long-context LLM training. Adaptively combines token-level and batch-level PP with sequence packing and adaptive checkpointing. Achieves 1.69x speedup over state-of-the-art systems.
* `training` `offloading` `hardware` [SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips](http://arxiv.org/abs/2509.21271v1)
  > **TL;DR**: Proposes SuperOffload, a Superchip-centric offloading system for large-scale LLM training. Introduces adaptive weight offloading, bucket repartitioning, and optimized Adam for Grace CPUs. Achieves 2.5x throughput improvement, enabling 25B model training on a single NVIDIA GH200 Superchip.
* `training` `offloading` `networking` [Go With The Flow: Churn-Tolerant Decentralized Training of Large Language Models](http://arxiv.org/abs/2509.21221v1)
  > **TL;DR**: Proposes GWTF, a churn-tolerant decentralized framework for collaborative LLM training on volunteer clients. Introduces a flow algorithm to optimize routing of microbatches amid node churn and network instability. Reduces training time by up to 45% in heterogeneous, high-churn environments across 10 locations.
* `hardware` `offloading` `training` [From GPUs to RRAMs: Distributed In-Memory Primal-Dual Hybrid Gradient Method for Solving Large-Scale Linear Optimization Problem](http://arxiv.org/abs/2509.21137v3)
  > **TL;DR**: Proposes a distributed in-memory PDHG method for large-scale linear optimization using RRAM arrays to reduce energy and latency. Minimizes write cycles and unifies operations across crossbars; achieves up to 1000× energy and latency reductions versus GPUs while maintaining accuracy.
* `kernel` `hardware` [Mojo: MLIR-Based Performance-Portable HPC Science Kernels on GPUs for the Python Ecosystem](http://arxiv.org/abs/2509.21039v1)
  > **TL;DR**: Explores Mojo, an MLIR-based language for efficient science kernels on GPUs. Combines Python compatibility with low-level optimizations for cross-platform GPU portability. Achieves competitive performance to CUDA/HIP on memory-bound kernels, reducing development time while maintaining efficiency.
* `RL` `training` `sparse` [RollPacker: Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training](http://arxiv.org/abs/2509.21009v1)
  > **TL;DR**: Addresses GPU underutilization in synchronous RL post-training for LLMs. Proposes RollPacker with tail batching to consolidate long-tail responses into designated steps. Achieves 2.03x-2.56x end-to-end training speedup on Qwen2.5 models.
* `edge` `RAG` [IoT-MCP: Bridging LLMs and IoT Systems Through Model Context Protocol](http://arxiv.org/abs/2510.01260v1)
  > **TL;DR**: Proposes IoT-MCP, a framework using Model Context Protocol for standardized communication between LLMs and IoT systems via edge servers. Achieves 100% task success rate and 205ms avg response time on complex tasks with only 74KB peak memory footprint on microcontrollers.
* `serving` [Prompt-Aware Scheduling for Low-Latency LLM Serving](http://arxiv.org/abs/2510.03243v2)
  > **TL;DR**: Tackles Head-of-Line blocking in LLM inference scheduling by introducing Prompt-Aware Scheduler (PARS), which approximates shortest-job-first via pairwise ranking. Integrated into vLLM, PARS predicts response lengths to reduce latency. Experiments show significant latency reduction with minimal overhead across various LLMs.
* `training` `inference` `edge` [Integrating and Characterizing HPC Task Runtime Systems for hybrid AI-HPC workloads](http://arxiv.org/abs/2509.20819v1)
  > **TL;DR**: Studies how to manage hybrid AI-HPC workloads combining MPI, training, and inference. Integrates RADICAL-Pilot with Flux and Dragon for hierarchical resource management. Achieves over 1,500 tasks/s with 99.6% utilization, reducing makespan by 30-60% and increasing throughput 4x on Frontier.
* `training` `serving` `storage` [Kant: An Efficient Unified Scheduling System for Large-Scale AI Clusters](http://arxiv.org/abs/2510.01256v1)
  > **TL;DR**: Proposes Kant, a unified scheduler for co-scheduling LLM training and inference jobs in large-scale AI clusters. Utilizes Backfill and E-Binpack strategies to optimize resource utilization and reduce fragmentation. Achieves higher GPU Allocation Ratio and Scheduling Occupancy Rate.
* `serving` `RAG` [Experience Deploying Containerized GenAI Services at an HPC Center](http://arxiv.org/abs/2509.20603v2)
  > **TL;DR**: Explores deployment of GenAI services, including LLM inference servers, at an HPC center. Proposes a converged HPC and Kubernetes architecture for containerized workloads using vLLM for inference. Achieves reproducible multi-platform deployment across different container runtimes.
* `training` `offline` `storage` [FZModules: A Heterogeneous Computing Framework for Customizable Scientific Data Compression Pipelines](http://arxiv.org/abs/2509.20563v1)
  > **TL;DR**: Addresses high data volume challenges in scientific computing with customizable compression pipelines. Proposes FZModules, a heterogeneous framework with asynchronous task execution and dependency management for modular lossy compression. Achieves significant end-to-end speedup comparable to GPU compressors with improved rate-distortion fidelity.
* `scheduling` `RL` `training` [Adaptive Approach to Enhance Machine Learning Scheduling Algorithms During Runtime Using Reinforcement Learning in Metascheduling Applications](http://arxiv.org/abs/2509.20520v1)
  > **TL;DR**: Proposes an online reinforcement learning unit for metaschedulers to adapt AI-based scheduling during runtime. Integrates RL models to discover new scheduling solutions and optimize existing ones under dynamic events. Achieves continuous refinement for robustness in safety-critical environments.
* `RL` `edge` `networking` [Reconstruction-Based Adaptive Scheduling Using AI Inferences in Safety-Critical Systems](http://arxiv.org/abs/2509.20513v1)
  > **TL;DR**: Proposes a reconstruction framework to dynamically validate and assemble schedules in time-triggered safety-critical systems. Transforms AI-generated or heuristic scheduling priorities into executable schedules with safety checks and recovery mechanisms. Improves system adaptability and runtime performance, achieving significant enhancement in operational integrity.
* `serving` `quantization` `hardware` [Energy Use of AI Inference: Efficiency Pathways and Test-Time Compute](http://arxiv.org/abs/2509.20241v1)
  > **TL;DR**: Estimates per-query energy use for large-scale LLM inference with token throughput methodology. Analyzes efficiency gains via model-level optimizations, hardware improvements, and serving platform enhancements. Achieves up to 20x reduction in energy per query with combined interventions.
* `training` `inference` `edge` [Fulcrum: Optimizing Concurrent DNN Training and Inferencing on Edge Accelerators](http://arxiv.org/abs/2509.20205v1)
  > **TL;DR**: Addresses optimizing concurrent DNN training and inference on edge accelerators under power and latency constraints. Proposes Fulcrum scheduler with gradient descent (GMD) and active learning (ALS) techniques to interleave workloads and select power modes. Satisfies latency/power budgets in 97% of runs, achieving within 7% of optimal throughput.
* `edge` `serving` `hardware` [Pagoda: An Energy and Time Roofline Study for DNN Workloads on Edge Accelerators](http://arxiv.org/abs/2509.20189v1)
  > **TL;DR**: Studies power-performance trade-offs for DNN workloads on Nvidia Jetson edge accelerators. Develops energy and time roofline models coupled with analytical workload analysis. Achieves up to 15% energy reduction with minimal latency degradation by tuning power modes.
* `training` `sparse` `kernel` [BurstEngine: an Efficient Distributed Framework for Training Transformers on Extremely Long Sequences of over 1M Tokens](http://arxiv.org/abs/2509.19836v1)
  > **TL;DR**: Proposes BurstEngine, a distributed framework with topology-aware communication (BurstAttention), fine-grained overlap, selective checkpointing, and workload balance for efficient training on sequences >1M tokens. Achieves 1.2× speedup with lower memory overhead than SOTA baselines.
* `serving` `offloading` `kernel` [Gyges: Dynamic Cross-Instance Parallelism Transformation for Efficient LLM Inference](http://arxiv.org/abs/2509.19729v1)
  > **TL;DR**: Addresses throughput degradation due to context length variance in LLM serving. Proposes Gyges, a system featuring KV cache layout optimizations, weight padding, and transformation-aware scheduling for dynamic parallelism adjustments. Achieves 1.75x-6.57x higher throughput versus state-of-the-art solutions.
* `kernel` `serving` `training` [gpu_ext: Extensible OS Policies for GPUs via eBPF](http://arxiv.org/abs/2512.12615v2)
  > **TL;DR**: Proposes gpu_ext, an eBPF-based runtime for extensible GPU resource management policies in drivers and devices. Enables safe programmability and fine-grained control for events like memory placement and scheduling. Achieves up to 4.8x throughput improvement and 2x tail latency reduction for inference and training workloads.
* `kernel` `serving` `offloading` [Principled Performance Tunability in Operating System Kernels](http://arxiv.org/abs/2512.12530v1)
  > **TL;DR**: Addresses inefficient in-situ tuning of performance-critical constants (perf-consts) in Linux kernels. Introduces KernelX using Scoped Indirect Execution (SIE) to safely transform perf-consts into runtime-tunable knobs. Achieves millisecond-scale policy updates with significant performance improvements in kernel subsystems.
* `training` `kernel` [VLCs: Managing Parallelism with Virtualized Libraries](http://arxiv.org/abs/2512.04320v1)
  > **TL;DR**: Addresses library contention in parallel computing by proposing Virtual Library Contexts (VLCs) that encapsulate and isolate library resources without code changes. Enables resource partitioning and parallel execution. Achieves up to 2.85x speedup in benchmarks with OpenMP, OpenBLAS, and LibTorch.
* `edge` `offloading` `storage` [SARA: A Stall-Aware Memory Allocation Strategy for Mixed-Criticality Systems](http://arxiv.org/abs/2511.19991v1)
  > **TL;DR**: Proposes SARA, a stall-aware real-time memory allocator for mixed-criticality edge systems. It balances memory allocation between soft RT tasks and non-RT applications, minimizes soft RT memory usage by modeling latency effects, and mitigates stalls via proactive job dropping. Achieves 97.13% deadline hit ratio and up to 22.32x throughput improvement under memory constraints.
* `training` `storage` [Crash-Consistent Checkpointing for AI Training on macOS/APFS](http://arxiv.org/abs/2511.18323v1)
  > **TL;DR**: Investigates crash-consistent checkpointing protocols for AI training on macOS/APFS. Implements three write modes with increasing durability and a SHA-256-based integrity guard for auto-rollback. Detects 99.8-100% corruptions with zero false positives at 56.5-570.6% overhead versus unsafe baseline.
* `kernel` `edge` `storage` [eBPF-PATROL: Protective Agent for Threat Recognition and Overreach Limitation using eBPF in Containerized and Virtualized Environments](http://arxiv.org/abs/2511.18155v1)
  > **TL;DR**: Introduces eBPF-PATROL, a lightweight runtime security agent using eBPF to monitor system calls in containerized/virtualized environments. It enforces user-defined policies with context awareness, achieving <2.5% overhead and high detection accuracy against real-world attacks.
* `offloading` `hardware` `kernel` [Taiji: A DPU Memory Elasticity Solution for In-production Cloud Environments](http://arxiv.org/abs/2511.09936v2)
  > **TL;DR**: Proposes Taiji, a memory elasticity solution for DPUs using hybrid virtualization and parallel memory swapping to enable memory overcommitment. Achieves over 50% memory expansion with 5% virtualization overhead and 90% swap-ins under 10μs.
* `training` `offloading` `storage` [GoCkpt: Gradient-Assisted Multi-Step overlapped Checkpointing for Efficient LLM Training](http://arxiv.org/abs/2511.07035v1)
  > **TL;DR**: Addresses checkpoint saving overhead in LLM training by overlapping multi-step transfers with gradient data. Proposes GoCkpt for partial checkpoint updates with GPU-CPU optimization. Achieves 38.4% higher throughput and 86.7% less interruption time versus traditional methods.
* `hardware` `storage` `kernel` [Guidelines for Building Indexes on Partially Cache-Coherent CXL Shared Memory](http://arxiv.org/abs/2511.06460v1)
  > **TL;DR**: Investigates efficient indexing on Partial Cache-Coherence (PCC) platforms like CXL. Proposes SP and P$^3$ guidelines to ensure correctness and mitigate overhead. Achieves up to 16× throughput gain versus non-optimized designs.
* `serving` `offloading` `agentic` [Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live](http://arxiv.org/abs/2511.02230v2)
  > **TL;DR**: Addresses inefficient KV cache eviction in LLM serving for multi-turn agentic workloads. Proposes Continuum with KV cache time-to-live mechanism to selectively retain KV cache during tool call pauses. Reduces average job completion time by up to 32% on real-world agentic benchmarks.
* `offloading` [Jenga: Responsive Tiered Memory Management without Thrashing](http://arxiv.org/abs/2510.22869v1)
  > **TL;DR**: Addresses tiered memory performance issues due to hot/cold object mixing and hotness fluctuation. Proposes Jenga with context-based page allocation and stable hotness measurement for timely migration without thrashing. Runs applications 28% faster with fast tier at working set size.
* `storage` `kernel` `serving` [LatticeHashForest: An Efficient Data Structure for Repetitive Data and Operations](http://arxiv.org/abs/2510.18496v2)
  > **TL;DR**: Presents LatticeHashForest (LHF), a data structure for efficient deduplication and computation in repetitive operations, such as compiler optimizations. LHF reduces memory usage and speeds up operations by enabling immediate deduplication and nested construction. In pointer analysis, it achieves over 4x speedup and negligible memory for 10 million inputs.
* `hardware` `training` `offline` [Funky: Cloud-Native FPGA Virtualization and Orchestration](http://arxiv.org/abs/2510.15755v1)
  > **TL;DR**: Addresses FPGA virtualization and orchestration challenges in cloud-native environments. Proposes Funky, a full-stack FPGA-aware orchestration engine with virtualization, state management, and standard-compliant components. Achieves 7.4% performance overhead vs native with strong isolation and 28.7x smaller images.
* `agentic` `offloading` `serving` [Toward Systems Foundations for Agentic Exploration](http://arxiv.org/abs/2510.05556v1)
  > **TL;DR**: Addresses the lack of systems support for agentic exploration in LLMs. Identifies limitations of snapshot/restore mechanisms and proposes fundamental challenges: fork semantics, side-effect management, and native forking. Benchmarks show generic tools fail in real deployments.
* `storage` `kernel` [An Early Exploration of Deep-Learning-Driven Prefetching for Far Memory](http://arxiv.org/abs/2510.04360v1)
  > **TL;DR**: Explores deep-learning-driven prefetching for far-memory systems to reduce on-demand data fetches. Proposes Memix, co-designing deep learning with system architecture to predict memory accesses via application semantics and runtime context. Achieves up to 42% fewer far-memory accesses than state-of-the-art.
* `RL` `agentic` `offloading` [Secure and Efficient Access Control for Computer-Use Agents via Context Space](http://arxiv.org/abs/2509.22256v2)
  > **TL;DR**: Addresses security risks in LLM-based computer-use agents by developing CSAgent, a system-level access control framework with intent- and context-aware policies enforced via an optimized OS service. Achieves 99.36% attack defense rate with 6.83% overhead.
* `serving` `offloading` `multi-modal` [Nova: Real-Time Agentic Vision-Language Model Serving with Adaptive Cross-Stage Parallelization](http://arxiv.org/abs/2509.21301v1)
  > **TL;DR**: Proposes Nova, a real-time scheduling framework for serving agentic vision-language models on a single GPU. Utilizes adaptive cross-stage parallelization with elastic GPU partitioning and Pareto-optimal resource calibration, plus vision encoder weight offloading. Reduces maximum latency by up to 23.3% while maintaining competitive throughput.
* `training` `storage` [Preparation Meets Opportunity: Enhancing Data Preprocessing for ML Training With Seneca](http://arxiv.org/abs/2511.13724v1)
  > **TL;DR**: Addresses input data preprocessing bottlenecks in concurrent ML training. Proposes Seneca, a data loading system featuring cache partitioning for encoded/decoded/augmented data and opportunistic cached data sampling. Reduces makespan by 45.23% and increases throughput by 3.45x over baselines.

### 2025-12-20
* `training` `serving` `networking` [RAPID-LLM: Resilience-Aware Performance analysis of Infrastructure for Distributed LLM Training and Inference](http://arxiv.org/abs/2512.19606v1)
  > **TL;DR**: Proposes RAPID-LLM, a unified performance modeling framework for LLM training and inference on GPU clusters. Combines DeepFlow-based frontend and Astra-Sim backend to simulate hardware-aware execution with network faults. Predicts latency within 10.4% of measurements, enabling configuration sweeps and resilience analysis.
* `serving` `offline` `networking` [Faster Distributed Inference-Only Recommender Systems via Bounded Lag Synchronous Collectives](http://arxiv.org/abs/2512.19342v1)
  > **TL;DR**: Addresses communication bottlenecks in distributed recommender system inference. Proposes bounded lag synchronous (BLS) alltoallv collective with adjustable lag bounds to mask process delays. Achieves improved latency and throughput in unbalanced scenarios, masking delays entirely in best cases.
* `serving` `offloading` `scheduling` [L4: Low-Latency and Load-Balanced LLM Serving via Length-Aware Scheduling](http://arxiv.org/abs/2512.19179v1)
  > **TL;DR**: Addresses GPU underutilization and latency in LLM serving due to request-length heterogeneity. Proposes L4, a runtime system for dynamic request rescheduling and instance partitioning based on length groups. Achieves up to 69% lower tail latency and 2.89× throughput improvement over state-of-the-art schedulers.
* `edge` `offline` `RL` [Evidential Trust-Aware Model Personalization in Decentralized Federated Learning for Wearable IoT](http://arxiv.org/abs/2512.19131v1)
  > **TL;DR**: Proposes Murmura, a trust-aware model personalization framework for decentralized federated learning on edge devices. Uses evidential deep learning to compute compatibility scores via cross-evaluation and adaptive aggregation. Achieves 7.4× faster convergence vs IID degradation in wearable IoT datasets.
* `edge` `networking` `serving` [QoS-Aware Load Balancing in the Computing Continuum via Multi-Player Bandits](http://arxiv.org/abs/2512.18915v1)
  > **TL;DR**: Proposes QEdgeProxy, a decentralized QoS-aware load balancer for edge computing, modeled as a Multi-Player MAB with kernel density estimation for per-client QoS. Kubernetes implementation achieves improved per-client QoS satisfaction in latency-sensitive workloads.
* `MoE` `offloading` `serving` [Remoe: Towards Efficient and Low-Cost MoE Inference in Serverless Computing](http://arxiv.org/abs/2512.18674v1)
  > **TL;DR**: Addresses high cost and memory overhead in MoE inference for serverless computing. Proposes Remoe, a heterogeneous system that assigns non-expert modules to GPUs and experts to CPUs, with SPS for activation prediction and MMP for SLO compliance. Reduces cost by 57% and cold start latency by 47%.
* `serving` `offloading` `kernel` [Asynchronous Pipeline Parallelism for Real-Time Multilingual Lip Synchronization in Video Communication Systems](http://arxiv.org/abs/2512.18318v1)
  > **TL;DR**: Proposes an asynchronous pipeline-parallel Transformer framework for real-time multilingual lip synchronization. Employs message-queue decoupling, low-level graph compilation, mixed-precision quantization, and kernel fusion to reduce latency. Achieves up to 3.1× lower end-to-end latency than sequential approaches.
* `serving` `offloading` `networking` [TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale](http://arxiv.org/abs/2512.18194v1)
  > **TL;DR**: Proposes TraCT, a rack-scale LLM serving system using CXL shared memory for KV cache, avoiding RDMA networks. Addresses synchronization and consistency via software solutions like two-tier synchronization. Achieves 9.8x lower TTFT, 6.2x lower P99 latency, and 1.6x higher throughput.
* `training` `networking` `sparse` [ACE-Sync: An Adaptive Cloud-Edge Synchronization Framework for Communication-Efficient Large-Scale Distributed Model Training](http://arxiv.org/abs/2512.18127v1)
  > **TL;DR**: Proposes ACE-Sync, an adaptive cloud-edge synchronization framework with attention-based gradient importance prediction, differentiated compression, and hierarchical coordination to reduce communication in distributed training. Reduces communication cost from 112.5 GB to 44.7 GB (60% reduction) while maintaining model accuracy within 0.3%.
* `serving` `training` `multi-modal` [Enabling Disaggregated Multi-Stage MLLM Inference via GPU-Internal Scheduling and Resource Sharing](http://arxiv.org/abs/2512.17574v1)
  > **TL;DR**: Addresses bottlenecks in multi-stage MLLM inference pipelines. Proposes FlashCodec for GPU-accelerated video decoding and UnifiedServe for resource sharing and decoupled execution. Achieves up to 4.4× higher throughput and serves 3.0× more requests compared to SOTA systems.
* `edge` `sparse` `networking` [Adaptive Graph Pruning with Sudden-Events Evaluation for Traffic Prediction using Online Semi-Decentralized ST-GNNs](http://arxiv.org/abs/2512.17352v1)
  > **TL;DR**: Proposes adaptive graph pruning to reduce communication overhead in online semi-decentralized ST-GNNs for traffic prediction. Dynamically filters redundant neighbor features based on model performance and event responsiveness. Reduces communication cost by 20-40% while maintaining accuracy via novel SEPA metric.
* `storage` `offline` `serving` [Scalable Distributed Vector Search via Accuracy Preserving Index Construction](http://arxiv.org/abs/2512.17264v1)
  > **TL;DR**: Addresses scalable distributed vector search for ANNS with accuracy-latency-throughput tradeoffs. Proposes SPIRE via balanced partition granularity and accuracy-preserving recursive index construction. Achieves 9.64x higher throughput vs state-of-the-art at 8B vector scale.
* `serving` `diffusion` `sparse` [Taming the Memory Footprint Crisis: System Design for Production Diffusion LLM Serving](http://arxiv.org/abs/2512.17077v1)
  > **TL;DR**: Addresses the memory footprint crisis in serving diffusion LLMs. Proposes dLLM-Serve with Logit-Aware Activation Budgeting, Phase-Multiplexed Scheduler, and Head-Centric Sparse Attention. Achieves up to 1.81× higher throughput and 4× lower tail latency versus baselines.
* `multi-modal` [LLM-HPC++: Evaluating LLM-Generated Modern C++ and MPI+OpenMP Codes for Scalable Mandelbrot Set Computation](http://arxiv.org/abs/2512.17023v1)
* `serving` `offloading` `MoE` [Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems](http://arxiv.org/abs/2512.16473v1)
  > **TL;DR**: Proposes CPU-GPU collaborative inference for MoE-based LLMs to overcome GPU memory limits. Introduces GPU expert caching to reduce transfers, offloads cache misses to optimized CPU threads. On consumer hardware, achieves up to 5.6x speedup over full offloading with minimal accuracy loss.
* `kernel` `inference` [Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference](http://arxiv.org/abs/2512.16391v1)
  > **TL;DR**: Proposes Kascade, a training-free sparse attention method for accelerating long-context LLM inference. Reuses Top-k indices from anchor layers through algorithmic layer selection and head-aware reuse, optimized for tile-level GPU operations. Achieves 4.1× decode attention speedup over FlashAttention-3 with matching accuracy on benchmarks.
* `offloading` `storage` `networking` [FlexKV: Flexible Index Offloading for Memory-Disaggregated Key-Value Store](http://arxiv.org/abs/2512.16148v1)
  > **TL;DR**: Proposes FlexKV, a memory-disaggregated key-value store with index proxying to address poor performance. Dynamically offloads index to compute nodes with load balancing, memory optimization, and RPC-aggregated cache coherence. Achieves up to 2.94× higher throughput and 85.2% lower latency.
* `serving` `storage` `networking` [Lotus: Optimizing Disaggregated Transactions with Disaggregated Locks](http://arxiv.org/abs/2512.16136v1)
  > **TL;DR**: Proposes Lotus, a distributed transaction system for disaggregated memory that moves lock management to compute nodes. Introduces lock-first protocol and application-aware lock partitioning to reduce RDMA bottlenecks. Achieves 1.5× higher throughput and 49.4% lower latency than state-of-the-art systems.
* `serving` `offloading` [Staggered Batch Scheduling: Co-optimizing Time-to-First-Token and Throughput for High-Efficiency LLM Inference](http://arxiv.org/abs/2512.16134v1)
  > **TL;DR**: Identifies queuing bubbles from immediate scheduling in distributed DP+EP LLM serving systems. Proposes Staggered Batch Scheduling that buffers and batches requests to eliminate queuing. Reduces TTFT by 30-40% and improves throughput by 15-20% on Deepseek-V3 serving.
* `serving` `offloading` `kernel` [An Online Fragmentation-Aware Scheduler for Managing GPU-Sharing Workloads on Multi-Instance GPUs](http://arxiv.org/abs/2512.16099v1)
  > **TL;DR**: Addresses GPU fragmentation and resource contention in Multi-Instance GPUs for efficient sharing. Proposes an online scheduler with dynamic partitioning, job migration, and load balancing to minimize contention and combat fragmentation. Achieves up to 35% makespan improvement.
* `serving` `offloading` `kernel` [MultiPath Transfer Engine: Breaking GPU and Host-Memory Bandwidth Bottlenecks in LLM Services](http://arxiv.org/abs/2512.16056v1)
  > **TL;DR**: Proposes Multipath Memory Access (MMA) to overcome PCIe bandwidth bottlenecks in GPU-host data transfer for LLM serving. Uses multipath data transfer via dynamic library injection to increase bandwidth. Achieves peak bandwidth of 245 GB/s (4.62x speedup) and reduces TTFT by up to 2.38x.
* `training` `storage` `offline` [LOG.io: Unified Rollback Recovery and Data Lineage Capture for Distributed Data Pipelines](http://arxiv.org/abs/2512.16038v1)
  > **TL;DR**: Introduces LOG.io, a log-based system for rollback recovery and data lineage in distributed data pipelines. It supports non-deterministic operators and dynamic scaling, with non-blocking recovery. Achieves marginal overhead (≤1.5%) for lineage capture and outperforms ABS in straggler scenarios.
* `serving` `offline` [Dynamic Rebatching for Efficient Early-Exit Inference with DREX](http://arxiv.org/abs/2512.15705v1)
  > **TL;DR**: Addresses inefficient batching for Early-Exit LLM inference. Proposes Dynamic Rebatching via DREX, featuring copy-free buffering and EE/SLA-aware scheduling. Achieves 2-12% higher throughput while eliminating involuntary exits.
* `offline` `agentic` `serving` [Optimizing Agentic Language Model Inference via Speculative Tool Calls](http://arxiv.org/abs/2512.15834v1)
  > **TL;DR**: Addresses performance bottlenecks in tool-using agentic LMs via speculative tool calls and sequence residency. Optimizations include speculative execution and tool caching to reduce inference overheads. Achieves hundreds of tokens per second throughput improvement over baselines.
* `kernel` `storage` `hardware` [Optimizing Bloom Filters for Modern GPU Architectures](http://arxiv.org/abs/2512.15595v1)
  > **TL;DR**: Explores Bloom filter optimization on GPUs for high-throughput approximate membership queries. Proposes a GPU design with vectorization, thread cooperation, and compute latency tuning. Achieves 11.35× faster lookups and above 92% of practical speed limit at iso error rate on a B200 GPU.
* `training` `offloading` `quantization` [LLMQ: Efficient Lower-Precision Pretraining for Consumer GPUs](http://arxiv.org/abs/2512.15306v1)
  > **TL;DR**: Introduces LLMQ, an efficient CUDA/C++ system for LLM training on consumer GPUs with low memory. Combines 8-bit quantization, activation checkpointing, weight offloading, and optimized collectives. Achieves 50% FLOP utilization when training a 7B model on a single 16GB GPU.
* `training` `sparse` `networking` [PruneX: A Hierarchical Communication-Efficient System for Distributed CNN Training with Structured Pruning](http://arxiv.org/abs/2512.14628v1)
  > **TL;DR**: Addresses high communication overhead in distributed CNN training. Proposes hierarchical structured pruning with buffer compaction for reduced inter-node transmissions. Reduces inter-node communication volume by 60% and achieves 6.75x speedup on ResNet at 64 GPUs.
* `serving` `multi-modal` [Cornserve: Efficiently Serving Any-to-Any Multimodal Models](http://arxiv.org/abs/2512.14098v2)
  > **TL;DR**: Proposes Cornserve, a serving system for Any-to-Any multimodal models that optimizes deployment by disaggregating components. Introduces a planner and runtime for efficient handling of heterogeneous computations. Achieves up to 3.81× higher throughput and 5.79× lower tail latency.
* `serving` `offloading` `RAG` [Trustworthy and Controllable Professional Knowledge Utilization in Large Language Models with TEE-GPU Execution](http://arxiv.org/abs/2512.16238v2)
  > **TL;DR**: Proposes PKUS, a system for trustworthy professional knowledge integration in LLMs, using TEE-GPU co-execution with separable adapters. Implements hardware-rooted protocols and split-execution scheduling. Achieves 8.1-11.9x speedup over CPU-only TEE inference with comparable accuracy.
* `serving` `offloading` `compression` [EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving](http://arxiv.org/abs/2512.14946v1)
  > **TL;DR**: Addresses KV-cache management inefficiencies in LLM inference. Proposes EVICPRESS, which jointly optimizes lossy compression and adaptive eviction across storage tiers via a unified utility function. Achieves up to 2.19x faster time-to-first-token while maintaining generation quality.

### 2025-12-19
* `serving` `multi-modal` `video` [Enabling Disaggregated Multi-Stage MLLM Inference via GPU-Internal Scheduling and Resource Sharing](http://arxiv.org/abs/2512.17574v1)
  > **TL;DR**: Addresses latency and throughput bottlenecks in multi-stage multimodal LLM (MLLM) serving. Proposes FlashCodec for GPU-accelerated video decoding and UnifiedServe for resource sharing and inter-stage optimization. Achieves up to 4.4× higher throughput and 3.0× more requests served vs. SOTA.
* `offline` `network` `edge` [Adaptive Graph Pruning with Sudden-Events Evaluation for Traffic Prediction using Online Semi-Decentralized ST-GNNs](http://arxiv.org/abs/2512.17352v1)
  > **TL;DR**: Proposes adaptive graph pruning to reduce communication in distributed ST-GNNs for traffic prediction. Dynamically adjusts pruning based on performance and introduces SEPA metric. Achieves reduced communication costs by 27% while maintaining accuracy in semi-decentralized edge settings.
* `storage` `offline` `networking` [Scalable Distributed Vector Search via Accuracy Preserving Index Construction](http://arxiv.org/abs/2512.17264v1)
  > **TL;DR**: Scaling distributed ANN search for billion-scale vectors. Proposes SPIRE with balanced partition granularity and recursive accuracy-preserving index construction. Achieves up to 9.64x higher throughput vs. SOTA.

### 2025-12-18
* `edge` `offloading` `training` [Delay-Aware Multi-Stage Edge Server Upgrade with Budget Constraint](http://arxiv.org/abs/2512.16792v1)
  > **TL;DR**: Proposes M-ESU algorithm for multi-stage edge server upgrade and task offloading under budget constraints. Optimizes deployment/upgrade decisions and offloading to maximize tasks meeting delay requirements. Achieves 21.57% higher task satisfaction with efficient heuristic for large networks.
* `MoE` `offloading` `serving` [Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems](http://arxiv.org/abs/2512.16473v1)
  > **TL;DR**: Proposes a CPU-GPU collaborative inference framework for memory-limited systems running MoE-based LLMs. Uses expert caching on GPU and CPU offloading with multithreading to minimize data transfer. Achieves faster inference with reduced latency compared to traditional offloading methods.
* `training` `offline` `storage` [AI4EOSC: a Federated Cloud Platform for Artificial Intelligence in Scientific Research](http://arxiv.org/abs/2512.16455v1)
  > **TL;DR**: Presents a federated cloud platform for AI in science, offering integrated tools for the ML lifecycle including distributed GPU training, model deployment, and storage resources. Achieves reproducible deployments across distributed infrastructures with unified service catalog.
* `serving` `kernel` `sparse` [Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference](http://arxiv.org/abs/2512.16391v1)
  > **TL;DR**: Proposes Kascade, a training-free sparse attention method for long-context LLM inference. It reuses top-k indices across anchor and reuse layers with head-aware selection, optimizing tile-level operations. Achieves up to 4.1x decode attention speedup over FlashAttention-3 on H100 GPUs with minimal accuracy loss.
* `storage` `networking` `hardware` [Lotus: Optimizing Disaggregated Transactions with Disaggregated Locks](http://arxiv.org/abs/2512.16136v1)
  > **TL;DR**: Addresses the bottleneck of RDMA NICs in disaggregated memory transaction systems by disaggregating locks to compute nodes. Introduces Lotus with application-aware lock sharding, lock-first protocol, and lock-rebuild-free recovery. Achieves 2.1× higher throughput and 49.4% lower latency.
* `serving` `networking` [Staggered Batch Scheduling: Co-optimizing Time-to-First-Token and Throughput for High-Efficiency LLM Inference](http://arxiv.org/abs/2512.16134v1)
  > **TL;DR**: Optimizes scheduling in P/D-separated LLM inference systems to reduce Time-to-First-Token (TTFT) and boost throughput. Proposes Staggered Batch Scheduling (SBS) with Load-Aware Global Allocation to eliminate queuing bubbles and balance DP load. Achieves 30-40% TTFT reduction and 15-20% throughput gain.
* `serving` `offline` `networking` [An Online Fragmentation-Aware Scheduler for Managing GPU-Sharing Workloads on Multi-Instance GPUs](http://arxiv.org/abs/2512.16099v1)
  > **TL;DR**: Addresses GPU fragmentation and resource contention in MIG-enabled clusters. Proposes an online scheduler integrating load balancing, dynamic partitioning, and migration for efficient GPU-sharing. Achieves up to 35% makespan improvement.
* `serving` `offloading` `kernel` [MultiPath Transfer Engine: Breaking GPU and Host-Memory Bandwidth Bottlenecks in LLM Services](http://arxiv.org/abs/2512.16056v1)
  > **TL;DR**: Proposes Multipath Memory Access (MMA) to overcome PCIe bandwidth bottlenecks for GPU-host data transfer in LLM serving. Utilizes dynamic library injection for deployment transparency. Achieves up to 4.62x higher bandwidth and reduces TTFT by up to 2.38x in vLLM.
* `serving` `offloading` `RAG` [Trustworthy and Controllable Professional Knowledge Utilization in Large Language Models with TEE-GPU Execution](http://arxiv.org/abs/2512.16238v1)
  > **TL;DR**: Proposes PKUS, a system for trustworthy and controllable utilization of professional knowledge adapters in LLM serving. Uses TEE-GPU split execution with attested adapters, hardware-rooted protocols, and scheduling to isolate knowledge. Achieves 8.1-11.9x speedup over CPU-only TEE inference while matching fine-tuning accuracy.

### 2025-12-17
* `serving` `offloading` [Dynamic Rebatching for Efficient Early-Exit Inference with DREX](http://arxiv.org/abs/2512.15705v1)
  > **TL;DR**: Addresses inefficiency in Early-Exit LLM inference due to static batching. Proposes DREX system with copy-free dynamic rebatching and scheduler to optimally regroup requests at exit points. Achieves 2-12% higher throughput while eliminating involuntary exits and preserving quality.
* `serving` `kernel` `offloading` [Optimizing Bloom Filters for Modern GPU Architectures](http://arxiv.org/abs/2512.15595v1)
  > **TL;DR**: Explores GPU-optimized Bloom filters to accelerate approximate membership queries. Introduces designs leveraging vectorization, thread cooperation, and compute latency optimization. Achieves up to 15.4× faster construction and 92% of practical speed-of-light throughput on B200 GPU.
* `training` `offloading` `quantization` [LLMQ: Efficient Lower-Precision Pretraining for Consumer GPUs](http://arxiv.org/abs/2512.15306v1)
  > **TL;DR**: Proposes LLMQ, a CUDA/C++ system for efficient 8-bit LLM training on consumer GPUs with limited memory. Uses activation checkpointing, weight offloading, and copy-engine collectives to handle memory and communication bottlenecks. Trains a 7B model on a single 16GB GPU and maintains 50% FLOP utilization.
* `hardware` `storage` `networking` [Reexamining Paradigms of End-to-End Data Movement](http://arxiv.org/abs/2512.15028v1)
  > **TL;DR**: Examines end-to-end data movement bottlenecks beyond raw bandwidth. Proposes holistic hardware-software co-design addressing host CPU, virtualization, and congestion control. Achieves consistent performance across 1-100 Gbps links, reducing latency and throughput disparities in edge-to-core transfers.

### 2025-12-16
* `training` `sparse` `offloading` [Performance and Stability of Barrier Mode Parallel Systems with Heterogeneous and Redundant Jobs](http://arxiv.org/abs/2512.14445v1)
  > **TL;DR**: Analyzes stability and performance overhead of barrier synchronization in parallel ML training systems. Models (s,k,l) barrier systems that allow partial task completion and hybrid workloads. Validated against Apache Spark, showing overhead from dual event/polling mechanisms with quantified performance bounds.
* `edge` `serving` [A Hybrid Reactive-Proactive Auto-scaling Algorithm for SLA-Constrained Edge Computing](http://arxiv.org/abs/2512.14290v1)
  > **TL;DR**: Proposes a hybrid reactive-proactive auto-scaling algorithm for SLA-constrained edge computing. Combines ML-based demand forecasting with reactive resource adjustment, integrated into Kubernetes. Reduces SLA violation rate from 23% to 6% compared to baselines.
* `serving` `multi-modal` `kernel` [Cornserve: Efficiently Serving Any-to-Any Multimodal Models](http://arxiv.org/abs/2512.14098v1)
  > **TL;DR**: Proposes Cornserve, a system for efficient online serving of multimodal Any-to-Any models. Automatically optimizes deployment plans by disaggregating models and handling heterogeneous components. Achieves up to 3.81× throughput improvement and 5.79× tail latency reduction over existing solutions.

### 2025-12-15
* `MoE` `serving` `networking` [Janus: Disaggregating Attention and Experts for Scalable MoE Inference](http://arxiv.org/abs/2512.13525v1)
  > **TL;DR**: Proposes Janus, a scalable MoE inference system that disaggregates attention and experts across GPU sub-clusters. Key designs include adaptive communication, lightweight scheduling, and dynamic resource scaling. Achieves up to 3.9× higher per-GPU throughput with latency guarantees.
* `training` `MoE` `storage` [SIGMA: An AI-Empowered Training Stack on Early-Life Hardware](http://arxiv.org/abs/2512.13488v1)
  > **TL;DR**: Proposes SIGMA, a training stack for large-scale distributed training on early-life AI accelerators. Combines LTP platform for reliability and LTF framework for efficient MoE model training. Achieves 94.45% accelerator utilization and trains 200B MoE model with 21.08% MFU.
* `kernel` `quantization` `hardware` [FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection](http://arxiv.org/abs/2512.12949v1)
  > **TL;DR**: Proposes FlashFuser, a compiler framework for kernel fusion using GPU inter-core connections to overcome memory limits. Utilizes Distributed Shared Memory (DSM) for complex operators, optimizing data movement and tile selection. Achieves 58% less memory access, 3.3x kernel speedup against tuned libraries.
* `serving` [PROSERVE: Unified Multi-Priority Request Scheduling for LLM Serving](http://arxiv.org/abs/2512.12928v1)
  > **TL;DR**: Addresses multi-priority request scheduling in LLM serving to maximize service gain balancing SLO attainment and client priority. Proposes PROSERVE, a two-tier scheduler with SlideBatching for batch formation and GoRouting for request dispatching. Achieves up to 35% higher system gain and 52% better SLO attainment.

### 2025-12-14
* `serving` `offline` `training` [Fine-Grained Energy Prediction For Parallellized LLM Inference With PIE-P](http://arxiv.org/abs/2512.12801v1)
  > **TL;DR**: Proposes PIE-P, a framework for fine-grained energy prediction in multi-GPU parallellized LLM inference. Uses precise sampling and modeling of inter-GPU communication to account for parallelism overhead. Achieves accurate energy predictions, significantly outperforming baselines in parallelized settings.
* `training` `networking` `quantization` [SPARK: Igniting Communication-Efficient Decentralized Learning via Stage-wise Projected NTK and Accelerated Regularization](http://arxiv.org/abs/2512.12737v1)
  > **TL;DR**: Addresses communication overhead in decentralized federated learning with statistical heterogeneity. Proposes SPARK, integrating Jacobian compression via random projection, stage-wise distillation, and Nesterov momentum. Reduces communication by 98.7% compared to NTK-DFL with 3× faster convergence.
* `edge` `networking` `storage` [Strategic Server Deployment under Uncertainty in Mobile Edge Computing](http://arxiv.org/abs/2512.12532v1)
  > **TL;DR**: Addresses strategic server deployment in mobile edge computing under uncertainty. Formulates as stochastic bilevel optimization and uses submodular approximation with greedy algorithms. Achieves up to 55% improvement over alternatives in real-world evaluations.
* `training` `serving` `kernel` [gpu_ext: Extensible OS Policies for GPUs via eBPF](http://arxiv.org/abs/2512.12615v1)
  > **TL;DR**: Addresses inflexible GPU resource management policies for diverse workloads. Proposes gpu_ext, an eBPF-based runtime enabling programmable GPU driver/device policies. Achieves up to 4.8x higher throughput and 2x lower tail latency for inference/training workloads.
* `kernel` `serving` [Principled Performance Tunability in Operating System Kernels](http://arxiv.org/abs/2512.12530v1)
  > **TL;DR**: Addresses the problem of safely tuning Linux kernel performance constants (perf-consts). Proposes KernelX, using Scoped Indirect Execution (SIE) to enable live updates without rebuilds. Achieves millisecond-scale updates and significant performance improvements in case studies.

### 2025-12-13
* `training` `RL` `networking` [HetRL: Efficient Reinforcement Learning for LLMs in Heterogeneous Environments](http://arxiv.org/abs/2512.12476v1)
  > **TL;DR**: Addresses efficient RL training for LLMs in heterogeneous GPU environments. Proposes HetRL system with a constrained joint optimization formulation and a multi-level search scheduling algorithm. Achieves up to 9.17x higher throughput over state-of-the-art systems.
* `edge` `RL` `serving` [A Conflict-Aware Resource Management Framework for the Computing Continuum](http://arxiv.org/abs/2512.12299v1)
  > **TL;DR**: Proposes a DRL-based framework for conflict-aware resource orchestration across edge, fog, and cloud. Integrates real-time performance feedback and historical data to mediate resource conflicts. Achieves efficient resource reallocation and adaptive learning in dynamic Kubernetes environments.
* `inference-side model updates` `resource efficiency` `latency reduction` [Near-Zero-Overhead Freshness for Recommendation Systems via Inference-Side Model Updates](http://arxiv.org/abs/2512.12295v1)
  > **TL;DR**: LiveUpdate introduces inference-side model updates for recommendation systems using Low-Rank Adaptation, reducing synchronization costs and improving freshness with minimal latency impact (<20ms P99) and higher accuracy (0.04-0.24% gain).
* `training` `sparse` [BOOST: BOttleneck-Optimized Scalable Training Framework for Low-Rank Large Language Models](http://arxiv.org/abs/2512.12131v1)
  > **TL;DR**: Proposes BOOST, a training framework for low-rank LLMs with bottleneck-aware tensor parallelism and other optimizations. Addresses poor scalability of low-rank architectures by reducing communication and improving GPU utilization. Achieves 1.46-1.91× speedup over full-rank baselines.

### 2025-12-12
* `video` `training` `RAG` [ECCO: Leveraging Cross-Camera Correlations for Efficient Live Video Continuous Learning](http://arxiv.org/abs/2512.11727v1)
  > **TL;DR**: Introduces ECCO, a framework for efficient continuous learning in multi-camera systems by grouping cameras with correlated data drift for shared model retraining. ECCO includes dynamic grouping, GPU allocation, and transmission control. Achieves 6.7%-18.1% higher accuracy or supports 3.3× more cameras at same resource.
* `serving` `edge` `kernel` [Parallax: Runtime Parallelization for Operator Fallbacks in Heterogeneous Edge Systems](http://arxiv.org/abs/2512.11532v1)
  > **TL;DR**: Addresses inefficient CPU fallbacks for unsupported DNN operators on edge devices. Proposes Parallax, a framework with DAG partitioning, branch-aware memory management, and an adaptive scheduler. Achieves up to 46% latency reduction compared to state-of-the-art frameworks.
* `training` `RL` `offloading` [RollMux: Phase-Level Multiplexing for Disaggregated RL Post-Training](http://arxiv.org/abs/2512.11306v1)
  > **TL;DR**: Proposes RollMux, a cluster scheduling framework for disaggregated RL post-training to reclaim dependency bubbles through cross-cluster orchestration. Uses co-execution groups for two-tier scheduling and warm caching. Achieves 1.84x higher cost efficiency than standard disaggregation on a 656-GPU testbed.
* `kernel` `hardware` `training` [Theoretical Foundations of GPU-Native Compilation for Rapid Code Iteration](http://arxiv.org/abs/2512.11200v1)
  > **TL;DR**: Proposes GPU-native compilation to eliminate CPU-GPU data transfer bottlenecks. Introduces parallel traditional, neural, and hybrid compilation strategies with probabilistic verification. Achieves 10-100x speedups in code iteration cycles through massive parallelism and transfer elimination.

### 2025-12-11
* `serving` `offloading` [ESS: An Offload-Centric Latent-Cache Management Architecture for DeepSeek-V3.2-Exp](http://arxiv.org/abs/2512.10576v1)
  > **TL;DR**: Addresses GPU memory bottleneck in DeepSeek-V3.2-Exp's decode stage caused by linear growth of latent-cache. Proposes ESS, an offload-centric system that selectively moves latent-cache to CPU memory to enable larger batch sizes. Achieves up to 123% throughput improvement at 128K context length.
* `training` `RL` `offline` [Hybrid Learning and Optimization-Based Dynamic Scheduling for DL Workloads on Heterogeneous GPU Clusters](http://arxiv.org/abs/2512.10271v1)
  > **TL;DR**: Proposes RLTune, an RL-based scheduling framework with MILP job mapping for DL workloads on heterogeneous GPU clusters. Aims to optimize GPU utilization, queueing delay, and JCT without per-job profiling. Achieves up to 20% higher GPU utilization, 81% lower queueing delay, and 70% shorter JCT.
* `training` `networking` `kernel` [Design Space Exploration of DMA based Finer-Grain Compute Communication Overlap](http://arxiv.org/abs/2512.10236v1)
  > **TL;DR**: Proposes FiCCO, a finer-grain compute-communication overlap method using DMA offloading for distributed ML training. Introduces heuristics to select optimal schedules by characterizing inefficiencies, achieving up to 1.6× speedup and 81% accuracy in unseen scenarios.

### 2025-12-10
* `edge` `kernel` [Ariel-ML: Computing Parallelization with Embedded Rust for Neural Networks on Heterogeneous Multi-core Microcontrollers](http://arxiv.org/abs/2512.09800v1)
  > **TL;DR**: Presents Ariel-ML, a Rust toolkit for parallelized ANN inference on multi-core microcontrollers. Combines a generic TinyML pipeline with embedded Rust to utilize multi-core capabilities. Achieves lower inference latency than prior art while maintaining comparable memory footprint to C/C++ toolkits.
* `training` `optimization` `scheduling` [Straggler Tolerant and Resilient DL Training on Homogeneous GPUs](http://arxiv.org/abs/2512.09685v1)
  > **TL;DR**: Investigate stragglers in distributed training on GPU clusters. Propose STAR with adaptive synchronization modes and resource reallocation to mitigate stragglers caused by CPU/bandwidth imbalance. Reduces Time-To-Accuracy by 70% compared to state-of-the-art.
* `serving` `offloading` [WarmServe: Enabling One-for-Many GPU Prewarming for Multi-LLM Serving](http://arxiv.org/abs/2512.09472v1)
  > **TL;DR**: Addresses performance degradation in multi-LLM GPU serving due to cold starts. Proposes WarmServe with universal GPU workers, evict-aware placement, proactive prewarming, and zero-overhead memory switching. Achieves 50.8× TTFT improvement and 2.5× request capacity.
* `RAG` `storage` `networking` [Passing the Baton: High Throughput Distributed Disk-Based Vector Search with BatANN](http://arxiv.org/abs/2512.09331v1)
  > **TL;DR**: Presents BatANN, a distributed disk-based ANN system for scalable vector search using batched query state handoff between servers to maintain locality. Achieves 2.5-6.49x higher throughput over baseline while keeping mean latency below 6ms on billion-point datasets.
* `edge` `offloading` `multi-modal` [A Distributed Framework for Privacy-Enhanced Vision Transformers on the Edge](http://arxiv.org/abs/2512.09309v1)
  > **TL;DR**: Proposes a distributed framework for privacy-enhanced Vision Transformers on edge devices. Uses hierarchical offloading to partition visual data across untrusted clouds and aggregates results locally on a trusted edge device. Reduces reconstruction risk while maintaining near-baseline segmentation performance in SAM case study.
* `training` `MoE` `offloading` [Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens](http://arxiv.org/abs/2512.09277v1)
  > **TL;DR**: Proposes METRO for efficient MoE serving in memory-bound regimes; balances activated experts per GPU instead of tokens to reduce memory pressure and improve performance. Achieves up to 22% lower decode latency and 4.11x higher throughput.

### 2025-12-09
* `training` `serving` `kernel` [Magneton: Optimizing Energy Efficiency of ML Systems via Differential Energy Debugging](http://arxiv.org/abs/2512.08365v1)
  > **TL;DR**: Proposes differential energy debugging to identify software-caused energy waste in ML systems. Magneton compares energy use at operator level across similar systems to pinpoint inefficient code/configuration. Applied to LLM inference, reduces energy consumption by up to 47% in diagnosed cases.
* `training` `kernel` [Chopper: A Multi-Level GPU Characterization Tool & Derived Insights Into LLM Training Inefficiency](http://arxiv.org/abs/2512.08242v1)
  > **TL;DR**: Introduces Chopper, a multi-level GPU profiling tool for LLM training analysis. Collects and aligns kernel traces and hardware performance counters across granularities to identify bottlenecks in FSDP. Identifies frequency overhead (DVFS) as largest inefficiency source (exceeding MFMA utilization loss etc.) in Llama 3 8B training.

### 2025-12-08
* `offline` `training` `sparse` [Quantifying the Carbon Reduction of DAG Workloads: A Job Shop Scheduling Perspective](http://arxiv.org/abs/2512.07799v1)
  > **TL;DR**: Quantifies carbon reduction for DAG workloads (e.g., video encoding, offline inference) by modeling as job-shop scheduling. Uses offline solver to compute upper bounds, achieving 25% lower carbon emissions without increased makespan; doubling makespan nearly doubles savings.
* `training` `serving` `networking` [Designing Co-operation in Systems of Hierarchical, Multi-objective Schedulers for Stream Processing](http://arxiv.org/abs/2512.07792v1)
  > **TL;DR**: Investigates scheduling co-operation to optimize resource allocation in hierarchical stream processing systems at Meta. Proposes integration of new schedulers into existing hierarchies for effective load balancing across compute resources. Enables processing terabytes of data within seconds.
* `training` `networking` [Bandwidth-Aware Network Topology Optimization for Decentralized Learning](http://arxiv.org/abs/2512.07536v1)
  > **TL;DR**: Proposes bandwidth-aware network topology optimization for decentralized learning to maximize consensus speed under edge constraints. Uses Mixed-Integer SDP reformulation and ADMM with conjugate gradient for scalability. Reduces training time by 1.21× for heterogeneous bandwidth settings.
* `video` `diffusion` `serving` [Communication-Efficient Serving for Video Diffusion Models with Latent Parallelism](http://arxiv.org/abs/2512.07350v1)
  > **TL;DR**: Addresses communication bottlenecks in video diffusion model serving. Proposes Latent Parallelism (LP), which dynamically rotates partitioning dimensions in latent space to reduce transfers. Achieves up to 97% communication overhead reduction while maintaining generation quality.
* `edge` `RAG` `video` [Venus: An Efficient Edge Memory-and-Retrieval System for VLM-based Online Video Understanding](http://arxiv.org/abs/2512.07344v1)
  > **TL;DR**: Addresses high deployment overhead for VLM-based online video understanding. Proposes Venus, an edge-cloud disaggregated system with hierarchical memory construction and threshold-based sampling for adaptive cost-accuracy tradeoff. Achieves 15x-131x latency speedup while maintaining accuracy.
* `hardware` `offloading` `kernel` [DCO: Dynamic Cache Orchestration for LLM Accelerators through Predictive Management](http://arxiv.org/abs/2512.07312v1)
  > **TL;DR**: Proposes DCO, a shared system-level cache with predictive management for LLM accelerators, using dataflow-guided cache replacement and thrashing mitigation. Achieves up to 1.80x speedup over conventional cache architectures and is validated with RTL implementation at 2 GHz.

### 2025-12-07
* `video` `serving` `offline` [Optimizing video analytics inference pipelines: a case study](http://arxiv.org/abs/2512.07009v1)
  > **TL;DR**: Optimizes video analytics inference pipelines for livestock monitoring. Introduces multi-level parallelization, GPU acceleration, vectorized clustering, and memory-efficient post-processing. Achieves 2x speedup across pipelines without accuracy loss.
* `storage` `serving` [A Chunked-Object Pattern for Multi-Region Large Payload Storage in Managed NoSQL Databases](http://arxiv.org/abs/2512.06852v1)
  > **TL;DR**: Proposes a 'chunked-object' pattern for storing large payloads exceeding NoSQL item size limits using chunking within the database. Eliminates replication lag risks by avoiding offloading to object storage, reducing p99 cross-region consistency latency for 1 MB payloads by keeping data in a single consistency domain.
* `MoE` `edge` `training` [Stable-MoE: Lyapunov-based Token Routing for Distributed Mixture-of-Experts Training over Edge Networks](http://arxiv.org/abs/2512.06784v1)
  > **TL;DR**: Proposes Lyapunov-based token routing for distributed MoE training on edge networks with heterogeneous resources. Formulates stochastic optimization for throughput and gating consistency via online routing/resource allocation. Gains 40% throughput and 5% accuracy on SVHN/CIFAR-100.

### 2025-12-06
* `training` `RL` `sparse` [A-3PO: Accelerating Asynchronous LLM Training with Staleness-aware Proximal Policy Approximation](http://arxiv.org/abs/2512.06547v1)
  > **TL;DR**: Addresses computational bottleneck in asynchronous RL training for LLMs caused by proximal policy. Proposes A-3PO, which approximates the proximal policy via interpolation instead of extra forward pass. Reduces training time by 18%.
* `offline` `quantization` `edge` [Vec-LUT: Vector Table Lookup for Parallel Ultra-Low-Bit LLM Inference on Edge Devices](http://arxiv.org/abs/2512.06443v1)
  > **TL;DR**: Proposes Vec-LUT for efficient parallel ultra-low-bit LLM inference. Replaces scalar LUTs with a vector lookup to reduce bandwidth underutilization via tensor layout and cache-aware techniques. On 5 edge devices, achieves up to 4.2× speedup over baselines.

### 2025-12-05
* `serving` `offloading` `networking` [Metronome: Differentiated Delay Scheduling for Serverless Functions](http://arxiv.org/abs/2512.05703v1)
  > **TL;DR**: Proposes differentiated delay scheduling for serverless FaaS to optimize locality-aware execution. Metronome uses online Random Forest Regression to predict function times and select optimal nodes. Achieves 64.88%-95.83% reduction in mean execution time over baselines.
* `serving` `RL` `RAG` [Model Gateway: Model Management Platform for Model-Driven Drug Discovery](http://arxiv.org/abs/2512.05462v1)
  > **TL;DR**: Proposes Model Gateway, a management platform for ML models in drug discovery. It integrates LLM Agents for dynamic consensus model, model registration, asynchronous execution, and result retrieval. Achieves 0% failure rate with 10k simultaneous clients.

### 2025-12-04
* `offloading` `hardware` `kernel` [Offloading to CXL-based Computational Memory](http://arxiv.org/abs/2512.04449v1)
  > **TL;DR**: Proposes KAI, a system using Asynchronous Back-Streaming protocol to offload operations to CXL-based Computational Memory. Utilizes layered data/control transfers for async data movement and pipelining, reducing end-to-end runtime by up to 50.4% and idle times by 22.11x/3.85x.

### 2025-12-03
* `MoE` `edge` `offloading` [OD-MoE: On-Demand Expert Loading for Cacheless Edge-Distributed MoE Inference](http://arxiv.org/abs/2512.03927v1)
  > **TL;DR**: Proposes OD-MoE, a distributed MoE inference framework for edge devices that eliminates expert caching via on-demand expert loading and an emulative predictor. Achieves 99.94% expert prediction accuracy and 75% decoding speed of fully GPU-cached deployment with only 1/3 GPU memory, enabling sub-1GB deployments.
* `training` `storage` `networking` [FFTrainer: Fast Failover in Large-Language Model Training with Almost-Free State Management](http://arxiv.org/abs/2512.03644v1)
  > **TL;DR**: Addresses recovery inefficiencies and storage overhead in large-scale LLM training. FFTrainer uses surplus network bandwidth for fast state management and failure rollback reduction. Achieves up to 98% faster recovery time and 68% higher GPU utilization preservation.
* `serving` `autoscaling` `resource management` [TokenScale: Timely and Accurate Autoscaling for Disaggregated LLM Serving with Token Velocity](http://arxiv.org/abs/2512.03416v1)
  > **TL;DR**: Proposes TokenScale for autoscaling disaggregated LLM serving using Token Velocity metric and convertible decoders to handle bursty workloads. Achieves SLO attainment up to 96% and reduces costs by 4-14%.

### 2025-12-02
* `serving` `kernel` `inference` [TokenPowerBench: Benchmarking the Power Consumption of LLM Inference](http://arxiv.org/abs/2512.03024v1)
  > **TL;DR**: Investigators tackle the lack of power consumption benchmarks for LLM inference by introducing TokenPowerBench, a tool enabling configuration of model, prompt, and engine; capturing multi-level power metrics and attributing energy per request phase. It quantifies joules per token across varying settings, achieving measurable energy efficiency assessment without specialized hardware.
* `offloading` `training` `storage` [Offloading Artificial Intelligence Workloads across the Computing Continuum by means of Active Storage Systems](http://arxiv.org/abs/2512.02646v1)
  > **TL;DR**: Proposes an active storage architecture for offloading AI workloads across heterogeneous devices. Embeds computation in storage to reduce data transfer and uses dataClay platform. Achieves improved memory efficiency and training speeds while maintaining accuracy.

### 2025-12-01
* `serving` `offloading` `storage` [Tangram: Accelerating Serverless LLM Loading through GPU Memory Reuse and Affinity](http://arxiv.org/abs/2512.01357v1)
  > **TL;DR**: Tackles high cold-start latency in serverless LLM serving via GPU memory reuse. Proposes Tangram with unified GPU memory pool, on-demand KV cache allocation, and affinity-aware scheduling. Achieves 6.2x faster model loading and 23-55% lower Time-To-First-Token.

### 2025-11-29
* `serving` `edge` `offloading` [IslandRun: Privacy-Aware Multi-Objective Orchestration for Distributed AI Inference](http://arxiv.org/abs/2512.00595v1)
  > **TL;DR**: Addresses the challenge of orchestrating LLM inference across heterogeneous devices (edge, cloud) with conflicting goals of performance, privacy, and cost. Proposes IslandRun, a system using agent-based routing and reversible anonymization to route compute to data. Achieves improved privacy and multi-objective optimization.

### 2025-11-28
* `training` `offloading` `edge` [Communication-Computation Pipeline Parallel Split Learning over Wireless Edge Networks](http://arxiv.org/abs/2511.23167v1)
  > **TL;DR**: Proposes a pipeline-parallel split learning method to overlap communication and computation in distributed training over wireless edge networks. Uses micro-batching and jointly optimizes task split and resource allocation. Reduces total training time by over 38% while maintaining accuracy.
* `serving` `RL` `networking` [Serving Heterogeneous LoRA Adapters in Distributed LLM Inference Systems](http://arxiv.org/abs/2511.22880v1)
  > **TL;DR**: Addresses performance skew from serving heterogeneous LoRA adapters. Proposes LoRAServe, a dynamic adapter placement and routing framework that rebalances adapters across GPUs using RDMA for remote access. Achieves up to 2× higher throughput and 50% fewer GPUs under SLOs compared to SOTA.

### 2025-11-27
* `serving` `edge` `networking` [DisCEdge: Distributed Context Management for Large Language Models at the Edge](http://arxiv.org/abs/2511.22599v1)
  > **TL;DR**: Proposes DisCEdge, a distributed context management system for edge LLM serving. It stores/replicates user context as token sequences instead of raw text to reduce synchronization overhead. Improves median response time by 14.46% and reduces client request size by 90%.
* `serving` `MoE` `sparse` [OmniInfer: System-Wide Acceleration Techniques for Optimizing LLM Serving Throughput and Latency](http://arxiv.org/abs/2511.22481v1)
  > **TL;DR**: Proposes OmniInfer, a unified LLM serving system that optimizes throughput and latency via Mixture-of-Experts scheduling and sparse attention acceleration. Built on vLLM, it reduces TPOT by 36% and TTFT by 38% in cluster evaluations.
* `kernel` `serving` [PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel](http://arxiv.org/abs/2511.22333v1)
  > **TL;DR**: Proposes PAT, a prefix-aware attention kernel for LLM decoding that packs queries by shared prefix and uses a multi-tile design to minimize KV cache memory accesses. Achieves 67.4% average attention latency reduction over state-of-the-art kernels.

### 2025-11-26
* `serving` `edge` `offloading` [DSD: A Distributed Speculative Decoding Solution for Edge-Cloud Agile Large Model Serving](http://arxiv.org/abs/2511.21669v1)
  > **TL;DR**: Addresses the challenge of speculative decoding for LLM inference in distributed edge-cloud environments. Introduces DSD, a framework for coordinated draft-target execution across devices, evaluated via a custom simulator and adaptive window control. Achieves up to 1.1x speedup and 9.7% higher throughput.
* `training` `MoE` `offloading` [MemFine: Memory-Aware Fine-Grained Scheduling for MoE Training](http://arxiv.org/abs/2511.21431v1)
  > **TL;DR**: Proposes MemFine, a memory-aware scheduling framework to solve the memory bottleneck in MoE training caused by dynamic token routing. It uses chunk-level decomposition and optimized recomputation guided by a memory model. Reduces activation memory by 48.03% and increases throughput by 4.42% versus full recomputation.
* `serving` `edge` [Automated Dynamic AI Inference Scaling on HPC-Infrastructure: Integrating Kubernetes, Slurm and vLLM](http://arxiv.org/abs/2511.21413v1)
  > **TL;DR**: Proposes a system to dynamically scale LLM inference on HPC infrastructure by integrating vLLM, Slurm, and Kubernetes to handle synchronous user-facing workloads. Achieves efficient scaling for up to 1000 concurrent requests with only ~500 ms end-to-end latency overhead.
* `serving` [A Dynamic PD-Disaggregation Architecture for Maximizing Goodput in LLM Inference Serving](http://arxiv.org/abs/2511.20982v1)
  > **TL;DR**: Proposes DOPD, a dynamic LLM inference system that adjusts prefill-to-decoding instance allocations based on real-time load to resolve workload imbalance. Achieves up to 1.5x higher goodput and reduces P90 TTFT by 67.5% compared to vLLM and DistServe.
* `serving` `agentic` `offline` [Aragog: Just-in-Time Model Routing for Scalable Serving of Agentic Workflows](http://arxiv.org/abs/2511.20975v1)
  > **TL;DR**: Proposes Aragog, a system for cost-aware LLM serving of agentic workflows using just-in-time model routing. It decouples configuration into an offline accuracy-preserving step and an online per-stage scheduler. At peak load, achieves 50-217% higher throughput and 32.5-78.9% lower median latency.

### 2025-11-25
* `serving` `offloading` `hardware` [Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management](http://arxiv.org/abs/2511.20172v1)
  > **TL;DR**: Introduces Beluga, a CXL-based memory architecture for efficient KVCache management in LLM serving. It enables GPUs to natively access a shared memory pool via CXL switches, reducing access latency. Achieves 89.6% lower TTFT and 7.35x higher throughput vs. RDMA-based solutions.
* `training` `kernel` [QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation](http://arxiv.org/abs/2511.20100v1)
  > **TL;DR**: Proposes Macro Thinking Micro Coding (MTMC), a hierarchical LLM framework for generating high-performance GPU kernels. Decouples optimization strategy (reinforcement learning) from implementation (LLM coding). Achieves up to 7.3x speedup over LLMs and 2.2x over expert-optimized PyTorch kernels.
* `training` `networking` `sparse` [ParaBlock: Communication-Computation Parallel Block Coordinate Federated Learning for Large Language Models](http://arxiv.org/abs/2511.19959v1)
  > **TL;DR**: Addresses high communication latency in federated fine-tuning of large LLM blocks. Proposes ParaBlock, a scheme with parallel communication and computation threads to overlap these phases. Reduces communication costs by up to 42% while maintaining model convergence and performance.
* `diffusion` `edge` [Batch Denoising for AIGC Service Provisioning in Wireless Edge Networks](http://arxiv.org/abs/2511.19847v1)
  > **TL;DR**: Optimizes image generation service quality in wireless edge networks under latency constraints. Proposes STACKING, an algorithm for batch denoising that exploits step importance for parallelism. Achieves lower per-step delay and higher service quality compared to baseline methods.

### 2025-11-24
* `edge` `storage` `RAG` [AME: An Efficient Heterogeneous Agentic Memory Engine for Smartphones](http://arxiv.org/abs/2511.19192v1)
  > **TL;DR**: Proposes AME, a hardware-aware vector database engine for on-device agents on smartphones. It co-designs an efficient matrix pipeline and workload scheduler with mobile SoC constraints. Achieves 1.4x higher query throughput and 7x faster index construction.
* `kernel` [Low-Rank GEMM: Efficient Matrix Multiplication via Low-Rank Approximation with FP8 Acceleration](http://arxiv.org/abs/2511.18674v1)
  > **TL;DR**: Proposes Low-Rank GEMM, a matrix multiplication method using low-rank approximations and FP8 precision to reduce complexity and improve hardware efficiency. Achieves a 7.8x speedup and 75% memory savings over PyTorch FP32 on large matrices.

### 2025-11-22
* `edge` `multi-modal` `offloading` [AVERY: Adaptive VLM Split Computing through Embodied Self-Awareness for Efficient Disaster Response Systems](http://arxiv.org/abs/2511.18151v1)
  > **TL;DR**: Presents AVERY, an adaptive split-computing framework for Vision-Language Models on UAVs. It uses a dual-stream split (context/insight) and a self-aware controller to dynamically offload processing. Outperforms baselines with 93.98% lower energy consumption versus full-edge execution.
* `training` `networking` [Pier: Efficient Large Language Model pretraining with Relaxed Global Communication](http://arxiv.org/abs/2511.17849v1)
  > **TL;DR**: Addresses the high cost of global communication (e.g., all-reduce) in LLM pretraining. Proposes Pier, an optimizer with relaxed communication via momentum warmup/decay. Achieves a 2.7x-3.7x speedup in GPT-2 XL training on 256 A100s without performance loss.

### 2025-11-21
* `training` `MoE` `hardware` [Training Foundation Models on a Full-Stack AMD Platform: Compute, Networking, and System Design](http://arxiv.org/abs/2511.17127v1)
  > **TL;DR**: Presents a full-stack system design for training MoE foundation models on AMD MI300X GPUs with Pollara interconnect. Characterizes cluster networking, memory bandwidth, and introduces hardware-aware transformer sizing rules. Achieves competitive model performance with optimized throughput on pure AMD hardware.
* `kernel` `serving` [Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2511.16964v1)
  > **TL;DR**: Investigates multi-agent LLM systems for optimizing PyTorch inference. Proposes a framework to compare different agent strategies, finding exploit-heavy strategies with error-fixing agents work best. Achieves an average 2.88x speedup on an H100 GPU across the KernelBench suite.
* `training` `MoE` [MicroMoE: Fine-Grained Load Balancing for Mixture-of-Experts with Token Scheduling](http://arxiv.org/abs/2511.16947v1)
  > **TL;DR**: Addresses load imbalance in MoE model training. Proposes MicroMoE, a distributed system with MicroEP, a fine-grained parallelization strategy using token scheduling for load balancing. Improves end-to-end training throughput by up to 47.6%.

### 2025-11-20
* `training` `RL` `kernel` [Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter](http://arxiv.org/abs/2511.16665v1)
  > **TL;DR**: Proposes TLT, a system to accelerate reasoning RL training by using adaptive speculative decoding to overcome the bottleneck of long-tail response generation. Achieves over 1.7x end-to-end training speedup while preserving model accuracy.
* `RL` `serving` [Fast LLM Post-training via Decoupled and Best-of-N Speculation](http://arxiv.org/abs/2511.16193v1)
  > **TL;DR**: Addresses the slow rollout phase in LLM post-training (e.g., RLHF). Introduces SpecActor, which uses dynamic decoupled speculation and Best-of-N drafting to maximize GPU efficiency and draft accuracy. Achieves 1.3–1.7× speedup over common post-training baselines.
* `kernel` `hardware` [Can Asymmetric Tile Buffering Be Beneficial?](http://arxiv.org/abs/2511.16041v1)
  > **TL;DR**: Proposes Asymmetric Tile Buffering (ATB) to decouple input and output operand tile dimensions in GEMM, increasing arithmetic intensity. A performance model balances ATB benefits against kernel switching overhead. On AMD XDNA2 AI Engine, achieves 4.54x speedup (4.8 to 24.6 TFLOPS) for BFP16-BF16 GEMM.
* `hardware` `serving` `offloading` [A Scalable NorthPole System with End-to-End Vertical Integration for Low-Latency and Energy-Efficient LLM Inference](http://arxiv.org/abs/2511.15950v1)
  > **TL;DR**: Presents a vertically integrated system with NorthPole accelerators for scalable, low-latency LLM inference. Combines custom hardware, a runtime stack, and containerized pipeline. Delivers 2.8 ms per-user inter-token latency for an 8B model while consuming only 30 kW.

### 2025-11-19
* `kernel` `hardware` `serving` [A Tensor Compiler for Processing-In-Memory Architectures](http://arxiv.org/abs/2511.15503v1)
  > **TL;DR**: Proposes DCC, a data-centric compiler optimizing data layouts and compute code jointly for ML kernels on PIM architectures. It uses a unified tuning process with performance prediction. Achieves up to 7.71x speedup for LLM inference over GPU-only execution.
* `networking` `MoE` `training` [GPU-Initiated Networking for NCCL](http://arxiv.org/abs/2511.15076v1)
  > **TL;DR**: Investigates how to reduce GPU communication latency for AI workloads like MoE models. Proposes GPU-Initiated Networking (GIN), an NCCL extension enabling direct, CPU-bypass GPU-to-GPU communication via RDMA. Achieves lower latency device-initiated operations within NCCL's collective runtime.

### 2025-11-18
* `RL` `serving` [Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning](http://arxiv.org/abs/2511.14617v1)
  > **TL;DR**: Addresses performance bottlenecks in synchronous LLM reinforcement learning. Proposes Seer, a system using online context learning with dynamic load balancing and grouped speculative decoding to exploit prompt similarities. Achieves up to 97% higher rollout throughput and 93% lower long-tail latency.
* `serving` `edge` `networking` [Hyperion: Hierarchical Scheduling for Parallel LLM Acceleration in Multi-tier Networks](http://arxiv.org/abs/2511.14450v1)
  > **TL;DR**: This paper tackles minimizing end-to-end latency for LLM inference in multi-tier networks. Hyperion uses a two-stage framework that jointly optimizes offline model partitioning and online request scheduling. It reduces latency by up to 52.1% compared to GPipe while increasing GPU utilization.
* `training` `offloading` `storage` [10Cache: Heterogeneous Resource-Aware Tensor Caching and Migration for LLM Training](http://arxiv.org/abs/2511.14124v1)
  > **TL;DR**: Addresses memory bottlenecks and tensor migration latency in LLM training. Proposes 10Cache, a system for intelligent tensor caching/migration across GPU, CPU, and NVMe tiers using prefetching and buffer reuse. Achieves up to 2x training speedup and an 86.6x higher GPU cache hit rate.
* `serving` `networking` [FailSafe: High-performance Resilient Serving](http://arxiv.org/abs/2511.14116v1)
  > **TL;DR**: Addresses how to sustain LLM serving performance under GPU failures in tensor-parallel settings. Proposes FailSafe with cyclic KVCache placement, hybrid attention, and dynamic routing, plus proactive backup and recovery. Achieves 2x higher throughput and 100x lower recovery latency vs. standard fault handling.
* `MoE` `offloading` `serving` [MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts](http://arxiv.org/abs/2511.14102v1)
  > **TL;DR**: Addresses the I/O bottleneck of expert offloading in MoE model inference. Proposes MoE-SpeQ, a system that uses a draft model for speculative expert prefetching and an adaptive governor to overlap I/O with computation. Achieves a 2.34x speedup over the state-of-the-art offloading framework.

### 2025-11-17
* `kernel` `inference` `sparse` [MACKO: Sparse Matrix-Vector Multiplication for Low Sparsity](http://arxiv.org/abs/2511.13061v1)
  > **TL;DR**: Proposes MACKO-SpMV, a GPU-optimized kernel and storage format for efficient sparse matrix-vector multiplication at low (30-90%) sparsity levels common in pruned LLMs. Applied to Llama2-7B, it achieves 1.5x memory reduction and 1.5x faster inference at 50% sparsity compared to a dense baseline.

### 2025-11-16
* `kernel` `training` [Iris: First-Class Multi-GPU Programming Experience in Triton](http://arxiv.org/abs/2511.12500v1)
  > **TL;DR**: Presents Iris, a multi-GPU communication library implemented in Python/Triton to simplify and optimize compute-communication overlap. Introduces tile-based symmetric memory abstractions for single-source kernels. Achieves up to 1.79x speedup over PyTorch and RCCL in GEMM+All-Scatter workloads.

### 2025-11-15
* `video` `serving` [PipeDiT: Accelerating Diffusion Transformers in Video Generation with Task Pipelining and Model Decoupling](http://arxiv.org/abs/2511.12056v1)
  > **TL;DR**: Proposes PipeDiT, a pipelining framework to reduce inference latency and memory consumption for DiT-based video generation. Key innovations include pipeline sequence parallelism, decoupling diffusion/VAE modules, and attention co-processing. Achieves 1.06x-4.02x speedups over baseline frameworks.
* `serving` `kernel` `offloading` [Striking the Right Balance between Compute and Copy: Improving LLM Inferencing Under Speculative Decoding](http://arxiv.org/abs/2511.12031v1)
  > **TL;DR**: Proposes BMC, a KV cache allocation method balancing memory copies and compute redundancy for LLM inference. It pre-allocates tensors with extra rows for in-place updates, repurposing redundancy for speculative decoding. Achieves up to 3.2x throughput acceleration over baseline HuggingFace.

### 2025-11-14
* `agentic` `edge` `serving` [UFO$^3$: Weaving the Digital Agent Galaxy](http://arxiv.org/abs/2511.11332v1)
  > **TL;DR**: Presents UFO3, a system for orchestrating LLM-powered agents across heterogeneous devices. Models user requests as mutable distributed DAGs with asynchronous execution and dynamic optimization. Achieves 31% lower end-to-end latency compared to a sequential baseline.
* `edge` `offloading` `networking` [SemanticNN: Compressive and Error-Resilient Semantic Offloading for Extremely Weak Devices](http://arxiv.org/abs/2511.11038v1)
  > **TL;DR**: Addresses how to perform accurate AI inference offloading for resource-constrained IoT devices over unreliable networks. Proposes SemanticNN, a semantic codec with a BER-aware decoder and soft quantization, that is resilient to bit errors. Reduces feature transmission volume by 56.82-344.83x while maintaining accuracy.

### 2025-11-13
* `training` `kernel` [Scalable Synthesis of distributed LLM workloads through Symbolic Tensor Graphs](http://arxiv.org/abs/2511.10480v1)
  > **TL;DR**: Presents STAGE, a framework to synthesize high-fidelity execution traces for modeling distributed LLM training workloads using symbolic tensor graphs. It enables systematic exploration of parallelization strategies and scales to model configurations spanning over 32K GPUs with tensor-level accuracy.
* `training` `hardware` [Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs](http://arxiv.org/abs/2511.09861v1)
  > **TL;DR**: Identifies Lit Silicon, a thermal imbalance causing straggler GPUs in multi-GPU LLM training systems. Proposes detection/mitigation techniques and power management models, including GPU/CPU power optimization. Achieves up to 6% performance and 4% power improvements on two LLM training frameworks.
* `training` [MoFa: A Unified Performance Modeling Framework for LLM Pretraining](http://arxiv.org/abs/2511.09837v1)
  > **TL;DR**: Presents MoFa, a performance modeling framework for distributed LLM pretraining that integrates optimization features and fault tolerance overhead. It uses an enhanced cost model and fault tolerance analysis to guide system tuning. Achieves high prediction accuracy across various scenarios.
* `hardware` `offloading` [Taiji: A DPU Memory Elasticity Solution for In-production Cloud Environments](http://arxiv.org/abs/2511.09936v1)
  > **TL;DR**: Proposes Taiji, a resource-elasticity architecture for Data Processing Units (DPUs) using hybrid virtualization and parallel memory swapping to enable memory overcommitment. Achieves over 50% memory expansion with ~5% virtualization overhead and 90% of swap-ins under 10 microseconds.

### 2025-11-12
* `training` `networking` [TawPipe: Topology-Aware Weight Pipeline Parallelism for Accelerating Long-Context Large Models Training](http://arxiv.org/abs/2511.09741v1)
  > **TL;DR**: Proposes TawPipe, a topology-aware weight pipeline parallelism method for LLM training that optimizes communication hierarchy to reduce cross-node traffic. Groups devices by topology, assigns fixed weight shards, and overlaps communication. Achieves 1.56x higher throughput than WeiPipe on 24 GPUs with long sequences.
* `serving` `networking` [LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations with Fast All-Reduce Communication](http://arxiv.org/abs/2511.09557v2)
  > **TL;DR**: Analyzes bottlenecks in multi-node LLM inference. Proposes NVRAR, a low-latency hierarchical all-reduce algorithm using NVSHMEM and recursive doubling. Reduces communication latency by up to 3.6x and end-to-end batch latency by 1.72x for large models like Llama 3.1 405B.
* `training` `kernel` [No Cords Attached: Coordination-Free Concurrent Lock-Free Queues](http://arxiv.org/abs/2511.09410v1)
  > **TL;DR**: Addresses high coordination overhead in concurrent queues for massively parallel AI workloads. Proposes Cyclic Memory Protection (CMP), a coordination-free lock-free queue using bounded protection windows. Outperforms state-of-the-art queues by 1.72-4x under high contention with hundreds of threads.
* `hardware` `serving` [Flex-MIG: Enabling Distributed Execution on MIG](http://arxiv.org/abs/2511.09143v2)
  > **TL;DR**: Addresses GPU underutilization from rigid NVIDIA MIG partitioning. Proposes Flex-MIG, a software framework enabling one-to-many allocation and shared-memory collectives for distributed execution across MIG instances. Improves cluster makespan by up to 17% on diverse traces.

### 2025-11-11
* `kernel` `sparse` [\uline{LO}w-c\uline{O}st yet High-\uline{P}erformant \uline{S}parse Matrix-Matrix Multiplication on Arm SME Architectures](http://arxiv.org/abs/2511.08158v1)
  > **TL;DR**: Proposes LOOPS, a hybrid framework for efficient SpMM on Arm SME architectures. Combines row-wise and vector-wise layouts to co-utilize NEON and SME units via an adaptive parallelization scheme. Achieves up to 14.4x (FP64) speedup over the baseline TACO on CPU.

### 2025-11-06
* `quantization` `sparse` `edge` [Enabling Dynamic Sparsity in Quantized LLM Inference](http://arxiv.org/abs/2511.04477v1)
  > **TL;DR**: Proposes techniques to enable dynamic sparsity in quantized LLM inference for edge devices. Key designs include a zigzag-patterned quantization layout and a specialized GEMV kernel for commodity GPUs. Achieves up to 1.55x faster decoding throughput while maintaining accuracy.

### 2025-11-05
* `edge` `serving` `kernel` [UMDAM: A Unified Data Layout and DRAM Address Mapping for Heterogenous NPU-PIM](http://arxiv.org/abs/2511.03293v1)
  > **TL;DR**: Proposes UMDAM, a unified data layout and DRAM mapping scheme for NPU-PIM co-execution to optimize LLM inference on edge devices. Uses a column-major, tile-based layout to eliminate bandwidth loss and redundant storage. Reduces TTFT by up to 3.0x and TTLT by 2.18x.
* `serving` `offloading` [SnapStream: Efficient Long Sequence Decoding on Dataflow Accelerators](http://arxiv.org/abs/2511.03092v1)
  > **TL;DR**: Proposes SnapStream, a KV cache compression method for efficient long-sequence LLM inference with continuous batching. It integrates sparsification into static-graph dataflow accelerators. Achieves 4× improved on-chip memory usage while maintaining throughput of 1832 tokens/sec for DeepSeek-671B at 128k context length.

### 2025-11-04
* `serving` `edge` `networking` [Federated Attention: A Distributed Paradigm for Collaborative LLM Inference over Edge Networks](http://arxiv.org/abs/2511.02647v1)
  > **TL;DR**: Proposes Federated Attention (FedAttn), a distributed LLM inference framework for edge networks that performs local self-attention and periodic KV matrix aggregation to preserve privacy and reduce communication. Achieves a 3.2x reduction in communication cost while maintaining response quality.
* `serving` `kernel` [From Models to Operators: Rethinking Autoscaling Granularity for Large Generative Models](http://arxiv.org/abs/2511.02248v1)
  > **TL;DR**: Proposes operator-level autoscaling for large generative models to replace inefficient model-level scaling. The framework profiles individual model operators and independently scales resources for each. Achieves up to 40% fewer GPUs while preserving SLOs, or 1.6x higher throughput under fixed resources.
* `training` `kernel` `networking` [Eliminating Multi-GPU Performance Taxes: A Systems Approach to Efficient Distributed LLMs](http://arxiv.org/abs/2511.02168v1)
  > **TL;DR**: Addresses performance inefficiencies in distributed LLM execution across multiple GPUs due to the Bulk Synchronous Parallel (BSP) model. Proposes fine-grained programming patterns using in-kernel communication primitives to replace global barriers with dataflow synchronization. Achieves a 10-20% speedup in end-to-end latency.
* `serving` `offloading` `agentic` [Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live](http://arxiv.org/abs/2511.02230v1)
  > **TL;DR**: Proposes Continuum, a serving system optimizing multi-turn LLM agent job completion time by predicting tool call durations to set KV cache time-to-live and enable program-level scheduling. Reduces average job completion time and prevents scheduling bubbles for real-world agent workloads.

### 2025-11-03
* `kernel` `training` `serving` [Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects](http://arxiv.org/abs/2511.02132v1)
  > **TL;DR**: Addresses performance degradation of attention kernels on disaggregated GPUs due to NUMA effects. Proposes Swizzled Head-first Mapping, a scheduling strategy aligning attention heads with NUMA domains for cache reuse. Achieves 50% higher performance on AMD MI300X with L2 cache hit rates of 80-97%.

### 2025-11-02
* `serving` `networking` [FREESH: Fair, Resource- and Energy-Efficient Scheduling for LLM Serving on Heterogeneous GPUs](http://arxiv.org/abs/2511.00807v1)
  > **TL;DR**: Explores fair, energy/carbon-efficient LLM serving on distributed, heterogeneous GPUs. Proposes FREESH, a joint routing and scheduling system that optimizes query placement and GPU frequency based on spatiotemporal power/carbon variability. Achieves 45.45% emissions reduction and 28.6% energy saving while improving SLO attainment.
* `RL` `training` [AReaL-Hex: Accommodating Asynchronous RL Training over Heterogeneous GPUs](http://arxiv.org/abs/2511.00796v1)
  > **TL;DR**: Introduces AReaL-Hex, a heterogeneity-aware system for asynchronous RL training on heterogeneous GPUs. It uses a two-phase scheduler (MILP-based planning and graph partitioning) to assign stages optimally. On reasoning tasks, it achieves 1.50x higher throughput or 1.46x cost reduction versus homogeneous deployments.

### 2025-11-01
* `serving` `edge` `multi-modal` [EPARA: Parallelizing Categorized AI Inference in Edge Clouds](http://arxiv.org/abs/2511.00603v1)
  > **TL;DR**: Proposes EPARA, an end-to-end parallel inference framework for edge clouds that categorizes AI tasks by latency sensitivity and GPU needs. It uses a task allocator, distributed handler, and state-aware scheduler to improve resource allocation. Achieves up to 2.1x higher goodput compared to prior frameworks.

### 2025-10-31
* `networking` `serving` `RL` [RDMA Point-to-Point Communication for LLM Systems](http://arxiv.org/abs/2510.27656v1)
  > **TL;DR**: Addresses the need for flexible point-to-point communication in LLM systems using diverse NICs. Proposes TransferEngine, a uniform interface that exposes one-sided WriteImm operations with completion primitives. Achieves 400 Gbps throughput and enables 1.3-second RL weight updates for trillion-parameter models.
* `training` `multi-modal` [Synergistic Tensor and Pipeline Parallelism](http://arxiv.org/abs/2510.27257v1)
  > **TL;DR**: Proposes a synergistic schedule for tensor and pipeline parallelism to reduce communication and synchronization bubbles during LLM/MLLM training. Decouples forward/backward passes into fine-grained units and braids them into a composite sequence. Improves training throughput by up to 16% over baselines.
* `serving` `offloading` [SERFLOW: A Cross-Service Cost Optimization Framework for SLO-Aware Dynamic ML Inference](http://arxiv.org/abs/2510.27182v1)
  > **TL;DR**: Addresses cost-efficient, SLO-aware serving for dynamic ML inference where requests may exit early. Proposes SERFLOW, a framework that offloads model stages between IaaS VMs and FaaS functions using stage-specific provisioning and adaptive load balancing. Reduces cloud costs by over 23%.
* `serving` `RL` [Glia: A Human-Inspired AI for Automated Systems Design and Optimization](http://arxiv.org/abs/2510.27176v1)
  > **TL;DR**: Presents Glia, an automated AI system for designing computer systems using a multi-agent LLM workflow with reasoning and experimentation. Applied to distributed GPU clusters for LLM inference, it generates novel algorithms for routing and scheduling, achieving human-expert level performance in less time.

### 2025-10-30
* `training` `agentic` `RL` [FlowMesh: A Service Fabric for Composable LLM Workflows](http://arxiv.org/abs/2510.26913v1)
  > **TL;DR**: Proposes FlowMesh, a service fabric for fine-grained execution of composite LLM workflows (e.g., RLHF, agent workflows). It decomposes workflows into operators for cross-user deduplication, batching, and global scheduling. Achieves up to 3.8x cost reduction compared to baseline pipelines.
* `serving` `MoE` `offloading` [ExpertFlow: Adaptive Expert Scheduling and Memory Coordination for Efficient MoE Inference](http://arxiv.org/abs/2510.26730v1)
  > **TL;DR**: Proposes ExpertFlow, a runtime system for MoE inference that uses adaptive expert prefetching and cache-aware routing to reduce latency from parameter transfers. It dynamically predicts expert needs to minimize cache misses, reducing model stall time to <0.1% of the baseline.
* `training` `sparse` `networking` [An All-Reduce Compatible Top-K Compressor for Communication-Efficient Distributed Learning](http://arxiv.org/abs/2510.26709v2)
  > **TL;DR**: Proposes ARC-Top-K, an All-Reduce-compatible gradient compressor for distributed training. It aligns sparsity patterns via a lightweight sketch to enable index-free communication. Achieves up to 60.7% reduction in wall-clock training time while matching Top-K accuracy.
* `RL` `training` `serving` [ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems](http://arxiv.org/abs/2510.26475v1)
  > **TL;DR**: Adapts speculative decoding to accelerate generation in RL training systems. Proposes ReSpec with dynamic SD tuning, drafter distillation, and reward-weighted updates to mitigate staleness and policy drift. Achieves up to 4.5x speedup on Qwen models (3B-14B) while preserving reward convergence.

### 2025-10-29
* `MoE` `serving` `networking` [MoEntwine: Unleashing the Potential of Wafer-scale Chips for Large-scale Expert Parallel Inference](http://arxiv.org/abs/2510.25258v1)
  > **TL;DR**: Proposes ER-Mapping and NI-Balancer to optimize MoE inference on wafer-scale chips by co-designing expert and attention layer mappings and overlapping expert migration with idle network links, achieving 62% communication reduction and 39% higher per-device performance than NVL72.

### 2025-10-27
* `edge` `offloading` `serving` [Bayes-Split-Edge: Bayesian Optimization for Constrained Collaborative Inference in Wireless Edge Systems](http://arxiv.org/abs/2510.23503v1)
  > **TL;DR**: Proposes Bayes-Split-Edge, a Bayesian optimization framework for jointly optimizing neural network split points and transmission power in edge-served inference to meet energy and latency constraints. Achieves 2.4x reduction in evaluation cost and near-linear convergence with as few as 20 function evaluations.
* `offloading` `edge` `serving` [Rethinking Inference Placement for Deep Learning across Edge and Cloud Platforms: A Multi-Objective Optimization Perspective and Future Directions](http://arxiv.org/abs/2510.22909v1)
  > **TL;DR**: Addresses optimal placement of DL model inference across edge and cloud to balance latency, cost, and privacy. Proposes a multi-objective optimization framework for partitioning and offloading, enabling tailored deployment for latency-sensitive applications like chatbots.

### 2025-10-23
* `training` `networking` `hardware` [Collective Communication for 100k+ GPUs](http://arxiv.org/abs/2510.20171v1)
  > **TL;DR**: Addresses communication bottlenecks in training LLMs at 100k+ GPU scales. Proposes NCCLX, a collective communication framework optimizing all-reduce and all-gather for ultra-scale clusters, achieving significant throughput and latency improvements over standard NCCL on Llama4 training.
* `training` `MoE` `offloading` [AsyncHZP: Hierarchical ZeRO Parallelism with Asynchronous Scheduling for Scalable LLM Training](http://arxiv.org/abs/2510.20111v1)
  > **TL;DR**: Proposes AsyncHZP, an asynchronous hierarchical ZeRO variant that reduces communication overhead and improves memory efficiency in large-scale LLM training. By adaptively resharding and overlapping communication with computation via multi-stream scheduling, it achieves superior scalability over ND parallelism on both dense and MoE models.

### 2025-10-22
* `MoE` `training` `networking` [HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid Expert/Data Transmission](http://arxiv.org/abs/2510.19470v1)
  > **TL;DR**: Addresses scalability limits of MoE training across datacenters due to low cross-DC bandwidth. Proposes HybridEP, a framework that dynamically optimizes expert and data communication patterns via modeling-guided hybrid transmission, achieving up to 5.6x speedup under constrained bandwidth.
* `training` `MoE` `networking` [RailS: Load Balancing for All-to-All Communication in Distributed Mixture-of-Experts Training](http://arxiv.org/abs/2510.19262v2)
  > **TL;DR**: Addresses all-to-all communication bottlenecks in distributed MoE training by leveraging Rail topology symmetry to enable local, topology-aware load balancing. RailS uses multipath spraying and LPT scheduling to reduce iteration time by up to 40% and boost bus bandwidth by up to 78%.
* `RL` `serving` `offloading` [RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs](http://arxiv.org/abs/2510.19225v1)
  > **TL;DR**: RLBoost improves cost-efficiency of LLM reinforcement learning by harvesting preemptible GPUs for rollout stages, using adaptive offloading, pull-based weight transfer, and token-level response migration. It achieves 1.51x–1.97x higher throughput and 28%–49% better cost efficiency than on-demand-only setups.

### 2025-10-21
* `training` `sparse` `distributed` [MTraining: Distributed Dynamic Sparse Attention for Efficient Ultra-Long Context Training](http://arxiv.org/abs/2510.18830v1)
  > **TL;DR**: Addresses inefficient training of LLMs with ultra-long contexts due to computational imbalance. Proposes MTraining, a distributed method with dynamic sparse attention and balanced ring attention, achieving 6× higher training throughput while scaling context from 32K to 512K tokens on 32 A100 GPUs.
* `serving` `offloading` [Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications](http://arxiv.org/abs/2510.18586v1)
  > **TL;DR**: Addresses KV-cache inefficiencies in LLM-based multi-agent serving due to memory contention and idle stalls. Tokencake introduces agent-aware space and time schedulers with proactive KV-cache offloading and predictive upload, reducing end-to-end latency by 47.06% and boosting GPU memory utilization by 16.9% over vLLM.
* `serving` `edge` `thinking` [SLICE: SLO-Driven Scheduling for LLM Inference on Edge Computing Devices](http://arxiv.org/abs/2510.18544v1)
  > **TL;DR**: Addresses SLO-aware LLM inference scheduling on edge devices by optimizing TTFT, TPOT, and end-to-end latency. Proposes SLICE, a utility-driven scheduler with dynamic generation rate control, achieving up to 35× higher SLO attainment than Orca and FastServe.

### 2025-10-20
* `training` `kernel` `sparse` [Efficient Long-context Language Model Training by Core Attention Disaggregation](http://arxiv.org/abs/2510.18121v1)
  > **TL;DR**: Proposes Core Attention Disaggregation (CAD) to eliminate load imbalance in long-context LLM training by offloading attention computation to dedicated devices. Leverages stateless, composable attention kernels with dynamic rebatching, achieving up to 1.35x training throughput on 512k context lengths.

### 2025-10-19
* `serving` `offloading` `thinking` [Justitia: Fair and Efficient Scheduling for LLM Applications](http://arxiv.org/abs/2510.17015v1)
  > **TL;DR**: Addresses unfair and inefficient scheduling of LLM applications in shared GPU environments. Proposes Justitia, a memory-centric scheduler using neural demand prediction and virtual-time fair queuing to balance efficiency and worst-case guarantees, reducing completion time by up to 3.2× while ensuring fairness compared to vLLM.

### 2025-10-18
* `networking` `training` [Reimagining RDMA Through the Lens of ML](http://arxiv.org/abs/2510.16606v1)
  > **TL;DR**: Celeris redesigns RDMA transport for distributed ML training by eliminating strict reliability and in-order delivery, leveraging ML's fault tolerance to reduce tail latency. By removing retransmissions and using software-level recovery (e.g., Hadamard Transform), it cuts 99th-percentile latency by 2.3x and reduces BRAM usage by 67%.
* `serving` `offloading` `quantization` [FourierCompress: Layer-Aware Spectral Activation Compression for Efficient and Accurate Collaborative LLM Inference](http://arxiv.org/abs/2510.16418v1)
  > **TL;DR**: Addresses communication bottlenecks in collaborative LLM inference by compressing intermediate activations across edge-client and server boundaries. Proposes FourierCompress, a layer-aware FFT-based method retaining low-frequency coefficients, achieving 7.6x compression with <0.3% accuracy loss and 32x faster compression than Top-k.
* `training` `offloading` `sparse` [MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization](http://arxiv.org/abs/2510.16415v1)
  > **TL;DR**: Proposes MeCeFO, a fault-tolerant LLM training system that transfers failed node workloads to neighbors with minimal overhead using skip-connections, recomputation, and low-rank gradient approximation. Achieves 5.0–6.7× higher resilience than SOTA with only 4.18% throughput drop under failures.

### 2025-10-17
* `training` `serving` `hardware` [GOGH: Correlation-Guided Orchestration of GPUs in Heterogeneous Clusters](http://arxiv.org/abs/2510.15652v1)
  > **TL;DR**: Proposes GOGH, a learning-based system for adaptive GPU resource orchestration in heterogeneous clusters to minimize energy and meet performance targets. Uses two neural networks to predict model-hardware compatibility and co-location effects, improving allocation over time. Reduces energy consumption by up to 27% while maintaining SLOs.
* `training` `kernel` `sparse` [PRISM: Probabilistic Runtime Insights and Scalable Performance Modeling for Large-Scale Distributed Training](http://arxiv.org/abs/2510.15596v1)
  > **TL;DR**: PRISM models stochastic performance variability in large-scale distributed training, offering probabilistic guarantees on training time. It identifies communication kernels (e.g., AllGather, ReduceScatter) as primary variability sources and enables 1.26x performance gains via placement-aware optimization, with 20.8% KS distance in prediction accuracy.
* `serving` `thinking` [BeLLMan: Controlling LLM Congestion](http://arxiv.org/abs/2510.15330v1)
  > **TL;DR**: Addresses LLM inference congestion due to uncontrolled autoregressive token generation. Proposes beLLMan, a feedback controller that dynamically adjusts output length based on system load, reducing end-to-end latency by up to 8× and energy use by 25% while increasing request throughput by 19%.

### 2025-10-16
* `serving` `multi-modal` `offloading` [xLLM Technical Report](http://arxiv.org/abs/2510.14686v1)
  > **TL;DR**: Designs xLLM, a high-performance LLM serving framework with decoupled service-engine architecture for multimodal inference, featuring dynamic PD/EPD disaggregation and global KV cache management. Achieves up to 2.2x higher throughput than vLLM-Ascend under identical TPOT constraints on Qwen models.
* `training` `offloading` `hardware` [ScalePool: Hybrid XLink-CXL Fabric for Composable Resource Disaggregation in Unified Scale-up Domains](http://arxiv.org/abs/2510.14580v1)
  > **TL;DR**: Proposes ScalePool, a hybrid XLink-CXL fabric for disaggregated memory and accelerator interconnection in LLM training clusters. Achieves 1.22x average speedup and 4.5x lower latency for memory-intensive workloads by enabling coherent, tiered memory pooling beyond RDMA.
* `serving` `offline` [FairBatching: Fairness-Aware Batch Formation for LLM Inference](http://arxiv.org/abs/2510.14392v1)
  > **TL;DR**: Addresses computational unfairness in LLM inference batching by redesigning batch formation to balance prefill and decode tasks. Introduces adaptive budgeting and dynamic scheduling to reduce TTFT tail latency by up to 2.29x while improving single-node and cluster capacity by 20.0% and 54.3%, respectively.

### 2025-10-15
* `serving` `offloading` `networking` [FIRST: Federated Inference Resource Scheduling Toolkit for Scientific AI Model Access](http://arxiv.org/abs/2510.13724v1)
  > **TL;DR**: FIRST enables federated LLM inference across distributed HPC clusters using a cloud-like API, auto-scaling resources and maintaining hot nodes for low-latency serving. It achieves scalable, on-premises generation of billions of tokens daily without commercial cloud reliance.
* `serving` `offloading` `thinking` [Adaptive Rescheduling in Prefill-Decode Disaggregated LLM Inference](http://arxiv.org/abs/2510.13668v1)
  > **TL;DR**: Addresses workload imbalance in disaggregated LLM inference due to unpredictable decode lengths. Proposes ARES, a system using LLM-internal state to predict output length and dynamically reschedule decode tasks, reducing P99 TPOT by 74.77% and improving goodput by 2.24×.
* `quantization` `hardware` `edge` [F-BFQ: Flexible Block Floating-Point Quantization Accelerator for LLMs](http://arxiv.org/abs/2510.13401v1)
  > **TL;DR**: Designs a hardware accelerator (F-BFQ) to efficiently execute mixed block-floating-point quantized LLMs on edge devices, dynamically switching between BFP variants without reconfiguration. Achieves 1.4x faster inference than Arm NEON CPU and 5.2 tokens/s on AMD Kria.
* `serving` `offloading` `RAG` [BanaServe: Unified KV Cache and Dynamic Module Migration for Balancing Disaggregated LLM Serving in AI Infrastructure](http://arxiv.org/abs/2510.13223v1)
  > **TL;DR**: Addresses load imbalance in disaggregated LLM serving by dynamically migrating KV cache and model layers between prefill and decode nodes. Enables cache-agnostic scheduling with overlapped transmission, achieving 1.2x-3.9x higher throughput and up to 78.4% lower latency than vLLM.

### 2025-10-14
* `RL` `training` `offloading` [Laminar: A Scalable Asynchronous RL Post-Training Framework](http://arxiv.org/abs/2510.12633v1)
  > **TL;DR**: Addresses scalability bottlenecks in RL post-training of LLMs caused by trajectory latency skew. Proposes Laminar, a fully decoupled architecture with tiered relay workers for asynchronous parameter updates and dynamic trajectory repackaging. Achieves up to 5.48× training throughput speedup on a 1024-GPU cluster.

### 2025-10-13
* `training` `serving` `offloading` [An Explorative Study on Distributed Computing Techniques in Training and Inference of Large Language Models](http://arxiv.org/abs/2510.11211v1)
  > **TL;DR**: Explores distributed computing techniques for training and serving LLMs, including system modifications to enable consumer-grade deployment and a comparative analysis of three serving frameworks. Implements a metaheuristic-based offloading method that reduces memory usage by up to 40% on consumer hardware.

### 2025-10-12
* `training` `sparse` `offloading` [DCP: Addressing Input Dynamism In Long-Context Training via Dynamic Context Parallelism](http://arxiv.org/abs/2510.10620v1)
  > **TL;DR**: Addresses dynamic sequence length and attention pattern variability in long-context LLM training. Proposes DCP, a fine-grained dynamic context parallelism framework that adapts data and computation blocks to device resources, achieving up to 1.46x end-to-end training speed-up over static methods.

### 2025-10-11
* `serving` `offloading` `MoE` [SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference](http://arxiv.org/abs/2510.10302v1)
  > **TL;DR**: Proposes SP-MoE, an SD-aware offloading framework for MoE-based LLM inference that prefetches experts using draft-target model correspondence and pipelines I/O to reduce latency. Achieves 1.07–3.5× speedup in tokens-per-token (TPOT) over state-of-the-art methods.

### 2025-10-10
* `serving` `hardware` [SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inference](http://arxiv.org/abs/2510.08544v1)
  > Designs specialized hardware chips for disaggregated LLM inference, tailoring prefill (compute-heavy) and decode (memory-heavy) stages to their distinct workloads. SPAD achieves 19%-41% lower hardware cost and 2%-17% lower TDP than GPU-based systems while maintaining performance.

### 2025-10-09
* `serving` `MoE` `offloading` [From Tokens to Layers: Redefining Stall-Free Scheduling for LLM Serving with Layered Prefill](http://arxiv.org/abs/2510.08055v1)
  > Addresses high energy and latency in MoE LLM serving due to redundant expert weight loads during chunked prefill. Proposes layered prefill, which schedules by layer groups instead of tokens to eliminate reloads. Reduces TTFT by up to 70% and per-token energy by 22% while maintaining stall-free decoding.

### 2025-10-08
* `kernel` [Vectorized FlashAttention with Low-cost Exponential Computation in RISC-V Vector Processors](http://arxiv.org/abs/2510.06834v1)
  > **TL;DR**: Vectorizes FlashAttention on RISC-V vector processors using low-cost exponential approximations and tiling to optimize attention kernels. Achieves significant performance gains without custom ISA extensions.
* `RL` `training` [EARL: Efficient Agentic Reinforcement Learning Systems for Large Language Models](http://arxiv.org/abs/2510.05943v1)
  > **TL;DR**: Addresses scalability bottlenecks in agentic RL training for LLMs by dynamically adapting parallelism and decentralizing intermediate data exchange. Achieves 2.1× higher throughput and eliminates OOM failures during long-context RL training.

### 2025-10-01 ~ 2025-10-07
* `agentic` `offloading` `serving` [Toward Systems Foundations for Agentic Exploration](http://arxiv.org/abs/2510.05556v1)
  > **TL;DR**: Addresses the need for efficient execution branching in LLM agents. Proposes foundational support for fork semantics, side-effect isolation, and microsecond-scale forking. Enables scalable agentic exploration with order-of-magnitude faster snapshot/restore than existing tools.
* `MoE` `serving` [Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting](http://arxiv.org/abs/2510.05497v1)
  > **TL;DR**: Addresses data movement bottlenecks in large-scale MoE LLM serving by forecasting expert selection patterns via massive profiling, enabling architectural optimizations that achieve up to 6.3X speedup on DeepSeek V3.
* `training` `offloading` [OptPipe: Memory- and Scheduling-Optimized Pipeline Parallelism for LLM Training](http://arxiv.org/abs/2510.05186v1)
  > **TL;DR**: Optimizes pipeline parallelism for LLM training by formulating scheduling as a constrained optimization problem that balances memory, computation, and bubble minimization. Dynamically adjusts activation offloading and scheduling to reduce idle time by up to 50% under fixed memory constraints.
* `agentic` `serving` `thinking` [Speculative Actions: A Lossless Framework for Faster Agentic Systems](http://arxiv.org/abs/2510.04371v1)
  > **TL;DR**: Proposes speculative actions to accelerate agentic systems by predicting future actions with faster models, enabling parallel execution. Achieves up to 55% accuracy in action prediction and significantly reduces end-to-end latency in real-world environments.
* `serving` `thinking` `MoE` [SATER: A Self-Aware and Token-Efficient Approach to Routing and Cascading](http://arxiv.org/abs/2510.05164v1)
  > **TL;DR**: How to reduce cost and latency of LLM inference while maintaining performance? SATER introduces a self-aware routing and cascade framework with confidence-aware rejection and preference optimization, cutting computational cost by >50% and cascade latency by >80%.
* `diffusion` `MoE` [Paris: A Decentralized Trained Open-Weight Diffusion Model](http://arxiv.org/abs/2510.03434v1)
  > **TL;DR**: Can high-quality diffusion models be trained decentralized without gradient synchronization? Paris uses 8 isolated expert diffusion models with a router for inference, achieving comparable quality to centralized baselines using 16× less compute and 14× less data.
* `serving` `diffusion` [TridentServe: A Stage-level Serving System for Diffusion Pipelines](http://arxiv.org/abs/2510.02838v1)
  > **TL;DR**: Addresses inefficient static serving of diffusion pipelines by introducing dynamic stage-level resource allocation. TridentServe co-optimizes stage placement and request routing, achieving up to 4.1x lower P95 latency while improving SLO attainment.
* `MoE` `serving` [ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models](http://arxiv.org/abs/2510.02613v1)
  > **TL;DR**: Enables fine-grained, zero-downtime scaling of MoE LLMs during inference by decoupling execution from memory operations and using zero-copy remapping. Achieves up to 9× lower scale-up latency and 2× higher throughput during scaling.
* `serving` `diffusion` [TetriServe: Efficient DiT Serving for Heterogeneous Image Generation](http://arxiv.org/abs/2510.01565v1)
  > **TL;DR**: Addresses efficient serving of DiT models under heterogeneous SLOs by introducing step-level sequence parallelism and round-based scheduling. TetriServe dynamically adjusts parallelism per request step, achieving up to 32% higher SLO attainment while maintaining image quality.
* `agentic` `serving` [FlashResearch: Real-time Agent Orchestration for Efficient Deep Research](http://arxiv.org/abs/2510.05145v1)
  > **TL;DR**: How to accelerate deep research agents by parallelizing sequential reasoning? FlashResearch dynamically decomposes queries into tree-structured tasks and orchestrates parallel execution across breadth and depth, achieving 5x speedup while maintaining report quality.
* `training` `networking` [An Efficient, Reliable and Observable Collective Communication Library in Large-scale GPU Training Clusters](http://arxiv.org/abs/2510.00991v1)
  > **TL;DR**: Designs ICCL, a collective communication library for large-scale LLM training, to improve P2P efficiency, tolerate NIC failures, and enable microsecond-level anomaly observability, achieving 28.5% lower latency and 6.02% higher training throughput than NCCL.
* `training` `offloading` [ElasWave: An Elastic-Native System for Scalable Hybrid-Parallel Training](http://arxiv.org/abs/2510.00606v3)
  > **TL;DR**: Designs an elastic-native LLM training system that maintains parameter consistency and low recovery time during dynamic scaling. Introduces multi-dimensional scheduling, online resharding, and asynchronous migration with in-memory snapshots. Achieves 1.6× higher throughput and 51% lower MTTR than baselines.
* `MoE` `training` [FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training](http://arxiv.org/abs/2510.00207v2)
  > **TL;DR**: FlowMoE develops a unified pipeline scheduling framework for distributed MoE training by integrating MHA, gating, expert computation, and communication. It uses tensor chunk-based priority scheduling to overlap all-reduce with computing, reducing training time by up to 57%.
* `training` `kernel` [LoRAFusion: Efficient LoRA Fine-Tuning for LLMs](http://arxiv.org/abs/2510.00206v1)
  > **TL;DR**: Improves LoRA fine-tuning efficiency by fusing memory-bound ops via kernel optimization and scheduling multiple LoRA adapters with adaptive batching. Achieves up to 1.96× speedup over Megatron-LM and 1.46× over mLoRA.
* `networking` `training` `serving` [Lattica: A Decentralized Cross-NAT Communication Framework for Scalable AI Inference and Training](http://arxiv.org/abs/2510.00183v2)
  > **TL;DR**: Designs a decentralized cross-NAT framework to enable scalable AI training and inference without centralized infrastructure. Uses NAT traversal, CRDTs, and DHT-based discovery for peer-to-peer model synchronization. Achieves reliable communication in permissionless environments with low latency and high throughput.
* `serving` [TASP: Topology-aware Sequence Parallelism](http://arxiv.org/abs/2509.26541v2)
  > **TL;DR**: Addresses inefficient communication in sequence parallelism for long-context LLMs by decomposing Ring AllGather into topology-aware concurrent ring paths. TASP exploits AlltoAll accelerator topology to boost communication efficiency, achieving up to 3.58x speedup over Ring Attention.
* `serving` `offloading` [Parallax: Efficient LLM Inference Service over Decentralized Environment](http://arxiv.org/abs/2509.26182v1)
  > **TL;DR**: How to efficiently serve LLMs over decentralized, heterogeneous GPU pools? Parallax uses a two-phase scheduler for layer-wise model allocation and dynamic pipeline construction, reducing latency by up to 40% and improving throughput versus decentralized baselines.

### 2025-09-16 ~ 2025-09-30
* `serving` `storage` [Accelerating LLM Inference with Precomputed Query Storage](http://arxiv.org/abs/2509.25919v1)
  > **TL;DR**: Reduces LLM inference latency by precomputing and storing response pairs for predictable queries. Uses LLM-driven query generation and disk-backed vector indexing for efficient retrieval. Achieves up to 17.3% latency reduction without compromising response quality.
* `edge` `hardware` `kernel` [Context-Driven Performance Modeling for Causal Inference Operators on Neural Processing Units](http://arxiv.org/abs/2509.25155v1)
  > **TL;DR**: How to enable efficient long-context LLM inference on edge NPUs? Analyzes quadratic vs. sub-quadratic attention operators on NPUs, identifying memory-bound vs. compute-bound bottlenecks, and proposes hardware-aware model co-design. Achieves up to 95% reduction in pipeline stalls for long contexts.
* `MoE` `serving` [GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference](http://arxiv.org/abs/2509.25041v1)
  > **TL;DR**: Addresses high communication overhead and load imbalance in distributed MoE inference. Proposes GRACE-MoE with expert grouping, dynamic replication, and locality-aware routing to co-optimize communication and computation. Achieves up to 3.79x latency reduction over SOTA systems.
* `MoE` `serving` `load-balancing` [From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing](http://arxiv.org/abs/2510.03293v1)
  > **TL;DR**: How to improve MoE inference efficiency without retraining? LASER dynamically balances expert load using gate score distributions, routing tokens to least-loaded experts when scores are ambiguous. Achieves up to 30% lower latency and higher throughput on Mixtral and DeepSeek-MoE with near-zero accuracy drop.
* `training` [HAPT: Heterogeneity-Aware Automated Parallel Training on Heterogeneous Clusters](http://arxiv.org/abs/2509.24859v1)
  > **TL;DR**: HAPT automates parallel training on heterogeneous GPU clusters by optimizing inter-operator parallel strategies and adaptive 1F1B scheduling to maximize computation-communication overlap, achieving 1.3x-1.6x higher training throughput than existing frameworks.
* `serving` `offloading` [SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving](http://arxiv.org/abs/2509.24626v1)
  > **TL;DR**: How to enable efficient serving of long-context LLMs using dynamic sparse attention? SparseServe introduces hierarchical HBM-DRAM KV cache management with fragmentation-aware transfers, working-set-aware batching, and layer-segmented prefill, achieving up to 9.26x lower TTFT and 3.14x higher throughput.
* `serving` `multi-modal` [RServe: Overlapping Encoding and Prefill for Efficient LMM Inference](http://arxiv.org/abs/2509.24381v1)
  > **TL;DR**: Addressing high latency in LMM inference, REDServe overlaps multimodal encoding with language model prefill via disaggregation and fine-grained scheduling, achieving up to 66% lower latency and 109% higher throughput.
* `RL` `training` [RL in the Wild: Characterizing RLVR Training in LLM Deployment](http://arxiv.org/abs/2509.25279v1)
  > **TL;DR**: Characterizes system challenges in RL with verifiable rewards (RLVR) for LLM training, identifying issues like GPU idling and load imbalance due to skewed workloads; proposes PolyTrace benchmark achieving 94.7% accuracy in workload simulation.
* `serving` `training` `edge` [MACE: A Hybrid LLM Serving System with Colocated SLO-aware Continuous Retraining Alignment](http://arxiv.org/abs/2510.03283v1)
  > **TL;DR**: How to jointly serve LLM inference and fine-tuning on edge devices without violating SLOs? MACE collocates inference and iteration-level retraining with dynamic GPU resource allocation, achieving up to 63% lower latency while maintaining >85% GPU utilization on NVIDIA AGX Orin.
* `training` `sparse` [AdaPtis: Reducing Pipeline Bubbles with Adaptive Pipeline Parallelism on Heterogeneous Models](http://arxiv.org/abs/2509.23722v1)
  > **TL;DR**: Addresses pipeline bubbles in LLM training caused by model heterogeneity by jointly optimizing partition, placement, and scheduling. AdaPtis uses a performance model to guide adaptive pipeline parallelism, achieving up to 2.14x speedup over Megatron-LM.
* `serving` [A Predictive and Synergistic Two-Layer Scheduling Framework for LLM Serving](http://arxiv.org/abs/2509.23384v3)
  > **TL;DR**: Addresses inefficient two-layer LLM serving by introducing predictive, synergistic scheduling to bridge cluster- and engine-layer information gaps. Uses a performancemodel for adaptive batching and state-driven routing, improving SLO attainment by 43% and throughput by 3x.
* `serving` `quantization` `edge` [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](http://arxiv.org/abs/2509.23324v1)
  > **TL;DR**: Can smaller LLMs match larger models' accuracy on smartphones by leveraging underused NPU compute via test-time scaling? Proposes hardware-aware quantization and LUT optimizations for NPU-efficient inference, achieving up to 2.2× speedup and matching larger model accuracy.
* `training` [A Flexible Programmable Pipeline Parallelism Framework for Efficient DNN Training](http://arxiv.org/abs/2510.05112v2)
  > **TL;DR**: Designs FlexPipe, a programmable framework for automated and customizable pipeline parallelism in DNN training, using a DSL and scheduler to explore efficient schedules; achieves up to 2.28× speedup over Megtron-LM.
* `MoE` `quantization` `offloading` [Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](http://arxiv.org/abs/2510.02345v1)
  > **TL;DR**: Addresses the MoE trilemma by dynamically clustering experts and applying structured compression with hierarchical routing, mixed-precision storage (FP16/INT4), and dynamic offloading, reducing parameters by 80% while improving throughput by 10–20% and load balance by 3×.
* `training` `hardware` [Efficient Fine-Grained GPU Performance Modeling for Distributed Deep Learning of LLM](http://arxiv.org/abs/2509.22832v1)
  > **TL;DR**: Predicts end-to-end LLM training time across distributed GPUs by decomposing models into primitives and using lightweight hardware-aware models. Achieves <10% prediction error on 20B models across 128 GPUs while running entirely on CPU.
* `training` `sparse` [Zeppelin: Balancing Variable-length Workloads in Data Parallel Large Model Training](http://arxiv.org/abs/2509.21841v2)
  > **TL;DR**: Addresses load imbalance in data-parallel LLM training due to variable sequence lengths. Introduces hierarchical sequence partitioning, dynamic NIC routing, and module-aware remapping to balance computation and communication. Achieves 2.80x speedup over state-of-the-art.
* `multi-modal` `serving` `edge` `kernel` [Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices](http://arxiv.org/abs/2510.05109v1)
  > **TL;DR**: Designs a software-hardware co-designed system to efficiently run multimodal models on battery-powered edge devices by modularly scheduling vision, audio, and language components across heterogeneous accelerators. Uses optimized low-bit kernels and token-aware buffering, reducing energy consumption by 42.3% and enabling all-day LMM inference on-device.
* `training` `sparse` [Data-Centric Elastic Pipeline Parallelism for Efficient Long-Context LLM Training](http://arxiv.org/abs/2509.21275v1)
  > **TL;DR**: Addresses inefficient pipeline parallelism in long-context LLM training by adaptively switching between token- and batch-level partitioning. InfiniPipe uses workload-aware sequence packing and stage-aware checkpointing to balance load and reduce memory. Achieves 1.69x speedup over SOTA.
* `training` `offloading` [SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips](http://arxiv.org/abs/2509.21271v1)
  > **TL;DR**: How to optimize large-scale LLM training on Superchips using offloading? SuperOffload introduces Superchip-aware techniques like adaptive weight offloading and CPU-optimized Adam, achieving 2.5x higher throughput and enabling 25B model training on a single GH200.
* `training` `networking` [Go With The Flow: Churn-Tolerant Decentralized Training of Large Language Models](http://arxiv.org/abs/2509.21221v1)
  > **TL;DR**: Proposes GWTF, a decentralized framework for LLM training tolerant to node churn and network instability. Uses a novel flow-based routing algorithm to optimize microbatch scheduling across heterogeneous clients. Reduces training time by up to 45% under high churn.
* `RL` `serving` [RollPacker: Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training](http://arxiv.org/abs/2509.21009v1)
  > **TL;DR**: Addresses GPU underutilization in synchronous RL post-training caused by long-tail response delays. Introduces tail batching to schedule long responses separately, enabling balanced workloads across training stages. Achieves up to 2.56x faster training time on 128 H800 GPUs.
* `serving` [PARS: Low-Latency LLM Serving via Pairwise Learning-to-Rank](http://arxiv.org/abs/2510.03243v1)
  > **TL;DR**: How to reduce LLM serving latency caused by Head-of-Line blocking? PARS uses pairwise learning-to-rank to predict optimal task ordering by response length, integrated into vLLM. Achieves up to 35% lower average latency without sacrificing throughput.
* `training` `serving` [Kant: An Efficient Unified Scheduling System for Large-Scale AI Clusters](http://arxiv.org/abs/2510.01256v1)
  > **TL;DR**: Designs Kant, a unified scheduler for co-scheduling LLM training and inference in large AI clusters. Uses Backfill and E-Binpack to improve GPU utilization and reduce fragmentation. Achieves up to 30% higher GPU Allocation Ratio (GAR) compared to baseline schedulers.
* `serving` `hardware` `offline` [Energy Use of AI Inference: Efficiency Pathways and Test-Time Compute](http://arxiv.org/abs/2509.20241v1)
  > **TL;DR**: Estimates energy use per LLM inference query at scale, accounting for real-world GPU utilization and PUE. Proposes a bottom-up methodology to quantify efficiency gains at model, platform, and hardware levels, achieving up to 20x reduction in energy per query for 1B queries/day.
* `training` `offloading` [BurstEngine: an Efficient Distributed Framework for Training Transformers on Extremely Long Sequences of over 1M Tokens](http://arxiv.org/abs/2509.19836v1)
  > **TL;DR**: Designs BurstEngine to efficiently train LLMs on sequences >1M tokens by introducing topology-aware BurstAttention, selective checkpointing, and fused loss computation, achieving 1.2× speedup and reduced memory overhead compared to state-of-the-art methods.
* `serving` `offloading` [Gyges: Dynamic Cross-Instance Parallelism Transformation for Efficient LLM Inference](http://arxiv.org/abs/2509.19729v1)
  > **TL;DR**: Addresses dynamic context length variance in LLM serving by adaptively transforming parallelism strategies across instances. Proposes a header-centric KV cache layout, weight padding, and transformation-aware scheduler, achieving up to 6.57× higher throughput than state-of-the-art systems.
* `RAG` [On The Reproducibility Limitations of RAG Systems](http://arxiv.org/abs/2509.18869v1)
  > **TL;DR**: Addresses the reproducibility limitations of RAG systems by introducing ReproRAG, a benchmarking framework that quantifies non-determinism across embedding models, retrieval algorithms, and hardware. Evaluates trade-offs using metrics like Exact Match Rate and Jaccard Similarity.
* `MoE` `serving` [Expert-as-a-Service: Towards Efficient, Scalable, and Robust Large-scale MoE Serving](http://arxiv.org/abs/2509.17863v1)
  > **TL;DR**: Addresses efficient serving of large-scale MoE models by disaggregating experts into stateless services. Uses peer-to-peer communication for low-overhead routing and dynamic resource scaling. Achieves 37.5% resource savings with <2% throughput loss under failures.
* `serving` `hardware` [Disaggregated Prefill and Decoding Inference System for Large Language Model Serving on Multi-Vendor GPUs](http://arxiv.org/abs/2509.17542v2)
  > **TL;DR**: Designs a disaggregated LLM inference system using heterogeneous GPUs to separate prefill and decoding stages, enabling cost-efficient deployment. Introduces a heterogeneous-compatible transmission module and joint optimization for parallelism and instance allocation. Achieves 38% higher resource utilization compared to homogeneous setups.
* `agentic` `serving` `RAG` [Asteria: Semantic-Aware Cross-Region Caching for Agentic LLM Tool Access](http://arxiv.org/abs/2509.17360v1)
  > **TL;DR**: Asteria improves agentic LLM performance by introducing semantic-aware cross-region caching for tool access. It uses semantic embeddings and a lightweight LLM judger for precise retrieval, achieving 3.6× higher throughput with 85%+ cache hit rates while maintaining accuracy.
* `serving` `heterogeneous` [Cronus: Efficient LLM inference on Heterogeneous GPU Clusters via Partially Disaggregated Prefill](http://arxiv.org/abs/2509.17357v1)
  > **TL;DR**: Cronus improves LLM inference throughput on heterogeneous GPU clusters by partially disaggregating prefill across low- and high-end GPUs, overlapping stages to balance load. It reduces P99 TTFT and TBT by up to 40% while maintaining high throughput.
* `multi-modal` `offloading` `edge` [MoA-Off: Adaptive Heterogeneous Modality-Aware Offloading with Edge-Cloud Collaboration for Efficient Multimodal LLM Inference](http://arxiv.org/abs/2509.16995v1)
  > **TL;DR**: How to efficiently infer multimodal LLMs on edge devices? MoA-Off proposes a modality-aware offloading framework that dynamically splits computation between edge and cloud based on input complexity, reducing latency by 30% and resource overhead by 30%-65% while preserving accuracy.
* `serving` `offloading` `hardware` [ShadowServe: Interference-Free KV Cache Fetching for Distributed Prefix Caching](http://arxiv.org/abs/2509.16857v1)
  > **TL;DR**: Addresses KV cache fetch interference in distributed prefix caching for LLM serving. Proposes ShadowServe, a SmartNIC-offloaded system with chunked pipelining and minimal-copy memory management. Achieves up to 2.2x lower TPOT and 1.38x lower TTFT in low-bandwidth scenarios.
* `serving` [Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads](http://arxiv.org/abs/2509.16495v1)
  > **TL;DR**: Addresses the latency-throughput tradeoff in LLM serving by introducing Shift Parallelism, which dynamically switches between tensor and sequence parallelism to leverage KV cache invariance. Achieves 1.51× lower latency in interactive workloads and 50% higher throughput in batch workloads than tensor parallelism alone.
* `training` [Robust LLM Training Infrastructure at ByteDance](http://arxiv.org/abs/2509.16293v2)
  > **TL;DR**: How to ensure stable large-scale LLM training amid frequent failures? ByteRobust introduces data-driven fault detection and recovery tailored to LLM parallelism, achieving 97% effective training time ratio (ETTR) over 9,600 GPUs.
* `RL` `training` [RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation](http://arxiv.org/abs/2509.15965v1)
  > **TL;DR**: Addresses low hardware utilization in RL training by introducing M2Flow, a macro-to-micro flow transformation that optimizes workflow execution via context switching and elastic pipelining. Achieves 1.1x–2.13x speedup in end-to-end training throughput.
* `networking` `training` [Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs](http://arxiv.org/abs/2509.15940v1)
  > **TL;DR**: Addresses communication inefficiencies in large-scale LLM training by aligning communication patterns with data center topology. Proposes Arnold, a scheduling system that reduces communication spread and improves end-to-end training throughput by 10.6% on 9600+ GPUs.
* `hardware` `networking` [PCCL: Photonic circuit-switched collective communication for distributed ML](http://arxiv.org/abs/2509.15450v1)
  > **TL;DR**: Addresses communication bottlenecks in distributed ML training by reconfiguring photonic networks to eliminate congestion. Proposes PCCL, a hardware-agnostic system that creates direct circuits for collective operations, achieving up to 3X faster communication and 1.3X end-to-end training speedup.
* `RAG` `agentic` [LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology](http://arxiv.org/abs/2509.13978v2)
  > **TL;DR**: Explores using LLM agents to interpret and query scientific workflow provenance via natural language. Combines metadata-driven design and RAG to translate prompts into structured queries, achieving high accuracy on real-world chemistry workflows.
* `serving` `kernel` `hardware` [FLAME: A Serving System Optimized for Large-Scale Generative Recommendation with Efficiency](http://arxiv.org/abs/2509.22681v1)
  > **TL;DR**: Designs a production-grade serving system for large-scale generative recommendation models by decoupling pre-processing and computation, optimizing memory with PDA, and accelerating inference via TensorRT-based fused kernels. Achieves up to 6.3x throughput gain and 2.3x latency reduction.
* `diffusion` `hardware` [AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions](http://arxiv.org/abs/2509.13523v1)
  > **TL;DR**: Designs AERIS, a billion-parameter Swin diffusion transformer for high-resolution weather prediction, using SWiPe to enable efficient window, sequence, and pipeline parallelism; achieves 10.21 ExaFLOPS on Aurora with 95.5% weak scaling efficiency.
* `serving` `offloading` [Scaling Up Throughput-oriented LLM Inference Applications on Heterogeneous Opportunistic GPU Clusters with Pervasive Context Management](http://arxiv.org/abs/2509.13201v1)
  > **TL;DR**: Addresses how to improve throughput of non-latency-sensitive LLM inference on opportunistic GPU clusters. Introduces pervasive context management to reuse computational context across dynamic resources, enabling 98.1% reduction in execution time.
* `agentic` `serving` `offloading` [Toward Systems Foundations for Agentic Exploration](http://arxiv.org/abs/2510.05556v1)
  > **TL;DR**: Addresses the lack of efficient system support for LLM agentic exploration. Proposes fundamental challenges in fork semantics, side-effect handling, and microsecond-level native forking. Achieves scalable branching with minimal overhead in multi-agent deployments.
* `agentic` `serving` `offloading` [Nova: Real-Time Agentic Vision-Language Model Serving with Adaptive Cross-Stage Parallelization](http://arxiv.org/abs/2509.21301v1)
  > **TL;DR**: Nova enables efficient real-time serving of agentic vision-language models by adaptively partitioning GPU resources across vision, prefill, and decode stages, plus lightweight vision encoder offloading. It achieves up to 23.3% lower max latency while maintaining high throughput.

### 2025-09-15

* [Characterizing the Efficiency of Distributed Training: A Power, Performance, and Thermal Perspective](https://arxiv.org/abs/2509.10371)
* [Ordered Consensus with Equal Opportunity](https://arxiv.org/abs/2509.09868)
* [The (R)evolution of Scientific Workflows in the Agentic AI Era: Towards Autonomous Science](https://arxiv.org/abs/2509.09915)
* [DBOS Network Sensing: A Web Services Approach to Collaborative Awareness](https://arxiv.org/abs/2509.09898)
* [XBOF: A Cost-Efficient CXL JBOF with Inter-SSD Compute Resource Sharing](https://arxiv.org/abs/2509.10251)
* [DBOS Network Sensing: A Web Services Approach to Collaborative Awareness](https://arxiv.org/abs/2509.09898)


### 2025-09-12

* [TrEnv: Transparently Share Serverless Execution Environments Across Different Functions and Nodes](https://arxiv.org/abs/2509.09525)
* [Weaker Assumptions for Asymmetric Trust](https://arxiv.org/abs/2509.09493)
* [Barycentric Coded Distributed Computing with Flexible Recovery Threshold for Collaborative Mobile Edge Computing](https://arxiv.org/abs/2509.09435)
* [WebAssembly and Unikernels: A Comparative Study for Serverless at the Edge](https://arxiv.org/abs/2509.09400)
* [Coherence-Aware Task Graph Modeling for Realistic Application](https://arxiv.org/abs/2509.09094)
* [Optimizing the Variant Calling Pipeline Execution on Human Genomes Using GPU-Enabled Machines](https://arxiv.org/abs/2509.09058)
* [A Comparative Analysis of Identifier Schemes: UUIDv4, UUIDv7, and ULID for Distributed Systems](https://arxiv.org/abs/2509.08969)
* [Towards A High-Performance Quantum Data Center Network Architecture](https://arxiv.org/abs/2509.09653)
* [HARD: A Performance Portable Radiation Hydrodynamics Code based on FleCSI Framework](https://arxiv.org/abs/2509.08971)
* [μFork: Supporting POSIX fork Within a Single-Address-Space OS](https://arxiv.org/abs/2509.09439)
* [TrEnv: Transparently Share Serverless Execution Environments Across Different Functions and Nodes](https://arxiv.org/abs/2509.09525)


### 2025-09-11

* [Reconfigurable Holographic Surfaces and Near Field Communication for Non-Terrestrial Networks: Potential and Challenges](https://arxiv.org/abs/2509.08770)
* [A 410GFLOP/s, 64 RISC-V Cores, 204.8GBps Shared-Memory Cluster in 12nm FinFET with Systolic Execution Support for Efficient B5G/6G AI-Enhanced O-RAN](https://arxiv.org/abs/2509.08608)
* [An HPC Benchmark Survey and Taxonomy for Characterization](https://arxiv.org/abs/2509.08347)
* [Hetis: Serving LLMs in Heterogeneous GPU Clusters with Fine-grained and Dynamic Parallelism](https://arxiv.org/abs/2509.08309)
* [Design and Implementation of Code Completion System Based on LLM and CodeBERT Hybrid Subsystem](https://arxiv.org/abs/2509.08215)
* [Aurora: Architecting Argonne's First Exascale Supercomputer for Accelerated Scientific Discovery](https://arxiv.org/abs/2509.08207)
* [Towards Scalable Proteomics: Opportunistic SMC Samplers on HTCondor](https://arxiv.org/abs/2509.08020)


### 2025-09-10

* [Scaling atomic ordering in shared memory](https://arxiv.org/abs/2509.07781)
* [AgentX: Towards Orchestrating Robust Agentic Workflow Patterns with FaaS-hosted MCP Services](https://arxiv.org/abs/2509.07595)
* [Navigating Energy Doldrums: Modeling the Impact of Energy Price Volatility on HPC Cost of Ownership](https://arxiv.org/abs/2509.07567)
* [Astra: A Multi-Agent System for GPU Kernel Performance Optimization](https://arxiv.org/abs/2509.07506)
* [DREAMS: Decentralized Resource Allocation and Service Management across the Compute Continuum Using Service Affinity](https://arxiv.org/abs/2509.07497)
* [Dependency-Aware Execution Mechanism in Hyperledger Fabric Architecture](https://arxiv.org/abs/2509.07425)
* [DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling for Efficient MoE LLM Inference](https://arxiv.org/abs/2509.07379)
* [Optimizing Task Scheduling in Fog Computing with Deadline Awareness](https://arxiv.org/abs/2509.07378)
* [A Study on Messaging Trade-offs in Data Streaming for Scientific Workflows](https://arxiv.org/abs/2509.07199)
* [Bodega: Serving Linearizable Reads Locally from Anywhere at Anytime via Roster Leases](https://arxiv.org/abs/2509.07158)
* [Crossword: Adaptive Consensus for Dynamic Data-Heavy Workloads](https://arxiv.org/abs/2509.07157)
* [MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Model?](https://arxiv.org/abs/2509.07727)
* [HYLU: Hybrid Parallel Sparse LU Factorization](https://arxiv.org/abs/2509.07690)
* [veScale: Consistent and Efficient Tensor Programming with Eager-Mode SPMD](https://arxiv.org/abs/2509.07003)


### 2025-09-09

* [IM-PIR: In-Memory Private Information Retrieval](https://arxiv.org/abs/2509.06514)
* [MaaSO: SLO-aware Orchestration of Heterogeneous Model Instances for MaaS](https://arxiv.org/abs/2509.06362)
* [FineServe: Precision-Aware KV Slab and Two-Level Scheduling for Heterogeneous Precision LLM Serving](https://arxiv.org/abs/2509.06261)
* [20 Years in Life of a Smart Building: A retrospective](https://arxiv.org/abs/2509.06229)
* [Gathering in Non-Vertex-Transitive Graphs Under Round Robin](https://arxiv.org/abs/2509.06064)
* [DISTRIBUTEDANN: Efficient Scaling of a Single DISKANN Graph Across Thousands of Computers](https://arxiv.org/abs/2509.06046)
* [A Simple and Robust Protocol for Distributed Counting](https://arxiv.org/abs/2509.05870)
* [Multi-IaC-Eval: Benchmarking Cloud Infrastructure as Code Across Multiple Formats](https://arxiv.org/abs/2509.05303)
* [Distributed Automatic Generation Control subject to Ramp-Rate-Limits: Anytime Feasibility and Uniform Network-Connectivity](https://arxiv.org/abs/2509.06588)
* [Tackling Device Data Distribution Real-time Shift via Prototype-based Parameter Editing](https://arxiv.org/abs/2509.06552)
* [Several Performance Bounds on Decentralized Online Optimization are Highly Conservative and Potentially Misleading](https://arxiv.org/abs/2509.06466)
* [Social Dynamics of DAOs: Power, Onboarding, and Inclusivity](https://arxiv.org/abs/2509.06163)
* [Introduction to Number Theoretic Transform](https://arxiv.org/abs/2509.05884)
* [Tiga: Accelerating Geo-Distributed Transactions with Synchronized Clocks [Technical Report]](https://arxiv.org/abs/2509.05759)
* [Distributed Deep Learning using Stochastic Gradient Staleness](https://arxiv.org/abs/2509.05679)
* [Workflow for High-Fidelity Dynamic Analysis of Structures with Pile Foundation](https://arxiv.org/abs/2509.05675)
* [Efficient Fault Localization in a Cloud Stack Using End-to-End Application Service Topology](https://arxiv.org/abs/2509.05511)
* [MambaLite-Micro: Memory-Optimized Mamba Inference on MCUs](https://arxiv.org/abs/2509.05488)


### 2025-09-08

* [Scaling Performance of Large Language Model Pretraining](https://arxiv.org/abs/2509.05258)
* [Dynamic reconfiguration for malleable applications using RMA](https://arxiv.org/abs/2509.05248)
* [Toward Distributed 3D Gaussian Splatting for High-Resolution Isosurface Visualization](https://arxiv.org/abs/2509.05216)
* [VoltanaLLM: Feedback-Driven Frequency Control and State-Space Routing for Energy-Efficient LLM Serving](https://arxiv.org/abs/2509.04827)
* [STADI: Fine-Grained Step-Patch Diffusion Parallelism for Heterogeneous GPUs](https://arxiv.org/abs/2509.04719)


### 2025-09-05

* [On the impact of unlimited computational power in OBLOT: consequences for synchronous robots on graphs](https://arxiv.org/abs/2509.04383)
* [Trustworthy Second-hand Marketplace for Built Environment](https://arxiv.org/abs/2509.04085)
* [LowDiff: Efficient Frequent Checkpointing via Low-Cost Differential for High-Performance Distributed Training Systems](https://arxiv.org/abs/2509.04084)
* [Counterfactual simulations for large scale systems with burnout variables](https://arxiv.org/abs/2509.04038)
* [Gathering of asynchronous robots on circle with limited visibility using finite communication](https://arxiv.org/abs/2509.04004)
* [Distributed Download from an External Data Source in Asynchronous Faulty Settings](https://arxiv.org/abs/2509.03755)
* [Combining Performance and Productivity: Accelerating the Network Sensing Graph Challenge with GPUs and Commodity Data Science Software](https://arxiv.org/abs/2509.03653)
* [Massively-Parallel Implementation of Inextensible Elastic Rods Using Inter-block GPU Synchronization](https://arxiv.org/abs/2509.04277)
* [Cloud-Assisted Remote Control for Aerial Robots: From Theory to Proof-of-Concept Implementation](https://arxiv.org/abs/2509.04095)
* [Prob-GParareal: A Probabilistic Numerical Parallel-in-Time Solver for Differential Equations](https://arxiv.org/abs/2509.03945)
* [Towards Deterministic Sub-0.5 us Response on Linux through Interrupt Isolation](https://arxiv.org/abs/2509.03855)


### 2025-09-04

* [CloudFormer: An Attention-based Performance Prediction for Public Clouds with Unknown Workload](https://arxiv.org/abs/2509.03394)
* [Efficient and Secure Sleepy Model for BFT Consensus](https://arxiv.org/abs/2509.03145)
* [The High Cost of Keeping Warm: Characterizing Overhead in Serverless Autoscaling Policies](https://arxiv.org/abs/2509.03104)
* [FlashRecovery: Fast and Low-Cost Recovery from Failures for Large-Scale Training of LLMs](https://arxiv.org/abs/2509.03047)
* [Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training](https://arxiv.org/abs/2509.03018)
* [A Novel IaaS Tax Model as Leverage Towards Green Cloud Computing](https://arxiv.org/abs/2509.02767)
* [DPQuant: Efficient and Differentially-Private Model Training via Dynamic Quantization Scheduling](https://arxiv.org/abs/2509.03472)
* [A description of the radio astronomy data processing tool DDF Pipeline](https://arxiv.org/abs/2509.03075)
* [Treasure Hunt in Anonymous Graphs with Quantum Pebbles by Oblivious Agents](https://arxiv.org/abs/2509.02909)
* [\textit{In Silico} Benchmarking of Detectable Byzantine Agreement in Noisy Quantum Networks](https://arxiv.org/abs/2509.02629)
* [On the Optimization of Methods for Establishing Well-Connected Communities](https://arxiv.org/abs/2509.02590)
* [Safe Sharing of Fast Kernel-Bypass I/O Among Nontrusting Applications](https://arxiv.org/abs/2509.02899)


### 2025-09-03

* [Energy-Efficient Split Learning for Resource-Constrained Environments: A Smart Farming Solution](https://arxiv.org/abs/2509.02549)
* [MLP-Offload: Multi-Level, Multi-Path Offloading for LLM Pre-training to Break the GPU Memory Wall](https://arxiv.org/abs/2509.02480)
* [Safe Memory Reclamation Techniques](https://arxiv.org/abs/2509.02457)
* [KubeIntellect: A Modular LLM-Orchestrated Agent Framework for End-to-End Kubernetes Management](https://arxiv.org/abs/2509.02449)
* [An Efficient and Adaptive Watermark Detection System with Tile-based Error Correction](https://arxiv.org/abs/2509.02447)
* [Efficient Pyramidal Analysis of Gigapixel Images on a Decentralized Modest Computer Cluster](https://arxiv.org/abs/2509.02440)
* [A Continuous Energy Ising Machine Leveraging Difference-of-Convex Programming](https://arxiv.org/abs/2509.01928)
* [Optimal Parallel Scheduling under Concave Speedup Functions](https://arxiv.org/abs/2509.01811)
* [STZ: A High Quality and High Speed Streaming Lossy Compression Framework for Scientific Data](https://arxiv.org/abs/2509.01626)
* [HiCR, an Abstract Model for Distributed Heterogeneous Programming](https://arxiv.org/abs/2509.01425)
* [LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving](https://arxiv.org/abs/2509.01229)
* [LobRA: Multi-tenant Fine-tuning over Heterogeneous Data](https://arxiv.org/abs/2509.01193)
* [DSDE: Dynamic Speculative Decoding with KLD Stability for Real-World Serving](https://arxiv.org/abs/2509.01083)
* [Parallelizing Drug Discovery: HPC Pipelines for Alzheimer's Molecular Docking and Simulation](https://arxiv.org/abs/2509.00937)
* [Accelerating Latency-Critical Applications with AI-Powered Semi-Automatic Fine-Grained Parallelization on SMT Processors](https://arxiv.org/abs/2509.00883)
* [HADIS: Hybrid Adaptive Diffusion Model Serving for Efficient Text-to-Image Generation](https://arxiv.org/abs/2509.00642)
* [KVComp: A High-Performance, LLM-Aware, Lossy Compression Framework for KV Cache](https://arxiv.org/abs/2509.00579)
* [HydroGAT: Distributed Heterogeneous Graph Attention Transformer for Spatiotemporal Flood Prediction](https://arxiv.org/abs/2509.02481)
* [Online Identification of IT Systems through Active Causal Learning](https://arxiv.org/abs/2509.02130)
* [Batch Query Processing and Optimization for Agentic Workflows](https://arxiv.org/abs/2509.02121)
* [OASIS: Object-based Analytics Storage for Intelligent SQL Query Offloading in Scientific Tabular Workloads](https://arxiv.org/abs/2509.01966)
* [AdaptCache: KV Cache Native Storage Hierarchy for Low-Delay and High-Quality Language Model Serving](https://arxiv.org/abs/2509.00105)
* [Towards Agentic OS: An LLM Agent Framework for Linux Schedulers](https://arxiv.org/abs/2509.01245)


### 2025-09-01

* [Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding](https://arxiv.org/abs/2508.21706)
* [Odyssey: Adaptive Policy Selection for Resilient Distributed Training](https://arxiv.org/abs/2508.21613)
* [Unpacking Maximum Extractable Value on Polygon: A Study on Atomic Arbitrage](https://arxiv.org/abs/2508.21473)
* [Addressing Reproducibility Challenges in HPC with Continuous Integration](https://arxiv.org/abs/2508.21289)
* [Fast and Scalable Mixed Precision Euclidean Distance Calculations Using GPU Tensor Cores](https://arxiv.org/abs/2508.21230)
* [An Optimistic Gradient Tracking Method for Distributed Minimax Optimization](https://arxiv.org/abs/2508.21431)


### 2025-08-29

* [Collaborative Evolution of Intelligent Agents in Large-Scale Microservice Systems](https://arxiv.org/abs/2508.20508)
* [pdGRASS: A Fast Parallel Density-Aware Algorithm for Graph Spectral Sparsification](https://arxiv.org/abs/2508.20403)
* [CoFormer: Collaborating with Heterogeneous Edge Devices for Scalable Transformer Inference](https://arxiv.org/abs/2508.20375)
* [Predictable LLM Serving on GPU Clusters](https://arxiv.org/abs/2508.20274)
* [SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization](https://arxiv.org/abs/2508.20258)
* [SpeedMalloc: Improving Multi-threaded Applications via a Lightweight Core for Memory Allocation](https://arxiv.org/abs/2508.20253)
* [A Hybrid Stochastic Gradient Tracking Method for Distributed Online Optimization Over Time-Varying Directed Networks](https://arxiv.org/abs/2508.20645)
* [High performance visualization for Astronomy and Cosmology: the VisIVO's pathway toward Exascale systems](https://arxiv.org/abs/2508.20603)
* [Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs](https://arxiv.org/abs/2508.20333)


### 2025-08-28

* [HPC Digital Twins for Evaluating Scheduling Policies, Incentive Structures and their Impact on Power and Cooling](https://arxiv.org/abs/2508.20016)
* [Separation of Three or More Autonomous Mobile Models under Hierarchical Schedulers](https://arxiv.org/abs/2508.19805)
* [Beyond the Bermuda Triangle of Contention: IOMMU Interference in Mixed Criticality Systems](https://arxiv.org/abs/2508.19670)
* [Taming the Chaos: Coordinated Autoscaling for Heterogeneous and Disaggregated LLM Inference](https://arxiv.org/abs/2508.19559)
* [Formal Modeling and Verification of the Algorand Consensus Protocol in CADP](https://arxiv.org/abs/2508.19452)
* [HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference](https://arxiv.org/abs/2508.19373)
* [New Tools, Programming Models, and System Support for Processing-in-Memory Architectures](https://arxiv.org/abs/2508.19868)
* [Aegis: Taxonomy and Optimizations for Overcoming Agent-Environment Failures in LLM Agents](https://arxiv.org/abs/2508.19504)


### 2025-08-27

* [Ab-initio Quantum Transport with the GW Approximation, 42,240 Atoms, and Sustained Exascale Performance](https://arxiv.org/abs/2508.19138)
* [CARMA: Collocation-Aware Resource Manager with GPU Memory Estimator](https://arxiv.org/abs/2508.19073)
* [Deep Learning-Enabled Supercritical Flame Simulation at Detailed Chemistry and Real-Fluid Accuracy Towards Trillion-Cell Scale](https://arxiv.org/abs/2508.18969)
* [SIREN: Software Identification and Recognition in HPC Systems](https://arxiv.org/abs/2508.18950)
* [ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive](https://arxiv.org/abs/2508.18850)
* [Examining MPI and its Extensions for Asynchronous Multithreaded Communication](https://arxiv.org/abs/2508.18667)
* [Strata: Hierarchical Context Caching for Long Context Language Model Serving](https://arxiv.org/abs/2508.18572)
* [Managing Multi Instance GPUs for High Throughput and Energy Savings](https://arxiv.org/abs/2508.18556)
* [Experiences with Model Context Protocol Servers for Science and High Performance Computing](https://arxiv.org/abs/2508.18489)
* [Architecting Distributed Quantum Computers: Design Insights from Resource Estimation](https://arxiv.org/abs/2508.19160)
* [History Rhymes: Accelerating LLM Reinforcement Learning with RhymeRL](https://arxiv.org/abs/2508.18588)
* [DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction](https://arxiv.org/abs/2508.18376)


### 2025-08-26

* [Flash Sparse Attention: An Alternative Efficient Implementation of Native Sparse Attention Kernel](https://arxiv.org/abs/2508.18224)
* [Practical GPU Choices for Earth Observation: ResNet-50 Training Throughput on Integrated, Laptop, and Cloud Accelerators](https://arxiv.org/abs/2508.18206)
* [Wait-free Replicated Data Types and Fair Reconciliation](https://arxiv.org/abs/2508.18193)
* [Scalable Engine and the Performance of Different LLM Models in a SLURM based HPC architecture](https://arxiv.org/abs/2508.17814)
* [ExpertWeave: Efficiently Serving Expert-Specialized Fine-Tuned Adapters at Scale](https://arxiv.org/abs/2508.17624)
* [Zen-Attention: A Compiler Framework for Dynamic Attention Folding on AMD NPUs](https://arxiv.org/abs/2508.17593)
* [Easy Acceleration with Distributed Arrays](https://arxiv.org/abs/2508.17493)
* [Bine Trees: Enhancing Collective Operations by Optimizing Communication Locality](https://arxiv.org/abs/2508.17311)
* [TokenLake: A Unified Segment-level Prefix Cache Pool for Fine-grained Elastic Long-Context LLM Serving](https://arxiv.org/abs/2508.17219)
* [PICO: Performance Insights for Collective Operations](https://arxiv.org/abs/2508.16809)
* [Neuromorphic Simulation of Drosophila Melanogaster Brain Connectome on Loihi 2](https://arxiv.org/abs/2508.16792)
* [Equinox: Holistic Fair Scheduling in Serving Large Language Models](https://arxiv.org/abs/2508.16646)
* [GPU Acceleration for Faster Evolutionary Spatial Cyclic Game Systems](https://arxiv.org/abs/2508.16639)
* [Performance measurements of modern Fortran MPI applications with Score-P](https://arxiv.org/abs/2508.16592)
* [Views: A Hardware-friendly Graph Database Model For Storing Semantic Information](https://arxiv.org/abs/2508.18123)
* [Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective](https://arxiv.org/abs/2508.16712)
* [GPT-OSS-20B: A Comprehensive Deployment-Centric Analysis of OpenAI's Open-Weight Mixture of Experts Model](https://arxiv.org/abs/2508.16700)
* [Scalable Hybrid quantum Monte Carlo simulation of U(1) gauge field coupled to fermions on GPU](https://arxiv.org/abs/2508.16298)
* [Iridescent: A Framework Enabling Online System Implementation Specialization](https://arxiv.org/abs/2508.16690)
* [Puzzle: Scheduling Multiple Deep Learning Models on Mobile Device with Heterogeneous Processors](https://arxiv.org/abs/2508.17764)


### 2025-08-25

* [Generalizing Brooks' theorem via Partial Coloring is Hard Classically and Locally](https://arxiv.org/abs/2508.16308)
* [HyperFlexis: Joint Design of Algorithms and Systems for Multi-SLO Serving and Fast Scaling](https://arxiv.org/abs/2508.15919)
* [On the Duality of Task and Actor Programming Models](https://arxiv.org/abs/2508.16522)
* [Hybrid Classical-Quantum Supercomputing: A demonstration of a multi-user, multi-QPU and multi-GPU environment](https://arxiv.org/abs/2508.16297)
* [Self-Healing Network of Interconnected Edge Devices Empowered by Infrastructure-as-Code and LoRa Communication](https://arxiv.org/abs/2508.16268)
* [Towards Integrated Energy-Communication-Transportation Hub: A Base-Station-Centric Design in 5G and Beyond](https://arxiv.org/abs/2508.15833)
* [CXLAimPod: CXL Memory is all you need in AI era](https://arxiv.org/abs/2508.15980)


### 2025-08-22

* [CausalMesh: A Formally Verified Causal Cache for Stateful Serverless Computing](https://arxiv.org/abs/2508.15647)
* [Efficient Mixed-Precision Large Language Model Inference with TurboMind](https://arxiv.org/abs/2508.15601)
* [Lower Bounds for $k$-Set Agreement in Fault-Prone Networks](https://arxiv.org/abs/2508.15562)
* [Universal Dancing by Luminous Robots under Sequential Schedulers](https://arxiv.org/abs/2508.15484)
* [Databelt: A Continuous Data Path for Serverless Workflows in the 3D Compute Continuum](https://arxiv.org/abs/2508.15351)
* [Declarative Data Pipeline for Large Scale ML Services](https://arxiv.org/abs/2508.15105)
* [Mitigating context switching in densely packed Linux clusters with Latency-Aware Group Scheduling](https://arxiv.org/abs/2508.15703)
* [On the Effectiveness of Graph Reordering for Accelerating Approximate Nearest Neighbor Search on GPU](https://arxiv.org/abs/2508.15436)
* [Optimizing Compilation for Distributed Quantum Computing via Clustering and Annealing](https://arxiv.org/abs/2508.15267)
* [Reliable Multi-view 3D Reconstruction for `Just-in-time' Edge Environments](https://arxiv.org/abs/2508.15158)
* [TOAST: Fast and scalable auto-partitioning based on principled static analysis](https://arxiv.org/abs/2508.15010)
* [Scalable FPGA Framework for Real-Time Denoising in High-Throughput Imaging: A DRAM-Optimized Pipeline using High-Level Synthesis](https://arxiv.org/abs/2508.14917)
* [Mitigating context switching in densely packed Linux clusters with Latency-Aware Group Scheduling](https://arxiv.org/abs/2508.15703)


### 2025-08-21

* [The Cost Advantage of Virtual Machine Migrations: Empirical Insights into Amazon's EC2 Marketspace](https://arxiv.org/abs/2508.14883)
* [Leveraging Hardware-Aware Computation in Mixed-Precision Matrix Multiply: A Tile-Centric Approach](https://arxiv.org/abs/2508.14848)
* [MOHAF: A Multi-Objective Hierarchical Auction Framework for Scalable and Fair Resource Allocation in IoT Ecosystems](https://arxiv.org/abs/2508.14830)
* [DAG it off: Latency Prefers No Common Coins](https://arxiv.org/abs/2508.14716)
* [A Systematic Evaluation of the Potential of Carbon-Aware Execution for Scientific Workflows](https://arxiv.org/abs/2508.14625)
* [Boosting Payment Channel Network Liquidity with Topology Optimization and Transaction Selection](https://arxiv.org/abs/2508.14524)
* [Auditable Shared Objects: From Registers to Synchronization Primitives](https://arxiv.org/abs/2508.14506)
* [SSSP-Del: Fully Dynamic Distributed Algorithm for Single-Source Shortest Path](https://arxiv.org/abs/2508.14319)
* [Pure Data Spaces](https://arxiv.org/abs/2508.14271)
* [Time-optimal Asynchronous Minimal Vertex Covering by Myopic Robots](https://arxiv.org/abs/2508.14247)
* [Cooperative SGD with Dynamic Mixing Matrices](https://arxiv.org/abs/2508.14565)
* [Lagrangian Simulation Volume-Based Contour Tree Simplification](https://arxiv.org/abs/2508.14339)
* [Power Stabilization for AI Training Datacenters](https://arxiv.org/abs/2508.14318)
* [A High Performance GPU CountSketch Implementation and Its Application to Multisketching and Least Squares Problems](https://arxiv.org/abs/2508.14209)


### 2025-08-20

* [Is RISC-V ready for High Performance Computing? An evaluation of the Sophon SG2044](https://arxiv.org/abs/2508.13840)
* [Estimating CO$_2$ emissions of distributed applications and platforms with SimGrid/Batsim](https://arxiv.org/abs/2508.13693)
* [LUNDIsim: model meshes for flow simulation and scientific data compression benchmarks](https://arxiv.org/abs/2508.13636)
* [LAMMPS-KOKKOS: Performance Portable Molecular Dynamics Across Exascale Architectures](https://arxiv.org/abs/2508.13523)
* [DDoS Attacks in Cloud Computing: Detection and Prevention](https://arxiv.org/abs/2508.13522)
* [Optimizing Allreduce Operations for Heterogeneous Architectures with Multiple Processes per GPU](https://arxiv.org/abs/2508.13397)
* [OrbitChain: Orchestrating In-orbit Real-time Analytics of Earth Observation Data](https://arxiv.org/abs/2508.13374)
* [Persistent and Partitioned MPI for Stencil Communication](https://arxiv.org/abs/2508.13370)
* [Harnessing the Full Potential of RRAMs through Scalable and Distributed In-Memory Computing with Integrated Error Correction](https://arxiv.org/abs/2508.13298)
* [Analog computation with transcriptional networks](https://arxiv.org/abs/2508.14017)
* [PennyLane-Lightning MPI: A massively scalable quantum circuit simulator based on distributed computing in CPU clusters](https://arxiv.org/abs/2508.13615)
* [X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms](https://arxiv.org/abs/2508.13337)
* [Sustainable AI Training via Hardware-Software Co-Design on NVIDIA, AMD, and Emerging GPU Architectures](https://arxiv.org/abs/2508.13163)
* [Towards Timing Isolation for Mixed-Criticality Communication in Software-Defined Vehicles](https://arxiv.org/abs/2508.13652)


### 2025-08-19

* [Team Formation and Applications](https://arxiv.org/abs/2508.13084)
* [Congested Clique Counting for Local Gibbs Distributions](https://arxiv.org/abs/2508.13083)
* [WANify: Gauging and Balancing Runtime WAN Bandwidth for Geo-distributed Data Analytics](https://arxiv.org/abs/2508.12961)
* [Accelerating Edge Inference for Distributed MoE Models with Latency-Optimized Expert Placement](https://arxiv.org/abs/2508.12851)
* [Dissecting CPU-GPU Unified Physical Memory on AMD MI300A APUs](https://arxiv.org/abs/2508.12743)
* [DIT: Dimension Reduction View on Optimal NFT Rarity Meters](https://arxiv.org/abs/2508.12671)
* [Proceedings 18th Interaction and Concurrency Experience](https://arxiv.org/abs/2508.12308)
* [Data-driven Trust Bootstrapping for Mobile Edge Computing-based Industrial IoT Services](https://arxiv.org/abs/2508.12560)
* [Attack Graph Generation on HPC Clusters](https://arxiv.org/abs/2508.12161)
* [OS-R1: Agentic Operating System Kernel Tuning with Reinforcement Learning](https://arxiv.org/abs/2508.12551)


### 2025-08-18

* [Efficient GPU-Centered Singular Value Decomposition Using the Divide-and-Conquer Method](https://arxiv.org/abs/2508.11467)
* [Time, Fences and the Ordering of Events in TSO](https://arxiv.org/abs/2508.11415)
* [Space-efficient population protocols for exact majority in general graphs](https://arxiv.org/abs/2508.11384)
* [Inter-APU Communication on AMD MI300A Systems via Infinity Fabric: a Deep Dive](https://arxiv.org/abs/2508.11298)
* [Element and Everything Tokens: Two-Tier Architecture for Mobilizing Alternative Assets](https://arxiv.org/abs/2508.11266)
* [EMLIO: Minimizing I/O Latency and Energy Consumption for Large-Scale AI Training](https://arxiv.org/abs/2508.11035)
* [OpenCXD: An Open Real-Device-Guided Hybrid Evaluation Framework for CXL-SSDs](https://arxiv.org/abs/2508.11477)


### 2025-08-15

* [Minimmit: Fast Finality with Even Faster Blocks](https://arxiv.org/abs/2508.10862)
* [Introducing CQ: A C-like API for Quantum Accelerated HPC](https://arxiv.org/abs/2508.10854)
* [Dalek: An Unconventional and Energy-Aware Heterogeneous Cluster](https://arxiv.org/abs/2508.10481)
* [GPZ: GPU-Accelerated Lossy Compressor for Particle Data](https://arxiv.org/abs/2508.10305)
* [Mixed-Precision Performance Portability of FFT-Based GPU-Accelerated Algorithms for Block-Triangular Toeplitz Matrices](https://arxiv.org/abs/2508.10202)
* [Hard Shell, Reliable Core: Improving Resilience in Replicated Systems with Selective Hybridization](https://arxiv.org/abs/2508.10141)
* [Leveraging OS-Level Primitives for Robotic Action Management](https://arxiv.org/abs/2508.10259)


### 2025-08-14

* [Closing the HPC-Cloud Convergence Gap: Multi-Tenant Slingshot RDMA for Kubernetes](https://arxiv.org/abs/2508.09663)
* [HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap](https://arxiv.org/abs/2508.09591)
* [Verify Distributed Deep Learning Model Implementation Refinement with Iterative Relation Inference](https://arxiv.org/abs/2508.09505)
* [Distributed Diamond Formation of Sliding Squares](https://arxiv.org/abs/2508.09638)
* [Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference](https://arxiv.org/abs/2508.09229)
* [Semantic-Aware LLM Orchestration for Proactive Resource Management in Predictive Digital Twin Vehicular Networks](https://arxiv.org/abs/2508.09149)
* [Holistic Heterogeneous Scheduling for Autonomous Applications using Fine-grained, Multi-XPU Abstraction](https://arxiv.org/abs/2508.09503)
* [A Limits Study of Memory-side Tiering Telemetry](https://arxiv.org/abs/2508.09351)


### 2025-08-13

* [P/D-Device: Disaggregated Large Language Model between Cloud and Devices](https://arxiv.org/abs/2508.09035)
* [A Reinforcement Learning-Driven Task Scheduling Algorithm for Multi-Tenant Distributed Systems](https://arxiv.org/abs/2508.08525)
* [Profiling Concurrent Vision Inference Workloads on NVIDIA Jetson -- Extended](https://arxiv.org/abs/2508.08430)
* [Ultra Ethernet's Design Principles and Architectural Innovations](https://arxiv.org/abs/2508.08906)
* [Scalable Graph Indexing using GPUs for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2508.08744)
* [Two for One, One for All: Deterministic LDC-based Robust Computation in Congested Clique](https://arxiv.org/abs/2508.08740)
* [A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models](https://arxiv.org/abs/2508.08712)
* [Vector-Centric Machine Learning Systems: A Cross-Stack Approach](https://arxiv.org/abs/2508.08469)
* [Towards Efficient and Practical GPU Multitasking in the Era of LLM](https://arxiv.org/abs/2508.08448)
* [Extremely Scalable Distributed Computation of Contour Trees via Pre-Simplification](https://arxiv.org/abs/2508.08433)
* [XDMA: A Distributed, Extensible DMA Architecture for Layout-Flexible Data Movements in Heterogeneous Multi-Accelerator SoCs](https://arxiv.org/abs/2508.08396)
* [Towards Efficient and Practical GPU Multitasking in the Era of LLM](https://arxiv.org/abs/2508.08448)
* [Ultra Ethernet's Design Principles and Architectural Innovations](https://arxiv.org/abs/2508.08906)
* [Selective KV-Cache Sharing to Mitigate Timing Side-Channels in LLM Inference](https://arxiv.org/abs/2508.08438)


### 2025-08-12

* [On the Operational Resilience of CBDC: Threats and Prospects of Formal Validation for Offline Payments](https://arxiv.org/abs/2508.08064)
* [Performance Evaluation of Brokerless Messaging Libraries](https://arxiv.org/abs/2508.07934)
* [Towards Lock Modularization for Heterogeneous Environments](https://arxiv.org/abs/2508.07756)
* [Over-the-Top Resource Broker System for Split Computing: An Approach to Distribute Cloud Computing Infrastructure](https://arxiv.org/abs/2508.07744)
* [Perpetual exploration in anonymous synchronous networks with a Byzantine black hole](https://arxiv.org/abs/2508.07703)
* [Taming Cold Starts: Proactive Serverless Scheduling with Model Predictive Control](https://arxiv.org/abs/2508.07640)
* [Coordinated Power Management on Heterogeneous Systems](https://arxiv.org/abs/2508.07605)
* [An Experimental Exploration of In-Memory Computing for Multi-Layer Perceptrons](https://arxiv.org/abs/2508.07317)
* [FlashMP: Fast Discrete Transform-Based Solver for Preconditioning Maxwell's Equations on GPUs](https://arxiv.org/abs/2508.07193)
* [The Fused Kernel Library: A C++ API to Develop Highly-Efficient GPU Libraries](https://arxiv.org/abs/2508.07071)
* [Convergence Sans Synchronization](https://arxiv.org/abs/2508.06949)
* [Kairos: Low-latency Multi-Agent Serving with Shared LLMs and Excessive Loads in the Public Cloud](https://arxiv.org/abs/2508.06948)
* [PiKV: KV Cache Management System for Mixture of Experts](https://arxiv.org/abs/2508.06526)
* [Fully-Fluctuating Participation in Sleepy Consensus](https://arxiv.org/abs/2508.08068)
* [GPU-Accelerated Syndrome Decoding for Quantum LDPC Codes below the 63 $μ$s Latency Threshold](https://arxiv.org/abs/2508.07879)
* [Enhancing Privacy in Decentralized Min-Max Optimization: A Differentially Private Approach](https://arxiv.org/abs/2508.07505)
* [Real-Time Analysis of Unstructured Data with Machine Learning on Heterogeneous Architectures](https://arxiv.org/abs/2508.07423)
* [DSperse: A Framework for Targeted Verification in Zero-Knowledge Machine Learning](https://arxiv.org/abs/2508.06972)
* [A Portable Multi-GPU Solver for Collisional Plasmas with Coulombic Interactions](https://arxiv.org/abs/2508.06771)
* [PANAMA: A Network-Aware MARL Framework for Multi-Agent Path Finding in Digital Twin Ecosystems](https://arxiv.org/abs/2508.06767)


### 2025-08-11

* [Performant Unified GPU Kernels for Portable Singular Value Computation Across Hardware and Precision](https://arxiv.org/abs/2508.06339)
* [KV Cache Compression for Inference Efficiency in LLMs: A Review](https://arxiv.org/abs/2508.06297)
* [EC2MoE: Adaptive End-Cloud Pipeline Collaboration Enabling Scalable Mixture-of-Experts Inference](https://arxiv.org/abs/2508.06024)
* [KnapFormer: An Online Load Balancer for Efficient Diffusion Transformers Training](https://arxiv.org/abs/2508.06001)
* [Snowpark: Performant, Secure, User-Friendly Data Engineering and AI/ML Next To Your Data](https://arxiv.org/abs/2508.05904)
* [A Dynamic Approach to Load Balancing in Cloud Infrastructure: Enhancing Energy Efficiency and Resource Utilization](https://arxiv.org/abs/2508.05821)
* [Accelerating Data Chunking in Deduplication Systems using Vector Instructions](https://arxiv.org/abs/2508.05797)
* [Voting-Based Semi-Parallel Proof-of-Work Protocol](https://arxiv.org/abs/2508.06489)


### 2025-08-08

* [Simulating LLM training workloads for heterogeneous compute and network infrastructure](https://arxiv.org/abs/2508.05370)


### 2025-08-07

* [S2M3: Split-and-Share Multi-Modal Models for Distributed Multi-Task Inference on the Edge](https://arxiv.org/abs/2508.04271)


### 2025-08-06

* [Block: Balancing Load in LLM Serving with Context, Knowledge and Predictive Scheduling](https://arxiv.org/abs/2508.03611)
* [Frontier: Simulating the Next Generation of LLM Inference Systems](https://arxiv.org/abs/2508.03148)


### 2025-08-05

* [PUSHtap: PIM-based In-Memory HTAP with Unified Data Storage Format](https://arxiv.org/abs/2508.02309)
* [Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving](https://arxiv.org/abs/2508.01989)


### 2025-08-04

* [SwarnRaft: Leveraging Consensus for Robust Drone Swarm Coordination in GNSS-Degraded Environments](https://arxiv.org/abs/2508.00622)
* [Adacc: Adaptive Compression and Activation Checkpointing for LLM Memory Management](https://arxiv.org/abs/2508.00806)
* [Quality-of-Service Aware LLM Routing for Edge Computing with Multiple Experts](https://arxiv.org/abs/2508.00234)


### 2025-07-31

* [DSPE: Profit Maximization in Edge-Cloud Storage System using Dynamic Space Partitioning with Erasure Code](https://arxiv.org/abs/2507.22801)
* [Leveraging Caliper and Benchpark to Analyze MPI Communication Patterns: Insights from AMG2023, Kripke, and Laghos](https://arxiv.org/abs/2507.22372)


### 2025-07-30

* [LeMix: Unified Scheduling for LLM Training and Inference on Multi-GPU Systems](https://arxiv.org/abs/2507.21276)
* [Advancing Compositional LLM Reasoning with Structured Task Relations in Interactive Multimodal Communications](https://arxiv.org/abs/2507.21199)


### 2025-07-29

* [MegatronApp: Efficient and Comprehensive Management on Distributed LLM Training](https://arxiv.org/abs/2507.19845)


### 2025-07-28

* [RailX: A Flexible, Scalable, and Low-Cost Network Architecture for Hyper-Scale LLM Training Systems](https://arxiv.org/abs/2507.18889)


### 2025-07-25

* [Cloud Native System for LLM Inference Serving](https://arxiv.org/abs/2507.18007)
* [Unlock the Potential of Fine-grained LLM Serving via Dynamic Module Scaling](https://arxiv.org/abs/2507.18006)
* [Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving](https://arxiv.org/abs/2507.18454)


### 2025-07-24

* [BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs](https://arxiv.org/abs/2507.17133)
* [BucketServe: Bucket-Based Dynamic Batching for Smart and Efficient LLM Inference Serving](https://arxiv.org/abs/2507.17120)


### 2025-07-23

* [Cooling Matters: Benchmarking Large Language Models and Vision-Language Models on Liquid-Cooled Versus Air-Cooled H100 GPU Systems](https://arxiv.org/abs/2507.16781)
* [Collaborative Inference and Learning between Edge SLMs and Cloud LLMs: A Survey of Algorithms, Execution, and Open Challenges](https://arxiv.org/abs/2507.16731)
* [Reducing GPU Memory Fragmentation via Spatio-Temporal Planning for Efficient Large-Scale Model Training](https://arxiv.org/abs/2507.16274)


### 2025-07-22

* [Efficient Routing of Inference Requests across LLM Instances in Cloud-Edge Computing](https://arxiv.org/abs/2507.15553)
* [GALE: Leveraging Heterogeneous Systems for Efficient Unstructured Mesh Data Analysis](https://arxiv.org/abs/2507.15230)
* [Byzantine-Robust Decentralized Coordination of LLM Agents](https://arxiv.org/abs/2507.14928)
* [Characterizing Communication Patterns in Distributed Large Language Model Inference](https://arxiv.org/abs/2507.14392)
* [IDSS, a Novel P2P Relational Data Storage Service](https://arxiv.org/abs/2507.14682)
* [A Sparsity Predicting Approach for Large Language Models via Activation Pattern Clustering](https://arxiv.org/abs/2507.14179)


### 2025-07-21

* [DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training](https://arxiv.org/abs/2507.13833)
* [Leveraging Multi-Instance GPUs through moldable task scheduling](https://arxiv.org/abs/2507.13601)
* [An End-to-End DNN Inference Framework for the SpiNNaker2 Neuromorphic MPSoC](https://arxiv.org/abs/2507.13736)


### 2025-07-18

* [BootSeer: Analyzing and Mitigating Initialization Bottlenecks in Large-Scale LLM Training](https://arxiv.org/abs/2507.12619)


### 2025-07-17

* [Toward Efficient SpMV in Sparse LLMs via Block Extraction and Compressed Storage](https://arxiv.org/abs/2507.12205)
* [Arctic Inference with Shift Parallelism: Fast and Efficient Open Source Inference System for Enterprise AI](https://arxiv.org/abs/2507.11830)


### 2025-07-16

* [Quantifying the Energy Consumption and Carbon Emissions of LLM Inference via Simulations](https://arxiv.org/abs/2507.11417)
* [MIRAGE: KV Cache Optimization through Parameter Remapping for Multi-tenant LLM Serving](https://arxiv.org/abs/2507.11507)


### 2025-07-15

* [Zorse: Optimizing LLM Training Efficiency on Heterogeneous GPU Clusters](https://arxiv.org/abs/2507.10392)
* [Cross-Timeslot Optimization for Distributed GPU Inference Using Reinforcement Learning](https://arxiv.org/abs/2507.10259)
* [Past-Future Scheduler for LLM Serving under SLA Guarantees](https://arxiv.org/abs/2507.10150)
* [ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism](https://arxiv.org/abs/2507.10069)
* [EAT: QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion Reinforcement Learning](https://arxiv.org/abs/2507.10026)
* [Green-LLM: Optimal Workload Allocation for Environmentally-Aware Distributed Inference](https://arxiv.org/abs/2507.09942)
* [SLIM: A Heterogeneous Accelerator for Edge Inference of Sparse Large Language Model via Adaptive Thresholding](https://arxiv.org/abs/2507.09201)
* [On Evaluating Performance of LLM Inference Serving Systems](https://arxiv.org/abs/2507.09019)


### 2025-07-11

* [KIS-S: A GPU-Aware Kubernetes Inference Simulator with RL-Based Auto-Scaling](https://arxiv.org/abs/2507.07932)
* [KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows](https://arxiv.org/abs/2507.07400)
* [Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding](https://arxiv.org/abs/2507.07120)
* [Analysing semantic data storage in Distributed Ledger Technologies for Data Spaces](https://arxiv.org/abs/2507.07116)


### 2025-07-10

* [Nexus: Taming Throughput-Latency Tradeoff in LLM Serving via Efficient GPU Sharing](https://arxiv.org/abs/2507.06608)
* [SlimCaching: Edge Caching of Mixture-of-Experts for Distributed Inference](https://arxiv.org/abs/2507.06567)


### 2025-07-08

* [On Fault Tolerance of Data Storage Systems: A Holistic Perspective](https://arxiv.org/abs/2507.03849)
* [Analysis and Optimized CXL-Attached Memory Allocation for Long-Context LLM Fine-Tuning](https://arxiv.org/abs/2507.03305)
* [Symbiosis: Multi-Adapter Inference and Fine-Tuning](https://arxiv.org/abs/2507.03220)
* [ZettaLith: An Architectural Exploration of Extreme-Scale AI Inference Acceleration](https://arxiv.org/abs/2507.02871)
* [Performance Evaluation of General Purpose Large Language Models for Basic Linear Algebra Subprograms Code Generation](https://arxiv.org/abs/2507.04697)


### 2025-07-04

* [FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference](https://arxiv.org/abs/2507.02620)
* [Dissecting the Impact of Mobile DVFS Governors on LLM Inference Performance and Energy Efficiency](https://arxiv.org/abs/2507.02135)


### 2025-07-03

* [Deep Recommender Models Inference: Automatic Asymmetric Data Flow Optimization](https://arxiv.org/abs/2507.01676)
* [EdgeLoRA: An Efficient Multi-Tenant LLM Serving System on Edge Devices](https://arxiv.org/abs/2507.01438)


### 2025-07-02

* [Accelerating Loading WebGraphs in ParaGrapher](https://arxiv.org/abs/2507.00716)
* [DynoStore: A wide-area distribution system for the management of data over heterogeneous storage](https://arxiv.org/abs/2507.00576)
* [LLM-Mesh: Enabling Elastic Sharing for Serverless LLM Inference](https://arxiv.org/abs/2507.00507)
* [Serving LLMs in HPC Clusters: A Comparative Study of Qualcomm Cloud AI 100 Ultra and High-Performance GPUs](https://arxiv.org/abs/2507.00418)
* [Toward Edge General Intelligence with Multiple-Large Language Model (Multi-LLM): Architecture, Trust, and Orchestration](https://arxiv.org/abs/2507.00672)
* [HelixPipe: Efficient Distributed Training of Long Sequence Transformers with Attention Parallel Pipeline Parallelism](https://arxiv.org/abs/2507.00394)


### 2025-07-01

* [Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC](https://arxiv.org/abs/2506.24045)
* [QPART: Adaptive Model Quantization and Dynamic Workload Balancing for Accuracy-aware Edge Inference](https://arxiv.org/abs/2506.23934)
* [Towards Building Private LLMs: Exploring Multi-Node Expert Parallelism on Apple Silicon for Mixture-of-Experts Large Language Model](https://arxiv.org/abs/2506.23635)


### 2025-06-30

* [MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism](https://arxiv.org/abs/2506.22175)
* [SiPipe: Bridging the CPU-GPU Utilization Gap for Efficient Pipeline-Parallel LLM Inference](https://arxiv.org/abs/2506.22033)


### 2025-06-27

* [ParEval-Repo: A Benchmark Suite for Evaluating LLMs with Repository-level HPC Translation Tasks](https://arxiv.org/abs/2506.20938)


### 2025-06-26

* [Breaking the Boundaries of Long-Context LLM Inference: Adaptive KV Management on a Single Commodity GPU](https://arxiv.org/abs/2506.20187)
* [MNN-AECS: Energy Optimization for LLM Decoding on Mobile Devices via Adaptive Core Selection](https://arxiv.org/abs/2506.19884)


### 2025-06-25

* [Shelby: Decentralized Storage Designed to Serve](https://arxiv.org/abs/2506.19233)
* [Vertex addition to a ball graph with application to reliability and area coverage in autonomous swarms](https://arxiv.org/abs/2506.19197)
* [Binsparse: A Specification for Cross-Platform Storage of Sparse Matrices and Tensors](https://arxiv.org/abs/2506.19175)


### 2025-06-24

* [Leveraging Cloud-Fog Automation for Autonomous Collision Detection and Classification in Intelligent Unmanned Surface Vehicles](https://arxiv.org/abs/2506.18024)
* [Research on Model Parallelism and Data Parallelism Optimization Methods in Large Language Model-Based Recommendation Systems](https://arxiv.org/abs/2506.17551)
* [Leveraging Large Language Model for Intelligent Log Processing and Autonomous Debugging in Cloud AI Platforms](https://arxiv.org/abs/2506.17900)
* [VeriLocc: End-to-End Cross-Architecture Register Allocation via LLM](https://arxiv.org/abs/2506.17506)


### 2025-06-23

* [TrainVerify: Equivalence-Based Verification for Distributed LLM Training](https://arxiv.org/abs/2506.15961)


### 2025-06-19

* [All is Not Lost: LLM Recovery without Checkpoints](https://arxiv.org/abs/2506.15461)
* [eLLM: Elastic Memory Management Framework for Efficient LLM Serving](https://arxiv.org/abs/2506.15155)
* [Cost-Efficient Serving of LLM Agents via Test-Time Plan Caching](https://arxiv.org/abs/2506.14852)
* [Efficient Serving of LLM Applications with Probabilistic Demand Modeling](https://arxiv.org/abs/2506.14851)


### 2025-06-18

* [Keigo: Co-designing Log-Structured Merge Key-Value Stores with a Non-Volatile, Concurrency-aware Storage Hierarchy (Extended Version)](https://arxiv.org/abs/2506.14630)


### 2025-06-17

* [Serving Large Language Models on Huawei CloudMatrix384](https://arxiv.org/abs/2506.12708)
* [HarMoEny: Efficient Multi-GPU Inference of MoE Models](https://arxiv.org/abs/2506.12417)
* [NaSh: Guardrails for an LLM-Powered Natural Language Shell](https://arxiv.org/abs/2506.13028)
* [Semantic Scheduling for LLM Inference](https://arxiv.org/abs/2506.12204)


### 2025-06-16

* [A retrospective on DISPEED -- Leveraging heterogeneity in a drone swarm for IDS execution](https://arxiv.org/abs/2506.11800)
* [SwiftSpec: Ultra-Low Latency LLM Decoding by Scaling Asynchronous Speculative Decoding](https://arxiv.org/abs/2506.11309)


### 2025-06-13

* [TD-Pipe: Temporally-Disaggregated Pipeline Parallelism Architecture for High-Throughput LLM Inference](https://arxiv.org/abs/2506.10470)
* [HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration](https://arxiv.org/abs/2506.10401)


### 2025-06-12

* [Understanding the Performance and Power of LLM Inferencing on Edge Accelerators](https://arxiv.org/abs/2506.09554)
* [SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving](https://arxiv.org/abs/2506.09397)
* [ScalableHD: Scalable and High-Throughput Hyperdimensional Computing Inference on Multi-Core CPUs](https://arxiv.org/abs/2506.09282)
* [EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model](https://arxiv.org/abs/2506.09061)


### 2025-06-11

* [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027)


### 2025-06-10

* [Addressing tokens dynamic generation, propagation, storage and renewal to secure the GlideinWMS pilot based jobs and system](https://arxiv.org/abs/2506.07379)
* [Cost-Efficient LLM Training with Lifetime-Aware Tensor Offloading via GPUDirect Storage](https://arxiv.org/abs/2506.06472)
* [Towards Efficient Multi-LLM Inference: Characterization and Analysis of LLM Routing and Hierarchical Techniques](https://arxiv.org/abs/2506.06579)


### 2025-06-09

* [Beyond the Buzz: A Pragmatic Take on Inference Disaggregation](https://arxiv.org/abs/2506.05508)


### 2025-06-06

* [FlashDMoE: Fast Distributed MoE in a Single Kernel](https://arxiv.org/abs/2506.04667)
* [SkimROOT: Accelerating LHC Data Filtering with Near-Storage Processing](https://arxiv.org/abs/2506.04507)
* [Knowledge-Guided Attention-Inspired Learning for Task Offloading in Vehicle Edge Computing](https://arxiv.org/abs/2506.04456)
* [Inference economics of language models](https://arxiv.org/abs/2506.04645)


### 2025-06-05

* [Cascadia: A Cascade Serving System for Large Language Models](https://arxiv.org/abs/2506.04203)
* [Crowd-SFT: Crowdsourcing for LLM Alignment](https://arxiv.org/abs/2506.04063)
* [Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs](https://arxiv.org/abs/2506.03296)


### 2025-06-04

* [Adaptive Configuration Selection for Multi-Model Inference Pipelines in Edge Computing](https://arxiv.org/abs/2506.02814)
* [Simplifying Root Cause Analysis in Kubernetes with StateGraph and LLM](https://arxiv.org/abs/2506.02490)
* [D-Rex: Heterogeneity-Aware Reliability Framework and Adaptive Algorithms for Distributed Storage](https://arxiv.org/abs/2506.02026)
* [Evaluating the Efficacy of LLM-Based Reasoning for Multiobjective HPC Job Scheduling](https://arxiv.org/abs/2506.02025)
* [NestedFP: High-Performance, Memory-Efficient Dual-Precision Floating Point Support for LLMs](https://arxiv.org/abs/2506.02024)
* [DistMLIP: A Distributed Inference Platform for Machine Learning Interatomic Potentials](https://arxiv.org/abs/2506.02023)
* [Efficient and Workload-Aware LLM Serving via Runtime Layer Swapping and KV Cache Resizing](https://arxiv.org/abs/2506.02006)


### 2025-06-03

* [Adaptive, Efficient and Fair Resource Allocation in Cloud Datacenters leveraging Weighted A3C Deep Reinforcement Learning](https://arxiv.org/abs/2506.00929)
* [Advancing AI-assisted Hardware Design with Hierarchical Decentralized Training and Personalized Inference-Time Optimization](https://arxiv.org/abs/2506.00002)


### 2025-06-02

* [Distributed Intelligence in the Computing Continuum with Active Inference](https://arxiv.org/abs/2505.24618)
* [SkyLB: A Locality-Aware Cross-Region Load Balancer for LLM Inference](https://arxiv.org/abs/2505.24095)
* [EmbAdvisor: Adaptive Cache Management for Sustainable LLM Serving](https://arxiv.org/abs/2505.23970)


### 2025-05-30

* [Sustainable Carbon-Aware and Water-Efficient LLM Scheduling in Geo-Distributed Cloud Datacenters](https://arxiv.org/abs/2505.23554)
* [MemAscend: System Memory Optimization for SSD-Offloaded LLM Fine-Tuning](https://arxiv.org/abs/2505.23254)
* [Ghidorah: Fast LLM Inference on Edge with Speculative Decoding and Hetero-Core Parallelism](https://arxiv.org/abs/2505.23219)
* [Accelerating AllReduce with a Persistent Straggler](https://arxiv.org/abs/2505.23523)


### 2025-05-29

* [Towards Efficient Key-Value Cache Management for Prefix Prefilling in LLM Inference](https://arxiv.org/abs/2505.21919)

### 2025-05-27

* [DGRAG: Distributed Graph-based Retrieval-Augmented Generation in Edge-Cloud Systems](https://arxiv.org/abs/2505.19847)
* [Win Fast or Lose Slow: Balancing Speed and Accuracy in Latency-Sensitive Decisions of LLMs](https://arxiv.org/abs/2505.19481)

### 2025-05-26

* [H2:Towards Efficient Large-Scale LLM Training on Hyper-Heterogeneous Cluster over 1,000 Chips](https://arxiv.org/abs/2505.17548)
* [Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models](https://arxiv.org/abs/2505.17826)

### 2025-05-23

* [Edge-First Language Model Inference: Models, Metrics, and Tradeoffs](https://arxiv.org/abs/2505.16508)
* [Recursive Offloading for LLM Serving in Multi-tier Networks](https://arxiv.org/abs/2505.16502)

### 2025-05-22

* [Balanced and Elastic End-to-end Training of Dynamic LLMs](https://arxiv.org/abs/2505.14864)


### 2025-05-21

* [ServerlessLoRA: Minimizing Latency and Cost in Serverless Inference for LoRA-Based LLMs](https://arxiv.org/abs/2505.14468)

### 2025-05-20

* [HydraInfer: Hybrid Disaggregated Scheduling for Multimodal Large Language Model Serving](https://arxiv.org/abs/2505.12658)
* [Arrow: Adaptive Scheduling Mechanisms for Disaggregated LLM Inference Architecture](https://arxiv.org/abs/2505.11916)
* [Occult: Optimizing Collaborative Communication across Experts for Accelerated Parallel MoE Training and Inference](https://arxiv.org/abs/2505.13345)

### 2025-05-19

* [TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference](https://arxiv.org/abs/2505.11329)
* [MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production](https://arxiv.org/abs/2505.11432)
* [MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse Mixture-of-Experts Systems](https://arxiv.org/abs/2505.11415)


### 2025-05-16

* [ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production](https://arxiv.org/abs/2505.09999)


### 2025-05-15

* [ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor](https://arxiv.org/abs/2505.09142)

### 2025-05-14

* [Fused3S: Fast Sparse Attention on Tensor Cores](https://arxiv.org/abs/2505.08098)
* [Patchwork: A Unified Framework for RAG Serving](https://arxiv.org/abs/2505.07833)

### 2025-05-13

* [PrefillOnly: An Inference Engine for Prefill-only Workloads in Large Language Model Applications](https://arxiv.org/abs/2505.07203)
* [SneakPeek: Data-Aware Model Selection and Scheduling for Inference Serving on the Edge](https://arxiv.org/abs/2505.06641)
* [Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference](https://arxiv.org/abs/2505.06461)
* [SpecRouter: Adaptive Routing for Multi-Level Speculative Decoding in Large Language Models](https://arxiv.org/abs/2505.07680)
* [QoS-Efficient Serving of Multiple Mixture-of-Expert LLMs Using Partial Runtime Reconfiguration](https://arxiv.org/abs/2505.06481)
* [Towards Efficient LLM Storage Reduction via Tensor Deduplication and Delta Compression](https://arxiv.org/abs/2505.06252)


### 2025-05-12

* [Understanding Stragglers in Large Model Training Using What-if Analysis](https://arxiv.org/abs/2505.05713)


### 2025-05-09

* [Walrus: An Efficient Decentralized Storage Network](https://arxiv.org/abs/2505.05370)
* [Exploring Influence Factors on LLM Suitability for No-Code Development of End User IoT Applications](https://arxiv.org/abs/2505.04710)
* [HiPerRAG: High-Performance Retrieval Augmented Generation for Scientific Insights](https://arxiv.org/abs/2505.04846)


### 2025-05-08

* [Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving](https://arxiv.org/abs/2505.04021)
* [Rollbaccine : Herd Immunity against Storage Rollback Attacks in TEEs [Technical Report]](https://arxiv.org/abs/2505.04014)
* [Splitwiser: Efficient LM inference with constrained resources](https://arxiv.org/abs/2505.03763)


### 2025-05-06

* [Large Language Model Partitioning for Low-Latency Inference at the Edge](https://arxiv.org/abs/2505.02533)
* [Opt-GPTQ: An Optimized GPTQ Combining Sparse Attention and Quantization Techniques](https://arxiv.org/abs/2505.02351)
* [HAS-GPU: Efficient Hybrid Auto-scaling with Fine-grained GPU Allocation for SLO-aware Serverless Inferences](https://arxiv.org/abs/2505.01968)
* [HSplitLoRA: A Heterogeneous Split Parameter-Efficient Fine-Tuning Framework for Large Language Models](https://arxiv.org/abs/2505.02795)

### 2025-05-05

* [CaGR-RAG: Context-aware Query Grouping for Disk-based Vector Search in RAG Systems](https://arxiv.org/abs/2505.01164)


### 2025-04-30

* [Leveraging Neural Graph Compilers in Machine Learning Research for Edge-Cloud Systems](https://arxiv.org/abs/2504.20198)
* [GenTorrent: Scaling Large Language Model Serving with An Overley Network](https://arxiv.org/abs/2504.20101)
* [Tempo: Application-aware LLM Serving with Mixed SLO Requirements](https://arxiv.org/abs/2504.20068)
* [OSVBench: Benchmarking LLMs on Specification Generation Tasks for Operating System Verification](https://arxiv.org/abs/2504.20964)


### 2025-04-29

* [Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration](https://arxiv.org/abs/2504.19516)
* [Adaptra: Straggler-Resilient Hybrid-Parallel Training with Pipeline Adaptation](https://arxiv.org/abs/2504.19232)
* [semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage](https://arxiv.org/abs/2504.19867)
* [Taming the Titans: A Survey of Efficient LLM Inference Serving](https://arxiv.org/abs/2504.19720)


### 2025-04-28

* [EcoServe: Enabling Cost-effective LLM Serving with Proactive Intra- and Inter-Instance Orchestration](https://arxiv.org/abs/2504.18154)


### 2025-04-24

* [Preemption Aware Task Scheduling for Priority and Deadline Constrained DNN Inference Task Offloading in Homogeneous Mobile-Edge Networks](https://arxiv.org/abs/2504.16792)
* [Real-time Bayesian inference at extreme scale: A digital twin for tsunami early warning applied to the Cascadia subduction zone](https://arxiv.org/abs/2504.16344)
* [HPU: High-Bandwidth Processing Unit for Scalable, Cost-effective LLM Inference via GPU Co-processing](https://arxiv.org/abs/2504.16112)

### 2025-04-23

* [SeaLLM: Service-Aware and Latency-Optimized Resource Sharing for Large Language Model Inference](https://arxiv.org/abs/2504.15720)
* [High-Throughput LLM inference on Heterogeneous Clusters](https://arxiv.org/abs/2504.15303)
* [RAGDoll: Efficient Offloading-based Online RAG System on a Single GPU](https://arxiv.org/abs/2504.15302)
* [D$^{2}$MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving](https://arxiv.org/abs/2504.15299)
* [Scalability Optimization in Cloud-Based AI Inference Services: Strategies for Real-Time Load Balancing and Automated Scaling](https://arxiv.org/abs/2504.15296)
* [StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation](https://arxiv.org/abs/2504.15930)
* [RAGDoll: Efficient Offloading-based Online RAG System on a Single GPU](https://arxiv.org/abs/2504.15302)

### 2025-04-22

* [SLO-Aware Scheduling for Large Language Model Inferences](https://arxiv.org/abs/2504.14966)
* [gLLM: Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling](https://arxiv.org/abs/2504.14775)
* [Joint Optimization of Offloading, Batching and DVFS for Multiuser Co-Inference](https://arxiv.org/abs/2504.14611)
* [MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core](https://arxiv.org/abs/2504.14960)
* [Optimizing SLO-oriented LLM Serving with PD-Multiplexing](https://arxiv.org/abs/2504.14489)

### 2025-04-18

* [You Don't Need All Attentions: Distributed Dynamic Fine-Tuning for Foundation Models](https://arxiv.org/abs/2504.12471)

### 2025-04-17

* [Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures](https://arxiv.org/abs/2504.11750)
* [Cost-Efficient LLM Serving in the Cloud: VM Selection with KV Cache Offloading](https://arxiv.org/abs/2504.11816)
* [70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float](https://arxiv.org/abs/2504.11651)

### 2025-04-16

* [Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints](https://arxiv.org/abs/2504.11320)

### 2025-04-15

* [Optimal Graph Stretching for Distributed Averaging](https://arxiv.org/abs/2504.10289)
* [Training LLMs on HPC Systems: Best Practices from the OpenGPT-X Project](https://arxiv.org/abs/2504.10013)
* [MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints](https://arxiv.org/abs/2504.09345)
* [Lumos: Efficient Performance Modeling and Estimation for Large-scale LLM Training](https://arxiv.org/abs/2504.09307)
* [DynaServe: Unified and Elastic Tandem-Style Execution for Dynamic Disaggregated LLM Serving](https://arxiv.org/abs/2504.09285)
* [SpecEE: Accelerating Large Language Model Inference with Speculative Early Exiting](https://arxiv.org/abs/2504.08850)
* [DARIS: An Oversubscribed Spatio-Temporal Scheduler for Real-Time DNN Inference on GPUs](https://arxiv.org/abs/2504.08795)
* [PRIMA.CPP: Speeding Up 70B-Scale LLM Inference on Low-Resource Everyday Home Clusters](https://arxiv.org/abs/2504.08791)
* [SLOs-Serve: Optimized Serving of Multi-SLO LLMs](https://arxiv.org/abs/2504.08784)
* [MigGPT: Harnessing Large Language Models for Automated Migration of Out-of-Tree Linux Kernel Patches Across Versions](https://arxiv.org/abs/2504.09474)

### 2025-04-14

* [Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices](https://arxiv.org/abs/2504.08242)

### 2025-04-11

* [Token Level Routing Inference System for Edge Devices](https://arxiv.org/abs/2504.07878)

### 2025-04-09

* [Nonuniform-Tensor-Parallelism: Mitigating GPU failure impact for Scaled-up LLM Training](https://arxiv.org/abs/2504.06095)
* [HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference](https://arxiv.org/abs/2504.05897)

### 2025-04-08

* [IntentContinuum: Using LLMs to Support Intent-Based Computing Across the Compute Continuum](https://arxiv.org/abs/2504.04429)
* [HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs](https://arxiv.org/abs/2504.03871)
* [FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling](https://arxiv.org/abs/2504.03775)
* [Adaptive Orchestration for Inference of Large Foundation Models at the Edge](https://arxiv.org/abs/2504.03668)
* [LLM & HPC:Benchmarking DeepSeek's Performance in High-Performance Computing Tasks](https://arxiv.org/abs/2504.03665)
* [PIPO: Pipelined Offloading for Efficient Inference on Consumer Devices](https://arxiv.org/abs/2504.03664)

### 2025-04-07

* [LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM Applications](https://arxiv.org/abs/2504.03444)

### 2025-04-04

* [FT-Transformer: Resilient and Reliable Transformer with End-to-End Fault Tolerant Attention](https://arxiv.org/abs/2504.02211)

### 2025-04-02

* [AMP4EC: Adaptive Model Partitioning Framework for Efficient Deep Learning Inference in Edge Computing Environments](https://arxiv.org/abs/2504.00407)

### 2025-04-01

* [OrchMLLM: Orchestrate Multimodal Data with Batch Post-Balancing to Accelerate Multimodal Large Language Model Training](https://arxiv.org/abs/2503.23830)
* [MVDRAM: Enabling GeMV Execution in Unmodified DRAM for Low-Bit LLM Acceleration](https://arxiv.org/abs/2503.23817)


### 2025-03-31

* [Niyama : Breaking the Silos of LLM Inference Serving](https://arxiv.org/abs/2503.22562)


### 2025-03-28

* [Robust DNN Partitioning and Resource Allocation Under Uncertain Inference Time](https://arxiv.org/abs/2503.21476)
* [Optimizing Multi-DNN Inference on Mobile Devices through Heterogeneous Processor Co-Execution](https://arxiv.org/abs/2503.21109)
* [Scalability Evaluation of HPC Multi-GPU Training for ECG-based LLMs](https://arxiv.org/abs/2503.21033)


### 2025-03-27

* [Injecting Adrenaline into LLM Serving: Boosting Resource Utilization and Throughput via Attention Disaggregation](https://arxiv.org/abs/2503.20552)
* [Harmonia: A Multi-Agent Reinforcement Learning Approach to Data Placement and Migration in Hybrid Storage Systems](https://arxiv.org/abs/2503.20507)
* [L4: Diagnosing Large-scale LLM Training Failures via Automated Log Analysis](https://arxiv.org/abs/2503.20263)


### 2025-03-26

* [Mist: Efficient Distributed Training of Large Language Models via Memory-Parallelism Co-Optimization](https://arxiv.org/abs/2503.19050)


### 2025-03-25

* [Jenga: Effective Memory Management for Serving LLM with Heterogeneity](https://arxiv.org/abs/2503.18292)
* [Risk Management for Distributed Arbitrage Systems: Integrating Artificial Intelligence](https://arxiv.org/abs/2503.18265)
* [WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model Training](https://arxiv.org/abs/2503.17924)
* [PipeBoost: Resilient Pipelined Architecture for Fast Serverless LLM Scaling](https://arxiv.org/abs/2503.17707)
* [A Generative Caching System for Large Language Models](https://arxiv.org/abs/2503.17603)


### 2025-03-24

* [Improving the End-to-End Efficiency of Offline Inference for Multi-LLM Applications Based on Sampling and Simulation](https://arxiv.org/abs/2503.16893)
* [Distributed LLMs and Multimodal Large Language Models: A Survey on Advances, Challenges, and Future Directions](https://arxiv.org/abs/2503.16585)


### 2025-03-21

* [SPIN: Accelerating Large Language Model Inference with Heterogeneous Speculative Models](https://arxiv.org/abs/2503.15921)
* [ATTENTION2D: Communication Efficient Distributed Self-Attention Mechanism](https://arxiv.org/abs/2503.15758)


### 2025-03-20

* [Efficient allocation of image recognition and LLM tasks on multi-GPU system](https://arxiv.org/abs/2503.15252)
* [Prada: Black-Box LLM Adaptation with Private Data on Resource-Constrained Devices](https://arxiv.org/abs/2503.14932)
* [RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving](https://arxiv.org/abs/2503.14649)


### 2025-03-19

* [Do Large Language Models Understand Performance Optimization?](https://arxiv.org/abs/2503.13772)


### 2025-03-18

* [Adaptive Fault Tolerance Mechanisms of Large Language Models in Cloud Computing Environments](https://arxiv.org/abs/2503.12228)
* [FAILS: A Framework for Automated Collection and Analysis of LLM Service Incidents](https://arxiv.org/abs/2503.12185)


### 2025-03-17

* [Beyond A Single AI Cluster: A Survey of Decentralized LLM Training](https://arxiv.org/abs/2503.11023)
* [LLMPerf: GPU Performance Modeling meets Large Language Models](https://arxiv.org/abs/2503.11244)
* [Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores](https://arxiv.org/abs/2503.10725)


### 2025-03-14

* [SPPO:Efficient Long-sequence LLM Training via Adaptive Sequence Pipeline Parallel Offloading](https://arxiv.org/abs/2503.10377)
* [Collaborative Speculative Inference for Efficient LLM Inference Serving](https://arxiv.org/abs/2503.10325)
* [MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching](https://arxiv.org/abs/2503.09716)


### 2025-03-13

* [Performance Models for a Two-tiered Storage System](https://arxiv.org/abs/2503.08966)
* [Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference](https://arxiv.org/abs/2503.09304)
* [Sometimes Painful but Certainly Promising: Feasibility and Trade-offs of Language Model Inference at the Edge](https://arxiv.org/abs/2503.09114)


### 2025-03-12

* [TokenSim: Enabling Hardware and Software Exploration for Large Language Model Inference Systems](https://arxiv.org/abs/2503.08415)
* [Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference](https://arxiv.org/abs/2503.08311)
* [Will LLMs Scaling Hit the Wall? Breaking Barriers via Distributed Resources on Massive Edge Devices](https://arxiv.org/abs/2503.08223)
* [Accelerating MoE Model Inference with Expert Sharding](https://arxiv.org/abs/2503.08467)
* [FastCache: Optimizing Multimodal LLM Serving through Lightweight KV-Cache Compression Framework](https://arxiv.org/abs/2503.08461)
* [DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous and Parallel LLM-based Multi-Agent Systems](https://arxiv.org/abs/2503.07675)


### 2025-03-11

* [Seesaw: High-throughput LLM Inference via Model Re-sharding](https://arxiv.org/abs/2503.06433)
* [eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference](https://arxiv.org/abs/2503.06823)
* [Distributed Graph Neural Network Inference With Just-In-Time Compilation For Industry-Scale Graphs](https://arxiv.org/abs/2503.06208)


### 2025-03-10

* [Optimizing LLM Inference Throughput via Memory-aware and SLA-constrained Dynamic Batching](https://arxiv.org/abs/2503.05248)
* [Linear-MoE: Linear Sequence Modeling Meets Mixture-of-Experts](https://arxiv.org/abs/2503.05447)


### 2025-03-07

* [Dynamic Pricing for On-Demand DNN Inference in the Edge-AI Market](https://arxiv.org/abs/2503.04521)
* [Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling](https://arxiv.org/abs/2503.04398)
* [Malware Detection at the Edge with Lightweight LLMs: A Performance Evaluation](https://arxiv.org/abs/2503.04302)


### 2025-03-06

* [Enhancing Memory Efficiency in Large Language Model Training Through Chronos-aware Pipeline Parallelism](https://arxiv.org/abs/2503.03182)
* [Environment-Aware Dynamic Pruning for Pipelined Edge Inference](https://arxiv.org/abs/2503.03070)


### 2025-03-05

* [SpecInF: Exploiting Idle GPU Resources in Distributed DL Training via Speculative Inference Filling](https://arxiv.org/abs/2503.02550)
* [CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory](https://arxiv.org/abs/2503.02354)
* [VQ-LLM: High-performance Code Generation for Vector Quantization Augmented LLM Inference](https://arxiv.org/abs/2503.02236)
### 2025-03-04

* [Improving inference time in multi-TPU systems with profiled model segmentation](https://arxiv.org/abs/2503.01025)

### 2025-03-03

* [ByteScale: Efficient Scaling of LLM Training with a 2048K Context Length on More Than 12,000 GPUs](https://arxiv.org/abs/2502.21231)
* [TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval](https://arxiv.org/abs/2502.20969)
* [Cicada: A Pipeline-Efficient Approach to Serverless Inference with Decoupled Management](https://arxiv.org/abs/2502.20959)
* [SkyStore: Cost-Optimized Object Storage Across Regions and Clouds](https://arxiv.org/abs/2502.20818)
* [LADs: Leveraging LLMs for AI-Driven DevOps](https://arxiv.org/abs/2502.20825)

### 2025-02-28

* [SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks](https://arxiv.org/abs/2502.19913)

### 2025-02-27

* [CLLoRA: An Approach to Measure the Effects of the Context Length for LLM Fine-Tuning](https://arxiv.org/abs/2502.18910)

### 2025-02-25

* [FairKV: Balancing Per-Head KV Cache for Fast Multi-GPU Inference](https://arxiv.org/abs/2502.15804)
* [Hybrid Offline-online Scheduling Method for Large Language Model Inference Optimization](https://arxiv.org/abs/2502.15763)
* [LoXR: Performance Evaluation of Locally Executing LLMs on XR Devices](https://arxiv.org/abs/2502.15761)
* [DistrEE: Distributed Early Exit of Deep Neural Network Inference on Edge Devices](https://arxiv.org/abs/2502.15735)

### 2025-02-24

* [Towards Swift Serverless LLM Cold Starts with ParaServe](https://arxiv.org/abs/2502.15524)
* [FlexPie: Accelerate Distributed Inference on Edge Devices with Flexible Combinatorial Optimization[Technical Report]](https://arxiv.org/abs/2502.15312)

### 2025-02-21

* [Serving Models, Fast and Slow:Optimizing Heterogeneous LLM Inferencing Workloads at Scale](https://arxiv.org/abs/2502.14617)
* [Optimizing the Longhorn Cloud-native Software Defined Storage Engine for High Performance](https://arxiv.org/abs/2502.14419)
* [CarbonEdge: Leveraging Mesoscale Spatial Carbon-Intensity Variations for Low Carbon Edge Computing](https://arxiv.org/abs/2502.14076)
* [LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention](https://arxiv.org/abs/2502.14866)
* [LLM4FaaS: No-Code Application Development using LLMs and FaaS](https://arxiv.org/abs/2502.14450)

### 2025-02-20

* [Autellix: An Efficient Serving Engine for LLM Agents as General Programs](https://arxiv.org/abs/2502.13965)

### 2025-02-19

* [SparkAttention: High-Performance Multi-Head Attention for Large Models on Volta GPU Architecture](https://arxiv.org/abs/2502.12784)
* [Distributed On-Device LLM Inference With Over-the-Air Computation](https://arxiv.org/abs/2502.12559)
* [Understanding Silent Data Corruption in LLM Training](https://arxiv.org/abs/2502.12340)
* [Semantica: Decentralized Search using a LLM-Guided Semantic Tree Overlay](https://arxiv.org/abs/2502.10151)

### 2025-02-18

* [Scalable and Cost-Efficient ML Inference: Parallel Batch Processing with Serverless Functions](https://arxiv.org/abs/2502.12017)
* [BagChain: A Dual-functional Blockchain Leveraging Bagging-based Distributed Learning](https://arxiv.org/abs/2502.11464)
* [DreamDDP: Accelerating Data Parallel Distributed LLM Training with Layer-wise Scheduled Partial Synchronization](https://arxiv.org/abs/2502.11058)
* [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880)
* [DiSCo: Device-Server Collaborative LLM-Based Text Streaming Services](https://arxiv.org/abs/2502.11417)
* [Local-Cloud Inference Offloading for LLMs in Multi-Modal, Multi-Task, Multi-Dialogue Settings](https://arxiv.org/abs/2502.11007)

### 2025-02-17

* [λScale: Enabling Fast Scaling for Serverless Large Language Model Inference](https://arxiv.org/abs/2502.09922)

### 2025-02-14

* [ThunderServe: High-performance and Cost-efficient LLM Serving in Cloud Environments](https://arxiv.org/abs/2502.09334)

### 2025-02-13

* [Memory Offloading for Large Language Model Inference with Latency SLO Guarantees](https://arxiv.org/abs/2502.08182)
* [HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment](https://arxiv.org/abs/2502.07903)
* [Democratizing AI: Open-source Scalable LLM Training on GPU-based Supercomputers](https://arxiv.org/abs/2502.08145)

### 2025-02-11

* [MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing](https://arxiv.org/abs/2502.06643)
* [fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving](https://arxiv.org/abs/2502.05370)

### 2025-02-10

* [EcoServe: Designing Carbon-Aware AI Inference Systems](https://arxiv.org/abs/2502.05043)
* [WaferLLM: A Wafer-Scale LLM Inference System](https://arxiv.org/abs/2502.04563)

### 2025-02-07

* [HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference](https://arxiv.org/abs/2502.03589)
* [InfinitePOD: Building Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching Transceivers](https://arxiv.org/abs/2502.03885)

### 2025-02-05

* [LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models](https://arxiv.org/abs/2502.02406)
* [Longer Attention Span: Increasing Transformer Context Length with Sparse Graph Processing Techniques](https://arxiv.org/abs/2502.01659)

### 2025-02-04

* [OCTOPINF: Workload-Aware Inference Serving for Edge Video Analytics](https://arxiv.org/abs/2502.01277)
* [Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs](https://arxiv.org/abs/2502.00722)
* [General Coded Computing in a Probabilistic Straggler Regime](https://arxiv.org/abs/2502.00645)
* [Leveraging InfiniBand Controller to Configure Deadlock-Free Routing Engines for Dragonflies](https://arxiv.org/abs/2502.01214)

### 2025-02-03

* [Infer-EDGE: Dynamic DNN Inference Optimization in 'Just-in-time' Edge-AI Implementations](https://arxiv.org/abs/2501.18842)

### 2025-01-30

* [Dual-Lagrange Encoding for Storage and Download in Elastic Computing for Resilience](https://arxiv.org/abs/2501.17275)

### 2025-01-29

* [On the Shape Containment Problem within the Amoebot Model with Reconfigurable Circuits](https://arxiv.org/abs/2501.16892)

### 2025-01-28

* [Static Batching of Irregular Workloads on GPUs: Framework and Application to Efficient MoE Model Inference](https://arxiv.org/abs/2501.16103)
* [Aging-aware CPU Core Management for Embodied Carbon Amortization in Cloud LLM Inference](https://arxiv.org/abs/2501.15829)
* [HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location](https://arxiv.org/abs/2501.14808)
* [HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators](https://arxiv.org/abs/2501.14794)
* [DeServe: Towards Affordable Offline LLM Inference via Decentralization](https://arxiv.org/abs/2501.14784)
* [Dynamic Adaptation in Data Storage: Real-Time Machine Learning for Enhanced Prefetching](https://arxiv.org/abs/2501.14771)

### 2025-01-27

* [Locality-aware Fair Scheduling in LLM Serving](https://arxiv.org/abs/2501.14312)

### 2025-01-22

* [Accelerating End-Cloud Collaborative Inference via Near Bubble-free Pipeline Optimization](https://arxiv.org/abs/2501.12388)
* [DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference](https://arxiv.org/abs/2501.10375)
* [AdaServe: SLO-Customized LLM Serving with Fine-Grained Speculative Decoding](https://arxiv.org/abs/2501.12162)
* [Glinthawk: A Two-Tiered Architecture for High-Throughput LLM Inference](https://arxiv.org/abs/2501.11779)

### 2025-01-20

* [Over-the-Air Multi-Sensor Inference with Neural Networks Using Memristor-Based Analog Computing](https://arxiv.org/abs/2501.10245)

### 2025-01-17

* [PICE: A Semantic-Driven Progressive Inference System for LLM Serving in Cloud-Edge Networks](https://arxiv.org/abs/2501.09367)

### 2025-01-15

* [PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving](https://arxiv.org/abs/2501.08192)
* [HgPCN: A Heterogeneous Architecture for E2E Embedded Point Cloud Inference](https://arxiv.org/abs/2501.07767)

### 2025-01-14

* [CoCoI: Distributed Coded Inference System for Straggler Mitigation](https://arxiv.org/abs/2501.06856)
* [Ladder-residual: parallelism-aware architecture for accelerating large model inference with communication overlapping](https://arxiv.org/abs/2501.06589)

### 2025-01-13

* [A Practical Cross-Layer Approach for ML-Driven Storage Placement in Warehouse-Scale Computers](https://arxiv.org/abs/2501.05651)

### 2025-01-10

* [Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing](https://arxiv.org/abs/2501.05313)

### 2025-01-09

* [Collaborative Inference Acceleration with Non-Penetrative Tensor Partitioning](https://arxiv.org/abs/2501.04489)
* [Scalable Data Notarization Leveraging Hybrid DLTs](https://arxiv.org/abs/2501.04571)

### 2025-01-07

* [TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms](https://arxiv.org/abs/2501.02600)

### 2025-01-06

* [Efficient LLM Inference with Activation Checkpointing and Hybrid Caching](https://arxiv.org/abs/2501.01792)

### 2025-01-03

* [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005)
* [Dynamic Optimization of Storage Systems Using Reinforcement Learning Techniques](https://arxiv.org/abs/2501.00068)

