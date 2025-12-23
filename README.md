
<div align="center">

# Daily Arxiv Papers (LMSys)

![Static Badge](https://img.shields.io/badge/total_papers-1275-blue?logo=gitbook)
![Static Badge](https://img.shields.io/badge/update-2025.12.23-red?logo=fireship)
[![Static Badge](https://img.shields.io/badge/arXiv-cs.DC-green)](https://arxiv.org/list/cs.DC/recent)
[![Static Badge](https://img.shields.io/badge/arXiv-cs.OS-green)](https://arxiv.org/list/cs.OS/recent)
[![Static Badge](https://img.shields.io/badge/arXiv-cs.LG-green)](https://arxiv.org/list/cs.LG/recent)

`Fetch from arxiv` → `LLM Filter` → `GitHub workflow update`

</div>

**👍Conference Papers on LMSys**: [conference.md](conference.md)

**⚠️NOTE**: Update papers up to last day every morning (8:00 UTC+8) automatically.

**🙋WANT**: Keyword subscription (email); Functional web page.

**🔖TAGS**:`serving` `training` `offline` `thinking` `RL` `MoE` `RAG` `video` `multi-modal` `sparse` `quantization` `offloading` `hardware` `storage` `kernel` `diffusion` `agentic` `edge` `networking`

---
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

