# KV Cache Simulator

**A discrete-event simulator for KV cache dynamics in multi-GPU LLM inference systems.**

Explore scheduling policies, memory pressure strategies, and multi-GPU routing decisions without deploying actual hardware. Built for researchers, infrastructure engineers, and anyone curious about the systems challenges behind serving large language models at scale.

---

## Why This Exists

Modern LLM inference is **memory-bound, not compute-bound**. The KV cache—storing key-value pairs for attention—grows with context length and concurrency, becoming the primary bottleneck for:

- **Throughput**: How many tokens/second can we serve?
- **Tail latency**: What's the p99 experience for users?
- **Admission control**: When do we reject vs queue vs evict?
- **Multi-GPU efficiency**: How do we route requests across heterogeneous hardware?

Production systems like vLLM, TGI, and TensorRT-LLM solve these problems with complex heuristics tuned through trial and error. This simulator lets you **experiment with those decisions in minutes instead of hours**, understand their tradeoffs, and develop intuition before touching real infrastructure.

---

## Design Philosophy

1. **Model consequences, not implementations**
   We don't implement CUDA kernels or actual tensors. We model their *performance and memory footprint*, which is the layer that matters for scheduling and capacity planning.

2. **Discrete-event precision**
   No time-stepping approximations. Events (arrivals, prefill completions, decode completions, handoffs) are processed in exact chronological order via a priority queue.

3. **Heterogeneity as a first-class citizen**
   Real clusters have mixed GPU types, asymmetric interconnects, and varying capacities. The simulator supports per-GPU configurations and topology-aware routing.

4. **Tail latency focus**
   Averages lie. We track p50/p95/p99 latencies and time-to-first-token (TTFT) because that's what users experience.

5. **Extensible policy layer**
   Scheduling, routing, eviction, and admission policies are modular. Add your own without touching the simulation core.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Simulator Core                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Request   │  │   Event     │  │      GPU States         │ │
│  │   Queue     │  │   Priority  │  │  (VRAM, active counts,  │ │
│  │  (Global)   │  │   Queue     │  │   per-GPU queues, LRU)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Event Handlers                         │  │
│  │  on_arrival → on_start_prefill → on_start_decode →       │  │
│  │  on_handoff_start → on_handoff_complete → on_finish      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Policy Layer                           │  │
│  │  • Routing (P2C, Round-Robin, Least-Loaded)              │  │
│  │  • Scheduling (FIFO, Shortest-Remaining)                 │  │
│  │  • Memory Pressure (Reject, Evict)                       │  │
│  │  • Eviction (FIFO, LRU)                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Output Files                            │
│  summary.json │ timeseries.csv │ events.jsonl │ run_meta.json  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts Modeled

### Request Lifecycle

```
Arrival → [Global Queue] → Queued → Prefill → [Handoff] → Decode → Finished
                                       ↓                      ↓
                                   Rejected              Evicted
```

1. **Arrival**: Request enters the system with prompt tokens and expected generation length
2. **Routing**: P2C algorithm selects target GPU (see [Algorithmic Highlights](#algorithmic-highlights))
3. **Admission**: Check VRAM capacity; queue, reject, or evict based on policy
4. **Prefill**: Process prompt tokens (compute-intensive, parallelizable)
5. **Decode Routing**: Re-evaluate optimal GPU for decode phase
6. **Handoff**: Transfer KV cache between GPUs if decode GPU differs (topology-aware)
7. **Decode**: Generate tokens one-by-one (memory-bandwidth bound)
8. **Completion**: Free KV cache, update metrics

### Multi-GPU Dynamics

- **Heterogeneous GPUs**: Different VRAM sizes, prefill/decode throughputs per GPU
- **Phase-level routing**: Prefill and decode can run on different GPUs
- **KV handoff**: Transfer KV cache with topology-aware latency and bandwidth
- **Global queue**: Cluster-wide waiting queue when all GPUs are at capacity
- **Cross-GPU retry**: On memory pressure, retry admission on alternate GPU

### Memory Management

- **Safe reservation**: Pre-allocate full KV (prompt + gen) at prefill time
- **Lazy allocation**: Allocate prompt KV at prefill, gen KV at decode (risks late rejection)
- **Eviction policies**: FIFO or LRU eviction under memory pressure
- **Per-request tracking**: Allocated bytes tracked per request per GPU

---

## Algorithmic Highlights

### Power of Two Choices (P2C) Routing

Instead of scanning all GPUs to find the least loaded (O(n)), we sample **two random GPUs** and pick the better one. This achieves:

- **O(1) routing decisions** regardless of cluster size
- **Exponentially better load balancing** than random assignment
- **Proven theoretical guarantees**: Max load drops from O(log n / log log n) to O(log log n)

```cpp
int a = random_gpu(), b = random_gpu();
return score(a) < score(b) ? a : b;
```

The scoring function accounts for active requests, queue depth, and GPU speed (normalized by TPS).

### Floyd-Warshall Topology Precomputation

For multi-GPU routing with arbitrary interconnect topologies, we precompute all-pairs shortest paths:

```cpp
// Precompute optimal bandwidth and latency between all GPU pairs
for (int k = 0; k < num_gpus; k++)
  for (int i = 0; i < num_gpus; i++)
    for (int j = 0; j < num_gpus; j++)
      // Update if routing through k is better
```

This enables **O(1) handoff cost estimation** during decode routing, even with complex topologies (NVLink rings, PCIe trees, etc.).

### Decode Routing Score

Balances load and transfer cost:

```
score = load × (500 / decode_tps) + weight × handoff_time
```

- **Load factor**: Normalized by GPU decode speed (faster GPUs tolerate higher load)
- **Handoff cost**: Latency + transfer time based on KV size and link bandwidth
- **Weight**: Configurable tradeoff between load balancing and locality

---

## Output Metrics

### Summary Statistics (`summary.json`)

| Metric | Description |
|--------|-------------|
| `finished` | Requests that completed successfully |
| `rejected` | Requests rejected due to capacity |
| `completion_rate` | finished / total |
| `reject_rate` | rejected / total |
| `throughput_tokens_per_sec` | Total generated tokens / makespan |
| `p50_latency_ms` | Median end-to-end latency |
| `p95_latency_ms` | 95th percentile latency |
| `p99_latency_ms` | 99th percentile latency |
| `p50_ttft_ms` | Median time-to-first-token |
| `p95_ttft_ms` | 95th percentile TTFT |
| `avg_vram_bytes` | Time-weighted average VRAM usage |
| `gpu_busy_ms` | Total time with active work |
| `makespan_ms` | Total simulation duration |
| `evictions` | Requests evicted under memory pressure |

### Multi-GPU Metrics

| Metric | Description |
|--------|-------------|
| `handoffs_total` | KV cache transfers between GPUs |
| `cross_gpu_decodes` | Requests decoded on different GPU than prefill |
| `retry_attempts` | Admission retries due to memory pressure |
| `retry_successes` | Successful retries on alternate GPU |
| `max_global_queue_depth` | Peak cluster-wide queue depth |
| `per_gpu[].peak_vram_bytes` | High-water mark per GPU |
| `per_gpu[].tokens_generated` | Tokens produced per GPU |
| `per_gpu[].requests_finished` | Completions per GPU |

### Time Series (`timeseries.csv`)

Sampled at configurable intervals:

| Column | Description |
|--------|-------------|
| `time_ms` | Simulation timestamp |
| `vram_used` | Total VRAM across all GPUs |
| `vram_gpu{N}` | Per-GPU VRAM usage |
| `active_prefill` | Concurrent prefill operations |
| `active_decode` | Concurrent decode operations |
| `queue_depth` | Requests in per-GPU queues |
| `global_queue_depth` | Requests in cluster-wide queue |
| `tokens_generated_delta` | Tokens generated since last sample |
| `rejects_delta` | Rejections since last sample |

### Event Log (`events.jsonl`)

Every state transition as line-delimited JSON:

```json
{"time_ms":0,"type":"arrival","request_id":"r1","gpu_index":0}
{"time_ms":100,"type":"start_prefill","request_id":"r1","gpu_index":0}
{"time_ms":500,"type":"handoff_start","request_id":"r1","gpu_index":1}
{"time_ms":505,"type":"handoff_complete","request_id":"r1","gpu_index":1}
{"time_ms":850,"type":"finish","request_id":"r1","gpu_index":1}
```

---

## Configuration Reference(example)

### Global Options

```bash
num_gpus 2                      # Number of GPUs
kv_bytes_per_token 2048         # KV cache size per token
safe_reservation 1              # 1=reserve full KV upfront, 0=lazy allocation
max_queue 64                    # Acceptance threshold: queued + active < max_queue
max_retries 2                   # Cross-GPU retry attempts on admission failure
scheduling fifo                 # fifo | shortest_remaining
memory_pressure_policy reject   # reject | evict
eviction_policy lru             # lru | fifo
timeseries_dt_ms 20             # Sampling interval for time series
```

### Per-GPU Options

```bash
gpu 0 vram_bytes 24000000000    # 24GB VRAM
gpu 0 max_concurrent 16         # Max concurrent requests
gpu 0 prefill_tps 2000          # Prefill tokens/second
gpu 0 decode_tps 500            # Decode tokens/second
gpu 0 decode_sharing_cap 8      # Max batch size for decode
gpu 0 decode_efficiency 0.8     # Throughput scaling factor

gpu 1 vram_bytes 16000000000    # Heterogeneous: smaller GPU
gpu 1 prefill_tps 1500
gpu 1 decode_tps 400
```

### Handoff/Topology Options

```bash
handoff_bandwidth_gbps 300      # Default link bandwidth (NVLink ~300, PCIe ~25)
handoff_latency_us 10           # Fixed latency overhead per transfer

# Custom topology (optional)
link 0 1 bandwidth_gbps 300 latency_ms 0.01   # NVLink between GPU 0-1
link 0 2 bandwidth_gbps 25 latency_ms 0.1     # PCIe between GPU 0-2
```

---

## Guide

### Build

```bash
cd cpp && mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

### Run

```bash
./kv_sim --config <config_file> --trace <trace_file> --out <output_dir> [--seed 12345]
```

### Trace Format

```
# id arrival_ms prompt_tokens gen_tokens streaming(0/1)
req1 0 500 200 0
req2 50 400 150 0
req3 100 600 250 0
```

---

## Web UI

Interactive dashboard for running simulations and visualizing results.

```bash
# Start backend
cd python/server && pip install -r requirements.txt && python main.py

# Start frontend (separate terminal)
cd web && npm install && npm run dev
```

Features:
- Configure all parameters including per-GPU settings
- Real-time plots: VRAM over time, queue depth, throughput
- Per-GPU breakdown table
- KPI cards for all metrics

---

## Future Work

### Routing Policies
- [ ] **Locality-aware routing**: Prefer GPUs with cached prefixes
- [ ] **Predictive routing**: Use request size predictions for better load balancing
- [ ] **Least-connections**: Route to GPU with fewest active requests

### Scheduling Policies
- [ ] **Fair scheduling**: Weighted fair queuing across request classes
- [ ] **Preemption**: Pause low-priority decodes for urgent prefills

### Memory Management
- [ ] **Compression**: Model KV quantization (FP8, INT8) memory savings(very cool way!)

### Multi-GPU Enhancements
- [ ] **Tensor parallelism**: Coordinated multi-GPU inference

### Workload Modeling
- [ ] **Realistic arrival patterns**: Poisson, bursty, diurnal
- [ ] **Token length distributions**: Fit to production traces

---

## References & Inspiration

- **vLLM**: PagedAttention and continuous batching ([paper](https://arxiv.org/abs/2309.06180))
- **Orca**: Iteration-level scheduling ([paper](https://www.usenix.org/conference/osdi22/presentation/yu))
- **Power of Two Choices**: Mitzenmacher's seminal work on randomized load balancing
- **Floyd-Warshall**: All-pairs shortest paths for topology precomputation
