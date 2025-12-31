# KV Cache Simulator — LLM Inference Memory & Scheduling

A systems-focused simulator for **KV cache behavior in large language model (LLM) inference**, designed to study how **memory pressure, scheduling, and admission policies** affect throughput and tail latency (p50/p90/p99).

This project models the *system-level tradeoffs* behind modern LLM serving engines (e.g. vLLM, TGI), rather than implementing low-level CUDA kernels.

---

## Motivation

Autoregressive LLM inference is increasingly **memory-bound**, not compute-bound.

While model weights are fixed, **KV cache memory grows with context length, concurrency, and generated tokens**, making it the primary bottleneck for:
- throughput
- tail latency
- admission control
- fairness across requests

This simulator explores how different **KV cache designs and policies** impact real-world serving behavior under realistic workloads.

---

## Key Concepts Modeled

- **Prefill vs Decode**
  - Prefill: prompt processing (compute-heavy, batched)
  - Decode: token-by-token generation (memory-bandwidth & KV-bound)

- **KV Cache Growth**
  - Per-token KV memory accumulation
  - VRAM pressure under concurrency

- **Scheduling Policies**
  - FIFO
  - Shortest-job / remaining-tokens first
  - Decode-priority vs prefill-priority

- **Admission Control**
  - Reject-on-OOM
  - Token-budget-based admission
  - Concurrency caps

- **KV Cache Layouts**
  - Contiguous per-sequence allocation
  - Blocked / paged KV cache (fixed-size token blocks)

---

## Simulator Model

### Inputs
- Request arrival times
- Prompt token lengths
- Output token lengths (or distributions)
- Model configuration (layers, heads, KV bytes/token)
- GPU configuration (VRAM size, bandwidth)
- Policy configuration

### Core Simulation Loop
1. Admit or reject incoming requests
2. Allocate KV cache blocks
3. Schedule prefill / decode steps
4. Track KV growth and VRAM occupancy
5. Record latency and throughput metrics

### Outputs
- Time-to-first-token (TTFT)
- End-to-end latency
- p50 / p90 / p99 latency
- Throughput (tokens/sec)
- KV cache utilization over time
- Rejection / eviction statistics

---

## Metrics Reported

- **Latency**
  - p50 / p90 / p99
  - TTFT vs total latency

- **Throughput**
  - Prefill throughput
  - Decode throughput

- **Memory**
  - KV cache occupancy
  - VRAM fragmentation (paged vs contiguous)
  - Peak memory usage

- **System Health**
  - Queue lengths
  - Rejection rate
  - GPU utilization proxy

---

## Example Experiments

- FIFO vs shortest-first scheduling under mixed workloads
- Impact of long-context requests on p99 latency
- Block size tradeoffs in paged KV cache
- Decode-priority vs prefill-priority scheduling
- Admission control thresholds vs throughput

---

## Design Philosophy

- **System-level realism over kernel-level detail**
- Explicit modeling of tradeoffs instead of black-box benchmarking
- Focus on tail behavior (p99), not just averages
- Extensible policy layer for rapid experimentation

This project intentionally **does not implement actual KV tensors or CUDA kernels**.  
Instead, it models their *performance and memory consequences*, which is the relevant layer for scheduling and capacity planning.

---

## Future Work

- Prefix KV reuse / prompt caching
- CPU offload and swap latency modeling
- Multi-GPU routing
- Fairness-aware scheduling
- Predictive token-length admission control

---

## References & Inspiration

- vLLM: PagedAttention
- FasterTransformer / TGI
- Production LLM serving systems
- Queueing theory & tail-latency analysis

---

## Quick Validation Scenarios (sanity checks)

Run from repo root (examples):

- Low load (no pressure, expect rejects/evicts = 0, latency ≈ service time):
  ```
  ./cpp/build/kv_sim configs/validation_low_load.txt data/validation_low_load.txt runs/val_low_load
  ```

- Saturation (tight VRAM + frequent arrivals, expect queue growth and evictions):
  ```
  ./cpp/build/kv_sim configs/validation_saturation.txt data/validation_saturation.txt runs/val_saturation
  ```

- VRAM monotonicity (same trace, low vs high VRAM; higher VRAM should not worsen rejects/evicts):
  ```
  ./cpp/build/kv_sim configs/validation_vram_low.txt data/validation_vram_base.txt runs/val_vram_low
  ./cpp/build/kv_sim configs/validation_vram_high.txt data/validation_vram_base.txt runs/val_vram_high
  ```

- Long-context tail (mix of shorts + one long; expect higher p95/p99 vs shorts-only baseline):
  ```
  ./cpp/build/kv_sim configs/validation_long_tail.txt data/validation_long_tail.txt runs/val_long_tail
  ```

Expected outcomes (qualitative):
- Low load: rejects=0, evictions=0, latency near (prompt/decode modeled times).
- Saturation: p95/p99 higher than low-load; evictions > 0 under evict policy; queue_depth grows.
- VRAM monotonicity: `val_vram_high` should have <= rejects/evicts vs `val_vram_low`.
- Long tail: tail latencies (p95/p99) worsen vs a short-only mix; evictions may rise if pressure hits.
