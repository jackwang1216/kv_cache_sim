## 2025-12-28 — C++ simulator skeleton

**What we added/changed**
- `cpp/CMakeLists.txt`: C++17 build, warnings on, builds `kv_sim` from `src/sim_main.cpp` with headers in `include/`.
- `cpp/include/types.hpp`: Core PODs
  - `RequestState`, `Request` (arrival_ms, prompt_tokens, gen_tokens, streaming, lifecycle timestamps).
  - `GPUConfig` (defaults: 24GB VRAM, `max_concurrent` 16, `prefill_tps` 1000, `decode_tps` 500).
  - `PolicyConfig` (safe decode-reservation flag, `max_queue`, `kv_bytes_per_token` default 2048 bytes).
  - `SimConfig` bundles GPU/policy, timeseries_dt placeholder, seed.
- `cpp/include/events.hpp`: Discrete-event defs (`EventType` for Arrival/StartPrefill/StartDecode/Finish/Reject/Evict, `Event` struct with time_ms + request_index, heap comparator).
- `cpp/include/rng.hpp`: Seeded RNG wrapper (mt19937, uniform01) for deterministic runs.
- `cpp/include/simulator.hpp`: Simulator interface + internal helpers (event loop, KV accounting, admission checks, prefill/decode duration helpers).
- `cpp/src/sim_main.cpp`: Minimal simulator implementation + demo `main`
  - Discrete-event loop using `std::priority_queue`.
  - Admission: prompt-only vs prompt+decode reservation (default safe reservation).
  - KV accounting with `kv_bytes_per_token`.
  - Prefill/Decode phase scheduling; frees KV on finish.
  - Demo config: 8GB VRAM, `max_concurrent` 2, `prefill_tps` 1000, `decode_tps` 500, `kv_bytes_per_token` 2048; two sample requests to validate flow.

**What we accomplished (behavior)**
- End-to-end event flow: arrival → prefill → decode → finish with VRAM checks and KV alloc/free.
- Deterministic ordering via event heap; simple completion summary at end.

**Next steps (short horizon)**
- Add a proper FIFO queue respecting `max_queue` and `max_concurrent`; start new prefill when capacity frees up.
- Implement timeseries sampling (every `timeseries_dt_ms`) for vram_used, active_prefill/decode, queue_depth, tokens_generated_delta, rejects_delta.
- Emit run artifacts: `summary.json`, `timeseries.csv`, `events.jsonl`, `run_meta.json` (buffered IO).
- Add CLI parsing (config path, trace path, out dir, seed) and load requests from a trace file.
- Add simple scheduling switch (FIFO vs shortest-remaining) and keep safe admission toggle.

**Later steps (broader)**
- Memory pressure behaviors (reject-on-pressure for V1; eviction later).
- Batching/shared throughput model (divide throughput by active decodes, cap by batch tokens).
- Validation scenarios: low-load sanity, saturation curve, VRAM monotonicity, long-context tail.
- Integrate with Python runner for sweeps/plots once binary outputs are stable.

## 2025-12-30 — IO split, sample inputs, runnable demo

**What we added/changed**
- Split simulator logic into `src/simulator.cpp`; slim `sim_main.cpp` now wires config/trace IO and runs the engine.
- Added IO helpers: `io_config` (load engine settings), `io_trace` (load replay trace), `io_output` (write minimal summary).
- Added getter `Simulator::requests()` for outputs.
- Fixed CMake target name to `kv_sim` and included all sources.
- Created sample inputs:
  - `configs/example_basic.txt`: 8 GiB VRAM, concurrency 2, prefill/decode tps 1200/600, kv_bytes_per_token 2048, safe reservation on, max_queue 1024, timeseries_dt_ms 20.
  - `data/trace_demo.txt`: three requests (req1/req2/req3) with arrivals and prompt/gen tokens, non-streaming.
- Ran demo build + run; produced `runs/demo_basic/summary.json` with finished=2, rejected=0.

**Next steps (near-term)**
- Expand outputs: `summary.json` with latency/throughput/TTFT/VRAM stats; `timeseries.csv`; `events.jsonl`; `run_meta.json`.
- Add queue correctness: ensure arrivals allocate KV and start immediately when capacity exists; continue to respect `max_queue`.
- Add scheduling toggle (FIFO vs shortest-remaining) and finalize admission behavior.
- Add timeseries sampling hook in the event loop at `timeseries_dt_ms`.
- Wire CLI flags for config/trace/out/seed cleanly (current positional arguments are minimal).

**Later steps**
- Memory-pressure behaviors (reject-on-pressure now; evict later), batching/shared throughput model, validation scenarios, Python sweep/plots integration, web viewer if desired.
