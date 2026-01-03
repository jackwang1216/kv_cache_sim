#pragma once
#include <cstdint>
#include <string>
#include <deque>
#include <list>
#include <vector>
#include "events.hpp"

enum class RequestState {
    Arrived, 
    Queued, 
    Prefill, 
    Decode, 
    Finished, 
    Rejected, 
    Evicted
};

enum class SchedulingMode {
    FIFO,
    ShortestRemaining
};

enum class MemoryPressurePolicy {
    Reject,
    Evict
};

enum class EvictionPolicy {
    FIFO,
    LRU
};

enum class RoutingPolicy {
    P2C,
    RoundRobin,
    LeastLoaded
};

struct EventRecord {
    double time_ms = 0.0;
    EventType type = EventType::Arrival;
    std::string request_id;
    int gpu_index = 0;
};

struct TimeseriesSample {
    double time_ms = 0.0;
    std::uint64_t vram_used = 0;
    int active_prefill = 0;
    int active_decode = 0;
    int queue_depth = 0;
    std::uint64_t tokens_generated_delta = 0;
    int rejects_delta = 0;
};

struct Request {
    std::string id;
    double arrival_time_ms = 0.0;
    int prompt_tokens = 0;
    int gen_tokens = 0;
    bool streaming = false;

    RequestState state = RequestState::Arrived;
    double start_prefill_ms = 0.0;
    double start_decode_ms = 0.0;
    double finish_ms = 0.0;

    int prefill_gpu = 0;
    int decode_gpu = 0;
};

struct GPUConfig {
    std::uint64_t vram_bytes = 24ull * 1024ull * 1024ull * 1024ull;
    int max_concurrent = 16;
    double prefill_tps = 1000.0;
    double decode_tps = 500.0;
    int decode_sharing_cap = 8;
    double decode_efficiency = 0.8;
};

struct GPUState {
    std::uint64_t vram_used = 0;
    int active_prefill = 0;
    int active_decode = 0;
    std::deque<int> prefill_queue;
    std::deque<int> evict_queue;
    std::list<int> lru_list;
    std::vector<std::list<int>::iterator> lru_iters;
    std::vector<std::uint64_t> allocated_bytes;
};

struct PolicyConfig {
    bool safe_reservation = true;
    int max_queue = 1024;
    std::uint64_t kv_bytes_per_token = 2048;
    double handoff_latency_us = 10.0;       // Fixed latency overhead in microseconds
    double handoff_bandwidth_gbps = 300.0;  // Default NVLink ~300 GB/s, PCIe 4.0 ~25 GB/s
    double handoff_cost_weight = 0.5;
    SchedulingMode scheduling = SchedulingMode::FIFO;
    MemoryPressurePolicy memory_pressure_policy = MemoryPressurePolicy::Reject;
    EvictionPolicy eviction_policy = EvictionPolicy::FIFO;
    RoutingPolicy routing_policy = RoutingPolicy::P2C;

    std::uint64_t vram_bytes = 24ull * 1024ull * 1024ull * 1024ull;
    double prefill_tps = 1000.0;
    double decode_tps = 500.0;
};

struct RawLink {
    int src = 0;
    int dest = 0;
    double bandwidth_gbps = 0.0;
    double latency_ms = 0.0;
};

struct SimConfig {
    std::vector<GPUConfig> gpus;
    std::vector<std::vector<double>> latency_matrix;
    std::vector<std::vector<double>> bandwidth_matrix;
    std::vector<RawLink> raw_links;
    PolicyConfig policy;
    double timeseries_dt_ms = 20.0;
    unsigned int seed = 12345;
};