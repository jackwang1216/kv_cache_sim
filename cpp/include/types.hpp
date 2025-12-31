#pragma once
#include <cstdint>
#include <string>
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

struct EventRecord {
    double time_ms = 0.0;
    EventType type = EventType::Arrival;
    std::string request_id;
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
};

struct GPUConfig {
    std::uint64_t vram_bytes = 24ull * 1024ull * 1024ull * 1024ull; // 24GB
    int max_concurrent = 16;
    double prefill_tps = 1000.0;
    double decode_tps = 500.0;
};

struct PolicyConfig {
    bool safe_reservation = true; 
    int max_queue = 1024;
    std::uint64_t kv_bytes_per_token = 2048;
    SchedulingMode scheduling = SchedulingMode::FIFO;
};

struct SimConfig {
    GPUConfig gpu;
    PolicyConfig policy;
    double timeseries_dt_ms = 20.0;
    unsigned int seed = 12345;
};