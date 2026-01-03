#include <iostream>
#include <algorithm>
#include <random>
#include <limits>
#include "simulator.hpp"

Simulator::Simulator(SimConfig cfg, std::vector<Request> requests)
    : cfg_(std::move(cfg)),
      requests_(std::move(requests)),
      next_sample_ms_(cfg_.timeseries_dt_ms),
      rng_(cfg_.seed) {
        if (cfg_.gpus.size() == 0){
            cfg_.gpus.push_back(GPUConfig{});
        }
        gpus_.resize(cfg_.gpus.size());
        precompute_topology();
        for (auto& gpu : gpus_) {
            gpu.vram_used = 0;
            gpu.active_prefill = 0;
            gpu.active_decode = 0;
            gpu.prefill_queue.clear();
            gpu.evict_queue.clear();
            gpu.lru_list.clear();
            gpu.allocated_bytes.assign(requests_.size(), 0);
            gpu.lru_iters.assign(requests_.size(), gpu.lru_list.end());
        }
      }

void Simulator::run() {
    schedule_arrivals();
    sample_until(0.0);

    while (!pq_.empty()) {
        Event event = pq_.top();
        pq_.pop();
        now_ms_ = event.time_ms;
        handle_event(event);
        sample_until(now_ms_);
    }

    int finished{0}, rejected{0}, evicted{0};
    for (const auto& req : requests_) {
        if (req.state == RequestState::Finished) finished++;
        if (req.state == RequestState::Rejected) rejected++;
        if (req.state == RequestState::Evicted) evicted++;
    }
    std::cout << "Finished: " << finished
              << ", Rejected: " << rejected
              << ", Evicted: " << evicted << '\n';
    sim_end_ms_ = now_ms_;
}

void Simulator::precompute_topology() {
    int num_gpus = static_cast<int>(gpus_.size());
    const double INF = std::numeric_limits<double>::infinity();
    double default_bw = cfg_.policy.handoff_bandwidth_gbps;
    double default_lat = cfg_.policy.handoff_latency_us / 1000.0;  // Convert to ms

    // Initialize matrices: diagonal = same GPU, off-diagonal = default link
    cfg_.bandwidth_matrix.assign(num_gpus, std::vector<double>(num_gpus, default_bw));
    cfg_.latency_matrix.assign(num_gpus, std::vector<double>(num_gpus, default_lat));
    for (int i = 0; i < num_gpus; i++) {
        cfg_.latency_matrix[i][i] = 0.0;
        cfg_.bandwidth_matrix[i][i] = INF; 
    }

    // Apply explicit link definitions (override defaults)
    for (const auto& link : cfg_.raw_links) {
        int src = link.src;
        int dest = link.dest;
        if (src < 0 || dest < 0 || src >= num_gpus || dest >= num_gpus) continue;
        // Take better values (lower latency, higher bandwidth)
        cfg_.latency_matrix[src][dest] = std::min(cfg_.latency_matrix[src][dest], link.latency_ms);
        cfg_.latency_matrix[dest][src] = std::min(cfg_.latency_matrix[dest][src], link.latency_ms);
        cfg_.bandwidth_matrix[src][dest] = std::max(cfg_.bandwidth_matrix[src][dest], link.bandwidth_gbps);
        cfg_.bandwidth_matrix[dest][src] = std::max(cfg_.bandwidth_matrix[dest][src], link.bandwidth_gbps);
    }

    // Floyd-Warshall
    for (int k = 0; k < num_gpus; k++) {
        for (int i = 0; i < num_gpus; i++) {
            if (cfg_.bandwidth_matrix[i][k] == 0 || cfg_.bandwidth_matrix[i][k] == INF) continue;
            for (int j = 0; j < num_gpus; j++) {
                if (cfg_.bandwidth_matrix[k][j] == 0 || cfg_.bandwidth_matrix[k][j] == INF) continue;
                double hop_latency = cfg_.latency_matrix[i][k] + cfg_.latency_matrix[k][j];
                double hop_bandwidth = 1.0 / (1.0 / cfg_.bandwidth_matrix[i][k] + 1.0 / cfg_.bandwidth_matrix[k][j]);
                if (hop_bandwidth > cfg_.bandwidth_matrix[i][j]) {
                    cfg_.bandwidth_matrix[i][j] = hop_bandwidth;
                    cfg_.latency_matrix[i][j] = hop_latency;
                }
            }
        }
    }
}

//simple score function for now
double Simulator::score_gpu(int gpu_idx) const {
    auto& gpu = gpus_[gpu_idx];
    return gpu.active_prefill + gpu.active_decode + static_cast<double>(gpu.prefill_queue.size());
}

int Simulator::route_gpu_for_request(const Request& req) {
    (void)req;
    int n = static_cast<int>(gpus_.size());
    if (n == 1) return 0;

    if (cfg_.policy.routing_policy == RoutingPolicy::P2C) {
        auto sample_idx = [n, this]() {
            int idx = static_cast<int>(rng_.uniform01() * n);
            return (idx >= n) ? n - 1 : idx;
        };
        int a = sample_idx();
        int b = sample_idx();
        if (n > 2) {
            while (b == a) b = sample_idx();
        } else if (a == b) {
            b = 1 - a;
        }
        double score_a = score_gpu(a);
        double score_b = score_gpu(b);
        if (score_a < score_b) return a;
        if (score_b < score_a) return b;
        // Tie: pick randomly to avoid bias
        return (rng_.uniform01() < 0.5) ? a : b;
    } else if (cfg_.policy.routing_policy == RoutingPolicy::RoundRobin) {
        return 0;  // TODO: implement round-robin
    } else if (cfg_.policy.routing_policy == RoutingPolicy::LeastLoaded) {
        return 0;  // TODO: implement least-loaded
    }
    return 0;
}

int Simulator::route_decode(int prefill_gpu, const Request& req) {
    int n = static_cast<int>(gpus_.size());
    if (n == 1) return prefill_gpu;

    double best_score = std::numeric_limits<double>::infinity();
    int best_gpu = -1;
    for (int gpu_idx = 0; gpu_idx < n; ++gpu_idx) {
        if (!can_fit_kv(gpu_idx, req)) continue;
        double score = compute_decode_score(prefill_gpu, gpu_idx, req);
        if (score < best_score) {
            best_score = score;
            best_gpu = gpu_idx;
        }
    }
    if (best_gpu == -1) return prefill_gpu;
    return best_gpu;
}

bool Simulator::can_fit_kv(int gpu_idx, const Request& req) const {
    const auto& gpu = gpus_[gpu_idx];
    const auto& gpu_cfg = cfg_.gpus[gpu_idx];
    std::uint64_t need = static_cast<std::uint64_t>(req.prompt_tokens + req.gen_tokens) * cfg_.policy.kv_bytes_per_token;
    return gpu.vram_used + need <= gpu_cfg.vram_bytes;
}

double Simulator::get_link_bandwidth(int src_idx, int dest_idx) const {
    if (src_idx == dest_idx) return std::numeric_limits<double>::infinity();
    return cfg_.bandwidth_matrix[src_idx][dest_idx];
}

double Simulator::get_link_latency(int src_idx, int dest_idx) const {
    if (src_idx == dest_idx) return 0.0;
    return cfg_.latency_matrix[src_idx][dest_idx];
}

double Simulator::estimate_handoff_ms(int src_idx, int dest_idx, const Request& req) const {
    if (src_idx == dest_idx) return 0.0;
    double bandwidth_gbps = get_link_bandwidth(src_idx, dest_idx);
    double latency_ms = get_link_latency(src_idx, dest_idx);
    double bytes = static_cast<double>(req.prompt_tokens + req.gen_tokens) * cfg_.policy.kv_bytes_per_token;
    double transfer_ms = bytes / (bandwidth_gbps * 1e6);
    return latency_ms + transfer_ms;
}

double Simulator::compute_decode_score(int src_idx, int dest_idx, const Request& req) const {
    return score_gpu(dest_idx) + cfg_.policy.handoff_cost_weight * estimate_handoff_ms(src_idx, dest_idx, req);
}

void Simulator::schedule_arrivals() {
    for (int i = 0; i < static_cast<int>(requests_.size()); ++i) {
        pq_.push(Event{requests_[i].arrival_time_ms, EventType::Arrival, i, -1});
    }
}

void Simulator::handle_event(const Event& event) {
    switch (event.type) {
        case EventType::Arrival:        on_arrival(event); break;
        case EventType::StartPrefill:   on_start_prefill(event); break;
        case EventType::StartDecode:    on_start_decode(event); break;
        case EventType::HandoffStart:   on_handoff_start(event); break;
        case EventType::HandoffComplete: on_handoff_complete(event); break;
        case EventType::Finish:         on_finish(event); break;
        default: break;
    }
}

bool Simulator::can_admit_prompt(int prompt_tokens, int gpu_idx) const {
    auto& gpu = gpus_[gpu_idx];
    auto& gpu_cfg = cfg_.gpus[gpu_idx];
    std::uint64_t need = static_cast<std::uint64_t>(prompt_tokens) * cfg_.policy.kv_bytes_per_token;
    return gpu.vram_used + need <= gpu_cfg.vram_bytes;
}

bool Simulator::can_reserve_decode(int prompt_tokens, int gen_tokens, int gpu_idx) const {
    auto& gpu = gpus_[gpu_idx];
    auto& gpu_cfg = cfg_.gpus[gpu_idx];
    std::uint64_t need = static_cast<std::uint64_t>(prompt_tokens + gen_tokens) * cfg_.policy.kv_bytes_per_token;
    return gpu.vram_used + need <= gpu_cfg.vram_bytes;
}

void Simulator::allocate_kv_bytes(int req_idx, std::uint64_t bytes, int gpu_idx) {
    auto& gpu = gpus_[gpu_idx];
    gpu.vram_used += bytes;
    gpu.allocated_bytes[req_idx] += bytes;
}

void Simulator::free_kv_bytes(int req_idx, std::uint64_t bytes, int gpu_idx) {
    auto& gpu = gpus_[gpu_idx];
    std::uint64_t to_free = std::min(bytes, gpu.allocated_bytes[req_idx]);
    if (to_free > gpu.vram_used) {
        gpu.vram_used = 0;
    } else {
        gpu.vram_used -= to_free;
    }
    gpu.allocated_bytes[req_idx] -= to_free;
}

double Simulator::prefill_duration_ms(int prompt_tokens, int gpu_idx) const {
    return 1000.0 * prompt_tokens / cfg_.gpus[gpu_idx].prefill_tps;
}

double Simulator::decode_duration_ms(int gen_tokens, int active_decode, int gpu_idx) const {
    int share = std::max(1, std::min(active_decode, cfg_.gpus[gpu_idx].decode_sharing_cap));
    double eff = cfg_.gpus[gpu_idx].decode_efficiency;
    double effective_tps = cfg_.gpus[gpu_idx].decode_tps * eff / static_cast<double>(share);
    if (effective_tps <= 0.0) return 0.0;
    return 1000.0 * gen_tokens / effective_tps;
}

void Simulator::on_arrival(const Event& event) {
    auto& req = requests_[event.request_index];
    if (req.state == RequestState::Evicted || req.state == RequestState::Rejected || req.state == RequestState::Finished) {
        return;
    }
    // Route at arrival time (when actual GPU state is known)
    int gpu_idx = route_gpu_for_request(req);
    auto& gpu = gpus_[gpu_idx];
    int queued = static_cast<int>(gpu.prefill_queue.size());
    int active = gpu.active_prefill + gpu.active_decode;
    if (queued + active >= cfg_.policy.max_queue) {
        req.state = RequestState::Rejected;
        rejects_total_++;
        record_event(EventType::Reject, req, gpu_idx);
        return;
    }

    // Reserve KV: prompt always; decode tokens only if safe_reservation is enabled.
    int reserved_tokens = req.prompt_tokens + (cfg_.policy.safe_reservation ? req.gen_tokens : 0);
    std::uint64_t need = static_cast<std::uint64_t>(reserved_tokens) * cfg_.policy.kv_bytes_per_token;
    if (!ensure_capacity_for(need, gpu_idx)) {
        req.state = RequestState::Rejected;
        rejects_total_++;
        record_event(EventType::Reject, req, gpu_idx);
        return;
    }
    allocate_kv_bytes(event.request_index, need, gpu_idx);
    req.state = RequestState::Queued;
    record_event(EventType::Arrival, req, gpu_idx);

    // Track for eviction policies
    gpu.evict_queue.push_back(event.request_index);
    touch_lru(event.request_index, gpu_idx);

    if (gpu.active_prefill + gpu.active_decode < cfg_.gpus[gpu_idx].max_concurrent) {
        gpu.active_prefill++;  // Increment now to prevent concurrent arrivals from over-scheduling
        pq_.push(Event{now_ms_, EventType::StartPrefill, event.request_index, gpu_idx});
    } else {
        gpu.prefill_queue.push_back(event.request_index);
    }
}

int Simulator::pick_next_from_queue(int gpu_idx) {
    auto& gpu = gpus_[gpu_idx];
    if (gpu.prefill_queue.empty()) return -1;
    if (cfg_.policy.scheduling == SchedulingMode::FIFO) {
        int idx = gpu.prefill_queue.front();
        gpu.prefill_queue.pop_front();
        return idx;
    }
    // Shortest remaining
    auto best_it = gpu.prefill_queue.begin();
    int best_tokens = requests_[*best_it].prompt_tokens + requests_[*best_it].gen_tokens;
    for (auto it = std::next(gpu.prefill_queue.begin()); it != gpu.prefill_queue.end(); ++it) {
        int tokens = requests_[*it].prompt_tokens + requests_[*it].gen_tokens;
        if (tokens < best_tokens) {
            best_it = it;
            best_tokens = tokens;
        }
    }
    int idx = *best_it;
    gpu.prefill_queue.erase(best_it);
    return idx;
}

void Simulator::try_start_prefill(int gpu_idx) {
    auto& gpu = gpus_[gpu_idx];
    while (!gpu.prefill_queue.empty() && gpu.active_prefill + gpu.active_decode < cfg_.gpus[gpu_idx].max_concurrent) {
        int req_idx = pick_next_from_queue(gpu_idx);
        if (req_idx < 0) break;
        gpu.active_prefill++;  // Increment now to prevent over-scheduling
        pq_.push(Event{now_ms_, EventType::StartPrefill, req_idx, gpu_idx});
    }
}

void Simulator::on_start_prefill(const Event& event) {
    int gpu_idx = event.gpu_index;
    auto& gpu = gpus_[gpu_idx];
    auto& req = requests_[event.request_index];
    if (req.state == RequestState::Evicted || req.state == RequestState::Rejected || req.state == RequestState::Finished) {
        gpu.active_prefill--;
        try_start_prefill(gpu_idx);
        return;
    }
    req.state = RequestState::Prefill;
    req.start_prefill_ms = now_ms_;
    req.prefill_gpu = gpu_idx;
    touch_lru(event.request_index, gpu_idx);
    record_event(EventType::StartPrefill, req, gpu_idx);
    double duration = prefill_duration_ms(req.prompt_tokens, gpu_idx);
    pq_.push(Event{now_ms_ + duration, EventType::StartDecode, event.request_index, gpu_idx});
}

void Simulator::on_start_decode(const Event& event) {
    int gpu_idx = event.gpu_index;
    auto& gpu = gpus_[gpu_idx];
    auto& req = requests_[event.request_index];
    if (req.state == RequestState::Evicted || req.state == RequestState::Rejected || req.state == RequestState::Finished) {
        return;
    }
    gpu.active_prefill--;
    int decode_gpu_idx = route_decode(gpu_idx, req);
    req.decode_gpu = decode_gpu_idx;
    if (decode_gpu_idx != gpu_idx) {
        // Schedule handoff to different GPU; convert latency from us to ms
        pq_.push(Event{now_ms_ + cfg_.policy.handoff_latency_us / 1000.0, EventType::HandoffStart, event.request_index, decode_gpu_idx});
        try_start_prefill(gpu_idx);
        return;
    }
    req.state = RequestState::Decode;
    req.start_decode_ms = now_ms_;
    gpu.active_decode++;

    if (!cfg_.policy.safe_reservation) {
        std::uint64_t need = static_cast<std::uint64_t>(req.gen_tokens) * cfg_.policy.kv_bytes_per_token;
        if (!ensure_capacity_for(need, gpu_idx)) {
            req.state = RequestState::Rejected;
            rejects_total_++;
            gpu.active_decode--;
            record_event(EventType::Reject, req, gpu_idx);
            free_kv_bytes(event.request_index, static_cast<std::uint64_t>(req.prompt_tokens) * cfg_.policy.kv_bytes_per_token, gpu_idx);
            try_start_prefill(gpu_idx);
            return;
        }
        allocate_kv_bytes(event.request_index, need, gpu_idx);
    }
    touch_lru(event.request_index, gpu_idx);
    record_event(EventType::StartDecode, req, gpu_idx);
    double duration = decode_duration_ms(req.gen_tokens, gpu.active_decode, gpu_idx);
    pq_.push(Event{now_ms_ + duration, EventType::Finish, event.request_index, gpu_idx});
}

void Simulator::on_handoff_start(const Event& event) {
    int dest_gpu_idx = event.gpu_index;
    auto& req = requests_[event.request_index];
    int src_gpu_idx = req.prefill_gpu;
    auto& src_gpu = gpus_[src_gpu_idx];

    std::uint64_t bytes_to_copy = src_gpu.allocated_bytes[event.request_index];

    if (!ensure_capacity_for(bytes_to_copy, dest_gpu_idx)) {
        req.state = RequestState::Rejected;
        rejects_total_++;
        record_event(EventType::Reject, req, dest_gpu_idx);
        free_kv_bytes(event.request_index, bytes_to_copy, src_gpu_idx);
        return;
    }

    allocate_kv_bytes(event.request_index, bytes_to_copy, dest_gpu_idx);
    double transfer_ms = estimate_handoff_ms(src_gpu_idx, dest_gpu_idx, req);
    record_event(EventType::HandoffStart, req, dest_gpu_idx);
    pq_.push(Event{now_ms_ + transfer_ms, EventType::HandoffComplete, event.request_index, dest_gpu_idx});
}

void Simulator::on_handoff_complete(const Event& event) {
    int dest_gpu_idx = event.gpu_index;
    int req_idx = event.request_index;
    auto& req = requests_[req_idx];
    if (req.state == RequestState::Evicted || req.state == RequestState::Rejected || req.state == RequestState::Finished) {
        return;
    }
    int src_gpu_idx = req.prefill_gpu;
    auto& src_gpu = gpus_[src_gpu_idx];
    auto& dest_gpu = gpus_[dest_gpu_idx];

    // Free KV from source GPU (handoff complete)
    free_kv_bytes(req_idx, src_gpu.allocated_bytes[req_idx], src_gpu_idx);
    record_event(EventType::HandoffComplete, req, dest_gpu_idx);

    // If safe_reservation=false, need to allocate decode bytes on dest GPU
    if (!cfg_.policy.safe_reservation) {
        std::uint64_t need = static_cast<std::uint64_t>(req.gen_tokens) * cfg_.policy.kv_bytes_per_token;
        if (!ensure_capacity_for(need, dest_gpu_idx)) {
            req.state = RequestState::Rejected;
            rejects_total_++;
            record_event(EventType::Reject, req, dest_gpu_idx);
            free_kv_bytes(req_idx, dest_gpu.allocated_bytes[req_idx], dest_gpu_idx);
            return;
        }
        allocate_kv_bytes(req_idx, need, dest_gpu_idx);
    }

    // Directly transition to decode on destination GPU (don't re-enter routing)
    req.state = RequestState::Decode;
    req.start_decode_ms = now_ms_;
    dest_gpu.active_decode++;

    touch_lru(req_idx, dest_gpu_idx);
    record_event(EventType::StartDecode, req, dest_gpu_idx);
    double duration = decode_duration_ms(req.gen_tokens, dest_gpu.active_decode, dest_gpu_idx);
    pq_.push(Event{now_ms_ + duration, EventType::Finish, req_idx, dest_gpu_idx});
}

void Simulator::on_finish(const Event& event) {
    int gpu_idx = event.gpu_index;
    auto& gpu = gpus_[gpu_idx];
    auto& req = requests_[event.request_index];
    if (req.state == RequestState::Evicted || req.state == RequestState::Rejected || req.state == RequestState::Finished) {
        return;
    }
    gpu.active_decode--;
    req.state = RequestState::Finished;
    req.finish_ms = now_ms_;
    tokens_generated_total_ += static_cast<std::uint64_t>(req.gen_tokens);
    record_event(EventType::Finish, req, gpu_idx);
    free_kv_bytes(event.request_index, gpu.allocated_bytes[event.request_index], gpu_idx);

    // Clean eviction tracking (lazy remove)
    if (cfg_.policy.eviction_policy == EvictionPolicy::LRU) {
        if (gpu.lru_iters[event.request_index] != gpu.lru_list.end()) {
            gpu.lru_list.erase(gpu.lru_iters[event.request_index]);
            gpu.lru_iters[event.request_index] = gpu.lru_list.end();
        }
    }
    // For FIFO, skip stale victims during eviction; optional eager cleanup:
    // remove all occurrences of this index
    gpu.evict_queue.erase(
        std::remove(gpu.evict_queue.begin(), gpu.evict_queue.end(), event.request_index),
        gpu.evict_queue.end());

    try_start_prefill(gpu_idx);
}

void Simulator::record_event(EventType type, const Request& req, int gpu_idx) {
    events_.push_back(EventRecord{now_ms_, type, req.id, gpu_idx});
}

void Simulator::sample_until(double target_time_ms) {
    while (next_sample_ms_ <= target_time_ms) {
        TimeseriesSample s;
        s.time_ms = next_sample_ms_;
        for (const auto& gpu : gpus_) {
            s.vram_used += gpu.vram_used;
            s.active_prefill += gpu.active_prefill;
            s.active_decode += gpu.active_decode;
            s.queue_depth += static_cast<int>(gpu.prefill_queue.size());
        }
        s.tokens_generated_delta = tokens_generated_total_ - last_tokens_sampled_;
        s.rejects_delta = rejects_total_ - last_rejects_sampled_;
        samples_.push_back(s);
        last_tokens_sampled_ = tokens_generated_total_;
        last_rejects_sampled_ = rejects_total_;
        next_sample_ms_ += cfg_.timeseries_dt_ms;
    }
    // Ensure we capture the tail interval up to target_time_ms (even if it is not on the sampling grid).
    if (samples_.empty() || samples_.back().time_ms < target_time_ms) {
        TimeseriesSample s;
        s.time_ms = target_time_ms;
        for (const auto& gpu : gpus_) {
            s.vram_used += gpu.vram_used;
            s.active_prefill += gpu.active_prefill;
            s.active_decode += gpu.active_decode;
            s.queue_depth += static_cast<int>(gpu.prefill_queue.size());
        }
        s.tokens_generated_delta = tokens_generated_total_ - last_tokens_sampled_;
        s.rejects_delta = rejects_total_ - last_rejects_sampled_;
        samples_.push_back(s);
        last_tokens_sampled_ = tokens_generated_total_;
        last_rejects_sampled_ = rejects_total_;
    }
}

bool Simulator::ensure_capacity_for(std::uint64_t bytes_needed, int gpu_idx) {
    auto& gpu = gpus_[gpu_idx];
    if (gpu.vram_used + bytes_needed <= cfg_.gpus[gpu_idx].vram_bytes) return true;
    if (cfg_.policy.memory_pressure_policy == MemoryPressurePolicy::Reject) return false;

    //evict until fits or no victims 
    while (gpu.vram_used + bytes_needed > cfg_.gpus[gpu_idx].vram_bytes) {
        if (!evict_one(gpu_idx)) return false;
    }
    return true;
}

bool Simulator::evict_one(int gpu_idx) {
    auto& gpu = gpus_[gpu_idx];
    int victim = -1;
    if (cfg_.policy.eviction_policy == EvictionPolicy::FIFO) {
        while (!gpu.evict_queue.empty()) {
            int cand = gpu.evict_queue.front();
            const auto& req = requests_[cand];
            if (req.state == RequestState::Rejected || req.state == RequestState::Evicted || req.state == RequestState::Finished) {
                gpu.evict_queue.pop_front();
                continue;
            }
            victim = cand;
            gpu.evict_queue.pop_front();
            break;
        }
        if (victim == -1) return false;
    } else { //LRU
        if (gpu.lru_list.empty()) return false;
        victim = gpu.lru_list.back();
        gpu.lru_list.pop_back();
        gpu.lru_iters[victim] = gpu.lru_list.end();
    }
    auto& req = requests_[victim];
    if (req.state == RequestState::Rejected || req.state == RequestState::Evicted || req.state == RequestState::Finished) {
        return false; // skip invalid victim
    }
    // Adjust active counters and queue bookkeeping
    if (req.state == RequestState::Prefill) {
        if (gpu.active_prefill > 0) gpu.active_prefill--;
    } else if (req.state == RequestState::Decode) {
        if (gpu.active_decode > 0) gpu.active_decode--;
    } else if (req.state == RequestState::Queued) {
        // remove from prefill_queue_ if present
        gpu.prefill_queue.erase(
            std::remove(gpu.prefill_queue.begin(), gpu.prefill_queue.end(), victim),
            gpu.prefill_queue.end());
    }
    free_kv_bytes(victim, gpu.allocated_bytes[victim], gpu_idx);
    req.state = RequestState::Evicted;
    record_event(EventType::Evict, req, gpu_idx);
    // After freeing, try to start more work
    try_start_prefill(gpu_idx);
    return true;
}

void Simulator::touch_lru(int req_idx, int gpu_idx) {
    auto& gpu = gpus_[gpu_idx];
    if (cfg_.policy.eviction_policy != EvictionPolicy::LRU) return;
    if (gpu.lru_iters[req_idx] != gpu.lru_list.end()) {
        gpu.lru_list.erase(gpu.lru_iters[req_idx]);
    }
    gpu.lru_list.push_front(req_idx);
    gpu.lru_iters[req_idx] = gpu.lru_list.begin();
}