#include <iostream>
#include <algorithm>
#include "simulator.hpp"

Simulator::Simulator(SimConfig cfg, std::vector<Request> requests)
    : cfg_(cfg),
      requests_(std::move(requests)),
      lru_iters_(requests_.size(), lru_list_.end()),
      next_sample_ms_(cfg.timeseries_dt_ms) {}

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

    int finished{0}, rejected{0};
    for (const auto& req : requests_) {
        if (req.state == RequestState::Finished) finished++;
        if (req.state == RequestState::Rejected) rejected++;
    }
    std::cout << "Finished: " << finished << ", Rejected: " << rejected << '\n';
    sim_end_ms_ = now_ms_;
}

void Simulator::schedule_arrivals() {
    for (int i = 0; i < static_cast<int>(requests_.size()); ++i) {
        pq_.push(Event{requests_[i].arrival_time_ms, EventType::Arrival, i});
    }
}

void Simulator::handle_event(const Event& event) {
    switch (event.type) {
        case EventType::Arrival:      on_arrival(event); break;
        case EventType::StartPrefill: on_start_prefill(event); break;
        case EventType::StartDecode:  on_start_decode(event); break;
        case EventType::Finish:       on_finish(event); break;
        default: break;
    }
}

bool Simulator::can_admit_prompt(int prompt_tokens) const {
    std::uint64_t need = static_cast<std::uint64_t>(prompt_tokens) * cfg_.policy.kv_bytes_per_token;
    return vram_used_ + need <= cfg_.gpu.vram_bytes;
}

bool Simulator::can_reserve_decode(int prompt_tokens, int gen_tokens) const {
    std::uint64_t need = static_cast<std::uint64_t>(prompt_tokens + gen_tokens) * cfg_.policy.kv_bytes_per_token;
    return vram_used_ + need <= cfg_.gpu.vram_bytes;
}

void Simulator::allocate_kv(int tokens) {
    vram_used_ += static_cast<std::uint64_t>(tokens) * cfg_.policy.kv_bytes_per_token;
}

void Simulator::free_kv(int tokens) {
    vram_used_ -= static_cast<std::uint64_t>(tokens) * cfg_.policy.kv_bytes_per_token;
}

double Simulator::prefill_duration_ms(int prompt_tokens) const {
    return 1000.0 * prompt_tokens / cfg_.gpu.prefill_tps;
}

double Simulator::decode_duration_ms(int gen_tokens, int active_decode) const {
    int share = std::max(1, std::min(active_decode, cfg_.gpu.decode_sharing_cap));
    double eff = cfg_.gpu.decode_efficiency;
    double effective_tps = cfg_.gpu.decode_tps * eff / static_cast<double>(share);
    if (effective_tps <= 0.0) return 0.0;
    return 1000.0 * gen_tokens / effective_tps;
}

void Simulator::on_arrival(const Event& event) {
    auto& req = requests_[event.request_index];
    int queued = static_cast<int>(prefill_queue_.size());
    int active = active_prefill_ + active_decode_;
    if (queued + active >= cfg_.policy.max_queue) {
        req.state = RequestState::Rejected;
        rejects_total_++;
        record_event(EventType::Reject, req);
        return;
    }

    // Reserve KV: prompt always; decode tokens only if safe_reservation is enabled.
    int reserved_tokens = req.prompt_tokens + (cfg_.policy.safe_reservation ? req.gen_tokens : 0);
    std::uint64_t need = static_cast<std::uint64_t>(reserved_tokens) * cfg_.policy.kv_bytes_per_token;
    if (!ensure_capacity_for(need)) {
        req.state = RequestState::Rejected;
        rejects_total_++;
        record_event(EventType::Reject, req);
        return;
    }
    allocate_kv(reserved_tokens);
    req.state = RequestState::Queued;
    record_event(EventType::Arrival, req);

    // Track for eviction policies
    evict_queue_.push_back(event.request_index);
    touch_lru(event.request_index);

    if (active_prefill_ + active_decode_ < cfg_.gpu.max_concurrent) {
        pq_.push(Event{now_ms_, EventType::StartPrefill, event.request_index});
    } else {
        prefill_queue_.push_back(event.request_index);
    }
}

int Simulator::pick_next_from_queue() {
    if (prefill_queue_.empty()) return -1;
    if (cfg_.policy.scheduling == SchedulingMode::FIFO) {
        int idx = prefill_queue_.front();
        prefill_queue_.pop_front();
        return idx;
    }
    // Shortest remaining
    auto best_it = prefill_queue_.begin();
    int best_tokens = requests_[*best_it].prompt_tokens + requests_[*best_it].gen_tokens;
    for (auto it = std::next(prefill_queue_.begin()); it != prefill_queue_.end(); ++it) {
        int tokens = requests_[*it].prompt_tokens + requests_[*it].gen_tokens;
        if (tokens < best_tokens) {
            best_it = it;
            best_tokens = tokens;
        }
    }
    int idx = *best_it;
    prefill_queue_.erase(best_it);
    return idx;
}

void Simulator::try_start_prefill() {
    while (!prefill_queue_.empty() && active_prefill_ + active_decode_ < cfg_.gpu.max_concurrent) {
        int req_idx = pick_next_from_queue();
        if (req_idx < 0) break;
        pq_.push(Event{now_ms_, EventType::StartPrefill, req_idx});
    }
}

void Simulator::on_start_prefill(const Event& event) {
    auto& req = requests_[event.request_index];
    req.state = RequestState::Prefill;
    req.start_prefill_ms = now_ms_;
    active_prefill_++;
    touch_lru(event.request_index);
    record_event(EventType::StartPrefill, req);
    double duration = prefill_duration_ms(req.prompt_tokens);
    pq_.push(Event{now_ms_ + duration, EventType::StartDecode, event.request_index});
}

void Simulator::on_start_decode(const Event& event) {
    auto& req = requests_[event.request_index];
    active_prefill_--;
    req.state = RequestState::Decode;
    req.start_decode_ms = now_ms_;
    active_decode_++;
    if (!cfg_.policy.safe_reservation) {
        std::uint64_t need = static_cast<std::uint64_t>(req.gen_tokens) * cfg_.policy.kv_bytes_per_token;
        if (!ensure_capacity_for(need)) {
            req.state = RequestState::Rejected;
            rejects_total_++;
            record_event(EventType::Reject, req);
            free_kv(req.prompt_tokens);
            return;
        }
        allocate_kv(req.gen_tokens);
    }
    touch_lru(event.request_index);
    record_event(EventType::StartDecode, req);
    double duration = decode_duration_ms(req.gen_tokens, active_decode_);
    pq_.push(Event{now_ms_ + duration, EventType::Finish, event.request_index});
}

void Simulator::on_finish(const Event& event) {
    auto& req = requests_[event.request_index];
    active_decode_--;
    req.state = RequestState::Finished;
    req.finish_ms = now_ms_;
    tokens_generated_total_ += static_cast<std::uint64_t>(req.gen_tokens);
    record_event(EventType::Finish, req);
    free_kv(req.prompt_tokens + req.gen_tokens);

    // Clean eviction tracking (lazy remove)
    if (cfg_.policy.eviction_policy == EvictionPolicy::LRU) {
        if (lru_iters_[event.request_index] != lru_list_.end()) {
            lru_list_.erase(lru_iters_[event.request_index]);
            lru_iters_[event.request_index] = lru_list_.end();
        }
    }
    // For FIFO, skip stale victims during eviction; optional eager cleanup:
    // remove all occurrences of this index
    evict_queue_.erase(
        std::remove(evict_queue_.begin(), evict_queue_.end(), event.request_index),
        evict_queue_.end());

    try_start_prefill();
}

void Simulator::record_event(EventType type, const Request& req) {
    events_.push_back(EventRecord{now_ms_, type, req.id});
}

void Simulator::sample_until(double target_time_ms) {
    while (next_sample_ms_ <= target_time_ms) {
        TimeseriesSample s;
        s.time_ms = next_sample_ms_;
        s.vram_used = vram_used_;
        s.active_prefill = active_prefill_;
        s.active_decode = active_decode_;
        s.queue_depth = static_cast<int>(prefill_queue_.size());
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
        s.vram_used = vram_used_;
        s.active_prefill = active_prefill_;
        s.active_decode = active_decode_;
        s.queue_depth = static_cast<int>(prefill_queue_.size());
        s.tokens_generated_delta = tokens_generated_total_ - last_tokens_sampled_;
        s.rejects_delta = rejects_total_ - last_rejects_sampled_;
        samples_.push_back(s);
        last_tokens_sampled_ = tokens_generated_total_;
        last_rejects_sampled_ = rejects_total_;
    }
}

bool Simulator::ensure_capacity_for(std::uint64_t bytes_needed) {
    if (vram_used_ + bytes_needed <= cfg_.gpu.vram_bytes) return true;
    if (cfg_.policy.memory_pressure_policy == MemoryPressurePolicy::Reject) return false;

    //evict until fits or no victims 
    while (vram_used_ + bytes_needed > cfg_.gpu.vram_bytes) {
        if (!evict_one()) return false;
    }
    return true;
}

bool Simulator::evict_one() {
    int victim = -1;
    if (cfg_.policy.eviction_policy == EvictionPolicy::FIFO) {
        if (evict_queue_.empty()) return false;
        victim = evict_queue_.front();
        evict_queue_.pop_front();
    } else { //LRU
        if (lru_list_.empty()) return false;
        victim = lru_list_.back();
        lru_list_.pop_back();
        lru_iters_[victim] = lru_list_.end();
    }
    auto& req = requests_[victim];
    if (req.state == RequestState::Rejected || req.state == RequestState::Evicted || req.state == RequestState::Finished) {
        return false; // skip invalid victim
    }
    free_kv(req.prompt_tokens + req.gen_tokens);
    req.state = RequestState::Evicted;
    record_event(EventType::Evict, req);
    return true;
}

void Simulator::touch_lru(int req_idx) {
    if (cfg_.policy.eviction_policy != EvictionPolicy::LRU) return;
    if (lru_iters_[req_idx] != lru_list_.end()) {
        lru_list_.erase(lru_iters_[req_idx]);
    }
    lru_list_.push_front(req_idx);
    lru_iters_[req_idx] = lru_list_.begin();
}