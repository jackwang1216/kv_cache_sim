#include <iostream>
#include <queue>
#include <vector>
#include <string>
#include "types.hpp"
#include "events.hpp"
#include "rng.hpp"
#include "simulator.hpp"

Simulator::Simulator(SimConfig cfg, std::vector<Request> requests)
    : cfg_(cfg), requests_(std::move(requests)) {}

void Simulator::run() {
    schedule_arrivals();

    while (!pq_.empty()) {
        Event event = pq_.top();
        pq_.pop();
        now_ms_ = event.time_ms;
        handle_event(event);
    }

    int finished{0}, rejected{0};
    for (const auto& req : requests_) {
        if (req.state == RequestState::Finished) finished++;
        if (req.state == RequestState::Rejected) rejected++;
    }
    std::cout << "Finished: " << finished << ", Rejected: " << rejected << '\n';
}
void Simulator::schedule_arrivals() {
    for (int i = 0; i < (int)requests_.size(); ++i) {
        pq_.push(Event{requests_[i].arrival_time_ms, EventType::Arrival, i});

    }
}
void Simulator::handle_event(const Event& event){
    switch(event.type){
        case EventType::Arrival: on_arrival(event); break;
        case EventType::StartPrefill: on_start_prefill(event); break;
        case EventType::StartDecode: on_start_decode(event); break;
        case EventType::Finish: on_finish(event); break;
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

double Simulator::decode_duration_ms(int gen_tokens, int /*active_decode*/) const {
    return 1000.0 * gen_tokens / cfg_.gpu.decode_tps;
}

void Simulator::on_arrival(const Event& event){
    auto& req = requests_[event.request_index];
    bool ok = cfg_.policy.safe_reservation
                  ? can_reserve_decode(req.prompt_tokens, req.gen_tokens)
                  : can_admit_prompt(req.prompt_tokens);

    if (!ok) {
        req.state = RequestState::Rejected;
        return;
    }

    allocate_kv(req.prompt_tokens);
    req.state = RequestState::Queued;
    if (active_prefill_ + active_decode_ < cfg_.gpu.max_concurrent) {
        pq_.push(Event{now_ms_, EventType::StartPrefill, event.request_index});
    } else {
        //For V1, skip queueing logic for now. We will add it later.
    }
}

void Simulator::on_start_prefill(const Event& event){
    auto& req = requests_[event.request_index];
    req.state = RequestState::Prefill;
    req.start_prefill_ms = now_ms_;
    active_prefill_++;
    double duration = prefill_duration_ms(req.prompt_tokens);
    pq_.push(Event{now_ms_ + duration, EventType::StartDecode, event.request_index});
}

void Simulator::on_start_decode(const Event& event){
    auto& req = requests_[event.request_index];
    active_prefill_--;
    req.state = RequestState::Decode;
    req.start_decode_ms = now_ms_;
    active_decode_++;
    double duration = decode_duration_ms(req.gen_tokens, active_decode_);
    pq_.push(Event{now_ms_ + duration, EventType::Finish, event.request_index});
}

void Simulator::on_finish(const Event& event){
    auto& req = requests_[event.request_index];
    active_decode_--;
    req.state = RequestState::Finished;
    req.finish_ms = now_ms_;
    free_kv(req.prompt_tokens + req.gen_tokens);
}

int main() {
    SimConfig cfg;
    cfg.gpu.vram_bytes = 8ull * 1024 * 1024 * 1024; // 8GB for demo
    cfg.gpu.max_concurrent = 2;
    cfg.gpu.prefill_tps = 1000;
    cfg.gpu.decode_tps = 500;
    cfg.policy.kv_bytes_per_token = 2048;
    cfg.policy.safe_reservation = true;

    std::vector<Request> reqs;
    reqs.push_back(Request{"req1", 0.0, 200, 400, false});
    reqs.push_back(Request{"req2", 50.0, 150, 300, false});

    Simulator sim(cfg, std::move(reqs));
    sim.run();
    return 0;
}

