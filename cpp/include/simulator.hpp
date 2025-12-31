#pragma once
#include <vector>
#include <queue>
#include <deque>
#include "types.hpp"
#include "events.hpp"
#include "rng.hpp"

class Simulator {
public:
    Simulator(SimConfig cfg, std::vector<Request> requests);
    void run();
    const std::vector<Request>& requests() const { return requests_; }
    const std::vector<EventRecord>& events() const { return events_; }
    const std::vector<TimeseriesSample>& samples() const { return samples_; }

private:
    void schedule_arrivals();
    void handle_event(const Event& event);
    void on_arrival(const Event& event);
    void on_start_prefill(const Event& event);
    void on_start_decode(const Event& event);
    void on_finish(const Event& event);

    void try_start_prefill();
    int pick_next_from_queue();
    void record_event(EventType type, const Request& req);
    void sample_until(double time_ms);

    double prefill_duration_ms(int prompt_tokens) const;
    double decode_duration_ms(int gen_tokens, int active_decode) const;
    bool can_admit_prompt(int prompt_tokens) const;
    bool can_reserve_decode(int prompt_tokens, int gen_tokens) const;
    void allocate_kv(int tokens);
    void free_kv(int tokens);

private:
    SimConfig cfg_;
    std::vector<Request> requests_;
    std::priority_queue<Event, std::vector<Event>, EventCompare> pq_;
    std::deque<int> prefill_queue_;
    std::vector<EventRecord> events_;
    std::vector<TimeseriesSample> samples_;

    double now_ms_ = 0.0;
    double next_sample_ms_ = 0.0;
    std::uint64_t vram_used_ = 0;
    int active_prefill_ = 0;
    int active_decode_ = 0;

    std::uint64_t tokens_generated_total_ = 0;
    int rejects_total_ = 0;
    std::uint64_t last_tokens_sampled_ = 0;
    int last_rejects_sampled_ = 0;
};