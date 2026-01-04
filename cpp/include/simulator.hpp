#pragma once
#include <vector>
#include <queue>
#include <deque>
#include <list>
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
    double sim_end_ms() const { return sim_end_ms_; }
    std::uint64_t tokens_generated_total() const { return tokens_generated_total_; }

    // Phase 8: Extended metrics getters
    int retry_attempts() const { return retry_attempts_; }
    int retry_successes() const { return retry_successes_; }
    int handoffs_total() const { return handoffs_total_; }
    int cross_gpu_decodes() const { return cross_gpu_decodes_; }
    int max_global_queue_depth() const { return max_global_queue_depth_; }
    const std::vector<std::uint64_t>& peak_vram_per_gpu() const { return peak_vram_per_gpu_; }
    const std::vector<std::uint64_t>& tokens_per_gpu() const { return tokens_per_gpu_; }
    const std::vector<int>& requests_finished_per_gpu() const { return requests_finished_per_gpu_; }
    int num_gpus() const { return static_cast<int>(gpus_.size()); }

private:
    int route_gpu_for_request(const Request& req);
    void schedule_arrivals();
    void handle_event(const Event& event);
    void on_arrival(const Event& event);
    void on_start_prefill(const Event& event);
    void on_start_decode(const Event& event);
    void on_finish(const Event& event);

    void try_start_prefill(int gpu_idx);
    int pick_next_from_queue(int gpu_idx);
    void record_event(EventType type, const Request& req, int gpu_idx);
    void sample_until(double time_ms);

    double prefill_duration_ms(int prompt_tokens, int gpu_idx) const;
    double decode_duration_ms(int gen_tokens, int active_decode, int gpu_idx) const;
    bool can_admit_prompt(int prompt_tokens, int gpu_idx) const;
    bool can_reserve_decode(int prompt_tokens, int gen_tokens, int gpu_idx) const;
    void allocate_kv_bytes(int req_idx, std::uint64_t bytes, int gpu_idx);
    void free_kv_bytes(int req_idx, std::uint64_t bytes, int gpu_idx);

    bool ensure_capacity_for(std::uint64_t bytes_needed, int gpu_idx);
    bool evict_one(int gpu_idx);
    void touch_lru(int req_idx, int gpu_idx);

    double score_gpu(int gpu_idx) const;

    int route_decode(int prefill_gpu, const Request& req);
    void on_handoff_start(const Event& event);
    void on_handoff_complete(const Event& event);

    void precompute_topology();
    bool can_fit_kv(int gpu_idx, const Request& req) const;
    double get_link_bandwidth(int src_gpu_idx, int dest_gpu_idx) const;
    double get_link_latency(int src_gpu_idx, int dest_gpu_idx) const;
    double estimate_handoff_ms(int src_gpu_idx, int dest_gpu_idx, const Request& req) const;
    double compute_decode_score(int src_gpu_idx, int dest_gpu_idx, const Request& req) const;

    void try_dispatch_global_queue();
    int find_alternate_gpu(int exclude_gpu, const Request& req) const;

private:
    SimConfig cfg_;
    std::vector<Request> requests_;
    std::vector<GPUState> gpus_;
    std::priority_queue<Event, std::vector<Event>, EventCompare> pq_;
    std::vector<EventRecord> events_;
    std::vector<TimeseriesSample> samples_;
    std::deque<int> global_queue_;

    double now_ms_ = 0.0;
    double next_sample_ms_ = 0.0;
    double sim_end_ms_ = 0.0;

    std::uint64_t tokens_generated_total_ = 0;
    int rejects_total_ = 0;
    std::uint64_t last_tokens_sampled_ = 0;
    int last_rejects_sampled_ = 0;

    // Phase 8: Extended metrics counters
    int retry_attempts_ = 0;
    int retry_successes_ = 0;
    int handoffs_total_ = 0;
    int cross_gpu_decodes_ = 0;
    int max_global_queue_depth_ = 0;
    std::vector<std::uint64_t> peak_vram_per_gpu_;
    std::vector<std::uint64_t> tokens_per_gpu_;
    std::vector<int> requests_finished_per_gpu_;

    RNG rng_;
};