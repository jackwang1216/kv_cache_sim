#pragma once
#include <string>
#include <vector>
#include "types.hpp"

// Phase 8: Extended metrics for summary output
struct ExtendedMetrics {
    int retry_attempts = 0;
    int retry_successes = 0;
    int handoffs_total = 0;
    int cross_gpu_decodes = 0;
    int max_global_queue_depth = 0;
    std::vector<std::uint64_t> peak_vram_per_gpu;
    std::vector<std::uint64_t> tokens_per_gpu;
    std::vector<int> requests_finished_per_gpu;
};

bool write_summary(
    const std::string& out_dir,
    const std::vector<Request>& reqs,
    const std::vector<TimeseriesSample>& samples,
    std::uint64_t tokens_generated_total,
    double sim_end_ms,
    const std::vector<EventRecord>& events,
    const SimConfig& cfg,
    const ExtendedMetrics& ext_metrics,
    std::string& err
);
bool write_timeseries_csv(const std::string& out_dir, const std::vector<TimeseriesSample>& samples, int num_gpus, std::string& err);
bool write_events_jsonl(const std::string& out_dir, const std::vector<EventRecord>& events, std::string& err);
bool write_run_meta(const std::string& out_dir, const SimConfig& cfg, std::string& err, const std::string& config_path = "");