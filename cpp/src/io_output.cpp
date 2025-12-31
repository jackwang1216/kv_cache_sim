#include "io_output.hpp"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <chrono>

namespace fs = std::filesystem;

static std::uint64_t fnv1a_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return 0;
    std::uint64_t hash = 1469598103934665603ull;
    char buf[4096];
    while (f.read(buf, sizeof(buf)) || f.gcount()) {
        std::streamsize n = f.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            hash ^= static_cast<unsigned char>(buf[i]);
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

static bool ensure_dir(const std::string& out_dir, std::string& err) {
    std::error_code ec;
    if (fs::exists(out_dir, ec)) {
        if (fs::is_directory(out_dir, ec)) return true;
        err = "out dir exists but is not a directory";
        return false;
    }
    fs::create_directories(out_dir, ec);
    if (ec) {
        err = "cannot create out dir: " + ec.message();
        return false;
    }
    return true;
}

bool write_summary(const std::string& out_dir,
                   const std::vector<Request>& reqs,
                   const std::vector<TimeseriesSample>& samples,
                   std::uint64_t tokens_generated_total,
                   double sim_end_ms,
                   const std::vector<EventRecord>& events,
                   const SimConfig& cfg,
                   std::string& err) {
    if (!ensure_dir(out_dir, err)) return false;
    std::ofstream ofs(out_dir + "/summary.json");
    if (!ofs.is_open()) { err = "cannot open summary"; return false; }

    // Latencies
    std::vector<double> latencies;
    int finished = 0, rejected = 0;
    for (const auto& r : reqs) {
        if (r.state == RequestState::Finished) {
            finished++;
            latencies.push_back(r.finish_ms - r.arrival_time_ms);
        }
        if (r.state == RequestState::Rejected) rejected++;
    }
    auto pct = [&](double p) {
        if (latencies.empty()) return 0.0;
        std::sort(latencies.begin(), latencies.end());
        size_t idx = static_cast<size_t>(p * (latencies.size() - 1));
        return latencies[idx];
    };
    double p50 = pct(0.50);
    double p95 = pct(0.95);
    double p99 = pct(0.99);

    // Throughput (tokens/sec) over makespan
    double makespan_ms = sim_end_ms > 0 ? sim_end_ms : 0.0;
    double throughput_tps = (makespan_ms > 0.0)
        ? (static_cast<double>(tokens_generated_total) / (makespan_ms / 1000.0))
        : 0.0;

    // Completion / reject rates
    int total = static_cast<int>(reqs.size());
    double completion_rate = (total > 0) ? static_cast<double>(finished) / total : 0.0;
    double reject_rate     = (total > 0) ? static_cast<double>(rejected) / total : 0.0;

    // Time-weighted averages from timeseries
    double avg_vram = 0.0;
    double busy_ms = 0.0;
    if (samples.size() >= 2) {
        double weighted_vram = 0.0;
        double total_ms = 0.0;
        for (size_t i = 1; i < samples.size(); ++i) {
            double dt = samples[i].time_ms - samples[i-1].time_ms;
            weighted_vram += dt * static_cast<double>(samples[i-1].vram_used);
            if (samples[i-1].active_prefill + samples[i-1].active_decode > 0) {
                busy_ms += dt;
            }
            total_ms += dt;
        }
        if (total_ms > 0.0) {
            avg_vram = weighted_vram / total_ms;
        }
    }

    // Policy strings and evict count
    auto policy_to_str = [](MemoryPressurePolicy p) {
        return (p == MemoryPressurePolicy::Evict) ? "evict" : "reject";
    };
    auto evict_policy_to_str = [](EvictionPolicy p) {
        return (p == EvictionPolicy::LRU) ? "lru" : "fifo";
    };
    int evict_count = 0;
    for (const auto& e : events) {
        if (e.type == EventType::Evict) evict_count++;
    }

    ofs << "{\n"
        << "  \"finished\": " << finished << ",\n"
        << "  \"rejected\": " << rejected << ",\n"
        << "  \"completion_rate\": " << completion_rate << ",\n"
        << "  \"reject_rate\": " << reject_rate << ",\n"
        << "  \"throughput_tokens_per_sec\": " << throughput_tps << ",\n"
        << "  \"p50_latency_ms\": " << p50 << ",\n"
        << "  \"p95_latency_ms\": " << p95 << ",\n"
        << "  \"p99_latency_ms\": " << p99 << ",\n"
        << "  \"avg_vram_bytes\": " << avg_vram << ",\n"
        << "  \"gpu_busy_ms\": " << busy_ms << ",\n"
        << "  \"makespan_ms\": " << makespan_ms << ",\n"
        << "  \"memory_pressure_policy\": \"" << policy_to_str(cfg.policy.memory_pressure_policy) << "\",\n";
    if (cfg.policy.memory_pressure_policy == MemoryPressurePolicy::Evict) {
        ofs << "  \"eviction_policy\": \"" << evict_policy_to_str(cfg.policy.eviction_policy) << "\",\n"
            << "  \"evictions\": " << evict_count << "\n";
    } else {
        ofs << "  \"evictions\": " << evict_count << "\n";
    }
    ofs << "}\n";
    return true;
}

bool write_timeseries_csv(const std::string& out_dir, const std::vector<TimeseriesSample>& samples, std::string& err) {
    if (!ensure_dir(out_dir, err)) return false;
    std::ofstream ofs(out_dir + "/timeseries.csv");
    if (!ofs.is_open()) { err = "cannot open timeseries"; return false; }
    ofs << "time_ms,vram_used,active_prefill,active_decode,queue_depth,tokens_generated_delta,rejects_delta\n";
    for (const auto& s : samples) {
        ofs << s.time_ms << "," << s.vram_used << "," << s.active_prefill << ","
            << s.active_decode << "," << s.queue_depth << ","
            << s.tokens_generated_delta << "," << s.rejects_delta << "\n";
    }
    return true;
}

static std::string event_type_str(EventType t) {
    switch (t) {
        case EventType::Arrival: return "arrival";
        case EventType::Enqueue: return "enqueue";
        case EventType::StartPrefill: return "start_prefill";
        case EventType::StartDecode: return "start_decode";
        case EventType::Finish: return "finish";
        case EventType::Reject: return "reject";
        case EventType::Evict: return "evict";
    }
    return "unknown";
}

bool write_events_jsonl(const std::string& out_dir, const std::vector<EventRecord>& events, std::string& err) {
    if (!ensure_dir(out_dir, err)) return false;
    std::ofstream ofs(out_dir + "/events.jsonl");
    if (!ofs.is_open()) { err = "cannot open events"; return false; }
    for (const auto& e : events) {
        ofs << "{"
            << "\"time_ms\":" << e.time_ms << ","
            << "\"type\":\"" << event_type_str(e.type) << "\","
            << "\"request_id\":\"" << e.request_id << "\""
            << "}\n";
    }
    return true;
}

bool write_run_meta(const std::string& out_dir, const SimConfig& cfg, std::string& err) {
    if (!ensure_dir(out_dir, err)) return false;
    std::ofstream ofs(out_dir + "/run_meta.json");
    if (!ofs.is_open()) { err = "cannot open run_meta"; return false; }
    auto now = std::chrono::system_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    // config_path parameter is needed; keep default hashing to 0 if missing
    // (overload signature includes config_path; adjust signature to accept it)
    // Placeholder to avoid compile error; real hash computed in overload below.
    std::uint64_t cfg_hash = 0;

    ofs << "{\n"
        << "  \"seed\": " << cfg.seed << ",\n"
        << "  \"timeseries_dt_ms\": " << cfg.timeseries_dt_ms << ",\n"
        << "  \"timestamp_ms\": " << ms << ",\n"
        << "  \"config_hash\": " << cfg_hash << ",\n"
        << "  \"scheduling\": \"" << (cfg.policy.scheduling == SchedulingMode::FIFO ? "fifo" : "shortest_remaining") << "\",\n"
        << "  \"memory_pressure_policy\": \"" << (cfg.policy.memory_pressure_policy == MemoryPressurePolicy::Evict ? "evict" : "reject") << "\",\n"
        << "  \"eviction_policy\": \"" << (cfg.policy.eviction_policy == EvictionPolicy::LRU ? "lru" : "fifo") << "\",\n"
        << "  \"decode_sharing_cap\": " << cfg.gpu.decode_sharing_cap << ",\n"
        << "  \"decode_efficiency\": " << cfg.gpu.decode_efficiency << "\n"
        << "}\n";
    return true;
}

bool write_run_meta(const std::string& out_dir, const SimConfig& cfg, std::string& err, const std::string& config_path) {
    if (!ensure_dir(out_dir, err)) return false;
    std::ofstream ofs(out_dir + "/run_meta.json");
    if (!ofs.is_open()) { err = "cannot open run_meta"; return false; }
    auto now = std::chrono::system_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    std::uint64_t cfg_hash = config_path.empty() ? 0 : fnv1a_file(config_path);

    ofs << "{\n"
        << "  \"seed\": " << cfg.seed << ",\n"
        << "  \"timeseries_dt_ms\": " << cfg.timeseries_dt_ms << ",\n"
        << "  \"timestamp_ms\": " << ms << ",\n"
        << "  \"config_hash\": " << cfg_hash << ",\n"
        << "  \"scheduling\": \"" << (cfg.policy.scheduling == SchedulingMode::FIFO ? "fifo" : "shortest_remaining") << "\",\n"
        << "  \"memory_pressure_policy\": \"" << (cfg.policy.memory_pressure_policy == MemoryPressurePolicy::Evict ? "evict" : "reject") << "\",\n"
        << "  \"eviction_policy\": \"" << (cfg.policy.eviction_policy == EvictionPolicy::LRU ? "lru" : "fifo") << "\",\n"
        << "  \"decode_sharing_cap\": " << cfg.gpu.decode_sharing_cap << ",\n"
        << "  \"decode_efficiency\": " << cfg.gpu.decode_efficiency << "\n"
        << "}\n";
    return true;
}