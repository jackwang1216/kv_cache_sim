#include "io_output.hpp"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <chrono>

namespace fs = std::filesystem;

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

bool write_summary(const std::string& out_dir, const std::vector<Request>& reqs, std::string& err) {
    if (!ensure_dir(out_dir, err)) return false;
    std::ofstream ofs(out_dir + "/summary.json");
    if (!ofs.is_open()) { err = "cannot open summary"; return false; }

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

    ofs << "{\n"
        << "  \"finished\": " << finished << ",\n"
        << "  \"rejected\": " << rejected << ",\n"
        << "  \"p50_latency_ms\": " << p50 << ",\n"
        << "  \"p95_latency_ms\": " << p95 << "\n"
        << "}\n";
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
    ofs << "{\n"
        << "  \"seed\": " << cfg.seed << ",\n"
        << "  \"timeseries_dt_ms\": " << cfg.timeseries_dt_ms << ",\n"
        << "  \"timestamp_ms\": " << ms << "\n"
        << "}\n";
    return true;
}