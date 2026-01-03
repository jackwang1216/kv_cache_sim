#include "io_config.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

bool load_config(const std::string& path, SimConfig& cfg, std::string& err) {
    if (cfg.gpus.empty()) {
        cfg.gpus.push_back(GPUConfig{});
    }
    int num_gpus_requested = static_cast<int>(cfg.gpus.size());

    // Stub: if file missing, keep defaults; otherwise parse a simple key=value
    std::ifstream f(path);
    if (!f.is_open()) {
        err = "config file not found, using defaults";
        return true;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key, eq;
        double dval;
        std::uint64_t uval;
        int ival;
        if (!(iss >> key)) continue;
        auto to_lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            return s;
        };
        std::string sval;
        if (key == "num_gpus" && (iss >> ival) && ival > 0) num_gpus_requested = ival;
        else if (key == "vram_bytes" && (iss >> uval)) cfg.gpus[0].vram_bytes = uval;
        else if (key == "max_concurrent" && (iss >> ival)) cfg.gpus[0].max_concurrent = ival;
        else if (key == "prefill_tps" && (iss >> dval)) cfg.gpus[0].prefill_tps = dval;
        else if (key == "decode_tps" && (iss >> dval)) cfg.gpus[0].decode_tps = dval;
        else if (key == "kv_bytes_per_token" && (iss >> uval)) cfg.policy.kv_bytes_per_token = uval;
        else if (key == "max_queue" && (iss >> ival)) cfg.policy.max_queue = ival;
        else if (key == "safe_reservation" && (iss >> ival)) cfg.policy.safe_reservation = (ival != 0);
        else if (key == "timeseries_dt_ms" && (iss >> dval)) cfg.timeseries_dt_ms = dval;
        else if (key == "scheduling" && (iss >> sval)) {
            sval = to_lower(sval);
            if (sval == "fifo") cfg.policy.scheduling = SchedulingMode::FIFO;
            else if (sval == "shortest" || sval == "srt" || sval == "shortest_remaining")
                cfg.policy.scheduling = SchedulingMode::ShortestRemaining;
        }
        else if (key == "handoff_latency_us" && (iss >> dval)) {
            cfg.policy.handoff_latency_us = dval;
        }
        else if (key == "handoff_bandwidth_gbps" && (iss >> dval)) {
            cfg.policy.handoff_bandwidth_gbps = dval;
        }
        else if (key == "handoff_cost_weight" && (iss >> dval)) {
            cfg.policy.handoff_cost_weight = dval;
        }
        else if (key == "routing_policy" && (iss >> sval)) {
            sval = to_lower(sval);
            if (sval == "p2c" || sval == "power2choices" || sval == "power_of_two_choices") {
                cfg.policy.routing_policy = RoutingPolicy::P2C;
            } else if (sval == "roundrobin" || sval == "rr") {
                cfg.policy.routing_policy = RoutingPolicy::RoundRobin;
            } else if (sval == "leastloaded" || sval == "least" || sval == "ll") {
                cfg.policy.routing_policy = RoutingPolicy::LeastLoaded;
            }
        }
        else if (key == "link") {
            // Expected format: link <src> <dest> <bandwidth_gbps> <latency_ms>
            int src = -1, dest = -1;
            double bw = 0.0, lat = 0.0;
            if (iss >> src >> dest >> bw >> lat) {
                cfg.raw_links.push_back(RawLink{src, dest, bw, lat});
            }
        }
        else if (key == "memory_pressure_policy" && (iss >> sval)) {
            sval = to_lower(sval);
            if (sval == "reject") cfg.policy.memory_pressure_policy = MemoryPressurePolicy::Reject;
            else if (sval == "evict") cfg.policy.memory_pressure_policy = MemoryPressurePolicy::Evict;
        }
        else if (key == "eviction_policy" && (iss >> sval)) {
            sval = to_lower(sval);
            if (sval == "fifo") cfg.policy.eviction_policy = EvictionPolicy::FIFO;
            else if (sval == "lru") cfg.policy.eviction_policy = EvictionPolicy::LRU;
        }
        else if (key == "decode_sharing_cap" && (iss >> ival)) cfg.gpus[0].decode_sharing_cap = ival;
        else if (key == "decode_efficiency" && (iss >> dval)) cfg.gpus[0].decode_efficiency = dval;
        else if (key == "gpu") {
            // Format: gpu <id> [vram <bytes>] [prefill_tps <val>] [decode_tps <val>]
            int gpu_id = -1;
            if (!(iss >> gpu_id) || gpu_id < 0) continue;
            
            // Ensure gpus vector is large enough
            if (gpu_id >= static_cast<int>(cfg.gpus.size())) {
                cfg.gpus.resize(gpu_id + 1, cfg.gpus.empty() ? GPUConfig{} : cfg.gpus[0]);
            }
            if (gpu_id >= num_gpus_requested) num_gpus_requested = gpu_id + 1;
            
            // Parse key-value pairs for this GPU
            std::string subkey;
            while (iss >> subkey) {
                subkey = to_lower(subkey);
                if (subkey == "vram" && (iss >> uval)) {
                    cfg.gpus[gpu_id].vram_bytes = uval;
                } else if (subkey == "prefill_tps" && (iss >> dval)) {
                    cfg.gpus[gpu_id].prefill_tps = dval;
                } else if (subkey == "decode_tps" && (iss >> dval)) {
                    cfg.gpus[gpu_id].decode_tps = dval;
                } 
            }
        }
    }

    if (num_gpus_requested < 1) num_gpus_requested = 1;
    // Expand (not replace) gpus vector to reach num_gpus_requested
    // This preserves any per-GPU custom settings already parsed
    GPUConfig base = cfg.gpus.empty() ? GPUConfig{} : cfg.gpus[0];
    while (static_cast<int>(cfg.gpus.size()) < num_gpus_requested) {
        cfg.gpus.push_back(base);
    }
    // Shrink if needed (rare case where explicit gpu IDs < num_gpus)
    if (static_cast<int>(cfg.gpus.size()) > num_gpus_requested) {
        cfg.gpus.resize(num_gpus_requested);
    }
    return true;
}