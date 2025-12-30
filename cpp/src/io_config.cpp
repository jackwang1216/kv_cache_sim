#include "io_config.hpp"
#include <fstream>
#include <sstream>

bool load_config(const std::string& path, SimConfig& cfg, std::string& err) {
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
        if (key == "vram_bytes" && (iss >> uval)) cfg.gpu.vram_bytes = uval;
        else if (key == "max_concurrent" && (iss >> ival)) cfg.gpu.max_concurrent = ival;
        else if (key == "prefill_tps" && (iss >> dval)) cfg.gpu.prefill_tps = dval;
        else if (key == "decode_tps" && (iss >> dval)) cfg.gpu.decode_tps = dval;
        else if (key == "kv_bytes_per_token" && (iss >> uval)) cfg.policy.kv_bytes_per_token = uval;
        else if (key == "max_queue" && (iss >> ival)) cfg.policy.max_queue = ival;
        else if (key == "safe_reservation" && (iss >> ival)) cfg.policy.safe_reservation = (ival != 0);
        else if (key == "timeseries_dt_ms" && (iss >> dval)) cfg.timeseries_dt_ms = dval;
    }
    return true;
}