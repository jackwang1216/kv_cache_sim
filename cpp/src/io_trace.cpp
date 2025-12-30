#include "io_trace.hpp"
#include <fstream>
#include <sstream>

bool load_trace(const std::string& path, std::vector<Request>& out, std::string& err) {
    std::ifstream f(path);
    if (!f.is_open()) {
        err = "trace file not found";
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Request r;
        int streaming_int = 0;
        if (!(iss >> r.id >> r.arrival_time_ms >> r.prompt_tokens >> r.gen_tokens >> streaming_int)) {
            err = "failed to parse line: " + line;
            return false;
        }
        r.streaming = (streaming_int != 0);
        out.push_back(std::move(r));
    }
    return true;
}