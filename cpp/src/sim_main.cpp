#include <iostream>
#include "simulator.hpp"
#include "io_config.hpp"
#include "io_trace.hpp"
#include "io_output.hpp"

int main(int argc, char** argv) {
    std::string config_path = (argc > 1) ? argv[1] : "";
    std::string trace_path  = (argc > 2) ? argv[2] : "";
    std::string out_dir     = (argc > 3) ? argv[3] : "runs/demo";

    SimConfig cfg;
    std::string err;

    if (!config_path.empty()) {
        load_config(config_path, cfg, err);
        if (!err.empty()) std::cerr << "config: " << err << "\n";
        err.clear();
    }

    std::vector<Request> reqs;
    if (!trace_path.empty()) {
        if (!load_trace(trace_path, reqs, err)) {
            std::cerr << "trace error: " << err << "\n";
            return 1;
        }
    } else {
        // Fallback demo data
        reqs.push_back(Request{"req1", 0.0, 200, 400, false});
        reqs.push_back(Request{"req2", 50.0, 150, 300, false});
    }

    Simulator sim(cfg, std::move(reqs));
    sim.run();

    if (!write_summary(out_dir, sim.requests_view_for_output(), err)) { // see note below
        std::cerr << "write_summary error: " << err << "\n";
    }
    return 0;
}