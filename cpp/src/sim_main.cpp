#include <iostream>
#include <unordered_map>
#include "simulator.hpp"
#include "io_config.hpp"
#include "io_trace.hpp"
#include "io_output.hpp"

static std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> m;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto eq = a.find('=');
        if (eq != std::string::npos) {
            m[a.substr(0, eq)] = a.substr(eq + 1);
        } else if (i+1 < argc && argv[i+1][0] != '-') {
            m[a] = argv[i+1];
            ++i;
        } else {
            m[a] = "";
        }
    }
    return m;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    std::string config_path = args.count("--config") ? args["--config"] : (argc > 1 ? argv[1] : "");
    std::string trace_path  = args.count("--trace")  ? args["--trace"]  : (argc > 2 ? argv[2] : "");
    std::string out_dir     = args.count("--out")    ? args["--out"]    : (argc > 3 ? argv[3] : "runs/demo");
    unsigned int seed = 12345;
    if (args.count("--seed")) seed = static_cast<unsigned int>(std::stoul(args["--seed"]));

    SimConfig cfg;
    cfg.seed = seed;
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
        reqs.push_back(Request{"req1", 0.0, 200, 400, false});
        reqs.push_back(Request{"req2", 50.0, 150, 300, false});
    }

    Simulator sim(cfg, std::move(reqs));
    sim.run();

    if (!write_summary(out_dir, sim.requests(), sim.samples(), sim.tokens_generated_total(), sim.sim_end_ms(), sim.events(), cfg, err)){
        std::cerr << "write_summary error: " << err << "\n";
    }
    if (!write_timeseries_csv(out_dir, sim.samples(), err)) std::cerr << "write_timeseries error: " << err << "\n";
    if (!write_events_jsonl(out_dir, sim.events(), err)) std::cerr << "write_events error: " << err << "\n";
    if (!write_run_meta(out_dir, cfg, err, config_path)) std::cerr << "write_run_meta error: " << err << "\n";

    return 0;
}