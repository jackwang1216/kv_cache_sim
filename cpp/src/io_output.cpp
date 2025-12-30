#include "io_output.hpp"
#include <fstream>
#include <filesystem>

bool write_summary(const std::string& out_dir, const std::vector<Request>& reqs, std::string& err) {
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);
    std::string path = out_dir + "/summary.json";
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        err = "cannot open summary file";
        return false;
    }
    int finished = 0, rejected = 0;
    for (const auto& r : reqs) {
        if (r.state == RequestState::Finished) finished++;
        if (r.state == RequestState::Rejected) rejected++;
    }
    ofs << "{\n"
        << "  \"finished\": " << finished << ",\n"
        << "  \"rejected\": " << rejected << "\n"
        << "}\n";
    return true;
}