#pragma once
#include <string>
#include <vector>
#include "types.hpp"

bool write_summary(const std::string& out_dir, const std::vector<Request>& reqs, std::string& err);
bool write_timeseries_csv(const std::string& out_dir, const std::vector<TimeseriesSample>& samples, std::string& err);
bool write_events_jsonl(const std::string& out_dir, const std::vector<EventRecord>& events, std::string& err);
bool write_run_meta(const std::string& out_dir, const SimConfig& cfg, std::string& err);