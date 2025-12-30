#pragma once
#include <string>
#include <vector>
#include "types.hpp"

bool write_summary(const std::string& out_dir, const std::vector<Request>& reqs, std::string& err);