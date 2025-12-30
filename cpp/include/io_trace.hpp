#pragma once
#include <string>
#include <vector>
#include "types.hpp"

bool load_trace(const std::string& path, std::vector<Request>& out, std::string& err);