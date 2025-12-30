#pragma once
#include <string>
#include "types.hpp"

bool load_config(const std::string& path, SimConfig& cfg, std::string& err);