#pragma once

#include <filesystem>
#include <vector>

namespace alice {

struct AliceConfig {
    std::filesystem::path project_root;
    std::vector<std::filesystem::path> allowed_roots;
    std::filesystem::path log_dir;
    int max_runtime_seconds = 300;
};

AliceConfig load_config(const std::filesystem::path& config_path);

}  // namespace alice
