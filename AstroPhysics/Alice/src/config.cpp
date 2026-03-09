#include "alice/config.hpp"

#include "alice/string_utils.hpp"

#include <fstream>
#include <regex>
#include <sstream>

namespace alice {

static std::filesystem::path resolve_path(const std::string& raw, const std::filesystem::path& base) {
    std::filesystem::path candidate(raw);
    if (candidate.is_relative()) {
        return std::filesystem::weakly_canonical(base / candidate);
    }
    return std::filesystem::weakly_canonical(candidate);
}

AliceConfig load_config(const std::filesystem::path& config_path_raw) {
    const std::filesystem::path config_path = std::filesystem::weakly_canonical(config_path_raw);
    const std::filesystem::path project_root = config_path.parent_path().parent_path();

    std::string text;
    if (const auto maybe = read_file(config_path.string())) {
        text = *maybe;
    }

    std::vector<std::filesystem::path> allowed_roots;
    {
        std::regex list_regex(R"("allowed_roots"\s*:\s*\[(.*?)\])", std::regex::icase | std::regex::nosubs);
        std::smatch m;
        if (std::regex_search(text, m, std::regex(R"("allowed_roots"\s*:\s*\[(.*?)\])", std::regex::icase | std::regex::dotall))) {
            const std::string body = m[1].str();
            std::regex item_regex(R"("([^"]+)")");
            auto begin = std::sregex_iterator(body.begin(), body.end(), item_regex);
            auto end = std::sregex_iterator();
            for (auto it = begin; it != end; ++it) {
                allowed_roots.push_back(resolve_path((*it)[1].str(), project_root));
            }
        }
    }
    if (allowed_roots.empty()) {
        allowed_roots.push_back(std::filesystem::weakly_canonical(project_root));
    }

    std::filesystem::path log_dir = std::filesystem::weakly_canonical(project_root / "logs");
    {
        std::smatch m;
        if (std::regex_search(text, m, std::regex(R"("log_dir"\s*:\s*"([^"]+)")", std::regex::icase))) {
            log_dir = resolve_path(m[1].str(), project_root);
        }
    }

    int max_runtime_seconds = 300;
    {
        std::smatch m;
        if (std::regex_search(text, m, std::regex(R"("max_runtime_seconds"\s*:\s*(\d+))", std::regex::icase))) {
            try {
                max_runtime_seconds = std::stoi(m[1].str());
            } catch (...) {
                max_runtime_seconds = 300;
            }
            if (max_runtime_seconds <= 0) {
                max_runtime_seconds = 300;
            }
        }
    }

    AliceConfig cfg;
    cfg.project_root = std::filesystem::weakly_canonical(project_root);
    cfg.allowed_roots = std::move(allowed_roots);
    cfg.log_dir = std::move(log_dir);
    cfg.max_runtime_seconds = max_runtime_seconds;
    return cfg;
}

}  // namespace alice
