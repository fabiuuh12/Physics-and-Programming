#include "alice/config.hpp"

#include "alice/string_utils.hpp"

#include <sstream>

namespace alice {

static std::filesystem::path resolve_path(const std::string& raw, const std::filesystem::path& base) {
    std::filesystem::path candidate(raw);
    if (candidate.is_relative()) {
        candidate = base / candidate;
    }
    return std::filesystem::absolute(candidate).lexically_normal();
}

static std::optional<std::string> extract_string_field(const std::string& text, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    const std::size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t colon_pos = text.find(':', key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t first_quote = text.find('"', colon_pos + 1);
    if (first_quote == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t second_quote = text.find('"', first_quote + 1);
    if (second_quote == std::string::npos) {
        return std::nullopt;
    }
    return text.substr(first_quote + 1, second_quote - first_quote - 1);
}

static std::optional<int> extract_int_field(const std::string& text, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    const std::size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t colon_pos = text.find(':', key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        return std::nullopt;
    }

    std::size_t start = colon_pos + 1;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }
    std::size_t end = start;
    while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end]))) {
        ++end;
    }
    if (start == end) {
        return std::nullopt;
    }

    try {
        return std::stoi(text.substr(start, end - start));
    } catch (...) {
        return std::nullopt;
    }
}

static std::vector<std::string> extract_string_array_field(const std::string& text, const std::string& key) {
    std::vector<std::string> out;
    const std::string marker = "\"" + key + "\"";
    const std::size_t key_pos = text.find(marker);
    if (key_pos == std::string::npos) {
        return out;
    }
    const std::size_t colon_pos = text.find(':', key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        return out;
    }
    const std::size_t open = text.find('[', colon_pos + 1);
    const std::size_t close = text.find(']', open == std::string::npos ? colon_pos + 1 : open + 1);
    if (open == std::string::npos || close == std::string::npos || close <= open) {
        return out;
    }

    std::size_t pos = open + 1;
    while (pos < close) {
        const std::size_t q1 = text.find('"', pos);
        if (q1 == std::string::npos || q1 >= close) {
            break;
        }
        const std::size_t q2 = text.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 > close) {
            break;
        }
        out.push_back(text.substr(q1 + 1, q2 - q1 - 1));
        pos = q2 + 1;
    }

    return out;
}

AliceConfig load_config(const std::filesystem::path& config_path_raw) {
    const std::filesystem::path config_path = std::filesystem::absolute(config_path_raw).lexically_normal();
    const std::filesystem::path project_root = config_path.parent_path().parent_path();

    std::string text;
    if (const auto maybe = read_file(config_path.string()); maybe.has_value()) {
        text = *maybe;
    }

    std::vector<std::filesystem::path> allowed_roots;
    for (const auto& raw : extract_string_array_field(text, "allowed_roots")) {
        allowed_roots.push_back(resolve_path(raw, project_root));
    }
    if (allowed_roots.empty()) {
        allowed_roots.push_back(std::filesystem::absolute(project_root).lexically_normal());
    }

    std::filesystem::path log_dir = std::filesystem::absolute(project_root / "logs").lexically_normal();
    if (const auto raw = extract_string_field(text, "log_dir"); raw.has_value() && !raw->empty()) {
        log_dir = resolve_path(*raw, project_root);
    }

    int max_runtime_seconds = 300;
    if (const auto n = extract_int_field(text, "max_runtime_seconds"); n.has_value() && *n > 0) {
        max_runtime_seconds = *n;
    }

    AliceConfig cfg;
    cfg.project_root = std::filesystem::absolute(project_root).lexically_normal();
    cfg.allowed_roots = std::move(allowed_roots);
    cfg.log_dir = std::move(log_dir);
    cfg.max_runtime_seconds = max_runtime_seconds;
    return cfg;
}

}  // namespace alice
