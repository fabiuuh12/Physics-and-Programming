#include "alice/string_utils.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace alice {

std::string trim(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string normalize_text(const std::string& value) {
    std::string lowered = to_lower(trim(value));
    std::string out;
    out.reserve(lowered.size());
    bool previous_space = false;
    for (unsigned char c : lowered) {
        if (std::isalnum(c) || c == '_' || c == '\'') {
            out.push_back(static_cast<char>(c));
            previous_space = false;
            continue;
        }
        if (!previous_space) {
            out.push_back(' ');
            previous_space = true;
        }
    }
    return trim(out);
}

std::vector<std::string> split_words(const std::string& value) {
    std::vector<std::string> out;
    std::istringstream stream(value);
    std::string part;
    while (stream >> part) {
        out.push_back(part);
    }
    return out;
}

bool starts_with_ci(const std::string& text, const std::string& prefix) {
    if (prefix.size() > text.size()) {
        return false;
    }
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(text[i])) !=
            std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

std::string replace_all(std::string text, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return text;
    }
    std::size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

std::string shell_quote(const std::string& value) {
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') {
            out += "'\\''";
            continue;
        }
        out.push_back(c);
    }
    out.push_back('\'');
    return out;
}

std::string join(const std::vector<std::string>& parts, const std::string& delim) {
    std::ostringstream out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            out << delim;
        }
        out << parts[i];
    }
    return out.str();
}

static std::tm local_tm_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_value{};
#if defined(_WIN32)
    localtime_s(&tm_value, &t);
#else
    localtime_r(&t, &tm_value);
#endif
    return tm_value;
}

std::string now_iso8601() {
    const auto tm_value = local_tm_now();
    std::ostringstream out;
    out << std::put_time(&tm_value, "%Y-%m-%dT%H:%M:%S");
    return out.str();
}

std::string format_clock_time() {
    const auto tm_value = local_tm_now();
    std::ostringstream out;
    out << std::put_time(&tm_value, "%I:%M %p");
    return out.str();
}

std::string format_long_date() {
    const auto tm_value = local_tm_now();
    std::ostringstream out;
    out << std::put_time(&tm_value, "%A, %B %d, %Y");
    return out.str();
}

std::optional<std::string> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return std::nullopt;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool write_file(const std::string& path, const std::string& content) {
    try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            return false;
        }
        file << content;
        return file.good();
    } catch (...) {
        return false;
    }
}

}  // namespace alice
