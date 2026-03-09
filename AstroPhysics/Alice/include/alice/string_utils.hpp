#pragma once

#include <optional>
#include <string>
#include <vector>

namespace alice {

std::string trim(const std::string& value);
std::string to_lower(std::string value);
std::string normalize_text(const std::string& value);
std::vector<std::string> split_words(const std::string& value);
bool starts_with_ci(const std::string& text, const std::string& prefix);
std::string replace_all(std::string text, const std::string& from, const std::string& to);
std::string shell_quote(const std::string& value);
std::string join(const std::vector<std::string>& parts, const std::string& delim);
std::string now_iso8601();
std::string format_clock_time();
std::string format_long_date();
std::optional<std::string> read_file(const std::string& path);
bool write_file(const std::string& path, const std::string& content);

}  // namespace alice
