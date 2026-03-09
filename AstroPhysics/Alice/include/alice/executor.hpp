#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "alice/types.hpp"

namespace alice {

class AliceExecutor {
public:
    AliceExecutor(std::vector<std::filesystem::path> allowed_roots, std::filesystem::path log_dir,
                  int max_runtime_seconds = 300);

    ExecResult list_files(const std::optional<std::string>& target);
    ExecResult open_folder(const std::optional<std::string>& target);
    ExecResult run_file(const std::optional<std::string>& target);
    ExecResult stop_process(const std::optional<int>& pid);
    void shutdown();

private:
    struct ProcessInfo {
        int pid = -1;
        std::filesystem::path log_path;
    };

    std::vector<std::filesystem::path> allowed_roots_;
    std::filesystem::path log_dir_;
    int max_runtime_seconds_ = 300;

    std::map<int, ProcessInfo> processes_;
    std::optional<int> last_pid_;
    std::vector<std::filesystem::path> indexed_files_;

    void cleanup_finished();
    bool is_allowed(const std::filesystem::path& path) const;
    bool is_runnable_candidate(const std::filesystem::path& path) const;
    std::string display_path(const std::filesystem::path& path) const;

    std::pair<std::optional<std::filesystem::path>, std::optional<std::string>> resolve_target(
        const std::optional<std::string>& raw_target, bool must_exist, std::optional<bool> expect_directory) const;

    std::pair<std::optional<std::filesystem::path>, std::optional<std::string>>
    resolve_runnable_target(const std::optional<std::string>& raw_target);

    void refresh_file_index();
    static std::string normalize_query(const std::string& text);
    static std::vector<std::string> query_tokens(const std::string& text);
    double score_candidate(const std::string& query, const std::filesystem::path& candidate) const;

    std::pair<std::optional<std::filesystem::path>, std::vector<std::filesystem::path>>
    find_best_file_match(const std::string& raw_target);

    static std::optional<std::string> find_command(const std::string& name);
    static std::vector<std::string> split_env_flags(const char* value);
    static std::string timestamp_for_filename();

    static bool run_blocking_command(const std::vector<std::string>& command, const std::filesystem::path& cwd,
                                     const std::filesystem::path& log_path);

    std::optional<int> spawn_process(const std::vector<std::string>& command, const std::filesystem::path& cwd,
                                     const std::filesystem::path& log_path) const;
};

}  // namespace alice
