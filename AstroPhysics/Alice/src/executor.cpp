#include "alice/executor.hpp"

#include "alice/string_utils.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace alice {

static std::filesystem::path normalize_path(const std::filesystem::path& path) {
    try {
        if (std::filesystem::exists(path)) {
            return std::filesystem::weakly_canonical(path);
        }
    } catch (...) {
    }
    return std::filesystem::absolute(path).lexically_normal();
}

static bool is_prefix_path(const std::filesystem::path& root, const std::filesystem::path& value) {
    const auto n_root = normalize_path(root);
    const auto n_value = normalize_path(value);

    auto it_root = n_root.begin();
    auto it_value = n_value.begin();
    for (; it_root != n_root.end(); ++it_root, ++it_value) {
        if (it_value == n_value.end() || *it_root != *it_value) {
            return false;
        }
    }
    return true;
}

AliceExecutor::AliceExecutor(std::vector<std::filesystem::path> allowed_roots, std::filesystem::path log_dir,
                             int max_runtime_seconds)
    : max_runtime_seconds_(max_runtime_seconds) {
    allowed_roots_.reserve(allowed_roots.size());
    for (auto& root : allowed_roots) {
        allowed_roots_.push_back(normalize_path(root));
    }

    log_dir_ = normalize_path(log_dir);
    std::filesystem::create_directories(log_dir_);
}

void AliceExecutor::cleanup_finished() {
    std::vector<int> finished;
    for (const auto& [pid, info] : processes_) {
        int status = 0;
        const pid_t result = waitpid(static_cast<pid_t>(pid), &status, WNOHANG);
        if (result == static_cast<pid_t>(pid)) {
            finished.push_back(pid);
        }
    }

    for (const int pid : finished) {
        processes_.erase(pid);
        if (last_pid_.has_value() && *last_pid_ == pid) {
            last_pid_ = std::nullopt;
        }
    }
}

bool AliceExecutor::is_allowed(const std::filesystem::path& path) const {
    return std::any_of(allowed_roots_.begin(), allowed_roots_.end(), [&](const auto& root) {
        return is_prefix_path(root, path);
    });
}

bool AliceExecutor::is_runnable_candidate(const std::filesystem::path& path) const {
    if (!std::filesystem::is_regular_file(path)) {
        return false;
    }
    const std::string ext = to_lower(path.extension().string());
    static const std::set<std::string> runnable = {".py", ".sh", ".bash", ".cpp", ".cc", ".cxx", ".c"};
    if (runnable.count(ext) > 0) {
        return true;
    }
    return ::access(path.c_str(), X_OK) == 0;
}

std::string AliceExecutor::display_path(const std::filesystem::path& path) const {
    const auto normalized = normalize_path(path);
    for (const auto& root : allowed_roots_) {
        if (is_prefix_path(root, normalized)) {
            std::error_code ec;
            auto rel = std::filesystem::relative(normalized, root, ec);
            if (!ec) {
                const std::string rel_text = rel.string();
                return rel_text.empty() || rel_text == "." ? "." : rel_text;
            }
        }
    }
    return normalized.string();
}

std::pair<std::optional<std::filesystem::path>, std::optional<std::string>> AliceExecutor::resolve_target(
    const std::optional<std::string>& raw_target, bool must_exist, std::optional<bool> expect_directory) const {
    std::filesystem::path target = raw_target.has_value() && !trim(*raw_target).empty() ? std::filesystem::path(trim(*raw_target)) : std::filesystem::path(".");

    if (target.is_relative()) {
        target = std::filesystem::current_path() / target;
    }
    target = normalize_path(target);

    if (must_exist && !std::filesystem::exists(target)) {
        return {std::nullopt, "Path does not exist: " + display_path(target)};
    }

    if (expect_directory.has_value() && std::filesystem::exists(target)) {
        if (*expect_directory && !std::filesystem::is_directory(target)) {
            return {std::nullopt, "Expected a folder but got: " + display_path(target)};
        }
        if (!*expect_directory && !std::filesystem::is_regular_file(target)) {
            return {std::nullopt, "Expected a file but got: " + display_path(target)};
        }
    }

    if (!is_allowed(target)) {
        std::vector<std::string> roots;
        roots.reserve(allowed_roots_.size());
        for (const auto& root : allowed_roots_) {
            roots.push_back(root.string());
        }
        return {std::nullopt, "Blocked by allowlist. Allowed roots: " + join(roots, ", ")};
    }

    return {target, std::nullopt};
}

void AliceExecutor::refresh_file_index() {
    indexed_files_.clear();
    static const std::set<std::string> skip_dirs = {".git", ".venv", "__pycache__", "node_modules", "third_party"};

    for (const auto& root : allowed_roots_) {
        if (!std::filesystem::exists(root)) {
            continue;
        }
        for (auto it = std::filesystem::recursive_directory_iterator(root);
             it != std::filesystem::recursive_directory_iterator(); ++it) {
            const auto& entry = *it;
            if (entry.is_directory()) {
                const std::string name = entry.path().filename().string();
                if (skip_dirs.count(name) > 0) {
                    it.disable_recursion_pending();
                }
                continue;
            }
            if (is_runnable_candidate(entry.path())) {
                indexed_files_.push_back(entry.path());
            }
        }
    }
}

std::string AliceExecutor::normalize_query(const std::string& text) {
    std::string lowered = to_lower(trim(text));
    lowered = replace_all(lowered, "vizualization", "viz");
    lowered = replace_all(lowered, "visualization", "viz");
    lowered = replace_all(lowered, "visualisation", "viz");
    lowered = replace_all(lowered, "black hole", "blackhole");
    return normalize_text(lowered);
}

std::vector<std::string> AliceExecutor::query_tokens(const std::string& text) {
    static const std::set<std::string> stop_words = {"the", "a", "an", "file", "script", "program", "please", "for", "me", "open", "run", "execute", "start", "launch"};
    std::vector<std::string> out;
    for (const auto& token : split_words(normalize_query(text))) {
        if (stop_words.count(token) == 0) {
            out.push_back(token);
        }
    }
    return out;
}

double AliceExecutor::score_candidate(const std::string& query, const std::filesystem::path& candidate) const {
    const std::string q_norm = normalize_query(query);
    const auto tokens = query_tokens(query);
    if (q_norm.empty() && tokens.empty()) {
        return 0.0;
    }

    std::string rel = candidate.string();
    for (const auto& root : allowed_roots_) {
        if (is_prefix_path(root, candidate)) {
            std::error_code ec;
            const auto rel_path = std::filesystem::relative(candidate, root, ec);
            if (!ec) {
                rel = rel_path.string();
            }
            break;
        }
    }

    const std::string rel_norm = normalize_query(rel);
    const std::string stem_norm = normalize_query(candidate.stem().string());

    auto compact = [](const std::string& value) {
        std::string out;
        out.reserve(value.size());
        for (char c : value) {
            if (!std::isspace(static_cast<unsigned char>(c))) {
                out.push_back(c);
            }
        }
        return out;
    };

    const std::string cq = compact(q_norm);
    const std::string cr = compact(rel_norm);
    const std::string cs = compact(stem_norm);

    double score = 0.0;
    if (!cq.empty() && cr.find(cq) != std::string::npos) {
        score += 10.0;
    }
    if (!cq.empty() && cs.find(cq) != std::string::npos) {
        score += 9.0;
    }

    for (const auto& token : tokens) {
        if (cs.find(token) != std::string::npos) {
            score += 3.0;
        } else if (cr.find(token) != std::string::npos) {
            score += 1.8;
        }
    }

    score -= std::min(1.5, static_cast<double>(rel_norm.size()) / 120.0);
    return score;
}

std::pair<std::optional<std::filesystem::path>, std::vector<std::filesystem::path>>
AliceExecutor::find_best_file_match(const std::string& raw_target) {
    refresh_file_index();
    if (indexed_files_.empty()) {
        return {std::nullopt, {}};
    }

    std::vector<std::pair<double, std::filesystem::path>> scored;
    scored.reserve(indexed_files_.size());
    for (const auto& path : indexed_files_) {
        scored.emplace_back(score_candidate(raw_target, path), path);
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    if (scored.empty() || scored.front().first < 3.6) {
        std::vector<std::filesystem::path> suggestions;
        for (std::size_t i = 0; i < scored.size() && i < 5; ++i) {
            suggestions.push_back(scored[i].second);
        }
        return {std::nullopt, suggestions};
    }

    std::vector<std::filesystem::path> suggestions;
    for (std::size_t i = 0; i < scored.size() && i < 5; ++i) {
        suggestions.push_back(scored[i].second);
    }
    return {scored.front().second, suggestions};
}

std::pair<std::optional<std::filesystem::path>, std::optional<std::string>>
AliceExecutor::resolve_runnable_target(const std::optional<std::string>& raw_target) {
    auto [path, error] = resolve_target(raw_target, false, false);
    if (error.has_value()) {
        return {std::nullopt, error};
    }
    if (path.has_value() && std::filesystem::exists(*path) && std::filesystem::is_regular_file(*path)) {
        if (is_runnable_candidate(*path)) {
            return {path, std::nullopt};
        }
        return {std::nullopt, "File exists but is not directly runnable: " + display_path(*path)};
    }

    if (!raw_target.has_value() || trim(*raw_target).empty()) {
        return {std::nullopt, "Missing file target."};
    }

    const auto [match, suggestions] = find_best_file_match(*raw_target);
    if (match.has_value()) {
        return {match, std::nullopt};
    }

    if (!suggestions.empty()) {
        std::vector<std::string> names;
        for (std::size_t i = 0; i < suggestions.size() && i < 3; ++i) {
            names.push_back(suggestions[i].filename().string());
        }
        return {std::nullopt, "Could not find a runnable file for '" + *raw_target + "'. Closest matches: " + join(names, ", ")};
    }

    return {std::nullopt, "Could not find a runnable file for '" + *raw_target + "'."};
}

std::optional<std::string> AliceExecutor::find_command(const std::string& name) {
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr) {
        return std::nullopt;
    }
    std::stringstream ss(path_env);
    std::string token;
    while (std::getline(ss, token, ':')) {
        if (token.empty()) {
            continue;
        }
        std::filesystem::path candidate = std::filesystem::path(token) / name;
        if (std::filesystem::exists(candidate) && ::access(candidate.c_str(), X_OK) == 0) {
            return candidate.string();
        }
    }
    return std::nullopt;
}

std::vector<std::string> AliceExecutor::split_env_flags(const char* value) {
    std::vector<std::string> out;
    if (value == nullptr) {
        return out;
    }
    std::istringstream stream(value);
    std::string token;
    while (stream >> token) {
        out.push_back(token);
    }
    return out;
}

std::string AliceExecutor::timestamp_for_filename() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_value{};
#if defined(_WIN32)
    localtime_s(&tm_value, &t);
#else
    localtime_r(&t, &tm_value);
#endif
    char buffer[32] = {0};
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm_value);
    return buffer;
}

bool AliceExecutor::run_blocking_command(const std::vector<std::string>& command, const std::filesystem::path& cwd,
                                         const std::filesystem::path& log_path) {
    std::vector<std::string> escaped;
    escaped.reserve(command.size());
    for (const auto& part : command) {
        escaped.push_back(shell_quote(part));
    }
    const std::string cmd = "cd " + shell_quote(cwd.string()) + " && " + join(escaped, " ") + " >> " +
                            shell_quote(log_path.string()) + " 2>&1";
    return std::system(cmd.c_str()) == 0;
}

std::optional<int> AliceExecutor::spawn_process(const std::vector<std::string>& command, const std::filesystem::path& cwd,
                                                const std::filesystem::path& log_path) const {
    if (command.empty()) {
        return std::nullopt;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        return std::nullopt;
    }
    if (pid == 0) {
        if (chdir(cwd.c_str()) != 0) {
            _exit(127);
        }
        const int fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            _exit(127);
        }
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        close(fd);

        std::vector<char*> argv;
        argv.reserve(command.size() + 1);
        for (const auto& part : command) {
            argv.push_back(const_cast<char*>(part.c_str()));
        }
        argv.push_back(nullptr);

        execvp(argv[0], argv.data());
        _exit(127);
    }
    return static_cast<int>(pid);
}

ExecResult AliceExecutor::list_files(const std::optional<std::string>& target) {
    const auto [path, error] = resolve_target(target, true, true);
    if (error.has_value()) {
        return ExecResult{false, *error};
    }

    std::vector<std::filesystem::directory_entry> entries;
    for (const auto& entry : std::filesystem::directory_iterator(*path)) {
        entries.push_back(entry);
    }
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        if (a.is_directory() != b.is_directory()) {
            return a.is_directory() > b.is_directory();
        }
        return to_lower(a.path().filename().string()) < to_lower(b.path().filename().string());
    });

    if (entries.empty()) {
        return ExecResult{true, display_path(*path) + " is empty."};
    }

    std::vector<std::string> lines;
    lines.reserve(std::min<std::size_t>(25, entries.size()) + 2);
    lines.push_back(display_path(*path));
    for (std::size_t i = 0; i < entries.size() && i < 25; ++i) {
        lines.push_back(entries[i].path().filename().string() + (entries[i].is_directory() ? "/" : ""));
    }
    if (entries.size() > 25) {
        lines.push_back("... (" + std::to_string(entries.size() - 25) + " more)");
    }

    return ExecResult{true, join(lines, "\n")};
}

ExecResult AliceExecutor::open_folder(const std::optional<std::string>& target) {
    const auto [path, error] = resolve_target(target, true, true);
    if (error.has_value()) {
        return ExecResult{false, *error};
    }
    return ExecResult{true, "Folder ready: " + display_path(*path)};
}

ExecResult AliceExecutor::run_file(const std::optional<std::string>& target) {
    cleanup_finished();

    const auto [path, error] = resolve_runnable_target(target);
    if (error.has_value()) {
        return ExecResult{false, *error};
    }
    if (!path.has_value()) {
        return ExecResult{false, "Missing runnable path."};
    }

    const std::string stamp = timestamp_for_filename();
    std::filesystem::create_directories(log_dir_);
    const std::filesystem::path log_path = log_dir_ / ("run_" + stamp + "_" + path->stem().string() + ".log");

    std::vector<std::string> run_command;
    const std::string ext = to_lower(path->extension().string());

    if (ext == ".py") {
        if (const auto py = find_command("python3"); py.has_value()) {
            run_command = {*py, path->string()};
        } else {
            return ExecResult{false, "python3 not found for .py file execution."};
        }
    } else if (ext == ".sh" || ext == ".bash") {
        if (const auto bash = find_command("bash"); bash.has_value()) {
            run_command = {*bash, path->string()};
        } else {
            return ExecResult{false, "bash not found for shell script execution."};
        }
    } else if (ext == ".cpp" || ext == ".cc" || ext == ".cxx") {
        std::string compiler = std::getenv("ALICE_CXX") ? std::getenv("ALICE_CXX") : "";
        if (compiler.empty()) {
            const auto clangpp = find_command("clang++");
            const auto gpp = find_command("g++");
            compiler = clangpp.value_or(gpp.value_or(""));
        }
        if (compiler.empty()) {
            return ExecResult{false, "C++ compiler not found. Install clang++ or g++."};
        }

        const std::filesystem::path bin_dir = log_dir_ / "bin";
        std::filesystem::create_directories(bin_dir);
        const std::filesystem::path bin_path = bin_dir / (path->stem().string() + "_" + stamp);

        std::vector<std::string> compile_command = {compiler, path->string(), "-std=c++20", "-O2"};
        const auto extra = split_env_flags(std::getenv("ALICE_CXXFLAGS"));
        compile_command.insert(compile_command.end(), extra.begin(), extra.end());
        const auto ld = split_env_flags(std::getenv("ALICE_LDFLAGS"));
        compile_command.insert(compile_command.end(), ld.begin(), ld.end());
        compile_command.push_back("-o");
        compile_command.push_back(bin_path.string());

        if (!run_blocking_command(compile_command, path->parent_path(), log_path)) {
            return ExecResult{false, "Compilation failed for " + path->filename().string() + ". Check " + log_path.filename().string() + " in Alice logs."};
        }
        run_command = {bin_path.string()};
    } else if (ext == ".c") {
        std::string compiler = std::getenv("ALICE_CC") ? std::getenv("ALICE_CC") : "";
        if (compiler.empty()) {
            const auto clang = find_command("clang");
            const auto gcc = find_command("gcc");
            compiler = clang.value_or(gcc.value_or(""));
        }
        if (compiler.empty()) {
            return ExecResult{false, "C compiler not found. Install clang or gcc."};
        }

        const std::filesystem::path bin_dir = log_dir_ / "bin";
        std::filesystem::create_directories(bin_dir);
        const std::filesystem::path bin_path = bin_dir / (path->stem().string() + "_" + stamp);

        std::vector<std::string> compile_command = {compiler, path->string(), "-O2"};
        const auto extra = split_env_flags(std::getenv("ALICE_CCFLAGS"));
        compile_command.insert(compile_command.end(), extra.begin(), extra.end());
        const auto ld = split_env_flags(std::getenv("ALICE_LDFLAGS"));
        compile_command.insert(compile_command.end(), ld.begin(), ld.end());
        compile_command.push_back("-o");
        compile_command.push_back(bin_path.string());

        if (!run_blocking_command(compile_command, path->parent_path(), log_path)) {
            return ExecResult{false, "Compilation failed for " + path->filename().string() + ". Check " + log_path.filename().string() + " in Alice logs."};
        }
        run_command = {bin_path.string()};
    } else {
        if (::access(path->c_str(), X_OK) != 0) {
            return ExecResult{false, "Unsupported file type. Use .py, .sh, .bash, .cpp, .cc, .cxx, .c, or executable files."};
        }
        run_command = {path->string()};
    }

    const auto pid = spawn_process(run_command, path->parent_path(), log_path);
    if (!pid.has_value()) {
        return ExecResult{false, "Failed to start process."};
    }

    processes_[*pid] = ProcessInfo{*pid, log_path};
    last_pid_ = *pid;
    return ExecResult{true, "Started " + path->filename().string() + " (pid " + std::to_string(*pid) + "). Logging to " + log_path.filename().string() + "."};
}

ExecResult AliceExecutor::stop_process(const std::optional<int>& pid) {
    cleanup_finished();

    const std::optional<int> target_pid = pid.has_value() ? pid : last_pid_;
    if (!target_pid.has_value()) {
        return ExecResult{false, "No tracked running process to stop."};
    }

    auto it = processes_.find(*target_pid);
    if (it == processes_.end()) {
        return ExecResult{false, "Process " + std::to_string(*target_pid) + " is not currently tracked."};
    }

    if (::kill(static_cast<pid_t>(*target_pid), SIGTERM) != 0) {
        if (errno == ESRCH) {
            processes_.erase(it);
            if (last_pid_.has_value() && *last_pid_ == *target_pid) {
                last_pid_ = std::nullopt;
            }
            return ExecResult{true, "Process " + std::to_string(*target_pid) + " already finished."};
        }
        return ExecResult{false, "Failed to stop process " + std::to_string(*target_pid) + "."};
    }

    int status = 0;
    waitpid(static_cast<pid_t>(*target_pid), &status, 0);

    processes_.erase(it);
    if (last_pid_.has_value() && *last_pid_ == *target_pid) {
        last_pid_ = std::nullopt;
    }

    return ExecResult{true, "Stopped process " + std::to_string(*target_pid) + "."};
}

void AliceExecutor::shutdown() {
    std::vector<int> pids;
    pids.reserve(processes_.size());
    for (const auto& [pid, _] : processes_) {
        pids.push_back(pid);
    }
    for (const int pid : pids) {
        stop_process(pid);
    }
}

}  // namespace alice
