#include "alice/brain.hpp"
#include "alice/config.hpp"
#include "alice/executor.hpp"
#include "alice/intent.hpp"
#include "alice/memory_store.hpp"
#include "alice/string_utils.hpp"
#include "alice/ui.hpp"
#include "alice/voice_listener.hpp"
#include "alice/face_tracker.hpp"

#include <chrono>
#include <array>
#include <cerrno>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <spawn.h>
#include <regex>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <thread>

extern char** environ;

namespace alice {

static const std::string kHelpText =
    "Try commands like: run <file>, list files in <folder>, open folder <folder>, "
    "stop process, what time is it, what is today's date, search the web for <topic>, research <topic>, remember that <fact>, "
    "what do you remember about <topic>, help, exit. Wake word is optional.";

struct Args {
    std::string mode = "text";
    bool require_wake = false;
    std::string wake_word = "alice";
    bool once = false;
    std::optional<bool> ui;
    bool camera = true;
    int camera_index = 0;
    bool no_tts = false;
    std::optional<std::string> command;
    std::filesystem::path config_path;
};

static AliceUI* g_ui = nullptr;
static bool g_tts_enabled = false;
static std::mutex g_tts_mutex;
static pid_t g_tts_pid = -1;
static std::string g_tts_active_normalized;
static bool g_tts_barge_engaged = false;
static std::string g_recent_tts_normalized;
static std::chrono::steady_clock::time_point g_recent_tts_finished_at{};
static std::optional<std::string> g_tts_voice_name;
static int g_tts_rate_wpm = 185;

static bool command_exists(const std::string& name) {
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr) {
        return false;
    }
    std::string all(path_env);
    std::size_t start = 0;
    while (start <= all.size()) {
        std::size_t end = all.find(':', start);
        if (end == std::string::npos) {
            end = all.size();
        }
        const std::filesystem::path candidate = std::filesystem::path(all.substr(start, end - start)) / name;
        if (std::filesystem::exists(candidate) && ::access(candidate.c_str(), X_OK) == 0) {
            return true;
        }
        if (end == all.size()) {
            break;
        }
        start = end + 1;
    }
    return false;
}

static std::string smalltalk_reply(const std::optional<std::string>& topic) {
    const std::string key = to_lower(trim(topic.value_or("")));
    if (key == "hello" || key == "hi" || key == "hey") {
        return "Hey Fabio. I am here. How are you feeling today?";
    }
    if (key == "sorry" || key == "i'm sorry" || key == "im sorry" || key == "i am sorry" || key == "my mistake" ||
        key == "i made a mistake") {
        return "No problem at all. We can keep going.";
    }
    if (key == "never mind" || key == "nevermind") {
        return "No problem. We can switch topics.";
    }
    if (key == "good morning" || key == "good afternoon" || key == "good evening") {
        return "Hello Fabio. Good to hear from you.";
    }
    if (key == "how are you") {
        return "I am doing well. Curious and ready to help with your project.";
    }
    if (key == "who are you" || key == "what is your name") {
        return "I am Alice, your local AI assistant.";
    }
    if (key == "thanks" || key == "thank you") {
        return "You are welcome.";
    }
    return "I am here and listening.";
}

static std::string sanitize_spoken_text(const std::string& text) {
    std::string out = trim(text);
    out = std::regex_replace(out, std::regex(R"(https?://\S+)", std::regex::icase), " link ");
    out = replace_all(out, "|", ". ");
    out = replace_all(out, "_", " ");
    out = std::regex_replace(out, std::regex(R"(\s+)"), " ");
    return trim(out);
}

static std::string url_encode_query(const std::string& value) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(value.size() * 3);
    for (unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back(static_cast<char>(c));
        } else if (c == ' ') {
            out.push_back('+');
        } else {
            out.push_back('%');
            out.push_back(hex[(c >> 4) & 0x0F]);
            out.push_back(hex[c & 0x0F]);
        }
    }
    return out;
}

static std::string run_capture(const std::string& command) {
    std::array<char, 4096> buffer{};
    std::string output;

    FILE* pipe = ::popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return "";
    }
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output.append(buffer.data());
    }
    const int rc = ::pclose(pipe);
    if (rc != 0) {
        return "";
    }
    return output;
}

static std::string json_unescape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '\\' || i + 1 >= value.size()) {
            out.push_back(value[i]);
            continue;
        }
        const char next = value[i + 1];
        switch (next) {
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case '\\':
                out.push_back('\\');
                break;
            case '"':
                out.push_back('"');
                break;
            default:
                out.push_back(next);
                break;
        }
        ++i;
    }
    return out;
}

static std::string extract_json_string_field(const std::string& json, const std::string& field) {
    const std::regex rx("\\\"" + field + "\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"\\\\])*)\\\"");
    std::smatch match;
    if (!std::regex_search(json, match, rx)) {
        return "";
    }
    return trim(json_unescape(match[1].str()));
}

static std::vector<std::string> extract_json_text_entries(const std::string& json, std::size_t limit) {
    std::vector<std::string> out;
    const std::regex rx(R"("Text"\s*:\s*"((?:\\.|[^"\\])*)")");
    for (auto it = std::sregex_iterator(json.begin(), json.end(), rx); it != std::sregex_iterator(); ++it) {
        std::string text = trim(json_unescape((*it)[1].str()));
        if (text.empty()) {
            continue;
        }
        if (std::find(out.begin(), out.end(), text) != out.end()) {
            continue;
        }
        out.push_back(text);
        if (out.size() >= limit) {
            break;
        }
    }
    return out;
}

static std::string clamp_spoken_summary(const std::string& text, std::size_t max_chars = 380) {
    std::string cleaned = trim(text);
    if (cleaned.size() <= max_chars) {
        return cleaned;
    }
    cleaned = trim(cleaned.substr(0, max_chars));
    const std::size_t last_space = cleaned.find_last_of(" ");
    if (last_space != std::string::npos && last_space > max_chars / 2) {
        cleaned = cleaned.substr(0, last_space);
    }
    return trim(cleaned) + "...";
}

static ExecResult research_web(const std::string& query) {
    const std::string cleaned_query = trim(query);
    if (cleaned_query.empty()) {
        return ExecResult{false, "Tell me what topic to research."};
    }

    const std::string search_url = "https://duckduckgo.com/?q=" + url_encode_query(cleaned_query);
    const auto open_search = [&]() {
        if (!command_exists("open")) {
            return;
        }
        const std::string cmd = "open " + shell_quote(search_url) + " >/dev/null 2>&1 &";
        (void)std::system(cmd.c_str());
    };

    if (!command_exists("curl")) {
        open_search();
        return ExecResult{false, "I cannot fetch research directly right now, so I opened web results for " + cleaned_query + "."};
    }

    const std::string api_url =
        "https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1&q=" + url_encode_query(cleaned_query);
    const std::string command = "curl -sS --fail -L -m 18 " + shell_quote(api_url);
    const std::string response = run_capture(command);
    if (response.empty()) {
        open_search();
        return ExecResult{false, "I could not fetch research results right now, so I opened web results for " + cleaned_query + "."};
    }

    const std::string heading = extract_json_string_field(response, "Heading");
    const std::string abstract = extract_json_string_field(response, "AbstractText");
    const std::string answer = extract_json_string_field(response, "Answer");
    const std::string source_url = extract_json_string_field(response, "AbstractURL");
    const auto related = extract_json_text_entries(response, 4);

    std::vector<std::string> snippets;
    if (!answer.empty()) {
        snippets.push_back(answer);
    }
    if (!abstract.empty()) {
        snippets.push_back(abstract);
    }
    for (const auto& item : related) {
        if (snippets.size() >= 2) {
            break;
        }
        snippets.push_back(item);
    }

    if (snippets.empty()) {
        open_search();
        return ExecResult{false, "I found limited direct data, so I opened web results for " + cleaned_query + "."};
    }

    std::string message = "Here is what I found";
    const std::string title = heading.empty() ? cleaned_query : heading;
    message += " about " + title + ": " + clamp_spoken_summary(join(snippets, " "));
    if (!source_url.empty()) {
        message += " Source: " + source_url + ".";
    }
    return ExecResult{true, message};
}

static int tts_rate_from_env() {
    if (const char* raw = std::getenv("ALICE_TTS_RATE"); raw != nullptr && raw[0] != '\0') {
        try {
            const int value = std::stoi(raw);
            if (value >= 120 && value <= 280) {
                return value;
            }
        } catch (...) {
        }
    }
    return 185;
}

static std::vector<std::string> installed_say_voices() {
    std::vector<std::string> out;
    FILE* pipe = ::popen("say -v ?", "r");
    if (pipe == nullptr) {
        return out;
    }

    char buffer[512];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line = trim(buffer);
        if (line.empty()) {
            continue;
        }
        const std::size_t split = line.find_first_of(" \t");
        const std::string voice = split == std::string::npos ? line : line.substr(0, split);
        if (!voice.empty()) {
            out.push_back(voice);
        }
    }
    (void)::pclose(pipe);
    return out;
}

static std::optional<std::string> natural_voice_from_system() {
    if (const char* explicit_voice = std::getenv("ALICE_VOICE"); explicit_voice != nullptr && explicit_voice[0] != '\0') {
        return std::string(explicit_voice);
    }

    if (const char* raw = std::getenv("ALICE_TTS_AUTO_VOICE"); raw != nullptr && raw[0] != '\0') {
        const std::string value = to_lower(trim(raw));
        if (value == "0" || value == "false" || value == "no" || value == "off") {
            return std::nullopt;
        }
    }

    const auto installed = installed_say_voices();
    if (installed.empty()) {
        return std::nullopt;
    }

    static const std::vector<std::string> preferred = {
        "Samantha", "Alex", "Ava", "Victoria", "Karen", "Moira", "Daniel", "Nora", "Zoe",
    };

    for (const auto& name : preferred) {
        for (const auto& installed_name : installed) {
            if (installed_name == name) {
                return name;
            }
        }
    }
    return installed.front();
}

static void configure_tts() {
    g_tts_rate_wpm = tts_rate_from_env();
    g_tts_voice_name = natural_voice_from_system();
}

static void reap_tts_process_locked() {
    if (g_tts_pid <= 0) {
        return;
    }
    int status = 0;
    const pid_t rc = ::waitpid(g_tts_pid, &status, WNOHANG);
    if (rc == g_tts_pid || (rc < 0 && errno == ECHILD)) {
        if (!g_tts_active_normalized.empty()) {
            g_recent_tts_normalized = g_tts_active_normalized;
            g_recent_tts_finished_at = std::chrono::steady_clock::now();
        }
        g_tts_pid = -1;
        g_tts_active_normalized.clear();
        g_tts_barge_engaged = false;
    }
}

static void stop_tts_locked() {
    reap_tts_process_locked();
    if (g_tts_pid <= 0) {
        g_tts_active_normalized.clear();
        g_tts_barge_engaged = false;
        return;
    }

    const pid_t pid = g_tts_pid;
    (void)::kill(pid, SIGTERM);

    for (int attempt = 0; attempt < 20; ++attempt) {
        int status = 0;
        const pid_t rc = ::waitpid(pid, &status, WNOHANG);
        if (rc == pid || (rc < 0 && errno == ECHILD)) {
            if (!g_tts_active_normalized.empty()) {
                g_recent_tts_normalized = g_tts_active_normalized;
                g_recent_tts_finished_at = std::chrono::steady_clock::now();
            }
            g_tts_pid = -1;
            g_tts_active_normalized.clear();
            g_tts_barge_engaged = false;
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    (void)::kill(pid, SIGKILL);
    int status = 0;
    (void)::waitpid(pid, &status, 0);
    if (!g_tts_active_normalized.empty()) {
        g_recent_tts_normalized = g_tts_active_normalized;
        g_recent_tts_finished_at = std::chrono::steady_clock::now();
    }
    g_tts_pid = -1;
    g_tts_active_normalized.clear();
    g_tts_barge_engaged = false;
}

static void stop_tts() {
    std::lock_guard<std::mutex> lock(g_tts_mutex);
    stop_tts_locked();
}

static bool start_tts(const std::string& spoken) {
    std::lock_guard<std::mutex> lock(g_tts_mutex);
    stop_tts_locked();
    if (spoken.empty()) {
        return false;
    }

    std::vector<std::string> args_storage;
    args_storage.push_back("say");
    if (g_tts_voice_name.has_value() && !g_tts_voice_name->empty()) {
        args_storage.push_back("-v");
        args_storage.push_back(*g_tts_voice_name);
    }
    args_storage.push_back("-r");
    args_storage.push_back(std::to_string(g_tts_rate_wpm));
    args_storage.push_back(spoken);

    std::vector<char*> argv;
    argv.reserve(args_storage.size() + 1);
    for (auto& arg : args_storage) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);

    pid_t pid = -1;
    const int spawn_rc = ::posix_spawnp(&pid, "say", nullptr, nullptr, argv.data(), ::environ);
    if (spawn_rc != 0 || pid <= 0) {
        g_tts_pid = -1;
        g_tts_active_normalized.clear();
        g_tts_barge_engaged = false;
        return false;
    }

    g_tts_pid = pid;
    g_tts_active_normalized = normalize_text(spoken);
    g_tts_barge_engaged = false;
    return true;
}

static bool should_interrupt_tts_from_partial(const std::string& partial_raw) {
    const std::string partial = normalize_text(partial_raw);
    if (partial.empty()) {
        return false;
    }

    const auto partial_tokens = split_words(partial);
    if (partial_tokens.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_tts_mutex);
    reap_tts_process_locked();
    if (g_tts_pid <= 0 || g_tts_barge_engaged) {
        return false;
    }

    const bool explicit_interrupt =
        partial == "alice" || starts_with_ci(partial, "alice ") || partial == "stop" || partial == "wait" ||
        partial == "hold on" || partial == "listen";
    if (explicit_interrupt) {
        g_tts_barge_engaged = true;
        return true;
    }

    if (partial.size() < 8 || partial_tokens.size() < 2) {
        return false;
    }

    if (!g_tts_active_normalized.empty()) {
        if (g_tts_active_normalized.find(partial) != std::string::npos) {
            return false;
        }

        const auto active_tokens = split_words(g_tts_active_normalized);
        std::set<std::string> active_set(active_tokens.begin(), active_tokens.end());
        std::size_t overlap = 0;
        for (const auto& token : partial_tokens) {
            if (active_set.find(token) != active_set.end()) {
                ++overlap;
            }
        }

        const double overlap_ratio = static_cast<double>(overlap) / static_cast<double>(partial_tokens.size());
        if (overlap_ratio >= 0.72) {
            return false;
        }
    }

    g_tts_barge_engaged = true;
    return true;
}

static bool tts_echo_reference_snapshot(std::string* echo_reference) {
    std::lock_guard<std::mutex> lock(g_tts_mutex);
    reap_tts_process_locked();

    if (g_tts_pid > 0 && !g_tts_active_normalized.empty()) {
        *echo_reference = g_tts_active_normalized;
        return true;
    }

    if (!g_recent_tts_normalized.empty()) {
        const auto since_finish = std::chrono::steady_clock::now() - g_recent_tts_finished_at;
        if (since_finish < std::chrono::seconds(24)) {
            *echo_reference = g_recent_tts_normalized;
            return true;
        }
    }
    return false;
}

static bool is_probable_tts_echo(const std::string& text_raw) {
    std::string active;
    if (!tts_echo_reference_snapshot(&active) || active.empty()) {
        return false;
    }

    const std::string text = normalize_text(text_raw);
    if (text.size() < 6) {
        return false;
    }
    if (active.find(text) != std::string::npos) {
        return true;
    }

    const auto text_tokens = split_words(text);
    const auto active_tokens = split_words(active);
    if (text_tokens.empty() || active_tokens.empty()) {
        return false;
    }

    std::set<std::string> active_set(active_tokens.begin(), active_tokens.end());
    std::size_t overlap = 0;
    for (const auto& token : text_tokens) {
        if (active_set.find(token) != active_set.end()) {
            ++overlap;
        }
    }

    const double overlap_ratio = static_cast<double>(overlap) / static_cast<double>(text_tokens.size());
    return overlap >= 3 && overlap_ratio >= 0.65;
}

static bool is_probable_noise_utterance(const std::string& text_raw) {
    const std::string normalized = normalize_text(text_raw);
    if (normalized.empty()) {
        return true;
    }
    const auto tokens = split_words(normalized);
    if (tokens.empty()) {
        return true;
    }
    if (tokens.size() == 1) {
        const std::string t = tokens[0];
        if (t.size() <= 2) {
            return true;
        }
        if (t == "um" || t == "uh" || t == "hmm" || t == "huh" || t == "mm") {
            return true;
        }
    }
    return false;
}

static bool barge_in_enabled() {
    if (const char* raw = std::getenv("ALICE_BARGE_IN"); raw != nullptr && raw[0] != '\0') {
        const std::string value = to_lower(trim(raw));
        return !(value == "0" || value == "false" || value == "no" || value == "off");
    }
    return true;
}

static double stt_chunk_timeout_seconds() {
    if (const char* raw = std::getenv("ALICE_STT_CHUNK_TIMEOUT"); raw != nullptr && raw[0] != '\0') {
        try {
            const double value = std::stod(raw);
            if (value >= 1.5 && value <= 20.0) {
                return value;
            }
        } catch (...) {
        }
    }
    return 8.0;
}

static void speak(const std::string& text) {
    std::cout << "Alice> " << text << std::endl;

    if (g_ui != nullptr) {
        g_ui->add_message("Alice", text);
        g_ui->set_state("speaking");
        g_ui->set_status("Speaking...");
        g_ui->pump();
    }

    if (g_tts_enabled) {
        const std::string spoken = sanitize_spoken_text(text);
        (void)start_tts(spoken);
    }

    if (g_ui != nullptr) {
        g_ui->set_state("idle");
        g_ui->set_status("Online");
    }
}

static bool read_line_with_ui(std::string& out) {
    while (true) {
        if (g_ui != nullptr) {
            g_ui->pump();
            if (!g_ui->running()) {
                return false;
            }
        }

        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(STDIN_FILENO, &readfds);
        timeval timeout{};
        timeout.tv_sec = 0;
        timeout.tv_usec = 20000;

        const int rc = select(STDIN_FILENO + 1, &readfds, nullptr, nullptr, &timeout);
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (rc == 0) {
            continue;
        }

        return static_cast<bool>(std::getline(std::cin, out));
    }
}

static bool read_voice_with_ui(VoiceListener& listener, std::string& out, double timeout_seconds = 6.0,
                               double phrase_time_limit_seconds = 8.0) {
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::milliseconds(static_cast<int>(timeout_seconds * 1000.0));

    while (std::chrono::steady_clock::now() < deadline) {
        const auto now = std::chrono::steady_clock::now();
        const double remaining_seconds =
            std::chrono::duration<double>(deadline - now).count();
        if (remaining_seconds <= 0.0) {
            break;
        }

        const double chunk_timeout = std::min(stt_chunk_timeout_seconds(), remaining_seconds);
        const auto maybe_text = listener.listen(
            chunk_timeout,
            phrase_time_limit_seconds,
            []() {
                if (g_ui != nullptr) {
                    g_ui->pump();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(8));
            },
            [](const std::string& partial_text) {
                if (!barge_in_enabled()) {
                    return;
                }
                if (should_interrupt_tts_from_partial(partial_text)) {
                    stop_tts();
                }
            });
        if (!maybe_text.has_value()) {
            continue;
        }
        if (is_probable_tts_echo(*maybe_text)) {
            continue;
        }
        if (is_probable_noise_utterance(*maybe_text)) {
            continue;
        }
        out = *maybe_text;
        return true;
    }

    out.clear();
    return false;
}

static void load_env_file(const std::filesystem::path& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }

        std::string key = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));
        if (key.empty()) {
            continue;
        }
        if (!value.empty() &&
            ((value.front() == '"' && value.back() == '"') || (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }

        if (std::getenv(key.c_str()) == nullptr) {
            setenv(key.c_str(), value.c_str(), 0);
        }
    }
}

static void load_project_env(const std::filesystem::path& project_root) {
    load_env_file(project_root / ".env");
    load_env_file(project_root / ".env.local");
}

static Args parse_args(int argc, char** argv) {
    Args args;
    args.config_path = std::filesystem::current_path() / "config" / "allowed_paths.json";

    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        if (token == "--mode" && i + 1 < argc) {
            args.mode = argv[++i];
            continue;
        }
        if (token == "--require-wake") {
            args.require_wake = true;
            continue;
        }
        if (token == "--wake-word" && i + 1 < argc) {
            args.wake_word = argv[++i];
            continue;
        }
        if (token == "--once") {
            args.once = true;
            continue;
        }
        if (token == "--command" && i + 1 < argc) {
            args.command = argv[++i];
            continue;
        }
        if (token == "--config" && i + 1 < argc) {
            args.config_path = argv[++i];
            continue;
        }
        if (token == "--ui") {
            args.ui = true;
            continue;
        }
        if (token == "--no-ui") {
            args.ui = false;
            continue;
        }
        if (token == "--no-camera") {
            args.camera = false;
            continue;
        }
        if (token == "--camera-index" && i + 1 < argc) {
            try {
                args.camera_index = std::stoi(argv[++i]);
            } catch (...) {
                args.camera_index = 0;
            }
            continue;
        }
        if (token == "--no-tts") {
            args.no_tts = true;
            continue;
        }
    }

    return args;
}

static std::optional<std::string> extract_memorable_fact(const std::string& text) {
    std::string cleaned = trim(text);
    if (cleaned.empty() || cleaned.find('?') != std::string::npos || cleaned.size() < 8 || cleaned.size() > 180) {
        return std::nullopt;
    }
    const std::string lowered = to_lower(cleaned);

    if (starts_with_ci(lowered, "i think ") || starts_with_ci(lowered, "i guess ") || starts_with_ci(lowered, "maybe ") ||
        starts_with_ci(lowered, "probably ")) {
        return std::nullopt;
    }

    static const std::vector<std::regex> patterns = {
        std::regex(R"(^my\s+(name|birthday|goal|project|major|school|city|hometown)\s+(is|are)\s+.+$)", std::regex::icase),
        std::regex(R"(^my\s+favorite\s+[a-z][a-z\s]{1,20}\s+(is|are)\s+.+$)", std::regex::icase),
        std::regex(R"(^i('m| am)\s+working\s+on\s+.+$)", std::regex::icase),
        std::regex(R"(^i('m| am)\s+(from|in|studying|learning|building)\s+.+$)", std::regex::icase),
        std::regex(R"(^i\s+(study|work\s+on|build|use)\s+.+$)", std::regex::icase),
        std::regex(R"(^i\s+(like|love|prefer|enjoy|hate)\s+.+$)", std::regex::icase),
        std::regex(R"(^call\s+me\s+.+$)", std::regex::icase),
    };

    bool matches = false;
    for (const auto& pattern : patterns) {
        if (std::regex_match(cleaned, pattern)) {
            matches = true;
            break;
        }
    }
    if (!matches) {
        return std::nullopt;
    }

    while (!cleaned.empty() &&
           (cleaned.back() == '.' || cleaned.back() == '!' || std::isspace(static_cast<unsigned char>(cleaned.back())))) {
        cleaned.pop_back();
    }
    return cleaned;
}

static std::string format_recalled_memories(const std::string& query, const std::vector<MemoryItem>& memories) {
    if (memories.empty()) {
        return "I do not have a saved memory about '" + query + "' yet.";
    }
    std::vector<std::string> lines;
    for (const auto& memory : memories) {
        lines.push_back(memory.content);
    }
    return "Here is what I remember about " + query + ": " + join(lines, " | ");
}

static ExecResult execute_intent(const Intent& intent, AliceExecutor& executor) {
    if (intent.action == "help") {
        return ExecResult{true, kHelpText};
    }
    if (intent.action == "greet") {
        return ExecResult{true, "I am listening."};
    }
    if (intent.action == "smalltalk") {
        return ExecResult{true, smalltalk_reply(intent.target)};
    }
    if (intent.action == "list_files") {
        return executor.list_files(intent.target);
    }
    if (intent.action == "open_folder") {
        return executor.open_folder(intent.target);
    }
    if (intent.action == "run_file") {
        return executor.run_file(intent.target);
    }
    if (intent.action == "stop_process") {
        return executor.stop_process(intent.pid);
    }
    if (intent.action == "get_time") {
        return ExecResult{true, "It is " + format_clock_time() + "."};
    }
    if (intent.action == "get_date") {
        return ExecResult{true, "Today is " + format_long_date() + "."};
    }
    if (intent.action == "web_search") {
        const std::string query = trim(intent.target.value_or(""));
        if (query.empty()) {
            return ExecResult{false, "Tell me what to search for."};
        }
        if (!command_exists("open")) {
            return ExecResult{false, "I cannot open a browser on this system."};
        }

        const std::string url = "https://duckduckgo.com/?q=" + url_encode_query(query);
        const std::string cmd = "open " + shell_quote(url) + " >/dev/null 2>&1 &";
        const int rc = std::system(cmd.c_str());
        if (rc != 0) {
            return ExecResult{false, "I could not open web search right now."};
        }
        return ExecResult{true, "Opened web search results for " + query + "."};
    }
    if (intent.action == "web_research") {
        return research_web(intent.target.value_or(""));
    }
    if (intent.action == "exit") {
        return ExecResult{true, "Shutting down."};
    }
    return ExecResult{false, "I did not understand that command. Say 'Alice help'."};
}

static bool handle_utterance(const std::string& utterance, const std::string& wake_word, bool require_wake,
                             AliceExecutor& executor, AliceBrain& brain, MemoryStore& memory_store,
                             bool voice_mode, VoiceListener* voice_listener) {
    if (g_ui != nullptr) {
        g_ui->set_state("thinking");
        g_ui->set_status("Thinking...");
    }

    bool matched = false;
    Intent intent = parse_intent(utterance, wake_word, require_wake, &matched);
    if (intent.action == "skip") {
        if (g_ui != nullptr) {
            g_ui->set_state("idle");
            g_ui->set_status("Online");
        }
        return true;
    }

    if (intent.action == "remember_memory") {
        const std::string fact = trim(intent.target.value_or(""));
        if (fact.empty()) {
            speak("Tell me what to remember.");
        } else if (memory_store.add(fact, "profile")) {
            speak("Saved. I will remember that.");
        } else {
            speak("I already remember that.");
        }
        return true;
    }

    if (intent.action == "recall_memory") {
        const std::string query = trim(intent.target.value_or("me")).empty() ? "me" : trim(intent.target.value_or("me"));
        const auto recalled = memory_store.search(query, 5);
        speak(format_recalled_memories(query, recalled));
        return true;
    }

    if (intent.action == "chat") {
        const std::string chat_text = intent.target.value_or(intent.raw);
        const auto related = memory_store.search(chat_text, 4);
        std::vector<std::string> memory_lines;
        for (const auto& item : related) {
            memory_lines.push_back(item.content);
        }
        speak(brain.reply(chat_text, memory_lines));

        const auto auto_fact = extract_memorable_fact(chat_text);
        if (auto_fact.has_value()) {
            memory_store.add(*auto_fact, "profile");
        }
        return true;
    }

    if (intent.requires_confirmation) {
        speak("Please confirm: " + describe_for_confirmation(intent) + ". Say yes or no.");
        bool decision_known = false;
        bool decision = false;
        bool approved = false;

        for (int attempt = 0; attempt < 3; ++attempt) {
            if (g_ui != nullptr) {
                g_ui->set_state("listening");
                g_ui->set_status("Waiting for confirmation...");
            }
            std::string confirmation;
            if (voice_mode && voice_listener != nullptr && voice_listener->available()) {
                std::cout << "Confirm (voice)> " << std::flush;
                const bool captured = read_voice_with_ui(*voice_listener, confirmation, 10.0, 12.0);
                if (!captured) {
                    const std::string reason = trim(voice_listener->last_error());
                    if (!reason.empty()) {
                        std::cout << "[no speech: " << reason << "]" << std::endl;
                    } else {
                        std::cout << "[no speech]" << std::endl;
                    }
                    speak("I did not catch yes or no. Please say yes or no.");
                    continue;
                }
                std::cout << confirmation << std::endl;
            } else {
                std::cout << "Confirm> " << std::flush;
                if (!read_line_with_ui(confirmation)) {
                    return false;
                }
            }
            if (g_ui != nullptr && !trim(confirmation).empty()) {
                g_ui->add_message("You", confirmation);
            }
            decision = parse_confirmation(confirmation, &decision_known);
            if (!decision_known) {
                speak("I did not catch yes or no. Please say yes or no.");
                continue;
            }
            approved = decision;
            break;
        }

        if (!decision_known || !approved) {
            speak("Canceled.");
            return true;
        }
    }

    const ExecResult result = execute_intent(intent, executor);
    speak(result.message);
    return intent.action != "exit";
}

int run(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const bool voice_mode = (to_lower(trim(args.mode)) == "voice");
    const bool ui_enabled = args.ui.has_value() ? *args.ui : voice_mode;

    const std::filesystem::path config_path =
        args.config_path.is_absolute() ? args.config_path : std::filesystem::current_path() / args.config_path;
    const AliceConfig config = load_config(config_path);
    load_project_env(config.project_root);

    std::filesystem::path memory_path;
    if (const char* env_memory = std::getenv("ALICE_MEMORY_DB"); env_memory != nullptr && env_memory[0] != '\0') {
        memory_path = env_memory;
        if (memory_path.extension() == ".db") {
            memory_path.replace_extension(".tsv");
        }
    } else {
        memory_path = config.project_root / "data" / "alice_memory.tsv";
    }

    MemoryStore memory_store(memory_path);
    AliceExecutor executor(config.allowed_roots, config.log_dir, config.max_runtime_seconds);
    AliceBrain brain;
    std::unique_ptr<VoiceListener> voice_listener;
    std::unique_ptr<FaceTracker> face_tracker;
    if (voice_mode) {
        voice_listener = std::make_unique<VoiceListener>();
    }

    AliceUI ui;
    if (ui_enabled) {
        if (ui.start()) {
            g_ui = &ui;
            g_ui->set_state("idle");
            g_ui->set_status("Online");
            std::cout << "[Alice] UI: enabled" << std::endl;
        } else {
            std::cout << "[Alice] UI: unavailable on this platform/runtime." << std::endl;
        }
    }

    if (ui_enabled && args.camera) {
        face_tracker = std::make_unique<FaceTracker>();
        if (face_tracker->start(args.camera_index)) {
            std::cout << "[Alice] Camera tracking: enabled (camera " << args.camera_index << ")" << std::endl;
        } else {
            const std::string camera_error = face_tracker->last_error();
            std::cout << "[Alice] Camera tracking: unavailable (" << camera_error << ")" << std::endl;
            const std::string lowered = to_lower(camera_error);
            if (lowered.find("permission denied") != std::string::npos) {
                std::cout << "[Alice] Tip: enable Camera for Terminal/iTerm in System Settings > Privacy & Security > Camera." << std::endl;
            }
            if (g_ui != nullptr) {
                g_ui->set_state("error");
                g_ui->set_status("Camera unavailable: " + camera_error);
                g_ui->pump();
            }
            face_tracker.reset();
        }
    } else if (ui_enabled && !args.camera && g_ui != nullptr) {
        g_ui->set_status("Camera disabled");
    }

    g_tts_enabled = !args.no_tts && command_exists("say");
    if (g_tts_enabled) {
        configure_tts();
    }

    std::cout << "[Alice] STT backend: ";
    if (voice_mode) {
        if (voice_listener != nullptr && voice_listener->available()) {
            std::cout << voice_listener->backend_name() << std::endl;
        } else {
            std::cout << "none (" << (voice_listener ? voice_listener->last_error() : "voice listener unavailable")
                      << ")" << std::endl;
            std::cout << "[Alice] Voice mode unavailable. Falling back to text input." << std::endl;
        }
    } else {
        std::cout << "disabled" << std::endl;
    }

    if (g_tts_enabled) {
        std::cout << "[Alice] TTS backend: say";
        if (g_tts_voice_name.has_value() && !g_tts_voice_name->empty()) {
            std::cout << " (" << *g_tts_voice_name << ", " << g_tts_rate_wpm << " wpm)";
        } else {
            std::cout << " (" << g_tts_rate_wpm << " wpm)";
        }
        std::cout << std::endl;
    } else {
        std::cout << "[Alice] TTS backend: none" << std::endl;
    }
    std::cout << "[Alice] LLM backend: " << brain.llm_backend() << std::endl;
    if (brain.using_llm()) {
        speak("Alice is online with conversational mode enabled using " + brain.llm_backend() + ".");
    } else {
        speak("Alice is online. Advanced AI chat is unavailable, so I will use built-in responses.");
    }
    std::cout << "[Alice] NLU mode: rule-based intent parsing" << std::endl;
    std::cout << "[Alice] Memory items: " << memory_store.count() << " (" << memory_store.db_path().string() << ")"
              << std::endl;

    bool keep_running = true;
    while (keep_running) {
        if (g_ui != nullptr && face_tracker != nullptr) {
            const FaceObservation obs = face_tracker->latest();
            g_ui->set_face_target(obs.x, obs.y, obs.found, obs.face_count);
        }
        if (g_ui != nullptr) {
            g_ui->pump();
            if (!g_ui->running()) {
                std::cout << "[Alice] UI window closed. Exiting." << std::endl;
                break;
            }
        }

        std::string utterance;
        if (args.command.has_value()) {
            utterance = *args.command;
        } else {
            if (g_ui != nullptr) {
                g_ui->set_state("listening");
                g_ui->set_status(voice_mode ? "Listening to microphone..." : "Listening...");
            }

            if (voice_mode && voice_listener != nullptr && voice_listener->available()) {
                std::cout << "You (voice)> " << std::flush;
                const bool captured = read_voice_with_ui(*voice_listener, utterance, 16.0, 24.0);
                if (!captured) {
                    const std::string reason = trim(voice_listener->last_error());
                    if (!reason.empty()) {
                        std::cout << "[no speech: " << reason << "]" << std::endl;
                    } else {
                        std::cout << "[no speech]" << std::endl;
                    }
                    if (args.once) {
                        break;
                    }
                    continue;
                }
                std::cout << utterance << std::endl;
            } else {
                std::cout << "You> " << std::flush;
                if (!read_line_with_ui(utterance)) {
                    break;
                }
            }
        }

        if (!trim(utterance).empty()) {
            if (g_ui != nullptr) {
                g_ui->add_message("You", utterance);
            }
            keep_running =
                handle_utterance(utterance, args.wake_word, args.require_wake, executor, brain, memory_store,
                                 voice_mode, voice_listener.get());
        }

        if (args.once || args.command.has_value()) {
            break;
        }
    }

    executor.shutdown();
    stop_tts();
    if (face_tracker != nullptr) {
        face_tracker->stop();
    }
    if (g_ui != nullptr) {
        g_ui->set_state("offline");
        g_ui->set_status("Offline");
        g_ui = nullptr;
        ui.stop();
    }

    return 0;
}

}  // namespace alice

int main(int argc, char** argv) {
    return alice::run(argc, argv);
}
