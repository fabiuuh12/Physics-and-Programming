#include "alice/brain.hpp"
#include "alice/config.hpp"
#include "alice/executor.hpp"
#include "alice/intent.hpp"
#include "alice/memory_store.hpp"
#include "alice/string_utils.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <regex>
#include <string>
#include <vector>

namespace alice {

static const std::string kHelpText =
    "Try commands like: run <file>, list files in <folder>, open folder <folder>, "
    "stop process, what time is it, what is today's date, remember that <fact>, "
    "what do you remember about <topic>, help, exit. Wake word is optional.";

struct Args {
    std::string mode = "text";
    bool require_wake = false;
    std::string wake_word = "alice";
    bool once = false;
    std::optional<std::string> command;
    std::filesystem::path config_path;
};

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

static void speak(const std::string& text) {
    std::cout << "Alice> " << text << std::endl;
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
        if (!value.empty() && ((value.front() == '\"' && value.back() == '\"') || (value.front() == '\'' && value.back() == '\''))) {
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

    while (!cleaned.empty() && (cleaned.back() == '.' || cleaned.back() == '!' || std::isspace(static_cast<unsigned char>(cleaned.back())))) {
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
    if (intent.action == "exit") {
        return ExecResult{true, "Shutting down."};
    }
    return ExecResult{false, "I did not understand that command. Say 'Alice help'."};
}

static bool handle_utterance(const std::string& utterance, const std::string& wake_word, bool require_wake,
                             AliceExecutor& executor, AliceBrain& brain, MemoryStore& memory_store) {
    bool matched = false;
    Intent intent = parse_intent(utterance, wake_word, require_wake, &matched);
    if (intent.action == "skip") {
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
            std::cout << "Confirm> " << std::flush;
            std::string confirmation;
            if (!std::getline(std::cin, confirmation)) {
                return false;
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

    if (args.mode != "text") {
        std::cout << "[Alice] Voice mode is not enabled in this C++ rewrite yet. Falling back to text mode." << std::endl;
    }

    const std::filesystem::path config_path = args.config_path.is_absolute() ? args.config_path : std::filesystem::current_path() / args.config_path;
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

    std::cout << "[Alice] TTS backend: none" << std::endl;
    std::cout << "[Alice] LLM backend: " << brain.llm_backend() << std::endl;
    if (brain.using_llm()) {
        speak("Alice is online with conversational mode enabled using " + brain.llm_backend() + ".");
    } else {
        speak("Alice is online. Advanced AI chat is unavailable, so I will use built-in responses.");
    }
    std::cout << "[Alice] NLU mode: rule-based intent parsing" << std::endl;
    std::cout << "[Alice] Memory items: " << memory_store.count() << " (" << memory_store.db_path().string() << ")" << std::endl;

    bool keep_running = true;
    while (keep_running) {
        std::string utterance;
        if (args.command.has_value()) {
            utterance = *args.command;
        } else {
            std::cout << "You> " << std::flush;
            if (!std::getline(std::cin, utterance)) {
                break;
            }
        }

        if (!trim(utterance).empty()) {
            keep_running = handle_utterance(utterance, args.wake_word, args.require_wake, executor, brain, memory_store);
        }

        if (args.once || args.command.has_value()) {
            break;
        }
    }

    executor.shutdown();
    return 0;
}

}  // namespace alice

int main(int argc, char** argv) {
    return alice::run(argc, argv);
}
