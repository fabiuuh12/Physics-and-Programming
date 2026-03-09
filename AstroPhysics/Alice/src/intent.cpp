#include "alice/intent.hpp"

#include "alice/string_utils.hpp"

#include <regex>
#include <set>

namespace alice {

static std::optional<std::string> strip_wake_phrase(const std::string& text, const std::string& wake_word) {
    const std::string cleaned = trim(text);
    const std::string lowered = to_lower(cleaned);
    const std::string wake = to_lower(trim(wake_word));
    const std::vector<std::string> candidates = {
        wake,
        "hey " + wake,
        "ok " + wake,
        "okay " + wake,
    };

    for (const auto& candidate : candidates) {
        if (!starts_with_ci(lowered, candidate)) {
            continue;
        }
        if (lowered.size() > candidate.size() && std::isalnum(static_cast<unsigned char>(lowered[candidate.size()]))) {
            continue;
        }
        return trim(cleaned.substr(candidate.size()));
    }
    return std::nullopt;
}

static std::string clean_target(const std::string& raw) {
    std::string t = trim(raw);
    if (!t.empty() && (t.front() == '\"' || t.front() == '\'')) {
        t.erase(t.begin());
    }
    if (!t.empty() && (t.back() == '\"' || t.back() == '\'' || t.back() == '.')) {
        t.pop_back();
    }
    const std::regex filler(R"(\b(for me|please|right now|now)\b)", std::regex::icase);
    t = std::regex_replace(t, filler, "");
    return trim(t);
}

Intent parse_intent(const std::string& text, const std::string& wake_word, bool require_wake, bool* matched) {
    if (matched != nullptr) {
        *matched = false;
    }

    const std::string spoken = trim(text);
    if (spoken.empty()) {
        return Intent{"chat", std::string{}, std::nullopt, false, text};
    }

    std::string command = spoken;
    const auto stripped = strip_wake_phrase(spoken, wake_word);
    if (require_wake && !stripped.has_value()) {
        return Intent{"skip", std::nullopt, std::nullopt, false, spoken};
    }
    if (stripped.has_value()) {
        command = *stripped;
    }
    if (matched != nullptr) {
        *matched = true;
    }

    const std::string lowered = to_lower(trim(command));
    if (lowered.empty()) {
        return Intent{"greet", std::nullopt, std::nullopt, false, spoken};
    }

    if (lowered == "help" || lowered == "what can you do" || lowered == "commands") {
        return Intent{"help", std::nullopt, std::nullopt, false, spoken};
    }

    std::smatch m;
    if (std::regex_match(command, m,
                         std::regex(R"((?:please\s+)?(?:remember|save|note|memorize|don't\s+forget|dont\s+forget)\s+(?:that\s+)?(.+))",
                                    std::regex::icase))) {
        std::string target = clean_target(m[1].str());
        if (!target.empty()) {
            return Intent{"remember_memory", target, std::nullopt, false, spoken};
        }
    }

    if (std::regex_match(command, m,
                         std::regex(R"((?:what\s+do\s+you\s+remember(?:\s+about)?|what\s+did\s+i\s+tell\s+you(?:\s+about)?|what\s+have\s+you\s+learned(?:\s+about)?|what\s+do\s+you\s+know(?:\s+about)?|recall|remember\s+about)\s*(.*))",
                                    std::regex::icase))) {
        std::string target = clean_target(m[1].str());
        if (target.empty()) {
            target = "me";
        }
        return Intent{"recall_memory", target, std::nullopt, false, spoken};
    }

    if (std::regex_search(
            lowered, std::regex(R"(\b(what\s+time\s+is\s+it|what('?s| is)?\s+the\s+time|tell\s+me\s+the\s+time|current\s+time|time\s+now|time\s+is\s+it|clock)\b)"))) {
        return Intent{"get_time", std::nullopt, std::nullopt, false, spoken};
    }
    if (std::regex_search(
            lowered, std::regex(R"(\b(what('?s| is)?\s+the\s+date|what\s+day\s+is\s+it|what\s+is\s+today('?s)?\s+date|today('?s| is)\s+date|today('?s| is)\s+day|current\s+date|date\s+today)\b)"))) {
        return Intent{"get_date", std::nullopt, std::nullopt, false, spoken};
    }

    if (std::regex_match(command, m,
                         std::regex(R"((?:search(?:\s+the\s+web)?(?:\s+for)?|look\s+up|google)\s+(.+))",
                                    std::regex::icase))) {
        std::string target = clean_target(m[1].str());
        if (!target.empty()) {
            return Intent{"web_search", target, std::nullopt, false, spoken};
        }
    }

    if (std::regex_match(command, m,
                         std::regex(R"((?:research(?:\s+the\s+web)?(?:\s+for)?|do\s+research\s+on|investigate|find\s+information\s+(?:on|about))\s+(.+))",
                                    std::regex::icase))) {
        std::string target = clean_target(m[1].str());
        if (!target.empty()) {
            return Intent{"web_research", target, std::nullopt, false, spoken};
        }
    }

    const std::set<std::string> smalltalk = {
        "hello", "hi", "hey", "sorry", "i'm sorry", "im sorry", "i am sorry", "my mistake",
        "i made a mistake", "never mind", "nevermind", "it's okay", "its okay", "good morning",
        "good afternoon", "good evening", "how are you", "who are you", "what is your name", "thanks",
        "thank you",
    };
    if (smalltalk.count(lowered) > 0) {
        return Intent{"smalltalk", lowered, std::nullopt, false, spoken};
    }

    if (lowered == "exit" || lowered == "quit" || lowered == "shutdown" || lowered == "stop listening" ||
        lowered == "goodbye" || lowered == "bye") {
        return Intent{"exit", std::nullopt, std::nullopt, false, spoken};
    }

    if (std::regex_match(command, m,
                         std::regex(R"((?:please\s+)?run\s+(?:this\s+)?(?:file\s+)?(.+))", std::regex::icase))) {
        const std::string target = clean_target(m[1].str());
        return Intent{"run_file", target.empty() ? std::optional<std::string>{} : std::optional<std::string>{target},
                      std::nullopt, true, spoken};
    }

    if (std::regex_match(command, m,
                         std::regex(R"((?:open|show)\s+(?:folder|directory)\s+(.+))", std::regex::icase))) {
        std::string target = clean_target(m[1].str());
        if (target.empty()) {
            target = ".";
        }
        return Intent{"open_folder", target, std::nullopt, false, spoken};
    }

    if (std::regex_match(command, m,
                         std::regex(R"((?:list|show)\s+(?:the\s+)?files(?:\s+in)?\s*(.*))", std::regex::icase))) {
        std::string target = clean_target(m[1].str());
        if (target.empty()) {
            target = ".";
        }
        return Intent{"list_files", target, std::nullopt, false, spoken};
    }

    if (std::regex_match(command, m,
                         std::regex(R"((?:stop|kill|terminate)\s*(?:process)?(?:\s+(\d+))?)", std::regex::icase))) {
        std::optional<int> pid;
        if (m[1].matched) {
            try {
                pid = std::stoi(m[1].str());
            } catch (...) {
                pid = std::nullopt;
            }
        }
        return Intent{"stop_process", std::nullopt, pid, true, spoken};
    }

    return Intent{"chat", trim(command), std::nullopt, false, spoken};
}

std::string describe_for_confirmation(const Intent& intent) {
    if (intent.action == "run_file") {
        return "run file '" + intent.target.value_or("(unknown)") + "'";
    }
    if (intent.action == "stop_process") {
        return intent.pid.has_value() ? "stop process " + std::to_string(*intent.pid) : "stop the latest process";
    }
    return intent.action;
}

bool parse_confirmation(const std::string& text, bool* known) {
    if (known != nullptr) {
        *known = false;
    }
    const std::string cleaned = normalize_text(text);
    if (cleaned.empty()) {
        return false;
    }

    const std::vector<std::string> yes_terms = {"yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "proceed", "go ahead"};
    const std::vector<std::string> no_terms = {"no", "nope", "nah", "cancel", "stop", "abort", "never mind", "nevermind"};

    bool has_yes = false;
    bool has_no = false;
    for (const auto& term : yes_terms) {
        if (cleaned.find(term) != std::string::npos) {
            has_yes = true;
            break;
        }
    }
    for (const auto& term : no_terms) {
        if (cleaned.find(term) != std::string::npos) {
            has_no = true;
            break;
        }
    }

    if (has_yes == has_no) {
        return false;
    }
    if (known != nullptr) {
        *known = true;
    }
    return has_yes;
}

}  // namespace alice
