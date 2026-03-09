#include "alice/brain.hpp"

#include "alice/string_utils.hpp"

#include <algorithm>

namespace alice {

AliceBrain::AliceBrain() = default;

bool AliceBrain::using_llm() const {
    return llm_.available();
}

std::string AliceBrain::llm_backend() const {
    return llm_.backend();
}

std::string AliceBrain::fallback_reply(const std::string& text) const {
    const std::string lowered = normalize_text(text);
    if (lowered.find("hello") != std::string::npos || lowered == "hi" || lowered == "hey") {
        return "Hello Fabio. I am here with you. How is your day going?";
    }
    if (lowered.find("sorry") != std::string::npos || lowered.find("never mind") != std::string::npos) {
        return "No problem at all. We can keep going.";
    }
    if (lowered.find("how are you") != std::string::npos) {
        return "I am doing well and ready to help.";
    }
    if (lowered.find("what can you do") != std::string::npos || lowered == "help") {
        return "I can chat with you and run local commands like listing files or running scripts with confirmation.";
    }
    if (lowered.find("time") != std::string::npos) {
        return "It is " + format_clock_time() + ".";
    }
    if (lowered.find("date") != std::string::npos || lowered.find("day") != std::string::npos) {
        return "Today is " + format_long_date() + ".";
    }
    return "I hear you. I can chat naturally, and I can also run files or manage folders when asked.";
}

std::string AliceBrain::reply(const std::string& text, const std::vector<std::string>& memories) {
    if (!llm_.available()) {
        const std::string answer = fallback_reply(text);
        history_.push_back({text, answer});
        if (history_.size() > 12) {
            history_.erase(history_.begin(), history_.begin() + static_cast<long>(history_.size() - 12));
        }
        return answer;
    }

    std::vector<ChatMessage> messages;
    messages.push_back(ChatMessage{
        "system",
        "You are Alice, a concise voice assistant for Fabio. Be natural, useful, and brief. Do not claim actions you did not perform."});

    if (!memories.empty()) {
        const std::vector<std::string> limited(memories.begin(), memories.begin() + std::min<std::size_t>(memories.size(), 6));
        messages.push_back(ChatMessage{"system", "Relevant long-term memory: " + join(limited, " ; ")});
    }

    const std::size_t history_start = history_.size() > 4 ? history_.size() - 4 : 0;
    for (std::size_t i = history_start; i < history_.size(); ++i) {
        messages.push_back(ChatMessage{"user", history_[i].first});
        messages.push_back(ChatMessage{"assistant", history_[i].second});
    }
    messages.push_back(ChatMessage{"user", text});

    std::string answer = trim(llm_.chat(messages, 0.4));
    if (answer.empty()) {
        answer = fallback_reply(text);
    }

    history_.push_back({text, answer});
    if (history_.size() > 12) {
        history_.erase(history_.begin(), history_.begin() + static_cast<long>(history_.size() - 12));
    }
    return answer;
}

}  // namespace alice
