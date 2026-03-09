#pragma once

#include <string>
#include <vector>

namespace alice {

struct ChatMessage {
    std::string role;
    std::string content;
};

class LLMClient {
public:
    LLMClient();

    [[nodiscard]] std::string backend() const;
    [[nodiscard]] bool available() const;

    std::string chat(const std::vector<ChatMessage>& messages, double temperature = 0.3);

private:
    std::string backend_ = "none";
    std::string ollama_host_;
    std::string ollama_model_;
    std::string openai_model_;

    bool ollama_available() const;
    bool openai_available() const;

    std::string ollama_chat(const std::vector<ChatMessage>& messages, double temperature);
    std::string openai_chat(const std::vector<ChatMessage>& messages, double temperature);

    static std::string json_escape(const std::string& value);
    static std::string json_unescape(const std::string& value);
    static std::string extract_json_string_field(const std::string& json, const std::string& field);
    static std::string run_capture(const std::string& command);
};

}  // namespace alice
