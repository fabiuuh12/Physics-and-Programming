#include "alice/llm_client.hpp"

#include "alice/string_utils.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <unistd.h>

namespace alice {

static std::string env_or_default(const char* key, const char* fallback) {
    if (const char* value = std::getenv(key); value != nullptr && value[0] != '\0') {
        return value;
    }
    return fallback;
}

LLMClient::LLMClient() {
    ollama_host_ = env_or_default("ALICE_OLLAMA_HOST", "http://127.0.0.1:11434");
    ollama_model_ = env_or_default("ALICE_OLLAMA_MODEL", "qwen2.5:3b");
    openai_model_ = env_or_default("ALICE_OPENAI_MODEL", "gpt-4o-mini");

    std::string desired = to_lower(trim(env_or_default("ALICE_LLM_BACKEND", "auto")));
    if (desired != "auto" && desired != "ollama" && desired != "openai" && desired != "none") {
        desired = "auto";
    }

    if ((desired == "auto" || desired == "ollama") && ollama_available()) {
        backend_ = "ollama";
        return;
    }
    if (desired == "ollama") {
        backend_ = "none";
        return;
    }

    if ((desired == "auto" || desired == "openai") && openai_available()) {
        backend_ = "openai";
        return;
    }

    backend_ = "none";
}

std::string LLMClient::backend() const {
    return backend_;
}

bool LLMClient::available() const {
    return backend_ == "ollama" || backend_ == "openai";
}

bool LLMClient::ollama_available() const {
    const std::string command = "curl -sS --fail -m 2 " + shell_quote(ollama_host_ + "/api/tags") + " >/dev/null 2>&1";
    return std::system(command.c_str()) == 0;
}

bool LLMClient::openai_available() const {
    const char* api_key = std::getenv("OPENAI_API_KEY");
    return api_key != nullptr && api_key[0] != '\0';
}

std::string LLMClient::json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 16);
    for (unsigned char c : value) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out.push_back(static_cast<char>(c));
                break;
        }
    }
    return out;
}

std::string LLMClient::json_unescape(const std::string& value) {
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

std::string LLMClient::extract_json_string_field(const std::string& json, const std::string& field) {
    const std::regex rx("\\\"" + field + "\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"\\\\])*)\\\"");
    std::smatch m;
    if (!std::regex_search(json, m, rx)) {
        return "";
    }
    return json_unescape(m[1].str());
}

std::string LLMClient::run_capture(const std::string& command) {
    std::array<char, 4096> buffer{};
    std::string output;

    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return "";
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output.append(buffer.data());
    }
    pclose(pipe);
    return output;
}

std::string LLMClient::ollama_chat(const std::vector<ChatMessage>& messages, double temperature) {
    char tmp_name[] = "/tmp/alice_ollama_payload_XXXXXX.json";
    const int fd = mkstemps(tmp_name, 5);
    if (fd < 0) {
        return "";
    }
    close(fd);

    std::ostringstream payload;
    payload << "{\"model\":\"" << json_escape(ollama_model_) << "\",\"messages\":[";
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (i > 0) {
            payload << ",";
        }
        payload << "{\"role\":\"" << json_escape(messages[i].role) << "\",\"content\":\""
                << json_escape(messages[i].content) << "\"}";
    }
    payload << "],\"stream\":false,\"options\":{\"temperature\":" << temperature << "}}";

    if (!write_file(tmp_name, payload.str())) {
        std::filesystem::remove(tmp_name);
        return "";
    }

    const std::string command = "curl -sS --fail -m 25 -H 'Content-Type: application/json' -X POST " +
                                shell_quote(ollama_host_ + "/api/chat") + " --data-binary @" + shell_quote(tmp_name);
    const std::string response = run_capture(command);
    std::filesystem::remove(tmp_name);

    std::string content = extract_json_string_field(response, "content");
    return trim(content);
}

std::string LLMClient::openai_chat(const std::vector<ChatMessage>& messages, double temperature) {
    const char* api_key = std::getenv("OPENAI_API_KEY");
    if (api_key == nullptr || api_key[0] == '\0') {
        return "";
    }

    char tmp_name[] = "/tmp/alice_openai_payload_XXXXXX.json";
    const int fd = mkstemps(tmp_name, 5);
    if (fd < 0) {
        return "";
    }
    close(fd);

    std::ostringstream payload;
    payload << "{\"model\":\"" << json_escape(openai_model_) << "\",\"messages\":[";
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (i > 0) {
            payload << ",";
        }
        payload << "{\"role\":\"" << json_escape(messages[i].role) << "\",\"content\":\""
                << json_escape(messages[i].content) << "\"}";
    }
    payload << "],\"temperature\":" << temperature << ",\"max_tokens\":180}";

    if (!write_file(tmp_name, payload.str())) {
        std::filesystem::remove(tmp_name);
        return "";
    }

    const std::string command = "curl -sS --fail -m 30 -H 'Content-Type: application/json' -H " +
                                shell_quote(std::string("Authorization: Bearer ") + api_key) +
                                " -X POST https://api.openai.com/v1/chat/completions --data-binary @" +
                                shell_quote(tmp_name);

    const std::string response = run_capture(command);
    std::filesystem::remove(tmp_name);

    std::string content = extract_json_string_field(response, "content");
    return trim(content);
}

std::string LLMClient::chat(const std::vector<ChatMessage>& messages, double temperature) {
    if (backend_ == "ollama") {
        return ollama_chat(messages, temperature);
    }
    if (backend_ == "openai") {
        return openai_chat(messages, temperature);
    }
    return "";
}

}  // namespace alice
