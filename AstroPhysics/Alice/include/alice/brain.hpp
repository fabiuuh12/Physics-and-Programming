#pragma once

#include <string>
#include <utility>
#include <vector>

#include "alice/llm_client.hpp"

namespace alice {

class AliceBrain {
public:
    AliceBrain();

    [[nodiscard]] bool using_llm() const;
    [[nodiscard]] std::string llm_backend() const;

    std::string reply(const std::string& text, const std::vector<std::string>& memories = {});

private:
    std::vector<std::pair<std::string, std::string>> history_;
    LLMClient llm_;

    std::string fallback_reply(const std::string& text) const;
};

}  // namespace alice
