#pragma once

#include <optional>
#include <string>

namespace alice {

struct Intent {
    std::string action;
    std::optional<std::string> target;
    std::optional<int> pid;
    bool requires_confirmation = false;
    std::string raw;
};

struct ExecResult {
    bool ok = false;
    std::string message;
};

}  // namespace alice
