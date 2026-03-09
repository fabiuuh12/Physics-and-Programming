#pragma once

#include <string>

#include "alice/types.hpp"

namespace alice {

Intent parse_intent(const std::string& text, const std::string& wake_word, bool require_wake, bool* matched = nullptr);
std::string describe_for_confirmation(const Intent& intent);
bool parse_confirmation(const std::string& text, bool* known);

}  // namespace alice
