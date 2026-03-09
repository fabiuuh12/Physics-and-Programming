#pragma once

#include <functional>
#include <optional>
#include <string>

namespace alice {

class VoiceListener {
public:
    VoiceListener();
    ~VoiceListener();

    VoiceListener(const VoiceListener&) = delete;
    VoiceListener& operator=(const VoiceListener&) = delete;

    [[nodiscard]] bool available() const;
    [[nodiscard]] std::string backend_name() const;
    [[nodiscard]] std::string last_error() const;

    std::optional<std::string> listen(double timeout_seconds = 6.0, double phrase_time_limit_seconds = 8.0,
                                      const std::function<void()>& tick = {},
                                      const std::function<void(const std::string&)>& on_partial_text = {});

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace alice
