#pragma once

#include <memory>
#include <string>

namespace alice {

class AliceUI {
public:
    AliceUI();
    ~AliceUI();

    AliceUI(const AliceUI&) = delete;
    AliceUI& operator=(const AliceUI&) = delete;

    bool start();
    void pump();
    void stop();
    [[nodiscard]] bool running() const;

    void set_state(const std::string& state);
    void set_status(const std::string& status);
    void add_message(const std::string& speaker, const std::string& text);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace alice
