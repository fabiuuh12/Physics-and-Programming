#include "alice/ui.hpp"

namespace alice {

struct AliceUI::Impl {
    bool is_running = false;
};

AliceUI::AliceUI() : impl_(std::make_unique<Impl>()) {}
AliceUI::~AliceUI() = default;

bool AliceUI::start() {
    impl_->is_running = false;
    return false;
}

void AliceUI::pump() {}

void AliceUI::stop() {
    impl_->is_running = false;
}

bool AliceUI::running() const {
    return impl_->is_running;
}

void AliceUI::set_state(const std::string&) {}
void AliceUI::set_status(const std::string&) {}
void AliceUI::add_message(const std::string&, const std::string&) {}
void AliceUI::set_face_target(float, float, bool, int) {}

}  // namespace alice
