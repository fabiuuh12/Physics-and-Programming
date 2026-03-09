#include "alice/voice_listener.hpp"

namespace alice {

struct VoiceListener::Impl {
    bool is_available = false;
    std::string backend = "none";
    std::string error = "Voice mode is unavailable on this platform.";
};

VoiceListener::VoiceListener() : impl_(new Impl()) {}
VoiceListener::~VoiceListener() { delete impl_; }

bool VoiceListener::available() const { return impl_->is_available; }
std::string VoiceListener::backend_name() const { return impl_->backend; }
std::string VoiceListener::last_error() const { return impl_->error; }

std::optional<std::string> VoiceListener::listen(double, double, const std::function<void()>&,
                                                 const std::function<void(const std::string&)>&) {
    return std::nullopt;
}

}  // namespace alice
