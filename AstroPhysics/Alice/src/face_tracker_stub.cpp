#include "alice/face_tracker.hpp"

namespace alice {

struct FaceTracker::Impl {
    bool is_running = false;
    std::string error = "Face tracking unavailable on this platform.";
    FaceObservation observation;
};

FaceTracker::FaceTracker() : impl_(new Impl()) {}
FaceTracker::~FaceTracker() { delete impl_; }

bool FaceTracker::start(int) {
    impl_->is_running = false;
    return false;
}

void FaceTracker::stop() { impl_->is_running = false; }

bool FaceTracker::running() const { return impl_->is_running; }

std::string FaceTracker::last_error() const { return impl_->error; }

FaceObservation FaceTracker::latest() const { return impl_->observation; }

}  // namespace alice
