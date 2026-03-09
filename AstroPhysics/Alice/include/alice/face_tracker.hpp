#pragma once

#include <optional>
#include <string>

namespace alice {

struct FaceObservation {
    bool found = false;
    float x = 0.0f;  // normalized [-1, 1]
    float y = 0.0f;  // normalized [-1, 1]
    int face_count = 0;
};

class FaceTracker {
public:
    FaceTracker();
    ~FaceTracker();

    FaceTracker(const FaceTracker&) = delete;
    FaceTracker& operator=(const FaceTracker&) = delete;

    bool start(int camera_index = 0);
    void stop();

    [[nodiscard]] bool running() const;
    [[nodiscard]] std::string last_error() const;
    [[nodiscard]] FaceObservation latest() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace alice
