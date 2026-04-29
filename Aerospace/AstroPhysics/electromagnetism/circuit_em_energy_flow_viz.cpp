#include "../vision/hand_tracking_scene_shared.h"

#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1360;
constexpr int kScreenHeight = 860;
constexpr float kCameraDistanceMin = 1.25f;
constexpr float kCameraDistanceMax = 24.0f;
constexpr float kBridgeLiveZoomMin = 0.05f;
constexpr float kBridgeLiveZoomMax = 2.60f;
constexpr float kCameraPitchMin = -3.05f;
constexpr float kCameraPitchMax = 3.05f;
constexpr std::int64_t kControlStaleMs = 1200;
constexpr std::int64_t kLeftPinchWindowMs = 450;
constexpr std::int64_t kRightPinchConfirmMs = 240;
constexpr float kSignalSpeedMin = 1.0f;
constexpr float kSignalSpeedMax = 8.0f;
constexpr float kSignalSpeedSlowStep = 0.45f;
constexpr float kSignalSpeedFastStep = 0.85f;

enum class ViewMode {
    Normal = 0,
    Field = 1,
    Cutaway = 2,
};

enum class FaultMode {
    None = 0,
    OpenNeutral = 1,
    ShortCircuit = 2,
    MissingGround = 3,
    Overload = 4,
    BreakerTrip = 5,
};

struct LiveControls {
    float zoom = 1.0f;
    float rotationDeg = 0.0f;
    float pitchDeg = 0.0f;
    int nIncCount = 0;
    int nDecCount = 0;
    bool zoomLineActive = false;
    std::string label = "Unknown";
    std::string gesture = "none";
    std::int64_t timestampMs = 0;
};

std::string Trim(std::string s) {
    auto notSpace = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}

std::int64_t UnixMsNow() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

float NormalizeDeg(float deg) {
    while (deg > 180.0f) deg -= 360.0f;
    while (deg < -180.0f) deg += 360.0f;
    return deg;
}

float AngleDeltaDeg(float current, float previous) {
    return NormalizeDeg(current - previous);
}

std::optional<LiveControls> ParseLiveControlsFile(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) return std::nullopt;

    LiveControls lc;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = Trim(line.substr(0, eq));
        const std::string val = Trim(line.substr(eq + 1));
        try {
            if (key == "zoom") lc.zoom = std::stof(val);
            else if (key == "rotation_deg") lc.rotationDeg = std::stof(val);
            else if (key == "pitch_deg") lc.pitchDeg = std::stof(val);
            else if (key == "n_inc_count") lc.nIncCount = std::stoi(val);
            else if (key == "n_dec_count") lc.nDecCount = std::stoi(val);
            else if (key == "zoom_line_active") lc.zoomLineActive = (val == "1" || val == "true" || val == "True");
            else if (key == "label") lc.label = val;
            else if (key == "gesture") lc.gesture = val;
            else if (key == "timestamp_ms") lc.timestampMs = std::stoll(val);
        } catch (...) {
        }
    }

    if (lc.timestampMs <= 0) return std::nullopt;
    return lc;
}

std::optional<LiveControls> LoadLiveControls() {
    static std::optional<std::filesystem::path> cachedPath;

    auto tryPath = [](const std::filesystem::path& p) -> std::optional<LiveControls> {
        if (!std::filesystem::exists(p)) return std::nullopt;
        return ParseLiveControlsFile(p);
    };

    if (cachedPath) {
        if (auto parsed = tryPath(*cachedPath)) return parsed;
    }

    const std::vector<std::filesystem::path> candidates = {
        "vision/live_controls.txt",
        "../vision/live_controls.txt",
        "../../vision/live_controls.txt",
        "AstroPhysics/vision/live_controls.txt",
        std::filesystem::path(__FILE__).parent_path().parent_path() / "vision" / "live_controls.txt",
    };

    for (const auto& p : candidates) {
        if (auto parsed = tryPath(p)) {
            cachedPath = p;
            return parsed;
        }
    }
    return std::nullopt;
}

void UpdateCameraFromOrbit(Camera3D* camera, float yaw, float pitch, float distance) {
    const float cp = std::cos(pitch);
    const Vector3 offset = {
        distance * cp * std::cos(yaw),
        distance * std::sin(pitch),
        distance * cp * std::sin(yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

struct PathSample {
    Vector3 pos;
    Vector3 tangent;
    float segment_distance;
};

struct ProbeState {
    bool active = false;
    Vector3 pos{};
    Vector3 tangent{};
    Vector3 eDir{};
    Vector3 bDir{};
    Vector3 sDir{};
    float distanceAlong = 0.0f;
    float eMag = 0.0f;
    float bMag = 0.0f;
    float sMag = 0.0f;
    std::string conductor = "";
    Color accent{255, 255, 255, 255};
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0036f;
        *pitch += delta.y * 0.0034f;
        *pitch = std::clamp(*pitch, kCameraPitchMin, kCameraPitchMax);
    }

    *distance -= GetMouseWheelMove() * 0.7f;
    *distance = std::clamp(*distance, kCameraDistanceMin, kCameraDistanceMax);

    const float cp = std::cos(*pitch);
    const Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float Saturate(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}

Color WithAlpha(Color c, unsigned char alpha) {
    c.a = alpha;
    return c;
}

Color MixColor(Color a, Color b, float t) {
    t = Saturate(t);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * t),
        static_cast<unsigned char>(a.g + (b.g - a.g) * t),
        static_cast<unsigned char>(a.b + (b.b - a.b) * t),
        static_cast<unsigned char>(a.a + (b.a - a.a) * t),
    };
}

Vector3 CatmullRom(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& p3, float t) {
    const float t2 = t * t;
    const float t3 = t2 * t;
    return {
        0.5f * ((2.0f * p1.x) + (-p0.x + p2.x) * t +
                (2.0f * p0.x - 5.0f * p1.x + 4.0f * p2.x - p3.x) * t2 +
                (-p0.x + 3.0f * p1.x - 3.0f * p2.x + p3.x) * t3),
        0.5f * ((2.0f * p1.y) + (-p0.y + p2.y) * t +
                (2.0f * p0.y - 5.0f * p1.y + 4.0f * p2.y - p3.y) * t2 +
                (-p0.y + 3.0f * p1.y - 3.0f * p2.y + p3.y) * t3),
        0.5f * ((2.0f * p1.z) + (-p0.z + p2.z) * t +
                (2.0f * p0.z - 5.0f * p1.z + 4.0f * p2.z - p3.z) * t2 +
                (-p0.z + 3.0f * p1.z - 3.0f * p2.z + p3.z) * t3),
    };
}

std::vector<Vector3> BuildCurvedWirePath() {
    const std::vector<Vector3> control = {
        {-2.70f, 2.04f, 0.20f},
        {-2.72f, 2.52f, 0.22f},
        {-2.62f, 3.14f, 0.24f},
        {-2.30f, 3.64f, 0.24f},
        {-1.70f, 3.96f, 0.23f},
        {-0.82f, 4.08f, 0.21f},
        {0.25f, 4.10f, 0.18f},
        {1.22f, 3.98f, 0.14f},
        {2.00f, 3.68f, 0.10f},
        {2.52f, 3.24f, 0.08f},
        {2.76f, 2.90f, 0.06f},
        {2.90f, 2.72f, 0.04f},
    };

    std::vector<Vector3> path;
    path.reserve((control.size() - 1) * 10);
    for (std::size_t i = 0; i + 1 < control.size(); ++i) {
        const Vector3& p0 = control[(i == 0) ? 0 : i - 1];
        const Vector3& p1 = control[i];
        const Vector3& p2 = control[i + 1];
        const Vector3& p3 = control[(i + 2 < control.size()) ? i + 2 : control.size() - 1];
        const int samples = 6;
        for (int j = 0; j < samples; ++j) {
            const float t = static_cast<float>(j) / static_cast<float>(samples);
            path.push_back(CatmullRom(p0, p1, p2, p3, t));
        }
    }
    path.push_back(control.back());
    return path;
}

std::vector<Vector3> BuildLiveFeedPath() {
    const std::vector<Vector3> control = {
        {-4.76f, 1.58f, 0.22f},
        {-4.10f, 1.58f, 0.22f},
        {-3.42f, 1.56f, 0.22f},
        {-2.98f, 1.48f, 0.20f},
        {-2.80f, 1.38f, 0.16f},
        {-2.70f, 1.31f, 0.12f},
    };

    std::vector<Vector3> path;
    path.reserve((control.size() - 1) * 10);
    for (std::size_t i = 0; i + 1 < control.size(); ++i) {
        const Vector3& p0 = control[(i == 0) ? 0 : i - 1];
        const Vector3& p1 = control[i];
        const Vector3& p2 = control[i + 1];
        const Vector3& p3 = control[(i + 2 < control.size()) ? i + 2 : control.size() - 1];
        const int samples = 6;
        for (int j = 0; j < samples; ++j) {
            const float t = static_cast<float>(j) / static_cast<float>(samples);
            path.push_back(CatmullRom(p0, p1, p2, p3, t));
        }
    }
    path.push_back(control.back());
    return path;
}

std::vector<Vector3> BuildNeutralPath() {
    const std::vector<Vector3> control = {
        {-4.76f, 1.02f, -0.34f},
        {-3.98f, 1.00f, -0.34f},
        {-3.08f, 1.00f, -0.34f},
        {-1.92f, 1.08f, -0.34f},
        {-0.54f, 1.18f, -0.34f},
        {0.88f, 1.38f, -0.32f},
        {1.96f, 1.78f, -0.28f},
        {2.62f, 2.28f, -0.22f},
        {2.94f, 2.52f, -0.14f},
    };

    std::vector<Vector3> path;
    path.reserve((control.size() - 1) * 10);
    for (std::size_t i = 0; i + 1 < control.size(); ++i) {
        const Vector3& p0 = control[(i == 0) ? 0 : i - 1];
        const Vector3& p1 = control[i];
        const Vector3& p2 = control[i + 1];
        const Vector3& p3 = control[(i + 2 < control.size()) ? i + 2 : control.size() - 1];
        const int samples = 6;
        for (int j = 0; j < samples; ++j) {
            const float t = static_cast<float>(j) / static_cast<float>(samples);
            path.push_back(CatmullRom(p0, p1, p2, p3, t));
        }
    }
    path.push_back(control.back());
    return path;
}

std::vector<Vector3> BuildGroundPath() {
    const std::vector<Vector3> control = {
        {-4.76f, 0.64f, -0.58f},
        {-3.98f, 0.64f, -0.58f},
        {-3.00f, 0.66f, -0.58f},
        {-1.80f, 0.76f, -0.56f},
        {-0.46f, 0.90f, -0.54f},
        {0.86f, 1.10f, -0.50f},
        {1.92f, 1.52f, -0.44f},
        {2.56f, 2.02f, -0.34f},
        {2.82f, 2.38f, -0.26f},
        {2.88f, 2.60f, -0.20f},
    };

    std::vector<Vector3> path;
    path.reserve((control.size() - 1) * 10);
    for (std::size_t i = 0; i + 1 < control.size(); ++i) {
        const Vector3& p0 = control[(i == 0) ? 0 : i - 1];
        const Vector3& p1 = control[i];
        const Vector3& p2 = control[i + 1];
        const Vector3& p3 = control[(i + 2 < control.size()) ? i + 2 : control.size() - 1];
        const int samples = 6;
        for (int j = 0; j < samples; ++j) {
            const float t = static_cast<float>(j) / static_cast<float>(samples);
            path.push_back(CatmullRom(p0, p1, p2, p3, t));
        }
    }
    path.push_back(control.back());
    return path;
}

std::vector<Vector3> BuildOutletLivePath() {
    const std::vector<Vector3> control = {
        {-4.76f, 1.26f, 0.44f},
        {-4.10f, 1.24f, 0.44f},
        {-3.14f, 1.20f, 0.44f},
        {-1.86f, 1.02f, 0.42f},
        {-0.30f, 0.78f, 0.38f},
        {0.70f, 0.66f, 0.30f},
        {1.20f, 0.72f, 0.20f},
        {1.46f, 0.80f, 0.10f},
    };

    std::vector<Vector3> path;
    path.reserve((control.size() - 1) * 10);
    for (std::size_t i = 0; i + 1 < control.size(); ++i) {
        const Vector3& p0 = control[(i == 0) ? 0 : i - 1];
        const Vector3& p1 = control[i];
        const Vector3& p2 = control[i + 1];
        const Vector3& p3 = control[(i + 2 < control.size()) ? i + 2 : control.size() - 1];
        const int samples = 6;
        for (int j = 0; j < samples; ++j) {
            const float t = static_cast<float>(j) / static_cast<float>(samples);
            path.push_back(CatmullRom(p0, p1, p2, p3, t));
        }
    }
    path.push_back(control.back());
    return path;
}

std::vector<Vector3> BuildOutletNeutralPath() {
    const std::vector<Vector3> control = {
        {-4.76f, 0.92f, -0.40f},
        {-4.02f, 0.92f, -0.40f},
        {-3.06f, 0.92f, -0.40f},
        {-1.88f, 0.86f, -0.40f},
        {-0.44f, 0.72f, -0.40f},
        {0.68f, 0.66f, -0.34f},
        {1.20f, 0.68f, -0.22f},
        {1.46f, 0.76f, -0.12f},
    };

    std::vector<Vector3> path;
    path.reserve((control.size() - 1) * 10);
    for (std::size_t i = 0; i + 1 < control.size(); ++i) {
        const Vector3& p0 = control[(i == 0) ? 0 : i - 1];
        const Vector3& p1 = control[i];
        const Vector3& p2 = control[i + 1];
        const Vector3& p3 = control[(i + 2 < control.size()) ? i + 2 : control.size() - 1];
        const int samples = 6;
        for (int j = 0; j < samples; ++j) {
            const float t = static_cast<float>(j) / static_cast<float>(samples);
            path.push_back(CatmullRom(p0, p1, p2, p3, t));
        }
    }
    path.push_back(control.back());
    return path;
}

std::vector<Vector3> BuildOutletGroundPath() {
    const std::vector<Vector3> control = {
        {-4.76f, 0.62f, -0.66f},
        {-4.04f, 0.62f, -0.66f},
        {-3.04f, 0.62f, -0.66f},
        {-1.84f, 0.60f, -0.64f},
        {-0.40f, 0.58f, -0.60f},
        {0.72f, 0.60f, -0.48f},
        {1.22f, 0.66f, -0.30f},
        {1.46f, 0.70f, -0.18f},
    };

    std::vector<Vector3> path;
    path.reserve((control.size() - 1) * 10);
    for (std::size_t i = 0; i + 1 < control.size(); ++i) {
        const Vector3& p0 = control[(i == 0) ? 0 : i - 1];
        const Vector3& p1 = control[i];
        const Vector3& p2 = control[i + 1];
        const Vector3& p3 = control[(i + 2 < control.size()) ? i + 2 : control.size() - 1];
        const int samples = 6;
        for (int j = 0; j < samples; ++j) {
            const float t = static_cast<float>(j) / static_cast<float>(samples);
            path.push_back(CatmullRom(p0, p1, p2, p3, t));
        }
    }
    path.push_back(control.back());
    return path;
}

void DrawTube(const Vector3& a, const Vector3& b, float radius, Color color) {
    DrawCylinderEx(a, b, radius, radius, 6, color);
}

float PathLength(const std::vector<Vector3>& path) {
    float total = 0.0f;
    for (std::size_t i = 1; i < path.size(); ++i) total += Vector3Distance(path[i - 1], path[i]);
    return total;
}

PathSample SampleAlongPath(const std::vector<Vector3>& path, float distance) {
    float remaining = std::max(0.0f, distance);
    float traveled = 0.0f;
    for (std::size_t i = 1; i < path.size(); ++i) {
        const Vector3 a = path[i - 1];
        const Vector3 b = path[i];
        const float segLen = Vector3Distance(a, b);
        if (remaining <= segLen || i == path.size() - 1) {
            const float t = (segLen > 1e-4f) ? remaining / segLen : 0.0f;
            return {Vector3Lerp(a, b, t), Vector3Normalize(Vector3Subtract(b, a)), traveled + remaining};
        }
        remaining -= segLen;
        traveled += segLen;
    }
    return {path.back(), {0.0f, 1.0f, 0.0f}, traveled};
}

void BuildFieldBasis(const Vector3& tangent, Vector3* n1, Vector3* n2) {
    Vector3 ref = (std::fabs(tangent.y) < 0.8f) ? Vector3{0.0f, 1.0f, 0.0f} : Vector3{1.0f, 0.0f, 0.0f};
    *n1 = Vector3Normalize(Vector3CrossProduct(tangent, ref));
    *n2 = Vector3Normalize(Vector3CrossProduct(tangent, *n1));
}

void DrawArrow3D(Vector3 from, Vector3 to, float radius, Color color) {
    const Vector3 delta = Vector3Subtract(to, from);
    const float len = Vector3Length(delta);
    if (len < 1e-4f) return;
    const Vector3 dir = Vector3Scale(delta, 1.0f / len);
    const Vector3 tipBase = Vector3Add(to, Vector3Scale(dir, -std::min(0.18f, 0.35f * len)));
    DrawCylinderEx(from, tipBase, radius, radius, 6, color);
    DrawCylinderEx(tipBase, to, radius * 1.8f, 0.0f, 6, color);
}

float DistanceRayToPoint(Ray ray, Vector3 point) {
    const Vector3 dir = Vector3Normalize(ray.direction);
    const Vector3 diff = Vector3Subtract(point, ray.position);
    return Vector3Length(Vector3CrossProduct(diff, dir));
}

bool TryProbePath(const Ray& ray,
                  const std::vector<Vector3>& path,
                  float totalLength,
                  const char* conductor,
                  Color accent,
                  ProbeState* bestProbe,
                  float* bestDistance) {
    for (float s = 0.0f; s <= totalLength; s += 0.08f) {
        const PathSample sample = SampleAlongPath(path, s);
        const float d = DistanceRayToPoint(ray, sample.pos);
        if (d < *bestDistance) {
            *bestDistance = d;
            bestProbe->active = true;
            bestProbe->pos = sample.pos;
            bestProbe->tangent = sample.tangent;
            bestProbe->distanceAlong = s;
            bestProbe->conductor = conductor;
            bestProbe->accent = accent;
        }
    }
    return bestProbe->active;
}

void DrawSwitchPlate(bool closed) {
    DrawCube({-2.7f, 1.55f, -0.01f}, 0.86f, 1.28f, 0.03f, Color{90, 98, 112, 120});
    DrawCube({-2.7f, 1.55f, 0.03f}, 0.72f, 1.14f, 0.08f, Color{235, 237, 242, 255});
    DrawCubeWires({-2.7f, 1.55f, 0.03f}, 0.72f, 1.14f, 0.08f, Color{140, 145, 154, 255});
    DrawSphere({-2.92f, 1.94f, 0.085f}, 0.028f, Color{188, 192, 201, 255});
    DrawSphere({-2.48f, 1.16f, 0.085f}, 0.028f, Color{188, 192, 201, 255});
    DrawSphere({-2.92f, 1.94f, 0.099f}, 0.01f, Color{120, 126, 136, 255});
    DrawSphere({-2.48f, 1.16f, 0.099f}, 0.01f, Color{120, 126, 136, 255});

    DrawCube({-2.7f, 1.55f, 0.10f}, 0.32f, 0.60f, 0.05f, Color{214, 218, 225, 255});
    DrawCubeWires({-2.7f, 1.55f, 0.10f}, 0.32f, 0.60f, 0.05f, Color{150, 156, 165, 255});

    const Vector3 lowerTerminal = {-2.70f, 1.31f, 0.12f};
    const Vector3 upperTerminal = {-2.70f, 1.79f, 0.12f};
    const Color copperDark{118, 78, 46, 255};
    const Color copperBright{214, 148, 86, 255};
    DrawCylinderEx(Vector3Add(lowerTerminal, {-0.16f, 0.0f, 0.0f}), Vector3Add(lowerTerminal, {0.16f, 0.0f, 0.0f}), 0.030f, 0.030f, 10, copperDark);
    DrawCylinderEx(Vector3Add(upperTerminal, {-0.16f, 0.0f, 0.0f}), Vector3Add(upperTerminal, {0.16f, 0.0f, 0.0f}), 0.030f, 0.030f, 10, copperDark);
    DrawSphere(lowerTerminal, 0.026f, copperBright);
    DrawSphere(upperTerminal, 0.026f, copperBright);

    const Vector3 bladePivot = {-2.70f, 1.43f, 0.122f};
    const Vector3 bladeTip = closed ? Vector3{-2.70f, 1.73f, 0.122f} : Vector3{-2.56f, 1.60f, 0.12f};
    DrawCylinderEx(bladePivot, bladeTip, 0.018f, 0.016f, 10, Color{205, 126, 70, 255});
    DrawCylinderEx(lowerTerminal, bladePivot, 0.012f, 0.012f, 8, Color{188, 122, 68, 255});
    if (!closed) {
        DrawSphere(Vector3Lerp(bladeTip, upperTerminal, 0.5f), 0.018f, WithAlpha(Color{255, 188, 120, 255}, 70));
    }

    const Vector3 pivot = {-2.7f, 1.56f, 0.09f};
    const Vector3 tip = closed ? Vector3{-2.7f, 1.78f, 0.17f} : Vector3{-2.7f, 1.34f, 0.17f};
    DrawCylinderEx(pivot, tip, 0.09f, 0.07f, 10, Color{224, 228, 235, 255});
    DrawSphere(pivot, 0.07f, Color{210, 214, 222, 255});
    DrawSphere(Vector3Add(tip, {0.0f, 0.0f, 0.01f}), 0.03f, Color{245, 247, 250, 255});
}

void DrawWirePath(const std::vector<Vector3>& path, float totalPath, ViewMode viewMode, Color accent) {
    const unsigned char jacketAlpha = (viewMode == ViewMode::Field) ? 80 : 255;
    const unsigned char shellAlpha = (viewMode == ViewMode::Cutaway) ? 90 : jacketAlpha;
    const unsigned char coreAlpha = (viewMode == ViewMode::Cutaway) ? 255 : 220;

    for (std::size_t i = 1; i < path.size(); ++i) {
        Vector3 shadowA = path[i - 1];
        Vector3 shadowB = path[i];
        shadowA.z -= 0.03f;
        shadowB.z -= 0.03f;
        DrawTube(shadowA, shadowB, 0.092f, WithAlpha(Color{18, 20, 28, 255}, (viewMode == ViewMode::Field) ? 28 : 65));
        DrawTube(path[i - 1], path[i], 0.086f, WithAlpha(Color{46, 50, 58, 255}, shellAlpha));
        DrawTube(path[i - 1], path[i], 0.078f, WithAlpha(Color{74, 80, 90, 255}, shellAlpha));
        DrawTube(path[i - 1], path[i], 0.060f, WithAlpha(Color{104, 112, 126, 255}, jacketAlpha));
        DrawTube(path[i - 1], path[i], 0.036f, WithAlpha(MixColor(Color{162, 170, 182, 255}, accent, 0.18f), coreAlpha));
        DrawTube(path[i - 1], path[i], 0.016f, WithAlpha(MixColor(Color{212, 216, 224, 255}, accent, 0.35f), 255));
    }

    for (float s = 0.9f; s < totalPath - 0.4f; s += 1.55f) {
        const PathSample sample = SampleAlongPath(path, s);
        DrawCylinderEx(Vector3Add(sample.pos, {0.0f, 0.0f, -0.035f}),
                       Vector3Add(sample.pos, {0.0f, 0.0f, 0.11f}),
                       0.028f,
                       0.028f,
                       8,
                       WithAlpha(Color{74, 80, 90, 255}, (viewMode == ViewMode::Field) ? 80 : 255));
        DrawCylinderEx(Vector3Add(sample.pos, {0.0f, 0.0f, -0.02f}),
                       Vector3Add(sample.pos, {0.0f, 0.0f, 0.09f}),
                       0.018f,
                       0.018f,
                       6,
                       WithAlpha(Color{110, 116, 126, 255}, (viewMode == ViewMode::Field) ? 90 : 255));
        DrawCylinderEx(Vector3Add(sample.pos, {0.0f, 0.0f, 0.06f}),
                       Vector3Add(sample.pos, {0.0f, 0.0f, 0.11f}),
                       0.010f,
                       0.010f,
                       6,
                       WithAlpha(Color{184, 189, 196, 255}, (viewMode == ViewMode::Cutaway) ? 255 : 210));
    }

    for (float s = 0.38f; s < totalPath - 0.2f; s += 0.78f) {
        const PathSample sample = SampleAlongPath(path, s);
        Vector3 n1{};
        Vector3 n2{};
        BuildFieldBasis(sample.tangent, &n1, &n2);

        constexpr int kWrapSegs = 12;
        Vector3 prev = Vector3Add(sample.pos, Vector3Scale(n1, 0.072f));
        for (int i = 1; i <= kWrapSegs; ++i) {
            const float a = 2.0f * PI * static_cast<float>(i) / static_cast<float>(kWrapSegs);
            Vector3 cur = sample.pos;
            cur = Vector3Add(cur, Vector3Scale(n1, 0.072f * std::cos(a)));
            cur = Vector3Add(cur, Vector3Scale(n2, 0.072f * std::sin(a)));
            DrawLine3D(prev, cur, WithAlpha(MixColor(Color{210, 216, 224, 255}, accent, 0.22f), (viewMode == ViewMode::Field) ? 40 : 70));
            prev = cur;
        }
    }

    if (viewMode == ViewMode::Cutaway) {
        for (float s = 0.3f; s < totalPath - 0.18f; s += 0.48f) {
            const PathSample sample = SampleAlongPath(path, s);
            Vector3 n1{};
            Vector3 n2{};
            BuildFieldBasis(sample.tangent, &n1, &n2);
            for (int strand = 0; strand < 6; ++strand) {
                const float angle = 2.0f * PI * static_cast<float>(strand) / 6.0f;
                const Vector3 offset = Vector3Add(Vector3Scale(n1, 0.014f * std::cos(angle)),
                                                  Vector3Scale(n2, 0.014f * std::sin(angle)));
                const Vector3 a = Vector3Add(sample.pos, offset);
                const Vector3 b = Vector3Add(a, Vector3Scale(sample.tangent, 0.10f));
                DrawTube(a, b, 0.0045f, MixColor(Color{212, 124, 74, 255}, accent, 0.12f));
            }
        }
    }
}

void DrawHelixSegment(const Vector3& a,
                      const Vector3& b,
                      float localActive,
                      float startDistance,
                      float time,
                      float amplitudeScale = 1.0f) {
    const float segLen = Vector3Distance(a, b);
    if (segLen < 1e-4f || localActive <= 0.0f) return;

    const Vector3 tangent = Vector3Normalize(Vector3Subtract(b, a));
    Vector3 n1{};
    Vector3 n2{};
    BuildFieldBasis(tangent, &n1, &n2);

    const float drawLen = std::min(segLen, localActive);
    const int segments = std::max(8, static_cast<int>(28.0f * drawLen));
    const float helixRadius = 0.25f * amplitudeScale;
    const float turnsPerUnit = 2.4f;

    auto pointAt = [&](float s) {
        const float angle = 2.0f * PI * turnsPerUnit * (startDistance + s) - 6.0f * time;
        Vector3 base = Vector3Add(a, Vector3Scale(tangent, s));
        base = Vector3Add(base, Vector3Scale(n1, helixRadius * std::cos(angle)));
        base = Vector3Add(base, Vector3Scale(n2, helixRadius * std::sin(angle)));
        return base;
    };

    Vector3 prev = pointAt(0.0f);
    for (int i = 1; i <= segments; ++i) {
        const float s = drawLen * static_cast<float>(i) / static_cast<float>(segments);
        const Vector3 cur = pointAt(s);
        const float bead = 0.5f + 0.5f * std::sin(18.0f * (startDistance + s) - 9.0f * time);
        const Color core = MixColor(Color{92, 214, 255, 255}, Color{190, 248, 255, 255}, bead);
        DrawTube(prev, cur, 0.021f * amplitudeScale, Color{235, 250, 255, 235});
        DrawTube(prev, cur, 0.041f * amplitudeScale, core);
        DrawTube(prev, cur, 0.086f * amplitudeScale, WithAlpha(core, static_cast<unsigned char>(42 + 20 * bead)));

        if (i % 10 == 0) {
            DrawSphere(cur, (0.040f + 0.014f * bead) * amplitudeScale, Color{240, 252, 255, 230});
            DrawSphere(cur, (0.11f + 0.03f * bead) * amplitudeScale, WithAlpha(core, 28));
        }
        prev = cur;
    }
}

void DrawHelixPulseAlongPath(const std::vector<Vector3>& path, float activeDistance, float time, float amplitudeScale = 1.0f) {
    float covered = 0.0f;
    for (std::size_t i = 1; i < path.size(); ++i) {
        const Vector3 a = path[i - 1];
        const Vector3 b = path[i];
        const float segLen = Vector3Distance(a, b);
        const float localActive = std::clamp(activeDistance - covered, 0.0f, segLen);
        if (localActive <= 0.0f) break;
        DrawHelixSegment(a, b, localActive, covered, time, amplitudeScale);
        covered += segLen;
    }
}

void DrawBreakerPanel(bool breakerClosed, bool acMode, FaultMode faultMode, float tripProgress, float branchGlow, float outletGlow) {
    DrawCube({-5.05f, 1.12f, -0.04f}, 0.84f, 1.70f, 0.54f, Color{64, 70, 82, 255});
    DrawCube({-5.00f, 1.12f, 0.06f}, 0.72f, 1.58f, 0.10f, Color{104, 110, 122, 255});
    DrawCubeWires({-5.00f, 1.12f, 0.06f}, 0.72f, 1.58f, 0.10f, Color{132, 138, 150, 255});

    DrawCylinderEx({-4.95f, 1.66f, -0.06f}, {-4.95f, 0.54f, -0.06f}, 0.040f, 0.040f, 12, Color{192, 162, 104, 255});
    DrawCylinderEx({-5.16f, 1.64f, 0.02f}, {-5.16f, 0.58f, 0.02f}, 0.026f, 0.026f, 10, Color{126, 190, 255, 255});
    DrawCylinderEx({-4.76f, 1.64f, -0.18f}, {-4.76f, 0.58f, -0.18f}, 0.026f, 0.026f, 10, Color{136, 216, 126, 255});
    DrawCylinderEx({-4.95f, 1.44f, 0.04f}, {-4.30f, 1.44f, 0.04f}, 0.018f, 0.018f, 10, WithAlpha(Color{226, 144, 88, 255}, static_cast<unsigned char>(120 + 100 * branchGlow)));
    DrawCylinderEx({-4.95f, 1.10f, 0.04f}, {-4.30f, 1.10f, 0.04f}, 0.018f, 0.018f, 10, WithAlpha(Color{226, 144, 88, 255}, static_cast<unsigned char>(120 + 100 * outletGlow)));
    DrawCylinderEx({-5.16f, 1.10f, 0.10f}, {-4.32f, 1.10f, 0.10f}, 0.012f, 0.012f, 8, Color{120, 188, 255, 220});
    DrawCylinderEx({-4.76f, 0.84f, -0.08f}, {-4.32f, 0.84f, -0.08f}, 0.012f, 0.012f, 8, Color{150, 214, 120, 220});

    const Vector3 breakerBody = {-4.88f, 1.40f, 0.14f};
    DrawCube(breakerBody, 0.22f, 0.40f, 0.18f, Color{32, 36, 44, 255});
    const Vector3 handleOn = {-4.82f, 1.52f, 0.28f};
    const Vector3 handleOff = {-4.98f, 1.26f, 0.28f};
    const Vector3 handleTip = Vector3Lerp(handleOn, handleOff, Saturate(tripProgress));
    DrawCylinderEx({-4.90f, 1.40f, 0.22f}, handleTip, 0.030f, 0.028f, 10, breakerClosed ? Color{236, 188, 102, 255} : Color{198, 94, 84, 255});
    DrawSphere(handleTip, 0.028f, breakerClosed ? Color{255, 216, 138, 255} : Color{244, 126, 114, 255});
    DrawCube({-4.48f, 1.44f, 0.14f}, 0.18f, 0.18f, 0.12f, Color{42, 46, 54, 255});
    DrawCube({-4.48f, 1.10f, 0.14f}, 0.18f, 0.18f, 0.12f, Color{42, 46, 54, 255});
    DrawSphere({-4.48f, 1.44f, 0.22f}, 0.024f, WithAlpha(Color{255, 216, 138, 255}, static_cast<unsigned char>(90 + 120 * branchGlow)));
    DrawSphere({-4.48f, 1.10f, 0.22f}, 0.024f, WithAlpha(Color{255, 216, 138, 255}, static_cast<unsigned char>(90 + 120 * outletGlow)));

    DrawCube({-4.94f, 0.84f, 0.16f}, 0.18f, 0.22f, 0.14f, acMode ? Color{104, 164, 255, 255} : Color{124, 136, 150, 255});
    DrawCube({-5.12f, 1.42f, 0.16f}, 0.10f, 0.12f, 0.14f, faultMode == FaultMode::None ? Color{74, 92, 112, 255} : Color{210, 106, 92, 255});

    DrawSphere({-4.94f, 1.66f, 0.18f}, 0.034f, Color{226, 144, 88, 255});
    DrawSphere({-5.16f, 1.04f, 0.14f}, 0.030f, Color{120, 188, 255, 255});
    DrawSphere({-4.76f, 0.76f, -0.02f}, 0.030f, Color{150, 214, 120, 255});
}

void DrawOutletAssembly(const Vector3& center, ViewMode viewMode, float outletPower, bool groundIntact, float time) {
    DrawCube({center.x, center.y, center.z - 0.03f}, 0.84f, 1.18f, 0.03f, Color{88, 94, 108, 110});
    DrawCube({center.x, center.y, center.z + 0.01f}, 0.68f, 1.00f, 0.08f, Color{232, 235, 240, 255});
    DrawCubeWires({center.x, center.y, center.z + 0.01f}, 0.68f, 1.00f, 0.08f, Color{144, 148, 156, 255});

    for (int row = 0; row < 2; ++row) {
        const float y = center.y + (row == 0 ? 0.22f : -0.22f);
        DrawCylinderEx({center.x - 0.10f, y + 0.08f, center.z + 0.06f},
                       {center.x - 0.10f, y - 0.08f, center.z + 0.06f},
                       0.022f,
                       0.022f,
                       8,
                       Color{76, 78, 84, 255});
        DrawCylinderEx({center.x + 0.10f, y + 0.08f, center.z + 0.06f},
                       {center.x + 0.10f, y - 0.08f, center.z + 0.06f},
                       0.014f,
                       0.014f,
                       8,
                       Color{112, 118, 128, 255});
        DrawCylinderEx({center.x, y - 0.16f, center.z + 0.06f},
                       {center.x, y - 0.06f, center.z + 0.06f},
                       0.018f,
                       0.018f,
                       8,
                       groundIntact ? Color{132, 210, 122, 255} : Color{70, 78, 88, 255});
    }

    const Vector3 plug = {center.x + 0.22f, center.y - 0.22f, center.z + 0.09f};
    DrawCube(plug, 0.20f, 0.18f, 0.14f, Color{34, 36, 42, 255});
    DrawCylinderEx({plug.x - 0.08f, plug.y + 0.03f, plug.z - 0.01f}, {plug.x - 0.18f, plug.y + 0.03f, plug.z - 0.01f}, 0.010f, 0.010f, 8, Color{214, 188, 112, 255});
    DrawCylinderEx({plug.x - 0.08f, plug.y - 0.03f, plug.z - 0.01f}, {plug.x - 0.18f, plug.y - 0.03f, plug.z - 0.01f}, 0.010f, 0.010f, 8, Color{186, 192, 204, 255});
    const Vector3 cordKnee = {plug.x + 0.34f, plug.y - 0.24f, plug.z - 0.14f};
    const Vector3 cordEnd = {plug.x + 0.94f, plug.y - 0.30f, plug.z - 0.42f};
    DrawCylinderEx({plug.x + 0.10f, plug.y, plug.z}, cordKnee, 0.040f, 0.036f, 10, Color{48, 54, 64, 255});
    DrawCylinderEx(cordKnee, cordEnd, 0.036f, 0.032f, 10, Color{48, 54, 64, 255});
    DrawCylinderEx({plug.x + 0.10f, plug.y, plug.z}, cordKnee, 0.020f, 0.018f, 10, Color{72, 78, 90, 255});
    DrawCylinderEx(cordKnee, cordEnd, 0.018f, 0.016f, 10, Color{72, 78, 90, 255});

    const Vector3 fanBase = {plug.x + 1.18f, plug.y - 0.30f, plug.z - 0.56f};
    DrawCube({fanBase.x, fanBase.y + 0.04f, fanBase.z}, 0.58f, 0.12f, 0.30f, Color{46, 54, 64, 255});
    DrawCylinderEx({fanBase.x, fanBase.y + 0.10f, fanBase.z}, {fanBase.x, fanBase.y + 0.44f, fanBase.z}, 0.022f, 0.022f, 10, Color{126, 134, 146, 255});
    DrawSphere({fanBase.x, fanBase.y + 0.60f, fanBase.z}, 0.07f, Color{96, 102, 112, 255});
    DrawSphereWires({fanBase.x, fanBase.y + 0.60f, fanBase.z}, 0.33f, 12, 12, Color{170, 178, 190, 230});
    for (int blade = 0; blade < 4; ++blade) {
        const float a = 2.0f * PI * static_cast<float>(blade) / 4.0f + time * (0.6f + 7.0f * outletPower);
        const Vector3 tip = {fanBase.x + 0.22f * std::cos(a), fanBase.y + 0.60f + 0.18f * std::sin(a), fanBase.z};
        DrawCylinderEx({fanBase.x, fanBase.y + 0.60f, fanBase.z}, tip, 0.020f, 0.010f, 8, Color{194, 200, 210, 255});
    }
    DrawSphere({fanBase.x, fanBase.y + 0.60f, fanBase.z}, 0.038f, Color{222, 226, 232, 255});

    if (outletPower > 0.05f) {
        DrawSphere({center.x, center.y - 0.22f, center.z + 0.16f}, 0.12f, WithAlpha(Color{130, 244, 170, 255}, static_cast<unsigned char>(80 + 80 * outletPower)));
        DrawSphere({fanBase.x, fanBase.y + 0.60f, fanBase.z}, 0.24f, WithAlpha(Color{150, 255, 184, 255}, static_cast<unsigned char>(34 + 42 * outletPower)));
        if (viewMode != ViewMode::Normal) {
            DrawArrow3D({center.x + 0.34f, center.y - 0.22f, center.z + 0.18f},
                        {center.x + 0.10f, center.y - 0.22f, center.z + 0.18f},
                        0.012f,
                        Color{148, 255, 184, 230});
            DrawArrow3D({fanBase.x - 0.30f, fanBase.y + 0.60f, fanBase.z + 0.04f},
                        {fanBase.x - 0.06f, fanBase.y + 0.60f, fanBase.z + 0.04f},
                        0.010f,
                        Color{148, 255, 184, 220});
        }
    }
}

void DrawStatusBadge(int x, int y, int w, const char* label, Color accent) {
    const Rectangle r{static_cast<float>(x), static_cast<float>(y), static_cast<float>(w), 28.0f};
    DrawRectangleRounded(r, 0.35f, 8, WithAlpha(accent, 34));
    DrawRectangleLinesEx(r, 1.2f, WithAlpha(accent, 150));
    DrawText(label, x + 10, y + 6, 16, accent);
}

void DrawWorldCallout(const Camera3D& camera, const Vector3& worldPos, const char* text, Color accent) {
    const Vector2 screen = GetWorldToScreen(worldPos, camera);
    const int textW = MeasureText(text, 16);
    const Rectangle r{screen.x - textW * 0.5f - 8.0f, screen.y - 26.0f, static_cast<float>(textW + 16), 22.0f};
    DrawRectangleRounded(r, 0.30f, 8, Color{8, 12, 18, 180});
    DrawRectangleLinesEx(r, 1.0f, WithAlpha(accent, 180));
    DrawText(text, static_cast<int>(r.x) + 8, static_cast<int>(r.y) + 4, 16, accent);
}

void DrawFieldIndicatorsAlongPath(const std::vector<Vector3>& path, float activeDistance, float time) {
    for (float s = 0.42f; s < activeDistance; s += 0.56f) {
        const PathSample sample = SampleAlongPath(path, s);
        Vector3 n1{};
        Vector3 n2{};
        BuildFieldBasis(sample.tangent, &n1, &n2);

        const float pulse = 0.5f + 0.5f * std::sin(8.2f * (s - activeDistance) - 5.5f * time);
        const float swirl = 7.8f * s - 5.5f * time;
        const Vector3 eDir = Vector3Add(Vector3Scale(n1, std::cos(swirl)), Vector3Scale(n2, std::sin(swirl)));
        const Vector3 eVec = Vector3Scale(eDir, 0.23f + 0.08f * pulse);
        const Vector3 sVec = Vector3Scale(sample.tangent, 0.30f + 0.12f * pulse);

        DrawArrow3D(Vector3Add(sample.pos, Vector3Scale(eDir, -0.08f)),
                    Vector3Add(sample.pos, Vector3Scale(eDir, 0.18f + 0.08f * pulse)),
                    0.0105f,
                    WithAlpha(Color{120, 225, 255, 255}, static_cast<unsigned char>(150 + 70 * pulse)));

        DrawArrow3D(Vector3Add(sample.pos, Vector3Scale(n1, -0.12f)),
                    Vector3Add(sample.pos, Vector3Add(Vector3Scale(n1, -0.10f), sVec)),
                    0.0105f,
                    WithAlpha(Color{150, 255, 170, 255}, static_cast<unsigned char>(130 + 90 * pulse)));

        constexpr int kLoops = 14;
        Vector3 prev = Vector3Add(sample.pos, Vector3Scale(n1, 0.11f));
        for (int i = 1; i <= kLoops; ++i) {
            const float a = 2.0f * PI * static_cast<float>(i) / static_cast<float>(kLoops);
            Vector3 cur = sample.pos;
            cur = Vector3Add(cur, Vector3Scale(n1, (0.11f + 0.02f * pulse) * std::cos(a)));
            cur = Vector3Add(cur, Vector3Scale(n2, (0.11f + 0.02f * pulse) * std::sin(a)));
            DrawLine3D(prev, cur, WithAlpha(Color{255, 182, 105, 255}, static_cast<unsigned char>(78 + 85 * pulse)));
            prev = cur;
        }

        DrawLine3D(Vector3Add(sample.pos, Vector3Scale(eDir, -0.16f)),
                   Vector3Add(sample.pos, Vector3Scale(eDir, 0.16f)),
                   WithAlpha(Color{120, 225, 255, 255}, 70));
        DrawLine3D(Vector3Add(sample.pos, Vector3Scale(n2, -0.10f)),
                   Vector3Add(sample.pos, Vector3Scale(n2, 0.10f)),
                   WithAlpha(Color{255, 182, 105, 255}, 45));
    }
}

void DrawBulbAssembly(const Vector3& bulbBase,
                      const Vector3& bulbCenter,
                      const Color& filamentColor,
                      const Color& bulbGlass,
                      ViewMode viewMode,
                      float bulbPower,
                      float time) {
    const Color brassDark{92, 86, 78, 255};
    const Color brassMid{146, 136, 118, 255};
    const Color brassBright{196, 186, 160, 255};
    const Vector3 shellTop = {bulbBase.x, bulbBase.y + 0.42f, bulbBase.z};
    const Vector3 shellBottom = {bulbBase.x, bulbBase.y - 0.02f, bulbBase.z};

    DrawCylinderEx({bulbBase.x, bulbBase.y + 0.98f, bulbBase.z}, shellTop, 0.036f, 0.032f, 12,
                   WithAlpha(Color{190, 198, 210, 255}, (viewMode == ViewMode::Field) ? 150 : 255));
    DrawCylinderEx(shellTop, shellBottom, 0.13f, 0.12f, 16, brassDark);
    DrawCylinderEx(shellTop, shellBottom, 0.112f, 0.106f, 16, brassMid);

    for (int i = 0; i < 7; ++i) {
        const float y = shellTop.y - 0.045f - 0.055f * static_cast<float>(i);
        DrawCircle3D({bulbBase.x, y, bulbBase.z}, 0.122f, {1.0f, 0.0f, 0.0f}, 90.0f, WithAlpha(brassBright, 150));
        DrawCircle3D({bulbBase.x, y - 0.014f, bulbBase.z}, 0.102f, {1.0f, 0.0f, 0.0f}, 90.0f, WithAlpha(brassDark, 160));
    }

    DrawCylinderEx({bulbBase.x, shellBottom.y - 0.05f, bulbBase.z}, {bulbBase.x, shellBottom.y - 0.16f, bulbBase.z}, 0.058f, 0.072f, 12,
                   brassDark);
    DrawSphere({bulbBase.x, shellBottom.y - 0.20f, bulbBase.z}, 0.040f, Color{214, 200, 170, 255});
    DrawSphere({bulbBase.x, shellBottom.y - 0.20f, bulbBase.z}, 0.016f, Color{242, 234, 208, 255});

    const Vector3 stemBottom = {bulbBase.x, bulbBase.y + 0.06f, bulbBase.z};
    const Vector3 stemTop = {bulbBase.x, bulbBase.y + 0.33f, bulbBase.z};
    DrawCylinderEx(stemBottom, stemTop, 0.074f, 0.052f, 12, WithAlpha(Color{170, 205, 230, 255}, 120));
    DrawSphere(Vector3Add(stemBottom, {0.0f, 0.01f, 0.0f}), 0.054f, WithAlpha(Color{180, 214, 236, 255}, 120));

    const Vector3 supportLeftTop = {bulbBase.x, bulbBase.y + 0.31f, bulbBase.z - 0.11f};
    const Vector3 supportRightTop = {bulbBase.x, bulbBase.y + 0.31f, bulbBase.z + 0.11f};
    DrawCylinderEx({bulbBase.x, bulbBase.y + 0.10f, bulbBase.z - 0.11f}, supportLeftTop, 0.010f, 0.010f, 10, Color{194, 196, 202, 255});
    DrawCylinderEx({bulbBase.x, bulbBase.y + 0.10f, bulbBase.z + 0.11f}, supportRightTop, 0.010f, 0.010f, 10, Color{194, 196, 202, 255});
    DrawCylinderEx(supportLeftTop, {bulbBase.x, bulbBase.y + 0.36f, bulbBase.z - 0.05f}, 0.008f, 0.008f, 10, Color{196, 198, 204, 255});
    DrawCylinderEx(supportRightTop, {bulbBase.x, bulbBase.y + 0.36f, bulbBase.z + 0.05f}, 0.008f, 0.008f, 10, Color{196, 198, 204, 255});

    const Vector3 filamentStart = {bulbBase.x, bulbBase.y + 0.36f, bulbBase.z - 0.05f};
    const Vector3 filamentEnd = {bulbBase.x, bulbBase.y + 0.36f, bulbBase.z + 0.05f};
    const int coils = 44;
    const float pulseBeat = 0.5f + 0.5f * std::sin(8.0f * time);
    const float pulseGlow = Saturate(0.25f + 1.15f * bulbPower);
    Vector3 prev = filamentStart;
    for (int i = 1; i <= coils; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(coils);
        Vector3 cur = {
            bulbBase.x,
            bulbBase.y + 0.36f + 0.010f * std::sin(2.0f * PI * 6.0f * u + 4.0f * time * bulbPower),
            bulbBase.z - 0.05f + 0.10f * u + 0.0105f * std::sin(2.0f * PI * 12.0f * u),
        };
        DrawTube(prev, cur, 0.0095f, filamentColor);
        prev = cur;
    }
    DrawTube(prev, filamentEnd, 0.0095f, filamentColor);
    DrawSphere(Vector3Lerp(filamentStart, filamentEnd, 0.5f), 0.024f + 0.018f * pulseGlow, WithAlpha(filamentColor, static_cast<unsigned char>(180 + 60 * pulseGlow)));

    const unsigned char glassAlpha = (viewMode == ViewMode::Cutaway) ? 72 : ((viewMode == ViewMode::Field) ? 105 : bulbGlass.a);
    DrawSphere(bulbCenter, 0.50f, WithAlpha(bulbGlass, glassAlpha));
    DrawSphereWires(bulbCenter, 0.50f, 16, 16, WithAlpha(Color{190, 220, 255, 255}, (viewMode == ViewMode::Field) ? 70 : 36));
    DrawSphere(Vector3Add(bulbCenter, {-0.14f, 0.16f, -0.22f}), 0.11f, WithAlpha(Color{255, 255, 255, 255}, 62));
    DrawSphere(Vector3Add(bulbCenter, {0.05f, -0.10f, 0.26f}), 0.05f, WithAlpha(Color{216, 238, 255, 255}, 28));
    DrawSphere(bulbCenter, 0.74f + 0.15f * pulseGlow, WithAlpha(Color{255, 228, 145, 255}, static_cast<unsigned char>(64 + 108.0f * pulseGlow)));
    DrawSphere(bulbCenter, 0.90f + 0.18f * pulseGlow, WithAlpha(Color{255, 210, 118, 255}, static_cast<unsigned char>(16 + 48.0f * pulseGlow)));

    if (bulbPower > 0.05f) {
        for (int ring = 0; ring < 2; ++ring) {
            const float phase = time * (3.0f + 0.8f * ring) + ring * 0.9f;
            const float radius = 0.52f + 0.10f * static_cast<float>(ring) + 0.04f * pulseBeat;
            const unsigned char alpha = static_cast<unsigned char>(16 + 48 * bulbPower / static_cast<float>(ring + 1));
            DrawCircle3D({bulbCenter.x, bulbCenter.y, bulbCenter.z}, radius, {1.0f, 0.0f, 0.0f}, 90.0f + 14.0f * std::sin(phase), WithAlpha(Color{255, 224, 132, 255}, alpha));
            DrawCircle3D({bulbCenter.x, bulbCenter.y, bulbCenter.z}, radius * 0.92f, {0.0f, 0.0f, 1.0f}, 90.0f + 14.0f * std::cos(phase), WithAlpha(Color{255, 244, 190, 255}, alpha / 2));
        }
    }
}

const char* ViewModeLabel(ViewMode mode) {
    switch (mode) {
        case ViewMode::Normal: return "normal";
        case ViewMode::Field: return "field";
        case ViewMode::Cutaway: return "cutaway";
    }
    return "normal";
}

const char* FaultModeLabel(FaultMode mode) {
    switch (mode) {
        case FaultMode::None: return "none";
        case FaultMode::OpenNeutral: return "open neutral";
        case FaultMode::ShortCircuit: return "short circuit";
        case FaultMode::MissingGround: return "missing ground";
        case FaultMode::Overload: return "overload";
        case FaultMode::BreakerTrip: return "breaker trip";
    }
    return "none";
}

void UpdateProbeVectors(ProbeState* probe, float time, float magnitudeScale, float flowSign) {
    Vector3 n1{};
    Vector3 n2{};
    BuildFieldBasis(probe->tangent, &n1, &n2);
    const float swirl = 7.8f * probe->distanceAlong - 5.5f * time;
    probe->eDir = Vector3Normalize(Vector3Add(Vector3Scale(n1, std::cos(swirl)), Vector3Scale(n2, std::sin(swirl))));
    probe->bDir = Vector3Normalize(Vector3CrossProduct(probe->tangent, probe->eDir));
    probe->sDir = Vector3Scale(probe->tangent, flowSign);
    probe->eMag = 38.0f * magnitudeScale;
    probe->bMag = 0.12f * magnitudeScale;
    probe->sMag = 18.0f * magnitudeScale;
}

void DrawProbeGlyph(const ProbeState& probe) {
    if (!probe.active) return;
    DrawSphere(probe.pos, 0.075f, probe.accent);
    DrawSphere(probe.pos, 0.16f, WithAlpha(probe.accent, 38));
    DrawArrow3D(probe.pos, Vector3Add(probe.pos, Vector3Scale(probe.eDir, 0.40f)), 0.012f, Color{120, 228, 255, 255});
    DrawArrow3D(probe.pos, Vector3Add(probe.pos, Vector3Scale(probe.bDir, 0.34f)), 0.012f, Color{255, 186, 110, 255});
    DrawArrow3D(probe.pos, Vector3Add(probe.pos, Vector3Scale(probe.sDir, 0.48f)), 0.012f, Color{150, 255, 176, 255});
}

std::string HudText(bool switchClosed, bool breakerClosed, bool acMode, FaultMode faultMode, float front, float bulbPower, bool paused) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(2);
    os << "switch=" << (switchClosed ? "on" : "off")
       << "  breaker=" << (breakerClosed ? "closed" : "tripped")
       << "  mode=" << (acMode ? "ac" : "pulse")
       << "  fault=" << FaultModeLabel(faultMode)
       << "  front=" << front
       << "  bulb=" << bulbPower;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(kScreenWidth, kScreenHeight, "Wall Switch EM Helix to Bulb 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {6.8f, 4.5f, 8.0f};
    camera.target = {0.0f, 2.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.95f;
    float camPitch = 0.18f;
    float camDistance = 11.5f;

    const std::vector<Vector3> liveFeedPath = BuildLiveFeedPath();
    const std::vector<Vector3> liveOutPath = BuildCurvedWirePath();
    const std::vector<Vector3> neutralPath = BuildNeutralPath();
    const std::vector<Vector3> groundPath = BuildGroundPath();
    const std::vector<Vector3> outletLivePath = BuildOutletLivePath();
    const std::vector<Vector3> outletNeutralPath = BuildOutletNeutralPath();
    const std::vector<Vector3> outletGroundPath = BuildOutletGroundPath();
    const float liveFeedLength = PathLength(liveFeedPath);
    const float liveOutLength = PathLength(liveOutPath);
    const float neutralLength = PathLength(neutralPath);
    const float groundLength = PathLength(groundPath);
    const float outletLiveLength = PathLength(outletLivePath);
    const float outletNeutralLength = PathLength(outletNeutralPath);
    const float outletGroundLength = PathLength(outletGroundPath);
    const float loopLength = liveFeedLength + liveOutLength + neutralLength;
    const float bulbDistance = liveFeedLength + liveOutLength;

    bool switchClosed = false;
    bool breakerClosed = true;
    bool paused = false;
    bool acMode = false;
    float t = 0.0f;
    float signalDistance = 0.0f;
    float signalSpeed = 3.8f;
    float bulbPower = 0.0f;
    float outletPower = 0.0f;
    float faultTimer = 0.0f;
    ViewMode viewMode = ViewMode::Normal;
    FaultMode faultMode = FaultMode::None;
    ProbeState probe{};
    bool hasPrevLive = false;
    float prevLiveRotDeg = 0.0f;
    float prevLivePitchDeg = 0.0f;
    int prevLiveNIncCount = 0;
    int prevLiveNDecCount = 0;
    int pendingLeftPinches = 0;
    std::int64_t leftPinchDeadlineMs = 0;
    int pendingRightPinches = 0;
    std::int64_t rightPinchDeadlineMs = 0;
    std::string bridgeStatus = "tracker: waiting for AstroPhysics/vision/live_controls.txt";
    astro_hand::UdpFrameReceiver frameReceiver;
    const bool frameReceiverOk = frameReceiver.Start(static_cast<uint16_t>(astro_hand::kFrameUdpPort));
    std::vector<unsigned char> previewFrameBytes;
    Texture2D webcamTexture{};
    float lastFrameWallClock = -100.0f;
    bool previewLive = false;

    auto applyFaultMode = [&](FaultMode nextMode) {
        faultMode = nextMode;
        faultTimer = 0.0f;
        signalDistance = 0.0f;
        if (faultMode == FaultMode::BreakerTrip) breakerClosed = false;
        if (faultMode == FaultMode::None) probe.active = false;
    };

    auto toggleSwitch = [&]() {
        switchClosed = !switchClosed;
        signalDistance = 0.0f;
        faultTimer = 0.0f;
        if (!switchClosed) bulbPower = 0.0f;
    };

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_F11)) ToggleFullscreen();
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_SPACE)) toggleSwitch();
        if (IsKeyPressed(KEY_R)) {
            switchClosed = false;
            breakerClosed = true;
            paused = false;
            acMode = false;
            t = 0.0f;
            signalDistance = 0.0f;
            signalSpeed = 3.8f;
            bulbPower = 0.0f;
            outletPower = 0.0f;
            faultTimer = 0.0f;
            viewMode = ViewMode::Normal;
            faultMode = FaultMode::None;
            probe.active = false;
        }
        if (IsKeyPressed(KEY_A)) acMode = !acMode;
        if (IsKeyPressed(KEY_B) && faultMode != FaultMode::BreakerTrip) breakerClosed = true;
        if (IsKeyPressed(KEY_F)) {
            const int next = (static_cast<int>(faultMode) + 1) % 6;
            applyFaultMode(static_cast<FaultMode>(next));
        }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) signalSpeed = std::max(kSignalSpeedMin, signalSpeed - 0.25f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) signalSpeed = std::min(kSignalSpeedMax, signalSpeed + 0.25f);
        if (IsKeyPressed(KEY_ONE)) viewMode = ViewMode::Normal;
        if (IsKeyPressed(KEY_TWO)) viewMode = ViewMode::Field;
        if (IsKeyPressed(KEY_THREE)) viewMode = ViewMode::Cutaway;

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        const float dt = GetFrameTime();
        const std::int64_t nowMs = UnixMsNow();
        int framePacketsRead = 0;
        const float wallNow = static_cast<float>(GetTime());
        if (frameReceiver.Poll(previewFrameBytes, framePacketsRead)) {
            lastFrameWallClock = wallNow;
            astro_hand::UpdatePreviewTexture(webcamTexture, previewFrameBytes);
        }
        previewLive = frameReceiverOk && webcamTexture.id > 0 && ((wallNow - lastFrameWallClock) < astro_hand::kLinkTimeout);

        if (auto live = LoadLiveControls()) {
            const std::int64_t ageMs = nowMs - live->timestampMs;
            if (ageMs <= kControlStaleMs) {
                if (!hasPrevLive) {
                    hasPrevLive = true;
                    prevLiveRotDeg = live->rotationDeg;
                    prevLivePitchDeg = live->pitchDeg;
                    prevLiveNIncCount = live->nIncCount;
                    prevLiveNDecCount = live->nDecCount;
                } else {
                    const float rotDeltaDeg = AngleDeltaDeg(live->rotationDeg, prevLiveRotDeg);
                    prevLiveRotDeg = live->rotationDeg;
                    camYaw += rotDeltaDeg * DEG2RAD;

                    const float pitchDeltaDeg = live->pitchDeg - prevLivePitchDeg;
                    prevLivePitchDeg = live->pitchDeg;
                    camPitch += pitchDeltaDeg * DEG2RAD;
                    camPitch = std::clamp(camPitch, kCameraPitchMin, kCameraPitchMax);

                    if (live->zoomLineActive) {
                        pendingLeftPinches = 0;
                        leftPinchDeadlineMs = 0;
                        pendingRightPinches = 0;
                        rightPinchDeadlineMs = 0;
                    } else {
                        if (live->nIncCount >= prevLiveNIncCount) {
                            const int incDelta = live->nIncCount - prevLiveNIncCount;
                            if (incDelta > 0) {
                                pendingRightPinches += incDelta;
                                rightPinchDeadlineMs = nowMs + kRightPinchConfirmMs;
                            }
                        }
                        if (live->nDecCount >= prevLiveNDecCount) {
                            const int decDelta = live->nDecCount - prevLiveNDecCount;
                            if (decDelta > 0) {
                                pendingLeftPinches += decDelta;
                                leftPinchDeadlineMs = nowMs + kLeftPinchWindowMs;
                            }
                        }
                    }
                    prevLiveNIncCount = live->nIncCount;
                    prevLiveNDecCount = live->nDecCount;
                }

                const float currentLiveZoom = std::clamp(live->zoom, kBridgeLiveZoomMin, kBridgeLiveZoomMax);
                const float liveZoomNorm = (currentLiveZoom - kBridgeLiveZoomMin) / (kBridgeLiveZoomMax - kBridgeLiveZoomMin);
                camDistance = kCameraDistanceMax + (kCameraDistanceMin - kCameraDistanceMax) * liveZoomNorm;
                UpdateCameraFromOrbit(&camera, camYaw, camPitch, camDistance);

                std::ostringstream cs;
                cs << "tracker: live  hand=" << live->label
                   << "  gesture=" << live->gesture
                   << "  age=" << ageMs << "ms";
                bridgeStatus = cs.str();
            } else {
                hasPrevLive = false;
                bridgeStatus = "tracker: stale";
            }
        } else {
            hasPrevLive = false;
            bridgeStatus = "tracker: waiting for AstroPhysics/vision/live_controls.txt";
        }

        if (pendingRightPinches > 0 && rightPinchDeadlineMs > 0 && nowMs >= rightPinchDeadlineMs) {
            toggleSwitch();
            pendingRightPinches = 0;
            rightPinchDeadlineMs = 0;
        }

        if (pendingLeftPinches >= 2) {
            signalSpeed = std::min(kSignalSpeedMax, signalSpeed + kSignalSpeedFastStep);
            pendingLeftPinches = 0;
            leftPinchDeadlineMs = 0;
        } else if (pendingLeftPinches == 1 && leftPinchDeadlineMs > 0 && nowMs >= leftPinchDeadlineMs) {
            signalSpeed = std::max(kSignalSpeedMin, signalSpeed - kSignalSpeedSlowStep);
            pendingLeftPinches = 0;
            leftPinchDeadlineMs = 0;
        }

        const bool neutralIntact = (faultMode != FaultMode::OpenNeutral);
        const bool groundIntact = (faultMode != FaultMode::MissingGround);
        const bool shortFaultArmed = (faultMode == FaultMode::ShortCircuit);
        const bool overloadArmed = (faultMode == FaultMode::Overload);
        const bool conductorClosed = switchClosed && breakerClosed;
        const float acCarrier = 0.5f + 0.5f * std::sin(2.0f * PI * 1.2f * t);
        const float shortTripProgress = shortFaultArmed ? Saturate(faultTimer / 0.38f) : 0.0f;
        const float overloadTripProgress = overloadArmed ? Saturate(faultTimer / 1.55f) : 0.0f;
        const float tripProgress = breakerClosed ? std::max(shortTripProgress, overloadTripProgress) : 1.0f;

        if (!paused) {
            t += dt;
            if (conductorClosed && shortFaultArmed) {
                faultTimer += dt;
                signalDistance = loopLength;
                if (faultTimer > 0.38f) breakerClosed = false;
            } else if (conductorClosed && overloadArmed) {
                faultTimer += dt;
                if (faultTimer > 1.55f) breakerClosed = false;
            } else {
                faultTimer = 0.0f;
            }

            if (switchClosed && breakerClosed) {
                if (acMode) {
                    signalDistance = loopLength;
                } else {
                    signalDistance = std::min(loopLength, signalDistance + signalSpeed * dt);
                }
            } else {
                signalDistance = std::max(0.0f, signalDistance - 2.4f * signalSpeed * dt);
            }

            float bulbTarget = 0.0f;
            if (switchClosed && breakerClosed && neutralIntact && !shortFaultArmed) {
                if (acMode) {
                    const float acWave = std::pow(std::abs(std::sin(2.0f * PI * 1.2f * t)), 1.45f);
                    bulbTarget = 0.18f + 0.90f * acWave;
                } else {
                    bulbTarget = Saturate((signalDistance - bulbDistance + 0.25f) / 0.7f);
                }
                if (overloadArmed) bulbTarget = std::min(1.35f, bulbTarget * 1.28f);
            }
            bulbPower += (bulbTarget - bulbPower) * std::min(1.0f, 4.2f * dt);

            float outletTarget = breakerClosed ? (acMode ? (0.72f + 0.22f * std::abs(std::sin(2.0f * PI * 1.2f * t + 0.4f))) : 0.82f) : 0.0f;
            if (!groundIntact && viewMode == ViewMode::Field) outletTarget *= 0.92f;
            outletPower += (outletTarget - outletPower) * std::min(1.0f, 3.8f * dt);
        }

        const Color filamentColor = MixColor(Color{120, 35, 15, 255}, Color{255, 244, 180, 255}, bulbPower);
        const Color bulbGlass = MixColor(Color{55, 68, 84, 70}, Color{255, 226, 140, 170}, bulbPower);
        const Color groundAccent = groundIntact ? Color{146, 214, 118, 255} : Color{88, 96, 102, 255};
        const Color neutralAccent = neutralIntact ? Color{118, 186, 255, 255} : Color{88, 96, 102, 255};

        const float liveFeedActive = breakerClosed ? (acMode ? liveFeedLength : std::clamp(signalDistance, 0.0f, liveFeedLength)) : 0.0f;
        const float liveOutActive = (switchClosed && breakerClosed) ? (acMode ? liveOutLength : std::clamp(signalDistance - liveFeedLength, 0.0f, liveOutLength)) : 0.0f;
        const float neutralActive = (switchClosed && breakerClosed && neutralIntact && !shortFaultArmed)
            ? (acMode ? neutralLength : std::clamp(signalDistance - bulbDistance, 0.0f, neutralLength))
            : 0.0f;
        const float groundFaultActive = (switchClosed && breakerClosed && shortFaultArmed && groundIntact)
            ? (acMode ? groundLength : std::clamp(signalDistance - liveFeedLength * 0.20f, 0.0f, groundLength))
            : 0.0f;
        const float outletLiveActive = breakerClosed ? outletLiveLength : 0.0f;
        const float outletNeutralActive = breakerClosed ? outletNeutralLength : 0.0f;
        const float outletGroundActive = (breakerClosed && groundIntact) ? outletGroundLength : 0.0f;

        if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)) {
            ProbeState candidate{};
            float bestDistance = 0.18f;
            const Ray pickRay = GetMouseRay(GetMousePosition(), camera);
            TryProbePath(pickRay, liveFeedPath, liveFeedLength, "live feed", Color{226, 144, 88, 255}, &candidate, &bestDistance);
            TryProbePath(pickRay, liveOutPath, liveOutLength, "switched live", Color{238, 166, 102, 255}, &candidate, &bestDistance);
            TryProbePath(pickRay, neutralPath, neutralLength, "neutral", neutralAccent, &candidate, &bestDistance);
            TryProbePath(pickRay, groundPath, groundLength, "ground", groundAccent, &candidate, &bestDistance);
            TryProbePath(pickRay, outletLivePath, outletLiveLength, "outlet live", Color{226, 144, 88, 255}, &candidate, &bestDistance);
            TryProbePath(pickRay, outletNeutralPath, outletNeutralLength, "outlet neutral", neutralAccent, &candidate, &bestDistance);
            TryProbePath(pickRay, outletGroundPath, outletGroundLength, "outlet ground", groundAccent, &candidate, &bestDistance);
            probe = candidate;
        }

        if (probe.active) {
            float magnitudeScale = 0.20f;
            if (probe.conductor == "live feed") magnitudeScale = liveFeedActive / std::max(0.001f, liveFeedLength);
            else if (probe.conductor == "switched live") magnitudeScale = liveOutActive / std::max(0.001f, liveOutLength);
            else if (probe.conductor == "neutral") magnitudeScale = neutralActive / std::max(0.001f, neutralLength);
            else if (probe.conductor == "ground") magnitudeScale = groundFaultActive / std::max(0.001f, groundLength);
            else if (probe.conductor == "outlet live") magnitudeScale = outletPower;
            else if (probe.conductor == "outlet neutral") magnitudeScale = outletPower * (neutralIntact ? 1.0f : 0.15f);
            else if (probe.conductor == "outlet ground") magnitudeScale = groundIntact ? outletPower * 0.35f : 0.05f;
            magnitudeScale = std::max(0.05f, magnitudeScale);
            UpdateProbeVectors(&probe, t, magnitudeScale, (acMode && acCarrier < 0.5f) ? -1.0f : 1.0f);
        }

        BeginDrawing();
        ClearBackground(Color{6, 8, 12, 255});

        BeginMode3D(camera);

        DrawBreakerPanel(breakerClosed, acMode, faultMode, tripProgress, bulbPower, outletPower);
        DrawSwitchPlate(switchClosed);
        DrawWirePath(liveFeedPath, liveFeedLength, viewMode, Color{226, 144, 88, 255});
        DrawWirePath(liveOutPath, liveOutLength, viewMode, Color{238, 166, 102, 255});
        DrawWirePath(neutralPath, neutralLength, viewMode, neutralAccent);
        DrawWirePath(groundPath, groundLength, viewMode, groundAccent);
        DrawWirePath(outletLivePath, outletLiveLength, viewMode, Color{226, 144, 88, 255});
        DrawWirePath(outletNeutralPath, outletNeutralLength, viewMode, neutralAccent);
        DrawWirePath(outletGroundPath, outletGroundLength, viewMode, groundAccent);

        const bool showCircuitFields = (viewMode != ViewMode::Normal) || liveFeedActive > 0.0f || liveOutActive > 0.0f || outletPower > 0.05f;
        if (showCircuitFields) {
            if (liveFeedActive > 0.0f) {
                DrawHelixPulseAlongPath(liveFeedPath, liveFeedActive, t + 0.12f, 1.18f);
                DrawFieldIndicatorsAlongPath(liveFeedPath, liveFeedActive, t + 0.12f);
            }
            if (liveOutActive > 0.0f) {
                DrawHelixPulseAlongPath(liveOutPath, liveOutActive, t, 1.15f);
                DrawFieldIndicatorsAlongPath(liveOutPath, liveOutActive, t);
                const PathSample front = SampleAlongPath(liveOutPath, liveOutActive);
                DrawSphere(front.pos, 0.11f, Color{255, 242, 165, 255});
                DrawSphere(front.pos, 0.22f, WithAlpha(Color{255, 240, 150, 255}, 36));
            }
            if (neutralActive > 0.0f) {
                DrawHelixPulseAlongPath(neutralPath, neutralActive, t + 0.65f, 1.05f);
                DrawFieldIndicatorsAlongPath(neutralPath, neutralActive, t + 0.65f);
                const PathSample front = SampleAlongPath(neutralPath, neutralActive);
                DrawSphere(front.pos, 0.09f, Color{188, 226, 255, 255});
                DrawSphere(front.pos, 0.18f, WithAlpha(Color{168, 214, 255, 255}, 28));
            }
            if (outletLiveActive > 0.0f) {
                DrawHelixPulseAlongPath(outletLivePath, outletLiveActive, t + 0.32f, 1.00f);
                if (viewMode != ViewMode::Normal) DrawFieldIndicatorsAlongPath(outletLivePath, outletLiveActive, t + 0.32f);
            }
            if (outletNeutralActive > 0.0f && neutralIntact) {
                DrawHelixPulseAlongPath(outletNeutralPath, outletNeutralActive, t + 0.92f, 0.95f);
            }
            if (groundFaultActive > 0.0f) {
                DrawHelixPulseAlongPath(groundPath, groundFaultActive, t + 1.20f, 0.88f);
                DrawFieldIndicatorsAlongPath(groundPath, groundFaultActive, t + 1.20f);
            }
            if (outletGroundActive > 0.0f && viewMode == ViewMode::Field && groundIntact) {
                DrawFieldIndicatorsAlongPath(outletGroundPath, outletGroundActive * 0.40f, t + 1.45f);
            }
        }

        if (viewMode == ViewMode::Field && switchClosed) {
            DrawArrow3D({-4.10f, 1.44f, 0.20f}, {-2.90f, 1.44f, 0.20f}, 0.018f, Color{255, 176, 116, 220});
            DrawArrow3D({-2.20f, 4.35f, 0.18f}, {1.86f, 4.35f, 0.18f}, 0.020f, Color{140, 255, 180, 215});
            DrawArrow3D({2.10f, 1.22f, -0.20f}, {-3.95f, 1.00f, -0.20f}, 0.020f, Color{148, 214, 255, 190});
            DrawArrow3D({2.22f, 1.02f, -0.36f}, {-3.92f, 0.76f, -0.36f}, 0.016f, Color{162, 228, 132, 120});
            DrawArrow3D({-3.42f, 0.86f, 0.24f}, {1.24f, 0.80f, 0.24f}, 0.016f, Color{255, 176, 116, 180});
        }

        if (viewMode == ViewMode::Cutaway) {
            for (float s = 0.7f; s < liveOutLength; s += 1.6f) {
                const PathSample sample = SampleAlongPath(liveOutPath, s);
                DrawLine3D(Vector3Add(sample.pos, {0.0f, 0.0f, -0.10f}), Vector3Add(sample.pos, {0.0f, 0.0f, 0.10f}),
                           WithAlpha(Color{190, 210, 230, 255}, 90));
            }
        }

        const Vector3 bulbBase = {2.9f, 2.72f, 0.04f};
        const Vector3 bulbCenter = {2.9f, 2.18f, 0.04f};
        DrawBulbAssembly(bulbBase, bulbCenter, filamentColor, bulbGlass, viewMode, bulbPower, t);
        DrawOutletAssembly({1.46f, 0.96f, 0.10f}, viewMode, outletPower, groundIntact, t);

        if (bulbPower > 0.08f) {
            for (int i = 0; i < 6; ++i) {
                const float a = 2.0f * PI * static_cast<float>(i) / 6.0f + 0.7f * t;
                Vector3 p = bulbCenter;
                p.y += 0.18f * std::sin(a);
                p.z += 0.18f * std::cos(a);
                DrawArrow3D(Vector3Add(p, {-0.38f, 0.0f, 0.0f}), p, 0.010f, Color{150, 255, 175, 210});
            }
        }

        if (switchClosed && breakerClosed && shortFaultArmed) {
            const Vector3 arcA = {2.84f, 2.62f, 0.05f};
            const Vector3 arcB = {2.90f, 2.50f, -0.09f};
            DrawLine3D(arcA, arcB, Color{255, 244, 180, 255});
            DrawSphere(Vector3Lerp(arcA, arcB, 0.5f), 0.10f, WithAlpha(Color{255, 188, 118, 255}, 170));
            DrawSphere(Vector3Lerp(arcA, arcB, 0.5f), 0.20f, WithAlpha(Color{255, 208, 132, 255}, 56));
        }

        DrawProbeGlyph(probe);

        EndMode3D();

        DrawText("House Circuit EM Visualizer", 22, 18, 28, Color{234, 238, 244, 255});
        DrawText("Hot feeds the switch and outlet. Neutral returns. Ground is safety only.", 22, 50, 18, Color{112, 128, 150, 255});

        DrawStatusBadge(22, 80, 112, breakerClosed ? "BREAKER ON" : "BREAKER TRIP", breakerClosed ? Color{255, 214, 140, 255} : Color{244, 126, 114, 255});
        DrawStatusBadge(142, 80, 96, switchClosed ? "SWITCH ON" : "SWITCH OFF", switchClosed ? Color{255, 214, 140, 255} : Color{172, 182, 196, 255});
        DrawStatusBadge(246, 80, 82, acMode ? "AC MODE" : "PULSE", acMode ? Color{120, 188, 255, 255} : Color{150, 255, 176, 255});
        DrawStatusBadge(336, 80, 148, TextFormat("FAULT %s", FaultModeLabel(faultMode)), faultMode == FaultMode::None ? Color{172, 182, 196, 255} : Color{244, 126, 114, 255});

        DrawStatusBadge(22, 114, 104, neutralIntact ? "NEUTRAL OK" : "NEUTRAL OPEN", neutralIntact ? Color{120, 188, 255, 255} : Color{244, 126, 114, 255});
        DrawStatusBadge(134, 114, 96, groundIntact ? "GROUND OK" : "GROUND OFF", groundIntact ? Color{150, 214, 120, 255} : Color{244, 126, 114, 255});
        DrawStatusBadge(238, 114, 100, bulbPower > 0.08f ? "LAMP LIVE" : "LAMP IDLE", bulbPower > 0.08f ? Color{255, 214, 140, 255} : Color{172, 182, 196, 255});
        DrawStatusBadge(346, 114, 110, outletPower > 0.10f ? "OUTLET LIVE" : "OUTLET OFF", outletPower > 0.10f ? Color{150, 255, 176, 255} : Color{172, 182, 196, 255});

        const std::string hud = HudText(switchClosed, breakerClosed, acMode, faultMode, signalDistance / std::max(0.01f, loopLength), bulbPower, paused);
        DrawText(hud.c_str(), 22, 154, 18, Color{84, 194, 230, 255});
        DrawText("Controls: F11 fullscreen   1/2/3 view   A AC   F fault   B reset breaker   RMB probe", 22, 178, 18, Color{130, 146, 168, 255});
        DrawText("Gestures: dual pinch move/zoom   right pinch switch   left pinch slow   left double pinch fast", 22, 202, 18, Color{130, 146, 168, 255});
        DrawText(bridgeStatus.c_str(), 22, 226, 18, Color{142, 255, 190, 255});

        DrawWorldCallout(camera, {-4.88f, 1.54f, 0.22f}, "breaker panel", Color{255, 214, 140, 255});
        DrawWorldCallout(camera, {-2.70f, 1.94f, 0.18f}, "switch cuts hot", Color{255, 214, 140, 255});
        DrawWorldCallout(camera, {2.92f, 2.84f, 0.24f}, "lamp load", Color{255, 214, 140, 255});
        DrawWorldCallout(camera, {1.70f, 1.46f, 0.14f}, "outlet + fan", Color{150, 255, 176, 255});

        if (probe.active) {
            DrawRectangleRounded({20.0f, 708.0f, 520.0f, 120.0f}, 0.05f, 10, Color{8, 12, 18, 220});
            DrawRectangleLinesEx({20.0f, 708.0f, 520.0f, 120.0f}, 1.2f, probe.accent);
            DrawText(TextFormat("probe: %s", probe.conductor.c_str()), 34, 722, 22, probe.accent);
            DrawText(TextFormat("E %.1f V/m   B %.3f T   S %.1f W/m^2", probe.eMag, probe.bMag, probe.sMag), 34, 752, 20, Color{214, 222, 236, 255});
            DrawText(TextFormat("position %.2f %.2f %.2f", probe.pos.x, probe.pos.y, probe.pos.z), 34, 780, 18, Color{130, 146, 168, 255});
        }

        const Rectangle panel{static_cast<float>(GetScreenWidth() - 302), 20.0f, 282.0f, 190.0f};
        DrawRectangleRounded(panel, 0.06f, 10, Color{8, 12, 18, 210});
        DrawRectangleLinesEx(panel, 1.5f, Color{92, 110, 138, 255});
        DrawText("Webcam Feed", static_cast<int>(panel.x) + 14, static_cast<int>(panel.y) + 12, 20, Color{222, 230, 244, 255});
        if (previewLive) {
            const Rectangle src = {0.0f, 0.0f, static_cast<float>(webcamTexture.width), static_cast<float>(webcamTexture.height)};
            const Rectangle dst = {panel.x + 12.0f, panel.y + 42.0f, panel.width - 24.0f, panel.height - 54.0f};
            DrawTexturePro(webcamTexture, src, dst, {0.0f, 0.0f}, 0.0f, WHITE);
        } else {
            DrawRectangle(static_cast<int>(panel.x) + 12, static_cast<int>(panel.y) + 42, static_cast<int>(panel.width) - 24, static_cast<int>(panel.height) - 54, Color{20, 24, 32, 255});
            DrawText(frameReceiverOk ? "waiting for preview" : "preview receiver failed",
                     static_cast<int>(panel.x) + 22,
                     static_cast<int>(panel.y) + 112,
                     18,
                     Color{180, 194, 214, 255});
        }

        DrawFPS(22, 250);
        EndDrawing();
    }

    frameReceiver.Close();
    if (webcamTexture.id > 0) UnloadTexture(webcamTexture);
    CloseWindow();
    return 0;
}
