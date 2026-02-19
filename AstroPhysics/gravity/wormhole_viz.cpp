#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr float kCameraDistanceMin = 1.6f;
constexpr float kCameraDistanceMax = 120.0f;
constexpr float kCameraPitchMin = -3.05f;
constexpr float kCameraPitchMax = 3.05f;
constexpr std::int64_t kControlStaleMs = 1200;
constexpr std::int64_t kPinchSequenceWindowMs = 650;
constexpr std::int64_t kZoomPinchSuppressMs = 260;
constexpr float kWarpStep = 0.02f;
constexpr float kThroatStep = 0.05f;
constexpr float kBridgePitchGain = 2.20f;

struct FlowParticle {
    float u;
    float theta;
    float speed;
    float swirl;
    Color color;
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
            // Ignore malformed values and keep defaults.
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

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0035f;
        *pitch = std::clamp(*pitch, kCameraPitchMin, kCameraPitchMax);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, kCameraDistanceMin, kCameraDistanceMax);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

void UpdateCameraFromOrbit(Camera3D* camera, float yaw, float pitch, float distance) {
    float cp = std::cos(pitch);
    Vector3 offset = {
        distance * cp * std::cos(yaw),
        distance * std::sin(pitch),
        distance * cp * std::sin(yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float RadiusProfile(float u, float throatRadius, float flare) {
    return throatRadius + flare * (u * u);
}

Vector3 WormholePoint(float u, float theta, float throatRadius, float flare) {
    float r = RadiusProfile(u, throatRadius, flare);
    return {r * std::cos(theta), r * std::sin(theta), u};
}

void DrawWormholeSurface(float throatRadius, float flare) {
    const int rings = 52;
    const int segs = 64;
    const float uMin = -4.8f;
    const float uMax = 4.8f;

    for (int i = 0; i < rings - 1; ++i) {
        float u0 = uMin + (uMax - uMin) * static_cast<float>(i) / static_cast<float>(rings - 1);
        float u1 = uMin + (uMax - uMin) * static_cast<float>(i + 1) / static_cast<float>(rings - 1);
        for (int j = 0; j < segs; ++j) {
            float t0 = 2.0f * PI * static_cast<float>(j) / static_cast<float>(segs);
            float t1 = 2.0f * PI * static_cast<float>(j + 1) / static_cast<float>(segs);

            Vector3 p00 = WormholePoint(u0, t0, throatRadius, flare);
            Vector3 p01 = WormholePoint(u0, t1, throatRadius, flare);
            Vector3 p10 = WormholePoint(u1, t0, throatRadius, flare);

            float glow = 0.22f + 0.58f * (1.0f - std::fabs(u0) / 4.8f);
            Color c = Color{
                static_cast<unsigned char>(70 + 70 * glow),
                static_cast<unsigned char>(110 + 90 * glow),
                static_cast<unsigned char>(170 + 70 * glow),
                static_cast<unsigned char>(40 + 80 * glow)
            };

            DrawTriangle3D(p00, p10, p01, c);
        }
    }

    for (int k = 0; k < 10; ++k) {
        float u = -4.8f + 9.6f * static_cast<float>(k) / 9.0f;
        int segCount = 80;
        for (int j = 0; j < segCount; ++j) {
            float a0 = 2.0f * PI * static_cast<float>(j) / static_cast<float>(segCount);
            float a1 = 2.0f * PI * static_cast<float>(j + 1) / static_cast<float>(segCount);
            Vector3 p0 = WormholePoint(u, a0, throatRadius, flare);
            Vector3 p1 = WormholePoint(u, a1, throatRadius, flare);
            DrawLine3D(p0, p1, Color{90, 150, 220, 70});
        }
    }
}

std::string Hud(float throatRadius, float flare, int particles, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "throat=" << throatRadius
       << "  flare=" << flare
       << "  particles=" << particles;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Wormhole 3D Visualization - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.5f, 4.8f, 8.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f;
    float camPitch = 0.34f;
    float camDistance = 13.5f;

    float throatRadius = 0.85f;
    float flare = 0.22f;
    bool paused = false;
    bool hasPrevLive = false;
    float prevLiveZoom = 1.0f;
    float prevLiveRotDeg = 0.0f;
    float prevLivePitchDeg = 0.0f;
    int prevLiveNIncCount = 0;
    int prevLiveNDecCount = 0;
    int pendingRightPinches = 0;
    int pendingLeftPinches = 0;
    std::int64_t rightPendingDeadlineMs = 0;
    std::int64_t leftPendingDeadlineMs = 0;
    std::int64_t zoomPinchSuppressUntilMs = 0;
    std::string bridgeStatus = "bridge: waiting for AstroPhysics/vision/live_controls.txt";

    std::vector<FlowParticle> flow;
    flow.reserve(320);
    for (int i = 0; i < 320; ++i) {
        float u = -4.7f + 9.4f * (static_cast<float>(i) / 319.0f);
        float theta = 2.0f * PI * std::fmod(i * 0.6180339f, 1.0f);
        float speed = 0.4f + 0.9f * std::fmod(i * 0.371f, 1.0f);
        float swirl = 0.8f + 1.4f * std::fmod(i * 0.529f, 1.0f);
        Color c = Color{
            static_cast<unsigned char>(90 + (i * 17) % 120),
            static_cast<unsigned char>(160 + (i * 11) % 90),
            static_cast<unsigned char>(220 + (i * 7) % 35),
            230
        };
        flow.push_back({u, theta, speed, swirl, c});
    }

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            throatRadius = 0.85f;
            flare = 0.22f;
        }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) throatRadius = std::max(0.45f, throatRadius - 0.05f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) throatRadius = std::min(1.8f, throatRadius + 0.05f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) flare = std::max(0.08f, flare - 0.02f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) flare = std::min(0.55f, flare + 0.02f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        const std::int64_t nowMs = UnixMsNow();
        if (auto live = LoadLiveControls()) {
            const std::int64_t ageMs = nowMs - live->timestampMs;
            if (ageMs <= kControlStaleMs) {
                if (!hasPrevLive) {
                    hasPrevLive = true;
                    prevLiveZoom = std::max(0.25f, live->zoom);
                    prevLiveRotDeg = live->rotationDeg;
                    prevLivePitchDeg = live->pitchDeg;
                    prevLiveNIncCount = live->nIncCount;
                    prevLiveNDecCount = live->nDecCount;
                } else {
                    const float currentLiveZoom = std::max(0.25f, live->zoom);
                    float zoomRatio = currentLiveZoom / std::max(0.25f, prevLiveZoom);
                    zoomRatio = std::clamp(zoomRatio, 0.65f, 1.55f);
                    camDistance = std::clamp(camDistance / zoomRatio, kCameraDistanceMin, kCameraDistanceMax);
                    prevLiveZoom = currentLiveZoom;

                    const float rotDeltaDeg = AngleDeltaDeg(live->rotationDeg, prevLiveRotDeg);
                    prevLiveRotDeg = live->rotationDeg;
                    camYaw += rotDeltaDeg * DEG2RAD;

                    const float pitchDeltaDeg = live->pitchDeg - prevLivePitchDeg;
                    prevLivePitchDeg = live->pitchDeg;
                    camPitch += pitchDeltaDeg * kBridgePitchGain * DEG2RAD;
                    camPitch = std::clamp(camPitch, kCameraPitchMin, kCameraPitchMax);

                    if (live->zoomLineActive) {
                        zoomPinchSuppressUntilMs = nowMs + kZoomPinchSuppressMs;
                        pendingRightPinches = 0;
                        pendingLeftPinches = 0;
                        rightPendingDeadlineMs = 0;
                        leftPendingDeadlineMs = 0;
                    }

                    const bool allowPinchActions = (nowMs >= zoomPinchSuppressUntilMs) && !live->zoomLineActive;
                    if (allowPinchActions) {
                        if (live->nIncCount >= prevLiveNIncCount) {
                            const int incDelta = live->nIncCount - prevLiveNIncCount;
                            if (incDelta > 0) {
                                pendingRightPinches += incDelta;
                                rightPendingDeadlineMs = nowMs + kPinchSequenceWindowMs;
                            }
                        }
                        if (live->nDecCount >= prevLiveNDecCount) {
                            const int decDelta = live->nDecCount - prevLiveNDecCount;
                            if (decDelta > 0) {
                                pendingLeftPinches += decDelta;
                                leftPendingDeadlineMs = nowMs + kPinchSequenceWindowMs;
                            }
                        }
                    }
                    prevLiveNIncCount = live->nIncCount;
                    prevLiveNDecCount = live->nDecCount;
                }

                UpdateCameraFromOrbit(&camera, camYaw, camPitch, camDistance);
                std::ostringstream cs;
                cs << "bridge: live  hand=" << live->label
                   << "  gesture=" << live->gesture
                   << "  age=" << ageMs << "ms  single=flare  double=throat";
                bridgeStatus = cs.str();
            } else {
                hasPrevLive = false;
                bridgeStatus = "bridge: stale";
            }
        } else {
            hasPrevLive = false;
            bridgeStatus = "bridge: waiting for AstroPhysics/vision/live_controls.txt";
        }

        auto applyPinchBuffer = [&](int* pending, std::int64_t* deadline, float flareDir, float throatDir) {
            if (*pending >= 2) {
                const int doubleCount = *pending / 2;
                throatRadius = std::clamp(
                    throatRadius + throatDir * kThroatStep * static_cast<float>(doubleCount),
                    0.45f,
                    1.8f
                );
                *pending %= 2;
                *deadline = (*pending > 0) ? (nowMs + kPinchSequenceWindowMs) : 0;
            }

            if (*pending == 1 && *deadline > 0 && nowMs >= *deadline) {
                flare = std::clamp(flare + flareDir * kWarpStep, 0.08f, 0.55f);
                *pending = 0;
                *deadline = 0;
            }
        };

        applyPinchBuffer(&pendingRightPinches, &rightPendingDeadlineMs, +1.0f, +1.0f);
        applyPinchBuffer(&pendingLeftPinches, &leftPendingDeadlineMs, -1.0f, -1.0f);

        float dt = GetFrameTime();
        if (!paused) {
            for (FlowParticle& p : flow) {
                p.u += dt * (0.35f + p.speed);
                p.theta += dt * p.swirl;
                if (p.u > 4.8f) p.u = -4.8f;
            }
        }

        BeginDrawing();
        ClearBackground(Color{5, 8, 18, 255});

        BeginMode3D(camera);


        DrawWormholeSurface(throatRadius, flare);

        DrawSphere({0.0f, 0.0f, -4.8f}, RadiusProfile(-4.8f, throatRadius, flare), Color{80, 120, 190, 35});
        DrawSphere({0.0f, 0.0f, 4.8f}, RadiusProfile(4.8f, throatRadius, flare), Color{80, 120, 190, 35});

        for (const FlowParticle& p : flow) {
            Vector3 pos = WormholePoint(p.u, p.theta, throatRadius, flare);
            DrawSphere(pos, 0.03f, p.color);
        }

        EndMode3D();

        DrawText("Wormhole Tunnel (Morris-Thorne Style Visual)", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] throat | +/- flare | P pause | R reset", 20, 54, 19, Color{164, 183, 210, 255});
        std::string hud = Hud(throatRadius, flare, static_cast<int>(flow.size()), paused);
        DrawText(hud.c_str(), 20, 82, 21, Color{126, 224, 255, 255});
        DrawText(bridgeStatus.c_str(), 20, 108, 19, Color{152, 234, 198, 255});
        DrawFPS(20, 136);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
