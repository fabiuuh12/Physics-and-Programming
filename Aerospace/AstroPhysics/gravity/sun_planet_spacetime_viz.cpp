#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
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

constexpr float kPi = 3.14159265358979323846f;
constexpr std::int64_t kControlStaleMs = 1200;
constexpr std::int64_t kPinchSequenceWindowMs = 850;
constexpr std::int64_t kZoomPinchSuppressMs = 260;
constexpr float kSpeedStep = 0.25f;
constexpr float kWarpStep = 0.05f;

struct SimState {
    float angle;
    float speed;
    bool paused;
    bool showHelp;
    float sheetScale;
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
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.7f;
    *distance = std::clamp(*distance, 6.0f, 40.0f);

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

float WarpContribution(float dx, float dz, float strength, float core) {
    float r2 = dx * dx + dz * dz + core * core;
    return -strength / std::sqrt(r2);
}

float SpacetimeHeight(float x, float z, Vector3 sunPos, Vector3 planetPos, float sheetScale) {
    float sunWell = WarpContribution(x - sunPos.x, z - sunPos.z, 2.3f, 0.55f);
    float planetWell = WarpContribution(x - planetPos.x, z - planetPos.z, 0.8f, 0.28f);
    float h = (sunWell + planetWell) * sheetScale;
    return std::max(-4.6f, h);
}

void DrawSpacetimeSheet(Vector3 sunPos, Vector3 planetPos, float sheetScale) {
    constexpr int kGrid = 54;
    constexpr float kExtent = 11.0f;

    for (int i = 0; i < kGrid; ++i) {
        float z = -kExtent + 2.0f * kExtent * static_cast<float>(i) / static_cast<float>(kGrid - 1);
        for (int j = 0; j < kGrid - 1; ++j) {
            float x0 = -kExtent + 2.0f * kExtent * static_cast<float>(j) / static_cast<float>(kGrid - 1);
            float x1 = -kExtent + 2.0f * kExtent * static_cast<float>(j + 1) / static_cast<float>(kGrid - 1);
            Vector3 p0 = {x0, SpacetimeHeight(x0, z, sunPos, planetPos, sheetScale), z};
            Vector3 p1 = {x1, SpacetimeHeight(x1, z, sunPos, planetPos, sheetScale), z};
            float glow = 1.0f - std::min(1.0f, std::fabs(p0.y) / 4.6f);
            Color c = {
                static_cast<unsigned char>(50 + 60 * glow),
                static_cast<unsigned char>(110 + 80 * glow),
                static_cast<unsigned char>(185 + 50 * glow),
                static_cast<unsigned char>(120 + 80 * glow),
            };
            DrawLine3D(p0, p1, c);
        }
    }

    for (int j = 0; j < kGrid; ++j) {
        float x = -kExtent + 2.0f * kExtent * static_cast<float>(j) / static_cast<float>(kGrid - 1);
        for (int i = 0; i < kGrid - 1; ++i) {
            float z0 = -kExtent + 2.0f * kExtent * static_cast<float>(i) / static_cast<float>(kGrid - 1);
            float z1 = -kExtent + 2.0f * kExtent * static_cast<float>(i + 1) / static_cast<float>(kGrid - 1);
            Vector3 p0 = {x, SpacetimeHeight(x, z0, sunPos, planetPos, sheetScale), z0};
            Vector3 p1 = {x, SpacetimeHeight(x, z1, sunPos, planetPos, sheetScale), z1};
            float glow = 1.0f - std::min(1.0f, std::fabs(p0.y) / 4.6f);
            Color c = {
                static_cast<unsigned char>(45 + 55 * glow),
                static_cast<unsigned char>(100 + 85 * glow),
                static_cast<unsigned char>(170 + 65 * glow),
                static_cast<unsigned char>(95 + 65 * glow),
            };
            DrawLine3D(p0, p1, c);
        }
    }
}

void DrawOrbitTrail(const std::deque<Vector3>& trail) {
    if (trail.size() < 2) return;
    for (size_t i = 1; i < trail.size(); ++i) {
        float fade = static_cast<float>(i) / static_cast<float>(trail.size());
        Color c = {130, 205, 255, static_cast<unsigned char>(35 + 180 * fade)};
        DrawLine3D(trail[i - 1], trail[i], c);
    }
}

std::string Hud(const SimState& s, float period, float radiusA, float radiusB) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "speed=" << s.speed << "x"
       << "  period~" << period
       << "s  orbit=(" << radiusA << "," << radiusB << ")"
       << "  warp=" << s.sheetScale;
    if (s.paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Sun-Planet Spacetime Curvature 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {12.0f, 8.5f, 11.0f};
    camera.target = {0.0f, -1.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.77f;
    float camPitch = 0.44f;
    float camDistance = 17.0f;

    SimState sim{};
    sim.angle = 0.0f;
    sim.speed = 1.0f;
    sim.paused = false;
    sim.showHelp = true;
    sim.sheetScale = 1.0f;

    constexpr float kOrbitA = 6.2f;
    constexpr float kOrbitB = 5.3f;
    constexpr float kOmega = 0.42f;
    constexpr int kTrailMax = 900;

    std::deque<Vector3> trail;
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

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) sim.paused = !sim.paused;
        if (IsKeyPressed(KEY_R)) {
            sim.angle = 0.0f;
            sim.speed = 1.0f;
            sim.sheetScale = 1.0f;
            sim.paused = false;
            trail.clear();
        }
        if (IsKeyPressed(KEY_H)) sim.showHelp = !sim.showHelp;
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) sim.speed = std::max(0.25f, sim.speed - 0.25f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) sim.speed = std::min(8.0f, sim.speed + 0.25f);
        if (IsKeyPressed(KEY_LEFT_BRACKET)) sim.sheetScale = std::max(0.45f, sim.sheetScale - 0.05f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) sim.sheetScale = std::min(1.65f, sim.sheetScale + 0.05f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);
        const std::int64_t nowMs = UnixMsNow();
        if (auto live = LoadLiveControls()) {
            const std::int64_t ageMs = nowMs - live->timestampMs;
            if (ageMs <= kControlStaleMs) {
                if (!hasPrevLive) {
                    hasPrevLive = true;
                    prevLiveZoom = std::max(0.05f, live->zoom);
                    prevLiveRotDeg = live->rotationDeg;
                    prevLivePitchDeg = live->pitchDeg;
                    prevLiveNIncCount = live->nIncCount;
                    prevLiveNDecCount = live->nDecCount;
                } else {
                    const float currentLiveZoom = std::max(0.05f, live->zoom);
                    float zoomRatio = currentLiveZoom / std::max(0.05f, prevLiveZoom);
                    zoomRatio = std::clamp(zoomRatio, 0.65f, 1.55f);
                    camDistance = std::clamp(camDistance / zoomRatio, 6.0f, 40.0f);
                    prevLiveZoom = currentLiveZoom;

                    const float rotDeltaDeg = AngleDeltaDeg(live->rotationDeg, prevLiveRotDeg);
                    prevLiveRotDeg = live->rotationDeg;
                    camYaw += rotDeltaDeg * DEG2RAD;

                    const float pitchDeltaDeg = live->pitchDeg - prevLivePitchDeg;
                    prevLivePitchDeg = live->pitchDeg;
                    camPitch += pitchDeltaDeg * DEG2RAD;
                    camPitch = std::clamp(camPitch, -1.35f, 1.35f);

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
                   << "  age=" << ageMs << "ms";
                bridgeStatus = cs.str();
            } else {
                hasPrevLive = false;
                bridgeStatus = "bridge: stale";
            }
        } else {
            hasPrevLive = false;
            bridgeStatus = "bridge: waiting for AstroPhysics/vision/live_controls.txt";
        }

        auto applyPinchBuffer = [&](int* pending, std::int64_t* deadline, float warpDir, float speedDir) {
            if (*pending >= 2) {
                const int doubleCount = *pending / 2;
                sim.speed = std::clamp(sim.speed + speedDir * kSpeedStep * static_cast<float>(doubleCount), 0.25f, 8.0f);
                *pending %= 2;
                *deadline = (*pending > 0) ? (nowMs + kPinchSequenceWindowMs) : 0;
            }

            if (*pending == 1 && *deadline > 0 && nowMs >= *deadline) {
                sim.sheetScale = std::clamp(sim.sheetScale + warpDir * kWarpStep, 0.45f, 1.65f);
                *pending = 0;
                *deadline = 0;
            }
        };

        applyPinchBuffer(&pendingRightPinches, &rightPendingDeadlineMs, +1.0f, +1.0f);
        applyPinchBuffer(&pendingLeftPinches, &leftPendingDeadlineMs, -1.0f, -1.0f);

        float dt = GetFrameTime() * sim.speed;
        if (!sim.paused) {
            sim.angle += kOmega * dt;
        }

        Vector3 sunPos = {0.0f, 0.58f, 0.0f};
        Vector3 planetPos = {
            kOrbitA * std::cos(sim.angle),
            0.22f,
            kOrbitB * std::sin(sim.angle),
        };

        trail.push_back(planetPos);
        if (static_cast<int>(trail.size()) > kTrailMax) trail.pop_front();

        float sunSheetY = SpacetimeHeight(sunPos.x, sunPos.z, sunPos, planetPos, sim.sheetScale);
        float planetSheetY = SpacetimeHeight(planetPos.x, planetPos.z, sunPos, planetPos, sim.sheetScale);

        BeginDrawing();
        ClearBackground(Color{5, 8, 18, 255});

        BeginMode3D(camera);

        DrawSpacetimeSheet(sunPos, planetPos, sim.sheetScale);
        DrawOrbitTrail(trail);

        DrawLine3D({sunPos.x, sunSheetY, sunPos.z}, sunPos, Color{255, 195, 115, 120});
        DrawLine3D({planetPos.x, planetSheetY, planetPos.z}, planetPos, Color{140, 210, 255, 130});

        DrawSphere({sunPos.x, sunSheetY, sunPos.z}, 0.15f, Color{255, 180, 75, 70});
        DrawSphere({planetPos.x, planetSheetY, planetPos.z}, 0.08f, Color{100, 165, 240, 90});

        DrawSphere(sunPos, 0.78f, Color{255, 196, 95, 255});
        DrawSphereWires(sunPos, 0.93f, 16, 16, Color{255, 210, 130, 90});

        DrawSphere(planetPos, 0.26f, Color{95, 160, 255, 255});
        DrawSphereWires(planetPos, 0.31f, 10, 10, Color{180, 220, 255, 100});

        EndMode3D();

        DrawText("Planet Orbiting the Sun with Spacetime Curvature", 20, 18, 30, Color{232, 238, 248, 255});
        if (sim.showHelp) {
            DrawText("Hold left mouse: orbit camera | wheel: zoom | +/- speed | [ ] warp | P pause | R reset | H help",
                     20, 56, 19, Color{164, 183, 210, 255});
            DrawText("Webcam bridge: pinch-line zoom+camera | right single/double: warp+/speed+ | left single/double: warp-/speed-",
                     20, 80, 19, Color{164, 215, 198, 255});
        } else {
            DrawText("Press H to show controls", 20, 56, 19, Color{164, 183, 210, 255});
        }

        float orbitalPeriod = 2.0f * kPi / kOmega;
        std::string hud = Hud(sim, orbitalPeriod, kOrbitA, kOrbitB);
        DrawText(hud.c_str(), 20, 108, 21, Color{126, 224, 255, 255});
        DrawText(bridgeStatus.c_str(), 20, 134, 19, Color{152, 234, 198, 255});
        DrawFPS(20, 160);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
