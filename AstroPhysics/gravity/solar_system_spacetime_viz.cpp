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
constexpr float kExtent = 24.0f;
constexpr int kGrid = 60;
constexpr int kTrailMax = 280;
constexpr std::int64_t kControlStaleMs = 1200;
constexpr std::int64_t kPinchSequenceWindowMs = 850;
constexpr std::int64_t kZoomPinchSuppressMs = 260;
constexpr float kSpeedStep = 0.25f;
constexpr float kWarpStep = 0.05f;

struct Planet {
    const char* name;
    float orbitRadius;
    float orbitOmega;
    float radius;
    float warpStrength;
    float warpCore;
    float phase0;
    Color color;
    Vector3 pos;
    std::deque<Vector3> trail;
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

    *distance -= GetMouseWheelMove() * 0.75f;
    *distance = std::clamp(*distance, 9.0f, 72.0f);

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

float SpacetimeHeight(float x, float z, const Vector3& sunPos, const std::vector<Planet>& planets, float scale) {
    float h = WarpContribution(x - sunPos.x, z - sunPos.z, 5.2f, 0.95f);
    for (const Planet& p : planets) {
        h += WarpContribution(x - p.pos.x, z - p.pos.z, p.warpStrength, p.warpCore);
    }
    return std::max(-7.2f, h * scale);
}

void DrawSpacetimeSheet(const Vector3& sunPos, const std::vector<Planet>& planets, float scale) {
    for (int i = 0; i < kGrid; ++i) {
        float z = -kExtent + 2.0f * kExtent * static_cast<float>(i) / static_cast<float>(kGrid - 1);
        for (int j = 0; j < kGrid - 1; ++j) {
            float x0 = -kExtent + 2.0f * kExtent * static_cast<float>(j) / static_cast<float>(kGrid - 1);
            float x1 = -kExtent + 2.0f * kExtent * static_cast<float>(j + 1) / static_cast<float>(kGrid - 1);
            Vector3 p0 = {x0, SpacetimeHeight(x0, z, sunPos, planets, scale), z};
            Vector3 p1 = {x1, SpacetimeHeight(x1, z, sunPos, planets, scale), z};
            float glow = 1.0f - std::min(1.0f, std::fabs(p0.y) / 7.2f);
            Color c = {
                static_cast<unsigned char>(46 + 54 * glow),
                static_cast<unsigned char>(110 + 85 * glow),
                static_cast<unsigned char>(178 + 68 * glow),
                static_cast<unsigned char>(70 + 85 * glow),
            };
            DrawLine3D(p0, p1, c);
        }
    }

    for (int j = 0; j < kGrid; ++j) {
        float x = -kExtent + 2.0f * kExtent * static_cast<float>(j) / static_cast<float>(kGrid - 1);
        for (int i = 0; i < kGrid - 1; ++i) {
            float z0 = -kExtent + 2.0f * kExtent * static_cast<float>(i) / static_cast<float>(kGrid - 1);
            float z1 = -kExtent + 2.0f * kExtent * static_cast<float>(i + 1) / static_cast<float>(kGrid - 1);
            Vector3 p0 = {x, SpacetimeHeight(x, z0, sunPos, planets, scale), z0};
            Vector3 p1 = {x, SpacetimeHeight(x, z1, sunPos, planets, scale), z1};
            float glow = 1.0f - std::min(1.0f, std::fabs(p0.y) / 7.2f);
            Color c = {
                static_cast<unsigned char>(40 + 48 * glow),
                static_cast<unsigned char>(96 + 80 * glow),
                static_cast<unsigned char>(165 + 75 * glow),
                static_cast<unsigned char>(56 + 76 * glow),
            };
            DrawLine3D(p0, p1, c);
        }
    }
}

void DrawOrbitRing(float radius, Color c) {
    constexpr int kSegs = 140;
    for (int i = 0; i < kSegs; ++i) {
        float a0 = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(kSegs);
        float a1 = 2.0f * kPi * static_cast<float>(i + 1) / static_cast<float>(kSegs);
        DrawLine3D(
            {radius * std::cos(a0), 0.0f, radius * std::sin(a0)},
            {radius * std::cos(a1), 0.0f, radius * std::sin(a1)},
            c
        );
    }
}

void DrawTrail(const std::deque<Vector3>& trail, Color color) {
    if (trail.size() < 2) return;
    for (size_t i = 1; i < trail.size(); ++i) {
        float fade = static_cast<float>(i) / static_cast<float>(trail.size());
        Color c = color;
        c.a = static_cast<unsigned char>(20 + 130 * fade);
        DrawLine3D(trail[i - 1], trail[i], c);
    }
}

std::string Hud(float simTimeYears, float speed, float warpScale, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "t=" << simTimeYears << " years"
       << "  speed=" << speed << "x"
       << "  warp=" << warpScale;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Solar System Spacetime Curvature 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {20.0f, 14.0f, 18.0f};
    camera.target = {0.0f, -1.4f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.79f;
    float camPitch = 0.42f;
    float camDistance = 30.0f;

    Vector3 sunPos = {0.0f, 0.95f, 0.0f};

    std::vector<Planet> planets = {
        {"Mercury", 2.2f, 8.2f, 0.12f, 0.10f, 0.15f, 0.2f, Color{205, 190, 165, 255}},
        {"Venus",   3.4f, 5.3f, 0.18f, 0.13f, 0.18f, 1.2f, Color{220, 180, 110, 255}},
        {"Earth",   4.6f, 4.1f, 0.19f, 0.14f, 0.18f, 2.1f, Color{105, 165, 255, 255}},
        {"Mars",    6.0f, 3.0f, 0.15f, 0.11f, 0.16f, 2.8f, Color{220, 110, 85, 255}},
        {"Jupiter", 8.7f, 1.7f, 0.45f, 0.34f, 0.30f, 0.6f, Color{220, 175, 120, 255}},
        {"Saturn",  11.8f, 1.2f, 0.40f, 0.30f, 0.28f, 1.8f, Color{220, 200, 130, 255}},
        {"Uranus",  15.3f, 0.82f, 0.30f, 0.22f, 0.24f, 2.7f, Color{145, 220, 225, 255}},
        {"Neptune", 18.7f, 0.64f, 0.29f, 0.22f, 0.24f, 0.9f, Color{100, 150, 255, 255}},
    };

    float simTimeYears = 0.0f;
    float speed = 1.0f;
    float warpScale = 1.0f;
    bool paused = false;
    bool showTrails = true;
    bool showLabels = true;
    bool showHelp = true;
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
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            simTimeYears = 0.0f;
            speed = 1.0f;
            warpScale = 1.0f;
            paused = false;
            for (Planet& p : planets) p.trail.clear();
        }
        if (IsKeyPressed(KEY_T)) showTrails = !showTrails;
        if (IsKeyPressed(KEY_L)) showLabels = !showLabels;
        if (IsKeyPressed(KEY_H)) showHelp = !showHelp;
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) speed = std::max(0.25f, speed - 0.25f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) speed = std::min(10.0f, speed + 0.25f);
        if (IsKeyPressed(KEY_LEFT_BRACKET)) warpScale = std::max(0.35f, warpScale - 0.05f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) warpScale = std::min(1.8f, warpScale + 0.05f);

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
                    camDistance = std::clamp(camDistance / zoomRatio, 9.0f, 72.0f);
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
                speed = std::clamp(speed + speedDir * kSpeedStep * static_cast<float>(doubleCount), 0.25f, 10.0f);
                *pending %= 2;
                *deadline = (*pending > 0) ? (nowMs + kPinchSequenceWindowMs) : 0;
            }

            if (*pending == 1 && *deadline > 0 && nowMs >= *deadline) {
                warpScale = std::clamp(warpScale + warpDir * kWarpStep, 0.35f, 1.8f);
                *pending = 0;
                *deadline = 0;
            }
        };

        applyPinchBuffer(&pendingRightPinches, &rightPendingDeadlineMs, +1.0f, +1.0f);
        applyPinchBuffer(&pendingLeftPinches, &leftPendingDeadlineMs, -1.0f, -1.0f);

        float dtYears = GetFrameTime() * speed * 0.38f;
        if (!paused) simTimeYears += dtYears;

        for (Planet& p : planets) {
            float ang = p.phase0 + simTimeYears * p.orbitOmega;
            p.pos = {p.orbitRadius * std::cos(ang), 0.26f, p.orbitRadius * std::sin(ang)};
            p.trail.push_back(p.pos);
            if (static_cast<int>(p.trail.size()) > kTrailMax) p.trail.pop_front();
        }

        BeginDrawing();
        ClearBackground(Color{5, 8, 18, 255});

        BeginMode3D(camera);

        DrawSpacetimeSheet(sunPos, planets, warpScale);

        for (const Planet& p : planets) {
            DrawOrbitRing(p.orbitRadius, Color{120, 145, 190, 40});
        }

        float sunSheetY = SpacetimeHeight(sunPos.x, sunPos.z, sunPos, planets, warpScale);
        DrawLine3D({sunPos.x, sunSheetY, sunPos.z}, sunPos, Color{255, 200, 115, 120});
        DrawSphere({sunPos.x, sunSheetY, sunPos.z}, 0.2f, Color{255, 180, 95, 80});

        DrawSphere(sunPos, 1.05f, Color{255, 198, 95, 255});
        DrawSphereWires(sunPos, 1.22f, 18, 18, Color{255, 214, 130, 100});

        for (const Planet& p : planets) {
            float py = SpacetimeHeight(p.pos.x, p.pos.z, sunPos, planets, warpScale);
            DrawLine3D({p.pos.x, py, p.pos.z}, p.pos, Color{140, 210, 255, 80});
            DrawSphere({p.pos.x, py, p.pos.z}, p.radius * 0.35f, Color{110, 175, 255, 70});
            DrawSphere(p.pos, p.radius, p.color);
            if (showTrails) DrawTrail(p.trail, p.color);
        }

        EndMode3D();

        if (showLabels) {
            for (const Planet& p : planets) {
                Vector2 s = GetWorldToScreen({p.pos.x, p.pos.y + p.radius + 0.2f, p.pos.z}, camera);
                DrawText(p.name, static_cast<int>(s.x), static_cast<int>(s.y), 16, Color{220, 230, 245, 230});
            }
        }

        DrawText("Solar System with Combined Spacetime Curvature", 20, 18, 30, Color{232, 238, 248, 255});
        if (showHelp) {
            DrawText("Hold left mouse: orbit | wheel: zoom | +/- speed | [ ] warp | P pause | R reset | T trails | L labels | H help",
                     20, 54, 19, Color{164, 183, 210, 255});
        } else {
            DrawText("Press H to show controls", 20, 54, 19, Color{164, 183, 210, 255});
        }
        std::string hud = Hud(simTimeYears, speed, warpScale, paused);
        DrawText(hud.c_str(), 20, 82, 21, Color{126, 224, 255, 255});
        DrawText(bridgeStatus.c_str(), 20, 108, 19, Color{152, 234, 198, 255});
        DrawFPS(20, 136);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
