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
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

constexpr float kEventHorizonRadius = 0.65f;
constexpr float kPhotonRingRadius = 1.15f;
constexpr float kDiskInnerRadius = 1.35f;
constexpr float kDiskOuterRadius = 4.5f;
constexpr float kGravityMu = 12.0f;
constexpr float kSheetExtent = 11.0f;
constexpr int kSheetGrid = 52;
constexpr int kMatterStep = 100;
constexpr float kSpeedStep = 0.25f;
constexpr std::int64_t kControlStaleMs = 1200;
constexpr std::int64_t kPinchSequenceWindowMs = 650;

struct DustParticle {
    Vector3 pos;
    Vector3 vel;
    float size;
    float heat;
};

struct LiveControls {
    float zoom = 1.0f;
    float rotationDeg = 0.0f;
    float pitchDeg = 0.0f;
    int nIncCount = 0;
    int nDecCount = 0;
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

float RandRange(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(rng);
}

DustParticle SpawnDust(std::mt19937& rng) {
    const float r = RandRange(rng, kDiskInnerRadius, kDiskOuterRadius);
    const float phi = RandRange(rng, 0.0f, 2.0f * PI);
    const float thickness = RandRange(rng, -0.15f, 0.15f);

    Vector3 pos = {r * std::cos(phi), thickness, r * std::sin(phi)};

    const float orbitalSpeed = std::sqrt(kGravityMu / r);
    Vector3 tangent = {-std::sin(phi), 0.0f, std::cos(phi)};
    Vector3 vel = Vector3Scale(tangent, orbitalSpeed * RandRange(rng, 0.92f, 1.06f));

    Vector3 inward = Vector3Normalize(Vector3Negate(pos));
    vel = Vector3Add(vel, Vector3Scale(inward, RandRange(rng, 0.03f, 0.1f)));
    vel.y += RandRange(rng, -0.05f, 0.05f);

    const float size = RandRange(rng, 0.03f, 0.07f);
    const float heat = RandRange(rng, 0.45f, 1.0f);
    return {pos, vel, size, heat};
}

void ResetDisk(std::vector<DustParticle>* disk, std::mt19937& rng, int count) {
    disk->clear();
    disk->reserve(count);
    for (int i = 0; i < count; ++i) {
        disk->push_back(SpawnDust(rng));
    }
}

Color DiskColor(float heat) {
    const float h = std::clamp(heat, 0.0f, 1.0f);
    const unsigned char r = static_cast<unsigned char>(180 + 70 * h);
    const unsigned char g = static_cast<unsigned char>(100 + 90 * h);
    const unsigned char b = static_cast<unsigned char>(45 + 35 * (1.0f - h));
    return Color{r, g, b, 220};
}

std::string HudText(float t, float speed, int particles, int swallowed, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "t=" << t
       << "  speed=" << speed << "x"
       << "  particles=" << particles
       << "  swallowed=" << swallowed;
    if (paused) {
        os << "  [PAUSED]";
    }
    return os.str();
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 3.0f, 26.0f);

    const float cp = std::cos(*pitch);
    const Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
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

void DrawCircle3DXZ(float radius, int segments, Color color) {
    for (int i = 0; i < segments; ++i) {
        const float a0 = 2.0f * PI * static_cast<float>(i) / static_cast<float>(segments);
        const float a1 = 2.0f * PI * static_cast<float>(i + 1) / static_cast<float>(segments);
        DrawLine3D(
            {radius * std::cos(a0), 0.0f, radius * std::sin(a0)},
            {radius * std::cos(a1), 0.0f, radius * std::sin(a1)},
            color
        );
    }
}

float WarpHeight(float x, float z, float scale) {
    float r2 = x * x + z * z;
    float well = -2.8f / std::sqrt(r2 + 0.45f * 0.45f);
    float h = well * scale;
    return std::max(-5.2f, h);
}

void DrawWarpSheet(float scale) {
    for (int i = 0; i < kSheetGrid; ++i) {
        float z = -kSheetExtent + 2.0f * kSheetExtent * static_cast<float>(i) / static_cast<float>(kSheetGrid - 1);
        for (int j = 0; j < kSheetGrid - 1; ++j) {
            float x0 = -kSheetExtent + 2.0f * kSheetExtent * static_cast<float>(j) / static_cast<float>(kSheetGrid - 1);
            float x1 = -kSheetExtent + 2.0f * kSheetExtent * static_cast<float>(j + 1) / static_cast<float>(kSheetGrid - 1);
            Vector3 p0 = {x0, WarpHeight(x0, z, scale), z};
            Vector3 p1 = {x1, WarpHeight(x1, z, scale), z};
            float glow = 1.0f - std::min(1.0f, std::fabs(p0.y) / 5.2f);
            Color c = {
                static_cast<unsigned char>(40 + 55 * glow),
                static_cast<unsigned char>(85 + 85 * glow),
                static_cast<unsigned char>(155 + 85 * glow),
                static_cast<unsigned char>(85 + 95 * glow),
            };
            DrawLine3D(p0, p1, c);
        }
    }

    for (int j = 0; j < kSheetGrid; ++j) {
        float x = -kSheetExtent + 2.0f * kSheetExtent * static_cast<float>(j) / static_cast<float>(kSheetGrid - 1);
        for (int i = 0; i < kSheetGrid - 1; ++i) {
            float z0 = -kSheetExtent + 2.0f * kSheetExtent * static_cast<float>(i) / static_cast<float>(kSheetGrid - 1);
            float z1 = -kSheetExtent + 2.0f * kSheetExtent * static_cast<float>(i + 1) / static_cast<float>(kSheetGrid - 1);
            Vector3 p0 = {x, WarpHeight(x, z0, scale), z0};
            Vector3 p1 = {x, WarpHeight(x, z1, scale), z1};
            float glow = 1.0f - std::min(1.0f, std::fabs(p0.y) / 5.2f);
            Color c = {
                static_cast<unsigned char>(35 + 45 * glow),
                static_cast<unsigned char>(75 + 75 * glow),
                static_cast<unsigned char>(140 + 90 * glow),
                static_cast<unsigned char>(65 + 80 * glow),
            };
            DrawLine3D(p0, p1, c);
        }
    }
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Black Hole 3D Visualization - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.5f, 4.0f, 7.5f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    std::random_device rd;
    std::mt19937 rng(rd());

    int desiredParticles = 520;
    int swallowed = 0;
    float simTime = 0.0f;
    float speed = 1.0f;
    bool paused = false;
    float camYaw = 0.78f;
    float camPitch = 0.38f;
    float camDistance = 11.0f;
    float warpScale = 1.0f;
    bool showWarp = true;
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
    std::string bridgeStatus = "bridge: waiting for AstroPhysics/vision/live_controls.txt";

    std::vector<DustParticle> disk;
    ResetDisk(&disk, rng, desiredParticles);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R)) {
            simTime = 0.0f;
            swallowed = 0;
            warpScale = 1.0f;
            ResetDisk(&disk, rng, desiredParticles);
        }
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
            speed = std::min(4.0f, speed + 0.25f);
        }
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
            speed = std::max(0.25f, speed - 0.25f);
        }
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) {
            desiredParticles = std::min(1400, desiredParticles + 100);
            ResetDisk(&disk, rng, desiredParticles);
        }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) {
            desiredParticles = std::max(120, desiredParticles - 100);
            ResetDisk(&disk, rng, desiredParticles);
        }
        if (IsKeyPressed(KEY_PERIOD)) warpScale = std::min(1.8f, warpScale + 0.05f);
        if (IsKeyPressed(KEY_COMMA)) warpScale = std::max(0.45f, warpScale - 0.05f);
        if (IsKeyPressed(KEY_W)) showWarp = !showWarp;

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
                    camDistance = std::clamp(camDistance / zoomRatio, 3.0f, 26.0f);
                    prevLiveZoom = currentLiveZoom;

                    const float rotDeltaDeg = AngleDeltaDeg(live->rotationDeg, prevLiveRotDeg);
                    prevLiveRotDeg = live->rotationDeg;
                    camYaw += rotDeltaDeg * DEG2RAD;

                    const float pitchDeltaDeg = live->pitchDeg - prevLivePitchDeg;
                    prevLivePitchDeg = live->pitchDeg;
                    camPitch += pitchDeltaDeg * DEG2RAD;
                    camPitch = std::clamp(camPitch, -1.35f, 1.35f);

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
                bridgeStatus = "bridge: stale";
                hasPrevLive = false;
            }
        } else {
            bridgeStatus = "bridge: waiting for AstroPhysics/vision/live_controls.txt";
            hasPrevLive = false;
        }

        auto applyPinchBuffer = [&](int* pending, std::int64_t* deadline, int matterDir, float speedDir) {
            if (*pending >= 2) {
                const int doubleCount = *pending / 2;
                speed = std::clamp(speed + speedDir * kSpeedStep * static_cast<float>(doubleCount), 0.25f, 4.0f);
                *pending %= 2;
                *deadline = (*pending > 0) ? (nowMs + kPinchSequenceWindowMs) : 0;
            }

            if (*pending == 1 && *deadline > 0 && nowMs >= *deadline) {
                desiredParticles = std::clamp(desiredParticles + matterDir * kMatterStep, 120, 1400);
                ResetDisk(&disk, rng, desiredParticles);
                *pending = 0;
                *deadline = 0;
            }
        };

        applyPinchBuffer(&pendingRightPinches, &rightPendingDeadlineMs, +1, +1.0f);
        applyPinchBuffer(&pendingLeftPinches, &leftPendingDeadlineMs, -1, -1.0f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            const float frameDt = GetFrameTime() * speed;
            simTime += frameDt;

            for (DustParticle& d : disk) {
                const Vector3 r = d.pos;
                const float r2 = Vector3DotProduct(r, r) + 0.04f;
                const float invR = 1.0f / std::sqrt(r2);
                const float invR3 = invR * invR * invR;

                Vector3 gravity = Vector3Scale(r, -kGravityMu * invR3);
                Vector3 drag = Vector3Scale(d.vel, -0.025f);
                d.vel = Vector3Add(d.vel, Vector3Scale(Vector3Add(gravity, drag), frameDt));
                d.pos = Vector3Add(d.pos, Vector3Scale(d.vel, frameDt));

                const float radius = std::sqrt(Vector3DotProduct(d.pos, d.pos));
                d.heat = std::clamp(1.25f - (radius - kDiskInnerRadius) / (kDiskOuterRadius - kDiskInnerRadius), 0.2f, 1.0f);
            }

            const int before = static_cast<int>(disk.size());
            disk.erase(
                std::remove_if(disk.begin(), disk.end(), [](const DustParticle& d) {
                    const float r = std::sqrt(Vector3DotProduct(d.pos, d.pos));
                    return r < kEventHorizonRadius * 1.02f || r > 11.0f;
                }),
                disk.end()
            );
            swallowed += before - static_cast<int>(disk.size());

            while (static_cast<int>(disk.size()) < desiredParticles) {
                disk.push_back(SpawnDust(rng));
            }
        }

        BeginDrawing();
        ClearBackground(Color{4, 6, 14, 255});

        BeginMode3D(camera);

        if (showWarp) DrawWarpSheet(warpScale);
        DrawCircle3DXZ(kDiskInnerRadius, 96, Color{255, 140, 70, 70});
        DrawCircle3DXZ(kPhotonRingRadius, 120, Color{150, 210, 255, 100});
        DrawCircle3DXZ(kDiskOuterRadius, 120, Color{90, 130, 190, 45});

        float sheetCenter = WarpHeight(0.0f, 0.0f, warpScale);
        DrawLine3D({0.0f, sheetCenter, 0.0f}, {0.0f, 0.0f, 0.0f}, Color{170, 220, 255, 110});
        DrawSphere({0.0f, sheetCenter, 0.0f}, 0.11f, Color{130, 190, 255, 90});

        DrawSphere({0.0f, 0.0f, 0.0f}, kPhotonRingRadius, Color{80, 130, 200, 18});
        DrawSphere({0.0f, 0.0f, 0.0f}, kEventHorizonRadius, BLACK);

        for (const DustParticle& d : disk) {
            DrawSphere(d.pos, d.size, DiskColor(d.heat));
        }

        EndMode3D();

        DrawText("Black Hole + Accretion Disk (3D)", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | P pause | +/- speed | [ ] density | , . warp | W warp | R reset", 20, 54, 20, Color{164, 183, 210, 255});
        const std::string hud = HudText(simTime, speed, static_cast<int>(disk.size()), swallowed, paused);
        DrawText(hud.c_str(), 20, 84, 21, Color{126, 224, 255, 255});
        std::ostringstream warpHud;
        warpHud << std::fixed << std::setprecision(2)
                << "warp=" << warpScale
                << "  warpVisible=" << (showWarp ? "yes" : "no");
        DrawText(warpHud.str().c_str(), 20, 110, 20, Color{149, 201, 255, 255});
        DrawText(bridgeStatus.c_str(), 20, 136, 19, Color{152, 234, 198, 255});
        DrawFPS(20, 162);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
