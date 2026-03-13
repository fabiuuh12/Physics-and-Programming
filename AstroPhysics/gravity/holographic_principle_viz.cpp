#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
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
constexpr std::int64_t kControlStaleMs = 1200;
constexpr int kBulkPointCount = 180;
constexpr int kBoundaryPointCount = 180;
constexpr float kBoundaryRadius = 5.2f;
constexpr float kCameraDistanceMin = 7.5f;
constexpr float kCameraDistanceMax = 26.0f;
constexpr float kProjectionBoost = 1.45f;

struct BulkPoint {
    Vector3 pos;
    float phase;
    float size;
};

struct LiveControls {
    float zoom = 1.0f;
    float rotationDeg = 0.0f;
    float pitchDeg = 0.0f;
    float waveAmp = 1.0f;
    bool paused = false;
    bool zoomLineActive = false;
    float zoomLineAx = 0.5f;
    float zoomLineAy = 0.5f;
    float zoomLineBx = 0.5f;
    float zoomLineBy = 0.5f;
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
    return duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

float Clamp01(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}

float NormalizeDeg(float deg) {
    while (deg > 180.0f) deg -= 360.0f;
    while (deg < -180.0f) deg += 360.0f;
    return deg;
}

Color ScaleColor(Color c, float scale) {
    return Color{
        static_cast<unsigned char>(std::clamp(c.r * scale, 0.0f, 255.0f)),
        static_cast<unsigned char>(std::clamp(c.g * scale, 0.0f, 255.0f)),
        static_cast<unsigned char>(std::clamp(c.b * scale, 0.0f, 255.0f)),
        c.a,
    };
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
            else if (key == "wave_amp") lc.waveAmp = std::stof(val);
            else if (key == "paused") lc.paused = (val == "1" || val == "true" || val == "True");
            else if (key == "zoom_line_active") lc.zoomLineActive = (val == "1" || val == "true" || val == "True");
            else if (key == "zoom_line_ax") lc.zoomLineAx = std::stof(val);
            else if (key == "zoom_line_ay") lc.zoomLineAy = std::stof(val);
            else if (key == "zoom_line_bx") lc.zoomLineBx = std::stof(val);
            else if (key == "zoom_line_by") lc.zoomLineBy = std::stof(val);
            else if (key == "label") lc.label = val;
            else if (key == "gesture") lc.gesture = val;
            else if (key == "timestamp_ms") lc.timestampMs = std::stoll(val);
        } catch (...) {
            // Ignore malformed values.
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
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

Vector3 RandomUnitVector(std::mt19937& rng) {
    float z = RandRange(rng, -1.0f, 1.0f);
    float a = RandRange(rng, 0.0f, 2.0f * PI);
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    return {r * std::cos(a), r * std::sin(a), z};
}

std::vector<BulkPoint> BuildBulkPoints(std::mt19937& rng) {
    std::vector<BulkPoint> points;
    points.reserve(kBulkPointCount);
    for (int i = 0; i < kBulkPointCount; ++i) {
        Vector3 dir = RandomUnitVector(rng);
        float radius = kBoundaryRadius * std::pow(RandRange(rng, 0.0f, 1.0f), 0.58f) * 0.92f;
        points.push_back({
            Vector3Scale(dir, radius),
            RandRange(rng, 0.0f, 2.0f * PI),
            RandRange(rng, 0.06f, 0.18f),
        });
    }
    return points;
}

std::vector<Vector3> BuildBoundaryPoints(std::mt19937& rng) {
    std::vector<Vector3> points;
    points.reserve(kBoundaryPointCount);
    for (int i = 0; i < kBoundaryPointCount; ++i) {
        points.push_back(Vector3Scale(RandomUnitVector(rng), kBoundaryRadius));
    }
    return points;
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

void UpdateCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.8f;
    *distance = std::clamp(*distance, kCameraDistanceMin, kCameraDistanceMax);
    UpdateCameraFromOrbit(camera, *yaw, *pitch, *distance);
}

void DrawBoundarySphere(float pulse, float projectionMix) {
    Color shell = ScaleColor(Color{90, 215, 255, 255}, 0.9f + 0.35f * pulse + 0.28f * projectionMix);
    Color shellDim = ScaleColor(Color{45, 125, 215, 255}, 0.7f + 0.2f * pulse);
    DrawSphereWires({0.0f, 0.0f, 0.0f}, kBoundaryRadius, 32, 32, shell);
    DrawSphereWires({0.0f, 0.0f, 0.0f}, kBoundaryRadius * 0.985f, 18, 18, shellDim);
}

void DrawInteriorGrid(float pulse) {
    constexpr int kGrid = 10;
    constexpr float kExtent = 4.4f;
    Color c = ScaleColor(Color{55, 110, 180, 150}, 0.75f + 0.3f * pulse);
    for (int i = 0; i < kGrid; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(kGrid - 1);
        float x = -kExtent + 2.0f * kExtent * t;
        DrawLine3D({x, -kExtent, -kExtent}, {x, -kExtent, kExtent}, c);
        DrawLine3D({-kExtent, -kExtent, x}, {kExtent, -kExtent, x}, c);
        DrawLine3D({x, kExtent, -kExtent}, {x, kExtent, kExtent}, c);
        DrawLine3D({-kExtent, kExtent, x}, {kExtent, kExtent, x}, c);
        DrawLine3D({x, -kExtent, -kExtent}, {x, kExtent, -kExtent}, c);
        DrawLine3D({-kExtent, x, -kExtent}, {kExtent, x, -kExtent}, c);
    }
}

std::string HudText(float projectionMix, float entropyScale, float camDistance, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "projection=" << projectionMix
       << "  entropy~area x" << entropyScale
       << "  cam=" << camDistance;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Holographic Principle 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {11.5f, 8.0f, 11.5f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    std::random_device rd;
    std::mt19937 rng(rd());
    const std::vector<BulkPoint> bulk = BuildBulkPoints(rng);
    const std::vector<Vector3> boundary = BuildBoundaryPoints(rng);

    float camYaw = 0.78f;
    float camPitch = 0.42f;
    float camDistance = 16.5f;
    float simTime = 0.0f;
    bool paused = false;
    bool showHelp = true;

    std::string bridgeStatus = "bridge: waiting for AstroPhysics/vision/live_controls.txt";

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_H)) showHelp = !showHelp;
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            simTime = 0.0f;
            paused = false;
            camYaw = 0.78f;
            camPitch = 0.42f;
            camDistance = 16.5f;
        }

        UpdateCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        const std::int64_t nowMs = UnixMsNow();
        LiveControls live{};
        bool bridgeActive = false;
        if (auto parsed = LoadLiveControls()) {
            if (nowMs - parsed->timestampMs <= kControlStaleMs) {
                live = *parsed;
                bridgeActive = true;
                paused = live.paused;
                camDistance = std::clamp(18.5f / std::max(0.25f, live.zoom), kCameraDistanceMin, kCameraDistanceMax);
                camYaw = live.rotationDeg * DEG2RAD;
                camPitch = std::clamp(live.pitchDeg * DEG2RAD * 0.85f, -1.20f, 1.20f);
                UpdateCameraFromOrbit(&camera, camYaw, camPitch, camDistance);
                std::ostringstream ss;
                ss << "bridge: " << live.label << " / " << live.gesture;
                bridgeStatus = ss.str();
            } else {
                bridgeStatus = "bridge: live_controls stale";
            }
        }

        if (!paused) simTime += GetFrameTime();

        const float wave = bridgeActive ? std::clamp(live.waveAmp, 0.25f, 1.8f) : 1.0f;
        const float pulse = 0.55f + 0.45f * std::sin(simTime * (1.2f + 0.3f * wave));
        const float projectionMix = bridgeActive && live.zoomLineActive ? std::clamp(0.45f + 0.35f * wave, 0.0f, 1.0f) : 0.10f;
        const float entropyScale = 1.0f + projectionMix * 0.85f;

        BeginDrawing();
        ClearBackground(Color{7, 9, 16, 255});

        for (int i = 0; i < 140; ++i) {
            float x = std::fmod(i * 97.0f + simTime * 14.0f, static_cast<float>(kScreenWidth));
            float y = std::fmod(i * 53.0f + simTime * 8.0f, static_cast<float>(kScreenHeight));
            DrawCircle(static_cast<int>(x), static_cast<int>(y), (i % 3 == 0) ? 2.0f : 1.0f, Color{18, 26, 48, 255});
        }

        BeginMode3D(camera);
        DrawInteriorGrid(pulse);
        DrawBoundarySphere(pulse, projectionMix);

        const int mappedCount = static_cast<int>(std::round(projectionMix * static_cast<float>(bulk.size())));
        for (size_t i = 0; i < bulk.size(); ++i) {
            const BulkPoint& bp = bulk[i];
            float wobble = 0.18f * std::sin(simTime * 1.4f + bp.phase);
            Vector3 innerPos = Vector3Add(bp.pos, Vector3Scale(Vector3Normalize(bp.pos), wobble));
            Vector3 boundaryPos = Vector3Scale(Vector3Normalize(bp.pos), kBoundaryRadius);

            if (static_cast<int>(i) < mappedCount) {
                Color lineColor = ScaleColor(Color{95, 225, 255, 255}, 0.45f + 0.45f * pulse);
                DrawLine3D(innerPos, boundaryPos, lineColor);
                DrawSphere(boundaryPos, bp.size * (0.85f + 0.55f * projectionMix), ScaleColor(Color{255, 220, 120, 255}, 0.9f + 0.15f * pulse));
            }

            DrawSphere(innerPos, bp.size, ScaleColor(Color{95, 155, 255, 255}, 0.75f + 0.15f * pulse));
        }

        for (size_t i = 0; i < boundary.size(); ++i) {
            float flare = 0.65f + 0.35f * std::sin(simTime * 1.1f + static_cast<float>(i) * 0.21f);
            DrawSphere(boundary[i], 0.07f + 0.05f * flare, ScaleColor(Color{120, 235, 255, 255}, 0.6f + 0.3f * projectionMix));
        }

        DrawSphere({0.0f, 0.0f, 0.0f}, 0.25f + 0.06f * pulse, ScaleColor(Color{255, 240, 180, 255}, 0.85f));
        EndMode3D();

        DrawRectangle(14, 14, 520, showHelp ? 132 : 88, Color{12, 16, 28, 220});
        DrawRectangleLines(14, 14, 520, showHelp ? 132 : 88, Color{90, 180, 255, 255});
        DrawText("Holographic Principle", 28, 28, 28, Color{235, 245, 255, 255});
        DrawText(HudText(projectionMix, entropyScale, camDistance, paused).c_str(), 28, 60, 20, Color{150, 220, 255, 255});
        DrawText(bridgeStatus.c_str(), 28, 86, 20, bridgeActive ? Color{120, 255, 180, 255} : Color{255, 190, 110, 255});
        if (showHelp) {
            DrawText("Run AstroPhysics/vision/holographic_principle_bridge.py to drive this scene.", 28, 112, 18, Color{220, 225, 235, 255});
            DrawText("The interior bulk points project onto the boundary shell when the zoom line is active.", 28, 134, 18, Color{220, 225, 235, 255});
        }

        if (bridgeActive && live.zoomLineActive) {
            int ax = static_cast<int>(live.zoomLineAx * static_cast<float>(kScreenWidth));
            int ay = static_cast<int>(live.zoomLineAy * static_cast<float>(kScreenHeight));
            int bx = static_cast<int>(live.zoomLineBx * static_cast<float>(kScreenWidth));
            int by = static_cast<int>(live.zoomLineBy * static_cast<float>(kScreenHeight));
            DrawLineEx({static_cast<float>(ax), static_cast<float>(ay)}, {static_cast<float>(bx), static_cast<float>(by)}, 4.0f, Color{115, 235, 255, 255});
            DrawText("bridge zoom line", std::min(ax, bx), std::max(24, std::min(ay, by) - 18), 18, Color{150, 240, 255, 255});
        }

        DrawText("keys: H help  P pause  R reset", 18, kScreenHeight - 28, 18, Color{170, 190, 215, 255});
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
