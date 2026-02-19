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
constexpr float kWaveAmpMin = 0.25f;
constexpr float kWaveAmpMax = 1.80f;
constexpr std::int64_t kControlStaleMs = 1200;

struct SamplePoint {
    Vector3 pos;
    float size;
    Color color;
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
    float pinchRatio = 0.0f;
    int nIncCount = 0;
    int nDecCount = 0;
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

void UpdateCameraFromOrbit(Camera3D* camera, float yaw, float pitch, float distance) {
    float cp = std::cos(pitch);
    Vector3 offset = {
        distance * cp * std::cos(yaw),
        distance * std::sin(pitch),
        distance * cp * std::sin(yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 26.0f);
    UpdateCameraFromOrbit(camera, *yaw, *pitch, *distance);
}

std::optional<LiveControls> ParseLiveControlsFile(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) return std::nullopt;

    LiveControls lc;
    std::string line;
    while (std::getline(in, line)) {
        std::size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = Trim(line.substr(0, eq));
        std::string val = Trim(line.substr(eq + 1));

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
            else if (key == "pinch_ratio") lc.pinchRatio = std::stof(val);
            else if (key == "n_inc_count") lc.nIncCount = std::stoi(val);
            else if (key == "n_dec_count") lc.nDecCount = std::stoi(val);
            else if (key == "timestamp_ms") lc.timestampMs = std::stoll(val);
        } catch (...) {
            // Ignore malformed values; keep defaults.
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

float PsiN(float x, float L, int n) {
    if (x < 0.0f || x > L) return 0.0f;
    return std::sqrt(2.0f / L) * std::sin(static_cast<float>(n) * PI * x / L);
}

float Density(float x, float L, int n) {
    float psi = PsiN(x, L, n);
    return psi * psi;
}

std::vector<SamplePoint> BuildCloud(float L, int n, float phase, float waveAmp) {
    std::vector<SamplePoint> pts;
    pts.reserve(420);

    const int bins = 220;
    float maxD = 0.0f;
    for (int i = 0; i < bins; ++i) {
        float x = L * static_cast<float>(i) / static_cast<float>(bins - 1);
        maxD = std::max(maxD, Density(x, L, n));
    }

    for (int i = 0; i < bins; ++i) {
        float x = L * static_cast<float>(i) / static_cast<float>(bins - 1);
        float d = Density(x, L, n) / std::max(1e-6f, maxD);

        int count = static_cast<int>(1 + 3.2f * d);
        for (int k = 0; k < count; ++k) {
            float z = -0.75f + 1.5f * (static_cast<float>((i * 13 + k * 37) % 97) / 96.0f);
            float y = 0.08f + 0.8f * d * waveAmp + 0.06f * std::sin(phase + 0.23f * static_cast<float>(i + k));
            y = std::clamp(y, 0.02f, 1.55f);

            float hue = 0.5f + 0.5f * std::sin(phase + 8.0f * x / L);
            Color c = Color{
                static_cast<unsigned char>(80 + 70 * hue),
                static_cast<unsigned char>(140 + 90 * (1.0f - hue)),
                static_cast<unsigned char>(210 + 40 * hue),
                static_cast<unsigned char>(100 + 130 * d)
            };
            float size = (0.015f + 0.02f * d) * (0.75f + 0.45f * waveAmp);
            pts.push_back({{x - L * 0.5f, y, z}, size, c});
        }
    }

    return pts;
}

std::string Hud(int n, float L, float waveAmp, bool paused) {
    std::ostringstream os;
    os << "n=" << n << "  L=" << std::fixed << std::setprecision(2) << L
       << "  waveAmp=" << std::setprecision(2) << waveAmp;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Quantum Particle in a Box 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.4f, 4.3f, 7.8f};
    camera.target = {0.0f, 0.6f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.86f;
    float camPitch = 0.30f;
    float camDistance = 11.2f;

    float L = 6.0f;
    int n = 2;
    bool paused = false;
    float waveAmp = 1.0f;
    float t = 0.0f;
    bool hasPrevLive = false;
    float prevLiveZoom = 1.0f;
    float prevLiveRotDeg = 0.0f;
    float prevLivePitchDeg = 0.0f;
    int prevLiveNIncCount = 0;
    int prevLiveNDecCount = 0;
    std::string controlStatus = "bridge: waiting for AstroPhysics/vision/live_controls.txt";

    while (!WindowShouldClose()) {
        bool drawZoomOverlay = false;
        Vector2 zoomP0 = {0.0f, 0.0f};
        Vector2 zoomP1 = {0.0f, 0.0f};
        std::string zoomOverlayText;

        if (IsKeyPressed(KEY_F11) || IsKeyPressed(KEY_F)) ToggleFullscreen();
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            L = 6.0f;
            n = 2;
            t = 0.0f;
            paused = false;
        }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) n = std::max(1, n - 1);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) n = std::min(8, n + 1);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) L = std::max(3.0f, L - 0.2f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) L = std::min(10.0f, L + 0.2f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);
        controlStatus = "bridge: waiting for AstroPhysics/vision/live_controls.txt";

        if (auto live = LoadLiveControls()) {
            std::int64_t ageMs = UnixMsNow() - live->timestampMs;
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
                    camDistance = std::clamp(camDistance / zoomRatio, 4.0f, 26.0f);
                    prevLiveZoom = currentLiveZoom;

                    float rotDeltaDeg = AngleDeltaDeg(live->rotationDeg, prevLiveRotDeg);
                    prevLiveRotDeg = live->rotationDeg;
                    camYaw += rotDeltaDeg * DEG2RAD;

                    float pitchDeltaDeg = live->pitchDeg - prevLivePitchDeg;
                    prevLivePitchDeg = live->pitchDeg;
                    camPitch += pitchDeltaDeg * DEG2RAD;
                    camPitch = std::clamp(camPitch, -1.35f, 1.35f);

                    int incDelta = 0;
                    int decDelta = 0;
                    if (live->nIncCount >= prevLiveNIncCount) {
                        incDelta = live->nIncCount - prevLiveNIncCount;
                    }
                    if (live->nDecCount >= prevLiveNDecCount) {
                        decDelta = live->nDecCount - prevLiveNDecCount;
                    }
                    if (incDelta != 0 || decDelta != 0) {
                        n = std::clamp(n + incDelta - decDelta, 1, 8);
                    }
                    prevLiveNIncCount = live->nIncCount;
                    prevLiveNDecCount = live->nDecCount;
                }

                paused = live->paused;
                waveAmp = std::clamp(live->waveAmp, kWaveAmpMin, kWaveAmpMax);
                UpdateCameraFromOrbit(&camera, camYaw, camPitch, camDistance);
                if (live->zoomLineActive) {
                    const float sw = static_cast<float>(GetScreenWidth());
                    const float sh = static_cast<float>(GetScreenHeight());
                    zoomP0 = {
                        std::clamp(live->zoomLineAx, 0.0f, 1.0f) * sw,
                        std::clamp(live->zoomLineAy, 0.0f, 1.0f) * sh,
                    };
                    zoomP1 = {
                        std::clamp(live->zoomLineBx, 0.0f, 1.0f) * sw,
                        std::clamp(live->zoomLineBy, 0.0f, 1.0f) * sh,
                    };
                    std::ostringstream zs;
                    zs << "Zoom x" << std::fixed << std::setprecision(2) << live->zoom;
                    zoomOverlayText = zs.str();
                    drawZoomOverlay = true;
                }

                std::ostringstream cs;
                cs << "bridge: connected  hand=" << live->label
                   << "  gesture=" << live->gesture
                   << "  age=" << ageMs << "ms";
                controlStatus = cs.str();
            } else {
                hasPrevLive = false;
                controlStatus = "bridge: stale data (tracker not updating)";
            }
        } else {
            hasPrevLive = false;
        }

        if (!paused) {
            t += GetFrameTime();
        }

        float phase = 2.6f * t;
        std::vector<SamplePoint> cloud = BuildCloud(L, n, phase, waveAmp);

        BeginDrawing();
        ClearBackground(Color{6, 9, 17, 255});

        BeginMode3D(camera);

        Vector3 boxCenter = {0.0f, 0.5f, 0.0f};
        DrawCubeWires(boxCenter, L, 1.2f, 1.8f, Color{120, 170, 240, 140});

        for (int i = 0; i <= n; ++i) {
            float xNode = -L * 0.5f + L * static_cast<float>(i) / static_cast<float>(n);
            DrawLine3D({xNode, 0.0f, -0.9f}, {xNode, 1.1f, -0.9f}, Color{255, 120, 120, 110});
        }

        const int segments = 220;
        for (int i = 0; i < segments - 1; ++i) {
            float x0 = L * static_cast<float>(i) / static_cast<float>(segments - 1);
            float x1 = L * static_cast<float>(i + 1) / static_cast<float>(segments - 1);

            float d0 = Density(x0, L, n);
            float d1 = Density(x1, L, n);
            float y0 = 0.03f + 0.95f * waveAmp * d0 / (2.0f / L);
            float y1 = 0.03f + 0.95f * waveAmp * d1 / (2.0f / L);
            y0 = std::clamp(y0, 0.03f, 1.15f);
            y1 = std::clamp(y1, 0.03f, 1.15f);

            Vector3 p0 = {x0 - L * 0.5f, y0, -0.95f};
            Vector3 p1 = {x1 - L * 0.5f, y1, -0.95f};
            DrawLine3D(p0, p1, Color{255, 210, 120, 210});
        }

        for (const SamplePoint& sp : cloud) {
            DrawSphere(sp.pos, sp.size, sp.color);
        }

        EndMode3D();

        if (drawZoomOverlay) {
            DrawLineEx(zoomP0, zoomP1, 3.0f, Color{0, 230, 255, 255});
            int mx = static_cast<int>((zoomP0.x + zoomP1.x) * 0.5f);
            int my = static_cast<int>((zoomP0.y + zoomP1.y) * 0.5f);
            DrawText(zoomOverlayText.c_str(), mx + 1, my - 8 + 1, 22, Color{0, 0, 0, 255});
            DrawText(zoomOverlayText.c_str(), mx, my - 8, 22, Color{235, 248, 255, 255});
        }

        DrawText("Quantum Particle in a 1D Box (3D view)", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Mouse+gestures active | 1 fist=cam lock | 2 fists=pause | dual pinch=zoom+wave | single pinch Right=] Left=[ | wrist=left/right yaw | hand Y=pitch | F/F11", 20, 54, 18, Color{164, 183, 210, 255});
        std::string hud = Hud(n, L, waveAmp, paused);
        DrawText(hud.c_str(), 20, 82, 20, Color{126, 224, 255, 255});
        DrawText(controlStatus.c_str(), 20, 110, 18, Color{255, 205, 140, 255});
        DrawFPS(20, 138);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
