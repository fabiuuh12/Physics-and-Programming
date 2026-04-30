#include "cislunar_nav.hpp"

#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr int kStartScreenWidth = 1280;
constexpr int kStartScreenHeight = 800;
constexpr int kMinScreenWidth = 1100;
constexpr int kMinScreenHeight = 700;
constexpr double kMu = 0.0121505856;
constexpr float kScale = 18.0f;
constexpr int kTrailWindow = 520;

struct StressWindow {
    int start;
    int end;
};

struct MissionPreset {
    const char* name;
    const char* subtitle;
    cnav::State initial;
    int steps;
    double dt;
    bool weakGeometry;
    std::vector<StressWindow> missedWindows;
    std::vector<StressWindow> lightingWindows;
    float lightingSeverity;
    cnav::RiskWeights weights;
};

struct Sample {
    cnav::State truth;
    cnav::State estimate;
    float risk;
    float clippedRisk;
    float geometry;
    float sigmaR;
    float sigmaV;
    bool missed;
    bool lighting;
};

struct Star {
    Vector3 position;
    float radius;
    Color color;
};

bool InWindow(int step, const std::vector<StressWindow>& windows) {
    for (const StressWindow& window : windows) {
        if (step >= window.start && step <= window.end) {
            return true;
        }
    }
    return false;
}

Vector3 ToWorld(double x, double y, float lift = 0.0f) {
    return {
        static_cast<float>((x - 0.5) * kScale),
        lift,
        static_cast<float>(y * kScale),
    };
}

Vector3 ToWorld(const cnav::State& state, float lift = 0.0f) {
    return ToWorld(state[0], state[1], lift);
}

Color RiskColor(float clippedRisk) {
    const float t = std::clamp(clippedRisk / 8.0f, 0.0f, 1.0f);
    const unsigned char r = static_cast<unsigned char>(55.0f + 200.0f * t);
    const unsigned char g = static_cast<unsigned char>(222.0f - 174.0f * t);
    const unsigned char b = static_cast<unsigned char>(255.0f - 210.0f * t);
    return {r, g, b, 255};
}

float GeometryStrength(const cnav::State& state, bool weakGeometry) {
    const Vector2 craft = {static_cast<float>(state[0]), static_cast<float>(state[1])};
    const Vector2 earthObs = {static_cast<float>(-kMu), 0.0f};
    const Vector2 moonObs = {static_cast<float>(1.0 - kMu), 0.0f};
    if (weakGeometry) {
        const float distance = Vector2Distance(craft, earthObs);
        return std::clamp(0.035f + 0.14f / (1.0f + 5.0f * distance), 0.035f, 0.20f);
    }

    const Vector2 a = Vector2Normalize(Vector2Subtract(earthObs, craft));
    const Vector2 b = Vector2Normalize(Vector2Subtract(moonObs, craft));
    const float cross = std::fabs(a.x * b.y - a.y * b.x);
    return std::clamp(cross, 0.05f, 1.0f);
}

std::vector<Star> BuildStars() {
    std::vector<Star> stars;
    stars.reserve(260);
    unsigned int seed = 2166136261u;
    auto next = [&seed]() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return static_cast<float>(seed % 10000u) / 10000.0f;
    };

    for (int i = 0; i < 260; ++i) {
        const float theta = next() * 6.2831853f;
        const float z = -1.0f + 2.0f * next();
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float radius = 32.0f + 35.0f * next();
        const unsigned char alpha = static_cast<unsigned char>(95 + static_cast<int>(120.0f * next()));
        stars.push_back({
            {radius * r * std::cos(theta), radius * z, radius * r * std::sin(theta)},
            0.018f + 0.030f * next(),
            {205, 225, 255, alpha},
        });
    }
    return stars;
}

cnav::RiskWeights MakeWeights(float wg, float wm, float wl) {
    cnav::RiskWeights weights;
    weights.w_r = 0.25;
    weights.w_v = 0.15;
    weights.w_g = wg;
    weights.w_m = wm;
    weights.w_l = wl;
    weights.r0 = 0.01;
    weights.v0 = 0.01;
    weights.epsilon = 0.05;
    weights.risk_max = 8.0;
    return weights;
}

std::vector<MissionPreset> MakePresets() {
    return {
        {
            "Lunar Proximity Ops",
            "CR3BP test arc tuned to remain near the Moon while risk stressors cycle",
            {0.86, 0.05, -0.025, -0.025},
            1800,
            0.004,
            false,
            {{510, 660}, {1220, 1380}},
            {{820, 1080}},
            1.0f,
            MakeWeights(0.30f, 0.22f, 0.12f),
        },
        {
            "Weak Geometry Stress",
            "single-observer cislunar tracking with deliberately poor geometry",
            {0.86, 0.07, 0.10, -0.225},
            1600,
            0.004,
            true,
            {{380, 540}, {980, 1160}},
            {{650, 890}},
            1.0f,
            MakeWeights(0.42f, 0.22f, 0.12f),
        },
        {
            "High-Uncertainty Recovery",
            "same dynamics with larger estimation uncertainty and recovery windows",
            {0.86, 0.05, 0.025, -0.075},
            1500,
            0.0045,
            false,
            {{260, 430}, {760, 940}},
            {{480, 780}},
            0.85f,
            MakeWeights(0.34f, 0.26f, 0.16f),
        },
    };
}

std::vector<Sample> BuildSamples(const MissionPreset& preset) {
    std::vector<Sample> samples;
    samples.reserve(static_cast<size_t>(preset.steps + 1));

    cnav::State truth = preset.initial;
    cnav::State estimate = {
        preset.initial[0] + 0.004,
        preset.initial[1] - 0.003,
        preset.initial[2] + 0.001,
        preset.initial[3] - 0.001,
    };
    double sigmaR = preset.name[0] == 'H' ? 0.0060 : 0.0025;
    double sigmaV = preset.name[0] == 'H' ? 0.0022 : 0.0008;

    for (int step = 0; step <= preset.steps; ++step) {
        const bool missed = InWindow(step, preset.missedWindows);
        const bool lighting = InWindow(step, preset.lightingWindows);
        const float geometry = GeometryStrength(truth, preset.weakGeometry);
        sigmaR += missed ? 0.000050 : 0.000006;
        sigmaV += missed ? 0.000018 : 0.000003;
        if (!missed) {
            sigmaR *= 0.992;
            sigmaV *= 0.994;
        }
        if (lighting) {
            sigmaR += 0.000015 * preset.lightingSeverity;
        }

        const double rawRisk = cnav::navigation_risk(
            sigmaR,
            sigmaV,
            geometry,
            missed ? 1.0 : 0.0,
            lighting ? preset.lightingSeverity : 0.0,
            preset.weights
        );
        const float clipped = static_cast<float>(cnav::clipped_risk(rawRisk, preset.weights));
        const float errorScale = 0.015f + 0.012f * clipped;
        estimate = {
            truth[0] + errorScale * std::sin(0.037 * step),
            truth[1] + errorScale * std::cos(0.031 * step),
            truth[2] + 0.002 * std::sin(0.021 * step),
            truth[3] + 0.002 * std::cos(0.019 * step),
        };

        samples.push_back({
            truth,
            estimate,
            static_cast<float>(rawRisk),
            clipped,
            geometry,
            static_cast<float>(sigmaR),
            static_cast<float>(sigmaV),
            missed,
            lighting,
        });
        truth = cnav::rk4_step(truth, preset.dt, kMu);
    }
    return samples;
}

void DrawStars(const std::vector<Star>& stars) {
    for (const Star& star : stars) {
        DrawSphere(star.position, star.radius, star.color);
    }
}

void DrawPlaneGrid() {
    constexpr int lines = 14;
    constexpr float extent = 14.0f;
    for (int i = -lines; i <= lines; ++i) {
        const float p = extent * static_cast<float>(i) / static_cast<float>(lines);
        const Color major = (i == 0) ? Fade(WHITE, 0.34f) : Fade(Color{73, 92, 118, 255}, 0.20f);
        DrawLine3D({-extent, 0.0f, p}, {extent, 0.0f, p}, major);
        DrawLine3D({p, 0.0f, -extent}, {p, 0.0f, extent}, major);
    }
}

void DrawLabel3D(Camera3D camera, Vector3 world, const char* text, Color color) {
    const Vector2 screen = GetWorldToScreen(world, camera);
    DrawText(text, static_cast<int>(screen.x), static_cast<int>(screen.y), 16, color);
}

void DrawTrajectoryRibbon(const std::vector<Sample>& samples) {
    for (size_t i = 1; i < samples.size(); i += 2) {
        const Vector3 a = ToWorld(samples[i - 1].truth, 0.06f);
        const Vector3 b = ToWorld(samples[i].truth, 0.06f);
        DrawLine3D(a, b, Fade(Color{80, 160, 255, 255}, 0.22f));
    }
}

void DrawRiskTrail(const std::vector<Sample>& samples, int current) {
    const int start = std::max(1, current - kTrailWindow);
    for (int i = start; i <= current; ++i) {
        const Sample& prev = samples[static_cast<size_t>(i - 1)];
        const Sample& now = samples[static_cast<size_t>(i)];
        const float prevLift = 0.18f + 0.13f * prev.clippedRisk;
        const float nowLift = 0.18f + 0.13f * now.clippedRisk;
        DrawLine3D(ToWorld(prev.truth, prevLift), ToWorld(now.truth, nowLift), RiskColor(now.clippedRisk));
    }
}

void DrawStressMarkers(const std::vector<Sample>& samples, int current) {
    const int start = std::max(1, current - kTrailWindow);
    for (int i = start; i <= current; i += 3) {
        const Sample& now = samples[static_cast<size_t>(i)];
        if (!now.missed && !now.lighting) {
            continue;
        }
        const Color color = now.missed ? Color{255, 70, 72, 135} : Color{255, 217, 76, 120};
        DrawCylinder(ToWorld(now.truth, 0.03f), 0.060f, 0.060f, 0.72f, 12, color);
    }
}

void DrawContextObjects(Camera3D camera) {
    const Vector3 earth = ToWorld(-kMu, 0.0);
    const Vector3 moon = ToWorld(1.0 - kMu, 0.0);
    const Vector3 l1 = ToWorld(0.8369, 0.0, 0.05f);
    const Vector3 l2 = ToWorld(1.1557, 0.0, 0.05f);

    DrawSphere(earth, 0.62f, Color{51, 128, 255, 255});
    DrawSphereWires(earth, 0.75f, 24, 12, Fade(SKYBLUE, 0.35f));
    DrawSphere(moon, 0.24f, Color{202, 207, 216, 255});
    DrawSphereWires(moon, 1.55f, 42, 10, Fade(LIGHTGRAY, 0.18f));
    DrawSphere(l1, 0.08f, Color{125, 255, 190, 255});
    DrawSphere(l2, 0.08f, Color{125, 255, 190, 255});
    DrawLine3D(ToWorld(-0.10, 0.0), ToWorld(1.18, 0.0), Fade(WHITE, 0.42f));
    DrawLabel3D(camera, Vector3Add(earth, {0.35f, 0.55f, 0.0f}), "Earth observer", SKYBLUE);
    DrawLabel3D(camera, Vector3Add(moon, {0.30f, 0.38f, 0.0f}), "Moon", LIGHTGRAY);
    DrawLabel3D(camera, Vector3Add(l1, {0.0f, 0.32f, 0.0f}), "L1", Color{125, 255, 190, 255});
    DrawLabel3D(camera, Vector3Add(l2, {0.0f, 0.32f, 0.0f}), "L2", Color{125, 255, 190, 255});
}

void DrawSpacecraftLayer(const MissionPreset& preset, const std::vector<Sample>& samples, int current) {
    const Sample& sample = samples[static_cast<size_t>(current)];
    const Vector3 earth = ToWorld(-kMu, 0.0);
    const Vector3 moon = ToWorld(1.0 - kMu, 0.0);
    const Vector3 truth = ToWorld(sample.truth, 0.20f + 0.13f * sample.clippedRisk);
    const Vector3 estimate = ToWorld(sample.estimate, 0.20f + 0.13f * sample.clippedRisk);

    DrawLine3D(earth, truth, Fade(SKYBLUE, preset.weakGeometry ? 0.70f : 0.40f));
    if (!preset.weakGeometry) {
        DrawLine3D(moon, truth, Fade(LIGHTGRAY, 0.34f));
    }
    DrawLine3D(truth, estimate, Fade(ORANGE, 0.75f));
    DrawSphere(estimate, 0.085f, Color{255, 183, 77, 255});
    DrawSphere(truth, 0.145f, WHITE);
    DrawSphereWires(truth, 0.26f + 0.050f * sample.clippedRisk, 18, 10, RiskColor(sample.clippedRisk));
    DrawCubeWires(truth, 0.42f + 0.04f * sample.clippedRisk, 0.42f + 0.04f * sample.clippedRisk, 0.42f + 0.04f * sample.clippedRisk, Fade(RiskColor(sample.clippedRisk), 0.42f));
}

void DrawTimeline(const MissionPreset& preset, int current) {
    const int screenW = GetScreenWidth();
    const int screenH = GetScreenHeight();
    const int w = std::clamp(screenW - 620, 420, 860);
    const int x = std::max(22, (screenW - w) / 2);
    const int y = screenH - 58;
    const int h = 16;
    DrawRectangle(x, y, w, h, Fade(Color{26, 36, 54, 255}, 0.95f));
    auto drawWindows = [&](const std::vector<StressWindow>& windows, Color color) {
        for (const StressWindow& window : windows) {
            const int x0 = x + static_cast<int>(w * static_cast<float>(window.start) / preset.steps);
            const int x1 = x + static_cast<int>(w * static_cast<float>(window.end) / preset.steps);
            DrawRectangle(x0, y, std::max(2, x1 - x0), h, color);
        }
    };
    drawWindows(preset.missedWindows, Color{255, 70, 72, 210});
    drawWindows(preset.lightingWindows, Color{255, 217, 76, 190});
    const int marker = x + static_cast<int>(w * static_cast<float>(current) / preset.steps);
    DrawRectangle(marker - 2, y - 5, 4, h + 10, RAYWHITE);
    DrawText("mission timeline: red = missed measurements, yellow = lighting/visibility loss", x, y - 26, 16, LIGHTGRAY);
}

void DrawHud(const MissionPreset& preset, const Sample& sample, int current, bool paused, int presetIndex) {
    const int screenW = GetScreenWidth();
    const int rightX = std::max(screenW - 355, 720);
    DrawRectangle(22, 20, 522, 238, Fade(Color{4, 9, 18, 255}, 0.88f));
    DrawText("Risk-Aware Cislunar Navigation", 44, 40, 24, RAYWHITE);
    DrawText(preset.name, 44, 75, 21, Color{142, 210, 255, 255});
    DrawText(preset.subtitle, 44, 103, 15, LIGHTGRAY);
    DrawText(TextFormat("step %04d / %04d     preset %d/3", current, preset.steps, presetIndex + 1), 44, 132, 16, GRAY);
    DrawText(TextFormat("raw risk %.2f     clipped risk %.2f / 8.00", sample.risk, sample.clippedRisk), 44, 158, 20, RiskColor(sample.clippedRisk));
    DrawText(TextFormat("geometry %.3f     sigma_r %.5f     sigma_v %.5f", sample.geometry, sample.sigmaR, sample.sigmaV), 44, 187, 17, LIGHTGRAY);
    DrawText(TextFormat("measurements %s     visibility %s", sample.missed ? "OUTAGE" : "nominal", sample.lighting ? "DEGRADED" : "nominal"), 44, 214, 17, sample.missed ? RED : (sample.lighting ? GOLD : LIGHTGRAY));
    DrawText(paused ? "SPACE resume | F fullscreen | R reset | 1-3 presets | drag orbit | wheel zoom" : "SPACE pause | F fullscreen | R reset | 1-3 presets | drag orbit | wheel zoom", 44, 238, 15, GRAY);

    DrawRectangle(rightX, 28, 330, 194, Fade(Color{4, 9, 18, 255}, 0.84f));
    DrawText("Visual Encoding", rightX + 22, 48, 20, RAYWHITE);
    DrawText("white body: propagated truth", rightX + 22, 80, 16, WHITE);
    DrawText("orange body: estimated state", rightX + 22, 105, 16, ORANGE);
    DrawText("cyan -> red: low -> high risk", rightX + 22, 130, 16, Color{100, 220, 255, 255});
    DrawText("trail height: clipped risk", rightX + 22, 155, 16, LIGHTGRAY);
    DrawText("wire cube/sphere: uncertainty envelope", rightX + 22, 180, 16, LIGHTGRAY);
}

void UpdateManualOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
        const Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0065f;
        *pitch -= delta.y * 0.0065f;
        *pitch = std::clamp(*pitch, -1.15f, 1.20f);
    }

    const float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        *distance = std::clamp(*distance - wheel * 0.85f, 5.0f, 34.0f);
    }

    const float cp = std::cos(*pitch);
    camera->position = {
        camera->target.x + *distance * cp * std::sin(*yaw),
        camera->target.y + *distance * std::sin(*pitch),
        camera->target.z + *distance * cp * std::cos(*yaw),
    };
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(kStartScreenWidth, kStartScreenHeight, "Risk-Aware Cislunar Navigation 3D");
    SetWindowMinSize(kMinScreenWidth, kMinScreenHeight);
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {4.8f, 9.2f, 13.2f};
    camera.target = {0.0f, 0.45f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 44.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float cameraYaw = 0.35f;
    float cameraPitch = 0.60f;
    float cameraDistance = 16.8f;

    const std::vector<Star> stars = BuildStars();
    const std::vector<MissionPreset> presets = MakePresets();
    int presetIndex = 0;
    std::vector<Sample> samples = BuildSamples(presets[static_cast<size_t>(presetIndex)]);
    int current = 0;
    bool paused = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R)) {
            current = 0;
        }
        if (IsKeyPressed(KEY_F) || IsKeyPressed(KEY_F11)) {
            if (!IsWindowFullscreen()) {
                const int monitor = GetCurrentMonitor();
                SetWindowSize(GetMonitorWidth(monitor), GetMonitorHeight(monitor));
                ToggleFullscreen();
            } else {
                ToggleFullscreen();
                SetWindowSize(kStartScreenWidth, kStartScreenHeight);
            }
        }
        for (int key = KEY_ONE; key <= KEY_THREE; ++key) {
            if (IsKeyPressed(key)) {
                presetIndex = key - KEY_ONE;
                samples = BuildSamples(presets[static_cast<size_t>(presetIndex)]);
                current = 0;
            }
        }

        UpdateManualOrbitCamera(&camera, &cameraYaw, &cameraPitch, &cameraDistance);
        if (!paused) {
            current += 2;
            if (current >= static_cast<int>(samples.size())) {
                current = 0;
            }
        }

        const MissionPreset& preset = presets[static_cast<size_t>(presetIndex)];
        const Sample& sample = samples[static_cast<size_t>(current)];

        BeginDrawing();
        ClearBackground(Color{2, 5, 11, 255});
        BeginMode3D(camera);
        DrawStars(stars);
        DrawPlaneGrid();
        DrawContextObjects(camera);
        DrawTrajectoryRibbon(samples);
        DrawRiskTrail(samples, current);
        DrawStressMarkers(samples, current);
        DrawSpacecraftLayer(preset, samples, current);
        EndMode3D();

        DrawHud(preset, sample, current, paused, presetIndex);
        DrawTimeline(preset, current);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
