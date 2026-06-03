#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int kScreenWidth = 1520;
constexpr int kScreenHeight = 920;
constexpr float kSceneMinRadius = 1.6f;
constexpr float kSceneMaxRadius = 34.0f;
constexpr double kMinDistanceLy = 0.001;
constexpr double kMaxDistanceLy = 46.5e9;
constexpr double kMaxLookbackGyr = 13.8;
constexpr double kMaxRedshift = 1100.0;
constexpr int kStarCount = 520;
constexpr int kFilamentParticleCount = 320;

enum class MetricMode {
    kDistanceNow = 0,
    kLookbackTime = 1,
    kRedshift = 2,
};

struct OrbitCameraState {
    float yaw = 0.74f;
    float pitch = 0.30f;
    float distance = 68.0f;
    Vector3 target = {0.0f, 0.0f, 0.0f};
};

struct Landmark {
    const char* name = "";
    const char* note = "";
    double distanceLy = 0.0;
    double lookbackGyr = 0.0;
    double redshift = 0.0;
    Color color{};
    float theta = 0.0f;
    float phi = 0.0f;
};

struct BackdropStar {
    Vector3 pos{};
    float size = 0.0f;
    float twinkle = 0.0f;
};

struct FilamentParticle {
    Vector3 pos{};
    float size = 0.0f;
    float pulse = 0.0f;
};

float Clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

float LerpFloat(float a, float b, float t) {
    return a + (b - a) * Clamp01(t);
}

float RandRange(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

Color LerpColor(Color a, Color b, float t) {
    const float u = Clamp01(t);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * u),
        static_cast<unsigned char>(a.g + (b.g - a.g) * u),
        static_cast<unsigned char>(a.b + (b.b - a.b) * u),
        static_cast<unsigned char>(a.a + (b.a - a.a) * u),
    };
}

double SafeLog10(double value) {
    return std::log10(std::max(value, 1.0e-9));
}

float MapDistanceLyToSceneRadius(double distanceLy) {
    const double lo = SafeLog10(kMinDistanceLy);
    const double hi = SafeLog10(kMaxDistanceLy);
    const double norm = (SafeLog10(std::clamp(distanceLy, kMinDistanceLy, kMaxDistanceLy)) - lo) / (hi - lo);
    return kSceneMinRadius + static_cast<float>(norm) * (kSceneMaxRadius - kSceneMinRadius);
}

float NormalizeMetric(const Landmark& landmark, MetricMode mode) {
    if (mode == MetricMode::kDistanceNow) {
        const double lo = SafeLog10(kMinDistanceLy);
        const double hi = SafeLog10(kMaxDistanceLy);
        return static_cast<float>((SafeLog10(landmark.distanceLy) - lo) / (hi - lo));
    }
    if (mode == MetricMode::kLookbackTime) {
        return static_cast<float>(landmark.lookbackGyr / kMaxLookbackGyr);
    }
    return static_cast<float>(SafeLog10(1.0 + landmark.redshift) / SafeLog10(1.0 + kMaxRedshift));
}

const char* MetricName(MetricMode mode) {
    if (mode == MetricMode::kDistanceNow) return "Distance Now";
    if (mode == MetricMode::kLookbackTime) return "Lookback Time";
    return "Redshift";
}

Vector3 SpherePoint(float radius, float theta, float phi) {
    return {
        radius * std::cos(phi) * std::cos(theta),
        radius * std::sin(phi),
        radius * std::cos(phi) * std::sin(theta),
    };
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, OrbitCameraState* orbit) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        orbit->yaw -= delta.x * 0.0037f;
        orbit->pitch += delta.y * 0.0037f;
        orbit->pitch = std::clamp(orbit->pitch, -1.22f, 1.22f);
    }

    orbit->distance -= GetMouseWheelMove() * 2.0f;
    orbit->distance = std::clamp(orbit->distance, 18.0f, 110.0f);

    const float cp = std::cos(orbit->pitch);
    camera->target = orbit->target;
    camera->position = Vector3Add(orbit->target, {
        orbit->distance * cp * std::cos(orbit->yaw),
        orbit->distance * std::sin(orbit->pitch),
        orbit->distance * cp * std::sin(orbit->yaw),
    });
}

std::vector<Landmark> BuildLandmarks() {
    return {
        {"Solar System Edge", "heliopause / outer planetary neighborhood", 0.0023, 0.0, 0.0, Color{120, 210, 255, 255}, 0.20f, 0.18f},
        {"Oort Cloud", "icy reservoir around the Sun", 1.6, 0.0000016, 0.0, Color{164, 224, 255, 255}, 1.05f, -0.08f},
        {"Alpha Centauri", "nearest stellar system", 4.37, 0.00000437, 0.0, Color{248, 244, 180, 255}, 1.78f, 0.20f},
        {"Orion Arm", "our Milky Way neighborhood", 3500.0, 0.0035, 0.0, Color{140, 200, 255, 255}, 2.62f, -0.12f},
        {"Milky Way Scale", "disk and halo on galactic scales", 100000.0, 0.10, 0.00002, Color{136, 160, 255, 255}, 3.35f, 0.10f},
        {"Andromeda", "nearest major galaxy", 2.54e6, 0.00254, 0.0006, Color{255, 172, 160, 255}, 4.12f, -0.18f},
        {"Virgo Cluster", "nearest large galaxy cluster", 5.4e7, 0.054, 0.004, Color{255, 196, 132, 255}, 4.82f, 0.15f},
        {"Laniakea", "our supercluster environment", 5.2e8, 0.52, 0.035, Color{255, 144, 164, 255}, 5.44f, -0.10f},
        {"Quasar Era", "peak growth of bright early galaxies", 1.05e10, 10.8, 2.1, Color{228, 126, 255, 255}, 0.86f, 0.32f},
        {"Cosmic Microwave Background", "last scattering surface", 4.65e10, 13.8, 1100.0, Color{255, 240, 144, 255}, 2.18f, 0.36f},
    };
}

std::vector<BackdropStar> BuildBackdropStars() {
    std::mt19937 rng(71231);
    std::vector<BackdropStar> stars;
    stars.reserve(kStarCount);
    for (int i = 0; i < kStarCount; ++i) {
        const float theta = RandRange(rng, 0.0f, 2.0f * PI);
        const float phi = RandRange(rng, -0.48f * PI, 0.48f * PI);
        const float radius = RandRange(rng, 78.0f, 112.0f);
        stars.push_back({
            SpherePoint(radius, theta, phi),
            RandRange(rng, 0.03f, 0.12f),
            RandRange(rng, 0.7f, 5.0f),
        });
    }
    return stars;
}

std::vector<FilamentParticle> BuildFilamentParticles() {
    std::mt19937 rng(94117);
    std::vector<Vector3> anchors = {
        {22.0f, 14.0f, -18.0f},
        {-26.0f, 8.0f, -12.0f},
        {-18.0f, -10.0f, 24.0f},
        {14.0f, -14.0f, 25.0f},
        {27.0f, 5.0f, 12.0f},
        {-30.0f, 15.0f, 8.0f},
        {5.0f, 18.0f, -29.0f},
    };

    std::vector<FilamentParticle> particles;
    particles.reserve(kFilamentParticleCount);
    for (int i = 0; i < kFilamentParticleCount; ++i) {
        const Vector3 a = anchors[i % anchors.size()];
        const Vector3 b = anchors[(i * 3 + 2) % anchors.size()];
        const float t = RandRange(rng, 0.0f, 1.0f);
        Vector3 pos = Vector3Lerp(a, b, t);
        pos.x += RandRange(rng, -1.6f, 1.6f);
        pos.y += RandRange(rng, -1.2f, 1.2f);
        pos.z += RandRange(rng, -1.6f, 1.6f);
        particles.push_back({pos, RandRange(rng, 0.04f, 0.14f), RandRange(rng, 0.0f, 2.0f * PI)});
    }
    return particles;
}

int FindNearestLandmarkIndex(const std::vector<Landmark>& landmarks, MetricMode mode, float slider) {
    int bestIndex = 0;
    float bestDistance = 1.0e9f;
    for (int i = 0; i < static_cast<int>(landmarks.size()); ++i) {
        const float delta = std::fabs(NormalizeMetric(landmarks[i], mode) - slider);
        if (delta < bestDistance) {
            bestDistance = delta;
            bestIndex = i;
        }
    }
    return bestIndex;
}

void FormatDistance(double ly, char* buffer, int size) {
    if (ly < 0.01) {
        std::snprintf(buffer, size, "%.0f AU", ly * 63241.1);
    } else if (ly < 1000.0) {
        std::snprintf(buffer, size, "%.2f ly", ly);
    } else if (ly < 1.0e6) {
        std::snprintf(buffer, size, "%.1f kly", ly / 1.0e3);
    } else if (ly < 1.0e9) {
        std::snprintf(buffer, size, "%.2f Mly", ly / 1.0e6);
    } else {
        std::snprintf(buffer, size, "%.2f Gly", ly / 1.0e9);
    }
}

void FormatLookback(double gyr, char* buffer, int size) {
    if (gyr < 0.001) {
        std::snprintf(buffer, size, "%.2f kyr", gyr * 1.0e6);
    } else if (gyr < 1.0) {
        std::snprintf(buffer, size, "%.1f Myr", gyr * 1000.0);
    } else {
        std::snprintf(buffer, size, "%.2f Gyr", gyr);
    }
}

void FormatRedshift(double redshift, char* buffer, int size) {
    if (redshift < 0.001) {
        std::snprintf(buffer, size, "z < 0.001");
    } else if (redshift < 1.0) {
        std::snprintf(buffer, size, "z = %.3f", redshift);
    } else {
        std::snprintf(buffer, size, "z = %.1f", redshift);
    }
}

void DrawMetricRuler(const std::vector<Landmark>& landmarks, MetricMode mode, int selectedIndex) {
    const int left = 74;
    const int top = kScreenHeight - 108;
    const int width = 620;
    const int height = 48;
    DrawRectangleRounded(Rectangle{static_cast<float>(left - 18), static_cast<float>(top - 22), static_cast<float>(width + 36), 88.0f}, 0.20f, 10, Fade(Color{18, 24, 36, 255}, 0.92f));
    DrawLine(left, top, left + width, top, Fade(SKYBLUE, 0.65f));

    for (int i = 0; i < static_cast<int>(landmarks.size()); ++i) {
        const float norm = NormalizeMetric(landmarks[i], mode);
        const int x = left + static_cast<int>(norm * width);
        const bool selected = i == selectedIndex;
        DrawLine(x, top - 8, x, top + 8, selected ? landmarks[i].color : Fade(landmarks[i].color, 0.50f));
        DrawCircle(x, top, selected ? 6.0f : 4.0f, selected ? landmarks[i].color : Fade(landmarks[i].color, 0.65f));
        if (selected || i == 0 || i == static_cast<int>(landmarks.size()) - 1 || i == 4 || i == 7) {
            DrawText(landmarks[i].name, x - 42, top + 16, 14, selected ? Color{235, 240, 248, 255} : Color{160, 176, 204, 255});
        }
    }

    DrawText(MetricName(mode), left - 2, top - 38, 20, Color{228, 236, 246, 255});
    DrawText("log-scaled scene radius, but metric focus can switch to time or redshift", left - 2, top + height, 16, Color{148, 164, 188, 255});
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Observable Universe Scale Explorer - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {56.0f, 28.0f, 56.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 43.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    OrbitCameraState orbit{};
    orbit.distance = 76.0f;

    const std::vector<Landmark> landmarks = BuildLandmarks();
    const std::vector<BackdropStar> stars = BuildBackdropStars();
    const std::vector<FilamentParticle> filamentParticles = BuildFilamentParticles();

    MetricMode mode = MetricMode::kDistanceNow;
    float slider = 0.78f;
    float focusRadius = MapDistanceLyToSceneRadius(landmarks.back().distanceLy);
    float titlePulse = 0.0f;
    bool autoplay = true;

    while (!WindowShouldClose()) {
        const float dt = GetFrameTime();
        titlePulse += dt;

        if (IsKeyPressed(KEY_ONE)) mode = MetricMode::kDistanceNow;
        if (IsKeyPressed(KEY_TWO)) mode = MetricMode::kLookbackTime;
        if (IsKeyPressed(KEY_THREE)) mode = MetricMode::kRedshift;
        if (IsKeyPressed(KEY_SPACE)) autoplay = !autoplay;
        if (IsKeyPressed(KEY_R)) {
            mode = MetricMode::kDistanceNow;
            slider = 0.78f;
            autoplay = true;
        }

        if (IsKeyDown(KEY_RIGHT)) {
            slider += 0.22f * dt;
            autoplay = false;
        }
        if (IsKeyDown(KEY_LEFT)) {
            slider -= 0.22f * dt;
            autoplay = false;
        }
        slider = std::fmod(slider + 1.0f, 1.0f);

        if (autoplay) {
            slider = std::fmod(slider + 0.045f * dt, 1.0f);
        }

        UpdateOrbitCameraDragOnly(&camera, &orbit);

        const int selectedIndex = FindNearestLandmarkIndex(landmarks, mode, slider);
        const Landmark& selected = landmarks[selectedIndex];
        focusRadius = LerpFloat(focusRadius, MapDistanceLyToSceneRadius(selected.distanceLy), 0.08f);

        BeginDrawing();
        ClearBackground(Color{4, 7, 15, 255});

        BeginMode3D(camera);

        for (const BackdropStar& star : stars) {
            const float pulse = 0.45f + 0.55f * std::sin(titlePulse * star.twinkle);
            DrawSphere(star.pos, star.size * (0.7f + 0.5f * pulse), Fade(Color{220, 232, 255, 255}, 0.22f + 0.30f * pulse));
        }

        for (int i = 0; i < 6; ++i) {
            const float radius = kSceneMinRadius + (kSceneMaxRadius - kSceneMinRadius) * static_cast<float>(i + 1) / 6.0f;
            DrawSphereWires({0.0f, 0.0f, 0.0f}, radius, 36, 24, Fade(Color{54, 94, 148, 255}, 0.13f));
        }

        for (int i = 0; i < static_cast<int>(filamentParticles.size()); ++i) {
            const FilamentParticle& particle = filamentParticles[i];
            const float pulse = 0.5f + 0.5f * std::sin(titlePulse * 0.55f + particle.pulse);
            const Color tint = LerpColor(Color{86, 112, 180, 255}, Color{168, 124, 224, 255}, pulse);
            DrawSphere(particle.pos, particle.size * (0.9f + 0.55f * pulse), Fade(tint, 0.08f + 0.12f * pulse));
            if (i % 24 != 0) {
                DrawLine3D(filamentParticles[i - (i > 0 ? 1 : 0)].pos, particle.pos, Fade(tint, 0.05f));
            }
        }

        DrawSphere({0.0f, 0.0f, 0.0f}, 0.32f, Color{92, 156, 255, 255});
        DrawSphereWires({0.0f, 0.0f, 0.0f}, 0.35f, 18, 10, Fade(SKYBLUE, 0.55f));

        for (int i = 0; i < static_cast<int>(landmarks.size()); ++i) {
            const Landmark& landmark = landmarks[i];
            const float radius = MapDistanceLyToSceneRadius(landmark.distanceLy);
            const bool selectedShell = i == selectedIndex;
            const float selectedGlow = selectedShell ? 0.60f + 0.40f * std::sin(titlePulse * 2.4f) : 0.0f;
            DrawSphereWires({0.0f, 0.0f, 0.0f}, radius, 40, 28, Fade(landmark.color, selectedShell ? 0.55f + 0.15f * selectedGlow : 0.20f));

            const Vector3 markerPos = SpherePoint(radius, landmark.theta, landmark.phi);
            const float markerSize = selectedShell ? 0.34f + 0.10f * selectedGlow : 0.20f;
            DrawSphere(markerPos, markerSize, selectedShell ? landmark.color : Fade(landmark.color, 0.85f));
            if (selectedShell) {
                DrawLine3D({0.0f, 0.0f, 0.0f}, markerPos, Fade(landmark.color, 0.30f));
            }
        }

        DrawSphereWires({0.0f, 0.0f, 0.0f}, focusRadius, 52, 32, Fade(selected.color, 0.78f));
        DrawSphereWires({0.0f, 0.0f, 0.0f}, focusRadius + 0.18f, 52, 32, Fade(selected.color, 0.18f));
        DrawSphereWires({0.0f, 0.0f, 0.0f}, focusRadius - 0.18f, 52, 32, Fade(selected.color, 0.18f));

        EndMode3D();

        DrawText("Observable Universe Scale Explorer", 28, 22, 34, Color{234, 239, 248, 255});
        DrawText("Mouse drag orbit | wheel zoom | 1 distance | 2 lookback | 3 redshift | Left/Right scrub | Space autoplay | R reset", 28, 60, 18, Color{154, 176, 205, 255});
        DrawText("This is a log-radius teaching model: outer shells represent present-day scale, not a literal galaxy map.", 28, 86, 18, Color{138, 156, 182, 255});

        DrawRectangleRounded(Rectangle{24.0f, 132.0f, 436.0f, 214.0f}, 0.12f, 10, Fade(Color{16, 22, 34, 255}, 0.92f));
        DrawText(selected.name, 46, 154, 28, selected.color);
        DrawText(selected.note, 46, 188, 20, Color{182, 196, 220, 255});

        char line[256];
        char distanceBuf[64];
        char lookbackBuf[64];
        char redshiftBuf[64];
        FormatDistance(selected.distanceLy, distanceBuf, sizeof(distanceBuf));
        FormatLookback(selected.lookbackGyr, lookbackBuf, sizeof(lookbackBuf));
        FormatRedshift(selected.redshift, redshiftBuf, sizeof(redshiftBuf));

        std::snprintf(line, sizeof(line), "Distance now: %s", distanceBuf);
        DrawText(line, 46, 228, 21, Color{226, 232, 242, 255});
        std::snprintf(line, sizeof(line), "Lookback time: %s", lookbackBuf);
        DrawText(line, 46, 256, 21, Color{226, 232, 242, 255});
        std::snprintf(line, sizeof(line), "Redshift: %s", redshiftBuf);
        DrawText(line, 46, 284, 21, Color{226, 232, 242, 255});

        const char* focusMetric = MetricName(mode);
        std::snprintf(line, sizeof(line), "Focus mode: %s", focusMetric);
        DrawText(line, 46, 316, 20, Color{148, 220, 255, 255});

        DrawRectangleRounded(Rectangle{1130.0f, 142.0f, 344.0f, 226.0f}, 0.12f, 10, Fade(Color{16, 22, 34, 255}, 0.92f));
        DrawText("Landmark Labels", 1152, 164, 24, Color{226, 232, 244, 255});
        DrawText("Screen labels stay attached to one marker on each shell.", 1152, 196, 18, Color{148, 164, 188, 255});
        DrawText("Inner scene = local neighborhood", 1152, 236, 18, Color{176, 188, 212, 255});
        DrawText("Outer scene = large-scale cosmic structure", 1152, 262, 18, Color{176, 188, 212, 255});
        DrawText("Final shell = cosmic microwave background", 1152, 288, 18, Color{176, 188, 212, 255});
        DrawText("Autoplay sweeps the selected metric and snaps", 1152, 328, 18, Color{176, 188, 212, 255});
        DrawText("the focus shell to the nearest landmark.", 1152, 352, 18, Color{176, 188, 212, 255});

        for (int i = 0; i < static_cast<int>(landmarks.size()); ++i) {
            const Landmark& landmark = landmarks[i];
            const Vector3 markerPos = SpherePoint(MapDistanceLyToSceneRadius(landmark.distanceLy), landmark.theta, landmark.phi);
            const Vector2 screen = GetWorldToScreen(markerPos, camera);
            if (screen.x < -80.0f || screen.x > static_cast<float>(kScreenWidth) + 80.0f ||
                screen.y < -60.0f || screen.y > static_cast<float>(kScreenHeight) + 60.0f) {
                continue;
            }

            const bool selectedLabel = i == selectedIndex;
            const int offsetX = selectedLabel ? 22 : 16;
            const int offsetY = selectedLabel ? -20 : -14;
            DrawLineEx(screen, {screen.x + static_cast<float>(offsetX), screen.y + static_cast<float>(offsetY)}, selectedLabel ? 2.4f : 1.5f, selectedLabel ? landmark.color : Fade(landmark.color, 0.55f));
            DrawText(landmark.name, static_cast<int>(screen.x) + offsetX + 6, static_cast<int>(screen.y) + offsetY - 10, selectedLabel ? 18 : 15, selectedLabel ? Color{238, 244, 252, 255} : Color{164, 178, 204, 255});
        }

        DrawMetricRuler(landmarks, mode, selectedIndex);
        DrawFPS(24, kScreenHeight - 36);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
