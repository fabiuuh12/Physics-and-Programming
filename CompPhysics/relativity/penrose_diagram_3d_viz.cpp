#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1480;
constexpr int kScreenHeight = 920;
constexpr int kWindowMinWidth = 980;
constexpr int kWindowMinHeight = 620;
constexpr float kSceneScale = 2.2f;
constexpr float kHalfDepth = 1.25f;
constexpr float kStarFieldRadius = 95.0f;

struct Region {
    const char* name;
    const char* detail;
    std::vector<Vector2> polygon;
    Vector2 labelPos;
    Color color;
    float pulsePhase;
};

struct Star {
    Vector3 position;
    float radius;
    float twinkle;
    float phase;
    Color color;
};

struct CameraRig {
    float yaw = -0.85f;
    float pitch = 0.48f;
    float distance = 28.0f;
    Vector3 target = {0.0f, 0.0f, 0.0f};
};

float Clamp01(float v) {
    return std::clamp(v, 0.0f, 1.0f);
}

float Mix(float a, float b, float t) {
    return a + (b - a) * Clamp01(t);
}

float SmoothStep(float a, float b, float x) {
    if (std::fabs(b - a) < 1e-5f) return x >= b ? 1.0f : 0.0f;
    float t = Clamp01((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
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

Color WithAlpha(Color c, unsigned char alpha) {
    c.a = alpha;
    return c;
}

Vector3 ToWorld(Vector2 p, float z = 0.0f) {
    return {p.x * kSceneScale, p.y * kSceneScale, z};
}

std::vector<Region> BuildRegions() {
    return {
        {
            "Universe",
            "Region I",
            {{0.0f, 0.0f}, {3.2f, 3.05f}, {6.2f, 0.0f}, {3.2f, -3.05f}},
            {3.45f, 0.0f},
            Color{82, 189, 255, 255},
            0.0f
        },
        {
            "Black Hole",
            "Region II",
            {{0.0f, 0.0f}, {3.2f, 3.05f}, {0.0f, 6.15f}, {-3.2f, 3.05f}},
            {0.0f, 3.55f},
            Color{255, 112, 92, 255},
            1.1f
        },
        {
            "White Hole",
            "Region IV",
            {{-3.2f, -3.05f}, {0.0f, -6.15f}, {3.2f, -3.05f}, {0.0f, 0.0f}},
            {0.0f, -3.65f},
            Color{255, 233, 143, 255},
            2.2f
        },
        {
            "Parallel Universe",
            "Region III",
            {{-6.2f, 0.0f}, {-3.2f, 3.05f}, {0.0f, 0.0f}, {-3.2f, -3.05f}},
            {-3.75f, 0.0f},
            Color{154, 124, 255, 255},
            3.0f
        },
    };
}

std::vector<Star> BuildStars() {
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::uniform_real_distribution<float> azimuth(0.0f, 2.0f * PI);
    std::uniform_real_distribution<float> elevation(-0.4f * PI, 0.4f * PI);

    std::vector<Star> stars;
    stars.reserve(220);
    for (int i = 0; i < 220; ++i) {
        const float theta = azimuth(rng);
        const float phi = elevation(rng);
        const float radius = Mix(kStarFieldRadius * 0.85f, kStarFieldRadius, unit(rng));
        const float cosPhi = std::cos(phi);
        const Vector3 p = {
            radius * cosPhi * std::cos(theta),
            radius * std::sin(phi),
            radius * cosPhi * std::sin(theta)
        };
        const float tint = unit(rng);
        stars.push_back({
            p,
            Mix(0.05f, 0.22f, unit(rng)),
            Mix(0.6f, 1.6f, unit(rng)),
            Mix(0.0f, 2.0f * PI, unit(rng)),
            LerpColor(Color{160, 205, 255, 255}, Color{255, 221, 170, 255}, tint)
        });
    }
    return stars;
}

void UpdateCameraRig(Camera3D* camera, CameraRig* rig, bool autoOrbit, float dt) {
    if (autoOrbit) rig->yaw += dt * 0.22f;

    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        rig->yaw -= delta.x * 0.0035f;
        rig->pitch += delta.y * 0.0035f;
    }

    rig->pitch = std::clamp(rig->pitch, -1.35f, 1.2f);
    rig->distance -= GetMouseWheelMove() * 1.5f;
    rig->distance = std::clamp(rig->distance, 14.0f, 44.0f);

    const float cp = std::cos(rig->pitch);
    const Vector3 offset = {
        rig->distance * cp * std::cos(rig->yaw),
        rig->distance * std::sin(rig->pitch),
        rig->distance * cp * std::sin(rig->yaw),
    };
    camera->position = Vector3Add(rig->target, offset);
    camera->target = rig->target;
}

void DrawExtrudedPolygon(const std::vector<Vector2>& polygon, float halfDepth, Color faceColor, Color edgeColor) {
    if (polygon.size() < 3) return;

    for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
        DrawTriangle3D(ToWorld(polygon[0], halfDepth), ToWorld(polygon[i], halfDepth), ToWorld(polygon[i + 1], halfDepth), faceColor);
        DrawTriangle3D(ToWorld(polygon[0], -halfDepth), ToWorld(polygon[i + 1], -halfDepth), ToWorld(polygon[i], -halfDepth), WithAlpha(faceColor, static_cast<unsigned char>(faceColor.a * 0.75f)));
    }

    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const std::size_t j = (i + 1) % polygon.size();
        const Vector3 a0 = ToWorld(polygon[i], -halfDepth);
        const Vector3 a1 = ToWorld(polygon[j], -halfDepth);
        const Vector3 b0 = ToWorld(polygon[i], halfDepth);
        const Vector3 b1 = ToWorld(polygon[j], halfDepth);
        const Color side = WithAlpha(edgeColor, static_cast<unsigned char>(std::min(255, edgeColor.a + 20)));
        DrawTriangle3D(a0, a1, b0, side);
        DrawTriangle3D(b0, a1, b1, side);
        DrawLine3D(a0, a1, edgeColor);
        DrawLine3D(b0, b1, edgeColor);
        DrawLine3D(a0, b0, WithAlpha(edgeColor, 180));
    }
}

void DrawRibbonSegment(Vector2 a, Vector2 b, float halfDepth, float radius, Color color) {
    DrawCylinderEx(ToWorld(a, -halfDepth), ToWorld(b, -halfDepth), radius, radius, 10, WithAlpha(color, 120));
    DrawCylinderEx(ToWorld(a, halfDepth), ToWorld(b, halfDepth), radius, radius, 10, WithAlpha(color, 120));
    DrawCylinderEx(ToWorld(a, 0.0f), ToWorld(b, 0.0f), radius * 1.3f, radius * 1.3f, 12, color);
}

void DrawRegionLattice(const Region& region, float sceneTime) {
    const int stripes = 7;
    const int layers = 5;
    for (int i = 1; i < stripes; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(stripes);
        const float pulse = 0.3f + 0.7f * (0.5f + 0.5f * std::sin(sceneTime * 1.8f + region.pulsePhase + t * 3.0f));
        for (int layer = 0; layer < layers; ++layer) {
            const float z = Mix(-kHalfDepth * 0.85f, kHalfDepth * 0.85f, static_cast<float>(layer) / static_cast<float>(layers - 1));
            const Vector2 p0 = Vector2Lerp(region.polygon[0], region.polygon[2], t);
            const Vector2 p1 = Vector2Lerp(region.polygon[1], region.polygon[3], t);
            DrawLine3D(ToWorld(p0, z), ToWorld(p1, z), WithAlpha(region.color, static_cast<unsigned char>(65 + 90 * pulse)));
        }
    }
}

void DrawStars(const std::vector<Star>& stars, float sceneTime) {
    for (const Star& star : stars) {
        const float glow = 0.55f + 0.45f * std::sin(sceneTime * star.twinkle + star.phase);
        DrawSphere(star.position, star.radius * glow, WithAlpha(star.color, static_cast<unsigned char>(120 + 120 * glow)));
    }
}

Vector3 SampleWorldlinePoint(int regionIndex, float s, int lane, float sceneTime) {
    const float laneShift = (static_cast<float>(lane) - 1.5f) * 0.42f;
    const float z = laneShift * 0.95f;
    const float wobble = 0.18f * std::sin(sceneTime * 1.25f + s * 9.0f + lane * 0.9f);

    switch (regionIndex) {
        case 0: {
            const float y = Mix(-2.55f, 2.55f, s);
            const float x = 3.3f + laneShift + 0.22f * std::sin(sceneTime * 1.1f + s * 7.0f + lane);
            return ToWorld({x + wobble * 0.3f, y}, z);
        }
        case 1: {
            const float bend = SmoothStep(0.0f, 1.0f, s);
            const float startX = laneShift * 3.1f;
            const float x = Mix(startX, 0.0f, std::pow(bend, 0.82f)) + wobble * (1.0f - s);
            const float y = Mix(0.35f, 4.85f, s);
            return ToWorld({x, y}, z);
        }
        case 2: {
            const float endX = laneShift * 3.0f;
            const float x = Mix(0.0f, endX, s) + wobble * s;
            const float y = Mix(-4.85f, -0.35f, s);
            return ToWorld({x, y}, z);
        }
        case 3: {
            const float y = Mix(-2.55f, 2.55f, s);
            const float x = -3.3f + laneShift - 0.22f * std::sin(sceneTime * 1.15f + s * 6.4f + lane);
            return ToWorld({x + wobble * 0.3f, y}, z);
        }
        default:
            return {0.0f, 0.0f, 0.0f};
    }
}

void DrawWorldline(int regionIndex, int lane, float sceneTime, Color color) {
    constexpr int kSegments = 34;
    std::array<Vector3, kSegments> points{};
    const float head = std::fmod(sceneTime * (0.13f + lane * 0.015f) + lane * 0.17f, 1.0f);

    for (int i = 0; i < kSegments; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(kSegments - 1);
        const float sample = std::fmod(head + u, 1.0f);
        points[i] = SampleWorldlinePoint(regionIndex, sample, lane, sceneTime);
    }

    for (int i = 0; i < kSegments - 1; ++i) {
        const float fade = 1.0f - static_cast<float>(i) / static_cast<float>(kSegments - 1);
        DrawLine3D(points[i], points[i + 1], WithAlpha(color, static_cast<unsigned char>(40 + 180 * fade)));
    }

    DrawSphere(points[0], 0.16f, color);
    DrawSphereWires(points[0], 0.22f, 10, 10, WithAlpha(color, 110));
}

void DrawSingularityBand(float y, Color color) {
    const std::vector<Vector2> band = {
        {-2.05f, y - 0.22f},
        {2.05f, y - 0.22f},
        {2.45f, y + 0.22f},
        {-2.45f, y + 0.22f},
    };
    DrawExtrudedPolygon(band, kHalfDepth * 0.78f, WithAlpha(color, 115), WithAlpha(color, 220));
}

void DrawFocusMarker(Vector2 center, float sceneTime, Color color) {
    const float pulse = 0.85f + 0.15f * std::sin(sceneTime * 3.0f);
    const Vector3 p = ToWorld(center, 0.0f);
    DrawSphereWires(p, 0.7f * pulse, 14, 14, WithAlpha(color, 180));
    DrawSphereWires(p, 1.1f * pulse, 14, 14, WithAlpha(color, 90));
}

std::string BuildFocusText(const Region& region) {
    std::ostringstream out;
    out << region.name << "  |  " << region.detail;
    return out.str();
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Penrose Diagram 3D Visualization - C++");
    SetWindowMinSize(kWindowMinWidth, kWindowMinHeight);
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {22.0f, 11.0f, 18.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    CameraRig rig;
    const std::vector<Region> regions = BuildRegions();
    const std::vector<Star> stars = BuildStars();

    bool paused = false;
    bool autoOrbit = true;
    bool showHud = true;
    int focusRegion = -1;
    float sceneTime = 0.0f;

    while (!WindowShouldClose()) {
        const float dt = GetFrameTime();
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_A)) autoOrbit = !autoOrbit;
        if (IsKeyPressed(KEY_H)) showHud = !showHud;
        if (IsKeyPressed(KEY_ZERO)) focusRegion = -1;
        if (IsKeyPressed(KEY_ONE)) focusRegion = 0;
        if (IsKeyPressed(KEY_TWO)) focusRegion = 1;
        if (IsKeyPressed(KEY_THREE)) focusRegion = 2;
        if (IsKeyPressed(KEY_FOUR)) focusRegion = 3;

        if (!paused) sceneTime += dt;

        const Vector3 desiredTarget = (focusRegion >= 0)
            ? ToWorld(regions[focusRegion].labelPos, 0.0f)
            : Vector3{0.0f, 0.0f, 0.0f};
        rig.target = Vector3Lerp(rig.target, desiredTarget, 0.06f);
        UpdateCameraRig(&camera, &rig, autoOrbit, dt);

        BeginDrawing();
        ClearBackground(Color{6, 8, 18, 255});

        BeginMode3D(camera);
        DrawStars(stars, sceneTime);

        const Color horizonColor = Color{228, 241, 255, 255};
        const Color infinityColor = Color{111, 128, 180, 255};
        const float axisDepth = kHalfDepth * 1.22f;

        DrawCylinderEx({0.0f, -15.0f, 0.0f}, {0.0f, 15.0f, 0.0f}, 0.03f, 0.03f, 8, Color{105, 130, 190, 90});
        DrawCylinderEx({-15.0f, 0.0f, 0.0f}, {15.0f, 0.0f, 0.0f}, 0.03f, 0.03f, 8, Color{105, 130, 190, 90});

        for (const Region& region : regions) {
            const float pulse = 0.45f + 0.55f * (0.5f + 0.5f * std::sin(sceneTime * 1.5f + region.pulsePhase));
            const Color face = WithAlpha(region.color, static_cast<unsigned char>(52 + 55 * pulse));
            const Color edge = WithAlpha(LerpColor(region.color, WHITE, 0.18f), static_cast<unsigned char>(140 + 70 * pulse));
            DrawExtrudedPolygon(region.polygon, kHalfDepth, face, edge);
            DrawRegionLattice(region, sceneTime);
        }

        DrawRibbonSegment({0.0f, 0.0f}, {3.2f, 3.05f}, axisDepth, 0.075f, horizonColor);
        DrawRibbonSegment({0.0f, 0.0f}, {-3.2f, 3.05f}, axisDepth, 0.075f, horizonColor);
        DrawRibbonSegment({0.0f, 0.0f}, {3.2f, -3.05f}, axisDepth, 0.075f, horizonColor);
        DrawRibbonSegment({0.0f, 0.0f}, {-3.2f, -3.05f}, axisDepth, 0.075f, horizonColor);

        DrawRibbonSegment({3.2f, 3.05f}, {6.2f, 0.0f}, kHalfDepth, 0.04f, infinityColor);
        DrawRibbonSegment({6.2f, 0.0f}, {3.2f, -3.05f}, kHalfDepth, 0.04f, infinityColor);
        DrawRibbonSegment({-3.2f, 3.05f}, {-6.2f, 0.0f}, kHalfDepth, 0.04f, infinityColor);
        DrawRibbonSegment({-6.2f, 0.0f}, {-3.2f, -3.05f}, kHalfDepth, 0.04f, infinityColor);

        DrawSingularityBand(4.7f, Color{255, 86, 120, 255});
        DrawSingularityBand(-4.7f, Color{255, 210, 110, 255});

        for (int lane = 0; lane < 4; ++lane) {
            DrawWorldline(0, lane, sceneTime, Color{110, 210, 255, 255});
            DrawWorldline(1, lane, sceneTime, Color{255, 130, 110, 255});
            DrawWorldline(2, lane, sceneTime, Color{255, 225, 135, 255});
            DrawWorldline(3, lane, sceneTime, Color{174, 144, 255, 255});
        }

        const float photonPhase = std::fmod(sceneTime * 0.38f, 1.0f);
        const std::array<Vector2, 4> horizonEnds = {{{3.2f, 3.05f}, {-3.2f, 3.05f}, {3.2f, -3.05f}, {-3.2f, -3.05f}}};
        for (std::size_t i = 0; i < horizonEnds.size(); ++i) {
            const float local = std::fmod(photonPhase + static_cast<float>(i) * 0.23f, 1.0f);
            const Vector2 p = Vector2Lerp({0.0f, 0.0f}, horizonEnds[i], local);
            DrawSphere(ToWorld(p, std::sin(sceneTime * 2.0f + static_cast<float>(i)) * 0.45f), 0.14f, WithAlpha(WHITE, 230));
        }

        DrawSphere({0.0f, 0.0f, 0.0f}, 0.18f, WHITE);
        DrawSphereWires({0.0f, 0.0f, 0.0f}, 0.42f, 12, 12, WithAlpha(WHITE, 120));

        if (focusRegion >= 0) DrawFocusMarker(regions[focusRegion].labelPos, sceneTime, regions[focusRegion].color);

        EndMode3D();

        DrawRectangleGradientV(0, 0, GetScreenWidth(), 130, Color{4, 6, 12, 220}, Color{4, 6, 12, 0});
        DrawText("Penrose Diagram in 3D", 26, 20, 30, RAYWHITE);
        DrawText("Animated interpretation of the eternal Schwarzschild extension", 28, 56, 20, Color{187, 201, 230, 255});

        int legendY = 102;
        for (std::size_t i = 0; i < regions.size(); ++i) {
            DrawRectangle(28 + static_cast<int>(i) * 245, legendY, 18, 18, regions[i].color);
            DrawText(regions[i].name, 54 + static_cast<int>(i) * 245, legendY - 1, 20, RAYWHITE);
            DrawText(regions[i].detail, 54 + static_cast<int>(i) * 245, legendY + 18, 16, Color{180, 188, 210, 255});
        }

        if (focusRegion >= 0) {
            DrawRectangleRounded(Rectangle{24.0f, static_cast<float>(GetScreenHeight() - 78), 430.0f, 42.0f}, 0.25f, 8, Color{10, 14, 26, 210});
            DrawText(BuildFocusText(regions[focusRegion]).c_str(), 40, GetScreenHeight() - 66, 24, regions[focusRegion].color);
        } else {
            DrawRectangleRounded(Rectangle{24.0f, static_cast<float>(GetScreenHeight() - 78), 540.0f, 42.0f}, 0.25f, 8, Color{10, 14, 26, 210});
            DrawText("Focus: whole causal structure", 40, GetScreenHeight() - 66, 24, RAYWHITE);
        }

        if (showHud) {
            DrawRectangleRounded(Rectangle{static_cast<float>(GetScreenWidth() - 370), 20.0f, 346.0f, 150.0f}, 0.2f, 8, Color{8, 12, 22, 210});
            DrawText("Controls", GetScreenWidth() - 340, 34, 24, RAYWHITE);
            DrawText("Mouse drag: orbit camera", GetScreenWidth() - 340, 66, 20, Color{194, 204, 228, 255});
            DrawText("Wheel: zoom", GetScreenWidth() - 340, 90, 20, Color{194, 204, 228, 255});
            DrawText("1-4: focus regions   0: reset", GetScreenWidth() - 340, 114, 20, Color{194, 204, 228, 255});
            DrawText("A: auto orbit   Space: pause   H: hide", GetScreenWidth() - 340, 138, 20, Color{194, 204, 228, 255});
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
