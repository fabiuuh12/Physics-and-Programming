#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kSunMu = 1600.0f;
constexpr float kDt = 0.0028f;
constexpr int kPreviewSteps = 5200;
constexpr int kTrailMax = 1400;

struct Planet {
    std::string name;
    float radius;
    float visualRadius;
    float period;
    float phase;
    Color color;
};

struct Craft {
    Vector2 pos;
    Vector2 vel;
};

Vector2 WorldToScreen(Vector2 p, Vector2 camera, float scale) {
    return {
        kScreenWidth * 0.5f + (p.x - camera.x) * scale,
        kScreenHeight * 0.5f + (p.y - camera.y) * scale,
    };
}

Vector2 ScreenToWorld(Vector2 p, Vector2 camera, float scale) {
    return {
        camera.x + (p.x - kScreenWidth * 0.5f) / scale,
        camera.y + (p.y - kScreenHeight * 0.5f) / scale,
    };
}

Vector2 Accel(Vector2 pos) {
    const float r2 = Vector2LengthSqr(pos) + 0.35f;
    const float invR = 1.0f / std::sqrt(r2);
    const float invR3 = invR * invR * invR;
    return Vector2Scale(pos, -kSunMu * invR3);
}

void StepCraft(Craft* craft, float dt) {
    const Vector2 a0 = Accel(craft->pos);
    craft->vel = Vector2Add(craft->vel, Vector2Scale(a0, dt * 0.5f));
    craft->pos = Vector2Add(craft->pos, Vector2Scale(craft->vel, dt));
    const Vector2 a1 = Accel(craft->pos);
    craft->vel = Vector2Add(craft->vel, Vector2Scale(a1, dt * 0.5f));
}

Vector2 PlanetPosition(const Planet& planet, float time) {
    const float angle = planet.phase + time * 2.0f * kPi / planet.period;
    return {std::cos(angle) * planet.radius, std::sin(angle) * planet.radius};
}

std::vector<Planet> MakePlanets() {
    return {
        {"Mercury", 28.0f, 3.2f, 7.5f, 0.7f, Color{170, 160, 145, 255}},
        {"Venus", 42.0f, 4.8f, 12.6f, 2.5f, Color{238, 195, 116, 255}},
        {"Earth", 58.0f, 5.2f, 18.0f, 4.2f, Color{86, 172, 255, 255}},
        {"Mars", 82.0f, 4.3f, 34.0f, 1.4f, Color{225, 96, 66, 255}},
        {"Jupiter", 145.0f, 9.0f, 210.0f, 5.1f, Color{224, 177, 128, 255}},
    };
}

Craft MakeEarthOrbitCraft() {
    const float r = 58.0f;
    const float speed = std::sqrt(kSunMu / r);
    return {{r, 0.0f}, {0.0f, speed}};
}

std::string Fixed(float value, int precision = 2) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << value;
    return os.str();
}

std::vector<Vector2> BuildPreview(Craft craft, Vector2 burn) {
    craft.vel = Vector2Add(craft.vel, burn);
    std::vector<Vector2> points;
    points.reserve(650);
    for (int i = 0; i < kPreviewSteps; ++i) {
        StepCraft(&craft, kDt);
        if (i % 8 == 0) points.push_back(craft.pos);
        if (Vector2Length(craft.pos) > 250.0f) break;
        if (Vector2Length(craft.pos) < 5.5f) break;
    }
    return points;
}

void DrawPolylineWorld(const std::vector<Vector2>& points, Vector2 camera, float scale, Color color) {
    if (points.size() < 2) return;
    for (size_t i = 1; i < points.size(); ++i) {
        const float fade = static_cast<float>(i) / static_cast<float>(points.size());
        Color c = color;
        c.a = static_cast<unsigned char>(55 + 170 * fade);
        DrawLineV(WorldToScreen(points[i - 1], camera, scale), WorldToScreen(points[i], camera, scale), c);
    }
}

void DrawTrail(const std::vector<Vector2>& trail, Vector2 camera, float scale) {
    if (trail.size() < 2) return;
    for (size_t i = 1; i < trail.size(); ++i) {
        const float fade = static_cast<float>(i) / static_cast<float>(trail.size());
        Color c{155, 210, 255, static_cast<unsigned char>(35 + 140 * fade)};
        DrawLineV(WorldToScreen(trail[i - 1], camera, scale), WorldToScreen(trail[i], camera, scale), c);
    }
}

void DrawGrid(Vector2 camera, float scale) {
    const float spacing = 20.0f;
    const Vector2 minWorld = ScreenToWorld({0.0f, 0.0f}, camera, scale);
    const Vector2 maxWorld = ScreenToWorld({static_cast<float>(kScreenWidth), static_cast<float>(kScreenHeight)}, camera, scale);
    const int minX = static_cast<int>(std::floor(minWorld.x / spacing)) - 1;
    const int maxX = static_cast<int>(std::ceil(maxWorld.x / spacing)) + 1;
    const int minY = static_cast<int>(std::floor(minWorld.y / spacing)) - 1;
    const int maxY = static_cast<int>(std::ceil(maxWorld.y / spacing)) + 1;

    for (int x = minX; x <= maxX; ++x) {
        const float wx = x * spacing;
        const Vector2 a = WorldToScreen({wx, minWorld.y - spacing}, camera, scale);
        const Vector2 b = WorldToScreen({wx, maxWorld.y + spacing}, camera, scale);
        DrawLineV(a, b, Color{25, 35, 50, 120});
    }
    for (int y = minY; y <= maxY; ++y) {
        const float wy = y * spacing;
        const Vector2 a = WorldToScreen({minWorld.x - spacing, wy}, camera, scale);
        const Vector2 b = WorldToScreen({maxWorld.x + spacing, wy}, camera, scale);
        DrawLineV(a, b, Color{25, 35, 50, 120});
    }
}

void DrawHud(float simTime, float timeScale, Vector2 burn, bool paused, bool followCraft, float zoom) {
    DrawRectangle(920, 48, 314, 306, Color{13, 21, 34, 236});
    DrawRectangleLines(920, 48, 314, 306, Color{82, 110, 146, 255});
    DrawText("SOLAR SYSTEM PLANNER", 944, 76, 22, RAYWHITE);
    DrawText(("time: " + Fixed(simTime, 1) + " y").c_str(), 944, 118, 18, Color{205, 224, 245, 255});
    DrawText(("warp: " + Fixed(timeScale, 1) + "x").c_str(), 944, 146, 18, Color{205, 224, 245, 255});
    DrawText(("burn dv: " + Fixed(Vector2Length(burn), 2)).c_str(), 944, 184, 18, Color{255, 232, 150, 255});
    DrawText("prograde/radial preview", 944, 212, 18, Color{135, 245, 170, 255});
    DrawText(paused ? "SPACE resume" : "SPACE pause", 944, 250, 17, Color{170, 184, 204, 255});
    DrawText(followCraft ? "F follow: craft" : "F follow: sun", 944, 276, 17, Color{170, 184, 204, 255});
    DrawText(("zoom: " + Fixed(zoom, 1)).c_str(), 944, 302, 17, Color{170, 184, 204, 255});
    DrawText("ENTER applies burn", 944, 328, 17, Color{255, 232, 150, 255});
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Orbital Mechanics - Solar System Orbit Planner");
    SetTargetFPS(60);

    std::vector<Planet> planets = MakePlanets();
    Craft craft = MakeEarthOrbitCraft();
    std::vector<Vector2> trail;
    trail.reserve(kTrailMax);

    float simTime = 0.0f;
    float timeScale = 1.0f;
    float zoom = 4.2f;
    bool paused = true;
    bool followCraft = false;
    bool draggingBurn = false;
    Vector2 burn{};
    Vector2 camera{};

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            craft = MakeEarthOrbitCraft();
            trail.clear();
            simTime = 0.0f;
            timeScale = 1.0f;
            burn = {};
            paused = true;
        }
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_F)) followCraft = !followCraft;
        if (IsKeyPressed(KEY_ENTER)) {
            craft.vel = Vector2Add(craft.vel, burn);
            burn = {};
            trail.clear();
        }
        if (IsKeyDown(KEY_EQUAL) || IsKeyDown(KEY_KP_ADD)) timeScale = std::min(10.0f, timeScale + 3.0f * GetFrameTime());
        if (IsKeyDown(KEY_MINUS) || IsKeyDown(KEY_KP_SUBTRACT)) timeScale = std::max(0.1f, timeScale - 3.0f * GetFrameTime());
        zoom += GetMouseWheelMove() * 0.35f;
        zoom = std::clamp(zoom, 1.6f, 9.0f);

        const float scale = zoom;
        camera = followCraft ? craft.pos : Vector2{0.0f, 0.0f};
        const Vector2 craftScreen = WorldToScreen(craft.pos, camera, scale);

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && Vector2Distance(GetMousePosition(), craftScreen) < 28.0f) {
            draggingBurn = true;
        }
        if (draggingBurn && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            const Vector2 mouseWorld = ScreenToWorld(GetMousePosition(), camera, scale);
            burn = Vector2Scale(Vector2Subtract(mouseWorld, craft.pos), 0.11f);
            const float maxBurn = 5.5f;
            if (Vector2Length(burn) > maxBurn) burn = Vector2Scale(Vector2Normalize(burn), maxBurn);
        }
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) draggingBurn = false;

        if (!paused) {
            const int steps = std::max(1, static_cast<int>(std::ceil(timeScale * GetFrameTime() / kDt)));
            const float dt = timeScale * GetFrameTime() / static_cast<float>(steps);
            for (int i = 0; i < steps; ++i) {
                StepCraft(&craft, dt);
                simTime += dt;
            }
            trail.push_back(craft.pos);
            if (static_cast<int>(trail.size()) > kTrailMax) trail.erase(trail.begin());
        }

        std::vector<Vector2> preview = BuildPreview(craft, burn);

        BeginDrawing();
        ClearBackground(Color{5, 9, 18, 255});
        DrawGrid(camera, scale);

        for (const Planet& planet : planets) {
            DrawCircleLinesV(WorldToScreen({0.0f, 0.0f}, camera, scale), planet.radius * scale, Color{48, 63, 82, 190});
        }

        DrawPolylineWorld(preview, camera, scale, Color{95, 255, 145, 255});
        DrawTrail(trail, camera, scale);

        const Vector2 sun = WorldToScreen({0.0f, 0.0f}, camera, scale);
        DrawCircleV(sun, 18.0f, Color{255, 190, 70, 255});
        DrawCircleLinesV(sun, 26.0f, Color{255, 225, 135, 135});

        for (const Planet& planet : planets) {
            const Vector2 p = PlanetPosition(planet, simTime);
            const Vector2 s = WorldToScreen(p, camera, scale);
            DrawCircleV(s, planet.visualRadius, planet.color);
            DrawCircleLinesV(s, planet.visualRadius + 9.0f, Color{planet.color.r, planet.color.g, planet.color.b, 70});
            DrawText(planet.name.c_str(), static_cast<int>(s.x + planet.visualRadius + 7.0f), static_cast<int>(s.y - 8.0f), 14, Color{190, 202, 218, 255});
        }

        const Vector2 craftNow = WorldToScreen(craft.pos, camera, scale);
        DrawCircleV(craftNow, 6.5f, Color{245, 248, 255, 255});
        DrawTriangle(
            {craftNow.x + 13.0f, craftNow.y},
            {craftNow.x - 7.0f, craftNow.y - 7.0f},
            {craftNow.x - 7.0f, craftNow.y + 7.0f},
            Color{245, 248, 255, 255});

        if (Vector2Length(burn) > 0.01f) {
            const Vector2 burnEnd = WorldToScreen(Vector2Add(craft.pos, Vector2Scale(burn, 9.0f)), camera, scale);
            DrawLineEx(craftNow, burnEnd, 4.0f, Color{255, 232, 100, 255});
            DrawCircleV(burnEnd, 7.0f, Color{255, 232, 100, 255});
        }

        DrawText("Drag from spacecraft to draw a maneuver. Green path previews the trajectory.", 54, 52, 20, RAYWHITE);
        DrawText("Mouse wheel zoom  +/- time warp  SPACE pause  ENTER apply burn  F follow  R reset", 54, kScreenHeight - 48, 18, Color{182, 195, 212, 255});
        DrawHud(simTime, timeScale, burn, paused, followCraft, zoom);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
