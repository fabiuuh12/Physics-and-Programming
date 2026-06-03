#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr float kMu = 1.0f;
constexpr float kPi = 3.14159265358979323846f;

Vector2 ScreenCenter() {
    return {kScreenWidth * 0.5f - 80.0f, kScreenHeight * 0.54f};
}

Vector2 OrbitPoint(float radius, float angle, float scale) {
    Vector2 center = ScreenCenter();
    return {center.x + std::cos(angle) * radius * scale, center.y + std::sin(angle) * radius * scale};
}

Vector2 TransferPoint(float r1, float r2, float theta, float scale) {
    const float a = 0.5f * (r1 + r2);
    const float e = std::fabs(r2 - r1) / (r1 + r2);
    const float p = a * (1.0f - e * e);
    const bool outward = r2 >= r1;
    const float trueAnomaly = outward ? theta : theta + kPi;
    const float radius = p / (1.0f + e * std::cos(trueAnomaly));
    return OrbitPoint(radius, theta, scale);
}

void DrawOrbit(float radius, float scale, Color color) {
    Vector2 c = ScreenCenter();
    DrawCircleLines(static_cast<int>(c.x), static_cast<int>(c.y), radius * scale, color);
}

void DrawTransfer(float r1, float r2, float scale) {
    constexpr int segments = 180;
    for (int i = 1; i <= segments; ++i) {
        float a0 = kPi * static_cast<float>(i - 1) / segments;
        float a1 = kPi * static_cast<float>(i) / segments;
        DrawLineV(TransferPoint(r1, r2, a0, scale), TransferPoint(r1, r2, a1, scale), Color{255, 188, 84, 255});
    }
}

std::string Fixed(float value, int precision = 3) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << value;
    return os.str();
}

void DrawPanel(float r1, float r2, float progress, bool paused) {
    const float a = 0.5f * (r1 + r2);
    const float dv1 = std::sqrt(kMu / r1) * (std::sqrt((2.0f * r2) / (r1 + r2)) - 1.0f);
    const float dv2 = std::sqrt(kMu / r2) * (1.0f - std::sqrt((2.0f * r1) / (r1 + r2)));
    const float transferTime = kPi * std::sqrt((a * a * a) / kMu);
    const float phase = kPi - std::sqrt(kMu / (r2 * r2 * r2)) * transferTime;

    DrawRectangle(935, 55, 300, 300, Color{15, 23, 36, 232});
    DrawRectangleLines(935, 55, 300, 300, Color{90, 122, 150, 255});
    DrawText("HOHMANN TRANSFER", 958, 82, 22, Color{235, 244, 255, 255});
    DrawText(("parking r: " + Fixed(r1, 2)).c_str(), 958, 124, 18, Color{170, 215, 255, 255});
    DrawText(("target r:  " + Fixed(r2, 2)).c_str(), 958, 150, 18, Color{180, 255, 190, 255});
    DrawText(("transfer a: " + Fixed(a, 2)).c_str(), 958, 176, 18, RAYWHITE);
    DrawText(("burn 1 dv: " + Fixed(std::fabs(dv1))).c_str(), 958, 212, 18, Color{255, 210, 135, 255});
    DrawText(("burn 2 dv: " + Fixed(std::fabs(dv2))).c_str(), 958, 238, 18, Color{255, 210, 135, 255});
    DrawText(("total dv:  " + Fixed(std::fabs(dv1) + std::fabs(dv2))).c_str(), 958, 264, 18, Color{255, 238, 180, 255});
    DrawText(("phase: " + Fixed(phase * 180.0f / kPi, 1) + " deg").c_str(), 958, 300, 18, Color{210, 230, 255, 255});
    DrawText(paused ? "SPACE animate" : "SPACE pause", 958, 326, 16, Color{155, 166, 180, 255});
    DrawText(("transfer: " + Fixed(progress * 100.0f, 0) + "%").c_str(), 70, 70, 20, RAYWHITE);
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Orbital Mechanics - Hohmann Transfer");
    SetTargetFPS(60);

    float r1 = 1.85f;
    float r2 = 3.85f;
    float progress = 0.0f;
    bool paused = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            r1 = 1.85f;
            r2 = 3.85f;
            progress = 0.0f;
            paused = false;
        }
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyDown(KEY_RIGHT)) r2 += 0.7f * GetFrameTime();
        if (IsKeyDown(KEY_LEFT)) r2 -= 0.7f * GetFrameTime();
        if (IsKeyDown(KEY_UP)) r1 += 0.55f * GetFrameTime();
        if (IsKeyDown(KEY_DOWN)) r1 -= 0.55f * GetFrameTime();
        r1 = std::clamp(r1, 1.15f, 3.4f);
        r2 = std::clamp(r2, r1 + 0.35f, 5.3f);

        if (IsKeyDown(KEY_D)) progress += 0.34f * GetFrameTime();
        if (IsKeyDown(KEY_A)) progress -= 0.34f * GetFrameTime();
        if (!paused) progress += 0.08f * GetFrameTime();
        if (progress > 1.0f) progress -= 1.0f;
        if (progress < 0.0f) progress += 1.0f;

        const float scale = 88.0f;
        const float theta = progress * kPi;
        const Vector2 craft = TransferPoint(r1, r2, theta, scale);
        const Vector2 burn1 = TransferPoint(r1, r2, 0.0f, scale);
        const Vector2 burn2 = TransferPoint(r1, r2, kPi, scale);

        BeginDrawing();
        ClearBackground(Color{6, 10, 19, 255});

        for (int i = 0; i < 180; ++i) {
            float x = static_cast<float>((i * 97) % kScreenWidth);
            float y = static_cast<float>((i * 53) % kScreenHeight);
            DrawPixelV({x, y}, Color{80, 95, 118, static_cast<unsigned char>(70 + (i % 4) * 22)});
        }

        DrawOrbit(r1, scale, Color{76, 171, 255, 180});
        DrawOrbit(r2, scale, Color{101, 235, 132, 170});
        DrawTransfer(r1, r2, scale);
        DrawCircleV(ScreenCenter(), 32.0f, Color{255, 188, 80, 255});
        DrawCircleLinesV(ScreenCenter(), 38.0f, Color{255, 232, 170, 150});
        DrawCircleV(burn1, 7.0f, Color{255, 105, 97, 255});
        DrawCircleV(burn2, 7.0f, Color{120, 220, 255, 255});
        DrawCircleV(craft, 9.0f, Color{245, 246, 255, 255});
        DrawLineV(craft, {craft.x - 24.0f * std::sin(theta), craft.y + 24.0f * std::cos(theta)}, Color{245, 246, 255, 175});

        DrawText("BURN 1", static_cast<int>(burn1.x + 14), static_cast<int>(burn1.y - 10), 16, Color{255, 170, 160, 255});
        DrawText("BURN 2", static_cast<int>(burn2.x - 76), static_cast<int>(burn2.y - 10), 16, Color{155, 230, 255, 255});
        DrawText("parking orbit", 72, 110, 18, Color{115, 195, 255, 255});
        DrawText("target orbit", 72, 136, 18, Color{135, 245, 155, 255});
        DrawText("LEFT/RIGHT target  UP/DOWN parking  A/D scrub  R reset", 70, kScreenHeight - 52, 18, Color{185, 195, 210, 255});

        DrawPanel(r1, r2, progress, paused);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
