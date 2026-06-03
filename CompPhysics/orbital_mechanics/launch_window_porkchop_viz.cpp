#include "raylib.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr float kPi = 3.14159265358979323846f;

struct WindowResult {
    float c3;
    float tof;
    float phaseError;
};

WindowResult Evaluate(float departureDay, float arrivalDay, float synodicPeriod) {
    const float tof = std::max(1.0f, arrivalDay - departureDay);
    const float preferredTof = 255.0f;
    const float phase = std::sin((departureDay / synodicPeriod) * 2.0f * kPi);
    const float launchEnergy = 11.0f + 0.0009f * (tof - preferredTof) * (tof - preferredTof);
    const float phasePenalty = 18.0f * std::fabs(phase - 0.34f);
    const float shortTripPenalty = tof < 145.0f ? (145.0f - tof) * 0.14f : 0.0f;
    return {launchEnergy + phasePenalty + shortTripPenalty, tof, phase - 0.34f};
}

Color Heat(float c3) {
    const float t = std::clamp((c3 - 10.0f) / 70.0f, 0.0f, 1.0f);
    const unsigned char r = static_cast<unsigned char>(50 + 205 * t);
    const unsigned char g = static_cast<unsigned char>(210 - 145 * std::fabs(t - 0.34f));
    const unsigned char b = static_cast<unsigned char>(245 - 205 * t);
    return {r, g, b, 255};
}

std::string Fixed(float value, int precision = 1) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << value;
    return os.str();
}

Vector2 CellToScreen(int x, int y, Rectangle plot, int cols, int rows) {
    return {
        plot.x + (static_cast<float>(x) + 0.5f) * plot.width / cols,
        plot.y + (static_cast<float>(y) + 0.5f) * plot.height / rows,
    };
}

void DrawAxes(Rectangle plot) {
    DrawRectangleLinesEx(plot, 2.0f, Color{190, 205, 220, 255});
    DrawText("departure day", static_cast<int>(plot.x + plot.width * 0.40f), static_cast<int>(plot.y + plot.height + 38), 20, RAYWHITE);
    DrawText("arrival day", static_cast<int>(plot.x - 90), static_cast<int>(plot.y - 34), 20, RAYWHITE);
    for (int i = 0; i <= 6; ++i) {
        float x = plot.x + i * plot.width / 6.0f;
        DrawLineV({x, plot.y + plot.height}, {x, plot.y + plot.height + 8}, Color{190, 205, 220, 255});
        DrawText(TextFormat("%d", i * 60), static_cast<int>(x - 12), static_cast<int>(plot.y + plot.height + 13), 16, Color{165, 178, 195, 255});
    }
    for (int i = 0; i <= 6; ++i) {
        float y = plot.y + plot.height - i * plot.height / 6.0f;
        DrawLineV({plot.x - 8, y}, {plot.x, y}, Color{190, 205, 220, 255});
        DrawText(TextFormat("%d", i * 80 + 120), static_cast<int>(plot.x - 54), static_cast<int>(y - 8), 16, Color{165, 178, 195, 255});
    }
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Orbital Mechanics - Launch Window Porkchop");
    SetTargetFPS(60);

    Rectangle plot{88.0f, 92.0f, 760.0f, 560.0f};
    constexpr int cols = 96;
    constexpr int rows = 72;

    float selectedDeparture = 132.0f;
    float selectedArrival = 410.0f;
    float synodicPeriod = 780.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            selectedDeparture = 132.0f;
            selectedArrival = 410.0f;
            synodicPeriod = 780.0f;
        }
        if (IsKeyDown(KEY_RIGHT)) selectedDeparture += 80.0f * GetFrameTime();
        if (IsKeyDown(KEY_LEFT)) selectedDeparture -= 80.0f * GetFrameTime();
        if (IsKeyDown(KEY_UP)) selectedArrival += 100.0f * GetFrameTime();
        if (IsKeyDown(KEY_DOWN)) selectedArrival -= 100.0f * GetFrameTime();
        if (IsKeyDown(KEY_W)) synodicPeriod += 60.0f * GetFrameTime();
        if (IsKeyDown(KEY_S)) synodicPeriod -= 60.0f * GetFrameTime();

        selectedDeparture = std::clamp(selectedDeparture, 0.0f, 360.0f);
        selectedArrival = std::clamp(selectedArrival, 120.0f, 600.0f);
        synodicPeriod = std::clamp(synodicPeriod, 500.0f, 980.0f);

        float bestC3 = 1.0e9f;
        Vector2 bestPoint{};
        for (int y = 0; y < rows; ++y) {
            const float arrival = 120.0f + (static_cast<float>(y) / (rows - 1)) * 480.0f;
            for (int x = 0; x < cols; ++x) {
                const float departure = (static_cast<float>(x) / (cols - 1)) * 360.0f;
                WindowResult result = Evaluate(departure, arrival, synodicPeriod);
                if (arrival > departure + 70.0f && result.c3 < bestC3) {
                    bestC3 = result.c3;
                    bestPoint = CellToScreen(x, rows - 1 - y, plot, cols, rows);
                }
            }
        }

        WindowResult selected = Evaluate(selectedDeparture, selectedArrival, synodicPeriod);
        const float sx = plot.x + selectedDeparture / 360.0f * plot.width;
        const float sy = plot.y + plot.height - (selectedArrival - 120.0f) / 480.0f * plot.height;

        BeginDrawing();
        ClearBackground(Color{7, 11, 20, 255});

        for (int y = 0; y < rows; ++y) {
            const float arrival = 120.0f + (static_cast<float>(rows - 1 - y) / (rows - 1)) * 480.0f;
            for (int x = 0; x < cols; ++x) {
                const float departure = (static_cast<float>(x) / (cols - 1)) * 360.0f;
                WindowResult result = Evaluate(departure, arrival, synodicPeriod);
                Color c = arrival <= departure + 70.0f ? Color{22, 29, 41, 255} : Heat(result.c3);
                DrawRectangle(
                    static_cast<int>(plot.x + x * plot.width / cols),
                    static_cast<int>(plot.y + y * plot.height / rows),
                    static_cast<int>(std::ceil(plot.width / cols)) + 1,
                    static_cast<int>(std::ceil(plot.height / rows)) + 1,
                    c);
            }
        }

        DrawAxes(plot);
        DrawCircleV(bestPoint, 9.0f, Color{255, 255, 255, 255});
        DrawCircleLinesV(bestPoint, 15.0f, Color{20, 25, 35, 255});
        DrawLineV({sx, plot.y}, {sx, plot.y + plot.height}, Color{255, 255, 255, 145});
        DrawLineV({plot.x, sy}, {plot.x + plot.width, sy}, Color{255, 255, 255, 145});
        DrawCircleV({sx, sy}, 8.0f, Color{255, 238, 120, 255});

        DrawRectangle(900, 90, 305, 330, Color{15, 23, 36, 235});
        DrawRectangleLines(900, 90, 305, 330, Color{91, 118, 150, 255});
        DrawText("LAUNCH WINDOW", 930, 118, 24, RAYWHITE);
        DrawText(("departure: " + Fixed(selectedDeparture, 0) + " d").c_str(), 930, 168, 19, Color{210, 230, 255, 255});
        DrawText(("arrival:   " + Fixed(selectedArrival, 0) + " d").c_str(), 930, 196, 19, Color{210, 230, 255, 255});
        DrawText(("tof:       " + Fixed(selected.tof, 0) + " d").c_str(), 930, 224, 19, Color{255, 235, 175, 255});
        DrawText(("C3 index:  " + Fixed(selected.c3, 2)).c_str(), 930, 262, 19, Color{255, 235, 175, 255});
        DrawText(("phase err: " + Fixed(selected.phaseError, 2)).c_str(), 930, 290, 19, Color{190, 220, 255, 255});
        DrawText(("synodic:   " + Fixed(synodicPeriod, 0) + " d").c_str(), 930, 318, 19, Color{190, 220, 255, 255});
        DrawText(("best C3:   " + Fixed(bestC3, 2)).c_str(), 930, 356, 19, Color{220, 255, 220, 255});
        DrawText("white dot = best window", 930, 386, 16, Color{158, 170, 185, 255});

        DrawText("LEFT/RIGHT departure  UP/DOWN arrival  W/S synodic period  R reset", 88, 724, 19, Color{190, 200, 214, 255});
        DrawText("Synthetic porkchop map: lower C3 regions represent easier departure-energy windows.", 88, 754, 18, Color{142, 154, 170, 255});
        EndDrawing();
    }

    CloseWindow();
    return 0;
}

