#include "raylib.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr int kGridW = 150;
constexpr int kGridH = 120;
constexpr int kCell = 6;
constexpr int kFieldX = 24;
constexpr int kFieldY = 56;
constexpr int kFieldW = kGridW * kCell;
constexpr int kFieldH = kGridH * kCell;
constexpr int kPanelX = kFieldX + kFieldW + 22;
constexpr int kPanelW = kScreenWidth - kPanelX - 24;

struct MatterType {
    float speed;
    float damping;
    Color color;
    const char* name;
};

constexpr std::array<MatterType, 4> kMaterials{{
    {1.00f, 0.010f, {16, 24, 38, 255}, "vacuum"},
    {0.58f, 0.025f, {42, 120, 150, 255}, "glass"},
    {0.34f, 0.040f, {72, 164, 111, 255}, "dense"},
    {0.04f, 0.185f, {170, 154, 122, 255}, "metal"},
}};

int Idx(int x, int y) {
    return y * kGridW + x;
}

float ClampAbs(float value, float limit) {
    return std::clamp(value, -limit, limit);
}

Color WaveColor(float e, int material) {
    const MatterType& m = kMaterials[material];
    float a = std::clamp(std::fabs(e) * 1.9f, 0.0f, 1.0f);
    Color wave = e >= 0.0f ? Color{60, 190, 255, 255} : Color{255, 138, 84, 255};
    return {
        static_cast<unsigned char>(m.color.r * (1.0f - a) + wave.r * a),
        static_cast<unsigned char>(m.color.g * (1.0f - a) + wave.g * a),
        static_cast<unsigned char>(m.color.b * (1.0f - a) + wave.b * a),
        255
    };
}

void ClearWaves(std::vector<float>& prev, std::vector<float>& curr, std::vector<float>& next) {
    std::fill(prev.begin(), prev.end(), 0.0f);
    std::fill(curr.begin(), curr.end(), 0.0f);
    std::fill(next.begin(), next.end(), 0.0f);
}

bool DrawButton(Rectangle r, const char* label, bool active = false) {
    Vector2 mouse = GetMousePosition();
    bool hover = CheckCollisionPointRec(mouse, r);
    Color fill = active ? Color{58, 128, 178, 255} : (hover ? Color{44, 58, 80, 255} : Color{24, 32, 48, 255});
    DrawRectangleRounded(r, 0.12f, 6, fill);
    DrawRectangleRoundedLines(r, 0.12f, 6, active ? Color{126, 224, 255, 255} : Color{76, 94, 122, 255});
    int textWidth = MeasureText(label, 16);
    DrawText(label, static_cast<int>(r.x + (r.width - textWidth) * 0.5f), static_cast<int>(r.y + 8), 16, {224, 232, 244, 255});
    return hover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
}

bool DrawSlider(Rectangle r, const char* label, float* value, float minValue, float maxValue) {
    Vector2 mouse = GetMousePosition();
    bool active = CheckCollisionPointRec(mouse, r) && IsMouseButtonDown(MOUSE_LEFT_BUTTON);
    if (active) {
        float t = std::clamp((mouse.x - r.x) / r.width, 0.0f, 1.0f);
        *value = minValue + (maxValue - minValue) * t;
    }

    std::ostringstream os;
    os << label << "  " << std::fixed << std::setprecision(2) << *value;
    DrawText(os.str().c_str(), static_cast<int>(r.x), static_cast<int>(r.y - 24), 17, {204, 216, 236, 255});
    DrawRectangleRounded(r, 0.5f, 8, {22, 30, 45, 255});
    float t = (*value - minValue) / (maxValue - minValue);
    DrawRectangleRounded({r.x, r.y, r.width * t, r.height}, 0.5f, 8, {58, 146, 184, 255});
    DrawCircle(static_cast<int>(r.x + r.width * t), static_cast<int>(r.y + r.height * 0.5f), 10.0f, {232, 238, 248, 255});
    return active;
}

bool DrawIntSlider(Rectangle r, const char* label, int* value, int minValue, int maxValue) {
    float f = static_cast<float>(*value);
    bool changed = DrawSlider(r, label, &f, static_cast<float>(minValue), static_cast<float>(maxValue));
    *value = static_cast<int>(std::round(f));
    return changed;
}

void DrawPanel(int* brush,
               int* radius,
               float* frequency,
               float* sourceAmp,
               bool* paused,
               std::vector<int>& mat,
               std::vector<float>& prev,
               std::vector<float>& curr,
               std::vector<float>& next) {
    DrawText("EM Wave Field Lab", kPanelX, 24, 28, {232, 238, 248, 255});
    DrawText("Paint matter. The light wave slows, bends, absorbs, and reflects.", kPanelX, 62, 16, {178, 195, 218, 255});

    DrawText("Mouse", kPanelX, 104, 20, {232, 238, 248, 255});
    DrawText("Left drag: paint selected material", kPanelX, 142, 17, {184, 199, 222, 255});
    DrawText("Right drag: erase to vacuum", kPanelX, 168, 17, {184, 199, 222, 255});

    DrawText("Materials", kPanelX, 214, 20, {232, 238, 248, 255});
    for (int i = 0; i < static_cast<int>(kMaterials.size()); ++i) {
        int y = 246 + i * 38;
        Rectangle button{static_cast<float>(kPanelX), static_cast<float>(y), static_cast<float>(kPanelW), 30.0f};
        if (DrawButton(button, kMaterials[i].name, i == *brush)) *brush = i;
        DrawRectangle(kPanelX + 8, y + 7, 16, 16, kMaterials[i].color);
        std::ostringstream os;
        os << "n~" << std::fixed << std::setprecision(2) << (1.0f / std::max(0.08f, kMaterials[i].speed));
        DrawText(os.str().c_str(), kPanelX + kPanelW - 54, y + 8, 14, {184, 199, 222, 255});
    }

    DrawText("Settings", kPanelX, 420, 20, {232, 238, 248, 255});
    DrawIntSlider({static_cast<float>(kPanelX), 464.0f, static_cast<float>(kPanelW), 12.0f}, "Brush size", radius, 1, 15);
    DrawSlider({static_cast<float>(kPanelX), 524.0f, static_cast<float>(kPanelW), 12.0f}, "Source frequency", frequency, 0.45f, 8.0f);
    DrawSlider({static_cast<float>(kPanelX), 584.0f, static_cast<float>(kPanelW), 12.0f}, "Source strength", sourceAmp, 0.15f, 1.5f);

    DrawText("Actions", kPanelX, 640, 20, {232, 238, 248, 255});
    if (DrawButton({static_cast<float>(kPanelX), 674.0f, 86.0f, 34.0f}, *paused ? "Resume" : "Pause", *paused)) *paused = !*paused;
    if (DrawButton({static_cast<float>(kPanelX + 96), 674.0f, 86.0f, 34.0f}, "Reset")) ClearWaves(prev, curr, next);
    if (DrawButton({static_cast<float>(kPanelX + 192), 674.0f, 86.0f, 34.0f}, "Clear")) std::fill(mat.begin(), mat.end(), 0);

    DrawText("Blue/orange: electric-field phase", kPanelX, 746, 16, {184, 199, 222, 255});
    DrawText("Bright fronts: stronger light amplitude", kPanelX, 770, 16, {184, 199, 222, 255});
}
} // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Electromagnetic Wave Field Lab - C++ (raylib)");
    SetTargetFPS(60);

    std::vector<float> prev(kGridW * kGridH, 0.0f);
    std::vector<float> curr(kGridW * kGridH, 0.0f);
    std::vector<float> next(kGridW * kGridH, 0.0f);
    std::vector<int> mat(kGridW * kGridH, 0);
    std::vector<Color> pixels(kGridW * kGridH);

    Image image = GenImageColor(kGridW, kGridH, BLACK);
    Texture2D texture = LoadTextureFromImage(image);
    UnloadImage(image);

    int brush = 1;
    int brushRadius = 4;
    float frequency = 2.5f;
    float sourceAmp = 0.95f;
    float time = 0.0f;
    bool paused = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ONE)) brush = 0;
        if (IsKeyPressed(KEY_TWO)) brush = 1;
        if (IsKeyPressed(KEY_THREE)) brush = 2;
        if (IsKeyPressed(KEY_FOUR)) brush = 3;
        if (IsKeyPressed(KEY_LEFT_BRACKET)) brushRadius = std::max(1, brushRadius - 1);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) brushRadius = std::min(15, brushRadius + 1);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) frequency = std::max(0.45f, frequency - 0.15f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) frequency = std::min(8.0f, frequency + 0.15f);
        if (IsKeyPressed(KEY_COMMA)) sourceAmp = std::max(0.15f, sourceAmp - 0.05f);
        if (IsKeyPressed(KEY_PERIOD)) sourceAmp = std::min(1.5f, sourceAmp + 0.05f);
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) ClearWaves(prev, curr, next);
        if (IsKeyPressed(KEY_C)) std::fill(mat.begin(), mat.end(), 0);

        Vector2 mouse = GetMousePosition();
        bool inField = mouse.x >= kFieldX && mouse.x < kFieldX + kFieldW && mouse.y >= kFieldY && mouse.y < kFieldY + kFieldH;
        if (inField && (IsMouseButtonDown(MOUSE_LEFT_BUTTON) || IsMouseButtonDown(MOUSE_RIGHT_BUTTON))) {
            int gx = static_cast<int>((mouse.x - kFieldX) / kCell);
            int gy = static_cast<int>((mouse.y - kFieldY) / kCell);
            int paint = IsMouseButtonDown(MOUSE_RIGHT_BUTTON) ? 0 : brush;
            for (int y = gy - brushRadius; y <= gy + brushRadius; ++y) {
                for (int x = gx - brushRadius; x <= gx + brushRadius; ++x) {
                    if (x <= 0 || x >= kGridW - 1 || y <= 0 || y >= kGridH - 1) continue;
                    int dx = x - gx;
                    int dy = y - gy;
                    if (dx * dx + dy * dy <= brushRadius * brushRadius) mat[Idx(x, y)] = paint;
                }
            }
        }

        if (!paused) {
            constexpr int kSteps = 2;
            constexpr float dt = 0.38f;
            for (int step = 0; step < kSteps; ++step) {
                time += GetFrameTime() / kSteps;

                for (int y = 1; y < kGridH - 1; ++y) {
                    for (int x = 1; x < kGridW - 1; ++x) {
                        int id = Idx(x, y);
                        const MatterType& material = kMaterials[mat[id]];
                        float lap = curr[Idx(x + 1, y)] + curr[Idx(x - 1, y)] + curr[Idx(x, y + 1)] + curr[Idx(x, y - 1)] - 4.0f * curr[id];
                        float edgeLoss = (x < 6 || x > kGridW - 7 || y < 6 || y > kGridH - 7) ? 0.10f : 0.0f;
                        float damping = material.damping + edgeLoss;
                        next[id] = (2.0f - damping) * curr[id] - (1.0f - damping) * prev[id] + material.speed * material.speed * dt * dt * lap;
                        next[id] = ClampAbs(next[id], 2.0f);
                    }
                }

                int sx = 18;
                int sy = kGridH / 2;
                for (int y = sy - 5; y <= sy + 5; ++y) {
                    if (y <= 0 || y >= kGridH - 1) continue;
                    float taper = 1.0f - std::fabs(static_cast<float>(y - sy)) / 7.0f;
                    next[Idx(sx, y)] += sourceAmp * taper * std::sin(time * frequency * 6.28318f) * 0.22f;
                }

                std::swap(prev, curr);
                std::swap(curr, next);
                std::fill(next.begin(), next.end(), 0.0f);
            }
        }

        for (int y = 0; y < kGridH; ++y) {
            for (int x = 0; x < kGridW; ++x) {
                pixels[Idx(x, y)] = WaveColor(curr[Idx(x, y)], mat[Idx(x, y)]);
            }
        }
        UpdateTexture(texture, pixels.data());

        BeginDrawing();
        ClearBackground({8, 11, 18, 255});

        DrawText("Light Source", kFieldX + 44, 24, 18, {224, 232, 244, 255});
        DrawRectangle(kFieldX - 2, kFieldY - 2, kFieldW + 4, kFieldH + 4, {58, 72, 96, 255});
        DrawTextureEx(texture, {static_cast<float>(kFieldX), static_cast<float>(kFieldY)}, 0.0f, static_cast<float>(kCell), WHITE);
        DrawCircle(kFieldX + 18 * kCell + kCell / 2, kFieldY + (kGridH / 2) * kCell + kCell / 2, 8.0f, {255, 238, 138, 255});
        DrawCircleLines(kFieldX + 18 * kCell + kCell / 2, kFieldY + (kGridH / 2) * kCell + kCell / 2, 14.0f, {255, 238, 138, 180});

        if (inField) {
            int gx = static_cast<int>((mouse.x - kFieldX) / kCell);
            int gy = static_cast<int>((mouse.y - kFieldY) / kCell);
            DrawCircleLines(kFieldX + gx * kCell + kCell / 2, kFieldY + gy * kCell + kCell / 2, static_cast<float>(brushRadius * kCell), WHITE);
        }

        DrawPanel(&brush, &brushRadius, &frequency, &sourceAmp, &paused, mat, prev, curr, next);
        DrawFPS(kPanelX, kScreenHeight - 36);
        EndDrawing();
    }

    UnloadTexture(texture);
    CloseWindow();
    return 0;
}
