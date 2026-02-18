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

struct SamplePoint {
    Vector3 pos;
    float size;
    Color color;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 26.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float PsiN(float x, float L, int n) {
    if (x < 0.0f || x > L) return 0.0f;
    return std::sqrt(2.0f / L) * std::sin(static_cast<float>(n) * PI * x / L);
}

float Density(float x, float L, int n) {
    float psi = PsiN(x, L, n);
    return psi * psi;
}

std::vector<SamplePoint> BuildCloud(float L, int n, float phase) {
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
            float y = 0.08f + 0.8f * d + 0.06f * std::sin(phase + 0.23f * static_cast<float>(i + k));

            float hue = 0.5f + 0.5f * std::sin(phase + 8.0f * x / L);
            Color c = Color{
                static_cast<unsigned char>(80 + 70 * hue),
                static_cast<unsigned char>(140 + 90 * (1.0f - hue)),
                static_cast<unsigned char>(210 + 40 * hue),
                static_cast<unsigned char>(100 + 130 * d)
            };
            pts.push_back({{x - L * 0.5f, y, z}, 0.015f + 0.02f * d, c});
        }
    }

    return pts;
}

std::string Hud(int n, float L, bool paused) {
    std::ostringstream os;
    os << "n=" << n << "  L=" << std::fixed << std::setprecision(2) << L;
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
    float t = 0.0f;

    while (!WindowShouldClose()) {
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

        if (!paused) {
            t += GetFrameTime();
        }

        float phase = 2.6f * t;
        std::vector<SamplePoint> cloud = BuildCloud(L, n, phase);

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
            float y0 = 0.03f + 0.95f * d0 / (2.0f / L);
            float y1 = 0.03f + 0.95f * d1 / (2.0f / L);

            Vector3 p0 = {x0 - L * 0.5f, y0, -0.95f};
            Vector3 p1 = {x1 - L * 0.5f, y1, -0.95f};
            DrawLine3D(p0, p1, Color{255, 210, 120, 210});
        }

        for (const SamplePoint& sp : cloud) {
            DrawSphere(sp.pos, sp.size, sp.color);
        }

        EndMode3D();

        DrawText("Quantum Particle in a 1D Box (3D view)", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] quantum number n | +/- box length L | P pause | R reset", 20, 54, 18, Color{164, 183, 210, 255});
        std::string hud = Hud(n, L, paused);
        DrawText(hud.c_str(), 20, 82, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 112);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
