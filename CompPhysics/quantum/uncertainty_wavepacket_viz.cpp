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

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 34.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

float Gaussian(float x, float mu, float sigma) {
    float u = (x - mu) / sigma;
    return std::exp(-0.5f * u * u) / std::max(0.001f, sigma);
}
}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Uncertainty Wavepacket 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.5f, 5.0f, 9.0f};
    camera.target = {0.0f, 0.6f, 0.0f};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.0f;

    float x0 = -5.0f;
    float p0 = 2.6f;
    float sigma0 = 0.55f;
    bool paused=false;
    float t=0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused=!paused;
        if (IsKeyPressed(KEY_R)) { x0=-5.0f; p0=2.6f; sigma0=0.55f; paused=false; t=0.0f; }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) sigma0 = std::max(0.2f, sigma0 - 0.05f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) sigma0 = std::min(1.5f, sigma0 + 0.05f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) p0 = std::max(0.2f, p0 - 0.1f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) p0 = std::min(6.0f, p0 + 0.1f);

        UpdateOrbitCameraDragOnly(&camera,&camYaw,&camPitch,&camDistance);
        if (!paused) t += GetFrameTime();

        float sigma = std::sqrt(sigma0*sigma0 + 0.22f*t*t/sigma0/sigma0);
        float center = x0 + p0 * t * 0.55f;
        float dx = sigma;
        float dp = 1.0f / std::max(0.05f, 2.0f*sigma);

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);



        for (int i=0;i<160;++i) {
            float x = -7.5f + 15.0f * i / 159.0f;
            float p = Gaussian(x, center, sigma);
            float y = 0.1f + 2.6f * p;
            float z = 0.35f * std::sin(6.0f*x - 4.0f*t) * p;
            DrawSphere({x,y,z}, 0.03f + 0.05f*p, Color{130,220,255,220});
        }

        DrawLine3D({center - dx, 0.08f, -1.3f}, {center + dx, 0.08f, -1.3f}, Color{255, 170, 120, 255});
        DrawLine3D({-6.8f, 0.08f, -1.6f}, {-6.8f, 0.08f + 1.8f*dp, -1.6f}, Color{255, 220, 130, 255});

        EndMode3D();

        DrawText("Heisenberg Uncertainty: Wavepacket Spreading", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] sigma_x(0) | +/- momentum | P pause | R reset", 20, 54, 18, Color{164,183,210,255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(3)
           << "x_center=" << center << "  sigma_x~" << dx << "  sigma_p~" << dp << "  product~" << dx*dp;
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126,224,255,255});
        DrawText("Orange bar: position spread  |  Yellow bar: momentum spread", 20, 110, 18, Color{190,205,225,255});
        DrawFPS(20, 138);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
