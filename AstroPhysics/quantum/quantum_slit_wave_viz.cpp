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

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 30.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float IntensityAtScreen(float y, float slitSep, float wavelength, float screenX) {
    Vector2 s1 = {0.0f, 0.5f * slitSep};
    Vector2 s2 = {0.0f, -0.5f * slitSep};
    Vector2 p = {screenX, y};

    float r1 = Vector2Distance(p, s1);
    float r2 = Vector2Distance(p, s2);
    float phase = 2.0f * PI * (r1 - r2) / wavelength;
    float envelope = std::exp(-0.045f * y * y);
    return envelope * 0.5f * (1.0f + std::cos(phase));
}

std::string Hud(float slitSep, float wavelength, float freq, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "slitSep=" << slitSep
       << "  lambda=" << wavelength
       << "  freq=" << freq;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Double Slit Quantum Wave 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.8f, 5.0f, 8.8f};
    camera.target = {2.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.84f;
    float camPitch = 0.31f;
    float camDistance = 13.0f;

    const float sourceX = -5.0f;
    const float barrierX = 0.0f;
    const float screenX = 7.0f;

    float slitSep = 1.45f;
    float slitGap = 0.24f;
    float wavelength = 1.05f;
    float waveFreq = 2.6f;
    float timeScale = 1.0f;
    bool paused = false;
    float t = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            slitSep = 1.45f;
            wavelength = 1.05f;
            waveFreq = 2.6f;
            timeScale = 1.0f;
            t = 0.0f;
            paused = false;
        }

        if (IsKeyPressed(KEY_LEFT_BRACKET)) slitSep = std::max(0.6f, slitSep - 0.05f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) slitSep = std::min(2.8f, slitSep + 0.05f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) wavelength = std::max(0.35f, wavelength - 0.04f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) wavelength = std::min(2.0f, wavelength + 0.04f);
        if (IsKeyPressed(KEY_COMMA)) waveFreq = std::max(0.4f, waveFreq - 0.1f);
        if (IsKeyPressed(KEY_PERIOD)) waveFreq = std::min(8.0f, waveFreq + 0.1f);
        if (IsKeyPressed(KEY_SEMICOLON)) timeScale = std::max(0.2f, timeScale - 0.2f);
        if (IsKeyPressed(KEY_APOSTROPHE)) timeScale = std::min(6.0f, timeScale + 0.2f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            t += GetFrameTime() * timeScale;
        }

        BeginDrawing();
        ClearBackground(Color{6, 9, 17, 255});

        BeginMode3D(camera);

        DrawSphere({sourceX, 0.0f, 0.0f}, 0.18f, Color{255, 210, 120, 255});

        DrawCube({barrierX, 0.0f, 0.0f}, 0.2f, 4.0f, 2.5f, Color{100, 110, 130, 120});

        DrawCube({barrierX,  0.5f * slitSep + slitGap, 0.0f}, 0.24f, 1.6f, 2.6f, Color{6, 9, 17, 255});
        DrawCube({barrierX, -0.5f * slitSep - slitGap, 0.0f}, 0.24f, 1.6f, 2.6f, Color{6, 9, 17, 255});

        DrawCube({screenX, 0.0f, 0.0f}, 0.12f, 5.0f, 2.8f, Color{110, 130, 170, 170});

        Vector3 slit1 = {barrierX,  0.5f * slitSep, 0.0f};
        Vector3 slit2 = {barrierX, -0.5f * slitSep, 0.0f};

        DrawLine3D({sourceX, 0.0f, 0.0f}, slit1, Color{255, 120, 100, 170});
        DrawLine3D({sourceX, 0.0f, 0.0f}, slit2, Color{255, 120, 100, 170});

        for (int i = 0; i < 18; ++i) {
            float r = std::fmod(t * waveFreq + 0.5f * i, 12.0f);
            float alpha = static_cast<float>(1.0 - i / 18.0);

            int segs = 80;
            for (int s = 0; s < segs; ++s) {
                float a0 = 2.0f * PI * static_cast<float>(s) / static_cast<float>(segs);
                float a1 = 2.0f * PI * static_cast<float>(s + 1) / static_cast<float>(segs);
                Vector3 p0a = {slit1.x + r * std::cos(a0), slit1.y + r * std::sin(a0), 0.0f};
                Vector3 p1a = {slit1.x + r * std::cos(a1), slit1.y + r * std::sin(a1), 0.0f};
                Vector3 p0b = {slit2.x + r * std::cos(a0), slit2.y + r * std::sin(a0), 0.0f};
                Vector3 p1b = {slit2.x + r * std::cos(a1), slit2.y + r * std::sin(a1), 0.0f};

                Color c1 = Color{110, 190, 255, static_cast<unsigned char>(30 + 90 * alpha)};
                DrawLine3D(p0a, p1a, c1);
                DrawLine3D(p0b, p1b, c1);
            }
        }

        const int bins = 180;
        for (int i = 0; i < bins; ++i) {
            float y0 = -2.3f + 4.6f * static_cast<float>(i) / static_cast<float>(bins - 1);
            float y1 = -2.3f + 4.6f * static_cast<float>(i + 1) / static_cast<float>(bins - 1);

            float I0 = IntensityAtScreen(y0, slitSep, wavelength, screenX - barrierX);
            float I1 = IntensityAtScreen(y1, slitSep, wavelength, screenX - barrierX);

            Color c = Color{
                static_cast<unsigned char>(80 + 170 * I0),
                static_cast<unsigned char>(90 + 130 * I0),
                static_cast<unsigned char>(160 + 90 * I0),
                255
            };

            Vector3 p0 = {screenX + 0.08f, y0, 0.0f};
            Vector3 p1 = {screenX + 0.08f, y1, 0.0f};
            DrawLine3D(p0, p1, c);

            DrawSphere({screenX + 0.15f + 0.4f * I0, y0, 0.0f}, 0.012f + 0.02f * I0, Color{255, 220, 130, 180});

            (void)I1;
        }

        EndMode3D();

        DrawText("Double-Slit Interference (Wave Picture)", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] slit sep | +/- wavelength | , . freq | ; ' time | P pause | R reset", 20, 54, 18, Color{164, 183, 210, 255});
        std::string hud = Hud(slitSep, wavelength, waveFreq, paused);
        DrawText(hud.c_str(), 20, 82, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 112);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
