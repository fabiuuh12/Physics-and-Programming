#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

constexpr int kScreenWidth = 1360;
constexpr int kScreenHeight = 860;

enum class PolarizationMode {
    kLinear,
    kCircular,
    kElliptical
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.2f, 1.2f);
    }

    *distance -= GetMouseWheelMove() * 0.65f;
    *distance = std::clamp(*distance, 5.0f, 28.0f);

    const float cp = std::cos(*pitch);
    const Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

Color WithAlpha(Color c, unsigned char alpha) {
    c.a = alpha;
    return c;
}

const char* PolarizationName(PolarizationMode mode) {
    switch (mode) {
        case PolarizationMode::kLinear: return "linear";
        case PolarizationMode::kCircular: return "circular";
        case PolarizationMode::kElliptical: return "elliptical";
    }
    return "unknown";
}

Vector3 ElectricField(float x, float t, float amplitude, float waveNumber, float omega, PolarizationMode mode, float ellipticity, bool rightHanded) {
    const float phi = waveNumber * x - omega * t;
    const float handed = rightHanded ? 1.0f : -1.0f;

    float ey = amplitude * std::cos(phi);
    float ez = 0.0f;

    if (mode == PolarizationMode::kCircular) {
        ez = handed * amplitude * std::sin(phi);
    } else if (mode == PolarizationMode::kElliptical) {
        ez = handed * amplitude * ellipticity * std::sin(phi);
    }

    return {0.0f, ey, ez};
}

Vector3 MagneticFieldFromE(const Vector3& eField) {
    return {0.0f, -eField.z, eField.y};
}

void DrawArrow3D(const Vector3& from, const Vector3& to, float radius, Color color) {
    const Vector3 d = Vector3Subtract(to, from);
    const float len = Vector3Length(d);
    if (len < 1e-4f) return;

    const Vector3 dir = Vector3Scale(d, 1.0f / len);
    const Vector3 tipBase = Vector3Add(to, Vector3Scale(dir, -std::min(0.22f, len * 0.35f)));
    DrawCylinderEx(from, tipBase, radius, radius, 8, color);
    DrawCylinderEx(tipBase, to, radius * 1.9f, 0.0f, 8, color);
}

void DrawHelixCurve(bool electric,
                    float t,
                    float amplitude,
                    float waveNumber,
                    float omega,
                    PolarizationMode mode,
                    float ellipticity,
                    bool rightHanded,
                    Color color) {
    constexpr int kSegments = 260;
    constexpr float kXMin = -7.0f;
    constexpr float kXMax = 7.0f;

    auto curvePoint = [&](float x) {
        const Vector3 field = electric
            ? ElectricField(x, t, amplitude, waveNumber, omega, mode, ellipticity, rightHanded)
            : MagneticFieldFromE(ElectricField(x, t, amplitude, waveNumber, omega, mode, ellipticity, rightHanded));
        return Vector3Add({x, 0.0f, 0.0f}, field);
    };

    Vector3 prev = curvePoint(kXMin);
    for (int i = 1; i <= kSegments; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(kSegments);
        const float x = kXMin + (kXMax - kXMin) * u;
        const Vector3 cur = curvePoint(x);
        DrawCylinderEx(prev, cur, electric ? 0.048f : 0.04f, electric ? 0.048f : 0.04f, 8, color);
        DrawCylinderEx(prev, cur, electric ? 0.082f : 0.072f, electric ? 0.082f : 0.072f, 8, WithAlpha(color, 22));
        prev = cur;
    }
}

std::string HudLine(PolarizationMode mode, bool rightHanded, float amplitude, float waveNumber, float omega, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "mode=" << PolarizationName(mode)
       << "  handedness=" << (rightHanded ? "right" : "left")
       << "  A=" << amplitude
       << "  k=" << waveNumber
       << "  omega=" << omega;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Helical EM Wave + Poynting Flow 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.8f, 5.4f, 8.8f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 44.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f;
    float camPitch = 0.34f;
    float camDistance = 14.0f;

    PolarizationMode mode = PolarizationMode::kCircular;
    bool rightHanded = true;
    bool paused = false;
    float t = 0.0f;

    float amplitude = 1.05f;
    float waveNumber = 1.45f;
    float omega = 2.0f;
    float ellipticity = 0.52f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            mode = PolarizationMode::kCircular;
            rightHanded = true;
            paused = false;
            t = 0.0f;
            amplitude = 1.05f;
            waveNumber = 1.45f;
            omega = 2.0f;
            ellipticity = 0.52f;
        }
        if (IsKeyPressed(KEY_ONE)) mode = PolarizationMode::kLinear;
        if (IsKeyPressed(KEY_TWO)) mode = PolarizationMode::kCircular;
        if (IsKeyPressed(KEY_THREE)) mode = PolarizationMode::kElliptical;
        if (IsKeyPressed(KEY_H)) rightHanded = !rightHanded;
        if (IsKeyPressed(KEY_LEFT_BRACKET)) amplitude = std::max(0.25f, amplitude - 0.08f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) amplitude = std::min(1.9f, amplitude + 0.08f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) omega = std::max(0.4f, omega - 0.1f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) omega = std::min(6.0f, omega + 0.1f);
        if (IsKeyPressed(KEY_COMMA)) waveNumber = std::max(0.5f, waveNumber - 0.08f);
        if (IsKeyPressed(KEY_PERIOD)) waveNumber = std::min(3.0f, waveNumber + 0.08f);
        if (IsKeyPressed(KEY_SEMICOLON)) ellipticity = std::max(0.1f, ellipticity - 0.05f);
        if (IsKeyPressed(KEY_APOSTROPHE)) ellipticity = std::min(1.0f, ellipticity + 0.05f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);
        if (!paused) t += GetFrameTime();

        BeginDrawing();
        ClearBackground(Color{6, 9, 16, 255});

        DrawRectangleGradientV(0, 0, kScreenWidth, 160, Color{10, 18, 30, 220}, Color{10, 18, 30, 20});
        DrawRectangleGradientV(0, kScreenHeight - 130, kScreenWidth, 130, Color{7, 12, 20, 20}, Color{7, 12, 20, 220});

        BeginMode3D(camera);

        DrawPlane({0.0f, -2.0f, 0.0f}, {18.0f, 18.0f}, Color{13, 18, 27, 255});
        DrawLine3D({-7.4f, 0.0f, 0.0f}, {7.4f, 0.0f, 0.0f}, Color{170, 200, 240, 180});
        DrawLine3D({-7.4f, 0.0f, 0.0f}, {7.4f, 0.0f, 0.0f}, Color{170, 200, 240, 40});

        for (int i = -7; i <= 7; ++i) {
            DrawLine3D({static_cast<float>(i), -0.04f, -1.8f}, {static_cast<float>(i), -0.04f, 1.8f}, Color{40, 60, 90, 70});
        }

        DrawHelixCurve(true, t, amplitude, waveNumber, omega, mode, ellipticity, rightHanded, Color{110, 225, 255, 255});
        DrawHelixCurve(false, t, amplitude * 0.82f, waveNumber, omega, mode, ellipticity, rightHanded, Color{255, 175, 115, 255});

        constexpr int kSamples = 13;
        for (int i = 0; i < kSamples; ++i) {
            const float u = static_cast<float>(i) / static_cast<float>(kSamples - 1);
            const float x = -6.0f + 12.0f * u;
            const Vector3 base = {x, 0.0f, 0.0f};
            const Vector3 e = ElectricField(x, t, amplitude, waveNumber, omega, mode, ellipticity, rightHanded);
            const Vector3 b = MagneticFieldFromE(e);
            const Vector3 s = Vector3Scale(Vector3Normalize(Vector3CrossProduct(e, b)), 0.78f);

            DrawArrow3D(base, Vector3Add(base, e), 0.018f, Color{110, 225, 255, 255});
            DrawArrow3D(base, Vector3Add(base, b), 0.018f, Color{255, 175, 115, 255});
            DrawArrow3D({x, -1.25f, 0.0f}, {x, -1.25f, 0.0f + 0.0f}, 0.0f, BLANK);
            DrawArrow3D({x, -1.2f, -0.9f}, Vector3Add({x, -1.2f, -0.9f}, s), 0.016f, Color{160, 255, 170, 255});
            DrawSphere(base, 0.04f, Color{220, 232, 248, 180});
        }

        const float probeX = 1.4f;
        const Vector3 probeBase = {probeX, 0.0f, 0.0f};
        const Vector3 probeE = ElectricField(probeX, t, amplitude, waveNumber, omega, mode, ellipticity, rightHanded);
        const Vector3 probeB = MagneticFieldFromE(probeE);
        const Vector3 probeS = Vector3Scale(Vector3Normalize(Vector3CrossProduct(probeE, probeB)), 1.35f);
        DrawSphere(probeBase, 0.07f, Color{255, 235, 170, 255});
        DrawArrow3D(probeBase, Vector3Add(probeBase, probeE), 0.03f, Color{110, 225, 255, 255});
        DrawArrow3D(probeBase, Vector3Add(probeBase, probeB), 0.03f, Color{255, 175, 115, 255});
        DrawArrow3D({probeX, -1.55f, 0.0f}, Vector3Add({probeX, -1.55f, 0.0f}, probeS), 0.03f, Color{160, 255, 170, 255});

        EndMode3D();

        DrawText("Helical EM Wave + Poynting Flow", 20, 18, 31, Color{235, 240, 252, 255});
        DrawText("The electric and magnetic fields rotate as the wave propagates along +x. The green arrows show forward energy transport via the Poynting vector.", 20, 56, 18, Color{170, 186, 214, 255});
        DrawText("Mouse drag: orbit | wheel: zoom | 1 linear | 2 circular | 3 elliptical | H handedness", 20, 84, 17, Color{170, 186, 214, 255});
        DrawText("[ ] amplitude | +/- omega | , . wave number | ; ' ellipticity | P pause | R reset", 20, 108, 17, Color{170, 186, 214, 255});

        const std::string hud = HudLine(mode, rightHanded, amplitude, waveNumber, omega, paused);
        DrawText(hud.c_str(), 20, 138, 20, Color{130, 225, 255, 255});

        DrawRectangleRounded({1010.0f, 20.0f, 322.0f, 118.0f}, 0.08f, 14, Color{10, 18, 31, 205});
        DrawRectangleRoundedLinesEx({1010.0f, 20.0f, 322.0f, 118.0f}, 0.08f, 14, 2.0f, Color{49, 79, 113, 255});
        DrawText("cyan helix", 1030, 38, 17, Color{110, 225, 255, 255});
        DrawText("electric field E", 1030, 60, 16, Color{178, 193, 216, 255});
        DrawText("orange helix", 1030, 84, 17, Color{255, 175, 115, 255});
        DrawText("magnetic field B", 1030, 106, 16, Color{178, 193, 216, 255});

        DrawText("green arrows below axis: Poynting vector S = E x B", 20, kScreenHeight - 44, 17, Color{170, 255, 180, 255});
        DrawFPS(20, 166);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
