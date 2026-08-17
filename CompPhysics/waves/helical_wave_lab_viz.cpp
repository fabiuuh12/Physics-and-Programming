#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1360;
constexpr int kScreenHeight = 860;
constexpr float kTunnelLength = 15.0f;
constexpr float kXMin = -7.5f;
constexpr float kXMax = 7.5f;

enum class WaveMode {
    kPolarizedLight,
    kInterference,
    kStandingHelix,
    kPlasmaTwist
};

struct Tracer {
    float x;
    float angle;
    float radius;
    float drift;
    Color color;
};

float ClampFloat(float value, float lo, float hi) {
    return std::max(lo, std::min(value, hi));
}

Color WithAlpha(Color c, unsigned char alpha) {
    c.a = alpha;
    return c;
}

const char* ModeName(WaveMode mode) {
    switch (mode) {
        case WaveMode::kPolarizedLight: return "circular polarization";
        case WaveMode::kInterference: return "counter-rotating interference";
        case WaveMode::kStandingHelix: return "standing helical wave";
        case WaveMode::kPlasmaTwist: return "plasma torsion wave";
    }
    return "unknown";
}

void UpdateOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = ClampFloat(*pitch, -1.25f, 1.25f);
    }

    *distance -= GetMouseWheelMove() * 0.75f;
    *distance = ClampFloat(*distance, 7.0f, 32.0f);

    const float cp = std::cos(*pitch);
    const Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

Vector3 HelixOffset(float x,
                    float t,
                    float amplitude,
                    float waveNumber,
                    float omega,
                    float phaseOffset,
                    float ellipticity,
                    bool rightHanded) {
    const float handed = rightHanded ? 1.0f : -1.0f;
    const float phase = waveNumber * x - omega * t + phaseOffset;
    return {0.0f, amplitude * std::cos(phase), handed * amplitude * ellipticity * std::sin(phase)};
}

Vector3 ModeDisplacement(WaveMode mode,
                         float x,
                         float t,
                         float amplitude,
                         float waveNumber,
                         float omega,
                         float phaseOffset,
                         float ellipticity,
                         bool rightHanded) {
    const Vector3 a = HelixOffset(x, t, amplitude, waveNumber, omega, phaseOffset, ellipticity, rightHanded);

    if (mode == WaveMode::kPolarizedLight) {
        return a;
    }

    if (mode == WaveMode::kInterference) {
        const Vector3 b = HelixOffset(x, t, amplitude * 0.78f, -waveNumber, omega * 1.06f, -phaseOffset + 0.8f, ellipticity, !rightHanded);
        return Vector3Scale(Vector3Add(a, b), 0.68f);
    }

    if (mode == WaveMode::kStandingHelix) {
        const float handed = rightHanded ? 1.0f : -1.0f;
        const float envelope = std::cos(waveNumber * x + phaseOffset);
        const float timeSpin = std::cos(omega * t);
        const float timeTwist = std::sin(omega * t);
        return {
            0.0f,
            amplitude * envelope * timeSpin,
            handed * amplitude * ellipticity * envelope * timeTwist,
        };
    }

    const float handed = rightHanded ? 1.0f : -1.0f;
    const float phase = waveNumber * x - omega * t + phaseOffset;
    const float pulse = 0.62f + 0.38f * std::sin(0.72f * x + 1.4f * t);
    return {
        0.0f,
        amplitude * pulse * std::cos(phase),
        handed * amplitude * ellipticity * pulse * std::sin(phase + 0.38f * std::sin(t + x)),
    };
}

void DrawTube(float radius, float halfLength) {
    constexpr int rings = 26;
    constexpr int sides = 36;
    for (int i = 0; i <= rings; ++i) {
        const float x = -halfLength + 2.0f * halfLength * static_cast<float>(i) / static_cast<float>(rings);
        for (int j = 0; j < sides; ++j) {
            const float a0 = 2.0f * PI * static_cast<float>(j) / static_cast<float>(sides);
            const float a1 = 2.0f * PI * static_cast<float>(j + 1) / static_cast<float>(sides);
            const Vector3 p0 = {x, radius * std::cos(a0), radius * std::sin(a0)};
            const Vector3 p1 = {x, radius * std::cos(a1), radius * std::sin(a1)};
            DrawLine3D(p0, p1, Color{55, 78, 112, 70});
        }
    }

    for (int j = 0; j < 12; ++j) {
        const float a = 2.0f * PI * static_cast<float>(j) / 12.0f;
        const Vector3 p0 = {-halfLength, radius * std::cos(a), radius * std::sin(a)};
        const Vector3 p1 = {halfLength, radius * std::cos(a), radius * std::sin(a)};
        DrawLine3D(p0, p1, Color{64, 95, 138, 80});
    }
}

void DrawArrow3D(Vector3 from, Vector3 to, float radius, Color color) {
    const Vector3 d = Vector3Subtract(to, from);
    const float len = Vector3Length(d);
    if (len < 1e-4f) {
        return;
    }

    const Vector3 dir = Vector3Scale(d, 1.0f / len);
    const Vector3 base = Vector3Add(to, Vector3Scale(dir, -std::min(0.22f, len * 0.36f)));
    DrawCylinderEx(from, base, radius, radius, 8, color);
    DrawCylinderEx(base, to, radius * 2.1f, 0.0f, 8, color);
}

void DrawCurve(WaveMode mode,
               float t,
               float amplitude,
               float waveNumber,
               float omega,
               float phaseOffset,
               float ellipticity,
               bool rightHanded,
               Color color,
               float radius,
               bool glow) {
    constexpr int segments = 320;
    auto pointAt = [&](float x) {
        return Vector3Add({x, 0.0f, 0.0f},
                          ModeDisplacement(mode, x, t, amplitude, waveNumber, omega, phaseOffset, ellipticity, rightHanded));
    };

    Vector3 prev = pointAt(kXMin);
    for (int i = 1; i <= segments; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(segments);
        const float x = kXMin + (kXMax - kXMin) * u;
        const Vector3 cur = pointAt(x);
        if (glow) {
            DrawCylinderEx(prev, cur, radius * 2.0f, radius * 2.0f, 8, WithAlpha(color, 30));
        }
        DrawCylinderEx(prev, cur, radius, radius, 8, color);
        prev = cur;
    }
}

void DrawEnergyRibbon(WaveMode mode,
                      float t,
                      float amplitude,
                      float waveNumber,
                      float omega,
                      float phaseOffset,
                      float ellipticity,
                      bool rightHanded) {
    constexpr int samples = 24;
    Vector3 prev{};
    bool hasPrev = false;
    for (int i = 0; i < samples; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(samples - 1);
        const float x = kXMin + (kXMax - kXMin) * u;
        const Vector3 disp = ModeDisplacement(mode, x, t, amplitude, waveNumber, omega, phaseOffset, ellipticity, rightHanded);
        const float localEnergy = std::min(1.0f, Vector3Length(disp) / std::max(0.001f, amplitude));
        const Vector3 p = {x, -2.05f + localEnergy * 0.55f, 0.0f};
        const Color c = Color{120, static_cast<unsigned char>(165 + localEnergy * 80.0f), 255, 210};
        DrawSphere(p, 0.055f + localEnergy * 0.06f, c);
        if (hasPrev) {
            DrawCylinderEx(prev, p, 0.025f, 0.025f, 8, WithAlpha(c, 150));
        }
        prev = p;
        hasPrev = true;
    }
}

std::vector<Tracer> MakeTracers() {
    std::vector<Tracer> tracers;
    tracers.reserve(96);
    for (int i = 0; i < 96; ++i) {
        const float u = static_cast<float>(i) / 95.0f;
        const float ring = static_cast<float>(i % 16) / 16.0f;
        const unsigned char blue = static_cast<unsigned char>(170 + (i % 5) * 15);
        tracers.push_back({
            kXMin + u * kTunnelLength,
            2.0f * PI * ring,
            0.45f + 0.95f * static_cast<float>((i * 7) % 13) / 12.0f,
            0.22f + 0.18f * static_cast<float>((i * 5) % 11) / 10.0f,
            Color{110, static_cast<unsigned char>(185 + (i % 4) * 15), blue, 210},
        });
    }
    return tracers;
}

void DrawTracers(const std::vector<Tracer>& tracers,
                 WaveMode mode,
                 float t,
                 float amplitude,
                 float waveNumber,
                 float omega,
                 float phaseOffset,
                 float ellipticity,
                 bool rightHanded) {
    for (const Tracer& tracer : tracers) {
        float x = tracer.x + tracer.drift * t;
        while (x > kXMax) {
            x -= kTunnelLength;
        }
        const Vector3 wave = ModeDisplacement(mode, x, t, amplitude * 0.72f, waveNumber, omega, phaseOffset, ellipticity, rightHanded);
        const float swirl = tracer.angle + 0.8f * std::sin(0.9f * x + t) + (rightHanded ? 1.0f : -1.0f) * t * 0.35f;
        const Vector3 shell = {0.0f, tracer.radius * std::cos(swirl), tracer.radius * std::sin(swirl)};
        const Vector3 p = Vector3Add({x, 0.0f, 0.0f}, Vector3Add(Vector3Scale(wave, 0.35f), shell));
        DrawSphere(p, 0.035f, tracer.color);
    }
}

std::string HudLine(WaveMode mode,
                    float amplitude,
                    float waveNumber,
                    float omega,
                    float phaseOffset,
                    float ellipticity,
                    bool rightHanded,
                    bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "mode=" << ModeName(mode)
       << "  handedness=" << (rightHanded ? "right" : "left")
       << "  A=" << amplitude
       << "  k=" << waveNumber
       << "  omega=" << omega
       << "  phase=" << phaseOffset
       << "  ellipse=" << ellipticity;
    if (paused) {
        os << "  [PAUSED]";
    }
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Helical Wave Laboratory 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {10.0f, 6.0f, 11.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f;
    float camPitch = 0.34f;
    float camDistance = 16.0f;

    WaveMode mode = WaveMode::kInterference;
    bool rightHanded = true;
    bool paused = false;
    bool showFieldArrows = true;
    bool showParticles = true;

    float t = 0.0f;
    float amplitude = 1.0f;
    float waveNumber = 1.35f;
    float omega = 2.0f;
    float phaseOffset = 0.0f;
    float ellipticity = 0.78f;

    const std::vector<Tracer> tracers = MakeTracers();

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ONE)) mode = WaveMode::kPolarizedLight;
        if (IsKeyPressed(KEY_TWO)) mode = WaveMode::kInterference;
        if (IsKeyPressed(KEY_THREE)) mode = WaveMode::kStandingHelix;
        if (IsKeyPressed(KEY_FOUR)) mode = WaveMode::kPlasmaTwist;
        if (IsKeyPressed(KEY_Q) || IsKeyPressed(KEY_E)) rightHanded = !rightHanded;
        if (IsKeyPressed(KEY_P) || IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_F)) showFieldArrows = !showFieldArrows;
        if (IsKeyPressed(KEY_T)) showParticles = !showParticles;

        if (IsKeyPressed(KEY_R)) {
            mode = WaveMode::kInterference;
            rightHanded = true;
            paused = false;
            showFieldArrows = true;
            showParticles = true;
            t = 0.0f;
            amplitude = 1.0f;
            waveNumber = 1.35f;
            omega = 2.0f;
            phaseOffset = 0.0f;
            ellipticity = 0.78f;
        }

        if (IsKeyDown(KEY_W)) amplitude = std::min(1.85f, amplitude + 0.9f * GetFrameTime());
        if (IsKeyDown(KEY_S)) amplitude = std::max(0.25f, amplitude - 0.9f * GetFrameTime());
        if (IsKeyDown(KEY_D)) waveNumber = std::min(3.1f, waveNumber + 0.9f * GetFrameTime());
        if (IsKeyDown(KEY_A)) waveNumber = std::max(0.45f, waveNumber - 0.9f * GetFrameTime());
        if (IsKeyDown(KEY_X)) phaseOffset += 1.8f * GetFrameTime();
        if (IsKeyDown(KEY_Z)) phaseOffset -= 1.8f * GetFrameTime();
        if (IsKeyDown(KEY_EQUAL) || IsKeyDown(KEY_KP_ADD)) omega = std::min(5.5f, omega + 1.4f * GetFrameTime());
        if (IsKeyDown(KEY_MINUS) || IsKeyDown(KEY_KP_SUBTRACT)) omega = std::max(0.25f, omega - 1.4f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT_BRACKET)) ellipticity = std::min(1.0f, ellipticity + 0.55f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT_BRACKET)) ellipticity = std::max(0.08f, ellipticity - 0.55f * GetFrameTime());

        UpdateOrbitCamera(&camera, &camYaw, &camPitch, &camDistance);
        if (!paused) {
            t += GetFrameTime();
        }

        BeginDrawing();
        ClearBackground(Color{5, 7, 14, 255});

        DrawRectangleGradientV(0, 0, kScreenWidth, 170, Color{9, 16, 29, 230}, Color{9, 16, 29, 30});
        DrawRectangleGradientV(0, kScreenHeight - 128, kScreenWidth, 128, Color{6, 10, 18, 20}, Color{6, 10, 18, 230});

        BeginMode3D(camera);

        DrawPlane({0.0f, -2.45f, 0.0f}, {18.0f, 8.0f}, Color{9, 13, 23, 255});
        DrawTube(2.1f, kTunnelLength * 0.5f);
        DrawLine3D({kXMin - 0.3f, 0.0f, 0.0f}, {kXMax + 0.3f, 0.0f, 0.0f}, Color{185, 210, 245, 150});

        DrawCurve(mode, t, amplitude, waveNumber, omega, phaseOffset, ellipticity, rightHanded, Color{90, 225, 255, 255}, 0.045f, true);

        if (mode == WaveMode::kInterference) {
            DrawCurve(WaveMode::kPolarizedLight, t, amplitude * 0.78f, -waveNumber, omega * 1.06f, -phaseOffset + 0.8f, ellipticity, !rightHanded, Color{255, 125, 205, 230}, 0.033f, true);
        } else if (mode == WaveMode::kStandingHelix) {
            for (int i = -4; i <= 4; ++i) {
                const float nodeX = (static_cast<float>(i) + 0.5f) * PI / std::max(0.01f, waveNumber);
                if (nodeX > kXMin && nodeX < kXMax) {
                    DrawCylinderEx({nodeX, -1.35f, 0.0f}, {nodeX, 1.35f, 0.0f}, 0.018f, 0.018f, 8, Color{255, 235, 120, 170});
                }
            }
        }

        if (showParticles) {
            DrawTracers(tracers, mode, t, amplitude, waveNumber, omega, phaseOffset, ellipticity, rightHanded);
        }

        if (showFieldArrows) {
            constexpr int samples = 15;
            for (int i = 0; i < samples; ++i) {
                const float u = static_cast<float>(i) / static_cast<float>(samples - 1);
                const float x = kXMin + (kXMax - kXMin) * u;
                const Vector3 base = {x, 0.0f, 0.0f};
                const Vector3 disp = ModeDisplacement(mode, x, t, amplitude, waveNumber, omega, phaseOffset, ellipticity, rightHanded);
                const Vector3 tangent = Vector3Normalize(Vector3Subtract(
                    ModeDisplacement(mode, x + 0.05f, t, amplitude, waveNumber, omega, phaseOffset, ellipticity, rightHanded),
                    ModeDisplacement(mode, x - 0.05f, t, amplitude, waveNumber, omega, phaseOffset, ellipticity, rightHanded)));
                DrawArrow3D(base, Vector3Add(base, disp), 0.018f, Color{120, 235, 255, 230});
                DrawArrow3D(Vector3Add(base, {0.0f, -1.55f, -1.25f}),
                            Vector3Add(Vector3Add(base, {0.0f, -1.55f, -1.25f}), Vector3Scale(Vector3Add({1.0f, 0.0f, 0.0f}, tangent), 0.6f)),
                            0.015f,
                            Color{145, 255, 165, 220});
            }
        }

        DrawEnergyRibbon(mode, t, amplitude, waveNumber, omega, phaseOffset, ellipticity, rightHanded);

        EndMode3D();

        DrawText("Helical Wave Laboratory", 20, 18, 32, Color{235, 241, 252, 255});
        DrawText("A 3D lab for circular polarization, counter-rotating helices, standing waves, and plasma-like torsion.", 20, 56, 18, Color{172, 190, 218, 255});
        DrawText("Mouse drag: orbit | wheel: zoom | 1 light | 2 interference | 3 standing | 4 plasma | Q/E handedness", 20, 86, 17, Color{172, 190, 218, 255});
        DrawText("W/S amplitude | A/D wave number | Z/X phase | +/- speed | [ ] ellipticity | F arrows | T particles | Space pause | R reset", 20, 112, 17, Color{172, 190, 218, 255});

        const std::string hud = HudLine(mode, amplitude, waveNumber, omega, phaseOffset, ellipticity, rightHanded, paused);
        DrawText(hud.c_str(), 20, 144, 19, Color{125, 230, 255, 255});

        DrawRectangleRounded({1004.0f, 22.0f, 332.0f, 136.0f}, 0.08f, 14, Color{9, 17, 31, 210});
        DrawRectangleRoundedLinesEx({1004.0f, 22.0f, 332.0f, 136.0f}, 0.08f, 14, 2.0f, Color{52, 83, 124, 255});
        DrawText("cyan: main helical displacement", 1024, 42, 16, Color{120, 235, 255, 255});
        DrawText("pink: opposing helix in mode 2", 1024, 66, 16, Color{255, 145, 210, 255});
        DrawText("green: local propagation/twist", 1024, 90, 16, Color{145, 255, 165, 255});
        DrawText("dots: tracer particles in the tunnel", 1024, 114, 16, Color{185, 205, 230, 255});

        DrawText("Best capture modes: 2 for interference braids, 3 for nodes, 4 for animated plasma torsion.", 20, kScreenHeight - 42, 17, Color{170, 205, 245, 255});
        DrawFPS(20, 174);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
