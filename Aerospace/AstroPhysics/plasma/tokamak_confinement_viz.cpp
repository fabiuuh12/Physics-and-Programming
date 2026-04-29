#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1600;
constexpr int kScreenHeight = 960;
constexpr float kMajorRadius = 5.65f;
constexpr float kMinorRadius = 1.42f;
constexpr int kPlasmaParticleCount = 1800;
constexpr int kSparkCount = 260;
constexpr int kFieldLineCount = 26;
constexpr int kTorusSegments = 132;
constexpr int kPoloidalSegments = 44;

struct OrbitCameraState {
    float yaw = 0.82f;
    float pitch = 0.39f;
    float distance = 19.5f;
};

struct PlasmaParticle {
    float theta = 0.0f;
    float phi = 0.0f;
    float thetaSpeed = 0.0f;
    float phiSpeed = 0.0f;
    float radius = 0.0f;
    float heat = 0.0f;
    float size = 0.0f;
    float phase = 0.0f;
};

struct Spark {
    float theta = 0.0f;
    float phi = 0.0f;
    float speed = 0.0f;
    float life = 0.0f;
    float phase = 0.0f;
};

struct Star {
    Vector3 pos{};
    float radius = 0.0f;
    float alpha = 0.0f;
};

float RandRange(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

float SmoothStep(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

Color LerpColor(Color a, Color b, float t) {
    const float u = std::clamp(t, 0.0f, 1.0f);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * u),
        static_cast<unsigned char>(a.g + (b.g - a.g) * u),
        static_cast<unsigned char>(a.b + (b.b - a.b) * u),
        static_cast<unsigned char>(a.a + (b.a - a.a) * u),
    };
}

Color PlasmaColor(float heat, float alpha = 1.0f) {
    Color c = LerpColor(Color{80, 178, 255, 255}, Color{255, 74, 180, 255}, heat);
    c = LerpColor(c, Color{255, 224, 98, 255}, std::max(0.0f, heat - 0.72f) / 0.28f);
    c.a = static_cast<unsigned char>(255.0f * std::clamp(alpha, 0.0f, 1.0f));
    return c;
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, OrbitCameraState* orbit) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        orbit->yaw -= delta.x * 0.0036f;
        orbit->pitch += delta.y * 0.0036f;
        orbit->pitch = std::clamp(orbit->pitch, -1.25f, 1.26f);
    }

    orbit->distance -= GetMouseWheelMove() * 0.9f;
    orbit->distance = std::clamp(orbit->distance, 8.5f, 39.0f);

    const float cp = std::cos(orbit->pitch);
    camera->target = {0.0f, 0.15f, 0.0f};
    camera->position = Vector3Add(camera->target, {
        orbit->distance * cp * std::cos(orbit->yaw),
        orbit->distance * std::sin(orbit->pitch),
        orbit->distance * cp * std::sin(orbit->yaw),
    });
}

Vector3 TokamakPoint(float theta, float phi, float minorRadius, float verticalStretch = 1.0f) {
    const float cphi = std::cos(phi);
    const float sphi = std::sin(phi);
    const float tube = kMajorRadius + minorRadius * cphi;
    return {tube * std::cos(theta), minorRadius * verticalStretch * sphi, tube * std::sin(theta)};
}

Vector3 RingPoint(float theta, float radius, float y = 0.0f) {
    return {radius * std::cos(theta), y, radius * std::sin(theta)};
}

std::vector<PlasmaParticle> MakePlasmaParticles() {
    std::mt19937 rng(8172);
    std::vector<PlasmaParticle> particles;
    particles.reserve(kPlasmaParticleCount);

    for (int i = 0; i < kPlasmaParticleCount; ++i) {
        const float heat = RandRange(rng, 0.0f, 1.0f);
        const float radius = kMinorRadius * RandRange(rng, 0.12f, 0.92f);
        particles.push_back({
            RandRange(rng, 0.0f, 2.0f * PI),
            RandRange(rng, 0.0f, 2.0f * PI),
            RandRange(rng, 0.55f, 1.45f),
            RandRange(rng, 2.5f, 7.8f) * (0.72f + heat * 0.72f),
            radius,
            heat,
            RandRange(rng, 0.018f, 0.066f),
            RandRange(rng, 0.0f, 2.0f * PI),
        });
    }

    return particles;
}

std::vector<Spark> MakeSparks() {
    std::mt19937 rng(991);
    std::vector<Spark> sparks;
    sparks.reserve(kSparkCount);
    for (int i = 0; i < kSparkCount; ++i) {
        sparks.push_back({RandRange(rng, 0.0f, 2.0f * PI), RandRange(rng, 0.0f, 2.0f * PI), RandRange(rng, 0.6f, 2.6f),
                          RandRange(rng, 0.2f, 1.0f), RandRange(rng, 0.0f, 2.0f * PI)});
    }
    return sparks;
}

std::vector<Star> MakeBackdrop() {
    std::mt19937 rng(4401);
    std::vector<Star> stars;
    stars.reserve(240);
    for (int i = 0; i < 240; ++i) {
        const float theta = RandRange(rng, 0.0f, 2.0f * PI);
        const float elev = RandRange(rng, -0.45f, 0.65f);
        const float r = RandRange(rng, 28.0f, 48.0f);
        stars.push_back({{r * std::cos(elev) * std::cos(theta), r * std::sin(elev), r * std::cos(elev) * std::sin(theta)},
                         RandRange(rng, 0.018f, 0.075f),
                         RandRange(rng, 0.12f, 0.76f)});
    }
    return stars;
}

void DrawPolylineLoop(const std::vector<Vector3>& points, Color color) {
    for (size_t i = 1; i < points.size(); ++i) DrawLine3D(points[i - 1], points[i], color);
    if (points.size() > 2) DrawLine3D(points.back(), points.front(), color);
}

void DrawDShapedToroidalCoil(float theta, float time, float power) {
    const Vector3 radial = {std::cos(theta), 0.0f, std::sin(theta)};
    const Vector3 tangent = {-std::sin(theta), 0.0f, std::cos(theta)};
    const Vector3 center = Vector3Scale(radial, kMajorRadius);
    std::vector<Vector3> loop;
    loop.reserve(72);

    for (int i = 0; i < 72; ++i) {
        const float a = 2.0f * PI * static_cast<float>(i) / 72.0f;
        const float side = std::cos(a);
        const float y = 2.85f * std::sin(a);
        const float outward = side > 0.0f ? 2.25f * side : 1.42f * side;
        loop.push_back(Vector3Add(center, Vector3Add(Vector3Scale(radial, outward), Vector3Scale(tangent, 0.28f * std::sin(a)))));
        loop.back().y = y;
    }

    DrawPolylineLoop(loop, Color{74, 94, 128, 255});
    for (size_t i = 0; i < loop.size(); i += 2) {
        DrawSphere(loop[i], 0.055f, Color{110, 135, 176, 210});
    }

    const float pulse = 0.45f + 0.55f * SmoothStep(0.5f + 0.5f * std::sin(time * 2.0f + theta * 3.0f));
    Color c = LerpColor(Color{74, 164, 255, 90}, Color{130, 236, 255, 220}, pulse * power);
    for (int i = 0; i < 72; i += 6) {
        DrawLine3D(loop[i], loop[(i + 1) % loop.size()], c);
    }
}

void DrawPoloidalCoils(float time, float current) {
    const float radii[] = {3.6f, 4.65f, 7.45f, 8.85f};
    const float ys[] = {-3.15f, 3.15f, -2.28f, 2.28f};
    for (int c = 0; c < 4; ++c) {
        for (int i = 0; i < 96; ++i) {
            const float a0 = 2.0f * PI * i / 96.0f;
            const float a1 = 2.0f * PI * (i + 1) / 96.0f;
            Color color = Color{204, 112, 255, static_cast<unsigned char>(80 + 90 * current)};
            DrawLine3D(RingPoint(a0, radii[c], ys[c]), RingPoint(a1, radii[c], ys[c]), color);
        }
        const float p = std::fmod(time * (0.28f + current) + c * 0.21f, 1.0f);
        DrawSphere(RingPoint(2.0f * PI * p, radii[c], ys[c]), 0.11f, Color{255, 194, 255, 230});
    }
}

void DrawVacuumVessel(float time, bool cutaway) {
    const Color steel = cutaway ? Color{102, 118, 136, 72} : Color{102, 118, 136, 138};
    const Color grid = Color{178, 198, 220, static_cast<unsigned char>(cutaway ? 90 : 132)};

    for (int band = 0; band < 9; ++band) {
        const float phi = 2.0f * PI * band / 9.0f;
        for (int i = 0; i < kTorusSegments; ++i) {
            const float a0 = 2.0f * PI * i / kTorusSegments;
            const float a1 = 2.0f * PI * (i + 1) / kTorusSegments;
            DrawLine3D(TokamakPoint(a0, phi, kMinorRadius + 0.36f, 1.08f), TokamakPoint(a1, phi, kMinorRadius + 0.36f, 1.08f), grid);
        }
    }

    for (int ring = 0; ring < 24; ++ring) {
        if (cutaway && ring > 1 && ring < 9) continue;
        const float theta = 2.0f * PI * ring / 24.0f;
        std::vector<Vector3> loop;
        loop.reserve(kPoloidalSegments);
        for (int p = 0; p < kPoloidalSegments; ++p) {
            const float phi = 2.0f * PI * p / kPoloidalSegments;
            loop.push_back(TokamakPoint(theta, phi, kMinorRadius + 0.36f, 1.08f));
        }
        DrawPolylineLoop(loop, steel);
    }

    for (int port = 0; port < 12; ++port) {
        const float a = 2.0f * PI * port / 12.0f;
        const float r0 = kMajorRadius + kMinorRadius + 0.42f;
        const float r1 = r0 + 1.55f;
        DrawCylinderEx(RingPoint(a, r0, 0.0f), RingPoint(a, r1, 0.0f), 0.18f, 0.18f, 14, Color{78, 92, 110, 245});
        DrawCylinderEx(RingPoint(a, r1, 0.0f), RingPoint(a, r1 + 0.28f, 0.0f), 0.27f, 0.27f, 18, Color{132, 150, 172, 220});
    }

    for (int i = 0; i < 36; ++i) {
        const float a = 2.0f * PI * i / 36.0f;
        const float flicker = 0.5f + 0.5f * std::sin(time * 3.0f + i * 1.7f);
        DrawSphere(RingPoint(a, kMajorRadius + 2.25f, -2.0f), 0.035f + 0.022f * flicker, Color{90, 190, 255, 110});
    }
}

void DrawPlasmaSurfaces(float time, float power, bool magneticLines) {
    BeginBlendMode(BLEND_ADDITIVE);

    for (int shell = 0; shell < 5; ++shell) {
        const float r = kMinorRadius * (0.22f + shell * 0.135f);
        const float alpha = (0.18f - shell * 0.025f) * power;
        const Color shellColor = PlasmaColor(0.38f + 0.13f * shell, alpha);
        for (int band = 0; band < 8; ++band) {
            const float phi = 2.0f * PI * band / 8.0f + time * (0.07f + shell * 0.012f);
            for (int i = 0; i < kTorusSegments; ++i) {
                const float a0 = 2.0f * PI * i / kTorusSegments;
                const float a1 = 2.0f * PI * (i + 1) / kTorusSegments;
                DrawLine3D(TokamakPoint(a0, phi, r), TokamakPoint(a1, phi, r), shellColor);
            }
        }
    }

    if (magneticLines) {
        for (int line = 0; line < kFieldLineCount; ++line) {
            const float base = 2.0f * PI * line / kFieldLineCount;
            Vector3 prev{};
            bool hasPrev = false;
            const float q = 3.7f + 0.25f * std::sin(time * 0.4f + line);
            for (int i = 0; i <= 300; ++i) {
                const float u = static_cast<float>(i) / 300.0f;
                const float theta = 2.0f * PI * (1.75f * u) + time * 0.18f;
                const float phi = base + theta * q + 0.16f * std::sin(time + theta * 2.0f);
                const Vector3 p = TokamakPoint(theta, phi, kMinorRadius * 0.76f);
                if (hasPrev) DrawLine3D(prev, p, Color{96, 226, 255, 82});
                prev = p;
                hasPrev = true;
            }
        }
    }

    EndBlendMode();
}

void DrawParticles(const std::vector<PlasmaParticle>& particles, const std::vector<Spark>& sparks, float time, float power) {
    BeginBlendMode(BLEND_ADDITIVE);

    for (const PlasmaParticle& particle : particles) {
        const float theta = particle.theta + time * particle.thetaSpeed;
        const float phi = particle.phi + time * particle.phiSpeed + 0.2f * std::sin(time * 1.3f + particle.phase);
        const float breathing = 1.0f + 0.08f * std::sin(time * 5.0f + particle.phase);
        const Vector3 p = TokamakPoint(theta, phi, particle.radius * breathing);
        const Color c = PlasmaColor(std::clamp(particle.heat * 0.82f + power * 0.28f, 0.0f, 1.0f), 0.72f);
        DrawSphere(p, particle.size * (0.8f + power), c);
        if (particle.heat > 0.72f) DrawSphere(p, particle.size * 3.8f, PlasmaColor(1.0f, 0.09f));
    }

    for (const Spark& spark : sparks) {
        const float life = 0.5f + 0.5f * std::sin(time * spark.speed + spark.phase);
        if (life < 0.16f) continue;
        const float theta = spark.theta + time * (1.15f + spark.speed);
        const float phi = spark.phi + theta * 4.6f + 0.35f * std::sin(time * 2.0f + spark.phase);
        const Vector3 p = TokamakPoint(theta, phi, kMinorRadius * (0.86f + 0.12f * life));
        DrawSphere(p, 0.035f + 0.08f * life, Color{255, 236, 132, static_cast<unsigned char>(185 * life)});
    }

    EndBlendMode();
}

void DrawHeatingSystems(float time, float power) {
    const float injectAngles[] = {0.12f * PI, 0.86f * PI, 1.42f * PI};
    for (int i = 0; i < 3; ++i) {
        const float a = injectAngles[i];
        const Vector3 outer = RingPoint(a, kMajorRadius + 5.2f, i == 1 ? 0.7f : -0.45f);
        const Vector3 inner = RingPoint(a + 0.22f, kMajorRadius + 1.15f, 0.08f);
        DrawCylinderEx(outer, inner, 0.18f, 0.11f, 18, Color{90, 104, 124, 255});
        DrawCylinderEx(Vector3Add(outer, {0.0f, -0.28f, 0.0f}), Vector3Add(outer, {0.0f, 0.28f, 0.0f}), 0.42f, 0.42f, 24,
                       Color{126, 144, 168, 255});
        BeginBlendMode(BLEND_ADDITIVE);
        const float pulse = std::fmod(time * (0.7f + 0.4f * i) + i * 0.23f, 1.0f);
        const Vector3 beamStart = Vector3Lerp(outer, inner, pulse);
        const Vector3 beamEnd = Vector3Lerp(outer, inner, std::min(1.0f, pulse + 0.25f));
        DrawCylinderEx(beamStart, beamEnd, 0.04f, 0.02f, 10, Color{255, 120, 70, static_cast<unsigned char>(190 * power)});
        EndBlendMode();
    }

    for (int i = 0; i < 4; ++i) {
        const float a = 2.0f * PI * (i + 0.35f) / 4.0f;
        const Vector3 p = RingPoint(a, kMajorRadius + 2.25f, 1.1f);
        DrawCube(p, 0.45f, 0.32f, 0.72f, Color{120, 136, 160, 255});
        BeginBlendMode(BLEND_ADDITIVE);
        for (int k = 0; k < 5; ++k) {
            const float u = std::fmod(time * 0.8f + k * 0.18f + i * 0.11f, 1.0f);
            DrawSphere(Vector3Lerp(p, TokamakPoint(a - 0.08f, 0.2f, kMinorRadius * 0.72f), u), 0.035f, Color{122, 238, 255, 170});
        }
        EndBlendMode();
    }
}

void DrawDivertorAndHeatTiles(float time, float exhaust) {
    for (int i = 0; i < 96; ++i) {
        const float a = 2.0f * PI * i / 96.0f;
        const float heat = SmoothStep(0.5f + 0.5f * std::sin(a * 6.0f + time * 1.6f)) * exhaust;
        const Color tile = LerpColor(Color{58, 67, 78, 255}, Color{255, 110, 62, 255}, heat);
        DrawCube(RingPoint(a, kMajorRadius - 0.75f, -2.05f), 0.22f, 0.09f, 0.32f, tile);
        DrawCube(RingPoint(a, kMajorRadius + 0.95f, -2.05f), 0.22f, 0.09f, 0.32f, tile);
    }

    BeginBlendMode(BLEND_ADDITIVE);
    for (int i = 0; i < 36; ++i) {
        const float a = 2.0f * PI * i / 36.0f + time * 0.08f;
        const float flicker = SmoothStep(0.5f + 0.5f * std::sin(time * 4.0f + i));
        DrawSphere(RingPoint(a, kMajorRadius, -1.72f), 0.055f + 0.08f * flicker, Color{255, 156, 72, static_cast<unsigned char>(130 * exhaust)});
    }
    EndBlendMode();
}

void DrawCryostatAndFloor(float time, const std::vector<Star>& stars) {
    for (const Star& star : stars) {
        DrawSphere(star.pos, star.radius, Color{130, 180, 230, static_cast<unsigned char>(255.0f * star.alpha)});
    }

    DrawGrid(34, 1.0f);
    DrawCylinder({0.0f, -3.05f, 0.0f}, 10.6f, 10.6f, 0.18f, 96, Color{34, 42, 50, 255});
    DrawCylinderWires({0.0f, -2.9f, 0.0f}, 10.7f, 10.7f, 0.08f, 96, Color{106, 130, 158, 90});

    for (int i = 0; i < 48; ++i) {
        const float a = 2.0f * PI * i / 48.0f;
        const float y = -2.75f + 0.08f * std::sin(time + i);
        DrawCube(RingPoint(a, 10.35f, y), 0.18f, 0.28f, 0.18f, Color{54, 66, 82, 255});
    }
}

void DrawDiagnostics(float time) {
    for (int i = 0; i < 8; ++i) {
        const float a = 2.0f * PI * (i + 0.08f) / 8.0f;
        const Vector3 head = RingPoint(a, kMajorRadius + 4.15f, 1.85f - (i % 2) * 1.15f);
        const Vector3 target = TokamakPoint(a - 0.06f, 0.0f, kMinorRadius * 0.38f);
        DrawCube(head, 0.36f, 0.36f, 0.62f, Color{88, 104, 128, 255});
        DrawCylinderEx(head, target, 0.018f, 0.012f, 8, Color{94, 214, 255, 82});
        if (std::fmod(time + i * 0.31f, 2.2f) < 0.08f) {
            BeginBlendMode(BLEND_ADDITIVE);
            DrawLine3D(head, target, Color{140, 236, 255, 210});
            EndBlendMode();
        }
    }
}

void DrawHud(float power, float q, float current, float density, bool paused, bool cutaway, bool magneticLines) {
    DrawRectangle(0, 0, GetScreenWidth(), 136, Color{5, 8, 14, 215});
    DrawText("Graphically Advanced Tokamak Fusion Reactor", 24, 18, 32, Color{240, 246, 255, 255});
    DrawText("Mouse orbit | wheel zoom | [ ] plasma power | , . field current | C cutaway | M field lines | P pause | R reset",
             24, 58, 18, Color{178, 195, 222, 255});

    std::ostringstream os;
    os << "D-T plasma  temp=" << std::fixed << std::setprecision(1) << (80.0f + 90.0f * power)
       << " MK   Q=" << q
       << "   toroidal field=" << (3.2f + 3.1f * current)
       << " T   density=" << density << "e20 m^-3";
    if (cutaway) os << "   [CUTAWAY]";
    if (magneticLines) os << "   [FIELD LINES]";
    if (paused) os << "   [PAUSED]";
    DrawText(os.str().c_str(), 24, 90, 20, Color{255, 218, 142, 255});

    const int x = GetScreenWidth() - 378;
    const int y = 24;
    const int w = 240;
    auto bar = [&](int row, const char* label, float value, Color color) {
        DrawText(label, x, y + row * 28 - 2, 16, Color{194, 210, 232, 255});
        DrawRectangle(x + 116, y + row * 28, w, 14, Color{26, 34, 48, 245});
        DrawRectangle(x + 116, y + row * 28, static_cast<int>(w * std::clamp(value, 0.0f, 1.0f)), 14, color);
        DrawRectangleLines(x + 116, y + row * 28, w, 14, Color{110, 136, 166, 170});
    };
    bar(0, "plasma power", power, Color{255, 94, 170, 255});
    bar(1, "field current", current, Color{112, 224, 255, 255});
    bar(2, "confinement", q / 12.0f, Color{255, 210, 94, 255});
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Tokamak Fusion Reactor 3D - C++ (raylib)");
    SetWindowMinSize(1100, 720);
    SetTargetFPS(120);

    Camera3D camera{};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 46.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    OrbitCameraState orbit{};
    std::vector<PlasmaParticle> particles = MakePlasmaParticles();
    std::vector<Spark> sparks = MakeSparks();
    std::vector<Star> stars = MakeBackdrop();

    float power = 0.78f;
    float fieldCurrent = 0.74f;
    float time = 0.0f;
    bool paused = false;
    bool cutaway = true;
    bool magneticLines = true;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_C)) cutaway = !cutaway;
        if (IsKeyPressed(KEY_M)) magneticLines = !magneticLines;
        if (IsKeyPressed(KEY_R)) {
            power = 0.78f;
            fieldCurrent = 0.74f;
            time = 0.0f;
        }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) power = std::max(0.18f, power - 0.08f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) power = std::min(1.0f, power + 0.08f);
        if (IsKeyPressed(KEY_COMMA)) fieldCurrent = std::max(0.22f, fieldCurrent - 0.06f);
        if (IsKeyPressed(KEY_PERIOD)) fieldCurrent = std::min(1.0f, fieldCurrent + 0.06f);

        UpdateOrbitCameraDragOnly(&camera, &orbit);
        if (!paused) time += std::min(GetFrameTime(), 1.0f / 45.0f);

        const float q = 4.2f + 6.4f * fieldCurrent + 1.2f * power;
        const float density = 0.7f + 1.4f * power;
        const float exhaust = std::clamp(0.35f + 0.75f * power - 0.2f * fieldCurrent, 0.0f, 1.0f);

        BeginDrawing();
        ClearBackground(Color{4, 7, 13, 255});

        BeginMode3D(camera);
        DrawCryostatAndFloor(time, stars);
        DrawVacuumVessel(time, cutaway);

        for (int i = 0; i < 18; ++i) {
            DrawDShapedToroidalCoil(2.0f * PI * i / 18.0f, time, fieldCurrent);
        }

        DrawPoloidalCoils(time, fieldCurrent);
        DrawDivertorAndHeatTiles(time, exhaust);
        DrawHeatingSystems(time, power);
        DrawDiagnostics(time);
        DrawPlasmaSurfaces(time, power, magneticLines);
        DrawParticles(particles, sparks, time, power);

        BeginBlendMode(BLEND_ADDITIVE);
        DrawSphere({0.0f, 0.0f, 0.0f}, 2.2f + 0.2f * std::sin(time * 2.0f), Color{90, 180, 255, static_cast<unsigned char>(20 + 25 * power)});
        DrawSphere({0.0f, 0.0f, 0.0f}, 1.15f, Color{255, 160, 80, static_cast<unsigned char>(16 + 20 * power)});
        EndBlendMode();

        EndMode3D();

        DrawHud(power, q, fieldCurrent, density, paused, cutaway, magneticLines);
        DrawFPS(GetScreenWidth() - 100, 18);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
