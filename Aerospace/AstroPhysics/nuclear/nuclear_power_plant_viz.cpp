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

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;
constexpr int kCoolantParticleCount = 180;
constexpr int kSteamParticleCount = 160;
constexpr int kBackgroundParticleCount = 180;

struct OrbitCameraState {
    float yaw = 0.78f;
    float pitch = 0.34f;
    float distance = 26.0f;
};

struct FlowParticle {
    Vector3 pos{};
    float t = 0.0f;
    float speed = 0.0f;
    float size = 0.0f;
    float lane = 0.0f;
};

struct SteamParticle {
    Vector3 pos{};
    float age = 0.0f;
    float life = 0.0f;
    float angle = 0.0f;
    float radius = 0.0f;
    float size = 0.0f;
};

struct BackgroundParticle {
    Vector3 pos{};
    float size = 0.0f;
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

void UpdateOrbitCameraDragOnly(Camera3D* camera, OrbitCameraState* orbit) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        orbit->yaw -= delta.x * 0.0038f;
        orbit->pitch += delta.y * 0.0038f;
        orbit->pitch = std::clamp(orbit->pitch, -1.18f, 1.18f);
    }

    orbit->distance -= GetMouseWheelMove() * 1.1f;
    orbit->distance = std::clamp(orbit->distance, 11.0f, 52.0f);

    const float cp = std::cos(orbit->pitch);
    camera->target = {0.0f, 1.9f, 0.0f};
    camera->position = Vector3Add(camera->target, {
        orbit->distance * cp * std::cos(orbit->yaw),
        orbit->distance * std::sin(orbit->pitch),
        orbit->distance * cp * std::sin(orbit->yaw),
    });
}

Vector3 Bezier(Vector3 a, Vector3 b, Vector3 c, Vector3 d, float t) {
    const float u = 1.0f - t;
    return Vector3Add(
        Vector3Add(Vector3Scale(a, u * u * u), Vector3Scale(b, 3.0f * u * u * t)),
        Vector3Add(Vector3Scale(c, 3.0f * u * t * t), Vector3Scale(d, t * t * t)));
}

Vector3 CoolantPath(float t, float lane) {
    t = std::fmod(t, 1.0f);
    if (t < 0.0f) t += 1.0f;

    const float side = lane < 0.5f ? -0.18f : 0.18f;
    if (t < 0.25f) {
        const float u = t / 0.25f;
        return Bezier({-1.2f, 1.3f, side}, {-1.8f, 3.2f, side}, {-3.5f, 3.9f, side}, {-5.9f, 3.25f, side}, u);
    }
    if (t < 0.5f) {
        const float u = (t - 0.25f) / 0.25f;
        return Bezier({-5.9f, 3.25f, side}, {-7.5f, 2.3f, side}, {-7.1f, 0.2f, side}, {-5.7f, 0.0f, side}, u);
    }
    if (t < 0.75f) {
        const float u = (t - 0.5f) / 0.25f;
        return Bezier({-5.7f, 0.0f, side}, {-3.8f, -0.45f, side}, {-1.8f, -0.2f, side}, {-1.05f, 0.95f, side}, u);
    }

    const float u = (t - 0.75f) / 0.25f;
    return Bezier({-1.05f, 0.95f, side}, {-0.35f, 1.7f, side}, {-0.55f, 2.4f, side}, {-1.2f, 1.3f, side}, u);
}

std::vector<FlowParticle> MakeCoolantParticles() {
    std::mt19937 rng(4207);
    std::vector<FlowParticle> particles;
    particles.reserve(kCoolantParticleCount);
    for (int i = 0; i < kCoolantParticleCount; ++i) {
        const float lane = static_cast<float>(i % 2);
        particles.push_back({CoolantPath(RandRange(rng, 0.0f, 1.0f), lane),
                             RandRange(rng, 0.0f, 1.0f),
                             RandRange(rng, 0.05f, 0.11f),
                             RandRange(rng, 0.035f, 0.075f),
                             lane});
    }
    return particles;
}

std::vector<SteamParticle> MakeSteamParticles() {
    std::mt19937 rng(9188);
    std::vector<SteamParticle> particles;
    particles.reserve(kSteamParticleCount);
    for (int i = 0; i < kSteamParticleCount; ++i) {
        const float age = RandRange(rng, 0.0f, 1.0f);
        particles.push_back({{}, age, RandRange(rng, 3.6f, 6.4f), RandRange(rng, 0.0f, 2.0f * PI), RandRange(rng, 0.2f, 0.9f),
                             RandRange(rng, 0.08f, 0.22f)});
    }
    return particles;
}

std::vector<BackgroundParticle> MakeBackgroundParticles() {
    std::mt19937 rng(1017);
    std::vector<BackgroundParticle> particles;
    particles.reserve(kBackgroundParticleCount);
    for (int i = 0; i < kBackgroundParticleCount; ++i) {
        particles.push_back({{RandRange(rng, -32.0f, 32.0f), RandRange(rng, 7.0f, 19.0f), RandRange(rng, -30.0f, -18.0f)},
                             RandRange(rng, 0.025f, 0.09f),
                             RandRange(rng, 0.12f, 0.55f)});
    }
    return particles;
}

void DrawPipe(Vector3 a, Vector3 b, float radius, Color color) {
    DrawCylinderEx(a, b, radius, radius, 18, color);
    DrawCylinderWires(a, radius, radius, Vector3Distance(a, b), 18, Color{145, 168, 188, 85});
}

void DrawCoolingTower(Vector3 base, float time, const std::vector<SteamParticle>& steam, int parity) {
    const Color concrete = {180, 186, 190, 245};
    for (int i = 0; i < 9; ++i) {
        const float y = i * 0.46f;
        const float u = static_cast<float>(i) / 8.0f;
        const float radius = 1.22f - 0.36f * std::sin(u * PI);
        DrawCylinder(Vector3Add(base, {0.0f, y + 0.2f, 0.0f}), radius, radius - 0.03f, 0.45f, 44, concrete);
    }
    DrawCylinderWires(Vector3Add(base, {0.0f, 4.45f, 0.0f}), 1.12f, 1.04f, 0.18f, 44, Color{235, 242, 247, 170});

    for (int i = parity; i < static_cast<int>(steam.size()); i += 2) {
        const SteamParticle& p = steam[i];
        const float u = p.age / p.life;
        const float swirl = p.angle + time * (0.35f + 0.18f * parity) + 1.6f * u;
        const float r = p.radius + 1.45f * u;
        const Vector3 pos = Vector3Add(base, {std::cos(swirl) * r, 4.25f + 4.6f * SmoothStep(u), std::sin(swirl) * r});
        Color c = {230, 238, 244, static_cast<unsigned char>(std::clamp(130.0f * (1.0f - u), 0.0f, 130.0f))};
        DrawSphere(pos, p.size * (1.0f + 1.8f * u), c);
    }
}

void DrawContainmentBuilding(float corePower, float rodDepth, bool cutaway) {
    DrawCylinder({0.0f, 1.1f, 0.0f}, 2.45f, 2.45f, 2.2f, 56, Color{178, 184, 192, 240});
    DrawSphere({0.0f, 2.23f, 0.0f}, 2.45f, Color{196, 202, 210, static_cast<unsigned char>(cutaway ? 80 : 220)});
    DrawCylinderWires({0.0f, 1.1f, 0.0f}, 2.48f, 2.48f, 2.2f, 56, Color{235, 240, 246, 135});
    DrawSphereWires({0.0f, 2.23f, 0.0f}, 2.47f, 24, 20, Color{235, 240, 246, 120});

    if (!cutaway) return;

    DrawCylinder({0.0f, 1.2f, 0.0f}, 0.72f, 0.72f, 1.9f, 36, Color{76, 90, 111, 230});
    DrawCylinderWires({0.0f, 1.2f, 0.0f}, 0.74f, 0.74f, 1.92f, 36, Color{165, 191, 220, 170});

    const float glow = 0.58f + 0.18f * std::sin(GetTime() * 8.0f);
    DrawSphere({0.0f, 1.28f, 0.0f}, 0.44f + 0.05f * corePower * glow, Color{255, 146, 70, 225});
    DrawSphere({0.0f, 1.28f, 0.0f}, 0.78f + 0.16f * corePower * glow, Color{255, 205, 80, 70});

    for (int i = -3; i <= 3; ++i) {
        const float x = i * 0.18f;
        const float lowered = 2.48f - 0.92f * rodDepth;
        DrawCube({x, lowered, -0.28f}, 0.055f, 1.4f, 0.055f, Color{43, 55, 72, 255});
        DrawCube({x, lowered, 0.28f}, 0.055f, 1.4f, 0.055f, Color{43, 55, 72, 255});
    }
}

void DrawTurbineHall(float turbineSpin, float demand) {
    DrawCube({5.15f, 0.65f, 0.0f}, 4.8f, 1.3f, 3.2f, Color{84, 103, 126, 255});
    DrawCubeWires({5.15f, 0.65f, 0.0f}, 4.8f, 1.3f, 3.2f, Color{181, 200, 216, 100});
    DrawCube({5.15f, 1.38f, 0.0f}, 4.95f, 0.18f, 3.35f, Color{118, 136, 154, 255});

    DrawCylinderEx({3.35f, 1.45f, 0.0f}, {6.95f, 1.45f, 0.0f}, 0.38f, 0.38f, 34, Color{178, 190, 202, 255});
    for (int i = 0; i < 7; ++i) {
        const float x = 3.72f + i * 0.46f;
        const float a = turbineSpin + i * 0.7f;
        const Vector3 center = {x, 1.45f, 0.0f};
        DrawCylinderEx(Vector3Add(center, {0.0f, std::sin(a) * 0.72f, std::cos(a) * 0.72f}),
                       Vector3Add(center, {0.0f, -std::sin(a) * 0.72f, -std::cos(a) * 0.72f}), 0.025f, 0.025f, 8,
                       Color{214, 225, 236, 255});
        DrawCylinderEx(Vector3Add(center, {0.0f, std::cos(a) * 0.72f, -std::sin(a) * 0.72f}),
                       Vector3Add(center, {0.0f, -std::cos(a) * 0.72f, std::sin(a) * 0.72f}), 0.025f, 0.025f, 8,
                       Color{214, 225, 236, 255});
    }

    DrawCylinderEx({7.2f, 1.45f, 0.0f}, {8.1f, 1.45f, 0.0f}, 0.58f, 0.58f, 36, Color{238, 179, 72, 255});
    DrawCylinderWires({8.2f, 1.45f, 0.0f}, 0.64f, 0.64f, 0.22f, 36, Color{255, 233, 128, 180});
    for (int i = 0; i < 5; ++i) {
        const float y = 2.35f + i * 0.35f;
        const float pulse = std::fmod(static_cast<float>(GetTime()) * (0.95f + demand * 1.8f) + i * 0.18f, 1.0f);
        DrawLine3D({8.35f + pulse * 3.6f, y, -0.92f}, {8.95f + pulse * 3.6f, y, -0.92f}, Color{255, 218, 84, 210});
    }
}

void DrawPlantBase() {
    DrawCube({0.0f, -0.08f, 0.0f}, 20.0f, 0.16f, 12.0f, Color{38, 48, 54, 255});
    DrawGrid(24, 1.0f);
    DrawCube({-0.2f, 0.03f, -3.95f}, 7.0f, 0.12f, 0.18f, Color{70, 82, 92, 255});
    DrawCube({6.4f, 0.03f, 2.65f}, 6.4f, 0.12f, 0.18f, Color{70, 82, 92, 255});
}

void DrawStatusBars(float corePower, float waterLevel, float demand, float rodDepth) {
    const int x = 24;
    const int y = 752;
    const int w = 258;
    const int h = 16;
    const Color bg = {32, 41, 54, 240};

    auto bar = [&](int row, const char* label, float v, Color fill) {
        DrawText(label, x, y + row * 32 - 2, 18, Color{205, 216, 230, 255});
        DrawRectangle(x + 128, y + row * 32, w, h, bg);
        DrawRectangle(x + 128, y + row * 32, static_cast<int>(w * std::clamp(v, 0.0f, 1.0f)), h, fill);
        DrawRectangleLines(x + 128, y + row * 32, w, h, Color{125, 145, 168, 180});
    };

    bar(0, "core power", corePower, Color{255, 145, 74, 255});
    bar(1, "water level", waterLevel, Color{79, 178, 255, 255});
    bar(2, "grid demand", demand, Color{255, 212, 74, 255});
    bar(3, "rod depth", rodDepth, Color{127, 150, 180, 255});
}

std::string Hud(float corePower, float demand, float rodDepth, float waterLevel, bool paused, bool cutaway) {
    std::ostringstream os;
    os << "thermal output=" << std::fixed << std::setprecision(0) << corePower * 100.0f << "%"
       << "  demand=" << demand * 100.0f << "%"
       << "  control rods inserted=" << rodDepth * 100.0f << "%"
       << "  secondary water=" << waterLevel * 100.0f << "%";
    if (cutaway) os << "  [CUTAWAY]";
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Nuclear Power Plant 3D Simulation - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    OrbitCameraState orbit;
    std::vector<FlowParticle> coolant = MakeCoolantParticles();
    std::vector<SteamParticle> steam = MakeSteamParticles();
    std::vector<BackgroundParticle> background = MakeBackgroundParticles();

    float time = 0.0f;
    float corePower = 0.72f;
    float targetDemand = 0.68f;
    float rodDepth = 0.32f;
    float waterLevel = 0.82f;
    float turbineSpin = 0.0f;
    bool paused = false;
    bool cutaway = true;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_C)) cutaway = !cutaway;
        if (IsKeyPressed(KEY_R)) {
            corePower = 0.72f;
            targetDemand = 0.68f;
            rodDepth = 0.32f;
            waterLevel = 0.82f;
            turbineSpin = 0.0f;
        }

        if (IsKeyDown(KEY_UP)) rodDepth -= 0.45f * GetFrameTime();
        if (IsKeyDown(KEY_DOWN)) rodDepth += 0.45f * GetFrameTime();
        if (IsKeyDown(KEY_RIGHT)) targetDemand += 0.38f * GetFrameTime();
        if (IsKeyDown(KEY_LEFT)) targetDemand -= 0.38f * GetFrameTime();
        rodDepth = std::clamp(rodDepth, 0.02f, 0.94f);
        targetDemand = std::clamp(targetDemand, 0.18f, 1.0f);

        UpdateOrbitCameraDragOnly(&camera, &orbit);

        if (!paused) {
            const float dt = GetFrameTime();
            time += dt;
            const float targetPower = std::clamp(1.08f - rodDepth * 0.95f + targetDemand * 0.22f, 0.06f, 1.0f);
            corePower += (targetPower - corePower) * (1.0f - std::pow(0.05f, dt));
            waterLevel += ((0.92f - targetDemand * 0.22f + rodDepth * 0.05f) - waterLevel) * (1.0f - std::pow(0.25f, dt));
            waterLevel = std::clamp(waterLevel, 0.35f, 1.0f);
            turbineSpin += dt * (4.0f + 22.0f * corePower * targetDemand);

            for (FlowParticle& p : coolant) {
                p.t = std::fmod(p.t + dt * p.speed * (0.55f + corePower), 1.0f);
                p.pos = CoolantPath(p.t, p.lane);
            }

            for (SteamParticle& p : steam) {
                p.age += dt * (0.7f + corePower * 0.55f);
                if (p.age > p.life) {
                    p.age = 0.0f;
                    p.life = 3.8f + 2.2f * static_cast<float>(GetRandomValue(0, 1000)) / 1000.0f;
                    p.angle = 2.0f * PI * static_cast<float>(GetRandomValue(0, 1000)) / 1000.0f;
                }
            }
        }

        BeginDrawing();
        ClearBackground(Color{8, 12, 18, 255});

        BeginMode3D(camera);
        DrawPlantBase();

        for (const BackgroundParticle& p : background) {
            DrawSphere(p.pos, p.size, Color{155, 190, 225, static_cast<unsigned char>(255.0f * p.alpha)});
        }

        DrawContainmentBuilding(corePower, rodDepth, cutaway);
        DrawCoolingTower({-6.6f, 0.0f, 2.95f}, time, steam, 0);
        DrawCoolingTower({-9.2f, 0.0f, 2.25f}, time + 1.3f, steam, 1);
        DrawTurbineHall(turbineSpin, targetDemand);

        DrawPipe({-1.9f, 2.72f, 0.18f}, {-5.5f, 2.72f, 0.18f}, 0.13f, Color{206, 222, 236, 255});
        DrawPipe({-5.7f, 0.42f, -0.18f}, {-1.8f, 0.42f, -0.18f}, 0.13f, Color{115, 165, 220, 255});
        DrawPipe({1.85f, 2.65f, 0.0f}, {3.12f, 1.45f, 0.0f}, 0.12f, Color{230, 235, 240, 255});
        DrawPipe({6.95f, 1.18f, 0.0f}, {-5.4f, 0.25f, 0.0f}, 0.08f, Color{109, 164, 222, 255});

        DrawCylinder({-5.85f, 1.45f, 0.0f}, 0.86f, 0.86f, 2.7f, 36, Color{102, 125, 150, 250});
        DrawCylinderWires({-5.85f, 1.45f, 0.0f}, 0.89f, 0.89f, 2.74f, 36, Color{195, 215, 230, 145});
        DrawSphere({-5.85f, 2.9f, 0.0f}, 0.74f, Color{225, 235, 240, 90});

        for (const FlowParticle& p : coolant) {
            const bool hot = p.t < 0.48f;
            Color c = hot ? LerpColor(Color{255, 91, 50, 240}, Color{255, 210, 88, 240}, corePower)
                          : Color{74, 181, 255, 235};
            DrawSphere(p.pos, p.size, c);
        }

        for (int i = 0; i < 7; ++i) {
            const float pulse = std::fmod(time * (0.55f + targetDemand) + i * 0.16f, 1.0f);
            DrawLine3D({8.2f + pulse * 4.5f, 3.95f, 1.05f}, {8.75f + pulse * 4.5f, 3.95f, 1.05f},
                       Color{255, 222, 92, static_cast<unsigned char>(120 + 90 * pulse)});
        }

        EndMode3D();

        DrawRectangle(0, 0, kScreenWidth, 122, Color{7, 10, 16, 190});
        DrawText("Nuclear Power Plant 3D Simulation", 24, 18, 30, Color{238, 244, 250, 255});
        DrawText("Left mouse: orbit | wheel: zoom | UP/DOWN rods | LEFT/RIGHT grid demand | C cutaway | P pause | R reset",
                 24, 56, 18, Color{176, 190, 210, 255});
        DrawText(Hud(corePower, targetDemand, rodDepth, waterLevel, paused, cutaway).c_str(), 24, 84, 19, Color{255, 215, 142, 255});

        DrawStatusBars(corePower, waterLevel, targetDemand, rodDepth);
        DrawFPS(kScreenWidth - 98, 18);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
