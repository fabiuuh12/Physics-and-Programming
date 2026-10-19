#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 820;
constexpr float kPi = 3.14159265358979323846f;

struct RoseSettings {
    float petals = 3.6f;
    float openness = 8.0f;
};

struct RoseVertex {
    Vector3 position;
    Color color;
};

float PositiveModulo(float value, float period) {
    float result = std::fmod(value, period);
    return result < 0.0f ? result + period : result;
}

RoseVertex EvaluateRose(float radial, float theta, const RoseSettings& settings) {
    const float phase = 0.5f * kPi * std::exp(-theta / (settings.openness * kPi));
    const float petalWave = 1.0f - PositiveModulo(settings.petals * theta, 2.0f * kPi) / kPi;
    const float envelope = 1.0f - 0.5f * std::pow(1.25f * petalWave * petalWave - 0.25f, 2.0f);
    const float fold = 1.995653f * radial * radial *
                       std::pow(1.27689f * radial - 1.0f, 2.0f) * std::sin(phase);
    const float radius = envelope * (radial * std::sin(phase) + fold * std::cos(phase));
    const float height = envelope * (radial * std::cos(phase) - fold * std::sin(phase));

    const float turn = (theta - 4.0f * kPi) / (20.0f * kPi);
    const float edge = std::pow(radial, 1.7f);
    const float hue = 345.0f + 11.0f * turn;
    const float saturation = 0.78f + 0.18f * edge;
    const float value = 0.48f + 0.43f * edge + 0.07f * std::sin(settings.petals * theta);
    Color color = ColorFromHSV(hue, std::clamp(saturation, 0.0f, 1.0f),
                               std::clamp(value, 0.0f, 1.0f));
    color.a = 255;

    return {{3.85f * radius * std::sin(theta),
             3.85f * height - 0.75f,
             3.85f * radius * std::cos(theta)},
            color};
}

Model BuildRoseModel(const RoseSettings& settings) {
    constexpr int kRadialSteps = 25;
    constexpr int kThetaSteps = 260;
    const float thetaMin = 4.0f * kPi;
    const float thetaMax = 24.0f * kPi;

    std::vector<RoseVertex> grid(kRadialSteps * kThetaSteps);
    for (int t = 0; t < kThetaSteps; ++t) {
        const float v = static_cast<float>(t) / static_cast<float>(kThetaSteps - 1);
        const float theta = Lerp(thetaMin, thetaMax, v);
        for (int r = 0; r < kRadialSteps; ++r) {
            const float radial = static_cast<float>(r) / static_cast<float>(kRadialSteps - 1);
            grid[t * kRadialSteps + r] = EvaluateRose(radial, theta, settings);
        }
    }

    const int triangleCount = (kThetaSteps - 1) * (kRadialSteps - 1) * 2;
    Mesh mesh{};
    mesh.triangleCount = triangleCount;
    mesh.vertexCount = triangleCount * 3;
    mesh.vertices = static_cast<float*>(MemAlloc(mesh.vertexCount * 3 * sizeof(float)));
    mesh.normals = static_cast<float*>(MemAlloc(mesh.vertexCount * 3 * sizeof(float)));
    mesh.colors = static_cast<unsigned char*>(MemAlloc(mesh.vertexCount * 4 * sizeof(unsigned char)));

    int vertexIndex = 0;
    auto writeTriangle = [&](const RoseVertex& a, const RoseVertex& b, const RoseVertex& c) {
        Vector3 normal = Vector3Normalize(Vector3CrossProduct(
            Vector3Subtract(b.position, a.position), Vector3Subtract(c.position, a.position)));
        const RoseVertex vertices[3] = {a, b, c};
        for (const RoseVertex& vertex : vertices) {
            const int p = vertexIndex * 3;
            const int col = vertexIndex * 4;
            mesh.vertices[p] = vertex.position.x;
            mesh.vertices[p + 1] = vertex.position.y;
            mesh.vertices[p + 2] = vertex.position.z;
            mesh.normals[p] = normal.x;
            mesh.normals[p + 1] = normal.y;
            mesh.normals[p + 2] = normal.z;
            mesh.colors[col] = vertex.color.r;
            mesh.colors[col + 1] = vertex.color.g;
            mesh.colors[col + 2] = vertex.color.b;
            mesh.colors[col + 3] = vertex.color.a;
            ++vertexIndex;
        }
    };

    for (int t = 0; t < kThetaSteps - 1; ++t) {
        for (int r = 0; r < kRadialSteps - 1; ++r) {
            const RoseVertex& a = grid[t * kRadialSteps + r];
            const RoseVertex& b = grid[(t + 1) * kRadialSteps + r];
            const RoseVertex& c = grid[(t + 1) * kRadialSteps + r + 1];
            const RoseVertex& d = grid[t * kRadialSteps + r + 1];
            writeTriangle(a, b, c);
            writeTriangle(a, c, d);
        }
    }

    UploadMesh(&mesh, false);
    return LoadModelFromMesh(mesh);
}

void UpdateOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        const Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0038f;
        *pitch += delta.y * 0.0038f;
        *pitch = std::clamp(*pitch, -0.15f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.75f;
    *distance = std::clamp(*distance, 6.5f, 24.0f);
    const float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target,
                                  {*distance * cp * std::cos(*yaw),
                                   *distance * std::sin(*pitch),
                                   *distance * cp * std::sin(*yaw)});
}

void DrawPanel(const RoseSettings& settings, bool wireframe, bool spinning) {
    const int screenWidth = GetScreenWidth();
    const int screenHeight = GetScreenHeight();
    DrawRectangleRounded({20, 18, 535, 132}, 0.06f, 5, Color{10, 12, 17, 224});
    DrawRectangleRoundedLinesEx({20, 18, 535, 132}, 0.06f, 5, 1.0f, Color{122, 50, 74, 220});
    DrawText("PARAMETRIC ROSE", 38, 34, 25, Color{247, 223, 228, 255});
    DrawText("A two-parameter surface: radial position x spiral angle", 38, 66, 17,
             Color{188, 178, 190, 255});
    char values[180];
    std::snprintf(values, sizeof(values), "petal frequency  %.1f     openness  %.1f", settings.petals,
                  settings.openness);
    DrawText(values, 38, 94, 19, Color{255, 119, 155, 255});
    std::snprintf(values, sizeof(values), "surface %s     rotation %s", wireframe ? "+ wireframe" : "shaded",
                  spinning ? "on" : "paused");
    DrawText(values, 38, 121, 16, Color{183, 153, 166, 255});

    const char* controls = "drag orbit   wheel zoom   LEFT/RIGHT petals   UP/DOWN openness   W mesh   SPACE spin   R reset";
    const int width = MeasureText(controls, 17);
    DrawRectangle(0, screenHeight - 43, screenWidth, 43, Color{8, 9, 13, 235});
    DrawText(controls, std::max(12, (screenWidth - width) / 2), screenHeight - 29, 17,
             Color{203, 190, 198, 255});
}
}  // namespace

int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(kWindowWidth, kWindowHeight, "Parametric Rose Surface - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.target = {0.0f, 0.35f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 43.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float yaw = 0.72f;
    float pitch = 0.55f;
    float distance = 11.5f;

    RoseSettings settings{};
    Model rose = BuildRoseModel(settings);
    bool wireframe = true;
    bool spinning = true;
    float rotation = 0.0f;

    while (!WindowShouldClose()) {
        bool rebuild = false;
        if (IsKeyPressed(KEY_LEFT)) {
            settings.petals = std::max(1.0f, settings.petals - 0.1f);
            rebuild = true;
        }
        if (IsKeyPressed(KEY_RIGHT)) {
            settings.petals = std::min(8.0f, settings.petals + 0.1f);
            rebuild = true;
        }
        if (IsKeyPressed(KEY_DOWN)) {
            settings.openness = std::max(3.0f, settings.openness - 0.5f);
            rebuild = true;
        }
        if (IsKeyPressed(KEY_UP)) {
            settings.openness = std::min(16.0f, settings.openness + 0.5f);
            rebuild = true;
        }
        if (IsKeyPressed(KEY_W)) wireframe = !wireframe;
        if (IsKeyPressed(KEY_SPACE)) spinning = !spinning;
        if (IsKeyPressed(KEY_R)) {
            settings = {};
            yaw = 0.72f;
            pitch = 0.55f;
            distance = 11.5f;
            rotation = 0.0f;
            rebuild = true;
        }
        if (rebuild) {
            UnloadModel(rose);
            rose = BuildRoseModel(settings);
        }

        UpdateOrbitCamera(&camera, &yaw, &pitch, &distance);
        if (spinning) rotation += 7.0f * GetFrameTime();

        BeginDrawing();
        ClearBackground(Color{4, 4, 7, 255});
        BeginMode3D(camera);
        rlDisableBackfaceCulling();
        DrawModelEx(rose, {0, 0, 0}, {0, 1, 0}, rotation, {1, 1, 1}, WHITE);
        if (wireframe) {
            DrawModelWiresEx(rose, {0, 0, 0}, {0, 1, 0}, rotation, {1, 1, 1},
                             Fade(Color{255, 196, 208, 255}, 0.32f));
        }
        rlEnableBackfaceCulling();
        DrawCircle3D({0, -1.05f, 0}, 3.4f, {1, 0, 0}, 90.0f, Fade(Color{116, 54, 73, 255}, 0.16f));
        EndMode3D();

        DrawPanel(settings, wireframe, spinning);
        DrawFPS(GetScreenWidth() - 92, 20);
        EndDrawing();

    }

    UnloadModel(rose);
    CloseWindow();
    return 0;
}
