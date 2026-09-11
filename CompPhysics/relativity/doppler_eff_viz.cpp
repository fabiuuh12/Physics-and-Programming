#include "raylib.h"
#include "../common/studio.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

struct Wavefront {
    Vector3 center;
    float radius;
};

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* distance) {
    studio::pan(c, *yaw, *pitch, *distance);
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && !studio::panGesture()) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 35.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

void DrawWavefrontXZ(Vector3 c, float r, Color col) {
    int seg = 80;
    for (int i = 0; i < seg; ++i) {
        float a0 = 2.0f * PI * static_cast<float>(i) / seg;
        float a1 = 2.0f * PI * static_cast<float>(i + 1) / seg;
        DrawLine3D({c.x + r * std::cos(a0), c.y, c.z + r * std::sin(a0)}, {c.x + r * std::cos(a1), c.y, c.z + r * std::sin(a1)}, col);
    }
}

} // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Doppler Effect 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 4.8f, 8.5f};
    camera.target = {0.0f, 0.7f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.84f, camPitch = 0.34f, camDistance = 13.0f;

    float sourceSpeed = 1.6f;
    float waveSpeed = 4.0f;
    float sourceFreq = 1.4f;
    bool paused = false;

    Vector3 source = {-5.0f, 0.7f, 0.0f};
    Vector3 observer = {4.8f, 0.7f, 0.0f};

    std::deque<Wavefront> waves;
    float emitTimer = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { paused = false; source = {-5.0f, 0.7f, 0.0f}; waves.clear(); emitTimer = 0.0f; sourceSpeed = 1.6f; sourceFreq = 1.4f; }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) sourceSpeed = std::max(-3.5f, sourceSpeed - 0.2f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) sourceSpeed = std::min(3.5f, sourceSpeed + 0.2f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) sourceFreq = std::max(0.2f, sourceFreq - 0.1f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) sourceFreq = std::min(4.0f, sourceFreq + 0.1f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float dt = std::min(GetFrameTime(),0.1f);
            source.x += sourceSpeed * dt;
            if (source.x > 5.5f) { source.x = -5.5f; waves.clear(); emitTimer=0; }
            if (source.x < -5.5f) { source.x = 5.5f; waves.clear(); emitTimer=0; }

            for (Wavefront& w : waves) w.radius+=waveSpeed*dt;
            emitTimer += dt;
            float period = 1.0f / sourceFreq;
            while (emitTimer >= period) {
                emitTimer -= period;
                Vector3 emission=source;
                emission.x-=sourceSpeed*emitTimer;
                waves.push_back({emission,waveSpeed*emitTimer});
            }

            while (!waves.empty() && waves.front().radius > 18.0f) waves.pop_front();
        }

        float beta = sourceSpeed / waveSpeed;
        float observedAhead = sourceFreq / std::max(0.1f, (1.0f - beta));
        float observedBehind = sourceFreq / std::max(0.1f, (1.0f + beta));

        BeginDrawing();
        ClearBackground(Color{6, 9, 16, 255});
        BeginMode3D(camera);
        DrawLine3D({-6.0f, 0.7f, 0.0f}, {6.0f, 0.7f, 0.0f}, Color{120, 140, 180, 100});

        for (const Wavefront& w : waves) {
            DrawWavefrontXZ(w.center, w.radius, Color{130, 210, 255, 110});
        }

        DrawSphere(source, 0.2f, Color{255, 180, 110, 255});
        DrawSphere(observer, 0.2f, Color{130, 220, 255, 255});

        DrawLine3D(source, Vector3Add(source, {sourceSpeed > 0 ? 0.8f : -0.8f, 0.0f, 0.0f}), Color{255, 200, 140, 255});

        EndMode3D();

        studio::title("Doppler Effect: Moving Source Wave Compression", studio::Style::Instrument);
        studio::help("Hold left mouse: orbit | wheel: zoom | [ ] source speed | +/- source freq | P pause | R reset");

        std::ostringstream os;
        os << std::fixed << std::setprecision(2)
           << "v_source=" << sourceSpeed << "  v_wave=" << waveSpeed << "  f_source=" << sourceFreq
           << "  f(+X)=" << observedAhead << "  f(-X)=" << observedBehind;
        if (paused) os << "  [PAUSED]";
        studio::readout(os.str().c_str());
        studio::note("Classical sound-wave cross-section / fronts centered at emission / source wraps with a fresh train");
        studio::fps();

        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    studio::unload();
    CloseWindow();
    return 0;
}
