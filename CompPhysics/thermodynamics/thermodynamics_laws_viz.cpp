#include "raylib.h"
#include "../common/studio.h"
#include "../common/physics_models.h"
#include <array>
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

struct GasParticle {
    Vector3 pos;
    Vector3 vel;
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
    *distance = std::clamp(*distance, 4.0f, 34.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

}  // namespace

int main() {
    if (std::getenv("COMPPHYSICS_SMOKE_FRAMES")) SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(kScreenWidth, kScreenHeight, "Thermodynamics Gas Laws 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 5.0f, 9.0f};
    camera.target = {0.0f, 0.8f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.85f, camPitch = 0.34f, camDistance = 13.0f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);

    std::normal_distribution<float> normal(0.0f,1.0f);
    physics::Clock clock;
    double impulseSum=0, pressureTime=0, measuredPressure=0;
    bool pressureReady=false;
    float halfX = 2.4f;
    float halfY = 1.6f;
    float halfZ = 1.8f;
    float temperature = 1.0f;

    std::vector<GasParticle> gas;
    gas.reserve(240);

    auto reset = [&]() {
        gas.clear();
        for (int i = 0; i < 240; ++i) {
            Vector3 p = {ur(rng) * halfX * 0.95f, ur(rng) * halfY * 0.95f + 0.8f, ur(rng) * halfZ * 0.95f};
            Vector3 v = {normal(rng), normal(rng), normal(rng)};
            gas.push_back({p, v});
        }
        double v2=0;
        for (const auto& p:gas) v2+=Vector3LengthSqr(p.vel);
        float scale=std::sqrt(3.0*gas.size()*temperature/v2);
        for (auto& p:gas) p.vel=Vector3Scale(p.vel,scale);
        clock.reset(); impulseSum=0; pressureTime=0; measuredPressure=0; pressureReady=false;
    };

    reset();
    bool paused = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { temperature = 1.0f; reset(); paused = false; }
        float previousTemperature=temperature;
        if (IsKeyPressed(KEY_LEFT_BRACKET)) temperature = std::max(0.2f, temperature - 0.1f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) temperature = std::min(4.0f, temperature + 0.1f);

        if (temperature!=previousTemperature) {
            float scale=std::sqrt(temperature/previousTemperature);
            for (auto& g:gas) g.vel=Vector3Scale(g.vel,scale);
            impulseSum=0; pressureTime=0; pressureReady=false;
        }
        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);
        if (!paused) clock.advance(GetFrameTime(),[&](double dt) {
            for (auto& gp:gas) {
                auto bounce=[&](float& p,float& v,double lo,double hi) {
                    double pp=p,vv=v;
                    impulseSum+=physics::reflect(pp,vv,lo,hi,dt);
                    p=float(pp); v=float(vv);
                };
                bounce(gp.pos.x,gp.vel.x,-halfX,halfX);
                bounce(gp.pos.y,gp.vel.y,0.8f-halfY,0.8f+halfY);
                bounce(gp.pos.z,gp.vel.z,-halfZ,halfZ);
            }
            pressureTime+=dt;
            if (pressureTime>=2.0) {
                double area=8*(halfX*halfY+halfX*halfZ+halfY*halfZ);
                measuredPressure=impulseSum/(area*pressureTime);
                impulseSum=0; pressureTime=0; pressureReady=true;
            }
        });
        double v2=0;
        std::array<float,24> speeds{};
        for (const auto& gp:gas) {
            double speed=Vector3Length(gp.vel); v2+=speed*speed;
            speeds[std::min(23,int(speed/8*24))]++;
        }
        double kineticTemperature=v2/(3*gas.size());

        float volume = (2.0f * halfX) * (2.0f * halfY) * (2.0f * halfZ);
        float n = static_cast<float>(gas.size());
        float pIdeal = n * kineticTemperature / std::max(0.1f, volume);

        BeginDrawing();
        ClearBackground(Color{7, 10, 16, 255});

        BeginMode3D(camera);

        DrawCubeWires({0.0f, 0.8f, 0.0f}, 2.0f * halfX, 2.0f * halfY, 2.0f * halfZ, Color{130, 180, 255, 180});

        for (const GasParticle& gp : gas) {
            float sp = Vector3Length(gp.vel);
            float heat = std::clamp(sp / 3.0f, 0.0f, 1.0f);
            Color c = Color{static_cast<unsigned char>(100 + 155 * heat), static_cast<unsigned char>(120 + 80 * (1.0f - heat)), 255, 230};
            DrawSphere(gp.pos, 0.05f, c);
        }

        EndMode3D();

        studio::title("Thermodynamics Laws: Ideal Gas in a Box", studio::Style::Instrument);
        studio::help("Hold left mouse: orbit | wheel: zoom | [ ] temperature | P pause | R reset");

        std::ostringstream os;
        os << std::fixed << std::setprecision(3)
           << "N=" << gas.size()
           << "  T=" << temperature
           << "  V=" << volume
           << "  P(kinetic)=" << pIdeal
           << "  P(wall,2s)=" << measuredPressure;
        if (paused) os << "  [PAUSED]";
        studio::readout(os.str().c_str());
        studio::note(pressureReady ? "Unit mass / kB = 1 / wall pressure from measured collision impulses"
                                   : "Measuring wall impulses... first pressure estimate after 2 simulated seconds");
        studio::plot({float(GetScreenWidth()-388),float(GetScreenHeight()-278),360,210},
                     "Speed histogram / 0 to 8 units",speeds,0,60,Color{235,182,123,255});
        studio::fps();

        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    studio::unload();
    CloseWindow();
    return 0;
}
