#include "../vision/hand_tracking_scene_shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <vector>

using namespace astro_hand;

namespace {

constexpr int kScreenWidth = 1480;
constexpr int kScreenHeight = 920;

struct Satellite {
    Vector3 pos{};
    Vector3 vel{};
    Color color{};
    int heldBy = -1;
    float stableTime = 0.0f;
};

Vector3 HandTarget(const HandControlState& hand, bool rightSide, float t) {
    if (hand.active) return hand.pinchPoint;
    const float a = t * 0.36f + (rightSide ? 0.0f : PI);
    return {(rightSide ? 1.0f : -1.0f) * 3.0f + std::cos(a) * 0.4f, 1.2f, std::sin(a) * 2.0f};
}

void ResetSatellite(Satellite& sat, float seed) {
    const float a = seed * 2.0f * PI;
    const float r = 3.0f + 2.8f * std::fmod(seed * 17.3f, 1.0f);
    sat.pos = {std::cos(a) * r, 1.1f + 0.25f * std::sin(seed * 12.0f), std::sin(a) * r};
    const float v = std::sqrt(10.5f / std::max(r, 0.8f));
    sat.vel = {-std::sin(a) * v, 0.0f, std::cos(a) * v};
    sat.heldBy = -1;
    sat.stableTime = 0.0f;
}

void DrawPlanet(float t) {
    DrawSphere({0.0f, 0.0f, 0.0f}, 1.45f, Color{94, 140, 240, 255});
    DrawSphere({0.0f, 0.0f, 0.0f}, 1.02f, Color{136, 186, 255, 255});
    DrawSphere({0.0f, 0.0f, 0.0f}, 1.68f, Fade(Color{126, 184, 255, 255}, 0.12f));
    DrawSphere({0.62f * std::cos(t * 0.2f), 0.54f, 0.62f * std::sin(t * 0.2f)}, 0.22f, Fade(Color{255, 255, 255, 255}, 0.30f));
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Orbital Construction Hand Lab");
    SetTargetFPS(120);
    SetWindowMinSize(1080, 720);

    Camera3D camera{};
    camera.target = {0.0f, 1.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 42.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw = 0.80f;
    float camPitch = 0.42f;
    float camDistance = 15.5f;

    HandSceneBridge bridge;
    bridge.Start();

    std::vector<Satellite> sats = {
        {{}, {}, Color{132, 224, 255, 255}},
        {{}, {}, Color{255, 214, 132, 255}},
        {{}, {}, Color{204, 255, 154, 255}},
        {{}, {}, Color{255, 168, 198, 255}},
        {{}, {}, Color{214, 190, 255, 255}},
    };
    for (size_t i = 0; i < sats.size(); ++i) ResetSatellite(sats[i], static_cast<float>(i) / static_cast<float>(sats.size()));

    std::array<bool, 2> prevPinched = {false, false};

    while (!WindowShouldClose()) {
        const float dt = std::max(GetFrameTime(), 1.0e-4f);
        const float t = static_cast<float>(GetTime());

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);
        bridge.Update(camera, t, dt);

        const std::array<const HandControlState*, 2> hands = {&bridge.Control(false), &bridge.Control(true)};

        for (size_t hi = 0; hi < hands.size(); ++hi) {
            const HandControlState& hand = *hands[hi];
            const bool pinched = hand.active && hand.pinched;
            if (pinched) {
                int held = -1;
                for (size_t si = 0; si < sats.size(); ++si) {
                    if (sats[si].heldBy == static_cast<int>(hi)) held = static_cast<int>(si);
                }
                if (held < 0) {
                    float bestDist = 0.75f;
                    for (size_t si = 0; si < sats.size(); ++si) {
                        if (sats[si].heldBy >= 0) continue;
                        const float dist = Vector3Distance(sats[si].pos, hand.pinchPoint);
                        if (dist < bestDist) {
                            bestDist = dist;
                            held = static_cast<int>(si);
                        }
                    }
                    if (held >= 0) sats[static_cast<size_t>(held)].heldBy = static_cast<int>(hi);
                }
            }
            if (prevPinched[hi] && !pinched) {
                for (Satellite& sat : sats) {
                    if (sat.heldBy == static_cast<int>(hi)) {
                        sat.heldBy = -1;
                        sat.vel = Vector3Scale(hand.velocity, 0.060f);
                    }
                }
            }
            prevPinched[hi] = pinched;
        }

        int stableCount = 0;
        for (size_t i = 0; i < sats.size(); ++i) {
            Satellite& sat = sats[i];
            if (sat.heldBy >= 0) {
                sat.pos = hands[static_cast<size_t>(sat.heldBy)]->pinchPoint;
                sat.vel = {0.0f, 0.0f, 0.0f};
                sat.stableTime = 0.0f;
                continue;
            }

            Vector3 d = Vector3Subtract({0.0f, 0.0f, 0.0f}, sat.pos);
            const float r = std::max(0.8f, Vector3Length(d));
            sat.vel = Vector3Add(sat.vel, Vector3Scale(d, 10.5f * dt / (r * r * r)));
            sat.pos = Vector3Add(sat.pos, Vector3Scale(sat.vel, dt));

            const float speed = Vector3Length(sat.vel);
            const float energy = 0.5f * speed * speed - 10.5f / r;
            if (r > 2.2f && r < 8.2f && energy < -0.05f) sat.stableTime += dt;
            else sat.stableTime = std::max(0.0f, sat.stableTime - 2.0f * dt);
            if (sat.stableTime > 2.0f) ++stableCount;

            if (r < 1.65f || r > 11.0f) ResetSatellite(sat, std::fmod(t * 0.13f + static_cast<float>(i) * 0.19f, 1.0f));
        }

        BeginDrawing();
        ClearBackground(Color{4, 8, 16, 255});
        DrawStarfieldBackdrop(280, 0x44AA17u, t * 0.10f, Color{228, 236, 255, 255});

        BeginMode3D(camera);
        DrawPlanet(t);
        for (const Satellite& sat : sats) {
            DrawSphere(sat.pos, 0.16f, sat.color);
            DrawSphere(sat.pos, 0.28f, Fade(sat.color, 0.18f));
            if (sat.stableTime > 1.2f) {
                DrawSphereWires({0.0f, 0.0f, 0.0f}, Vector3Length(sat.pos), 32, 20, Fade(sat.color, 0.18f));
            }
        }
        if (bridge.AnyTracked()) bridge.DrawHands(false);
        EndMode3D();

        DrawText("Orbital Construction Hand Lab", 20, 18, 34, Color{236, 240, 248, 255});
        DrawText("Pinch a moon or satellite, move it, then release it with a good tangent velocity to build a stable orbit.", 20, 58, 20, Color{182, 198, 226, 255});
        std::ostringstream os;
        os.setf(std::ios::fixed);
        os.precision(2);
        os << "stable orbits=" << stableCount << "/" << sats.size();
        DrawText(os.str().c_str(), 20, 88, 19, Color{132, 228, 255, 255});
        DrawBridgeStatus(bridge, 20, 116);
        DrawText("This is more game-like: grab, place, release, and see whether the orbit survives or crashes out.", 20, 142, 18, Color{255, 218, 142, 255});
        DrawFPS(20, 170);
        bridge.DrawPreviewPanel({static_cast<float>(GetScreenWidth() - 392), 20.0f, 360.0f, 220.0f}, "Python Webcam Feed");
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
