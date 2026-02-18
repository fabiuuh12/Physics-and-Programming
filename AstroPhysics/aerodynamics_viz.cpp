#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <vector>

namespace {
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr float kPi = 3.14159265358979323846f;

struct Marker {
    Vector3 pos;
    std::deque<Vector3> trail;
};

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.7f;
    *distance = std::clamp(*distance, 6.0f, 42.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

Vector2 Rotate2D(Vector2 p, float a) {
    float c = std::cos(a);
    float s = std::sin(a);
    return {c * p.x - s * p.y, s * p.x + c * p.y};
}

float ClampMinR(float r, float minR) { return (r < minR) ? minR : r; }

float RocketRadiusAtLocalX(float lx, float halfLen, float bodyRadius) {
    if (lx < -halfLen || lx > halfLen) return 0.0f;
    const float noseLen = 0.52f * halfLen;
    const float tailLen = 0.18f * halfLen;

    if (lx > halfLen - noseLen) {
        float t = (halfLen - lx) / noseLen;   // 1 at nose base, 0 at tip
        return bodyRadius * std::max(0.08f, t);
    }
    if (lx < -halfLen + tailLen) {
        float t = (lx + halfLen) / tailLen;   // 0 at tail end, 1 at tail shoulder
        return bodyRadius * (0.58f + 0.42f * std::max(0.0f, t));
    }
    return bodyRadius;
}

bool InsideRocketBody(Vector3 world, float angle, float halfLen, float bodyRadius) {
    Vector2 localXY = Rotate2D({world.x, world.y}, -angle);
    float lx = localXY.x;
    float r = RocketRadiusAtLocalX(lx, halfLen, bodyRadius);
    if (r <= 0.0f) return false;

    float radial = std::sqrt(localXY.y * localXY.y + world.z * world.z);
    bool inHull = radial <= r;

    // Simple fin volumes near tail.
    const float finLen = 0.30f * halfLen;
    const float finThick = 0.04f;
    const float finSpan = 0.34f;
    bool tailBand = (lx >= -halfLen && lx <= -halfLen + finLen);
    bool finY = tailBand && (std::fabs(world.z) < finThick) &&
                (std::fabs(localXY.y) > bodyRadius) && (std::fabs(localXY.y) < bodyRadius + finSpan);
    bool finZ = tailBand && (std::fabs(localXY.y) < finThick) &&
                (std::fabs(world.z) > bodyRadius) && (std::fabs(world.z) < bodyRadius + finSpan);

    return inHull || finY || finZ;
}

Vector3 VelocityWorld(Vector3 pWorld, float angle, float uInf, float circulation, float bodyRadius, float halfLen) {
    Vector2 pLocal = Rotate2D({pWorld.x, pWorld.y}, -angle);
    float r = ClampMinR(std::sqrt(pLocal.x * pLocal.x + pLocal.y * pLocal.y), bodyRadius + 0.001f);
    float theta = std::atan2(pLocal.y, pLocal.x);
    float a2_r2 = (bodyRadius * bodyRadius) / (r * r);

    float vr = uInf * (1.0f - a2_r2) * std::cos(theta);
    float vt = -uInf * (1.0f + a2_r2) * std::sin(theta) + circulation / (2.0f * kPi * r);
    Vector2 eR{std::cos(theta), std::sin(theta)};
    Vector2 eT{-std::sin(theta), std::cos(theta)};
    Vector2 vLocalXY = Vector2Add(Vector2Scale(eR, vr), Vector2Scale(eT, vt));
    Vector2 vWorldXY = Rotate2D(vLocalXY, angle);

    // Wake-like swirl behind the rocket (tail side) as a turbulence proxy.
    float xLocal = pLocal.x;
    float radialYZ = std::sqrt(pLocal.y * pLocal.y + pWorld.z * pWorld.z) + 0.02f;
    float wakeX = std::max(0.0f, (-halfLen + 0.15f) - xLocal);
    float wakeFactor = std::exp(-0.8f * wakeX) * std::exp(-3.0f * radialYZ * radialYZ);
    float swirl = 0.9f * wakeFactor * (0.5f + 0.12f * circulation) / (0.25f + radialYZ);
    float vz = swirl * (pLocal.y / radialYZ);
    vWorldXY.y += -swirl * (pWorld.z / radialYZ);

    float liftLike = std::clamp(circulation * 0.16f + angle * 0.8f, -2.0f, 2.0f);
    float downwash = -0.35f * liftLike * std::exp(-0.18f * pWorld.x * pWorld.x) * std::exp(-0.35f * pWorld.y * pWorld.y);
    vWorldXY.y += downwash;

    return {vWorldXY.x, vWorldXY.y, vz};
}

float FlowBadnessScore(Vector3 pWorld, Vector3 v, float angle, float uInf, float circulation, float bodyRadius, float halfLen) {
    float speed = Vector3Length(v);
    float speedMismatch = std::fabs(speed - uInf) / std::max(0.2f, uInf);

    float crossflow = (std::fabs(v.y) + 1.4f * std::fabs(v.z)) / std::max(0.2f, uInf);

    // Finite-difference estimate of local shear (a proxy for turbulent/draggy regions).
    const float h = 0.14f;
    Vector3 vyP = VelocityWorld(Vector3Add(pWorld, {0.0f, h, 0.0f}), angle, uInf, circulation, bodyRadius, halfLen);
    Vector3 vyM = VelocityWorld(Vector3Add(pWorld, {0.0f, -h, 0.0f}), angle, uInf, circulation, bodyRadius, halfLen);
    Vector3 vzP = VelocityWorld(Vector3Add(pWorld, {0.0f, 0.0f, h}), angle, uInf, circulation, bodyRadius, halfLen);
    Vector3 vzM = VelocityWorld(Vector3Add(pWorld, {0.0f, 0.0f, -h}), angle, uInf, circulation, bodyRadius, halfLen);
    float shear = (Vector3Length(Vector3Subtract(vyP, vyM)) + Vector3Length(Vector3Subtract(vzP, vzM))) / (2.0f * h);
    float shearNorm = shear / (6.0f * std::max(0.2f, uInf));

    float badness = 0.45f * speedMismatch + 0.30f * crossflow + 0.25f * shearNorm;
    return std::clamp(badness, 0.0f, 1.0f);
}

Color FlowQualityColor(float badness) {
    // 0.0 -> green (clean flow), 1.0 -> red (high losses/turbulence proxy).
    badness = std::clamp(badness, 0.0f, 1.0f);
    unsigned char r = static_cast<unsigned char>(35.0f + 220.0f * badness);
    unsigned char g = static_cast<unsigned char>(230.0f - 190.0f * badness);
    unsigned char b = static_cast<unsigned char>(70.0f - 40.0f * badness);
    return Color{r, g, b, 255};
}

void ResetMarkers(std::vector<Marker>* marks) {
    int idx = 0;
    for (int iz = 0; iz < 12; ++iz) {
        for (int iy = 0; iy < 10; ++iy) {
            for (int ix = 0; ix < 3; ++ix) {
                Vector3 p{-6.5f - 0.45f * ix, -2.2f + 0.48f * iy, -1.7f + 0.32f * iz};
                (*marks)[idx].pos = p;
                (*marks)[idx].trail.clear();
                (*marks)[idx].trail.push_back(p);
                idx++;
            }
        }
    }
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Aerodynamics 3D - Wing Flow and Streamlines");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {10.0f, 5.4f, 10.5f};
    camera.target = {0.3f, 0.1f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f;
    float camPitch = 0.31f;
    float camDistance = 16.2f;

    const float rocketHalfLen = 2.15f;
    const float rocketRadius = 0.34f;
    const float flowRadius = 0.86f;

    float angleDeg = 8.0f;
    float uInf = 2.7f;
    float circulation = 4.0f;
    bool paused = false;

    std::vector<Marker> marks(12 * 10 * 3);
    ResetMarkers(&marks);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            paused = false;
            angleDeg = 8.0f;
            uInf = 2.7f;
            circulation = 4.0f;
            ResetMarkers(&marks);
        }

        float dt = GetFrameTime();
        if (IsKeyDown(KEY_UP)) angleDeg = std::min(24.0f, angleDeg + 23.0f * dt);
        if (IsKeyDown(KEY_DOWN)) angleDeg = std::max(-10.0f, angleDeg - 23.0f * dt);
        if (IsKeyDown(KEY_RIGHT)) uInf = std::min(6.0f, uInf + 2.0f * dt);
        if (IsKeyDown(KEY_LEFT)) uInf = std::max(0.7f, uInf - 2.0f * dt);
        if (IsKeyDown(KEY_RIGHT_BRACKET)) circulation = std::min(10.5f, circulation + 4.2f * dt);
        if (IsKeyDown(KEY_LEFT_BRACKET)) circulation = std::max(-2.5f, circulation - 4.2f * dt);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float angle = angleDeg * DEG2RAD;
        if (!paused) {
            for (auto& m : marks) {
                Vector3 v = VelocityWorld(m.pos, angle, uInf, circulation, flowRadius, rocketHalfLen);
                m.pos = Vector3Add(m.pos, Vector3Scale(v, dt));

                if (InsideRocketBody(m.pos, angle, rocketHalfLen, rocketRadius) ||
                    m.pos.x > 7.0f || std::fabs(m.pos.y) > 3.2f || std::fabs(m.pos.z) > 2.3f) {
                    m.pos = {-6.6f, GetRandomValue(-230, 230) / 100.0f, GetRandomValue(-180, 180) / 100.0f};
                    m.trail.clear();
                }
                m.trail.push_back(m.pos);
                if (m.trail.size() > 120) m.trail.pop_front();
            }
        }

        BeginDrawing();
        ClearBackground(Color{6, 9, 16, 255});
        BeginMode3D(camera);

        DrawGrid(20, 0.7f);

        for (float z = -1.8f; z <= 1.8f; z += 0.6f) {
            for (float y = -2.2f; y <= 2.2f; y += 0.5f) {
                for (float x = -5.8f; x <= 5.8f; x += 0.75f) {
                    Vector3 p{x, y, z};
                    if (InsideRocketBody(p, angle, rocketHalfLen, rocketRadius)) continue;
                    Vector3 v = VelocityWorld(p, angle, uInf, circulation, flowRadius, rocketHalfLen);
                    float speed = Vector3Length(v);
                    Vector3 dir = Vector3Normalize(v);
                    float len = 0.12f + 0.06f * std::min(speed / std::max(uInf, 0.1f), 2.5f);
                    float badness = FlowBadnessScore(p, v, angle, uInf, circulation, flowRadius, rocketHalfLen);
                    DrawLine3D(p, Vector3Add(p, Vector3Scale(dir, len)), Fade(FlowQualityColor(badness), 0.45f));
                }
            }
        }

        Vector3 axis = {std::cos(angle), std::sin(angle), 0.0f};
        Vector3 tail = Vector3Scale(axis, -rocketHalfLen);
        Vector3 bodyTail = Vector3Scale(axis, -rocketHalfLen + 0.35f);
        Vector3 noseBase = Vector3Scale(axis, rocketHalfLen - 0.62f);
        Vector3 noseTip = Vector3Scale(axis, rocketHalfLen + 0.58f);
        DrawCylinderEx(tail, bodyTail, rocketRadius * 0.72f, rocketRadius, 22, Color{210, 218, 232, 255});
        DrawCylinderEx(bodyTail, noseBase, rocketRadius, rocketRadius, 28, Color{232, 238, 248, 255});
        DrawCylinderEx(noseBase, noseTip, rocketRadius, 0.03f, 26, Color{244, 248, 255, 255});
        DrawSphere(tail, rocketRadius * 0.70f, Color{188, 196, 214, 255});
        DrawCylinderEx(Vector3Add(tail, Vector3Scale(axis, -0.20f)), tail, rocketRadius * 0.28f, rocketRadius * 0.20f, 18,
                       Color{72, 82, 105, 255});

        for (const auto& m : marks) {
            Vector3 vHere = VelocityWorld(m.pos, angle, uInf, circulation, flowRadius, rocketHalfLen);
            float badness = FlowBadnessScore(m.pos, vHere, angle, uInf, circulation, flowRadius, rocketHalfLen);
            Color qColor = FlowQualityColor(badness);
            for (size_t i = 1; i < m.trail.size(); ++i) {
                float a = static_cast<float>(i) / static_cast<float>(m.trail.size());
                DrawLine3D(m.trail[i - 1], m.trail[i], Fade(qColor, a * 0.80f));
            }
            DrawSphere(m.pos, 0.035f, Fade(qColor, 0.95f));
        }

        EndMode3D();

        DrawText("Aerodynamics 3D: Rocket Flow, Drag and Stability", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | Up/Down AoA | Left/Right speed | [ ] circulation | P pause | R reset",
                 20, 54, 18, Color{164, 183, 210, 255});
        char buf[256];
        std::snprintf(buf, sizeof(buf), "AoA=%.1f deg   U_inf=%.2f   circulation=%.2f   color: green=clean flow, red=high turbulence/drag%s",
                      angleDeg, uInf, circulation, paused ? "   [PAUSED]" : "");
        DrawText(buf, 20, 82, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
