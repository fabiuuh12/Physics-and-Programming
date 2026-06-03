#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1500;
constexpr int kScreenHeight = 940;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kHalfLength = 2.95f;

struct TunnelState {
    Vector3 bodyCenter = {0.0f, 0.60f, 0.0f};
    float windSpeed = 19.0f;
    float wakeStrength = 0.72f;
    float bodyScale = 1.0f;
    float roofBias = 0.0f;
};

struct FlowParticle {
    Vector3 pos{};
    std::deque<Vector3> trail;
    float age = 0.0f;
    float life = 0.0f;
    float lane = 0.0f;
};

struct WakeBlob {
    Vector3 pos{};
    float radius = 0.18f;
    float strength = 0.0f;
    float age = 0.0f;
};

float Saturate(float value) { return std::clamp(value, 0.0f, 1.0f); }

float Lerp(float a, float b, float t) { return a + (b - a) * t; }

float SmoothStep(float edge0, float edge1, float x) {
    float t = Saturate((x - edge0) / std::max(0.0001f, edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

float RandomFloat(float minValue, float maxValue) {
    return Lerp(minValue, maxValue, static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f);
}

Color LerpColor(Color a, Color b, float t) {
    t = Saturate(t);
    return Color{
        static_cast<unsigned char>(Lerp(static_cast<float>(a.r), static_cast<float>(b.r), t)),
        static_cast<unsigned char>(Lerp(static_cast<float>(a.g), static_cast<float>(b.g), t)),
        static_cast<unsigned char>(Lerp(static_cast<float>(a.b), static_cast<float>(b.b), t)),
        static_cast<unsigned char>(Lerp(static_cast<float>(a.a), static_cast<float>(b.a), t))};
}

void UpdateOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0034f;
        *pitch += delta.y * 0.0030f;
        *pitch = std::clamp(*pitch, -1.15f, 1.15f);
    }

    *distance -= GetMouseWheelMove() * 0.85f;
    *distance = std::clamp(*distance, 8.0f, 30.0f);

    float cosPitch = std::cos(*pitch);
    camera->position = Vector3Add(
        camera->target,
        {*distance * cosPitch * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cosPitch * std::sin(*yaw)});
}

float SampleCurve(const std::array<Vector2, 6>& knots, float x) {
    if (x <= knots.front().x) return knots.front().y;
    if (x >= knots.back().x) return knots.back().y;
    for (size_t i = 1; i < knots.size(); ++i) {
        if (x <= knots[i].x) {
            float t = SmoothStep(knots[i - 1].x, knots[i].x, x);
            return Lerp(knots[i - 1].y, knots[i].y, t);
        }
    }
    return knots.back().y;
}

float TopProfile(float x, const TunnelState& state) {
    static constexpr std::array<Vector2, 6> kTop = {
        Vector2{-2.95f, 0.34f}, Vector2{-2.20f, 0.42f}, Vector2{-1.15f, 1.02f},
        Vector2{ 0.25f, 1.18f}, Vector2{ 1.55f, 1.00f}, Vector2{ 2.95f, 0.36f},
    };
    return state.bodyCenter.y + state.bodyScale * (SampleCurve(kTop, x) + state.roofBias * 0.18f * SmoothStep(-1.3f, 0.8f, x));
}

float FloorProfile(float x, const TunnelState& state) {
    static constexpr std::array<Vector2, 6> kFloor = {
        Vector2{-2.95f, 0.06f}, Vector2{-2.10f, 0.08f}, Vector2{-0.70f, 0.10f},
        Vector2{ 1.25f, 0.08f}, Vector2{ 2.35f, 0.10f}, Vector2{ 2.95f, 0.14f},
    };
    return state.bodyCenter.y + state.bodyScale * SampleCurve(kFloor, x);
}

float HalfWidthProfile(float x, const TunnelState& state) {
    static constexpr std::array<Vector2, 6> kWidth = {
        Vector2{-2.95f, 0.16f}, Vector2{-2.10f, 0.62f}, Vector2{-0.85f, 1.02f},
        Vector2{ 0.75f, 1.06f}, Vector2{ 1.95f, 0.88f}, Vector2{ 2.95f, 0.20f},
    };
    return state.bodyScale * SampleCurve(kWidth, x);
}

float SectionMidY(float x, const TunnelState& state) { return 0.5f * (TopProfile(x, state) + FloorProfile(x, state)); }

float SectionHalfHeight(float x, const TunnelState& state) { return 0.5f * (TopProfile(x, state) - FloorProfile(x, state)); }

Vector3 ToLocal(Vector3 world, const TunnelState& state) { return Vector3Subtract(world, state.bodyCenter); }

Vector3 ToWorld(Vector3 local, const TunnelState& state) { return Vector3Add(local, state.bodyCenter); }

float BodySignedDistanceLocal(Vector3 local, const TunnelState& state) {
    float xClamped = std::clamp(local.x, -kHalfLength, kHalfLength);
    float top = TopProfile(xClamped, state) - state.bodyCenter.y;
    float floor = FloorProfile(xClamped, state) - state.bodyCenter.y;
    float yMid = 0.5f * (top + floor);
    float halfHeight = std::max(0.001f, 0.5f * (top - floor));
    float halfWidth = std::max(0.001f, HalfWidthProfile(xClamped, state));

    float dy = local.y - yMid;
    float radial = std::sqrt((dy * dy) / (halfHeight * halfHeight) + (local.z * local.z) / (halfWidth * halfWidth));
    float crossDistance = (radial - 1.0f) * std::min(halfHeight, halfWidth);
    float dx = local.x - xClamped;

    if (std::fabs(dx) < 0.0001f) return crossDistance;
    if (crossDistance < 0.0f) return -std::sqrt(dx * dx + crossDistance * crossDistance);
    return std::sqrt(dx * dx + crossDistance * crossDistance);
}

Vector3 SurfaceNormalLocal(Vector3 local, const TunnelState& state) {
    const float h = 0.03f;
    float dx = BodySignedDistanceLocal({local.x + h, local.y, local.z}, state) - BodySignedDistanceLocal({local.x - h, local.y, local.z}, state);
    float dy = BodySignedDistanceLocal({local.x, local.y + h, local.z}, state) - BodySignedDistanceLocal({local.x, local.y - h, local.z}, state);
    float dz = BodySignedDistanceLocal({local.x, local.y, local.z + h}, state) - BodySignedDistanceLocal({local.x, local.y, local.z - h}, state);
    return Vector3Normalize(Vector3{dx, dy, dz});
}

bool InsideBody(Vector3 world, const TunnelState& state) { return BodySignedDistanceLocal(ToLocal(world, state), state) <= 0.0f; }

float AeroSurfaceScore(Vector3 world, const TunnelState& state, const std::vector<WakeBlob>& wake);

Color AeroSurfaceColor(float score) {
    Color clean{88, 217, 255, 255};
    Color neutral{214, 227, 236, 255};
    Color dirty{255, 142, 82, 255};
    if (score < 0.5f) return LerpColor(dirty, neutral, score / 0.5f);
    return LerpColor(neutral, clean, (score - 0.5f) / 0.5f);
}

Color StreamColor(float speedRatio) {
    Color slow{255, 160, 90, 255};
    Color mid{208, 230, 236, 255};
    Color fast{105, 220, 255, 255};
    float t = Saturate((speedRatio - 0.55f) / 0.80f);
    if (t < 0.5f) return LerpColor(slow, mid, t / 0.5f);
    return LerpColor(mid, fast, (t - 0.5f) / 0.5f);
}

Vector3 ComputeWakeVelocity(Vector3 world, const TunnelState& state, const std::vector<WakeBlob>& wake) {
    Vector3 local = ToLocal(world, state);
    Vector3 velocity{0.0f, 0.0f, 0.0f};

    if (local.x > 2.30f * state.bodyScale) {
        float wakeX = local.x - 2.30f * state.bodyScale;
        float spreadZ = state.bodyScale * (0.90f + 0.22f * std::sqrt(std::max(0.0f, wakeX)));
        float spreadY = state.bodyScale * (0.55f + 0.18f * wakeX);
        float profile = std::exp(-(local.z * local.z) / (spreadZ * spreadZ) -
                                 ((local.y - 0.25f * state.bodyScale) * (local.y - 0.25f * state.bodyScale)) / (spreadY * spreadY));
        float deficit = state.wakeStrength * std::exp(-0.16f * wakeX) * profile;
        float recirculation = 0.55f * std::exp(-0.85f * wakeX) * profile;
        velocity.x -= state.windSpeed * deficit;
        velocity.x -= state.windSpeed * recirculation * (wakeX < 1.2f * state.bodyScale ? 1.0f : 0.0f);
        velocity.y += -local.y * state.windSpeed * 0.06f * profile;
    }

    for (const WakeBlob& blob : wake) {
        Vector3 fromBlob = Vector3Subtract(world, blob.pos);
        float rho = std::sqrt(fromBlob.y * fromBlob.y + fromBlob.z * fromBlob.z);
        float falloff = std::exp(-(rho * rho) / (2.0f * blob.radius * blob.radius)) * std::exp(-0.65f * std::fabs(fromBlob.x));
        if (rho > 0.0001f) {
            Vector3 tangent = {0.0f, -fromBlob.z / rho, fromBlob.y / rho};
            velocity = Vector3Add(velocity, Vector3Scale(tangent, blob.strength * falloff));
        }
    }

    return velocity;
}

Vector3 ComputeFlowVelocity(Vector3 world, const TunnelState& state, const std::vector<WakeBlob>& wake) {
    Vector3 local = ToLocal(world, state);
    float distance = BodySignedDistanceLocal(local, state);
    if (distance <= 0.0f) return {0.0f, 0.0f, 0.0f};

    Vector3 velocity{state.windSpeed, 0.0f, 0.0f};

    float shell = 0.85f * state.bodyScale;
    if (distance < shell && local.x > -3.4f * state.bodyScale && local.x < 3.2f * state.bodyScale) {
        Vector3 normal = SurfaceNormalLocal(local, state);
        Vector3 tangent = Vector3Subtract(velocity, Vector3Scale(normal, Vector3DotProduct(velocity, normal)));
        if (Vector3Length(tangent) < 0.001f) tangent = {state.windSpeed, 0.0f, 0.0f};
        tangent = Vector3Normalize(tangent);

        float proximity = 1.0f - distance / shell;
        float sideSpeedup = 1.0f + 0.35f * proximity + 0.10f * std::fabs(normal.z);
        Vector3 guided = Vector3Scale(tangent, state.windSpeed * sideSpeedup);
        velocity = Vector3Lerp(velocity, guided, 0.88f * proximity);
        velocity = Vector3Add(velocity, Vector3Scale(normal, state.windSpeed * 0.08f * proximity * std::max(0.0f, -normal.x)));
    }

    float frontEnv = std::exp(-0.95f * (local.x + 2.65f * state.bodyScale) * (local.x + 2.65f * state.bodyScale) -
                              0.55f * local.y * local.y - 0.45f * local.z * local.z);
    velocity.x -= state.windSpeed * 0.58f * frontEnv;
    velocity.y += local.y * state.windSpeed * 0.10f * frontEnv;
    velocity.z += local.z * state.windSpeed * 0.10f * frontEnv;

    float roofEnv = std::exp(-0.18f * (local.x + 0.15f) * (local.x + 0.15f) -
                             1.25f * (local.y - 1.10f * state.bodyScale) * (local.y - 1.10f * state.bodyScale) -
                             0.45f * local.z * local.z);
    velocity.x += state.windSpeed * 0.18f * roofEnv;

    float floorEnv = std::exp(-0.12f * local.x * local.x -
                              2.30f * (local.y - 0.08f * state.bodyScale) * (local.y - 0.08f * state.bodyScale) -
                              0.60f * local.z * local.z);
    velocity.x += state.windSpeed * 0.10f * floorEnv;

    velocity = Vector3Add(velocity, ComputeWakeVelocity(world, state, wake));
    return velocity;
}

float AeroSurfaceScore(Vector3 world, const TunnelState& state, const std::vector<WakeBlob>& wake) {
    Vector3 local = ToLocal(world, state);
    Vector3 normal = SurfaceNormalLocal(local, state);
    Vector3 samplePoint = Vector3Add(world, Vector3Scale(normal, 0.18f * state.bodyScale));
    float speed = Vector3Length(ComputeFlowVelocity(samplePoint, state, wake));
    float speedRatio = speed / std::max(0.001f, state.windSpeed);
    return Saturate((speedRatio - 0.45f) / 0.85f);
}

void ResetParticle(FlowParticle* particle) {
    particle->pos = {-11.6f - RandomFloat(0.0f, 2.6f), RandomFloat(0.05f, 2.85f), RandomFloat(-3.6f, 3.6f)};
    particle->trail.clear();
    particle->trail.push_back(particle->pos);
    particle->age = 0.0f;
    particle->life = RandomFloat(5.2f, 8.8f);
    particle->lane = Saturate((particle->pos.y - 0.05f) / 2.8f);
}

void ResetParticles(std::vector<FlowParticle>* particles) {
    for (FlowParticle& particle : *particles) ResetParticle(&particle);
}

void UpdateParticles(std::vector<FlowParticle>* particles, const TunnelState& state, const std::vector<WakeBlob>& wake, float dt) {
    for (FlowParticle& particle : *particles) {
        particle.age += dt;
        Vector3 velocity = ComputeFlowVelocity(particle.pos, state, wake);
        particle.pos = Vector3Add(particle.pos, Vector3Scale(velocity, dt * 0.22f));

        bool outOfBounds = particle.pos.x > 14.0f || particle.pos.x < -14.5f || particle.pos.y < -0.5f || particle.pos.y > 5.2f ||
                           std::fabs(particle.pos.z) > 5.2f;
        if (InsideBody(particle.pos, state) || outOfBounds || particle.age > particle.life) {
            ResetParticle(&particle);
            continue;
        }

        particle.trail.push_back(particle.pos);
        if (particle.trail.size() > 42) particle.trail.pop_front();
    }
}

void SpawnWakeBlobs(std::vector<WakeBlob>* wake, const TunnelState& state) {
    for (int i = 0; i < 3; ++i) {
        WakeBlob blob;
        blob.pos = {state.bodyCenter.x + 2.55f * state.bodyScale + RandomFloat(0.05f, 0.35f),
                    state.bodyCenter.y + RandomFloat(0.30f, 1.05f) * state.bodyScale,
                    RandomFloat(-0.95f, 0.95f) * state.bodyScale};
        blob.radius = RandomFloat(0.16f, 0.38f) * state.bodyScale;
        blob.strength = RandomFloat(-1.0f, 1.0f) * state.windSpeed * 0.18f;
        wake->push_back(blob);
    }
}

void UpdateWake(std::vector<WakeBlob>* wake, const TunnelState& state, float dt) {
    for (WakeBlob& blob : *wake) {
        blob.age += dt;
        blob.pos.x += state.windSpeed * 0.16f * dt;
        blob.pos.y += 0.08f * std::sin(blob.age * 1.7f + blob.pos.z) * dt;
        blob.pos.z += 0.12f * std::cos(blob.age * 1.2f + blob.pos.y) * dt;
        blob.radius = std::min(blob.radius + 0.07f * dt, 0.95f * state.bodyScale);
        blob.strength *= (1.0f - 0.18f * dt);
    }

    wake->erase(std::remove_if(wake->begin(), wake->end(),
                               [](const WakeBlob& blob) { return blob.age > 7.0f || blob.pos.x > 13.5f; }),
                wake->end());
}

void DrawTunnel(float time) {
    DrawCube({0.4f, -0.10f, 0.0f}, 30.0f, 0.18f, 12.5f, Color{22, 28, 40, 255});
    DrawCubeWires({0.4f, -0.10f, 0.0f}, 30.0f, 0.18f, 12.5f, Fade(Color{93, 108, 132, 255}, 0.35f));

    for (int i = 0; i < 17; ++i) {
        float x = -12.2f + static_cast<float>(i) * 1.55f;
        Color frame = Fade(Color{84, 124, 176, 255}, 0.12f + 0.18f * std::sin(time * 1.9f + static_cast<float>(i) * 0.33f));
        DrawLine3D({x, 0.0f, -4.2f}, {x, 3.8f, -4.2f}, frame);
        DrawLine3D({x, 3.8f, -4.2f}, {x, 3.8f, 4.2f}, frame);
        DrawLine3D({x, 3.8f, 4.2f}, {x, 0.0f, 4.2f}, frame);
        DrawLine3D({x, 0.0f, 4.2f}, {x, 0.0f, -4.2f}, frame);
    }

    for (int row = 0; row < 3; ++row) {
        float y = 0.45f + static_cast<float>(row) * 1.35f;
        for (int i = 0; i < 20; ++i) {
            float x = -11.5f + static_cast<float>(i) * 1.20f;
            float phase = std::fmod(time * 1.7f + static_cast<float>(i) * 0.17f + static_cast<float>(row) * 0.3f, 1.0f);
            DrawLine3D({x + phase * 0.9f, y, -3.6f}, {x + 0.55f + phase * 0.9f, y, -3.6f}, Fade(Color{107, 220, 255, 255}, 0.08f));
            DrawLine3D({x + phase * 0.9f, y, 3.6f}, {x + 0.55f + phase * 0.9f, y, 3.6f}, Fade(Color{107, 220, 255, 255}, 0.08f));
        }
    }
}

void DrawWindField(float time) {
    for (int iy = 0; iy < 5; ++iy) {
        for (int iz = 0; iz < 5; ++iz) {
            float y = 0.35f + static_cast<float>(iy) * 0.75f;
            float z = -3.3f + static_cast<float>(iz) * 1.65f;
            for (int ix = 0; ix < 8; ++ix) {
                float x = -11.8f + static_cast<float>(ix) * 2.8f;
                float phase = std::fmod(time * 1.8f + static_cast<float>(ix) * 0.14f + static_cast<float>(iy + iz) * 0.08f, 1.0f);
                Vector3 start{x + phase * 1.2f, y, z};
                Vector3 end{x + 0.65f + phase * 1.2f, y, z};
                DrawLine3D(start, end, Fade(Color{102, 217, 255, 255}, 0.12f));
            }
        }
    }
}

Vector3 BodySurfacePoint(float x, float angle, const TunnelState& state) {
    float yMid = SectionMidY(x, state) - state.bodyCenter.y;
    float halfHeight = SectionHalfHeight(x, state);
    float halfWidth = HalfWidthProfile(x, state);
    Vector3 local{x, yMid + halfHeight * std::cos(angle), halfWidth * std::sin(angle)};
    return ToWorld(local, state);
}

void DrawBody(const TunnelState& state, const std::vector<WakeBlob>& wake) {
    const int xSegments = 44;
    const int ringSegments = 28;

    for (int ix = 0; ix < xSegments; ++ix) {
        float x0 = Lerp(-kHalfLength, kHalfLength, static_cast<float>(ix) / xSegments);
        float x1 = Lerp(-kHalfLength, kHalfLength, static_cast<float>(ix + 1) / xSegments);
        for (int ir = 0; ir < ringSegments; ++ir) {
            float a0 = 2.0f * kPi * static_cast<float>(ir) / ringSegments;
            float a1 = 2.0f * kPi * static_cast<float>(ir + 1) / ringSegments;

            Vector3 p00 = BodySurfacePoint(x0, a0, state);
            Vector3 p10 = BodySurfacePoint(x1, a0, state);
            Vector3 p01 = BodySurfacePoint(x0, a1, state);
            Vector3 p11 = BodySurfacePoint(x1, a1, state);

            float scoreA = 0.25f * (AeroSurfaceScore(p00, state, wake) + AeroSurfaceScore(p10, state, wake) +
                                    AeroSurfaceScore(p01, state, wake) + AeroSurfaceScore(p11, state, wake));
            Color color = AeroSurfaceColor(scoreA);
            DrawTriangle3D(p00, p10, p11, color);
            DrawTriangle3D(p00, p11, p01, color);
        }
    }

    for (int ix = 0; ix <= xSegments; ix += 2) {
        float x = Lerp(-kHalfLength, kHalfLength, static_cast<float>(ix) / xSegments);
        for (int ir = 0; ir < ringSegments; ir += 2) {
            float a0 = 2.0f * kPi * static_cast<float>(ir) / ringSegments;
            float a1 = 2.0f * kPi * static_cast<float>(ir + 1) / ringSegments;
            DrawLine3D(BodySurfacePoint(x, a0, state), BodySurfacePoint(x, a1, state), Fade(Color{255, 255, 255, 255}, 0.12f));
        }
    }
}

void DrawParticles(const std::vector<FlowParticle>& particles, const TunnelState& state, const std::vector<WakeBlob>& wake) {
    for (const FlowParticle& particle : particles) {
        float speedRatio = Vector3Length(ComputeFlowVelocity(particle.pos, state, wake)) / std::max(0.001f, state.windSpeed);
        Color base = StreamColor(speedRatio);
        for (size_t i = 1; i < particle.trail.size(); ++i) {
            float alpha = static_cast<float>(i) / static_cast<float>(particle.trail.size());
            DrawLine3D(particle.trail[i - 1], particle.trail[i], Fade(base, 0.05f + 0.58f * alpha));
        }
        DrawSphere(particle.pos, 0.024f, Fade(base, 0.88f));
    }
}

void DrawWakeRibbons(const std::vector<WakeBlob>& wake, float time) {
    for (const WakeBlob& blob : wake) {
        Color tint = blob.strength >= 0.0f ? Color{103, 216, 255, 255} : Color{255, 154, 92, 255};
        Vector3 previous{};
        bool hasPrevious = false;
        for (int i = 0; i < 16; ++i) {
            float x = blob.pos.x - static_cast<float>(i) * 0.18f;
            float radius = blob.radius + 0.02f * i;
            float angle = time * 2.5f + static_cast<float>(i) * 0.45f * (blob.strength >= 0.0f ? 1.0f : -1.0f);
            Vector3 point{x, blob.pos.y + std::cos(angle) * radius, blob.pos.z + std::sin(angle) * radius};
            if (hasPrevious) DrawLine3D(previous, point, Fade(tint, 0.04f + 0.36f * (1.0f - static_cast<float>(i) / 16.0f)));
            previous = point;
            hasPrevious = true;
        }
    }
}

void DrawMinimalOverlay() {
    DrawRectangleRounded({18.0f, 16.0f, 360.0f, 54.0f}, 0.16f, 10, Fade(Color{8, 12, 20, 255}, 0.76f));
    DrawText("Wind Tunnel Aero View", 34, 28, 26, Color{234, 239, 245, 255});

    DrawRectangleRounded({18.0f, static_cast<float>(kScreenHeight - 54), 560.0f, 34.0f}, 0.16f, 10, Fade(Color{8, 12, 20, 255}, 0.72f));
    DrawText("Mouse drag: orbit   Wheel: zoom   Left/Right: wind   Up/Down: body scale   [ ]: roofline   P: pause   R: reset", 30,
             kScreenHeight - 45, 18, Color{189, 203, 221, 255});
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Aerodynamics Wind Tunnel View");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {10.2f, 4.9f, 10.4f};
    camera.target = {1.0f, 1.1f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 42.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.83f;
    float camPitch = 0.24f;
    float camDistance = 16.8f;

    TunnelState state;
    std::vector<FlowParticle> particles(480);
    ResetParticles(&particles);

    std::vector<WakeBlob> wake;
    float wakeTimer = 0.0f;
    bool paused = false;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        float time = static_cast<float>(GetTime());

        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            paused = false;
            state.windSpeed = 19.0f;
            state.bodyScale = 1.0f;
            state.roofBias = 0.0f;
            wake.clear();
            ResetParticles(&particles);
            wakeTimer = 0.0f;
        }

        if (IsKeyDown(KEY_LEFT)) state.windSpeed = std::max(6.0f, state.windSpeed - 14.0f * dt);
        if (IsKeyDown(KEY_RIGHT)) state.windSpeed = std::min(42.0f, state.windSpeed + 14.0f * dt);
        if (IsKeyDown(KEY_DOWN)) state.bodyScale = std::max(0.72f, state.bodyScale - 0.45f * dt);
        if (IsKeyDown(KEY_UP)) state.bodyScale = std::min(1.35f, state.bodyScale + 0.45f * dt);
        if (IsKeyDown(KEY_LEFT_BRACKET)) state.roofBias = std::max(-1.0f, state.roofBias - 0.90f * dt);
        if (IsKeyDown(KEY_RIGHT_BRACKET)) state.roofBias = std::min(1.0f, state.roofBias + 0.90f * dt);

        state.wakeStrength = 0.62f + 0.16f * state.bodyScale + 0.12f * std::fabs(state.roofBias);
        UpdateOrbitCamera(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            wakeTimer += dt;
            if (wakeTimer >= 0.07f) {
                SpawnWakeBlobs(&wake, state);
                wakeTimer = 0.0f;
            }
            UpdateWake(&wake, state, dt);
            UpdateParticles(&particles, state, wake, dt);
        }

        BeginDrawing();
        ClearBackground(Color{4, 8, 14, 255});
        BeginMode3D(camera);

        DrawTunnel(time);
        DrawGrid(28, 1.0f);
        DrawWindField(time);
        DrawWakeRibbons(wake, time);
        DrawBody(state, wake);
        DrawParticles(particles, state, wake);

        EndMode3D();

        DrawMinimalOverlay();
        DrawFPS(kScreenWidth - 96, 18);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
