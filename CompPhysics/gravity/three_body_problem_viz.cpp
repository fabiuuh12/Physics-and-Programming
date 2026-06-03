#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;

constexpr float kG = 14.0f;
constexpr float kSoftening = 0.18f;
constexpr int kTrailMax = 1800;
constexpr float kFixedStep = 1.0f / 240.0f;
constexpr float kPi = 3.14159265358979323846f;

struct Body {
    const char* name;
    float mass;
    float radius;
    Vector3 pos;
    Vector3 vel;
    Color color;
};

struct Preset {
    const char* name;
    std::array<Body, 3> bodies;
    float suggestedDistance;
    const char* description;
};

struct Derivative {
    std::array<Vector3, 3> dPos{};
    std::array<Vector3, 3> dVel{};
};

struct State {
    std::array<Vector3, 3> pos{};
    std::array<Vector3, 3> vel{};
};

Vector3 ComputeAcceleration(
    const std::array<Vector3, 3>& positions,
    const std::array<float, 3>& masses,
    int idx
) {
    Vector3 acc = {0.0f, 0.0f, 0.0f};
    for (int other = 0; other < 3; ++other) {
        if (other == idx) {
            continue;
        }
        Vector3 delta = Vector3Subtract(positions[other], positions[idx]);
        float dist2 = Vector3LengthSqr(delta) + kSoftening * kSoftening;
        float invDist = 1.0f / std::sqrt(dist2);
        float invDist3 = invDist * invDist * invDist;
        acc = Vector3Add(acc, Vector3Scale(delta, kG * masses[other] * invDist3));
    }
    return acc;
}

Derivative EvaluateDerivative(const State& state, const std::array<float, 3>& masses) {
    Derivative deriv{};
    for (int i = 0; i < 3; ++i) {
        deriv.dPos[i] = state.vel[i];
        deriv.dVel[i] = ComputeAcceleration(state.pos, masses, i);
    }
    return deriv;
}

State AdvanceState(const State& state, const Derivative& deriv, float dt) {
    State next = state;
    for (int i = 0; i < 3; ++i) {
        next.pos[i] = Vector3Add(state.pos[i], Vector3Scale(deriv.dPos[i], dt));
        next.vel[i] = Vector3Add(state.vel[i], Vector3Scale(deriv.dVel[i], dt));
    }
    return next;
}

void StepRK4(std::array<Body, 3>* bodies, float dt) {
    State start{};
    std::array<float, 3> masses{};
    for (int i = 0; i < 3; ++i) {
        start.pos[i] = (*bodies)[i].pos;
        start.vel[i] = (*bodies)[i].vel;
        masses[i] = (*bodies)[i].mass;
    }

    const Derivative k1 = EvaluateDerivative(start, masses);
    const Derivative k2 = EvaluateDerivative(AdvanceState(start, k1, dt * 0.5f), masses);
    const Derivative k3 = EvaluateDerivative(AdvanceState(start, k2, dt * 0.5f), masses);
    const Derivative k4 = EvaluateDerivative(AdvanceState(start, k3, dt), masses);

    for (int i = 0; i < 3; ++i) {
        Vector3 posDelta = Vector3Add(k1.dPos[i], Vector3Scale(k2.dPos[i], 2.0f));
        posDelta = Vector3Add(posDelta, Vector3Scale(k3.dPos[i], 2.0f));
        posDelta = Vector3Add(posDelta, k4.dPos[i]);

        Vector3 velDelta = Vector3Add(k1.dVel[i], Vector3Scale(k2.dVel[i], 2.0f));
        velDelta = Vector3Add(velDelta, Vector3Scale(k3.dVel[i], 2.0f));
        velDelta = Vector3Add(velDelta, k4.dVel[i]);

        (*bodies)[i].pos = Vector3Add((*bodies)[i].pos, Vector3Scale(posDelta, dt / 6.0f));
        (*bodies)[i].vel = Vector3Add((*bodies)[i].vel, Vector3Scale(velDelta, dt / 6.0f));
    }
}

float TotalMass(const std::array<Body, 3>& bodies) {
    return bodies[0].mass + bodies[1].mass + bodies[2].mass;
}

Vector3 ComputeBarycenter(const std::array<Body, 3>& bodies) {
    Vector3 weighted = {0.0f, 0.0f, 0.0f};
    float totalMass = TotalMass(bodies);
    for (const Body& body : bodies) {
        weighted = Vector3Add(weighted, Vector3Scale(body.pos, body.mass));
    }
    return Vector3Scale(weighted, 1.0f / totalMass);
}

Vector3 ComputeLinearMomentum(const std::array<Body, 3>& bodies) {
    Vector3 momentum = {0.0f, 0.0f, 0.0f};
    for (const Body& body : bodies) {
        momentum = Vector3Add(momentum, Vector3Scale(body.vel, body.mass));
    }
    return momentum;
}

float TotalEnergy(const std::array<Body, 3>& bodies) {
    float kinetic = 0.0f;
    float potential = 0.0f;

    for (const Body& body : bodies) {
        kinetic += 0.5f * body.mass * Vector3LengthSqr(body.vel);
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            float dist = std::sqrt(Vector3DistanceSqr(bodies[i].pos, bodies[j].pos) + kSoftening * kSoftening);
            potential -= kG * bodies[i].mass * bodies[j].mass / dist;
        }
    }

    return kinetic + potential;
}

float TotalAngularMomentum(const std::array<Body, 3>& bodies) {
    Vector3 angular = {0.0f, 0.0f, 0.0f};
    for (const Body& body : bodies) {
        angular = Vector3Add(angular, Vector3CrossProduct(body.pos, Vector3Scale(body.vel, body.mass)));
    }
    return Vector3Length(angular);
}

Preset MakeLagrangePreset() {
    constexpr float radius = 4.2f;
    constexpr float mass = 3.4f;
    constexpr float orbitalSpeed = 2.45f;

    std::array<Body, 3> bodies{};
    for (int i = 0; i < 3; ++i) {
        float angle = (2.0f * kPi * static_cast<float>(i)) / 3.0f;
        Vector3 pos = {radius * std::cos(angle), 0.75f * std::sin(angle), radius * std::sin(angle)};
        Vector3 radial = Vector3Normalize(pos);
        Vector3 axis = Vector3Normalize(Vector3{0.0f, 1.0f, 0.18f});
        Vector3 tangent = Vector3Normalize(Vector3CrossProduct(axis, radial));
        bodies[i] = {
            i == 0 ? "Aurelia" : (i == 1 ? "Cerulean" : "Rose"),
            mass,
            0.42f,
            pos,
            Vector3Scale(tangent, orbitalSpeed),
            i == 0 ? Color{255, 196, 104, 255} : (i == 1 ? Color{112, 214, 255, 255} : Color{255, 124, 168, 255}),
        };
    }

    return {
        "Lagrange Triangle",
        bodies,
        18.0f,
        "stable rotating equilateral choreography in 3D view"
    };
}

Preset MakeBraidedChaosPreset() {
    std::array<Body, 3> bodies{{
        {"Alpha", 4.2f, 0.45f, {-4.8f, 1.1f, -1.6f}, {0.58f, 0.18f, 1.24f}, Color{255, 200, 120, 255}},
        {"Beta", 3.6f, 0.40f, {4.5f, -0.8f, 1.8f}, {-0.82f, 0.36f, -1.08f}, Color{118, 225, 255, 255}},
        {"Gamma", 2.4f, 0.33f, {0.2f, 0.9f, -5.2f}, {0.36f, -0.92f, 0.41f}, Color{188, 132, 255, 255}},
    }};

    return {
        "Braided Chaos",
        bodies,
        22.0f,
        "generic 3-body interaction with strong out-of-plane motion"
    };
}

Preset MakeBinaryIntruderPreset() {
    std::array<Body, 3> bodies{{
        {"Primary", 6.5f, 0.50f, {-1.9f, 0.0f, 0.0f}, {0.0f, 0.32f, 1.58f}, Color{255, 194, 110, 255}},
        {"Companion", 5.2f, 0.46f, {1.9f, 0.0f, 0.0f}, {0.0f, -0.28f, -1.92f}, Color{120, 213, 255, 255}},
        {"Intruder", 1.5f, 0.28f, {0.0f, 7.0f, -6.5f}, {-0.10f, -1.88f, 1.65f}, Color{255, 118, 166, 255}},
    }};

    return {
        "Binary Capture",
        bodies,
        24.0f,
        "close binary disturbed by a lighter incoming body"
    };
}

std::array<Preset, 3> BuildPresets() {
    return {MakeLagrangePreset(), MakeBraidedChaosPreset(), MakeBinaryIntruderPreset()};
}

void AppendTrails(
    const std::array<Body, 3>& bodies,
    std::array<std::deque<Vector3>, 3>* trails
) {
    for (int i = 0; i < 3; ++i) {
        (*trails)[i].push_back(bodies[i].pos);
        if (static_cast<int>((*trails)[i].size()) > kTrailMax) {
            (*trails)[i].pop_front();
        }
    }
}

void ResetSimulation(
    const Preset& preset,
    std::array<Body, 3>* bodies,
    std::array<std::deque<Vector3>, 3>* trails,
    float* simTime
) {
    *bodies = preset.bodies;
    for (std::deque<Vector3>& trail : *trails) {
        trail.clear();
    }
    AppendTrails(*bodies, trails);
    *simTime = 0.0f;
}

void DrawTrail(const std::deque<Vector3>& trail, Color color) {
    if (trail.size() < 2) {
        return;
    }

    for (size_t i = 1; i < trail.size(); ++i) {
        float alpha = static_cast<float>(i) / static_cast<float>(trail.size());
        Color segColor = color;
        segColor.a = static_cast<unsigned char>(18 + 170 * alpha);
        DrawLine3D(trail[i - 1], trail[i], segColor);
    }
}

void DrawStarfield(const std::vector<Vector3>& stars) {
    for (size_t i = 0; i < stars.size(); ++i) {
        unsigned char alpha = static_cast<unsigned char>(120 + (i % 120));
        DrawPoint3D(stars[i], Color{220, 232, 255, alpha});
    }
}

void UpdateOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance, Vector3 target) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0034f;
        *pitch += delta.y * 0.0030f;
        *pitch = std::clamp(*pitch, -1.42f, 1.42f);
    }

    *distance -= GetMouseWheelMove() * 1.4f;
    *distance = std::clamp(*distance, 6.0f, 70.0f);

    camera->target = Vector3Lerp(camera->target, target, 0.09f);
    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

std::vector<Vector3> BuildStarfield() {
    std::vector<Vector3> stars;
    stars.reserve(360);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> azimuthDist(0.0f, 2.0f * kPi);
    std::uniform_real_distribution<float> heightDist(-0.9f, 0.9f);
    std::uniform_real_distribution<float> radiusDist(42.0f, 70.0f);

    for (int i = 0; i < 360; ++i) {
        float azimuth = azimuthDist(rng);
        float y = heightDist(rng);
        float ring = std::sqrt(std::max(0.0f, 1.0f - y * y));
        float radius = radiusDist(rng);
        stars.push_back({
            radius * ring * std::cos(azimuth),
            radius * y,
            radius * ring * std::sin(azimuth),
        });
    }

    return stars;
}

Color Brighten(Color color, float amount) {
    amount = std::clamp(amount, 0.0f, 1.0f);
    Color result = color;
    result.r = static_cast<unsigned char>(color.r + (255 - color.r) * amount);
    result.g = static_cast<unsigned char>(color.g + (255 - color.g) * amount);
    result.b = static_cast<unsigned char>(color.b + (255 - color.b) * amount);
    result.a = 255;
    return result;
}

void DrawBodyWithGlow(const Body& body) {
    Color shell = Brighten(body.color, 0.18f);
    Color core = Brighten(body.color, 0.38f);
    Color rim = Brighten(body.color, 0.60f);

    DrawSphere(body.pos, body.radius * 1.10f, shell);
    DrawSphere(body.pos, body.radius * 0.92f, core);
    DrawSphereWires(body.pos, body.radius * 1.12f, 14, 14, rim);
}

void DrawVelocityVector(const Body& body) {
    Vector3 tip = Vector3Add(body.pos, Vector3Scale(body.vel, 0.65f));
    DrawLine3D(body.pos, tip, Fade(body.color, 0.85f));
    DrawSphere(tip, body.radius * 0.18f, Fade(body.color, 0.92f));
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Three Body Problem 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {18.0f, 10.0f, 18.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 42.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.85f;
    float camPitch = 0.45f;
    float camDistance = 18.0f;

    const std::array<Preset, 3> presets = BuildPresets();
    int presetIndex = 0;
    std::array<Body, 3> bodies{};
    std::array<std::deque<Vector3>, 3> trails;
    float simTime = 0.0f;
    ResetSimulation(presets[presetIndex], &bodies, &trails, &simTime);

    float speed = 1.0f;
    bool paused = false;
    bool showTrails = true;
    bool showVectors = false;
    bool showBarycenter = true;

    const std::vector<Vector3> stars = BuildStarfield();

    while (!WindowShouldClose()) {
        int requestedPreset = presetIndex;
        if (IsKeyPressed(KEY_ONE)) requestedPreset = 0;
        if (IsKeyPressed(KEY_TWO)) requestedPreset = 1;
        if (IsKeyPressed(KEY_THREE)) requestedPreset = 2;

        if (requestedPreset != presetIndex) {
            presetIndex = requestedPreset;
            camDistance = presets[presetIndex].suggestedDistance;
            ResetSimulation(presets[presetIndex], &bodies, &trails, &simTime);
        }

        if (IsKeyPressed(KEY_R)) {
            ResetSimulation(presets[presetIndex], &bodies, &trails, &simTime);
        }
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_T)) showTrails = !showTrails;
        if (IsKeyPressed(KEY_V)) showVectors = !showVectors;
        if (IsKeyPressed(KEY_B)) showBarycenter = !showBarycenter;
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) speed = std::min(6.0f, speed + 0.25f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) speed = std::max(0.25f, speed - 0.25f);

        Vector3 barycenter = ComputeBarycenter(bodies);
        UpdateOrbitCamera(&camera, &camYaw, &camPitch, &camDistance, barycenter);

        if (!paused) {
            float frameAdvance = GetFrameTime() * speed;
            int steps = std::max(1, static_cast<int>(std::ceil(frameAdvance / kFixedStep)));
            float dt = frameAdvance / static_cast<float>(steps);
            for (int i = 0; i < steps; ++i) {
                StepRK4(&bodies, dt);
                simTime += dt;
            }
            AppendTrails(bodies, &trails);
            barycenter = ComputeBarycenter(bodies);
        }

        BeginDrawing();
        ClearBackground(Color{4, 6, 14, 255});

        DrawRectangleGradientV(0, 0, kScreenWidth, kScreenHeight, Color{7, 11, 24, 255}, Color{2, 3, 8, 255});

        BeginMode3D(camera);

        DrawStarfield(stars);

        if (showTrails) {
            for (int i = 0; i < 3; ++i) {
                DrawTrail(trails[i], bodies[i].color);
            }
        }

        if (showBarycenter) {
            DrawSphere(ComputeBarycenter(bodies), 0.14f, Color{245, 245, 255, 235});
        }

        for (const Body& body : bodies) {
            DrawBodyWithGlow(body);
            if (showVectors) {
                DrawVelocityVector(body);
            }
        }

        EndMode3D();

        DrawText("Three Body Problem", 20, 18, 34, Color{236, 241, 248, 255});
        DrawText(presets[presetIndex].description, 20, 58, 18, Color{145, 189, 231, 255});
        DrawText("Mouse orbit | wheel zoom | 1/2/3 presets | P pause | R reset | +/- speed | T trails | V vectors | B barycenter",
                 20, 84, 18, Color{166, 186, 212, 255});

        char status[256];
        std::snprintf(
            status,
            sizeof(status),
            "Preset: %s   t=%.2f   speed=%.2fx   energy=%.3f   |L|=%.3f%s",
            presets[presetIndex].name,
            simTime,
            speed,
            TotalEnergy(bodies),
            TotalAngularMomentum(bodies),
            paused ? "   [PAUSED]" : ""
        );
        DrawText(status, 20, 112, 20, Color{126, 224, 255, 255});

        Vector3 momentum = ComputeLinearMomentum(bodies);
        char metrics[220];
        std::snprintf(
            metrics,
            sizeof(metrics),
            "Barycenter=(%.2f, %.2f, %.2f)   |P|=%.3f",
            ComputeBarycenter(bodies).x,
            ComputeBarycenter(bodies).y,
            ComputeBarycenter(bodies).z,
            Vector3Length(momentum)
        );
        DrawText(metrics, 20, 138, 18, Color{199, 216, 238, 255});

        int legendY = 176;
        for (const Body& body : bodies) {
            DrawCircle(28, legendY + 10, 7.0f, body.color);
            char bodyLine[180];
            std::snprintf(
                bodyLine,
                sizeof(bodyLine),
                "%s  m=%.1f  pos=(%.1f, %.1f, %.1f)",
                body.name,
                body.mass,
                body.pos.x,
                body.pos.y,
                body.pos.z
            );
            DrawText(bodyLine, 44, legendY, 18, Color{226, 232, 242, 255});
            legendY += 26;
        }

        DrawFPS(20, legendY + 6);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
