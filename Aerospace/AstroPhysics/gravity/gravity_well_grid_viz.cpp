#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr float kBaseGridExtent = 8.0f;
constexpr int kLines = 7;
constexpr int kSegments = 58;
constexpr float kFlowRepeats = 4.25f;
constexpr int kTrailLimit = 520;
constexpr float kG = 1.45f;
constexpr float kLightSpeed = 5.4f;
constexpr float kPulsarMinMass = 6.0f;
constexpr float kBlackHoleMinMass = 8.0f;

struct MassObject {
    Vector3 pos;
    Vector3 vel;
    float mass;
    float radius;
    Color color;
    std::string label;
    bool blackHole;
    bool pulsar;
    std::deque<Vector3> trail;
};

struct CollisionEvent {
    Vector3 pos;
    float age;
    float strength;
};

struct ExplosionParticle {
    Vector3 pos;
    Vector3 vel;
    float age;
    float life;
    Color color;
};

Vector3 CenterOfMass(const std::vector<MassObject>& masses);

std::vector<Color> BodyPalette() {
    return {
        WHITE,
        Color{255, 210, 120, 255},
        Color{125, 215, 255, 255},
        Color{190, 255, 170, 255},
        Color{255, 125, 170, 255},
        Color{185, 150, 255, 255},
    };
}

std::string LabelForIndex(int index) {
    std::ostringstream os;
    os << "body " << (index + 1);
    return os.str();
}

void UpdateOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::fmod(*pitch, 2.0f * PI);
    }

    *distance -= GetMouseWheelMove() * 1.15f;
    *distance = std::clamp(*distance, 0.75f, 250.0f);

    float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    });
    camera->up = std::cos(*pitch) >= 0.0f ? Vector3{0.0f, 1.0f, 0.0f} : Vector3{0.0f, -1.0f, 0.0f};
}

Color WithAlpha(Color c, unsigned char alpha) {
    c.a = alpha;
    return c;
}

float Coordinate(int i, float extent) {
    return -extent + 2.0f * extent * static_cast<float>(i) / static_cast<float>(kLines - 1);
}

float SchwarzschildRadius(float mass) {
    return 2.0f * kG * mass / (kLightSpeed * kLightSpeed);
}

float EffectiveRadius(const MassObject& body) {
    return body.blackHole ? std::max(body.radius, SchwarzschildRadius(body.mass)) : body.radius;
}

Vector3 WarpOffset(Vector3 p, const std::vector<MassObject>& masses, const std::vector<CollisionEvent>& collisions, float core, float pulse) {
    Vector3 offset = {0.0f, 0.0f, 0.0f};
    for (const MassObject& body : masses) {
        Vector3 toMass = Vector3Subtract(body.pos, p);
        float r = Vector3Length(toMass);
        if (r < 0.001f) continue;

        Vector3 dir = Vector3Scale(toMass, 1.0f / r);
        float rs = SchwarzschildRadius(body.mass);
        float compactness = std::clamp(rs / std::max(r, rs + 0.08f), 0.0f, body.blackHole ? 0.96f : 0.86f);
        float pull = body.mass / (r * r + core * core) * (1.0f + (body.blackHole ? 5.2f : 3.2f) * compactness);
        pull = std::min(pull, r * 0.76f);
        float wave = 0.075f * std::sin(4.6f * r - pulse) * std::exp(-0.18f * r);
        offset = Vector3Add(offset, Vector3Scale(dir, pull + wave));
    }

    for (const CollisionEvent& event : collisions) {
        Vector3 fromEvent = Vector3Subtract(p, event.pos);
        float r = Vector3Length(fromEvent);
        if (r < 0.001f) continue;
        Vector3 dir = Vector3Scale(fromEvent, 1.0f / r);
        float shell = event.age * kLightSpeed * 0.85f;
        float profile = std::exp(-3.6f * std::fabs(r - shell));
        float amplitude = 0.34f * event.strength * std::exp(-0.72f * event.age) * profile;
        offset = Vector3Add(offset, Vector3Scale(dir, amplitude));
    }

    float len = Vector3Length(offset);
    if (len > 2.75f) offset = Vector3Scale(Vector3Normalize(offset), 2.75f);
    return offset;
}

Vector3 WarpedPoint(Vector3 p, const std::vector<MassObject>& masses, const std::vector<CollisionEvent>& collisions, float core, float pulse) {
    return Vector3Add(p, WarpOffset(p, masses, collisions, core, pulse));
}

float LocalWarpStrength(Vector3 p, const std::vector<MassObject>& masses, float core) {
    float strength = 0.0f;
    for (const MassObject& body : masses) {
        float r = Vector3Distance(p, body.pos);
        float compactness = SchwarzschildRadius(body.mass) / std::max(r, SchwarzschildRadius(body.mass) + 0.08f);
        strength += body.mass / (r * r + core * core) + (body.blackHole ? 4.6f : 2.8f) * compactness;
    }
    return std::clamp(strength / 2.25f, 0.0f, 1.0f);
}

Color WarpColor(Vector3 p, const std::vector<MassObject>& masses, float core) {
    float near = LocalWarpStrength(p, masses, core);
    return Color{
        static_cast<unsigned char>(0 + 38 * near),
        static_cast<unsigned char>(68 + 170 * near),
        static_cast<unsigned char>(178 + 46 * (1.0f - near)),
        static_cast<unsigned char>(68 + 130 * near),
    };
}

void DrawWarpedLine(Vector3 a, Vector3 b, const std::vector<MassObject>& masses, const std::vector<CollisionEvent>& collisions, float core, float pulse) {
    Vector3 lineDir = Vector3Normalize(Vector3Subtract(b, a));
    Vector3 prevBase = a;
    Vector3 prev = WarpedPoint(a, masses, collisions, core, pulse);
    for (int i = 1; i <= kSegments; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(kSegments);
        Vector3 base = Vector3Lerp(a, b, t);
        Vector3 current = WarpedPoint(base, masses, collisions, core, pulse);
        DrawLine3D(prev, current, WarpColor(current, masses, core));

        Vector3 midBase = Vector3Lerp(prevBase, base, 0.5f);
        for (const MassObject& body : masses) {
            Vector3 toBody = Vector3Subtract(body.pos, midBase);
            float distance = Vector3Length(toBody);
            float influenceRadius = std::clamp(1.20f + std::sqrt(body.mass) * 0.78f + (body.blackHole ? 1.15f : 0.0f), 2.15f, 5.4f);
            if (distance < 0.001f || distance > influenceRadius) continue;

            float alignment = std::fabs(Vector3DotProduct(lineDir, Vector3Scale(toBody, 1.0f / distance)));
            if (alignment < 0.36f) continue;

            float normalizedDistance = distance / influenceRadius;
            float localStrength = (1.0f - normalizedDistance) * alignment;
            float direction = Vector3DotProduct(lineDir, toBody) >= 0.0f ? 1.0f : -1.0f;
            float localCoordinate = Vector3DotProduct(midBase, lineDir) * direction;
            float speed = 0.22f + 0.045f * body.mass;
            float phase = std::fmod(localCoordinate * kFlowRepeats - pulse * speed, 1.0f);
            if (phase < 0.0f) phase += 1.0f;

            if (phase < 0.10f) {
                float head = 1.0f - phase / 0.10f;
                unsigned char alpha = static_cast<unsigned char>(std::clamp(150.0f * localStrength * head, 0.0f, 145.0f));
                Color flowColor = body.blackHole ? Color{255, 205, 90, alpha} : Color{145, 230, 255, alpha};
                DrawLine3D(prev, current, flowColor);
            }
        }

        prevBase = base;
        prev = current;
    }
}

float TargetGridExtent(const std::vector<MassObject>& masses) {
    float required = kBaseGridExtent;
    for (const MassObject& body : masses) {
        float axisExtent = std::max({std::fabs(body.pos.x), std::fabs(body.pos.y), std::fabs(body.pos.z)});
        float speedMargin = std::min(Vector3Length(body.vel) * 0.35f, 8.0f);
        required = std::max(required, axisExtent + EffectiveRadius(body) + 3.0f + speedMargin);
    }
    return std::max(required, kBaseGridExtent);
}

void DrawWarpedCubeGrid(const std::vector<MassObject>& masses, const std::vector<CollisionEvent>& collisions, float core, float pulse, float extent) {
    for (int iy = 0; iy < kLines; ++iy) {
        float y = Coordinate(iy, extent);
        for (int iz = 0; iz < kLines; ++iz) {
            float z = Coordinate(iz, extent);
            DrawWarpedLine({-extent, y, z}, {extent, y, z}, masses, collisions, core, pulse);
        }
    }

    for (int ix = 0; ix < kLines; ++ix) {
        float x = Coordinate(ix, extent);
        for (int iz = 0; iz < kLines; ++iz) {
            float z = Coordinate(iz, extent);
            DrawWarpedLine({x, -extent, z}, {x, extent, z}, masses, collisions, core, pulse);
        }
    }

    for (int ix = 0; ix < kLines; ++ix) {
        float x = Coordinate(ix, extent);
        for (int iy = 0; iy < kLines; ++iy) {
            float y = Coordinate(iy, extent);
            DrawWarpedLine({x, y, -extent}, {x, y, extent}, masses, collisions, core, pulse);
        }
    }
}

void DrawSphericalRipple(Vector3 center, float radius, Color color) {
    DrawCircle3D(center, radius, {0.0f, 1.0f, 0.0f}, 90.0f, color);
    DrawCircle3D(center, radius, {1.0f, 0.0f, 0.0f}, 90.0f, WithAlpha(color, color.a * 3 / 4));
    DrawCircle3D(center, radius, {0.0f, 0.0f, 1.0f}, 90.0f, WithAlpha(color, color.a * 3 / 4));
    DrawCircle3D(center, radius * 0.72f, {0.0f, 1.0f, 0.0f}, 90.0f, WithAlpha(color, color.a / 2));
    DrawCircle3D(center, radius * 0.72f, {1.0f, 0.0f, 0.0f}, 90.0f, WithAlpha(color, color.a / 2));
    DrawCircle3D(center, radius * 0.72f, {0.0f, 0.0f, 1.0f}, 90.0f, WithAlpha(color, color.a / 2));
}

void DrawGravitationalWaveRipples(const std::vector<MassObject>& masses, const std::vector<CollisionEvent>& collisions, float time, bool decayOn) {
    if (masses.size() >= 2) {
        Vector3 center = CenterOfMass(masses);
        float activity = 0.0f;
        for (const MassObject& body : masses) {
            activity += body.mass * Vector3Length(body.vel);
        }
        activity = std::clamp(activity / 16.0f, 0.25f, 1.0f);

        for (int i = 0; i < 7; ++i) {
            float phase = std::fmod(time * 1.55f + static_cast<float>(i) * 2.15f, 13.5f);
            float radius = 1.6f + phase;
            float fade = std::clamp(1.0f - phase / 13.5f, 0.0f, 1.0f);
            unsigned char alpha = static_cast<unsigned char>((decayOn ? 95 : 58) * fade * activity);
            Color c = decayOn ? Color{130, 230, 255, alpha} : Color{90, 165, 230, alpha};
            DrawSphericalRipple(center, radius, c);
        }
    }

    for (const CollisionEvent& event : collisions) {
        float radius = 0.25f + event.age * kLightSpeed * 0.85f;
        float fade = std::clamp(1.0f - event.age / 3.2f, 0.0f, 1.0f);
        unsigned char alpha = static_cast<unsigned char>(190.0f * fade * std::clamp(event.strength, 0.45f, 1.4f));
        DrawSphericalRipple(event.pos, radius, Color{255, 205, 90, alpha});
        DrawSphere(event.pos, 0.18f + 0.18f * fade, Color{255, 245, 190, static_cast<unsigned char>(90 * fade)});
    }
}

void DrawBlackHole(const MassObject& body, bool selected, float time) {
    float rs = SchwarzschildRadius(body.mass);
    float horizon = std::max(body.radius * 1.25f, rs);
    DrawSphere(body.pos, horizon * 0.82f, BLACK);
    DrawSphereWires(body.pos, horizon, 28, 16, selected ? Color{255, 150, 80, 220} : Color{255, 90, 70, 155});
    DrawSphere(body.pos, horizon * 1.34f, Color{255, 120, 35, 32});

    for (int i = 0; i < 4; ++i) {
        float radius = horizon * (1.65f + 0.34f * i);
        unsigned char alpha = static_cast<unsigned char>(120 - 20 * i);
        Color diskColor = i % 2 == 0 ? Color{255, 190, 80, alpha} : Color{120, 215, 255, alpha};
        DrawCircle3D(body.pos, radius, {0.0f, 1.0f, 0.0f}, 90.0f + 8.0f * std::sin(time + i), diskColor);
    }
}

void DrawPulsar(const MassObject& body, bool selected, float time) {
    DrawSphere(body.pos, body.radius * 1.08f, Color{170, 220, 255, 255});
    DrawSphereWires(body.pos, body.radius * (selected ? 2.0f : 1.55f), 22, 12,
                    selected ? Color{120, 230, 255, 190} : Color{210, 245, 255, 120});
    DrawSphere(body.pos, body.radius * 2.35f, Color{85, 180, 255, 38});

    float spin = time * (1.6f + 0.18f * body.mass);
    Vector3 axis = Vector3Normalize({0.46f * std::cos(spin), 0.72f, 0.46f * std::sin(spin)});
    float beamLength = 3.2f + body.mass * 0.34f;
    for (int side = -1; side <= 1; side += 2) {
        Vector3 dir = Vector3Scale(axis, static_cast<float>(side));
        Vector3 end = Vector3Add(body.pos, Vector3Scale(dir, beamLength));
        DrawLine3D(body.pos, end, Color{115, 230, 255, 210});
        DrawLine3D(Vector3Add(body.pos, Vector3Scale(dir, body.radius * 0.8f)), end, Color{255, 255, 255, 115});
        DrawSphere(end, body.radius * 0.24f, Color{150, 235, 255, 135});
    }
}

void SpawnExplosionParticles(std::vector<ExplosionParticle>* particles, Vector3 pos, float strength, Color baseColor) {
    const int count = 34;
    for (int i = 0; i < count; ++i) {
        float fi = static_cast<float>(i);
        float azimuth = fi * 2.399963f;
        float z = -1.0f + 2.0f * (fi + 0.5f) / static_cast<float>(count);
        float ring = std::sqrt(std::max(0.0f, 1.0f - z * z));
        Vector3 dir = {ring * std::cos(azimuth), z, ring * std::sin(azimuth)};
        float speed = (1.6f + 0.09f * static_cast<float>(i % 7)) * std::clamp(strength, 0.55f, 2.2f);
        Color c = i % 3 == 0 ? Color{255, 235, 155, 255} : baseColor;
        particles->push_back({pos, Vector3Scale(dir, speed), 0.0f, 0.78f + 0.025f * static_cast<float>(i % 9), c});
    }
}

void UpdateExplosionParticles(std::vector<ExplosionParticle>* particles, float dt) {
    for (ExplosionParticle& particle : *particles) {
        particle.age += dt;
        particle.vel = Vector3Scale(particle.vel, std::max(0.0f, 1.0f - 1.25f * dt));
        particle.pos = Vector3Add(particle.pos, Vector3Scale(particle.vel, dt));
    }
    particles->erase(std::remove_if(particles->begin(), particles->end(), [](const ExplosionParticle& particle) {
        return particle.age >= particle.life;
    }), particles->end());
}

void DrawExplosionParticles(const std::vector<ExplosionParticle>& particles) {
    for (const ExplosionParticle& particle : particles) {
        float fade = std::clamp(1.0f - particle.age / particle.life, 0.0f, 1.0f);
        Color c = WithAlpha(particle.color, static_cast<unsigned char>(210.0f * fade));
        DrawSphere(particle.pos, 0.035f + 0.08f * fade, c);
    }
}

Vector3 BodyAcceleration(int index, const std::vector<MassObject>& masses, bool relativisticMode) {
    constexpr float kSoftening = 0.76f;
    Vector3 accel = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < static_cast<int>(masses.size()); ++i) {
        if (i == index) continue;
        const MassObject& body = masses[i];
        Vector3 toMass = Vector3Subtract(body.pos, masses[index].pos);
        float r2 = Vector3LengthSqr(toMass) + kSoftening * kSoftening;
        float invR = 1.0f / std::sqrt(r2);
        float correction = 1.0f;
        if (relativisticMode) {
            Vector3 relativeVelocity = Vector3Subtract(masses[index].vel, body.vel);
            float h = Vector3Length(Vector3CrossProduct(toMass, relativeVelocity));
            correction += 3.0f * h * h / (r2 * kLightSpeed * kLightSpeed);
            correction = std::min(correction, 1.75f);
        }
        accel = Vector3Add(accel, Vector3Scale(toMass, kG * body.mass * correction * invR * invR * invR));
    }
    return accel;
}

std::vector<Vector3> BodyAccelerations(const std::vector<MassObject>& masses, bool relativisticMode) {
    std::vector<Vector3> accel(masses.size());
    for (int i = 0; i < static_cast<int>(masses.size()); ++i) {
        accel[i] = BodyAcceleration(i, masses, relativisticMode);
    }
    return accel;
}

Vector3 CenterOfMass(const std::vector<MassObject>& masses) {
    Vector3 weighted = {0.0f, 0.0f, 0.0f};
    float totalMass = 0.0f;
    for (const MassObject& body : masses) {
        weighted = Vector3Add(weighted, Vector3Scale(body.pos, body.mass));
        totalMass += body.mass;
    }
    return totalMass > 0.0f ? Vector3Scale(weighted, 1.0f / totalMass) : Vector3Zero();
}

Vector3 CenterOfMassVelocity(const std::vector<MassObject>& masses) {
    Vector3 weighted = {0.0f, 0.0f, 0.0f};
    float totalMass = 0.0f;
    for (const MassObject& body : masses) {
        weighted = Vector3Add(weighted, Vector3Scale(body.vel, body.mass));
        totalMass += body.mass;
    }
    return totalMass > 0.0f ? Vector3Scale(weighted, 1.0f / totalMass) : Vector3Zero();
}

void RecenterSystem(std::vector<MassObject>* masses) {
    Vector3 com = CenterOfMass(*masses);
    Vector3 comVel = CenterOfMassVelocity(*masses);
    for (MassObject& body : *masses) {
        body.pos = Vector3Subtract(body.pos, com);
        body.vel = Vector3Subtract(body.vel, comVel);
        for (Vector3& point : body.trail) {
            point = Vector3Subtract(point, com);
        }
    }
}

void HandleCollisions(std::vector<MassObject>* masses, std::vector<CollisionEvent>* collisions, std::vector<ExplosionParticle>* particles, int* selected) {
    for (int i = 0; i < static_cast<int>(masses->size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(masses->size()); ++j) {
            MassObject& a = (*masses)[i];
            MassObject& b = (*masses)[j];
            float mergeDistance = (EffectiveRadius(a) + EffectiveRadius(b)) * (a.blackHole || b.blackHole ? 1.05f : 1.18f);
            if (Vector3Distance(a.pos, b.pos) > mergeDistance) continue;

            float totalMass = a.mass + b.mass;
            Vector3 pos = Vector3Scale(Vector3Add(Vector3Scale(a.pos, a.mass), Vector3Scale(b.pos, b.mass)), 1.0f / totalMass);
            Vector3 vel = Vector3Scale(Vector3Add(Vector3Scale(a.vel, a.mass), Vector3Scale(b.vel, b.mass)), 1.0f / totalMass);
            float impact = Vector3Length(Vector3Subtract(a.vel, b.vel));
            bool blackHole = a.blackHole || b.blackHole || SchwarzschildRadius(totalMass) > 0.42f;
            bool pulsar = !blackHole && (a.pulsar || b.pulsar);
            float radius = blackHole ? std::max(0.36f, SchwarzschildRadius(totalMass) * 0.72f)
                                     : std::min(0.55f, std::cbrt(a.radius * a.radius * a.radius + b.radius * b.radius * b.radius) * 1.18f);
            Color mergedColor = blackHole ? Color{20, 20, 24, 255} : (pulsar ? Color{170, 220, 255, 255} : Color{255, 245, 190, 255});
            a = {pos, vel, totalMass, radius, mergedColor, a.label + "+" + b.label, blackHole, pulsar, {}};
            masses->erase(masses->begin() + j);
            float burst = std::clamp(impact * 0.55f + totalMass * 0.08f, 0.5f, 1.8f);
            Vector3 recenteredPos = Vector3Subtract(pos, CenterOfMass(*masses));
            collisions->push_back({recenteredPos, 0.0f, burst});
            SpawnExplosionParticles(particles, recenteredPos, burst, mergedColor);
            *selected = std::min(i, static_cast<int>(masses->size()) - 1);
            RecenterSystem(masses);
            return;
        }
    }
}

void UpdateCollisionEvents(std::vector<CollisionEvent>* collisions, float dt) {
    for (CollisionEvent& event : *collisions) event.age += dt;
    collisions->erase(std::remove_if(collisions->begin(), collisions->end(), [](const CollisionEvent& event) {
        return event.age > 3.4f;
    }), collisions->end());
}

void UpdateMassObjects(std::vector<MassObject>* masses, std::vector<CollisionEvent>* collisions, std::vector<ExplosionParticle>* particles, int* selected, float dt, bool relativisticMode, bool radiationDecay) {
    if (masses->size() < 2) {
        if (!masses->empty()) {
            MassObject& body = (*masses)[0];
            body.trail.push_back(body.pos);
            if (body.trail.size() > kTrailLimit) body.trail.pop_front();
        }
        return;
    }

    std::vector<Vector3> accel = BodyAccelerations(*masses, relativisticMode);
    for (int i = 0; i < static_cast<int>(masses->size()); ++i) {
        MassObject& body = (*masses)[i];
        body.vel = Vector3Add(body.vel, Vector3Scale(accel[i], 0.5f * dt));
        body.pos = Vector3Add(body.pos, Vector3Scale(body.vel, dt));
    }

    std::vector<Vector3> nextAccel = BodyAccelerations(*masses, relativisticMode);
    for (int i = 0; i < static_cast<int>(masses->size()); ++i) {
        MassObject& body = (*masses)[i];
        body.vel = Vector3Add(body.vel, Vector3Scale(nextAccel[i], 0.5f * dt));
        if (radiationDecay && masses->size() >= 2) {
            body.vel = Vector3Scale(body.vel, 1.0f - 0.0065f * dt);
        }

        body.trail.push_back(body.pos);
        if (body.trail.size() > kTrailLimit) body.trail.pop_front();
    }
    RecenterSystem(masses);
    HandleCollisions(masses, collisions, particles, selected);
}

std::vector<MassObject> MakeDefaultMasses() {
    return {
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 3.25f, 0.30f, WHITE, "body 1", false, false, {}},
    };
}

std::vector<MassObject> MakeBinaryMasses() {
    std::vector<Color> colors = BodyPalette();
    float separation = 5.6f;
    float primaryMass = 3.25f;
    float companionMass = 2.35f;
    float totalMass = primaryMass + companionMass;
    float orbitalSpeed = std::sqrt(kG * totalMass / separation);
    float primaryRadius = separation * companionMass / totalMass;
    float companionRadius = separation * primaryMass / totalMass;
    return {
        {{-primaryRadius, 0.0f, 0.0f}, {0.0f, 0.0f, -orbitalSpeed * companionMass / totalMass}, primaryMass, 0.30f, colors[0], "body 1", false, false, {}},
        {{companionRadius, 0.0f, 0.0f}, {0.0f, 0.0f, orbitalSpeed * primaryMass / totalMass}, companionMass, 0.27f, colors[1], "body 2", false, false, {}},
    };
}

std::vector<MassObject> MakeTriangularTripleMasses() {
    std::vector<Color> colors = BodyPalette();
    std::vector<MassObject> result;
    const float orbitRadius = 4.4f;
    const float mass = 2.35f;
    const float omega = std::sqrt(kG * mass / (std::sqrt(3.0f) * orbitRadius * orbitRadius * orbitRadius));
    const float speed = omega * orbitRadius;
    for (int i = 0; i < 3; ++i) {
        float angle = static_cast<float>(i) * 2.0f * PI / 3.0f;
        Vector3 radial = {std::cos(angle), 0.0f, std::sin(angle)};
        Vector3 tangent = {-std::sin(angle), 0.0f, std::cos(angle)};
        result.push_back({Vector3Scale(radial, orbitRadius), Vector3Scale(tangent, speed), mass, 0.25f, colors[i], LabelForIndex(i), false, false, {}});
    }
    RecenterSystem(&result);
    return result;
}

std::vector<MassObject> MakeClusterMasses() {
    std::vector<Color> colors = BodyPalette();
    std::vector<MassObject> result;
    const int count = 5;
    const float radius = 5.7f;
    for (int i = 0; i < count; ++i) {
        float angle = static_cast<float>(i) * 2.0f * PI / static_cast<float>(count);
        float mass = 1.15f + 0.22f * static_cast<float>(i % 3);
        Vector3 radial = {std::cos(angle), 0.24f * std::sin(angle * 2.0f), std::sin(angle)};
        Vector3 tangent = {-std::sin(angle), 0.0f, std::cos(angle)};
        result.push_back({Vector3Scale(radial, radius), Vector3Scale(tangent, 0.55f + 0.08f * i), mass, 0.18f, colors[i % static_cast<int>(colors.size())], LabelForIndex(i), false, false, {}});
    }
    RecenterSystem(&result);
    return result;
}

void AddOrbitingMass(std::vector<MassObject>* masses, int selected) {
    if (masses->size() >= 6) return;

    const std::vector<Color> colors = BodyPalette();
    int n = static_cast<int>(masses->size());
    if (n == 1) {
        float originalMass = (*masses)[0].mass;
        *masses = MakeBinaryMasses();
        (*masses)[0].mass = originalMass;
        RecenterSystem(masses);
        return;
    }

    if (n == 2) {
        *masses = MakeTriangularTripleMasses();
        return;
    }

    float angle = 0.85f + static_cast<float>(n) * 1.45f;
    float radius = 6.2f + 0.8f * static_cast<float>(n % 3);
    Vector3 barycenter = CenterOfMass(*masses);
    Vector3 systemVelocity = CenterOfMassVelocity(*masses);
    Vector3 offset = {radius * std::cos(angle), 0.35f * std::sin(angle * 0.7f), radius * std::sin(angle)};
    Vector3 pos = Vector3Add(barycenter, offset);

    Vector3 radial = Vector3Normalize(Vector3Subtract(pos, barycenter));
    Vector3 tangent = Vector3Normalize(Vector3CrossProduct({0.0f, 1.0f, 0.0f}, radial));
    if (Vector3Length(tangent) < 0.001f) tangent = {0.0f, 0.0f, 1.0f};

    float totalMass = 0.0f;
    for (const MassObject& body : *masses) totalMass += body.mass;
    float orbitalSpeed = std::sqrt(kG * std::max(totalMass, 0.1f) / std::max(radius, 0.4f));
    Vector3 vel = Vector3Add(systemVelocity, Vector3Scale(tangent, orbitalSpeed * 1.05f));

    std::ostringstream label;
    label << "body " << (n + 1);
    masses->push_back({pos, vel, 0.85f, 0.22f, colors[n % static_cast<int>(colors.size())], label.str(), false, false, {}});
    RecenterSystem(masses);
}

std::string Hud(const std::vector<MassObject>& masses, int selected, bool relativisticMode, bool radiationDecay, bool paused, float simSpeed) {
    std::ostringstream os;
    float rs = SchwarzschildRadius(masses[selected].mass);
    os << std::fixed << std::setprecision(2)
       << "selected=" << masses[selected].label
       << "  mass=" << masses[selected].mass
       << "  bodies=" << masses.size()
       << "  Rs=" << rs
       << "  speed=" << simSpeed << "x";
    os << (relativisticMode ? "  [weak-field GR]" : "  [Newtonian]");
    if (radiationDecay) os << "  [GW decay]";
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(kScreenWidth, kScreenHeight, "3D Multi-Mass Gravity Grid - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw = 0.72f;
    float camPitch = 0.28f;
    float camDistance = 18.0f;

    std::vector<MassObject> masses = MakeDefaultMasses();
    std::vector<CollisionEvent> collisions;
    std::vector<ExplosionParticle> explosionParticles;

    int selected = 0;
    float core = 1.05f;
    float time = 0.0f;
    float gridExtent = kBaseGridExtent;
    float simSpeed = 1.0f;
    bool paused = false;
    bool animateGrid = true;
    bool bodyMotionOn = true;
    bool relativisticMode = true;
    bool radiationDecay = false;
    int windowedWidth = kScreenWidth;
    int windowedHeight = kScreenHeight;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_F)) {
            if (!IsWindowFullscreen()) {
                windowedWidth = GetScreenWidth();
                windowedHeight = GetScreenHeight();
                int monitor = GetCurrentMonitor();
                SetWindowSize(GetMonitorWidth(monitor), GetMonitorHeight(monitor));
                ToggleFullscreen();
            } else {
                ToggleFullscreen();
                SetWindowSize(windowedWidth, windowedHeight);
            }
        }
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_G)) relativisticMode = !relativisticMode;
        if (IsKeyPressed(KEY_C)) radiationDecay = !radiationDecay;
        if (IsKeyPressed(KEY_B)) {
            MassObject& body = masses[selected];
            body.blackHole = !body.blackHole;
            if (body.blackHole) {
                body.pulsar = false;
                std::size_t pulsarSuffix = body.label.find(" PSR");
                if (pulsarSuffix != std::string::npos) body.label.erase(pulsarSuffix);
                body.mass = std::max(body.mass, kBlackHoleMinMass);
                body.radius = std::max(0.36f, SchwarzschildRadius(body.mass) * 0.72f);
                body.color = Color{20, 20, 24, 255};
                if (body.label.find(" BH") == std::string::npos) body.label += " BH";
            } else {
                body.radius = 0.30f;
                body.color = BodyPalette()[selected % static_cast<int>(BodyPalette().size())];
                std::size_t suffix = body.label.find(" BH");
                if (suffix != std::string::npos) body.label.erase(suffix);
            }
        }
        if (IsKeyPressed(KEY_L)) {
            MassObject& body = masses[selected];
            if (!body.blackHole) {
                body.pulsar = !body.pulsar;
                if (body.pulsar) {
                    body.mass = std::max(body.mass, kPulsarMinMass);
                    body.radius = 0.22f;
                    body.color = Color{170, 220, 255, 255};
                    if (body.label.find(" PSR") == std::string::npos) body.label += " PSR";
                } else {
                    body.radius = 0.30f;
                    body.color = BodyPalette()[selected % static_cast<int>(BodyPalette().size())];
                    std::size_t suffix = body.label.find(" PSR");
                    if (suffix != std::string::npos) body.label.erase(suffix);
                }
            }
        }
        if (IsKeyPressed(KEY_SPACE)) animateGrid = !animateGrid;
        if (IsKeyPressed(KEY_M)) bodyMotionOn = !bodyMotionOn;
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) simSpeed = std::max(0.05f, simSpeed * 0.8f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) simSpeed = std::min(12.0f, simSpeed * 1.25f);
        if (IsKeyPressed(KEY_ZERO) || IsKeyPressed(KEY_KP_0)) simSpeed = 1.0f;
        if (IsKeyPressed(KEY_ONE)) { masses = MakeDefaultMasses(); collisions.clear(); explosionParticles.clear(); selected = 0; }
        if (IsKeyPressed(KEY_TWO)) { masses = MakeBinaryMasses(); collisions.clear(); explosionParticles.clear(); selected = 1; }
        if (IsKeyPressed(KEY_THREE)) { masses = MakeTriangularTripleMasses(); collisions.clear(); explosionParticles.clear(); selected = 2; }
        if (IsKeyPressed(KEY_FOUR)) { masses = MakeClusterMasses(); collisions.clear(); explosionParticles.clear(); selected = 0; }
        if (IsKeyPressed(KEY_N)) {
            AddOrbitingMass(&masses, selected);
            selected = static_cast<int>(masses.size()) - 1;
        }
        if ((IsKeyPressed(KEY_BACKSPACE) || IsKeyPressed(KEY_DELETE)) && masses.size() > 1) {
            masses.erase(masses.begin() + selected);
            selected = std::min(selected, static_cast<int>(masses.size()) - 1);
        }
        if (IsKeyPressed(KEY_TAB)) selected = (selected + 1) % static_cast<int>(masses.size());
        if (IsKeyPressed(KEY_R)) {
            masses = MakeDefaultMasses();
            collisions.clear();
            explosionParticles.clear();
            selected = 0;
            core = 1.05f;
            time = 0.0f;
            gridExtent = kBaseGridExtent;
            simSpeed = 1.0f;
            paused = false;
            animateGrid = true;
            bodyMotionOn = true;
            relativisticMode = true;
            radiationDecay = false;
        }

        float dt = GetFrameTime();
        MassObject& active = masses[selected];
        if (IsKeyDown(KEY_UP)) active.mass += 1.4f * dt;
        if (IsKeyDown(KEY_DOWN)) active.mass = std::max(0.35f, active.mass - 1.4f * dt);
        if (active.blackHole) active.radius = std::max(0.36f, SchwarzschildRadius(active.mass) * 0.72f);
        if (IsKeyDown(KEY_RIGHT)) core = std::min(2.4f, core + 0.75f * dt);
        if (IsKeyDown(KEY_LEFT)) core = std::max(0.45f, core - 0.75f * dt);
        bool movedActive = false;
        if (IsKeyDown(KEY_W)) { active.pos.y = std::min(3.8f, active.pos.y + 1.8f * dt); movedActive = true; }
        if (IsKeyDown(KEY_S)) { active.pos.y = std::max(-3.8f, active.pos.y - 1.8f * dt); movedActive = true; }
        if (IsKeyDown(KEY_A)) { active.pos.x = std::max(-3.8f, active.pos.x - 1.8f * dt); movedActive = true; }
        if (IsKeyDown(KEY_D)) { active.pos.x = std::min(3.8f, active.pos.x + 1.8f * dt); movedActive = true; }
        if (IsKeyDown(KEY_Q)) { active.pos.z = std::max(-3.8f, active.pos.z - 1.8f * dt); movedActive = true; }
        if (IsKeyDown(KEY_E)) { active.pos.z = std::min(3.8f, active.pos.z + 1.8f * dt); movedActive = true; }
        if (movedActive) {
            active.vel = {0.0f, 0.0f, 0.0f};
            active.trail.clear();
        }

        if (!paused) {
            float scaledDt = dt * simSpeed;
            if (animateGrid) time += scaledDt * 4.2f;
            UpdateCollisionEvents(&collisions, scaledDt);
            UpdateExplosionParticles(&explosionParticles, scaledDt);
            if (bodyMotionOn) {
                float totalSimDt = std::min(0.014f, dt) * 5.0f * simSpeed;
                int steps = std::clamp(static_cast<int>(std::ceil(totalSimDt / 0.014f)), 1, 96);
                float simDt = totalSimDt / static_cast<float>(steps);
                for (int i = 0; i < steps; ++i) UpdateMassObjects(&masses, &collisions, &explosionParticles, &selected, simDt, relativisticMode, radiationDecay);
            }
        }

        float targetExtent = TargetGridExtent(masses);
        if (targetExtent > gridExtent) {
            gridExtent = targetExtent;
        } else {
            gridExtent += (targetExtent - gridExtent) * std::clamp(dt * 0.85f, 0.0f, 1.0f);
        }

        UpdateOrbitCamera(&camera, &camYaw, &camPitch, &camDistance);

        BeginDrawing();
        ClearBackground(BLACK);

        BeginMode3D(camera);
        DrawWarpedCubeGrid(masses, collisions, core, time, gridExtent);
        DrawGravitationalWaveRipples(masses, collisions, time, radiationDecay);
        DrawExplosionParticles(explosionParticles);

        for (int i = 0; i < static_cast<int>(masses.size()); ++i) {
            const MassObject& body = masses[i];
            for (std::size_t j = 1; j < body.trail.size(); ++j) {
                float a = static_cast<float>(j) / static_cast<float>(body.trail.size());
                DrawLine3D(body.trail[j - 1], body.trail[j], WithAlpha(body.color, static_cast<unsigned char>(30 + 145 * a)));
            }
            float rs = SchwarzschildRadius(body.mass);
            if (body.blackHole) {
                DrawBlackHole(body, i == selected, time);
            } else if (body.pulsar) {
                DrawPulsar(body, i == selected, time);
            } else {
                DrawSphere(body.pos, body.radius, body.color);
                DrawSphereWires(body.pos, std::max(body.radius * 1.15f, rs), 20, 12, Color{255, 90, 80, 105});
                DrawSphereWires(body.pos, body.radius * (i == selected ? 1.9f : 1.45f), 18, 12,
                                i == selected ? Color{120, 230, 255, 180} : Color{210, 245, 255, 110});
                DrawSphere(body.pos, body.radius * 2.15f, WithAlpha(body.color, 42));
            }
        }
        EndMode3D();

        DrawText("Weak-Field Relativistic Gravity Grid", 20, 18, 28, Color{238, 242, 252, 255});
        DrawText("1-4 presets | N add | B black hole | L pulsar | -/+ speed | 0 reset speed | SPACE flow | R reset", 20, 52, 18, Color{166, 184, 214, 255});
        std::string hud = Hud(masses, selected, relativisticMode, radiationDecay, paused, simSpeed);
        DrawText(hud.c_str(), 20, 82, 19, Color{255, 220, 120, 255});
        DrawFPS(20, 112);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
