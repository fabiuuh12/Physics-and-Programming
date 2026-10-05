#include "raylib.h"
#include "raymath.h"
#include "../common/cosmic_studio.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;
constexpr float kPi = 3.14159265358979323846f;

struct DiskParticle {
    float radiusBase;
    float radialNoise;
    float theta;
    float omega;
    float yBase;
    float heat;
    float alpha;
    float size;
    float streak;
    float phase;
    int band;
};

struct CoronaParticle {
    float radius;
    float theta;
    float omega;
    float height;
    float pulse;
    float size;
};

struct JetPacket {
    float axial;
    float radial;
    float theta;
    float speed;
    float age;
    float ttl;
    float width;
    float brightness;
    int direction;
};

struct Star {
    Vector3 position;
    float size;
    Color color;
};

struct CameraPreset {
    float yaw;
    float pitch;
    float distance;
    Vector3 target;
};

float RandomFloat(float minValue, float maxValue) {
    return minValue + (maxValue - minValue) * (static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f);
}

Color LerpColor(Color a, Color b, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * t),
        static_cast<unsigned char>(a.g + (b.g - a.g) * t),
        static_cast<unsigned char>(a.b + (b.b - a.b) * t),
        static_cast<unsigned char>(a.a + (b.a - a.a) * t),
    };
}

Color DiskHeatColor(float heat) {
    if (heat > 0.82f) {
        return LerpColor(Color{160, 214, 255, 255}, Color{255, 249, 228, 255}, (heat - 0.82f) / 0.18f);
    }
    if (heat > 0.52f) {
        return LerpColor(Color{255, 188, 92, 255}, Color{255, 244, 212, 255}, (heat - 0.52f) / 0.30f);
    }
    return LerpColor(Color{214, 72, 30, 255}, Color{255, 190, 92, 255}, heat / 0.52f);
}

void UpdateOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && studio::over(cosmic::view)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0034f;
        *pitch += delta.y * 0.0032f;
        *pitch = std::clamp(*pitch, -1.42f, 1.42f);
    }

    if (studio::over(cosmic::view)) *distance -= GetMouseWheelMove() * 0.95f;
    *distance = std::clamp(*distance, 5.0f, 75.0f);

    const float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    });
}

void ApplyPreset(const CameraPreset& preset, Camera3D* camera, float* yaw, float* pitch, float* distance) {
    *yaw = preset.yaw;
    *pitch = preset.pitch;
    *distance = preset.distance;
    camera->target = preset.target;
    const float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    });
}

Vector3 TorusPoint(float u, float v, float majorRadius, float minorRadius) {
    Vector3 p{
        (majorRadius + minorRadius * std::cos(v)) * std::cos(u),
        minorRadius * std::sin(v),
        (majorRadius + minorRadius * std::cos(v)) * std::sin(u),
    };
    return p;
}

void InitializeDisk(std::vector<DiskParticle>* particles) {
    particles->clear();
    const std::array<int, 5> counts = {170, 220, 250, 230, 170};
    const std::array<std::array<float, 2>, 5> radii = {{
        {2.05f, 2.75f},
        {2.75f, 3.85f},
        {3.85f, 5.05f},
        {5.05f, 6.55f},
        {6.55f, 8.10f},
    }};

    for (int band = 0; band < static_cast<int>(counts.size()); ++band) {
        for (int i = 0; i < counts[band]; ++i) {
            const float radius = RandomFloat(radii[band][0], radii[band][1]);
            const float heat = std::clamp(1.0f - band * 0.18f + RandomFloat(-0.05f, 0.05f), 0.12f, 1.0f);
            const float omega = (1.8f - band * 0.18f) / std::pow(radius, 0.94f);
            particles->push_back(DiskParticle{
                radius,
                RandomFloat(0.06f, 0.60f),
                RandomFloat(0.0f, 2.0f * kPi),
                omega,
                RandomFloat(-0.7f, 0.7f),
                heat,
                RandomFloat(0.38f, 0.96f),
                RandomFloat(0.045f, 0.095f),
                RandomFloat(0.10f, 0.38f),
                RandomFloat(0.0f, 2.0f * kPi),
                band,
            });
        }
    }
}

void InitializeCorona(std::vector<CoronaParticle>* particles) {
    particles->clear();
    particles->reserve(160);
    for (int i = 0; i < 160; ++i) {
        particles->push_back(CoronaParticle{
            RandomFloat(1.6f, 3.4f),
            RandomFloat(0.0f, 2.0f * kPi),
            RandomFloat(0.7f, 2.6f),
            RandomFloat(-1.9f, 1.9f),
            RandomFloat(0.0f, 2.0f * kPi),
            RandomFloat(0.035f, 0.090f),
        });
    }
}

void InitializeStars(std::vector<Star>* stars) {
    stars->clear();
    stars->reserve(220);
    for (int i = 0; i < 220; ++i) {
        float theta = RandomFloat(0.0f, 2.0f * kPi);
        float phi = RandomFloat(0.25f, kPi - 0.25f);
        float radius = RandomFloat(30.0f, 80.0f);
        Vector3 pos{
            radius * std::sin(phi) * std::cos(theta),
            radius * std::cos(phi) * RandomFloat(0.4f, 1.1f),
            radius * std::sin(phi) * std::sin(theta),
        };
        float tint = RandomFloat(0.0f, 1.0f);
        stars->push_back(Star{
            pos,
            RandomFloat(0.05f, 0.18f),
            LerpColor(Color{150, 176, 255, 255}, Color{255, 236, 210, 255}, tint),
        });
    }
}

void SpawnJetPacket(std::vector<JetPacket>* packets, int direction, float jetPower, float flareStrength) {
    packets->push_back(JetPacket{
        RandomFloat(1.35f, 1.85f),
        RandomFloat(0.04f, 0.26f + 0.12f * jetPower),
        RandomFloat(0.0f, 2.0f * kPi),
        RandomFloat(4.0f, 8.2f) + jetPower * 1.6f + flareStrength * 2.0f,
        0.0f,
        RandomFloat(2.0f, 3.6f),
        RandomFloat(0.07f, 0.17f),
        RandomFloat(0.55f, 1.0f),
        direction,
    });
}

float DiskAngularSpeed(const DiskParticle& p,float mass,float flare) {
    return p.omega*(0.72f+mass/260.0f)*(1+(1-p.radiusBase/8.2f)*(0.35f+flare*0.65f));
}
Vector3 DiskPosition(const DiskParticle& p,float time,float thickness) {
    float radius=p.radiusBase+std::sin(time*(1.2f+0.2f*p.band)+p.phase)*p.radialNoise*0.12f;
    float y=p.yBase*0.22f*thickness+std::sin(time*2.8f+p.phase*1.7f)*0.05f*thickness*(1.2f-p.band/5.0f);
    return {radius*std::cos(p.theta),y,radius*std::sin(p.theta)};
}
Vector3 DiskVelocity(const DiskParticle& p,float time,float thickness,float mass,float flare) {
    Vector3 pos=DiskPosition(p,time,thickness);
    float r=std::hypot(pos.x,pos.z);
    float dr=std::cos(time*(1.2f+0.2f*p.band)+p.phase)*(1.2f+0.2f*p.band)*p.radialNoise*0.12f;
    float w=DiskAngularSpeed(p,mass,flare);
    float dy=std::cos(time*2.8f+p.phase*1.7f)*2.8f*0.05f*thickness*(1.2f-p.band/5.0f);
    return {dr*std::cos(p.theta)-r*w*std::sin(p.theta),dy,dr*std::sin(p.theta)+r*w*std::cos(p.theta)};
}
constexpr float kJetContraction=-0.481929f; // former 0.992 per frame at 60 Hz
Vector3 JetPosition(const JetPacket& p) {
    return {p.radial*std::cos(p.theta),p.direction*p.axial,p.radial*std::sin(p.theta)};
}
Vector3 JetVelocity(const JetPacket& p,float power) {
    float w=1+0.8f*power, dr=p.radial*kJetContraction;
    return {dr*std::cos(p.theta)-p.radial*w*std::sin(p.theta),
            p.direction*p.speed*(1+0.25f*power),
            dr*std::sin(p.theta)+p.radial*w*std::cos(p.theta)};
}

void DrawRing(float radius, float y, int segments, Color color, float wobble, float time) {
    for (int i = 0; i < segments; ++i) {
        float a0 = (2.0f * kPi * i) / static_cast<float>(segments);
        float a1 = (2.0f * kPi * (i + 1)) / static_cast<float>(segments);
        float r0 = radius + wobble * std::sin(time * 2.8f + a0 * 5.0f);
        float r1 = radius + wobble * std::sin(time * 2.8f + a1 * 5.0f);
        Vector3 p0{r0 * std::cos(a0), y, r0 * std::sin(a0)};
        Vector3 p1{r1 * std::cos(a1), y, r1 * std::sin(a1)};
        DrawLine3D(p0, p1, color);
    }
}

void DrawTorus(float majorRadius, float minorRadius, float opacity, float time, bool guides) {
    const int majorSegments = 56;
    const int minorSegments = 12;
    for (int i = 0; i < majorSegments; ++i) {
        float u0 = (2.0f * kPi * i) / static_cast<float>(majorSegments);
        float u1 = (2.0f * kPi * (i + 1)) / static_cast<float>(majorSegments);
        for (int j = 0; j < minorSegments; ++j) {
            float v0 = (2.0f * kPi * j) / static_cast<float>(minorSegments);
            float v1 = (2.0f * kPi * (j + 1)) / static_cast<float>(minorSegments);
            Vector3 p00 = TorusPoint(u0, v0, majorRadius, minorRadius);
            Vector3 p10 = TorusPoint(u1, v0, majorRadius, minorRadius);
            Vector3 p01 = TorusPoint(u0, v1, majorRadius, minorRadius);
            float rim = 0.5f + 0.5f * std::sin(v0 + time * 0.8f);
            Color c = Color{
                static_cast<unsigned char>(80 + 120 * rim),
                static_cast<unsigned char>(42 + 65 * rim),
                static_cast<unsigned char>(20 + 25 * rim),
                static_cast<unsigned char>(20 + opacity * (45 + 55 * rim)),
            };
            Vector3 p11=TorusPoint(u1,v1,majorRadius,minorRadius);
            c.a=static_cast<unsigned char>(opacity*(22+20*rim));
            DrawTriangle3D(p00,p11,p10,c);
            DrawTriangle3D(p00,p01,p11,c);
            if (guides && j%3==0) DrawLine3D(p00,p10,Fade(cosmic::warm,0.18f));
        }
    }
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Quasar Core | Active Nucleus Lab");
    RenderTexture2D scene=LoadRenderTexture(cosmic::viewWidth,cosmic::viewHeight);
    SetTextureFilter(scene.texture,TEXTURE_FILTER_BILINEAR);
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {15.0f, 7.2f, 15.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 42.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    std::array<CameraPreset, 4> presets = {{
        {0.82f, 0.33f, 32.0f, {0.0f, 0.0f, 0.0f}},
        {1.58f, 0.04f, 32.0f, {0.0f, 0.1f, 0.0f}},
        {0.24f, 0.64f, 24.0f, {0.0f, 0.5f, 0.0f}},
        {1.55f, 1.02f, 23.0f, {0.0f, 6.0f, 0.0f}},
    }};

    float camYaw = presets[0].yaw;
    float camPitch = presets[0].pitch;
    float camDistance = presets[0].distance;

    float blackHoleMass = 280.0f;
    float jetPower = 1.25f;
    float diskThickness = 0.72f;
    float torusOpacity = 0.65f;
    float beamingScale = 1.45f;
    bool paused = false;
    bool showVectors=true, showTrails=true, showGuides=false;
    float arrowScale=1.0f;

    float time = 0.0f;
    float flareTimer = 0.0f;
    float flareStrength = 0.0f;
    float jetSpawnAccumulator = 0.0f;

    std::vector<DiskParticle> diskParticles;
    std::vector<CoronaParticle> coronaParticles;
    std::vector<JetPacket> jetPackets;
    std::vector<Star> stars;
    InitializeDisk(&diskParticles);
    InitializeCorona(&coronaParticles);
    InitializeStars(&stars);
    jetPackets.reserve(540);

    ApplyPreset(presets[0], &camera, &camYaw, &camPitch, &camDistance);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_V) || studio::clicked({1130,450,128,40})) showVectors=!showVectors;
        if (IsKeyPressed(KEY_T) || studio::clicked({1268,450,128,40})) showTrails=!showTrails;
        if (IsKeyPressed(KEY_G) || studio::clicked({1130,500,128,40})) showGuides=!showGuides;
        if (IsKeyDown(KEY_COMMA)) arrowScale=std::max(0.25f,arrowScale*std::exp(-GetFrameTime()));
        if (IsKeyDown(KEY_PERIOD)) arrowScale=std::min(3.0f,arrowScale*std::exp(GetFrameTime()));
        if (IsKeyPressed(KEY_ONE)) ApplyPreset(presets[0], &camera, &camYaw, &camPitch, &camDistance);
        if (IsKeyPressed(KEY_TWO)) ApplyPreset(presets[1], &camera, &camYaw, &camPitch, &camDistance);
        if (IsKeyPressed(KEY_THREE)) ApplyPreset(presets[2], &camera, &camYaw, &camPitch, &camDistance);
        if (IsKeyPressed(KEY_FOUR)) ApplyPreset(presets[3], &camera, &camYaw, &camPitch, &camDistance);

        if (IsKeyPressed(KEY_P) || studio::clicked({1268,500,128,40})) paused = !paused;
        if (IsKeyPressed(KEY_B)) beamingScale = (beamingScale > 1.46f) ? 1.0f : 2.25f;
        if (IsKeyPressed(KEY_F) || studio::clicked({1130,559,266,40})) flareTimer = 1.65f;
        if (IsKeyPressed(KEY_R)) {
            blackHoleMass = 280.0f;
            jetPower = 1.25f;
            diskThickness = 0.72f;
            torusOpacity = 0.65f;
            beamingScale = 1.45f;
            paused = false;
            time = 0.0f;
            flareTimer = 0.0f;
            flareStrength = 0.0f;
            jetSpawnAccumulator = 0.0f;
            InitializeDisk(&diskParticles);
            InitializeCorona(&coronaParticles);
            InitializeStars(&stars);
            jetPackets.clear();
            ApplyPreset(presets[0], &camera, &camYaw, &camPitch, &camDistance);
        }

        if (IsKeyDown(KEY_UP)) blackHoleMass = std::min(520.0f, blackHoleMass + 110.0f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) blackHoleMass = std::max(120.0f, blackHoleMass - 110.0f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT)) jetPower = std::min(3.2f, jetPower + 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT)) jetPower = std::max(0.2f, jetPower - 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT_BRACKET)) diskThickness = std::min(1.45f, diskThickness + 0.55f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT_BRACKET)) diskThickness = std::max(0.18f, diskThickness - 0.55f * GetFrameTime());
        if (IsKeyDown(KEY_EQUAL)) torusOpacity = std::min(1.0f, torusOpacity + 0.7f * GetFrameTime());
        if (IsKeyDown(KEY_MINUS)) torusOpacity = std::max(0.05f, torusOpacity - 0.7f * GetFrameTime());

        UpdateOrbitCamera(&camera, &camYaw, &camPitch, &camDistance);

        const float dt = std::min(GetFrameTime(),0.05f);
        if (!paused) {
            time += dt;
            flareTimer = std::max(0.0f, flareTimer - dt);
            flareStrength = std::sin((flareTimer / 1.65f) * kPi);
            flareStrength = std::max(flareStrength, 0.0f);

            for (DiskParticle& particle : diskParticles)
                particle.theta=std::fmod(particle.theta+DiskAngularSpeed(particle,blackHoleMass,flareStrength)*dt,2*kPi);
            for (CoronaParticle& particle : coronaParticles) {
                particle.theta += particle.omega * (1.0f + flareStrength * 0.5f) * dt;
            }

            jetSpawnAccumulator += dt * (18.0f + jetPower * 22.0f + flareStrength * 18.0f);
            while (jetSpawnAccumulator >= 1.0f) {
                jetSpawnAccumulator -= 1.0f;
                SpawnJetPacket(&jetPackets, 1, jetPower, flareStrength);
                SpawnJetPacket(&jetPackets, -1, jetPower, flareStrength);
            }

            for (JetPacket& packet : jetPackets) {
                packet.age += dt;
                packet.axial += packet.speed * dt * (1.0f + 0.25f * jetPower);
                packet.theta += dt * (1.0f + 0.8f * jetPower);
                packet.radial *= std::exp(kJetContraction*dt);
            }
            jetPackets.erase(
                std::remove_if(jetPackets.begin(), jetPackets.end(), [](const JetPacket& packet) {
                    return packet.age > packet.ttl || packet.axial > 24.0f;
                }),
                jetPackets.end());
        }

        const float horizonRadius = 1.0f + (blackHoleMass - 120.0f) / 500.0f;
        const float photonRingRadius = horizonRadius * 1.85f;
        const Vector3 observerDirection = Vector3Normalize(Vector3Subtract(camera.position, camera.target));

        cosmic::beginScene(scene);

        BeginMode3D(camera);
        if (showGuides) for (int i=-10;i<=10;++i) {
            DrawLine3D({float(i)*1.4f,-0.8f,-14},{float(i)*1.4f,-0.8f,14},Color{40,60,86,95});
            DrawLine3D({-14,-0.8f,float(i)*1.4f},{14,-0.8f,float(i)*1.4f},Color{40,60,86,95});
        }

        for (const Star& star : stars) {
            DrawSphereEx(star.position, star.size*0.6f, 4, 4, Fade(star.color, 0.80f));
        }

        for (int i = 0; i < 8; ++i) {
            float radius = 10.0f + i * 0.95f;
            float alpha = 0.07f - i * 0.006f;
            DrawRing(radius, -0.04f * i, 84, Fade(Color{90, 130, 185, 255}, alpha), 0.04f, time * 0.2f + i);
        }

        rlDrawRenderBatchActive();
        rlDisableDepthMask();
        DrawTorus(8.6f, 0.75f + diskThickness * 0.25f, torusOpacity, time, showGuides);
        rlDrawRenderBatchActive();
        rlEnableDepthMask();

        for (const DiskParticle& particle : diskParticles) {
            float bandFactor = 1.0f - particle.band / 5.0f;
            Vector3 position=DiskPosition(particle,time,diskThickness);

            Vector3 tangent=Vector3Normalize(DiskVelocity(particle,time,diskThickness,blackHoleMass,flareStrength));
            float viewBoost = std::clamp((Vector3DotProduct(tangent, observerDirection) + 1.0f) * 0.5f, 0.0f, 1.0f);
            float beaming = 0.65f + std::pow(viewBoost, 1.0f + beamingScale) * (0.8f + jetPower * 0.25f);

            Color baseColor = DiskHeatColor(std::clamp(particle.heat + flareStrength * bandFactor * 0.20f, 0.0f, 1.0f));
            Color drawColor = Fade(baseColor, std::clamp(particle.alpha * beaming, 0.0f, 1.0f));

            float streakLength = particle.streak * (1.0f + bandFactor * 0.7f + flareStrength * bandFactor);
            Vector3 tail = Vector3Subtract(position, Vector3Scale(tangent, streakLength));
            if (showTrails) DrawLine3D(tail, position, Fade(drawColor, 0.75f));
            DrawSphereEx(position, particle.size * (0.9f + 0.8f * bandFactor + flareStrength * 0.4f), 4, 6, drawColor);
        }

        for (const CoronaParticle& particle : coronaParticles) {
            float lift = particle.height + std::sin(time * 1.8f + particle.pulse) * 0.25f;
            Vector3 position{
                particle.radius * std::cos(particle.theta),
                lift,
                particle.radius * std::sin(particle.theta),
            };
            float pulse = 0.55f + 0.45f * std::sin(time * 4.0f + particle.pulse);
            Color coronaColor = Fade(Color{185, 226, 255, 255}, 0.10f + pulse * 0.24f + flareStrength * 0.12f);
            DrawSphereEx(position, particle.size * (1.0f + flareStrength * 0.8f), 4, 6, coronaColor);
        }

        for (int side = -1; side <= 1; side += 2) {
            float spineLength = 10.0f + jetPower * 2.0f;
            Color spineColor = (side > 0) ? Color{120, 220, 255, 120} : Color{92, 178, 250, 96};
            Color plumeColor = (side > 0) ? Color{80, 186, 255, 40} : Color{70, 160, 230, 32};
            DrawCylinderEx({0.0f, horizonRadius * side, 0.0f}, {0.0f, spineLength * side, 0.0f},
                           0.24f + flareStrength * 0.08f, 0.08f, 16, spineColor);
            DrawCylinderEx({0.0f, horizonRadius * side, 0.0f}, {0.0f, (8.0f + jetPower * 2.0f) * side, 0.0f},
                           0.25f + jetPower * 0.08f, 0.55f + jetPower * 0.12f, 20, Fade(plumeColor,0.08f));
        }

        for (const JetPacket& packet : jetPackets) {
            Vector3 position=JetPosition(packet);
            float ageT = std::clamp(packet.age / packet.ttl, 0.0f, 1.0f);
            float beaming = (packet.direction > 0)
                                ? (0.7f + 0.6f * std::pow(std::max(0.0f, Vector3DotProduct(observerDirection, {0.0f, 1.0f, 0.0f})), 2.0f))
                                : 0.85f;
            Color packetColor = LerpColor(Color{110, 200, 255, 255}, Color{255, 255, 255, 255}, packet.brightness);
            packetColor = Fade(packetColor, (1.0f - ageT) * (0.25f + packet.brightness * 0.65f) * beaming);
            Vector3 tail=Vector3Subtract(position,Vector3Scale(JetVelocity(packet,jetPower),0.055f));
            if (showTrails) DrawLine3D(tail, position, Fade(packetColor, 0.75f));
            DrawSphereEx(position, packet.width * (1.0f + 0.5f * packet.brightness), 4, 6, packetColor);
        }

        DrawSphere({0.0f, 0.0f, 0.0f}, horizonRadius, Color{6, 6, 8, 255});
        if (showGuides) DrawSphereWires({0,0,0},horizonRadius*1.08f,18,18,Fade(cosmic::motion,0.2f));
        DrawRing(photonRingRadius, 0.0f, 120, Fade(Color{255, 230, 176, 255}, 0.55f + flareStrength * 0.25f), 0.04f, time);
        DrawRing(photonRingRadius * 1.12f, 0.03f, 96, Fade(Color{255, 144, 76, 255}, 0.35f), 0.07f, time * 1.2f);
        DrawRing(photonRingRadius * 0.92f, -0.02f, 96, Fade(Color{196, 220, 255, 255}, 0.22f + flareStrength * 0.15f), 0.03f, time * 1.5f);

        if (showVectors) {
            for (size_t i=40;i<diskParticles.size();i+=104) {
                const auto& p=diskParticles[i];
                cosmic::arrow(DiskPosition(p,time,diskThickness),DiskVelocity(p,time,diskThickness,blackHoleMass,flareStrength),0.52f*arrowScale,cosmic::warm);
            }
            int count=0;
            for (size_t i=0;i<jetPackets.size() && count<8;i+=23) {
                const auto& p=jetPackets[i];
                if (p.axial>3 && p.axial<12) { cosmic::arrow(JetPosition(p),JetVelocity(p,jetPower),0.18f*arrowScale,cosmic::motion); ++count; }
            }
        }
        EndMode3D();
        cosmic::drawMotion(camera);
        cosmic::glow(camera,{0,0,0},cosmic::warm,0.07f+flareStrength*0.04f,160+flareStrength*40);
        cosmic::label(camera,{0,0,0},"BLACK HOLE",cosmic::muted,{16,16});
        cosmic::label(camera,{-5,0,0},"ACCRETION DISK",cosmic::warm,{-40,20});
        cosmic::label(camera,{0,7,0},"OUTFLOW +",cosmic::motion,{18,0});
        cosmic::label(camera,{0,-7,0},"OUTFLOW -",cosmic::motion,{18,0});
        cosmic::compose(scene,"ACTIVE GALACTIC NUCLEUS / 03","Quasar core","Orbiting disk material and bipolar jets");
        float x=cosmic::panelX+20;
        studio::text("OBSERVATORY",x,40,13,cosmic::muted);
        studio::text(paused ? "PAUSED" : "NUCLEUS ACTIVE",x,66,19,paused ? cosmic::warm : cosmic::motion);
        studio::text(TextFormat("Animation time   %.1f",time),x,104,16,cosmic::muted);
        studio::text(TextFormat("Core mass parameter    %.0f",blackHoleMass),x,148,18,cosmic::text);
        cosmic::bar(x,177,266,(blackHoleMass-120)/400,cosmic::warm);
        studio::text(TextFormat("Jet power                        %.2f",jetPower),x,203,18,cosmic::motion);
        cosmic::bar(x,232,266,(jetPower-0.2f)/3,cosmic::motion);
        studio::text(TextFormat("Disk thickness     %.2f",diskThickness),x,267,16,cosmic::text);
        studio::text(TextFormat("Torus opacity      %.2f",torusOpacity),x,298,16,cosmic::text);
        studio::text(TextFormat("Beaming scale     %.2f",beamingScale),x,329,16,cosmic::text);
        studio::text("DISK HEAT / RELATIVE",x,379,13,cosmic::muted);
        for (int i=0;i<266;++i) DrawRectangle(int(x)+i,408,1,9,DiskHeatColor(1-i/265.0f));
        studio::text("inner / hot",x,425,12,cosmic::warm);
        studio::text("outer / cooler",x+177,425,12,cosmic::muted);
        cosmic::button({x,450,128,40},"V  Motion",showVectors);
        cosmic::button({x+138,450,128,40},"T  Trails",showTrails,cosmic::warm);
        cosmic::button({x,500,128,40},"G  Guides",showGuides,cosmic::purple);
        cosmic::button({x+138,500,128,40},"P  Pause",paused);
        cosmic::button({x,559,266,40},"F  Trigger flare",flareTimer>0,cosmic::warm);
        cosmic::bar(x,614,266,flareStrength,cosmic::warm);
        studio::text(TextFormat("Arrow multiplier    %.2fx",arrowScale),x,655,16,cosmic::text);
        studio::text(", / .  adjust arrow size",x,681,14,cosmic::muted);
        studio::text("Gold: disk motion   Cyan: jet motion",x,721,13,cosmic::muted);
        studio::text("Illustrative animation; not GR / MHD",x,748,13,cosmic::muted);
        studio::text("Separate arrow scales; capped at 2.6",x,773,13,cosmic::muted);
        studio::text("FOLLOW THE FLOW",28,799,13,cosmic::warm);
        studio::text("Inner disk rotates faster. Jets travel away from the core in both directions.",200,798,14,cosmic::muted);
        cosmic::help("Drag: orbit   Wheel: zoom   1-4: views   Up/Down: mass   Left/Right: jets   [ / ]: thickness   - / =: torus   B: beaming   R: reset");
        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    UnloadRenderTexture(scene);
    studio::unload();
    CloseWindow();
    return 0;
}
