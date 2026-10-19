#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>

namespace {
constexpr int kScreenW = 1280;
constexpr int kScreenH = 820;
constexpr float kPi = 3.14159265358979323846f;

struct Vec2f {
    float y;
    float z;
};

struct SceneFrame {
    Vector3 origin;
    Vector3 xAxis;
    Vector3 yAxis;
    Vector3 zAxis;
};

Vector3 ToWorld(const SceneFrame& frame, Vector3 p) {
    return Vector3Add(
        Vector3Add(frame.origin, Vector3Scale(frame.xAxis, p.x - 6.0f)),
        Vector3Add(Vector3Scale(frame.yAxis, p.y), Vector3Scale(frame.zAxis, p.z))
    );
}

SceneFrame DefaultFrame() {
    return {{6.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
}

SceneFrame CameraFollowFrame(Camera3D camera) {
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
    if (Vector3Length(right) < 0.001f) right = {0.0f, 1.0f, 0.0f};
    Vector3 up = Vector3Normalize(Vector3CrossProduct(right, forward));
    return {camera.position, Vector3Negate(forward), right, up};
}

float Length(Vec2f v) {
    return std::sqrt(v.y * v.y + v.z * v.z);
}

Vec2f Scale(Vec2f v, float s) {
    return {v.y * s, v.z * s};
}

Vec2f Add(Vec2f a, Vec2f b) {
    return {a.y + b.y, a.z + b.z};
}

Vec2f ImagePosition(Vec2f beta, float thetaE, bool majorImage) {
    float b = Length(beta);
    if (b < 0.001f) return {majorImage ? thetaE : -thetaE, 0.0f};

    Vec2f dir = Scale(beta, 1.0f / b);
    float root = std::sqrt(b * b + 4.0f * thetaE * thetaE);
    float radius = 0.5f * (b + (majorImage ? root : -root));
    return Scale(dir, radius);
}

void UpdateOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.28f, 1.28f);
    }

    *distance -= GetMouseWheelMove() * 0.75f;
    *distance = std::clamp(*distance, 7.0f, 28.0f);

    float cp = std::cos(*pitch);
    Vector3 offset{
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw)
    };
    camera->position = Vector3Add(camera->target, offset);
}

Vector3 QuadraticPoint(Vector3 a, Vector3 b, Vector3 c, float t) {
    float u = 1.0f - t;
    return Vector3Add(Vector3Add(Vector3Scale(a, u * u), Vector3Scale(b, 2.0f * u * t)), Vector3Scale(c, t * t));
}

void DrawCircleYZ(const SceneFrame& frame, float x, float radius, Color color, int segments) {
    for (int i = 0; i < segments; ++i) {
        float a0 = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(segments);
        float a1 = 2.0f * kPi * static_cast<float>(i + 1) / static_cast<float>(segments);
        DrawLine3D(ToWorld(frame, {x, radius * std::cos(a0), radius * std::sin(a0)}),
                   ToWorld(frame, {x, radius * std::cos(a1), radius * std::sin(a1)}), color);
    }
}

void DrawPlaneGridYZ(float x, float extent, float spacing, Color color) {
    for (float p = -extent; p <= extent + 0.001f; p += spacing) {
        DrawLine3D({x, -extent, p}, {x, extent, p}, color);
        DrawLine3D({x, p, -extent}, {x, p, extent}, color);
    }
}

float Hash1(int n) {
    unsigned int x = static_cast<unsigned int>(n) * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return static_cast<float>(x) / static_cast<float>(0xffffffffu);
}

void DrawDeepStarfield(float phase) {
    for (int i = 0; i < 260; ++i) {
        float u = Hash1(i * 3 + 1);
        float v = Hash1(i * 3 + 2);
        float w = Hash1(i * 3 + 3);
        float angle = 2.0f * kPi * u;
        float z = -1.0f + 2.0f * v;
        float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        float shell = 15.0f + 7.0f * w;
        Vector3 p{shell * r * std::cos(angle), shell * z * 0.58f, shell * r * std::sin(angle)};
        float twinkle = 0.65f + 0.35f * std::sin(phase * (0.7f + w) + 19.0f * u);
        Color c{
            static_cast<unsigned char>(150 + 90 * Hash1(i + 77)),
            static_cast<unsigned char>(165 + 70 * Hash1(i + 113)),
            static_cast<unsigned char>(205 + 45 * Hash1(i + 151)),
            255
        };
        DrawSphere(p, 0.010f + 0.025f * Hash1(i + 211), Fade(c, 0.36f + 0.52f * twinkle));
    }
}

void DrawGlowSphere(Vector3 center, float radius, Color core, Color glow) {
    DrawSphere(center, radius, core);
    DrawSphereWires(center, radius * 1.45f, 18, 10, Fade(glow, 0.38f));
    DrawSphereWires(center, radius * 2.15f, 22, 12, Fade(glow, 0.18f));
    DrawSphereWires(center, radius * 3.00f, 26, 12, Fade(glow, 0.08f));
}

void DrawCurvedRay(const SceneFrame& frame, Vector3 source, Vector3 impact, Vector3 observer, Color beforeLens, Color afterLens) {
    Vector3 prev = source;
    Vector3 controlA{
        (source.x + impact.x) * 0.5f,
        source.y * 0.42f + impact.y * 0.58f,
        source.z * 0.42f + impact.z * 0.58f
    };
    for (int i = 1; i <= 26; ++i) {
        float t = static_cast<float>(i) / 26.0f;
        Vector3 now = QuadraticPoint(source, controlA, impact, t);
        DrawLine3D(ToWorld(frame, prev), ToWorld(frame, now), beforeLens);
        prev = now;
    }

    prev = impact;
    Vector3 controlB{
        (impact.x + observer.x) * 0.5f,
        impact.y * 0.78f,
        impact.z * 0.78f
    };
    for (int i = 1; i <= 34; ++i) {
        float t = static_cast<float>(i) / 34.0f;
        Vector3 now = QuadraticPoint(impact, controlB, observer, t);
        DrawLine3D(ToWorld(frame, prev), ToWorld(frame, now), afterLens);
        prev = now;
    }
}

Vector3 CurvedRayPoint(Vector3 source, Vector3 impact, Vector3 observer, float t) {
    if (t < 0.5f) {
        float localT = t * 2.0f;
        Vector3 controlA{
            (source.x + impact.x) * 0.5f,
            source.y * 0.42f + impact.y * 0.58f,
            source.z * 0.42f + impact.z * 0.58f
        };
        return QuadraticPoint(source, controlA, impact, localT);
    }

    float localT = (t - 0.5f) * 2.0f;
    Vector3 controlB{
        (impact.x + observer.x) * 0.5f,
        impact.y * 0.78f,
        impact.z * 0.78f
    };
    return QuadraticPoint(impact, controlB, observer, localT);
}

void DrawPhotonPulse(const SceneFrame& frame, Vector3 source, Vector3 impact, Vector3 observer, float phase, Color color) {
    for (int i = 0; i < 3; ++i) {
        float t = std::fmod(phase + 0.33f * static_cast<float>(i), 1.0f);
        Vector3 p = ToWorld(frame, CurvedRayPoint(source, impact, observer, t));
        DrawSphere(p, 0.055f, color);
        DrawSphereWires(p, 0.095f, 8, 6, Fade(color, 0.25f));
    }
}

void DrawBrightStar(const SceneFrame& frame, Vector3 center, float radius, float phase) {
    DrawGlowSphere(ToWorld(frame, center), radius, Color{255, 246, 190, 255}, Color{255, 191, 74, 255});

    for (int arm = 0; arm < 8; ++arm) {
        Vector3 prev = center;
        float a = 2.0f * kPi * static_cast<float>(arm) / 8.0f + 0.25f * std::sin(phase);
        for (int i = 1; i <= 18; ++i) {
            float t = static_cast<float>(i) / 18.0f;
            float flare = radius * (1.0f + 2.8f * t);
            Vector3 now{center.x, center.y + flare * std::cos(a), center.z + flare * std::sin(a)};
            DrawLine3D(ToWorld(frame, prev), ToWorld(frame, now), Fade(Color{255, 202, 96, 255}, 0.50f * (1.0f - t)));
            prev = now;
        }
    }
}

Vector3 DiskPoint(float radius, float angle, float tilt, float phase) {
    float wobble = 0.035f * std::sin(3.0f * angle + phase);
    return {
        radius * std::sin(angle) * tilt,
        radius * std::cos(angle),
        radius * std::sin(angle) * (0.28f + wobble)
    };
}

void DrawBlackHole(const SceneFrame& frame, float thetaE, float skyScale, float phase) {
    for (int ring = 0; ring < 9; ++ring) {
        float radius = 0.72f + 0.15f * static_cast<float>(ring);
        float baseAlpha = 0.34f - 0.022f * static_cast<float>(ring);
        Vector3 prev = DiskPoint(radius, phase * (0.22f + ring * 0.015f), 0.22f, phase);
        for (int i = 1; i <= 160; ++i) {
            float a = 2.0f * kPi * static_cast<float>(i) / 160.0f + phase * (0.22f + ring * 0.015f);
            Vector3 now = DiskPoint(radius, a, 0.22f, phase);
            float beam = 0.55f + 0.45f * std::max(0.0f, std::sin(a - 0.7f));
            Color c = (ring < 4)
                ? Color{255, static_cast<unsigned char>(108 + 80 * beam), 42, 255}
                : Color{255, static_cast<unsigned char>(170 + 70 * beam), 96, 255};
            DrawLine3D(ToWorld(frame, prev), ToWorld(frame, now), Fade(c, baseAlpha * (0.65f + 1.1f * beam)));
            prev = now;
        }
    }

    for (int i = 0; i < 48; ++i) {
        float a = 2.0f * kPi * static_cast<float>(i) / 48.0f + phase * 0.36f;
        float radius = 1.08f + 0.18f * std::sin(a * 2.0f + phase);
        Vector3 inner = DiskPoint(radius * 0.82f, a, 0.22f, phase);
        Vector3 outer = DiskPoint(radius * 1.16f, a + 0.10f, 0.22f, phase);
        DrawLine3D(ToWorld(frame, inner), ToWorld(frame, outer), Fade(Color{255, 184, 78, 255}, 0.16f));
    }

    DrawSphere(ToWorld(frame, {0.0f, 0.0f, 0.0f}), 0.43f, BLACK);
    DrawSphereWires(ToWorld(frame, {0.0f, 0.0f, 0.0f}), 0.50f, 30, 18, Fade(Color{25, 31, 45, 255}, 0.92f));
    DrawCircleYZ(frame, 0.0f, 0.61f, Fade(Color{255, 166, 78, 255}, 0.88f), 128);
    DrawCircleYZ(frame, 0.0f, 0.74f, Fade(Color{255, 222, 142, 255}, 0.28f), 128);
    DrawCircleYZ(frame, 0.0f, thetaE * skyScale, Fade(Color{255, 231, 152, 255}, 0.44f), 160);
}

void DrawImageTrails(const SceneFrame& frame, const std::deque<Vec2f>& trail, float thetaE, float skyScale) {
    for (int i = 1; i < static_cast<int>(trail.size()); ++i) {
        float alpha = static_cast<float>(i) / static_cast<float>(trail.size());
        Vec2f a0 = ImagePosition(trail[i - 1], thetaE, true);
        Vec2f a1 = ImagePosition(trail[i], thetaE, true);
        Vec2f b0 = ImagePosition(trail[i - 1], thetaE, false);
        Vec2f b1 = ImagePosition(trail[i], thetaE, false);
        DrawLine3D(ToWorld(frame, {0.0f, a0.y * skyScale, a0.z * skyScale}), ToWorld(frame, {0.0f, a1.y * skyScale, a1.z * skyScale}),
                   Fade(Color{255, 234, 150, 255}, 0.05f + 0.28f * alpha));
        DrawLine3D(ToWorld(frame, {0.0f, b0.y * skyScale, b0.z * skyScale}), ToWorld(frame, {0.0f, b1.y * skyScale, b1.z * skyScale}),
                   Fade(Color{255, 148, 105, 255}, 0.04f + 0.18f * alpha));
    }
}

void DrawDistortedStarImages(const SceneFrame& frame, Vec2f sourceSky, float sourceRadius, float thetaE, float skyScale, float phase) {
    for (int ring = 0; ring <= 4; ++ring) {
        float rr = sourceRadius * static_cast<float>(ring) / 4.0f;
        int segments = (ring == 0) ? 1 : 44;
        Vector3 prevMajor{};
        Vector3 prevMinor{};
        bool hasPrev = false;

        for (int i = 0; i <= segments; ++i) {
            float a = (segments == 1) ? 0.0f : 2.0f * kPi * static_cast<float>(i) / static_cast<float>(segments);
            Vec2f local{rr * std::cos(a), rr * std::sin(a)};
            Vec2f beta = Add(sourceSky, local);
            Vec2f imgMajor = ImagePosition(beta, thetaE, true);
            Vec2f imgMinor = ImagePosition(beta, thetaE, false);
            Vector3 pMajor{0.0f, imgMajor.y * skyScale, imgMajor.z * skyScale};
            Vector3 pMinor{0.0f, imgMinor.y * skyScale, imgMinor.z * skyScale};

            float edge = (ring == 4) ? 1.0f : 0.62f;
            Color major = Fade(Color{255, 238, 156, 255}, 0.34f + 0.30f * edge);
            Color minor = Fade(Color{255, 156, 100, 255}, 0.24f + 0.22f * edge);

            if (hasPrev) {
                DrawLine3D(ToWorld(frame, prevMajor), ToWorld(frame, pMajor), major);
                DrawLine3D(ToWorld(frame, prevMinor), ToWorld(frame, pMinor), minor);
            }

            if (ring == 4 && i % 4 == 0) {
                DrawSphere(ToWorld(frame, pMajor), 0.045f, Color{255, 242, 174, 235});
                DrawSphere(ToWorld(frame, pMinor), 0.035f, Color{255, 165, 112, 210});
            }

            prevMajor = pMajor;
            prevMinor = pMinor;
            hasPrev = true;
        }
    }

    for (int spoke = 0; spoke < 12; ++spoke) {
        float a = 2.0f * kPi * static_cast<float>(spoke) / 12.0f + phase * 0.15f;
        Vec2f beta = Add(sourceSky, {sourceRadius * std::cos(a), sourceRadius * std::sin(a)});
        Vec2f major = ImagePosition(beta, thetaE, true);
        Vec2f minor = ImagePosition(beta, thetaE, false);
        DrawLine3D(ToWorld(frame, {0.0f, major.y * skyScale, major.z * skyScale}),
                   ToWorld(frame, {0.0f, minor.y * skyScale, minor.z * skyScale}),
                   Fade(Color{255, 210, 126, 255}, 0.12f));
    }
}

void DrawLensingScene(Camera3D camera, Vec2f sourceSky, float thetaE, float phase, const std::deque<Vec2f>& trail, bool followCamera) {
    const float skyScale = 2.55f;
    const float sourceRadius = 0.18f;
    const Vector3 observer{6.0f, 0.0f, 0.0f};
    const Vector3 source{-6.0f, sourceSky.y * skyScale, sourceSky.z * skyScale};
    const SceneFrame frame = followCamera ? CameraFollowFrame(camera) : DefaultFrame();

    Vec2f imageA = ImagePosition(sourceSky, thetaE, true);
    Vec2f imageB = ImagePosition(sourceSky, thetaE, false);
    Vector3 impactA{0.0f, imageA.y * skyScale, imageA.z * skyScale};
    Vector3 impactB{0.0f, imageB.y * skyScale, imageB.z * skyScale};

    BeginMode3D(camera);

    DrawDeepStarfield(phase);

    DrawLine3D(ToWorld(frame, {-6.4f, 0.0f, 0.0f}), ToWorld(frame, {6.25f, 0.0f, 0.0f}), Fade(Color{205, 218, 238, 255}, 0.13f));
    DrawCircleYZ(frame, -6.0f, sourceRadius * skyScale, Fade(Color{255, 230, 148, 255}, 0.44f), 96);
    DrawCircleYZ(frame, 6.0f, 0.52f, Fade(Color{120, 220, 255, 255}, 0.24f), 72);

    DrawBrightStar(frame, source, sourceRadius * skyScale, phase);
    DrawBlackHole(frame, thetaE, skyScale, phase);
    if (!followCamera) {
        DrawSphere(ToWorld(frame, observer), 0.24f, Color{150, 226, 255, 255});
        DrawSphereWires(ToWorld(frame, observer), 0.48f, 20, 12, Fade(Color{150, 226, 255, 255}, 0.22f));
    }

    DrawSphere(ToWorld(frame, impactA), 0.105f, Color{255, 238, 172, 255});
    DrawSphere(ToWorld(frame, impactB), 0.085f, Color{255, 168, 116, 255});
    DrawImageTrails(frame, trail, thetaE, skyScale);
    DrawDistortedStarImages(frame, sourceSky, sourceRadius, thetaE, skyScale, phase);

    DrawCurvedRay(frame, source, impactA, observer, Fade(Color{112, 196, 255, 255}, 0.72f), Fade(Color{255, 232, 150, 255}, 0.92f));
    DrawCurvedRay(frame, source, impactB, observer, Fade(Color{112, 196, 255, 255}, 0.46f), Fade(Color{255, 158, 116, 255}, 0.66f));
    if (!followCamera) {
        DrawPhotonPulse(frame, source, impactA, observer, std::fmod(phase * 0.26f, 1.0f), Color{255, 244, 180, 255});
        DrawPhotonPulse(frame, source, impactB, observer, std::fmod(phase * 0.21f + 0.18f, 1.0f), Color{255, 174, 122, 235});
    }

    for (int i = 0; i < 10; ++i) {
        float a = 2.0f * kPi * static_cast<float>(i) / 10.0f + 0.4f * std::sin(phase);
        Vec2f beta = Add(sourceSky, {sourceRadius * 0.75f * std::cos(a), sourceRadius * 0.75f * std::sin(a)});
        Vec2f major = ImagePosition(beta, thetaE, true);
        Vector3 impact{0.0f, major.y * skyScale, major.z * skyScale};
        Vector3 surfacePoint{-6.0f, beta.y * skyScale, beta.z * skyScale};
        DrawCurvedRay(frame, surfacePoint, impact, observer, Fade(Color{255, 222, 126, 255}, 0.20f), Fade(Color{255, 222, 126, 255}, 0.28f));
    }

    for (int i = 0; i < 26; ++i) {
        float a = 2.0f * kPi * static_cast<float>(i) / 26.0f;
        Vector3 ringImpact{0.0f, thetaE * skyScale * std::cos(a), thetaE * skyScale * std::sin(a)};
        DrawLine3D(ToWorld(frame, ringImpact), ToWorld(frame, observer), Fade(Color{255, 220, 130, 255}, 0.12f));
    }

    EndMode3D();
}

void DrawWorldLabel(Camera3D camera, Vector3 p, const char* text, Color color) {
    Vector2 s = GetWorldToScreen(p, camera);
    if (s.x < -80.0f || s.x > kScreenW + 80.0f || s.y < -40.0f || s.y > kScreenH + 40.0f) return;
    DrawText(text, static_cast<int>(s.x + 10.0f), static_cast<int>(s.y - 8.0f), 17, color);
    DrawCircleV(s, 3.0f, color);
}

void DrawLabels(Camera3D camera, Vec2f sourceSky, float thetaE) {
    const float skyScale = 2.55f;
    Vec2f major = ImagePosition(sourceSky, thetaE, true);
    Vec2f minor = ImagePosition(sourceSky, thetaE, false);
    DrawWorldLabel(camera, {-6.0f, sourceSky.y * skyScale, sourceSky.z * skyScale}, "source star", Color{255, 230, 150, 255});
    DrawWorldLabel(camera, {0.0f, 0.0f, 0.0f}, "black hole", Color{255, 170, 90, 255});
    DrawWorldLabel(camera, {0.0f, major.y * skyScale, major.z * skyScale}, "major image", Color{255, 240, 170, 255});
    DrawWorldLabel(camera, {0.0f, minor.y * skyScale, minor.z * skyScale}, "minor image", Color{255, 170, 120, 255});
    DrawWorldLabel(camera, {6.0f, 0.0f, 0.0f}, "observer", Color{145, 225, 255, 255});
}
}  // namespace

int main() {
    InitWindow(kScreenW, kScreenH, "3D Gravitational Lensing Simulation - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.8f, 5.6f, 8.4f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.74f;
    float camPitch = 0.36f;
    float camDistance = 12.8f;
    float thetaE = 0.58f;
    float time = 0.0f;
    Vec2f sourceSky{0.64f, 0.10f};
    bool paused = false;
    bool autoDrift = true;
    bool showLabels = false;
    bool followCamera = false;
    std::deque<Vec2f> imageTrail;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();

        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_SPACE)) autoDrift = !autoDrift;
        if (IsKeyPressed(KEY_L)) showLabels = !showLabels;
        if (IsKeyPressed(KEY_GRAVE)) {
            followCamera = !followCamera;
        }
        if (IsKeyPressed(KEY_R)) {
            camYaw = 0.74f;
            camPitch = 0.36f;
            camDistance = 12.8f;
            thetaE = 0.58f;
            time = 0.0f;
            sourceSky = {0.64f, 0.10f};
            paused = false;
            autoDrift = true;
            followCamera = false;
            imageTrail.clear();
        }

        if (IsKeyDown(KEY_RIGHT_BRACKET)) thetaE = std::min(1.08f, thetaE + 0.36f * dt);
        if (IsKeyDown(KEY_LEFT_BRACKET)) thetaE = std::max(0.20f, thetaE - 0.36f * dt);

        float nudge = 0.72f * dt;
        if (IsKeyDown(KEY_LEFT)) sourceSky.y -= nudge;
        if (IsKeyDown(KEY_RIGHT)) sourceSky.y += nudge;
        if (IsKeyDown(KEY_UP)) sourceSky.z += nudge;
        if (IsKeyDown(KEY_DOWN)) sourceSky.z -= nudge;

        if (!paused && autoDrift) {
            time += dt;
            sourceSky = {
                0.72f * std::sin(time * 0.42f),
                0.18f * std::sin(time * 0.76f + 0.85f)
            };
        }
        sourceSky.y = std::clamp(sourceSky.y, -1.18f, 1.18f);
        sourceSky.z = std::clamp(sourceSky.z, -0.84f, 0.84f);

        imageTrail.push_back(sourceSky);
        if (imageTrail.size() > 150) imageTrail.pop_front();

        camera.target = {0.0f, 0.0f, 0.0f};
        UpdateOrbitCamera(&camera, &camYaw, &camPitch, &camDistance);

        BeginDrawing();
        ClearBackground(Color{5, 8, 16, 255});
        DrawLensingScene(camera, sourceSky, thetaE, time, imageTrail, followCamera);
        if (showLabels) DrawLabels(camera, sourceSky, thetaE);

        DrawRectangle(20, 18, 764, 104, Fade(Color{5, 8, 16, 255}, 0.58f));
        DrawText("Distorted Starlight Around a Black Hole", 38, 34, 30, Color{232, 238, 248, 255});
        DrawText("A bright star is stretched into glowing lens arcs as its light bends past the event horizon.", 40, 70, 18, Color{176, 195, 222, 255});
        DrawText("` camera-follow sim | mouse orbit | wheel zoom | arrows move star | [ ] mass | Space drift | L labels",
                 40, 96, 18, Color{176, 195, 222, 255});

        DrawRectangle(kScreenW - 300, 18, 272, 138, Fade(Color{5, 8, 16, 255}, 0.56f));
        char status[220];
        std::snprintf(status, sizeof(status), "mass lens %.2f\nstar y %.2f  z %.2f\n%s  %s\nview %s",
                      thetaE, sourceSky.y, sourceSky.z, autoDrift ? "auto drift" : "manual",
                      paused ? "paused" : "running", followCamera ? "follow" : "free");
        DrawText(status, kScreenW - 280, 38, 20, Color{126, 224, 255, 255});
        DrawFPS(38, kScreenH - 34);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
