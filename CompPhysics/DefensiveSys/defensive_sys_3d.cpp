#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

constexpr int kPreferredWindowWidth = 1280;
constexpr int kPreferredWindowHeight = 800;
constexpr float kWindowFillRatio = 0.90f;
constexpr int kMinWindowWidth = 960;
constexpr int kMinWindowHeight = 640;
constexpr int kUdpPort = 50505;
constexpr float kTurretTurnRate = 8.0f;
constexpr float kTurretPitchRate = 8.0f;
constexpr float kMaxYaw = 120.0f * DEG2RAD;
constexpr float kMaxPitch = 78.0f * DEG2RAD;
constexpr float kMinPitch = -18.0f * DEG2RAD;
constexpr float kRearAimEnterY = 0.94f;  // enter rear mode when hand is very high
constexpr float kRearAimExitY = 0.84f;   // exit rear mode after moving hand down enough
constexpr float kRearAimBlendRate = 7.5f;
constexpr float kShotCooldown = 0.22f;
constexpr float kShotRange = 180.0f;
constexpr float kShotFxTime = 0.12f;
constexpr float kPlaneExplosionTime = 0.72f;
constexpr float kPlaneRespawnDelay = 1.45f;

struct TrackingInput {
    bool leftValid = false;
    float leftX = 0.5f;
    float leftY = 0.5f;
    bool rightValid = false;
    bool rightPinch = false;
    double timestamp = 0.0;
};

struct Plane {
    Vector3 pos{};
    float speed = 0.0f;
    float size = 1.5f;
    bool enemy = false;
    bool alive = true;
    float explosionUntil = 0.0f;
    float respawnAt = 0.0f;
};

struct ShotFx {
    bool active = false;
    bool hit = false;
    float until = 0.0f;
    Vector3 from{};
    Vector3 to{};
};

float RandomRange(float minV, float maxV) {
    const float u = static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f;
    return minV + (maxV - minV) * u;
}

float NormalizeAngle(float a) {
    return std::atan2(std::sin(a), std::cos(a));
}

float LerpAngle(float from, float to, float alpha) {
    const float delta = NormalizeAngle(to - from);
    return NormalizeAngle(from + delta * alpha);
}

void SpawnPlane(Plane &plane, int direction) {
    plane.size = RandomRange(1.35f, 2.15f);
    plane.speed = RandomRange(15.0f, 24.0f) * static_cast<float>(direction);
    plane.pos.x = (direction > 0) ? -78.0f : 78.0f;
    plane.pos.y = RandomRange(8.0f, 22.0f);
    plane.pos.z = RandomRange(-30.0f, 30.0f);
    plane.alive = true;
    plane.explosionUntil = 0.0f;
    plane.respawnAt = 0.0f;
}

void InitPlanes(std::vector<Plane> &planes) {
    planes.clear();
    planes.reserve(10);
    for (int i = 0; i < 10; ++i) {
        Plane plane{};
        plane.enemy = (i % 2 == 1);
        const int direction = (i % 2 == 0) ? 1 : -1;
        SpawnPlane(plane, direction);
        plane.pos.z += (i - 5) * 1.6f;
        planes.push_back(plane);
    }
}

void DrawPlane3D(const Plane &plane) {
    const Color body = plane.enemy ? Color{230, 70, 72, 255} : Color{72, 132, 236, 255};
    const Color wing = plane.enemy ? Color{190, 60, 62, 255} : Color{62, 112, 196, 255};
    const Color glass = Color{220, 236, 255, 255};
    const float nose = (plane.speed >= 0.0f) ? 1.0f : -1.0f;
    const Vector3 p = plane.pos;
    const float s = plane.size;

    DrawCube({p.x - 1.2f * s * nose, p.y, p.z}, 1.8f * s, 0.45f * s, 0.55f * s, body);
    DrawCube({p.x - 2.0f * s * nose, p.y + 0.1f * s, p.z}, 0.8f * s, 0.36f * s, 2.6f * s, wing);
    DrawCube({p.x - 0.35f * s * nose, p.y + 0.05f * s, p.z}, 1.1f * s, 0.25f * s, 0.44f * s, glass);
    DrawSphere({p.x + 0.05f * s * nose, p.y, p.z}, 0.3f * s, body);
}

class UdpBridgeReceiver {
  public:
    bool Start(uint16_t port) {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) return false;
#endif
        socket_ = static_cast<int>(socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP));
        if (socket_ < 0) return false;

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port);
        if (bind(socket_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
            Close();
            return false;
        }

#ifdef _WIN32
        u_long mode = 1;
        if (ioctlsocket(socket_, FIONBIO, &mode) != 0) {
            Close();
            return false;
        }
#else
        const int flags = fcntl(socket_, F_GETFL, 0);
        if (flags < 0 || fcntl(socket_, F_SETFL, flags | O_NONBLOCK) < 0) {
            Close();
            return false;
        }
#endif
        ready_ = true;
        return true;
    }

    void Close() {
        if (socket_ >= 0) {
#ifdef _WIN32
            closesocket(socket_);
            WSACleanup();
#else
            close(socket_);
#endif
            socket_ = -1;
        }
        ready_ = false;
    }

    bool Poll(TrackingInput &outInput, int &packetsRead) {
        packetsRead = 0;
        if (!ready_) return false;
        bool gotAny = false;
        std::array<char, 256> buffer{};

        while (true) {
            sockaddr_in src{};
            socklen_t srcLen = sizeof(src);
            const int n = recvfrom(socket_, buffer.data(), static_cast<int>(buffer.size()) - 1, 0,
                                   reinterpret_cast<sockaddr *>(&src), &srcLen);
            if (n <= 0) {
#ifdef _WIN32
                if (WSAGetLastError() == WSAEWOULDBLOCK) break;
#else
                if (errno == EWOULDBLOCK || errno == EAGAIN) break;
#endif
                break;
            }

            buffer[static_cast<size_t>(n)] = '\0';
            TrackingInput parsed{};
            int lv = 0;
            int rv = 0;
            int rp = 0;
            if (std::sscanf(buffer.data(), "%lf,%d,%f,%f,%d,%d",
                            &parsed.timestamp, &lv, &parsed.leftX, &parsed.leftY, &rv, &rp) == 6) {
                parsed.leftValid = (lv != 0);
                parsed.rightValid = (rv != 0);
                parsed.rightPinch = (rp != 0);
                parsed.leftX = std::clamp(parsed.leftX, 0.0f, 1.0f);
                parsed.leftY = std::clamp(parsed.leftY, 0.0f, 1.0f);
                outInput = parsed;
                gotAny = true;
                packetsRead++;
            }
        }
        return gotAny;
    }

    bool ready() const { return ready_; }

  private:
    int socket_ = -1;
    bool ready_ = false;
};

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(kPreferredWindowWidth, kPreferredWindowHeight, "DefensiveSys 3D Turret + Planes (raylib)");
    SetTargetFPS(120);
    SetWindowMinSize(kMinWindowWidth, kMinWindowHeight);
    const int monitor = GetCurrentMonitor();
    const int monitorW = GetMonitorWidth(monitor);
    const int monitorH = GetMonitorHeight(monitor);
    const int desiredW = std::min(kPreferredWindowWidth, static_cast<int>(monitorW * kWindowFillRatio));
    const int desiredH = std::min(kPreferredWindowHeight, static_cast<int>(monitorH * kWindowFillRatio));
    SetWindowSize(std::max(kMinWindowWidth, desiredW), std::max(kMinWindowHeight, desiredH));
    SetWindowPosition((monitorW - GetScreenWidth()) / 2, (monitorH - GetScreenHeight()) / 2);

    Camera3D cam{};
    cam.position = {0.0f, 24.0f, 46.0f};
    cam.target = {0.0f, 10.0f, 0.0f};
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.fovy = 50.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    UdpBridgeReceiver receiver;
    const bool receiverOk = receiver.Start(static_cast<uint16_t>(kUdpPort));
    TrackingInput tracking{};
    double lastPacketWallClock = 0.0;

    std::vector<Plane> planes;
    InitPlanes(planes);

    float turretYaw = 0.0f;
    float turretPitch = 0.0f;
    float targetYaw = 0.0f;
    float targetPitch = 0.0f;
    bool rearAimMode = false;
    float rearAimBlend = 0.0f;
    bool prevRightPinch = false;
    float lastShotTime = -10.0f;
    ShotFx shotFx{};

    int shots = 0;
    int enemyKills = 0;
    int friendlyHits = 0;
    std::string lastEvent = "ready";

    while (!WindowShouldClose()) {
        const float now = static_cast<float>(GetTime());
        const float dt = GetFrameTime();

        int packetsRead = 0;
        if (receiver.Poll(tracking, packetsRead)) {
            lastPacketWallClock = static_cast<double>(now);
        }
        const bool linkLive = receiver.ready() && ((now - static_cast<float>(lastPacketWallClock)) < 0.60f);

        if (tracking.leftValid) {
            const float yNorm = std::clamp(1.0f - tracking.leftY, 0.0f, 1.0f);
            const float xNorm = std::clamp((tracking.leftX - 0.5f) * 2.0f, -1.0f, 1.0f);

            if (!rearAimMode && yNorm >= kRearAimEnterY) {
                rearAimMode = true;
            } else if (rearAimMode && yNorm <= kRearAimExitY) {
                rearAimMode = false;
            }
            const float rearTarget = rearAimMode ? 1.0f : 0.0f;
            rearAimBlend = Lerp(rearAimBlend, rearTarget, 1.0f - std::exp(-kRearAimBlendRate * dt));

            const float baseYaw = xNorm * kMaxYaw;
            targetYaw = NormalizeAngle(baseYaw + rearAimBlend * PI);
            targetPitch = kMinPitch + (kMaxPitch - kMinPitch) * yNorm;
            targetPitch = std::clamp(targetPitch, kMinPitch, kMaxPitch);
        }
        turretYaw = LerpAngle(turretYaw, targetYaw, 1.0f - std::exp(-kTurretTurnRate * dt));
        turretPitch = Lerp(turretPitch, targetPitch, 1.0f - std::exp(-kTurretPitchRate * dt));

        const Vector3 turretPivot = {0.0f, 1.6f, 0.0f};
        Vector3 fireDir = {
            std::sin(turretYaw) * std::cos(turretPitch),
            std::sin(turretPitch),
            std::cos(turretYaw) * std::cos(turretPitch),
        };
        fireDir = Vector3Normalize(fireDir);
        const Vector3 muzzle = Vector3Add(turretPivot, Vector3Scale(fireDir, 9.0f));

        for (Plane &plane : planes) {
            if (plane.alive) {
                plane.pos.x += plane.speed * dt;
                if ((plane.speed > 0.0f && plane.pos.x > 88.0f) || (plane.speed < 0.0f && plane.pos.x < -88.0f)) {
                    SpawnPlane(plane, (plane.speed >= 0.0f) ? 1 : -1);
                }
            } else if (now >= plane.respawnAt) {
                SpawnPlane(plane, (plane.speed >= 0.0f) ? 1 : -1);
            }
        }

        const bool rightPinch = tracking.rightValid && tracking.rightPinch;
        if (rightPinch && !prevRightPinch && (now - lastShotTime >= kShotCooldown)) {
            shots++;
            lastShotTime = now;
            shotFx.active = true;
            shotFx.hit = false;
            shotFx.until = now + kShotFxTime;
            shotFx.from = muzzle;
            shotFx.to = Vector3Add(muzzle, Vector3Scale(fireDir, kShotRange));

            int bestIdx = -1;
            float bestT = 1e9f;
            for (int i = 0; i < static_cast<int>(planes.size()); ++i) {
                const Plane &plane = planes[static_cast<size_t>(i)];
                if (!plane.alive) continue;

                const Vector3 rel = Vector3Subtract(plane.pos, muzzle);
                const float t = Vector3DotProduct(rel, fireDir);
                if (t < 0.0f || t > kShotRange) continue;

                const Vector3 closest = Vector3Add(muzzle, Vector3Scale(fireDir, t));
                const float dist = Vector3Distance(closest, plane.pos);
                const float hitRadius = 1.15f * plane.size + 0.85f;
                if (dist <= hitRadius && t < bestT) {
                    bestT = t;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0) {
                Plane &hit = planes[static_cast<size_t>(bestIdx)];
                hit.alive = false;
                hit.explosionUntil = now + kPlaneExplosionTime;
                hit.respawnAt = now + kPlaneRespawnDelay;
                shotFx.hit = true;
                shotFx.to = hit.pos;
                if (hit.enemy) {
                    enemyKills++;
                    lastEvent = "enemy down";
                } else {
                    friendlyHits++;
                    lastEvent = "friendly hit";
                }
            } else {
                lastEvent = "miss";
            }
        }
        prevRightPinch = rightPinch;

        BeginDrawing();
        ClearBackground(Color{14, 18, 26, 255});

        BeginMode3D(cam);
        DrawGrid(36, 2.0f);
        DrawPlane({0.0f, 0.0f, 0.0f}, {220.0f, 220.0f}, Color{22, 30, 38, 255});

        for (const Plane &plane : planes) {
            if (plane.alive) {
                DrawPlane3D(plane);
            } else if (now < plane.explosionUntil) {
                const float life = std::clamp((plane.explosionUntil - now) / kPlaneExplosionTime, 0.0f, 1.0f);
                const float progress = 1.0f - life;
                const int aCore = static_cast<int>(255.0f * life);
                const int aWave = static_cast<int>(200.0f * life);
                const float coreRadius = (0.9f + 1.2f * progress) * plane.size;
                const float waveRadius = (1.8f + 4.6f * progress) * plane.size;

                DrawSphere(plane.pos, coreRadius, Color{255, 232, 156, static_cast<unsigned char>(aCore)});
                DrawSphereWires(plane.pos, waveRadius, 10, 12, Color{255, 120, 70, static_cast<unsigned char>(aWave)});

                for (int j = 0; j < 8; ++j) {
                    const float jf = static_cast<float>(j);
                    const float ang = jf * PI / 4.0f + progress * 5.0f;
                    const float ring = (1.5f + progress * 3.8f) * plane.size;
                    const Vector3 spark{
                        plane.pos.x + std::cos(ang) * ring,
                        plane.pos.y + std::sin(ang * 1.7f) * 0.9f * plane.size,
                        plane.pos.z + std::sin(ang) * ring,
                    };
                    DrawSphere(spark, 0.22f * plane.size * life, Color{255, 170, 84, static_cast<unsigned char>(aCore)});
                    DrawLine3D(plane.pos, spark, Color{255, 136, 82, static_cast<unsigned char>(aWave)});
                }
            }
        }

        DrawCylinder({0.0f, 0.7f, 0.0f}, 2.6f, 2.8f, 1.4f, 24, Color{70, 78, 90, 255});
        DrawCylinder({0.0f, 1.45f, 0.0f}, 1.75f, 1.85f, 0.95f, 24, Color{96, 106, 118, 255});
        DrawCylinderEx(turretPivot, muzzle, 0.50f, 0.42f, 12, Color{142, 154, 170, 255});
        DrawSphere(muzzle, 0.48f, Color{212, 222, 236, 255});

        if (shotFx.active && now <= shotFx.until) {
            const Color c = shotFx.hit ? Color{120, 246, 176, 255} : Color{120, 196, 255, 255};
            DrawLine3D(shotFx.from, shotFx.to, c);
            DrawSphere(shotFx.to, shotFx.hit ? 0.65f : 0.35f, c);
        } else {
            shotFx.active = false;
        }
        EndMode3D();

        const int screenW = GetScreenWidth();
        const int screenH = GetScreenHeight();
        const int hudHeight = 94;
        const int hudY = std::max(0, screenH - hudHeight);
        DrawRectangle(0, hudY, screenW, hudHeight, Color{8, 10, 14, 190});
        DrawText("DefensiveSys 3D Bridge", 18, hudY + 10, 24, RAYWHITE);

        const char *linkText = linkLive ? "link:connected" : "link:waiting";
        DrawText(TextFormat("%s  udp:%d  left:%s (%.2f, %.2f)  right:%s pinch:%s",
                            linkText, kUdpPort,
                            tracking.leftValid ? "ok" : "none", tracking.leftX, tracking.leftY,
                            tracking.rightValid ? "ok" : "none",
                            rightPinch ? "down" : "up"),
                 18, hudY + 38, 18, Color{186, 206, 232, 255});

        DrawText(TextFormat("enemy kills:%d  friendly hits:%d  shots:%d  event:%s",
                            enemyKills, friendlyHits, shots, lastEvent.c_str()),
                 18, hudY + 62, 18, Color{172, 196, 224, 255});

        EndDrawing();
    }

    receiver.Close();
    CloseWindow();
    return 0;
}
