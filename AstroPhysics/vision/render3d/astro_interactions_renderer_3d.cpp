#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;
constexpr float kPi = 3.14159265358979323846f;
constexpr int kUdpPort = 50506;
constexpr float kBridgeTimeoutSec = 0.55f;
constexpr float kRemoteSmooth = 10.0f;
constexpr float kObjectLiftScale = 1.10f;

enum class AstroKind : int {
    Planet = 0,
    Star = 1,
    BlackHole = 2,
};

struct RemoteInput {
    bool leftValid = false;
    float leftX = 0.5f;
    float leftY = 0.5f;
    bool rightValid = false;
    float rightX = 0.5f;
    float rightY = 0.5f;
    int leftKind = 0;
    int rightKind = 1;
    bool interactionHint = false;
    float interactionStrength = 0.0f;
    float leftGrip = 0.0f;
    float rightGrip = 0.0f;
    double timestamp = 0.0;
};

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
        if (bind(socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
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

    bool Poll(RemoteInput& outInput, int& packetsRead) {
        packetsRead = 0;
        if (!ready_) return false;

        bool gotAny = false;
        std::array<char, 320> buffer{};

        while (true) {
            sockaddr_in src{};
            socklen_t srcLen = sizeof(src);
            const int n = recvfrom(socket_, buffer.data(), static_cast<int>(buffer.size()) - 1, 0,
                                   reinterpret_cast<sockaddr*>(&src), &srcLen);
            if (n <= 0) {
#ifdef _WIN32
                if (WSAGetLastError() == WSAEWOULDBLOCK) break;
#else
                if (errno == EWOULDBLOCK || errno == EAGAIN) break;
#endif
                break;
            }
            buffer[static_cast<size_t>(n)] = '\0';

            RemoteInput parsed{};
            int lv = 0;
            int rv = 0;
            int ia = 0;
            int got = std::sscanf(
                buffer.data(),
                "%lf,%d,%f,%f,%d,%f,%f,%d,%d,%d,%f,%f,%f",
                &parsed.timestamp,
                &lv,
                &parsed.leftX,
                &parsed.leftY,
                &rv,
                &parsed.rightX,
                &parsed.rightY,
                &parsed.leftKind,
                &parsed.rightKind,
                &ia,
                &parsed.interactionStrength,
                &parsed.leftGrip,
                &parsed.rightGrip
            );

            if (got >= 11) {
                parsed.leftValid = (lv != 0);
                parsed.rightValid = (rv != 0);
                parsed.interactionHint = (ia != 0);
                if (got < 13) {
                    parsed.leftGrip = 0.0f;
                    parsed.rightGrip = 0.0f;
                }

                parsed.leftX = std::clamp(parsed.leftX, 0.0f, 1.0f);
                parsed.leftY = std::clamp(parsed.leftY, 0.0f, 1.0f);
                parsed.rightX = std::clamp(parsed.rightX, 0.0f, 1.0f);
                parsed.rightY = std::clamp(parsed.rightY, 0.0f, 1.0f);
                parsed.leftKind = std::clamp(parsed.leftKind, 0, 2);
                parsed.rightKind = std::clamp(parsed.rightKind, 0, 2);
                parsed.interactionStrength = std::clamp(parsed.interactionStrength, 0.0f, 1.0f);
                parsed.leftGrip = std::clamp(parsed.leftGrip, 0.0f, 1.0f);
                parsed.rightGrip = std::clamp(parsed.rightGrip, 0.0f, 1.0f);

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

struct BodyState {
    AstroKind kind = AstroKind::Planet;
    Vector3 pos = {0.0f, 1.8f, 0.0f};
    float radius = 1.0f;
    bool valid = true;
};

struct InteractionState {
    bool active = false;
    float closeTimer = 0.0f;
    float blend = 0.0f;
    Vector3 center = {0.0f, 1.8f, 0.0f};
    float radius = 1.2f;
};

float BaseRadiusFor(AstroKind kind) {
    switch (kind) {
        case AstroKind::Planet: return 1.0f;
        case AstroKind::Star: return 1.15f;
        case AstroKind::BlackHole: return 1.05f;
    }
    return 1.0f;
}

AstroKind KindFromId(int id) {
    id = std::clamp(id, 0, 2);
    return static_cast<AstroKind>(id);
}

const char* KindName(AstroKind kind) {
    switch (kind) {
        case AstroKind::Planet: return "Planet";
        case AstroKind::Star: return "Star";
        case AstroKind::BlackHole: return "Black Hole";
    }
    return "Unknown";
}

AstroKind NextKind(AstroKind kind) {
    return static_cast<AstroKind>((static_cast<int>(kind) + 1) % 3);
}

Vector3 HandSpaceToWorld(float x, float y, bool rightSide) {
    const float worldX = Lerp(-9.0f, 9.0f, x) + (rightSide ? 0.75f : -0.75f);
    const float worldY = Lerp(0.8f, 6.2f, 1.0f - y);
    const float worldZ = rightSide ? 1.7f : -1.7f;
    return {worldX, worldY, worldZ};
}

void DrawFingerChain3D(
    const Vector3& base,
    Vector3 dir,
    const Vector3& bendAxis,
    const std::array<float, 3>& segLen,
    float curl,
    float baseRadius,
    Color color
) {
    Vector3 p0 = base;
    dir = Vector3Normalize(dir);
    for (int i = 0; i < 3; ++i) {
        const float w = (i == 0) ? 0.85f : (i == 1 ? 1.10f : 0.95f);
        const float a = curl * w;
        dir = Vector3Normalize(Vector3RotateByAxisAngle(dir, bendAxis, a));
        const Vector3 p1 = Vector3Add(p0, Vector3Scale(dir, segLen[static_cast<size_t>(i)]));

        const float r0 = baseRadius * (1.00f - 0.20f * static_cast<float>(i));
        const float r1 = baseRadius * (0.82f - 0.18f * static_cast<float>(i));
        DrawCylinderEx(p0, p1, r0, r1, 10, color);
        DrawSphere(p1, r1 * 0.92f, color);
        p0 = p1;
    }
}

void DrawHandAvatar3D(const Vector3& palm, bool rightHand, float grip, float t) {
    const float side = rightHand ? 1.0f : -1.0f;
    const float clench = std::clamp(grip, 0.0f, 1.0f);
    const Vector3 sideDir = {side, 0.0f, 0.0f};
    const Vector3 upDir = {0.0f, 1.0f, 0.0f};
    const Vector3 fwdDir = {0.0f, 0.0f, 1.0f};
    const Color palmColor = rightHand ? Color{246, 180, 130, 205} : Color{128, 198, 255, 205};
    const Color fingerColor = rightHand ? Color{236, 166, 114, 210} : Color{106, 176, 245, 210};
    const Color glowColor = rightHand ? Color{255, 222, 150, 150} : Color{138, 224, 255, 150};

    DrawSphere(palm, 0.38f, palmColor);
    DrawSphere(Vector3Add(palm, Vector3Scale(upDir, 0.10f)), 0.34f, Fade(palmColor, 0.85f));
    DrawCylinderEx(
        Vector3Add(palm, Vector3Scale(sideDir, -0.22f)),
        Vector3Add(palm, Vector3Scale(sideDir, 0.22f)),
        0.25f, 0.25f, 12, Fade(palmColor, 0.92f)
    );
    DrawSphereWires(palm, 0.43f, 10, 10, Fade(BLACK, 0.45f));
    DrawSphere(Vector3Add(palm, {0.0f, 0.07f, 0.0f}), 0.46f, Color{255, 255, 255, 24});

    const std::array<float, 4> spread = {0.24f, 0.08f, -0.08f, -0.24f};
    const std::array<std::array<float, 3>, 4> segs = {
        std::array<float, 3>{0.27f, 0.23f, 0.18f},  // index
        std::array<float, 3>{0.31f, 0.25f, 0.20f},  // middle
        std::array<float, 3>{0.29f, 0.24f, 0.19f},  // ring
        std::array<float, 3>{0.24f, 0.21f, 0.17f},  // pinky
    };
    for (int i = 0; i < 4; ++i) {
        const float sf = spread[static_cast<size_t>(i)] * side;
        const Vector3 base = Vector3Add(
            palm,
            Vector3Add(
                Vector3Scale(sideDir, sf),
                Vector3Add(
                    Vector3Scale(upDir, 0.16f + 0.01f * std::cos(t * 1.6f + i)),
                    Vector3Scale(fwdDir, 0.04f - 0.03f * std::abs(sf))
                )
            )
        );
        const float idleCurl = 0.08f * std::sin(t * 2.2f + 0.7f * static_cast<float>(i));
        const float curl = (0.10f + 1.00f * clench) + idleCurl;
        const Vector3 dir = Vector3Normalize(
            Vector3Add(
                Vector3Scale(upDir, 1.0f),
                Vector3Add(
                    Vector3Scale(fwdDir, 0.16f + 0.05f * std::sin(t + i)),
                    Vector3Scale(sideDir, 0.10f * sf)
                )
            )
        );
        DrawFingerChain3D(base, dir, sideDir, segs[static_cast<size_t>(i)], curl, 0.073f, fingerColor);
    }

    // Thumb with its own articulation axis and diagonal base direction.
    const Vector3 thumbBase = Vector3Add(
        palm,
        Vector3Add(Vector3Scale(sideDir, 0.28f), Vector3Add(Vector3Scale(upDir, 0.02f), Vector3Scale(fwdDir, 0.14f)))
    );
    const Vector3 thumbDir = Vector3Normalize(
        Vector3Add(Vector3Scale(sideDir, 0.85f), Vector3Add(Vector3Scale(upDir, 0.36f), Vector3Scale(fwdDir, 0.26f)))
    );
    const Vector3 thumbAxis = Vector3Normalize(Vector3CrossProduct(thumbDir, upDir));
    const float thumbIdle = 0.07f * std::sin(t * 2.6f + (rightHand ? 1.2f : 0.4f));
    const float thumbCurl = (0.24f + 0.92f * clench) + thumbIdle;
    DrawFingerChain3D(thumbBase, thumbDir, thumbAxis, {0.24f, 0.20f, 0.16f}, thumbCurl, 0.077f, fingerColor);

    const float pulse = 0.5f + 0.5f * std::sin(2.0f * t + (rightHand ? 1.0f : 0.0f));
    DrawSphere(palm, 0.52f + 0.04f * pulse, Fade(glowColor, 0.25f));
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0033f;
        *pitch += d.y * 0.0033f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.95f;
    *distance = std::clamp(*distance, 8.0f, 52.0f);

    const float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    });
}

void DrawRingXZ(Vector3 c, float rx, float rz, float yOff, int seg, float phase, Color color) {
    Vector3 prev = {
        c.x + rx * std::cos(phase),
        c.y + yOff,
        c.z + rz * std::sin(phase),
    };
    for (int i = 1; i <= seg; ++i) {
        const float a = phase + (2.0f * kPi * static_cast<float>(i)) / static_cast<float>(seg);
        const Vector3 cur = {
            c.x + rx * std::cos(a),
            c.y + yOff,
            c.z + rz * std::sin(a),
        };
        DrawLine3D(prev, cur, color);
        prev = cur;
    }
}

void DrawPlanet3D(Vector3 pos, float r, float t) {
    DrawSphere(pos, r, Color{70, 140, 255, 255});
    DrawSphere(pos, r * 1.03f, Color{120, 190, 255, 52});

    for (int i = -3; i <= 3; ++i) {
        const float y = static_cast<float>(i) * r * 0.16f;
        const float rr = r * std::sqrt(std::max(0.0f, 1.0f - (y * y) / (r * r)));
        DrawRingXZ(pos, rr, rr, y, 44, t * 0.25f + i * 0.4f, Color{90, 180, 255, 180});
    }

    for (int i = 0; i < 18; ++i) {
        const float a = (2.0f * kPi * i) / 18.0f + t * 0.65f;
        const Vector3 p = {
            pos.x + 0.86f * r * std::cos(a),
            pos.y + 0.22f * r * std::sin(t * 0.8f + i),
            pos.z + 0.86f * r * std::sin(a),
        };
        DrawSphere(p, 0.045f * r, Color{210, 235, 255, 120});
    }
}

void DrawStar3D(Vector3 pos, float r, float t) {
    const float pulse = 0.90f + 0.18f * (0.5f + 0.5f * std::sin(t * 3.4f));
    DrawSphere(pos, r * pulse, Color{255, 214, 118, 255});
    DrawSphere(pos, r * (1.35f + 0.12f * std::sin(t * 2.5f)), Color{255, 186, 92, 46});
    DrawSphere(pos, r * (1.80f + 0.14f * std::sin(t * 1.8f + 1.1f)), Color{255, 164, 80, 24});

    for (int i = 0; i < 26; ++i) {
        const float a = (2.0f * kPi * i) / 26.0f + t * 1.9f;
        const float inner = r * 1.02f;
        const float outer = r * (1.55f + 0.25f * std::sin(t * 5.0f + i * 0.63f));
        const Vector3 p0 = {pos.x + inner * std::cos(a), pos.y, pos.z + inner * std::sin(a)};
        const Vector3 p1 = {pos.x + outer * std::cos(a), pos.y + 0.18f * r * std::sin(t * 2.4f + i), pos.z + outer * std::sin(a)};
        DrawLine3D(p0, p1, Color{255, 228, 145, 185});
    }
}

void DrawBlackHole3D(Vector3 pos, float r, float t) {
    DrawSphere(pos, r * 0.95f, Color{7, 8, 12, 255});
    DrawSphereWires(pos, r * 1.03f, 16, 16, Color{45, 55, 80, 190});

    for (int j = 0; j < 4; ++j) {
        const float rr = r * (1.45f + 0.20f * static_cast<float>(j));
        const float wobble = 0.05f * std::sin(t * 2.2f + j * 0.9f);
        DrawRingXZ(pos, rr, rr * (0.60f + wobble), 0.04f * static_cast<float>(j - 1), 64, t * (1.4f + 0.2f * j), Color{255, 170, 88, 190});
    }
    DrawRingXZ(pos, r * 1.15f, r * 1.12f, 0.0f, 72, t * 1.2f, Color{120, 190, 255, 175});
}

void DrawBody3D(const BodyState& body, float t) {
    if (!body.valid) return;
    switch (body.kind) {
        case AstroKind::Planet: DrawPlanet3D(body.pos, body.radius, t); break;
        case AstroKind::Star: DrawStar3D(body.pos, body.radius, t); break;
        case AstroKind::BlackHole: DrawBlackHole3D(body.pos, body.radius, t); break;
    }
}

int PairCode(AstroKind a, AstroKind b) {
    int x = static_cast<int>(a);
    int y = static_cast<int>(b);
    if (x > y) std::swap(x, y);
    return (x << 8) | y;
}

const char* PairLabel(int code) {
    if (code == PairCode(AstroKind::Planet, AstroKind::Planet)) return "planet collision";
    if (code == PairCode(AstroKind::Planet, AstroKind::Star)) return "planet evaporation";
    if (code == PairCode(AstroKind::Planet, AstroKind::BlackHole)) return "spaghettification";
    if (code == PairCode(AstroKind::Star, AstroKind::Star)) return "supernova";
    if (code == PairCode(AstroKind::Star, AstroKind::BlackHole)) return "tidal disruption";
    if (code == PairCode(AstroKind::BlackHole, AstroKind::BlackHole)) return "bh merger";
    return "interaction";
}

void DrawInteraction3D(const BodyState& left, const BodyState& right, const InteractionState& st, float t) {
    const int code = PairCode(left.kind, right.kind);
    const Vector3 c = st.center;
    const float r = st.radius;
    const float s = std::clamp(st.blend, 0.0f, 1.0f);

    if (code == PairCode(AstroKind::Planet, AstroKind::Planet)) {
        DrawPlanet3D(Vector3Add(c, {-0.36f * r, 0.0f, 0.0f}), 0.62f * r, t);
        DrawPlanet3D(Vector3Add(c, {+0.36f * r, 0.0f, 0.0f}), 0.62f * r, t + 0.7f);
        for (int i = 0; i < 44; ++i) {
            const float a = t * 1.7f + (2.0f * kPi * i) / 44.0f;
            const float rr = r * (1.05f + 0.45f * static_cast<float>(i % 7) / 6.0f);
            const Vector3 p = {c.x + rr * std::cos(a), c.y + 0.17f * r * std::sin(t * 1.2f + i), c.z + rr * std::sin(a)};
            DrawSphere(p, 0.032f * r, Color{95, 178, 255, 210});
        }
    } else if (code == PairCode(AstroKind::Planet, AstroKind::Star)) {
        DrawStar3D(c, 0.95f * r, t);
        const float dir = t * 0.32f;
        const Vector3 u = {std::cos(dir), 0.0f, std::sin(dir)};
        const Vector3 v = {-u.z, 0.0f, u.x};
        for (int i = 0; i < 50; ++i) {
            const float f = static_cast<float>(i) / 49.0f;
            const Vector3 p = {
                c.x - (1.0f + 2.1f * f) * r * u.x + 0.35f * r * std::sin(t * 4.2f + i) * v.x,
                c.y + 0.12f * r * std::sin(t * 2.0f + i * 0.41f),
                c.z - (1.0f + 2.1f * f) * r * u.z + 0.35f * r * std::sin(t * 4.2f + i) * v.z,
            };
            DrawSphere(p, (0.030f + 0.036f * (1.0f - f)) * r, Color{135, 218, 255, 220});
        }
    } else if (code == PairCode(AstroKind::Planet, AstroKind::BlackHole)) {
        DrawBlackHole3D(c, 0.96f * r, t);
        for (int i = 0; i < 58; ++i) {
            const float f = static_cast<float>(i) / 57.0f;
            const float a = t * 2.5f + 12.0f * f;
            const float rr = r * (2.2f - 1.8f * f);
            const Vector3 p = {c.x + rr * std::cos(a), c.y + 0.35f * r * (0.5f - f), c.z + 0.62f * rr * std::sin(a)};
            DrawSphere(p, (0.022f + 0.045f * (1.0f - f)) * r, Color{115, 198, 255, 220});
        }
    } else if (code == PairCode(AstroKind::Star, AstroKind::Star)) {
        DrawStar3D(c, 0.42f * r, t * 1.4f);
        const float shell = r * (1.20f + 1.05f * (0.5f + 0.5f * std::sin(t * 5.2f)));
        DrawSphereWires(c, shell, 16, 16, Color{205, 245, 255, 235});
        DrawSphereWires(c, shell * 0.72f, 12, 12, Color{145, 220, 255, 210});
        for (int i = 0; i < 36; ++i) {
            const float a = t * 2.4f + (2.0f * kPi * i) / 36.0f;
            const float rr = r * (1.2f + 1.1f * static_cast<float>(i % 8) / 7.0f);
            const Vector3 p = {c.x + rr * std::cos(a), c.y + 0.22f * r * std::sin(t * 3.0f + i), c.z + rr * std::sin(a)};
            DrawSphere(p, 0.030f * r, Color{160, 234, 255, 230});
        }
        const float beam = r * (2.1f + 0.9f * s);
        const float ba = t * 2.2f;
        const Vector3 d = {beam * std::cos(ba), beam * 0.24f * std::sin(t * 0.7f), beam * std::sin(ba)};
        DrawLine3D(Vector3Subtract(c, d), Vector3Add(c, d), Color{165, 235, 255, 220});
    } else if (code == PairCode(AstroKind::Star, AstroKind::BlackHole)) {
        DrawBlackHole3D(c, 1.03f * r, t);
        for (int i = 0; i < 34; ++i) {
            const float f = static_cast<float>(i) / 33.0f;
            const float a = t * 2.2f + 8.6f * f;
            const float rr = r * (2.1f - 1.1f * f);
            const Vector3 p = {c.x + rr * std::cos(a), c.y + 0.18f * r * std::sin(t * 2.1f + i), c.z + 0.58f * rr * std::sin(a)};
            DrawSphere(p, (0.032f + 0.032f * (1.0f - f)) * r, Color{135, 220, 255, 220});
        }
        const float jet = r * (2.6f + 0.8f * s);
        DrawLine3D(c, Vector3Add(c, {0.28f * r, jet, 0.0f}), Color{120, 220, 255, 235});
        DrawLine3D(c, Vector3Add(c, {0.28f * r, -jet, 0.0f}), Color{120, 220, 255, 235});
    } else if (code == PairCode(AstroKind::BlackHole, AstroKind::BlackHole)) {
        DrawBlackHole3D(c, 1.16f * r, t);
        for (int i = 0; i < 5; ++i) {
            const float rr = r * (1.15f + 0.42f * static_cast<float>(i) + 0.22f * std::sin(t * 4.4f + i));
            DrawSphereWires(c, rr, 12, 12, Color{120, 200, 255, 180});
        }
        const float a = t * 3.0f;
        const Vector3 p = {c.x + 0.56f * r * std::cos(a), c.y, c.z + 0.56f * r * std::sin(a)};
        DrawSphere(p, 0.12f * r, Color{145, 220, 255, 210});
    }
}

void DrawBackdrop(const std::vector<Vector3>& stars) {
    for (const Vector3& p : stars) {
        DrawSphere(p, 0.05f, Color{198, 218, 255, 185});
    }
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Vision Astro Interactions 3D (UDP Bridge)");
    SetWindowMinSize(960, 640);
    SetTargetFPS(120);

    Camera3D camera{};
    camera.position = {13.0f, 8.0f, 17.0f};
    camera.target = {0.0f, 2.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 46.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.90f;
    float camPitch = 0.28f;
    float camDistance = 21.0f;

    UdpBridgeReceiver receiver;
    const bool receiverOk = receiver.Start(static_cast<uint16_t>(kUdpPort));
    RemoteInput remote{};
    float lastPacketWallClock = -100.0f;

    BodyState left{};
    left.kind = AstroKind::Planet;
    left.radius = BaseRadiusFor(left.kind);
    left.valid = true;

    BodyState right{};
    right.kind = AstroKind::Star;
    right.radius = BaseRadiusFor(right.kind);
    right.valid = true;
    Vector3 leftHand = {-4.2f, 1.0f, 0.0f};
    Vector3 rightHand = {4.2f, 1.0f, 0.0f};
    left.pos = Vector3Add(leftHand, {0.0f, kObjectLiftScale * left.radius, 0.0f});
    right.pos = Vector3Add(rightHand, {0.0f, kObjectLiftScale * right.radius, 0.0f});
    float leftGrip = 0.0f;
    float rightGrip = 0.0f;

    InteractionState interaction{};

    std::vector<Vector3> stars;
    stars.reserve(140);
    for (int i = 0; i < 140; ++i) {
        const float x = static_cast<float>(GetRandomValue(-210, 210)) * 0.25f;
        const float y = static_cast<float>(GetRandomValue(12, 120)) * 0.25f;
        const float z = static_cast<float>(GetRandomValue(-210, 210)) * 0.25f;
        stars.push_back({x, y, z});
    }

    float t = 0.0f;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_Q)) break;

        const float dt = std::max(1.0e-4f, GetFrameTime());
        t += dt;
        const float now = static_cast<float>(GetTime());

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        int packetsRead = 0;
        if (receiver.Poll(remote, packetsRead)) {
            lastPacketWallClock = now;
        }
        const bool remoteLive = receiver.ready() && (now - lastPacketWallClock) < kBridgeTimeoutSec;

        if (remoteLive) {
            left.kind = KindFromId(remote.leftKind);
            right.kind = KindFromId(remote.rightKind);
            left.radius = BaseRadiusFor(left.kind);
            right.radius = BaseRadiusFor(right.kind);
            leftGrip = remote.leftGrip;
            rightGrip = remote.rightGrip;

            left.valid = remote.leftValid;
            right.valid = remote.rightValid;
            if (left.valid) {
                const Vector3 target = HandSpaceToWorld(remote.leftX, remote.leftY, false);
                const float alpha = 1.0f - std::exp(-kRemoteSmooth * dt);
                leftHand = Vector3Lerp(leftHand, target, alpha);
            }
            if (right.valid) {
                const Vector3 target = HandSpaceToWorld(remote.rightX, remote.rightY, true);
                const float alpha = 1.0f - std::exp(-kRemoteSmooth * dt);
                rightHand = Vector3Lerp(rightHand, target, alpha);
            }
        } else {
            left.valid = true;
            right.valid = true;
            leftGrip = 0.5f + 0.5f * std::sin(t * 1.6f);
            rightGrip = 0.5f + 0.5f * std::sin(t * 1.35f + 1.1f);

            if (IsKeyPressed(KEY_Z)) {
                left.kind = NextKind(left.kind);
                left.radius = BaseRadiusFor(left.kind);
            }
            if (IsKeyPressed(KEY_X)) {
                right.kind = NextKind(right.kind);
                right.radius = BaseRadiusFor(right.kind);
            }

            const float move = 5.2f * dt;
            if (IsKeyDown(KEY_A)) leftHand.x -= move;
            if (IsKeyDown(KEY_D)) leftHand.x += move;
            if (IsKeyDown(KEY_W)) leftHand.z -= move;
            if (IsKeyDown(KEY_S)) leftHand.z += move;
            if (IsKeyDown(KEY_R)) leftHand.y += move;
            if (IsKeyDown(KEY_F)) leftHand.y -= move;

            if (IsKeyDown(KEY_LEFT)) rightHand.x -= move;
            if (IsKeyDown(KEY_RIGHT)) rightHand.x += move;
            if (IsKeyDown(KEY_UP)) rightHand.z -= move;
            if (IsKeyDown(KEY_DOWN)) rightHand.z += move;
            if (IsKeyDown(KEY_PAGE_UP)) rightHand.y += move;
            if (IsKeyDown(KEY_PAGE_DOWN)) rightHand.y -= move;

            leftHand.x = std::clamp(leftHand.x, -8.5f, 8.5f);
            rightHand.x = std::clamp(rightHand.x, -8.5f, 8.5f);
            leftHand.y = std::clamp(leftHand.y, 0.3f, 6.2f);
            rightHand.y = std::clamp(rightHand.y, 0.3f, 6.2f);
            leftHand.z = std::clamp(leftHand.z, -8.5f, 8.5f);
            rightHand.z = std::clamp(rightHand.z, -8.5f, 8.5f);
        }

        if (left.valid) left.pos = Vector3Add(leftHand, {0.0f, kObjectLiftScale * left.radius, 0.0f});
        if (right.valid) right.pos = Vector3Add(rightHand, {0.0f, kObjectLiftScale * right.radius, 0.0f});

        float d = -1.0f;
        float enter = -1.0f;
        if (left.valid && right.valid) {
            d = Vector3Distance(left.pos, right.pos);
            const float threshold = 2.05f * (left.radius + right.radius);
            enter = threshold * 1.22f;
            const float exit = threshold * 1.85f;

            if (!interaction.active) {
                if (d <= enter) {
                    interaction.closeTimer += dt;
                    if (interaction.closeTimer >= 0.08f) {
                        interaction.active = true;
                        interaction.closeTimer = 0.0f;
                    }
                } else {
                    interaction.closeTimer = 0.0f;
                }
            } else if (d >= exit) {
                interaction.active = false;
                interaction.closeTimer = 0.0f;
            }

            if (remoteLive && remote.interactionHint) {
                interaction.active = true;
                interaction.closeTimer = 0.0f;
            }

            interaction.center = Vector3Lerp(
                interaction.center,
                Vector3Scale(Vector3Add(left.pos, right.pos), 0.5f),
                std::clamp(9.0f * dt, 0.0f, 1.0f)
            );
            interaction.radius = Lerp(
                interaction.radius,
                0.68f * (left.radius + right.radius),
                std::clamp(8.0f * dt, 0.0f, 1.0f)
            );
        } else {
            interaction.active = false;
            interaction.closeTimer = 0.0f;
        }

        const float blendTarget = interaction.active ? 1.0f : 0.0f;
        interaction.blend = Lerp(interaction.blend, blendTarget, std::clamp(8.0f * dt, 0.0f, 1.0f));
        if (remoteLive) {
            interaction.blend = std::max(interaction.blend, remote.interactionStrength);
        }

        BeginDrawing();
        ClearBackground(Color{7, 10, 18, 255});

        BeginMode3D(camera);
        DrawBackdrop(stars);
        DrawGrid(22, 1.0f);
        if (left.valid) DrawHandAvatar3D(leftHand, false, leftGrip, t);
        if (right.valid) DrawHandAvatar3D(rightHand, true, rightGrip, t);
        if (left.valid) DrawLine3D(leftHand, left.pos, Color{110, 200, 255, 120});
        if (right.valid) DrawLine3D(rightHand, right.pos, Color{255, 205, 140, 120});

        if (interaction.blend > 0.06f && left.valid && right.valid) {
            DrawInteraction3D(left, right, interaction, t);
        } else {
            DrawBody3D(left, t * 0.9f);
            DrawBody3D(right, t * 0.9f + 0.8f);
            if (left.valid && right.valid && enter > 0.0f) {
                const float s = std::clamp((enter - d) / std::max(0.001f, enter), 0.0f, 1.0f);
                const Color linkColor = Color{
                    static_cast<unsigned char>(70 + 120 * s),
                    static_cast<unsigned char>(85 + 110 * s),
                    static_cast<unsigned char>(120 + 100 * s),
                    185
                };
                DrawLine3D(left.pos, right.pos, linkColor);
            }
        }

        EndMode3D();

        const int pairCode = PairCode(left.kind, right.kind);
        const char* modeText = remoteLive ? "mode: UDP bridge (python)" : "mode: local controls";
        const char* stateText = (interaction.active && left.valid && right.valid) ? PairLabel(pairCode) : "none";

        DrawRectangle(10, 10, GetScreenWidth() - 20, 64, Color{12, 18, 28, 180});
        DrawText(TextFormat("L[%s %s]  R[%s %s]  dist: %.2f  trigger: %.2f  state: %s",
                            KindName(left.kind), left.valid ? "on" : "off",
                            KindName(right.kind), right.valid ? "on" : "off",
                            d, enter, stateText),
                 18, 16, 19, Color{210, 228, 250, 255});

        DrawText(
            receiverOk
                ? TextFormat("%s  |  UDP:%d  |  grip L/R: %.2f / %.2f  |  Z/X + move keys work when bridge idle",
                             modeText, kUdpPort, leftGrip, rightGrip)
                : "udp receiver failed: running in local-only mode",
            18, 40, 16, Color{165, 188, 220, 255}
        );
        DrawFPS(GetScreenWidth() - 96, 16);

        EndDrawing();
    }

    receiver.Close();
    CloseWindow();
    return 0;
}
