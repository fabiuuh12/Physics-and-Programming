#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

constexpr int kScreenWidth = 1360;
constexpr int kScreenHeight = 860;

constexpr float kBoneSides = 10.0f;
constexpr float kJointSphereScale = 1.18f;
constexpr int kUdpPort = 50515;
constexpr float kLinkTimeout = 0.75f;

enum class HandPreset {
    Relaxed = 0,
    Fist,
    Pinch,
    Point,
    Splay,
};

struct FingerSpec {
    std::array<float, 3> lengths;
    float baseX = 0.0f;
    float baseY = 0.0f;
    float baseZ = 0.0f;
    float baseSplay = 0.0f;
    float baseLift = 0.0f;
    float curlBias = 0.0f;
};

struct HandControls {
    float grip = 0.15f;
    float spread = 0.10f;
    float pinch = 0.0f;
    float thumbOpposition = 0.20f;
    float wristPitch = -0.22f;
    float wristYaw = 0.18f;
    float wristRoll = 0.08f;
    std::array<float, 5> fingerAdd = {0.0f, 0.0f, 0.0f, 0.02f, 0.05f};
};

struct HandGeometry {
    std::array<Vector3, 21> landmarks{};
    std::array<float, 21> radii{};
    std::array<Vector3, 7> palmRim{};
    Vector3 palmCenter{};
    Vector3 wristLeft{};
    Vector3 wristRight{};
};

struct TrackingPacket {
    bool valid = false;
    bool pinched = false;
    float score = 0.0f;
    double timestamp = 0.0;
    std::array<Vector3, 21> landmarks{};
};

class UdpHandReceiver {
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

    ~UdpHandReceiver() {
        Close();
    }

    bool Poll(TrackingPacket& outPacket, int& packetsRead) {
        packetsRead = 0;
        if (!ready_) return false;

        bool gotAny = false;
        std::array<char, 4096> buffer{};
        while (true) {
            sockaddr_in src{};
            socklen_t srcLen = sizeof(src);
            const int n = recvfrom(
                socket_,
                buffer.data(),
                static_cast<int>(buffer.size()) - 1,
                0,
                reinterpret_cast<sockaddr*>(&src),
                &srcLen);
            if (n <= 0) {
#ifdef _WIN32
                if (WSAGetLastError() == WSAEWOULDBLOCK) break;
#else
                if (errno == EWOULDBLOCK || errno == EAGAIN) break;
#endif
                break;
            }

            buffer[static_cast<size_t>(n)] = '\0';
            TrackingPacket parsed{};
            if (ParsePacket(buffer.data(), parsed)) {
                outPacket = parsed;
                gotAny = true;
                packetsRead++;
            }
        }
        return gotAny;
    }

    bool ready() const { return ready_; }

  private:
    static bool ParsePacket(const char* text, TrackingPacket& outPacket) {
        std::vector<float> values;
        values.reserve(67);

        const char* p = text;
        char* end = nullptr;
        while (*p != '\0') {
            const float value = std::strtof(p, &end);
            if (end == p) {
                if (*p == ',' || *p == '\n' || *p == '\r' || *p == ' ' || *p == '\t') {
                    ++p;
                    continue;
                }
                return false;
            }
            values.push_back(value);
            p = end;
            while (*p == ',' || *p == '\n' || *p == '\r' || *p == ' ' || *p == '\t') ++p;
        }

        if (values.size() < 67) return false;

        outPacket.timestamp = static_cast<double>(values[0]);
        outPacket.valid = values[1] > 0.5f;
        outPacket.pinched = values[2] > 0.5f;
        outPacket.score = values[3];

        size_t idx = 4;
        for (Vector3& point : outPacket.landmarks) {
            point = {values[idx], values[idx + 1], values[idx + 2]};
            idx += 3;
        }
        return true;
    }

    int socket_ = -1;
    bool ready_ = false;
};

float Clamp01(float v) {
    return std::clamp(v, 0.0f, 1.0f);
}

float Smooth01(float x) {
    x = Clamp01(x);
    return x * x * (3.0f - 2.0f * x);
}

float LerpFloat(float a, float b, float t) {
    return a + (b - a) * t;
}

Vector3 RotateAroundAxis(Vector3 v, Vector3 axis, float angle) {
    axis = Vector3Normalize(axis);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return Vector3Add(
        Vector3Add(Vector3Scale(v, c), Vector3Scale(Vector3CrossProduct(axis, v), s)),
        Vector3Scale(axis, Vector3DotProduct(axis, v) * (1.0f - c)));
}

Vector3 RotateEulerXYZ(Vector3 v, Vector3 angles) {
    v = RotateAroundAxis(v, {1.0f, 0.0f, 0.0f}, angles.x);
    v = RotateAroundAxis(v, {0.0f, 1.0f, 0.0f}, angles.y);
    v = RotateAroundAxis(v, {0.0f, 0.0f, 1.0f}, angles.z);
    return v;
}

Vector3 LocalToWorld(Vector3 origin, Vector3 wristAngles, float sideSign, Vector3 local) {
    Vector3 mirrored = {sideSign * local.x, local.y, local.z};
    return Vector3Add(origin, RotateEulerXYZ(mirrored, wristAngles));
}

Vector3 Average(const std::initializer_list<Vector3>& pts) {
    Vector3 sum{0.0f, 0.0f, 0.0f};
    for (const Vector3& p : pts) sum = Vector3Add(sum, p);
    return Vector3Scale(sum, 1.0f / static_cast<float>(pts.size()));
}

HandControls PresetControls(HandPreset preset) {
    HandControls c;
    switch (preset) {
        case HandPreset::Relaxed:
            c.grip = 0.16f;
            c.spread = 0.08f;
            c.pinch = 0.0f;
            c.thumbOpposition = 0.22f;
            c.wristPitch = -0.22f;
            c.wristYaw = 0.18f;
            c.wristRoll = 0.08f;
            c.fingerAdd = {0.0f, 0.0f, 0.0f, 0.02f, 0.06f};
            break;
        case HandPreset::Fist:
            c.grip = 0.98f;
            c.spread = -0.18f;
            c.pinch = 0.0f;
            c.thumbOpposition = 0.55f;
            c.wristPitch = -0.18f;
            c.wristYaw = 0.10f;
            c.wristRoll = 0.05f;
            c.fingerAdd = {0.10f, 0.06f, 0.08f, 0.12f, 0.18f};
            break;
        case HandPreset::Pinch:
            c.grip = 0.30f;
            c.spread = 0.02f;
            c.pinch = 1.0f;
            c.thumbOpposition = 1.0f;
            c.wristPitch = -0.20f;
            c.wristYaw = 0.14f;
            c.wristRoll = 0.10f;
            c.fingerAdd = {0.02f, -0.10f, 0.14f, 0.28f, 0.34f};
            break;
        case HandPreset::Point:
            c.grip = 0.62f;
            c.spread = -0.02f;
            c.pinch = 0.0f;
            c.thumbOpposition = 0.36f;
            c.wristPitch = -0.26f;
            c.wristYaw = 0.10f;
            c.wristRoll = 0.06f;
            c.fingerAdd = {0.08f, -0.78f, 0.14f, 0.22f, 0.28f};
            break;
        case HandPreset::Splay:
            c.grip = 0.06f;
            c.spread = 1.0f;
            c.pinch = 0.0f;
            c.thumbOpposition = 0.12f;
            c.wristPitch = -0.18f;
            c.wristYaw = 0.18f;
            c.wristRoll = 0.04f;
            c.fingerAdd = {-0.02f, -0.08f, -0.08f, -0.04f, 0.00f};
            break;
    }
    return c;
}

HandControls LerpControls(const HandControls& a, const HandControls& b, float t) {
    HandControls out{};
    out.grip = LerpFloat(a.grip, b.grip, t);
    out.spread = LerpFloat(a.spread, b.spread, t);
    out.pinch = LerpFloat(a.pinch, b.pinch, t);
    out.thumbOpposition = LerpFloat(a.thumbOpposition, b.thumbOpposition, t);
    out.wristPitch = LerpFloat(a.wristPitch, b.wristPitch, t);
    out.wristYaw = LerpFloat(a.wristYaw, b.wristYaw, t);
    out.wristRoll = LerpFloat(a.wristRoll, b.wristRoll, t);
    for (size_t i = 0; i < out.fingerAdd.size(); ++i) {
        out.fingerAdd[i] = LerpFloat(a.fingerAdd[i], b.fingerAdd[i], t);
    }
    return out;
}

HandControls DemoControls(float t) {
    constexpr std::array<HandPreset, 5> cycle = {
        HandPreset::Relaxed,
        HandPreset::Splay,
        HandPreset::Pinch,
        HandPreset::Point,
        HandPreset::Fist,
    };

    const float segmentLength = 3.4f;
    const float wrapped = std::fmod(t, segmentLength * static_cast<float>(cycle.size()));
    const int idx = static_cast<int>(wrapped / segmentLength);
    const float localT = Smooth01((wrapped - idx * segmentLength) / segmentLength);
    return LerpControls(
        PresetControls(cycle[static_cast<size_t>(idx)]),
        PresetControls(cycle[static_cast<size_t>((idx + 1) % cycle.size())]),
        localT);
}

HandGeometry BuildHandGeometry(const HandControls& controls, Vector3 origin, bool rightHand) {
    HandGeometry g{};

    const float sideSign = rightHand ? 1.0f : -1.0f;
    const Vector3 wristAngles = {controls.wristPitch, controls.wristYaw, controls.wristRoll};

    const std::array<FingerSpec, 4> fingerSpecs = {{
        {{1.70f, 1.12f, 0.92f}, -0.72f, 4.12f, 0.24f, -0.14f, 0.08f, -0.02f},
        {{1.88f, 1.26f, 1.00f},  0.02f, 4.28f, 0.30f, -0.05f, 0.10f,  0.00f},
        {{1.72f, 1.12f, 0.90f},  0.74f, 4.10f, 0.26f,  0.08f, 0.10f,  0.05f},
        {{1.28f, 0.86f, 0.72f},  1.46f, 3.62f, 0.20f,  0.20f, 0.06f,  0.10f},
    }};

    g.landmarks[0] = LocalToWorld(origin, wristAngles, sideSign, {0.0f, 0.0f, 0.0f});
    g.radii[0] = 0.34f;

    const Vector3 thumbCmcLocal = {-1.72f, 1.12f, 0.04f};
    g.landmarks[1] = LocalToWorld(origin, wristAngles, sideSign, thumbCmcLocal);
    g.radii[1] = 0.24f;

    Vector3 thumbDir = Vector3Normalize({
        -0.80f,
        0.56f + 0.16f * controls.thumbOpposition,
        0.08f + 0.18f * controls.thumbOpposition,
    });
    thumbDir = RotateAroundAxis(thumbDir, {0.0f, 0.0f, 1.0f}, 0.16f + 0.34f * controls.thumbOpposition - 0.10f * controls.spread);
    thumbDir = RotateAroundAxis(thumbDir, {0.0f, 1.0f, 0.0f}, -0.08f + 0.24f * controls.thumbOpposition + 0.14f * controls.pinch);
    thumbDir = RotateAroundAxis(thumbDir, {1.0f, 0.0f, 0.0f}, -0.06f + 0.12f * controls.pinch);
    const Vector3 thumbBaseAxis = Vector3Normalize({0.58f, 0.12f, 0.80f});
    const Vector3 thumbDistalAxis = Vector3Normalize({0.34f, -0.10f, 0.94f});
    const float thumbGrip = Clamp01(0.16f + 0.75f * controls.grip + 0.60f * controls.pinch + 0.40f * controls.fingerAdd[0]);
    const std::array<float, 3> thumbAngles = {
        0.06f + 0.26f * thumbGrip,
        0.18f + 0.52f * thumbGrip,
        0.14f + 0.44f * thumbGrip,
    };
    const std::array<float, 3> thumbSweep = {
        0.05f + 0.18f * controls.thumbOpposition,
        0.02f + 0.14f * controls.thumbOpposition + 0.10f * controls.pinch,
        0.04f * controls.pinch,
    };
    const std::array<float, 3> thumbLengths = {0.94f, 0.82f, 0.72f};

    Vector3 thumbPos = thumbCmcLocal;
    for (int joint = 0; joint < 3; ++joint) {
        const Vector3 bendAxis = (joint == 0) ? thumbBaseAxis : thumbDistalAxis;
        thumbDir = RotateAroundAxis(thumbDir, bendAxis, -thumbAngles[static_cast<size_t>(joint)]);
        thumbDir = RotateAroundAxis(thumbDir, {0.0f, 1.0f, 0.0f}, thumbSweep[static_cast<size_t>(joint)]);
        thumbPos = Vector3Add(thumbPos, Vector3Scale(thumbDir, thumbLengths[static_cast<size_t>(joint)]));
        g.landmarks[2 + joint] = LocalToWorld(origin, wristAngles, sideSign, thumbPos);
        g.radii[2 + joint] = 0.22f - 0.03f * static_cast<float>(joint);
    }

    const std::array<int, 4> baseIndices = {5, 9, 13, 17};
    for (size_t finger = 0; finger < fingerSpecs.size(); ++finger) {
        const FingerSpec& spec = fingerSpecs[finger];
        const float totalGrip = Clamp01(spec.curlBias + controls.grip + controls.fingerAdd[finger + 1] + 0.12f * controls.pinch);
        const float splay = spec.baseSplay + controls.spread * (0.22f + 0.06f * static_cast<float>(finger));
        const float lift = spec.baseLift;

        const std::array<float, 3> bendAngles = {
            0.12f + 1.08f * totalGrip,
            0.08f + 1.20f * totalGrip,
            0.04f + 0.92f * totalGrip,
        };

        Vector3 dir = {0.0f, 1.0f, 0.0f};
        dir = RotateAroundAxis(dir, {0.0f, 0.0f, 1.0f}, splay);
        dir = RotateAroundAxis(dir, {1.0f, 0.0f, 0.0f}, lift);

        Vector3 jointPos = {spec.baseX, spec.baseY, spec.baseZ};
        const int baseIndex = baseIndices[finger];
        g.landmarks[baseIndex] = LocalToWorld(origin, wristAngles, sideSign, jointPos);
        g.radii[baseIndex] = 0.23f - 0.02f * static_cast<float>(finger);

        for (int segment = 0; segment < 3; ++segment) {
            dir = RotateAroundAxis(dir, {1.0f, 0.0f, 0.0f}, -bendAngles[static_cast<size_t>(segment)]);
            jointPos = Vector3Add(jointPos, Vector3Scale(dir, spec.lengths[static_cast<size_t>(segment)]));
            g.landmarks[baseIndex + segment + 1] = LocalToWorld(origin, wristAngles, sideSign, jointPos);
            g.radii[baseIndex + segment + 1] = (0.20f - 0.03f * static_cast<float>(segment)) - 0.02f * static_cast<float>(finger);
        }
    }

    g.wristLeft = LocalToWorld(origin, wristAngles, sideSign, {-1.62f, 0.26f, -0.44f});
    g.wristRight = LocalToWorld(origin, wristAngles, sideSign, {1.66f, 0.30f, -0.38f});
    g.palmRim = {
        g.wristLeft,
        g.landmarks[1],
        g.landmarks[5],
        g.landmarks[9],
        g.landmarks[13],
        g.landmarks[17],
        g.wristRight,
    };
    g.palmCenter = Average({g.landmarks[0], g.landmarks[1], g.landmarks[5], g.landmarks[9], g.landmarks[13], g.landmarks[17]});

    return g;
}

float LandmarkPalmNorm(const std::array<Vector3, 21>& pts) {
    const auto dist2 = [&](int a, int b) {
        const float dx = pts[static_cast<size_t>(a)].x - pts[static_cast<size_t>(b)].x;
        const float dy = pts[static_cast<size_t>(a)].y - pts[static_cast<size_t>(b)].y;
        return std::sqrt(dx * dx + dy * dy);
    };
    return std::max(1.0e-4f, (dist2(0, 5) + dist2(0, 17) + dist2(5, 17)) / 3.0f);
}

HandGeometry BuildTrackedGeometry(const TrackingPacket& packet) {
    HandGeometry g{};
    constexpr std::array<float, 21> kRadii = {
        0.34f, 0.24f, 0.22f, 0.19f, 0.16f,
        0.23f, 0.20f, 0.17f, 0.14f,
        0.23f, 0.20f, 0.17f, 0.14f,
        0.21f, 0.18f, 0.15f, 0.12f,
        0.19f, 0.16f, 0.13f, 0.10f,
    };

    const float palm = LandmarkPalmNorm(packet.landmarks);
    Vector3 palmCenterNorm{0.0f, 0.0f, 0.0f};
    for (int idx : {0, 5, 9, 13, 17}) {
        palmCenterNorm = Vector3Add(palmCenterNorm, packet.landmarks[static_cast<size_t>(idx)]);
    }
    palmCenterNorm = Vector3Scale(palmCenterNorm, 1.0f / 5.0f);

    const Vector3 origin = {
        (palmCenterNorm.x - 0.50f) * 9.0f,
        2.5f + (0.56f - palmCenterNorm.y) * 7.0f,
        0.0f,
    };

    for (size_t i = 0; i < g.landmarks.size(); ++i) {
        const Vector3 rel = Vector3Subtract(packet.landmarks[i], palmCenterNorm);
        const Vector3 local = {
            -(rel.x / palm) * 2.80f,
            -(rel.y / palm) * 2.80f,
            -(rel.z / palm) * 3.60f,
        };
        g.landmarks[i] = Vector3Add(origin, local);
        g.radii[i] = kRadii[i];
    }

    const Vector3 across = Vector3Subtract(g.landmarks[17], g.landmarks[5]);
    Vector3 wristAxis = {1.0f, 0.0f, 0.0f};
    if (Vector3Length(across) > 1.0e-4f) {
        wristAxis = Vector3Normalize(across);
    }

    g.wristLeft = Vector3Add(g.landmarks[0], Vector3Scale(wristAxis, -0.88f));
    g.wristRight = Vector3Add(g.landmarks[0], Vector3Scale(wristAxis, 0.88f));
    g.palmRim = {
        g.wristLeft,
        g.landmarks[1],
        g.landmarks[5],
        g.landmarks[9],
        g.landmarks[13],
        g.landmarks[17],
        g.wristRight,
    };
    g.palmCenter = Average({g.landmarks[0], g.landmarks[1], g.landmarks[5], g.landmarks[9], g.landmarks[13], g.landmarks[17]});

    return g;
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0032f;
        *pitch += d.y * 0.0030f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.7f;
    *distance = std::clamp(*distance, 6.0f, 28.0f);

    const float cp = std::cos(*pitch);
    const Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

void DrawBone(Vector3 a, Vector3 b, float ra, float rb, Color color) {
    DrawCylinderEx(a, b, ra, rb, static_cast<int>(kBoneSides), color);
    DrawCylinderWiresEx(a, b, ra, rb, static_cast<int>(kBoneSides), Fade(BLACK, 0.25f));
}

void DrawPalmSurface(const HandGeometry& g, Color palmColor) {
    for (size_t i = 0; i + 1 < g.palmRim.size(); ++i) {
        DrawTriangle3D(g.palmCenter, g.palmRim[i], g.palmRim[i + 1], palmColor);
    }

    const Color webColor = Fade(palmColor, 0.92f);
    DrawTriangle3D(g.landmarks[1], g.landmarks[5], g.landmarks[6], webColor);
    DrawTriangle3D(g.landmarks[5], g.landmarks[9], g.landmarks[6], webColor);
    DrawTriangle3D(g.landmarks[9], g.landmarks[13], g.landmarks[10], webColor);
    DrawTriangle3D(g.landmarks[13], g.landmarks[17], g.landmarks[14], webColor);
}

void DrawPalmLines(const HandGeometry& g, Color color) {
    for (size_t i = 0; i + 1 < g.palmRim.size(); ++i) {
        DrawLine3D(g.palmRim[i], g.palmRim[i + 1], color);
    }
    DrawLine3D(g.wristLeft, g.wristRight, color);
    DrawLine3D(g.landmarks[0], g.landmarks[5], Fade(color, 0.85f));
    DrawLine3D(g.landmarks[0], g.landmarks[9], Fade(color, 0.90f));
    DrawLine3D(g.landmarks[0], g.landmarks[13], Fade(color, 0.85f));
    DrawLine3D(g.landmarks[0], g.landmarks[17], Fade(color, 0.80f));
}

void DrawTendonLines(const HandGeometry& g, Color color) {
    const std::array<int, 4> fingerStarts = {5, 9, 13, 17};
    for (int idx : fingerStarts) {
        DrawLine3D(g.landmarks[0], g.landmarks[idx + 1], Fade(color, 0.65f));
        DrawLine3D(g.landmarks[idx], g.landmarks[idx + 2], Fade(color, 0.50f));
    }
    DrawLine3D(g.landmarks[1], g.landmarks[3], Fade(color, 0.55f));
}

void DrawHandModel(const HandGeometry& g, bool showLandmarks) {
    const Color palmColor = Color{222, 182, 146, 255};
    const Color boneColor = Color{205, 156, 120, 255};
    const Color tipColor = Color{235, 198, 166, 255};
    const Color tendonColor = Color{142, 104, 82, 255};

    DrawPalmSurface(g, palmColor);

    const std::array<std::pair<int, int>, 20> bones = {{
        {0, 1}, {1, 2}, {2, 3}, {3, 4},
        {0, 5}, {5, 6}, {6, 7}, {7, 8},
        {0, 9}, {9, 10}, {10, 11}, {11, 12},
        {0, 13}, {13, 14}, {14, 15}, {15, 16},
        {0, 17}, {17, 18}, {18, 19}, {19, 20},
    }};

    for (const auto& [a, b] : bones) {
        const bool fingertipBone = (b == 4 || b == 8 || b == 12 || b == 16 || b == 20);
        DrawBone(g.landmarks[a], g.landmarks[b], g.radii[a] * 0.78f, g.radii[b] * 0.82f, fingertipBone ? tipColor : boneColor);
    }

    const std::array<int, 4> metacarpalTargets = {5, 9, 13, 17};
    for (int idx : metacarpalTargets) {
        DrawBone(g.landmarks[0], g.landmarks[idx], 0.14f, g.radii[idx] * 0.90f, Fade(boneColor, 0.75f));
    }
    DrawBone(g.landmarks[0], g.landmarks[1], 0.16f, g.radii[1] * 0.95f, Fade(boneColor, 0.72f));

    DrawTendonLines(g, tendonColor);
    DrawPalmLines(g, Fade(tendonColor, 0.65f));

    for (size_t i = 0; i < g.landmarks.size(); ++i) {
        const bool tip = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20);
        DrawSphere(g.landmarks[i], g.radii[i] * kJointSphereScale, tip ? tipColor : palmColor);
    }

    DrawSphere(g.palmCenter, 0.33f, Fade(palmColor, 0.85f));

    if (showLandmarks) {
        for (size_t i = 0; i < g.landmarks.size(); ++i) {
            DrawSphere(g.landmarks[i], g.radii[i] * 0.42f, Color{80, 210, 255, 255});
        }
    }
}

std::string HudLine(const HandControls& c, bool autoDemo, bool rightHand, bool showLandmarks) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "grip=" << c.grip
       << "  spread=" << c.spread
       << "  pinch=" << c.pinch
       << "  thumb=" << c.thumbOpposition
       << "  wrist=(" << c.wristPitch << ", " << c.wristYaw << ", " << c.wristRoll << ")";
    if (autoDemo) os << "  [AUTO]";
    os << (rightHand ? "  [RIGHT]" : "  [LEFT]");
    if (showLandmarks) os << "  [LANDMARKS]";
    return os.str();
}

void DrawLandmarkLabels(const HandGeometry& g, const Camera3D& camera) {
    for (size_t i = 0; i < g.landmarks.size(); ++i) {
        const Vector2 pos = GetWorldToScreen(g.landmarks[i], camera);
        DrawCircleV(pos, 10.0f, Fade(BLACK, 0.45f));
        DrawText(TextFormat("%d", static_cast<int>(i)), static_cast<int>(pos.x) - 6, static_cast<int>(pos.y) - 8, 16, Color{240, 248, 255, 255});
    }
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "3D Human Hand Biomechanics - C++ (raylib)");
    SetTargetFPS(120);
    SetWindowMinSize(980, 640);

    Camera3D camera{};
    camera.target = {0.0f, 3.9f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 46.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.78f;
    float camPitch = 0.28f;
    float camDistance = 14.0f;

    bool autoDemo = true;
    bool rightHand = true;
    bool showLandmarks = false;
    HandControls manual = PresetControls(HandPreset::Relaxed);
    float demoTime = 0.0f;

    UdpHandReceiver receiver;
    const bool receiverOk = receiver.Start(static_cast<uint16_t>(kUdpPort));
    TrackingPacket tracking{};
    float lastPacketWallClock = -100.0f;

    while (!WindowShouldClose()) {
        const float dt = std::max(GetFrameTime(), 1.0e-4f);
        const float now = static_cast<float>(GetTime());

        if (IsKeyPressed(KEY_SPACE)) autoDemo = !autoDemo;
        if (IsKeyPressed(KEY_L)) showLandmarks = !showLandmarks;
        if (IsKeyPressed(KEY_M)) rightHand = !rightHand;
        if (IsKeyPressed(KEY_R)) {
            manual = PresetControls(HandPreset::Relaxed);
            autoDemo = false;
        }

        if (IsKeyPressed(KEY_ONE)) {
            manual = PresetControls(HandPreset::Relaxed);
            autoDemo = false;
        }
        if (IsKeyPressed(KEY_TWO)) {
            manual = PresetControls(HandPreset::Fist);
            autoDemo = false;
        }
        if (IsKeyPressed(KEY_THREE)) {
            manual = PresetControls(HandPreset::Pinch);
            autoDemo = false;
        }
        if (IsKeyPressed(KEY_FOUR)) {
            manual = PresetControls(HandPreset::Point);
            autoDemo = false;
        }
        if (IsKeyPressed(KEY_FIVE)) {
            manual = PresetControls(HandPreset::Splay);
            autoDemo = false;
        }

        if (IsKeyDown(KEY_RIGHT_BRACKET)) {
            manual.grip = std::min(1.0f, manual.grip + 0.75f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_LEFT_BRACKET)) {
            manual.grip = std::max(0.0f, manual.grip - 0.75f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_APOSTROPHE)) {
            manual.spread = std::min(1.1f, manual.spread + 0.9f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_SEMICOLON)) {
            manual.spread = std::max(-0.4f, manual.spread - 0.9f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_PERIOD)) {
            manual.thumbOpposition = std::min(1.0f, manual.thumbOpposition + 0.9f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_COMMA)) {
            manual.thumbOpposition = std::max(0.0f, manual.thumbOpposition - 0.9f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_EQUAL) || IsKeyDown(KEY_KP_ADD)) {
            manual.pinch = std::min(1.0f, manual.pinch + 0.95f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_MINUS) || IsKeyDown(KEY_KP_SUBTRACT)) {
            manual.pinch = std::max(0.0f, manual.pinch - 0.95f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_W)) {
            manual.wristPitch = std::min(0.7f, manual.wristPitch + 0.8f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_S)) {
            manual.wristPitch = std::max(-0.9f, manual.wristPitch - 0.8f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_D)) {
            manual.wristYaw = std::min(0.9f, manual.wristYaw + 0.9f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_A)) {
            manual.wristYaw = std::max(-0.9f, manual.wristYaw - 0.9f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_E)) {
            manual.wristRoll = std::min(1.1f, manual.wristRoll + 1.0f * dt);
            autoDemo = false;
        }
        if (IsKeyDown(KEY_Q)) {
            manual.wristRoll = std::max(-1.1f, manual.wristRoll - 1.0f * dt);
            autoDemo = false;
        }

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        int packetsRead = 0;
        if (receiver.Poll(tracking, packetsRead)) {
            lastPacketWallClock = now;
        }

        const bool linkLive = receiver.ready() && ((now - lastPacketWallClock) < kLinkTimeout);
        const bool trackedActive = linkLive && tracking.valid;

        if (autoDemo && !trackedActive) demoTime += dt;
        const HandControls pose = autoDemo ? DemoControls(demoTime) : manual;
        const HandGeometry geometry = trackedActive
            ? BuildTrackedGeometry(tracking)
            : BuildHandGeometry(pose, {0.0f, 0.65f, 0.0f}, rightHand);

        BeginDrawing();
        ClearBackground(Color{7, 10, 18, 255});

        BeginMode3D(camera);
        DrawGrid(20, 1.0f);
        DrawCube({0.0f, -0.05f, 0.0f}, 7.5f, 0.08f, 7.5f, Color{34, 44, 62, 255});
        DrawCube({0.0f, 0.02f, 0.0f}, 2.8f, 0.05f, 2.8f, Color{46, 58, 78, 255});
        DrawHandModel(geometry, showLandmarks);
        EndMode3D();

        DrawText("3D Human Hand Biomechanics", 20, 18, 31, Color{230, 236, 245, 255});
        DrawText(
            "Mouse drag/wheel: camera | Space auto-demo | 1-5 presets | M mirror hand | L landmark ids | R relaxed reset",
            20, 56, 19, Color{164, 183, 210, 255});
        DrawText(
            "[ ] grip | ; ' spread | , . thumb opposition | -/+ pinch | W/S pitch | A/D yaw | Q/E roll",
            20, 82, 19, Color{164, 183, 210, 255});

        const std::string hud = HudLine(pose, autoDemo, rightHand, showLandmarks);
        DrawText(hud.c_str(), 20, 114, 20, Color{126, 224, 255, 255});
        DrawText(
            trackedActive
                ? "Live right-hand webcam tracking is driving the 21-point rig."
                : "Landmark layout matches a 21-point hand skeleton, so the webcam bridge can drive this model later without replacing the rig.",
            20, 142, 18, Color{194, 205, 223, 255});

        const char* bridgeStatus =
            !receiverOk ? "bridge: UDP receiver failed to start"
            : trackedActive ? "bridge: tracking live on udp:50515"
            : linkLive ? "bridge: packets arriving, waiting for valid right hand"
            : "bridge: idle on udp:50515  run AstroPhysics/vision/hand_biomechanics_bridge.py";
        DrawText(bridgeStatus, 20, 168, 18, trackedActive ? Color{142, 255, 190, 255} : Color{188, 198, 220, 255});
        if (trackedActive && tracking.pinched) {
            DrawText("gesture: pinch", 20, 194, 18, Color{255, 224, 132, 255});
        }
        DrawFPS(20, 222);

        if (showLandmarks) {
            DrawLandmarkLabels(geometry, camera);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
