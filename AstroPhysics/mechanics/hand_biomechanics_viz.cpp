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
constexpr int kFrameUdpPort = 50516;
constexpr float kLinkTimeout = 0.75f;
constexpr float kTrackedDepthPalmRef = 0.155f;
constexpr float kTrackedDepthRange = 5.4f;

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

struct HandKinematics {
    bool active = false;
    bool pinched = false;
    Vector3 palm{};
    Vector3 indexTip{};
    Vector3 thumbTip{};
    Vector3 grabPoint{};
    Vector3 velocity{};
};

struct BallState {
    Vector3 pos{0.0f, 3.1f, 0.0f};
    Vector3 vel{0.0f, 0.0f, 0.0f};
    float radius = 0.52f;
    int grabbedBy = -1;
};

struct TrackedHandPacket {
    bool valid = false;
    bool pinched = false;
    float score = 0.0f;
    std::array<Vector3, 21> landmarks{};
};

struct TrackingPacket {
    double timestamp = 0.0;
    TrackedHandPacket left{};
    TrackedHandPacket right{};
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
        values.reserve(133);

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

        if (values.size() < 133) return false;

        outPacket.timestamp = static_cast<double>(values[0]);
        auto readHand = [&](size_t start, TrackedHandPacket& hand) {
            hand.valid = values[start] > 0.5f;
            hand.pinched = values[start + 1] > 0.5f;
            hand.score = values[start + 2];
            size_t idx = start + 3;
            for (Vector3& point : hand.landmarks) {
                point = {values[idx], values[idx + 1], values[idx + 2]};
                idx += 3;
            }
        };

        readHand(1, outPacket.left);
        readHand(67, outPacket.right);
        return true;
    }

    int socket_ = -1;
    bool ready_ = false;
};

class UdpFrameReceiver {
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

    ~UdpFrameReceiver() {
        Close();
    }

    bool Poll(std::vector<unsigned char>& outFrameBytes, int& packetsRead) {
        packetsRead = 0;
        if (!ready_) return false;

        bool gotAny = false;
        std::array<unsigned char, 65536> buffer{};
        while (true) {
            sockaddr_in src{};
            socklen_t srcLen = sizeof(src);
            const int n = recvfrom(
                socket_,
                reinterpret_cast<char*>(buffer.data()),
                static_cast<int>(buffer.size()),
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

            outFrameBytes.assign(buffer.begin(), buffer.begin() + n);
            gotAny = true;
            packetsRead++;
        }
        return gotAny;
    }

    bool ready() const { return ready_; }

  private:
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

Vector3 SafeNormalize(Vector3 v, Vector3 fallback) {
    if (Vector3Length(v) < 1.0e-4f) return fallback;
    return Vector3Normalize(v);
}

HandGeometry BlendGeometry(const HandGeometry& a, const HandGeometry& b, float t) {
    HandGeometry out{};
    for (size_t i = 0; i < out.landmarks.size(); ++i) {
        out.landmarks[i] = Vector3Lerp(a.landmarks[i], b.landmarks[i], t);
        out.radii[i] = LerpFloat(a.radii[i], b.radii[i], t);
    }
    for (size_t i = 0; i < out.palmRim.size(); ++i) {
        out.palmRim[i] = Vector3Lerp(a.palmRim[i], b.palmRim[i], t);
    }
    out.palmCenter = Vector3Lerp(a.palmCenter, b.palmCenter, t);
    out.wristLeft = Vector3Lerp(a.wristLeft, b.wristLeft, t);
    out.wristRight = Vector3Lerp(a.wristRight, b.wristRight, t);
    return out;
}

HandGeometry OffsetGeometry(const HandGeometry& g, Vector3 offset) {
    HandGeometry out = g;
    for (Vector3& point : out.landmarks) point = Vector3Add(point, offset);
    for (Vector3& point : out.palmRim) point = Vector3Add(point, offset);
    out.palmCenter = Vector3Add(out.palmCenter, offset);
    out.wristLeft = Vector3Add(out.wristLeft, offset);
    out.wristRight = Vector3Add(out.wristRight, offset);
    return out;
}

float EstimateTrackedDepthShift(const TrackedHandPacket& packet) {
    // MediaPipe normalized landmarks do not provide stable absolute camera depth.
    // Use palm size in image space as a smoother depth hint for whole-hand motion.
    const float palm = LandmarkPalmNorm(packet.landmarks);
    const float relative = std::clamp((kTrackedDepthPalmRef / palm) - 1.0f, -0.55f, 0.90f);
    return -relative * kTrackedDepthRange;
}

bool UpdatePreviewTexture(Texture2D& texture, const std::vector<unsigned char>& jpgBytes) {
    if (jpgBytes.empty()) return false;

    Image image = LoadImageFromMemory(".jpg", jpgBytes.data(), static_cast<int>(jpgBytes.size()));
    if (image.data == nullptr) return false;

    if (texture.id > 0 && texture.width == image.width && texture.height == image.height) {
        UpdateTexture(texture, image.data);
    } else {
        if (texture.id > 0) {
            UnloadTexture(texture);
            texture = Texture2D{};
        }
        texture = LoadTextureFromImage(image);
        SetTextureFilter(texture, TEXTURE_FILTER_BILINEAR);
    }

    UnloadImage(image);
    return texture.id > 0;
}

HandGeometry BuildTrackedGeometry(const TrackedHandPacket& packet, bool rightHand) {
    HandGeometry g{};
    constexpr std::array<float, 21> kRadii = {
        0.34f, 0.24f, 0.22f, 0.19f, 0.16f,
        0.23f, 0.20f, 0.17f, 0.14f,
        0.23f, 0.20f, 0.17f, 0.14f,
        0.21f, 0.18f, 0.15f, 0.12f,
        0.19f, 0.16f, 0.13f, 0.10f,
    };

    const Vector3 wristNorm = packet.landmarks[0];
    const Vector3 indexNorm = packet.landmarks[5];
    const Vector3 pinkyNorm = packet.landmarks[17];
    const Vector3 middleNorm = packet.landmarks[9];
    const Vector3 ringNorm = packet.landmarks[13];
    const Vector3 knuckleCenterNorm = Vector3Scale(Vector3Add(Vector3Add(indexNorm, middleNorm), Vector3Add(ringNorm, pinkyNorm)), 0.25f);
    const Vector3 sideKnuckleCenterNorm = Vector3Scale(Vector3Add(indexNorm, pinkyNorm), 0.5f);
    const Vector3 stableAnchorNorm = Vector3Add(Vector3Scale(wristNorm, 0.62f), Vector3Scale(sideKnuckleCenterNorm, 0.38f));
    const Vector3 rootAnchorNorm = Vector3Add(Vector3Scale(wristNorm, 0.88f), Vector3Scale(sideKnuckleCenterNorm, 0.12f));
    const float palm = LandmarkPalmNorm(packet.landmarks);

    const Vector3 origin = {
        (rootAnchorNorm.x - 0.50f) * 24.0f + (rightHand ? 0.4f : -0.4f),
        2.6f + (0.66f - rootAnchorNorm.y) * 8.2f,
        rightHand ? 0.45f : -0.45f,
    };

    const float xyScale = 3.8f / palm;
    const float zScale = 5.2f / palm;

    for (size_t i = 0; i < g.landmarks.size(); ++i) {
        const Vector3 rel = Vector3Subtract(packet.landmarks[i], rootAnchorNorm);
        const Vector3 local = {
            rel.x * xyScale,
            -rel.y * xyScale,
            -rel.z * zScale,
        };
        g.landmarks[i] = Vector3Add(origin, local);
        g.radii[i] = kRadii[i];
    }

    const Vector3 across = Vector3Subtract(g.landmarks[17], g.landmarks[5]);
    const Vector3 wristAxis = SafeNormalize(across, {1.0f, 0.0f, 0.0f});
    const Vector3 forearmDir = SafeNormalize(Vector3Subtract(g.landmarks[0], Average({g.landmarks[5], g.landmarks[9], g.landmarks[13], g.landmarks[17]})), {0.0f, -1.0f, 0.0f});

    g.wristLeft = Vector3Add(Vector3Add(g.landmarks[0], Vector3Scale(wristAxis, -0.88f)), Vector3Scale(forearmDir, 0.18f));
    g.wristRight = Vector3Add(Vector3Add(g.landmarks[0], Vector3Scale(wristAxis, 0.88f)), Vector3Scale(forearmDir, 0.18f));
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

struct HandVisualStyle {
    Color palm{};
    Color bone{};
    Color tip{};
    Color tendon{};
    Color accent{};
    Color shadow{};
};

HandVisualStyle StyleForHand(bool rightHand, bool tracked) {
    if (rightHand) {
        return tracked
            ? HandVisualStyle{Color{236, 184, 146, 255}, Color{212, 155, 118, 255}, Color{248, 212, 176, 255},
                              Color{142, 96, 74, 255}, Color{255, 210, 126, 255}, Color{76, 44, 28, 120}}
            : HandVisualStyle{Color{222, 182, 146, 255}, Color{205, 156, 120, 255}, Color{235, 198, 166, 255},
                              Color{142, 104, 82, 255}, Color{255, 206, 118, 255}, Color{76, 44, 28, 120}};
    }
    return tracked
        ? HandVisualStyle{Color{150, 190, 236, 255}, Color{112, 156, 214, 255}, Color{188, 220, 255, 255},
                          Color{74, 98, 148, 255}, Color{132, 224, 255, 255}, Color{28, 40, 76, 120}}
        : HandVisualStyle{Color{144, 180, 228, 255}, Color{104, 146, 205, 255}, Color{176, 210, 248, 255},
                          Color{72, 96, 138, 255}, Color{126, 214, 255, 255}, Color{28, 40, 76, 120}};
}

void UpdateHandKinematics(
    HandKinematics& out,
    const HandGeometry& g,
    bool active,
    bool pinched,
    Vector3* prevAnchor,
    bool* prevValid,
    float dt) {
    out.active = active;
    out.pinched = pinched;
    if (!active) {
        out.velocity = {0.0f, 0.0f, 0.0f};
        *prevValid = false;
        return;
    }

    out.palm = g.palmCenter;
    out.indexTip = g.landmarks[8];
    out.thumbTip = g.landmarks[4];
    out.grabPoint = Vector3Scale(Vector3Add(out.indexTip, out.thumbTip), 0.5f);

    const Vector3 anchor = pinched ? out.grabPoint : out.palm;
    if (*prevValid && dt > 1.0e-4f) {
        out.velocity = Vector3Scale(Vector3Subtract(anchor, *prevAnchor), 1.0f / dt);
    } else {
        out.velocity = {0.0f, 0.0f, 0.0f};
    }
    *prevAnchor = anchor;
    *prevValid = true;
}

void UpdateBall(
    BallState& ball,
    const std::array<HandKinematics, 2>& hands,
    float dt) {
    const float hoverY = 3.0f;
    const float worldBoundX = 9.2f;
    const float worldBoundZ = 5.5f;

    if (ball.grabbedBy >= 0) {
        const int idx = ball.grabbedBy;
        if (idx >= static_cast<int>(hands.size()) || !hands[static_cast<size_t>(idx)].active || !hands[static_cast<size_t>(idx)].pinched) {
            ball.grabbedBy = -1;
        } else {
            const HandKinematics& hand = hands[static_cast<size_t>(idx)];
            const Vector3 target = Vector3Add(hand.grabPoint, Vector3{0.0f, 0.0f, 0.0f});
            ball.pos = Vector3Lerp(ball.pos, target, 1.0f - std::exp(-16.0f * dt));
            ball.vel = Vector3Scale(hand.velocity, 0.72f);
            return;
        }
    }

    int grabCandidate = -1;
    float grabBestDist = 1.0e9f;
    for (size_t i = 0; i < hands.size(); ++i) {
        const HandKinematics& hand = hands[i];
        if (!hand.active || !hand.pinched) continue;
        const float dist = Vector3Distance(hand.grabPoint, ball.pos);
        if (dist < ball.radius + 0.55f && dist < grabBestDist) {
            grabBestDist = dist;
            grabCandidate = static_cast<int>(i);
        }
    }
    if (grabCandidate >= 0) {
        ball.grabbedBy = grabCandidate;
        return;
    }

    for (const HandKinematics& hand : hands) {
        if (!hand.active) continue;

        const std::array<Vector3, 2> probes = {hand.indexTip, hand.palm};
        const std::array<float, 2> reach = {0.34f, 0.62f};
        const std::array<float, 2> strength = {1.00f, 0.70f};

        for (size_t j = 0; j < probes.size(); ++j) {
            const Vector3 delta = Vector3Subtract(ball.pos, probes[j]);
            const float dist = Vector3Length(delta);
            const float hitRadius = ball.radius + reach[j];
            if (dist < hitRadius) {
                const Vector3 normal = SafeNormalize(delta, {0.0f, 1.0f, 0.0f});
                const float penetration = hitRadius - dist;
                ball.pos = Vector3Add(ball.pos, Vector3Scale(normal, penetration * 0.55f));
                const float toward = Vector3DotProduct(hand.velocity, normal);
                const float impulse = std::max(0.0f, toward) * 0.16f + penetration * 8.0f;
                ball.vel = Vector3Add(ball.vel, Vector3Scale(normal, impulse * strength[j]));
                ball.vel = Vector3Add(ball.vel, Vector3Scale(hand.velocity, 0.020f * strength[j]));
            }
        }
    }

    ball.vel.y += (hoverY - ball.pos.y) * 2.4f * dt;
    ball.vel = Vector3Scale(ball.vel, std::exp(-1.45f * dt));
    ball.pos = Vector3Add(ball.pos, Vector3Scale(ball.vel, dt));

    if (ball.pos.y < ball.radius + 0.04f) {
        ball.pos.y = ball.radius + 0.04f;
        ball.vel.y = std::fabs(ball.vel.y) * 0.55f;
        ball.vel.x *= 0.92f;
        ball.vel.z *= 0.92f;
    }
    if (ball.pos.x > worldBoundX || ball.pos.x < -worldBoundX) {
        ball.pos.x = std::clamp(ball.pos.x, -worldBoundX, worldBoundX);
        ball.vel.x *= -0.68f;
    }
    if (ball.pos.z > worldBoundZ || ball.pos.z < -worldBoundZ) {
        ball.pos.z = std::clamp(ball.pos.z, -worldBoundZ, worldBoundZ);
        ball.vel.z *= -0.68f;
    }
}

void DrawBall(const BallState& ball) {
    const float pulse = 0.5f + 0.5f * std::sin(2.0f * static_cast<float>(GetTime()));
    const Color glow = Color{120, 220, 255, static_cast<unsigned char>(90 + 50 * pulse)};
    DrawSphere(ball.pos, ball.radius * (1.22f + 0.05f * pulse), Fade(glow, 0.22f));
    DrawSphere(ball.pos, ball.radius, Color{112, 208, 255, 255});
    DrawSphere(ball.pos, ball.radius * 0.76f, Color{214, 244, 255, 255});
    DrawSphereWires(ball.pos, ball.radius * 1.03f, 12, 12, Color{20, 52, 86, 180});
    DrawLine3D(
        ball.pos,
        Vector3Add(ball.pos, Vector3Scale(ball.vel, 0.08f)),
        Color{255, 236, 160, 200});
}

void DrawBone(Vector3 a, Vector3 b, float ra, float rb, Color color) {
    DrawCylinderEx(a, b, ra, rb, static_cast<int>(kBoneSides), color);
    DrawCylinderWiresEx(a, b, ra, rb, static_cast<int>(kBoneSides), Fade(BLACK, 0.25f));
}

void DrawPalmSurface(const HandGeometry& g, Color palmColor, Color highlightColor) {
    for (size_t i = 0; i + 1 < g.palmRim.size(); ++i) {
        DrawTriangle3D(g.palmCenter, g.palmRim[i], g.palmRim[i + 1], palmColor);
    }

    const Color webColor = Fade(palmColor, 0.92f);
    DrawTriangle3D(g.landmarks[1], g.landmarks[5], g.landmarks[6], webColor);
    DrawTriangle3D(g.landmarks[5], g.landmarks[9], g.landmarks[6], webColor);
    DrawTriangle3D(g.landmarks[9], g.landmarks[13], g.landmarks[10], webColor);
    DrawTriangle3D(g.landmarks[13], g.landmarks[17], g.landmarks[14], webColor);
    DrawTriangle3D(g.landmarks[1], g.landmarks[5], g.palmCenter, Fade(highlightColor, 0.35f));
    DrawTriangle3D(g.landmarks[5], g.landmarks[9], g.palmCenter, Fade(highlightColor, 0.18f));
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

void DrawForearm(const HandGeometry& g, const HandVisualStyle& style) {
    const Vector3 wristMid = Vector3Scale(Vector3Add(g.wristLeft, g.wristRight), 0.5f);
    const Vector3 knuckleMid = Average({g.landmarks[5], g.landmarks[9], g.landmarks[13], g.landmarks[17]});
    const Vector3 dir = SafeNormalize(Vector3Subtract(wristMid, knuckleMid), {0.0f, -1.0f, 0.0f});
    const Vector3 forearmEnd = Vector3Add(wristMid, Vector3Scale(dir, 1.55f));
    DrawBone(wristMid, forearmEnd, 0.48f, 0.34f, Fade(style.bone, 0.82f));
    DrawSphere(forearmEnd, 0.34f, Fade(style.palm, 0.78f));
}

void DrawKnuckleBridge(const HandGeometry& g, const HandVisualStyle& style) {
    const std::array<int, 4> ridge = {5, 9, 13, 17};
    for (size_t i = 0; i + 1 < ridge.size(); ++i) {
        DrawLine3D(g.landmarks[ridge[i]], g.landmarks[ridge[i + 1]], Fade(style.accent, 0.55f));
    }
    for (int idx : ridge) {
        DrawSphere(g.landmarks[idx], g.radii[idx] * 0.72f, Fade(style.accent, 0.55f));
    }
}

void DrawPinchCue(const HandGeometry& g, const HandVisualStyle& style) {
    const Vector3 thumb = g.landmarks[4];
    const Vector3 index = g.landmarks[8];
    const Vector3 center = Vector3Scale(Vector3Add(thumb, index), 0.5f);
    DrawLine3D(thumb, index, style.accent);
    DrawSphere(center, 0.16f, Fade(style.accent, 0.92f));
    DrawSphere(thumb, 0.10f, style.accent);
    DrawSphere(index, 0.10f, style.accent);
}

void DrawPalmNormalCue(const HandGeometry& g, const HandVisualStyle& style) {
    const Vector3 across = Vector3Subtract(g.landmarks[17], g.landmarks[5]);
    const Vector3 fingers = Vector3Subtract(Average({g.landmarks[5], g.landmarks[9], g.landmarks[13], g.landmarks[17]}), g.landmarks[0]);
    Vector3 normal = SafeNormalize(Vector3CrossProduct(across, fingers), {0.0f, 0.0f, 1.0f});
    if (normal.z < 0.0f) normal = Vector3Scale(normal, -1.0f);
    const Vector3 tip = Vector3Add(g.palmCenter, Vector3Scale(normal, 0.72f));
    DrawLine3D(g.palmCenter, tip, style.accent);
    DrawSphere(tip, 0.10f, style.accent);
}

void DrawHandShadow(const HandGeometry& g, const HandVisualStyle& style) {
    for (const Vector3& point : g.palmRim) {
        const Vector3 shadow = {point.x, 0.021f, point.z};
        DrawSphere(shadow, 0.18f, Fade(style.shadow, 0.25f));
    }
}

void DrawHandModel(const HandGeometry& g, const HandVisualStyle& style, bool showLandmarks, bool pinched) {
    DrawHandShadow(g, style);
    DrawForearm(g, style);
    DrawPalmSurface(g, style.palm, style.tip);

    const std::array<std::pair<int, int>, 20> bones = {{
        {0, 1}, {1, 2}, {2, 3}, {3, 4},
        {0, 5}, {5, 6}, {6, 7}, {7, 8},
        {0, 9}, {9, 10}, {10, 11}, {11, 12},
        {0, 13}, {13, 14}, {14, 15}, {15, 16},
        {0, 17}, {17, 18}, {18, 19}, {19, 20},
    }};

    for (const auto& [a, b] : bones) {
        const bool fingertipBone = (b == 4 || b == 8 || b == 12 || b == 16 || b == 20);
        DrawBone(g.landmarks[a], g.landmarks[b], g.radii[a] * 0.78f, g.radii[b] * 0.82f, fingertipBone ? style.tip : style.bone);
    }

    const std::array<int, 4> metacarpalTargets = {5, 9, 13, 17};
    for (int idx : metacarpalTargets) {
        DrawBone(g.landmarks[0], g.landmarks[idx], 0.14f, g.radii[idx] * 0.90f, Fade(style.bone, 0.75f));
    }
    DrawBone(g.landmarks[0], g.landmarks[1], 0.16f, g.radii[1] * 0.95f, Fade(style.bone, 0.72f));

    DrawTendonLines(g, style.tendon);
    DrawPalmLines(g, Fade(style.tendon, 0.65f));
    DrawKnuckleBridge(g, style);

    for (size_t i = 0; i < g.landmarks.size(); ++i) {
        const bool tip = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20);
        DrawSphere(g.landmarks[i], g.radii[i] * kJointSphereScale, tip ? style.tip : style.palm);
    }

    DrawSphere(g.palmCenter, 0.33f, Fade(style.palm, 0.85f));
    DrawSphere(g.landmarks[1], 0.18f, Fade(style.accent, 0.30f));
    DrawPalmNormalCue(g, style);

    if (pinched) {
        DrawPinchCue(g, style);
    }

    if (showLandmarks) {
        for (size_t i = 0; i < g.landmarks.size(); ++i) {
            DrawSphere(g.landmarks[i], g.radii[i] * 0.42f, style.accent);
        }
    }
}

std::string HudLine(const HandControls& c, bool autoDemo, bool showLandmarks, bool showMirrorDemo) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "grip=" << c.grip
       << "  spread=" << c.spread
       << "  pinch=" << c.pinch
       << "  thumb=" << c.thumbOpposition
       << "  wrist=(" << c.wristPitch << ", " << c.wristYaw << ", " << c.wristRoll << ")";
    if (autoDemo) os << "  [AUTO]";
    if (showMirrorDemo) os << "  [DUAL]";
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
    bool showLandmarks = false;
    bool showMirrorDemo = true;
    HandControls manual = PresetControls(HandPreset::Relaxed);
    float demoTime = 0.0f;

    UdpHandReceiver receiver;
    const bool receiverOk = receiver.Start(static_cast<uint16_t>(kUdpPort));
    UdpFrameReceiver frameReceiver;
    const bool frameReceiverOk = frameReceiver.Start(static_cast<uint16_t>(kFrameUdpPort));
    TrackingPacket tracking{};
    float lastPacketWallClock = -100.0f;
    float lastFramePacketWallClock = -100.0f;
    std::array<HandGeometry, 2> liveGeometry{};
    std::array<bool, 2> liveGeometryInit = {false, false};
    std::array<HandKinematics, 2> handState{};
    std::array<Vector3, 2> prevHandAnchor = {Vector3{0.0f, 0.0f, 0.0f}, Vector3{0.0f, 0.0f, 0.0f}};
    std::array<bool, 2> prevHandAnchorValid = {false, false};
    BallState ball{};
    std::vector<unsigned char> previewFrameBytes;
    Texture2D webcamTexture{};

    while (!WindowShouldClose()) {
        const float dt = std::max(GetFrameTime(), 1.0e-4f);
        const float now = static_cast<float>(GetTime());

        if (IsKeyPressed(KEY_SPACE)) autoDemo = !autoDemo;
        if (IsKeyPressed(KEY_L)) showLandmarks = !showLandmarks;
        if (IsKeyPressed(KEY_M)) showMirrorDemo = !showMirrorDemo;
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
        int framePacketsRead = 0;
        if (frameReceiver.Poll(previewFrameBytes, framePacketsRead)) {
            lastFramePacketWallClock = now;
            UpdatePreviewTexture(webcamTexture, previewFrameBytes);
        }

        const bool linkLive = receiver.ready() && ((now - lastPacketWallClock) < kLinkTimeout);
        const bool previewLive = frameReceiver.ready() && webcamTexture.id > 0 && ((now - lastFramePacketWallClock) < kLinkTimeout);
        const bool leftTracked = linkLive && tracking.left.valid;
        const bool rightTracked = linkLive && tracking.right.valid;
        const bool anyTracked = leftTracked || rightTracked;

        if (autoDemo && !anyTracked) demoTime += dt;
        const HandControls pose = autoDemo ? DemoControls(demoTime) : manual;
        const float liveBlend = 1.0f - std::exp(-12.0f * dt);
        Vector3 depthAxis = Vector3Subtract(camera.position, camera.target);
        depthAxis.y *= 0.16f;
        depthAxis = SafeNormalize(depthAxis, {0.72f, 0.08f, 0.69f});

        if (leftTracked) {
            HandGeometry target = BuildTrackedGeometry(tracking.left, false);
            target = OffsetGeometry(target, Vector3Scale(depthAxis, EstimateTrackedDepthShift(tracking.left)));
            liveGeometry[0] = liveGeometryInit[0] ? BlendGeometry(liveGeometry[0], target, liveBlend) : target;
            liveGeometryInit[0] = true;
        } else {
            liveGeometryInit[0] = false;
        }

        if (rightTracked) {
            HandGeometry target = BuildTrackedGeometry(tracking.right, true);
            target = OffsetGeometry(target, Vector3Scale(depthAxis, EstimateTrackedDepthShift(tracking.right)));
            liveGeometry[1] = liveGeometryInit[1] ? BlendGeometry(liveGeometry[1], target, liveBlend) : target;
            liveGeometryInit[1] = true;
        } else {
            liveGeometryInit[1] = false;
        }

        UpdateHandKinematics(handState[0], liveGeometry[0], leftTracked && liveGeometryInit[0], tracking.left.pinched, &prevHandAnchor[0], &prevHandAnchorValid[0], dt);
        UpdateHandKinematics(handState[1], liveGeometry[1], rightTracked && liveGeometryInit[1], tracking.right.pinched, &prevHandAnchor[1], &prevHandAnchorValid[1], dt);
        UpdateBall(ball, handState, dt);

        const HandGeometry demoRight = BuildHandGeometry(pose, {4.5f, 0.65f, 0.0f}, true);
        const HandGeometry demoLeft = BuildHandGeometry(pose, {-4.5f, 0.65f, 0.0f}, false);
        const bool demoPinch = pose.pinch > 0.65f;

        BeginDrawing();
        ClearBackground(Color{7, 10, 18, 255});

        BeginMode3D(camera);
        DrawCube({0.0f, -0.05f, 0.0f}, 10.5f, 0.08f, 9.0f, Color{34, 44, 62, 255});
        DrawCube({0.0f, 0.02f, 0.0f}, 4.2f, 0.05f, 3.2f, Color{46, 58, 78, 255});
        DrawBall(ball);

        if (anyTracked) {
            if (leftTracked) DrawHandModel(liveGeometry[0], StyleForHand(false, true), showLandmarks, tracking.left.pinched);
            if (rightTracked) DrawHandModel(liveGeometry[1], StyleForHand(true, true), showLandmarks, tracking.right.pinched);
        } else {
            if (showMirrorDemo) DrawHandModel(demoLeft, StyleForHand(false, false), showLandmarks, demoPinch);
            DrawHandModel(demoRight, StyleForHand(true, false), showLandmarks, demoPinch);
        }
        EndMode3D();

        DrawText("3D Human Hand Biomechanics", 20, 18, 31, Color{230, 236, 245, 255});
        DrawText(
            "Mouse drag/wheel: camera | Space auto-demo | 1-5 presets | M demo pair | L landmark ids | R relaxed reset",
            20, 56, 19, Color{164, 183, 210, 255});
        DrawText(
            "[ ] grip | ; ' spread | , . thumb opposition | -/+ pinch | W/S pitch | A/D yaw | Q/E roll",
            20, 82, 19, Color{164, 183, 210, 255});

        const std::string hud = HudLine(pose, autoDemo, showLandmarks, showMirrorDemo);
        DrawText(hud.c_str(), 20, 114, 20, Color{126, 224, 255, 255});
        DrawText(
            anyTracked
                ? "Live webcam tracking is driving the full two-hand 21-point rig with palm-oriented 3D reconstruction."
                : "Demo mode now renders both hands and the same 21-point rig used by the webcam bridge.",
            20, 142, 18, Color{194, 205, 223, 255});
        DrawText("Ball: pinch near it to grab, or tap/push it with fingertips or palm.", 20, 168, 18, Color{194, 205, 223, 255});

        const char* bridgeStatus =
            !receiverOk ? "bridge: UDP receiver failed to start"
            : anyTracked ? "bridge: tracking live on udp:50515"
            : linkLive ? "bridge: packets arriving, waiting for valid left/right hands"
            : "bridge: idle on udp:50515  run AstroPhysics/vision/hand_biomechanics_bridge.py";
        DrawText(bridgeStatus, 20, 194, 18, anyTracked ? Color{142, 255, 190, 255} : Color{188, 198, 220, 255});

        if (anyTracked) {
            std::ostringstream liveOs;
            liveOs << "live hands:"
                   << (leftTracked ? " L" : "")
                   << (rightTracked ? " R" : "")
                   << "  pinch:"
                   << (tracking.left.pinched ? " L" : "")
                   << (tracking.right.pinched ? " R" : "")
                   << "  ball:"
                   << (ball.grabbedBy == 0 ? " grabbed L" : ball.grabbedBy == 1 ? " grabbed R" : " free");
            DrawText(liveOs.str().c_str(), 20, 220, 18, Color{255, 224, 132, 255});
        }
        DrawFPS(20, 248);

        const Rectangle previewPanel = {static_cast<float>(GetScreenWidth() - 392), 20.0f, 360.0f, 220.0f};
        DrawRectangleRounded(previewPanel, 0.06f, 10, Fade(BLACK, 0.38f));
        DrawRectangleLinesEx(previewPanel, 1.5f, Color{92, 110, 138, 255});
        DrawText("Python Webcam Feed", static_cast<int>(previewPanel.x) + 14, static_cast<int>(previewPanel.y) + 12, 20, Color{222, 230, 244, 255});
        if (previewLive) {
            const Rectangle src = {0.0f, 0.0f, static_cast<float>(webcamTexture.width), static_cast<float>(webcamTexture.height)};
            const Rectangle dst = {previewPanel.x + 12.0f, previewPanel.y + 42.0f, previewPanel.width - 24.0f, previewPanel.height - 54.0f};
            DrawTexturePro(webcamTexture, src, dst, {0.0f, 0.0f}, 0.0f, WHITE);
        } else {
            DrawRectangle(static_cast<int>(previewPanel.x) + 12, static_cast<int>(previewPanel.y) + 42, static_cast<int>(previewPanel.width) - 24, static_cast<int>(previewPanel.height) - 54, Color{20, 24, 32, 255});
            const char* previewStatus =
                !frameReceiverOk ? "preview receiver failed"
                : "waiting for webcam preview";
            DrawText(previewStatus, static_cast<int>(previewPanel.x) + 24, static_cast<int>(previewPanel.y) + 122, 18, Color{180, 194, 214, 255});
        }

        if (showLandmarks) {
            if (anyTracked) {
                if (leftTracked) DrawLandmarkLabels(liveGeometry[0], camera);
                if (rightTracked) DrawLandmarkLabels(liveGeometry[1], camera);
            } else {
                if (showMirrorDemo) DrawLandmarkLabels(demoLeft, camera);
                DrawLandmarkLabels(demoRight, camera);
            }
        }

        EndDrawing();
    }

    if (webcamTexture.id > 0) {
        UnloadTexture(webcamTexture);
    }
    CloseWindow();
    return 0;
}
