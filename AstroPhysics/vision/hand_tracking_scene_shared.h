#pragma once

#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

namespace astro_hand {

constexpr int kUdpPort = 50515;
constexpr int kFrameUdpPort = 50516;
constexpr float kLinkTimeout = 0.75f;
constexpr float kTrackedDepthPalmRef = 0.155f;
constexpr float kTrackedDepthRange = 5.4f;
constexpr float kTrackedHandModelScale = 0.18f;
constexpr float kBoneSides = 8.0f;
constexpr float kJointSphereScale = 1.18f;
constexpr float kPreviewUpdateInterval = 1.0f / 24.0f;

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

inline float Clamp01(float v) {
    return std::clamp(v, 0.0f, 1.0f);
}

inline float LerpFloat(float a, float b, float t) {
    return a + (b - a) * t;
}

inline Vector3 SafeNormalize(Vector3 v, Vector3 fallback) {
    if (Vector3Length(v) < 1.0e-4f) return fallback;
    return Vector3Normalize(v);
}

inline Vector3 Average(const std::initializer_list<Vector3>& pts) {
    Vector3 sum{0.0f, 0.0f, 0.0f};
    for (const Vector3& p : pts) sum = Vector3Add(sum, p);
    return Vector3Scale(sum, 1.0f / static_cast<float>(pts.size()));
}

inline float LandmarkPalmNorm(const std::array<Vector3, 21>& pts) {
    const auto dist2 = [&](int a, int b) {
        const float dx = pts[static_cast<size_t>(a)].x - pts[static_cast<size_t>(b)].x;
        const float dy = pts[static_cast<size_t>(a)].y - pts[static_cast<size_t>(b)].y;
        return std::sqrt(dx * dx + dy * dy);
    };
    return std::max(1.0e-4f, (dist2(0, 5) + dist2(0, 17) + dist2(5, 17)) / 3.0f);
}

inline bool UpdatePreviewTexture(Texture2D& texture, const std::vector<unsigned char>& jpgBytes) {
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

struct HandGeometry {
    std::array<Vector3, 21> landmarks{};
    std::array<float, 21> radii{};
    std::array<Vector3, 7> palmRim{};
    Vector3 palmCenter{};
    Vector3 wristLeft{};
    Vector3 wristRight{};
};

inline HandGeometry BlendGeometry(const HandGeometry& a, const HandGeometry& b, float t) {
    HandGeometry out{};
    for (size_t i = 0; i < out.landmarks.size(); ++i) {
        out.landmarks[i] = Vector3Lerp(a.landmarks[i], b.landmarks[i], t);
        out.radii[i] = LerpFloat(a.radii[i], b.radii[i], t);
    }
    for (size_t i = 0; i < out.palmRim.size(); ++i) out.palmRim[i] = Vector3Lerp(a.palmRim[i], b.palmRim[i], t);
    out.palmCenter = Vector3Lerp(a.palmCenter, b.palmCenter, t);
    out.wristLeft = Vector3Lerp(a.wristLeft, b.wristLeft, t);
    out.wristRight = Vector3Lerp(a.wristRight, b.wristRight, t);
    return out;
}

inline HandGeometry OffsetGeometry(const HandGeometry& g, Vector3 offset) {
    HandGeometry out = g;
    for (Vector3& point : out.landmarks) point = Vector3Add(point, offset);
    for (Vector3& point : out.palmRim) point = Vector3Add(point, offset);
    out.palmCenter = Vector3Add(out.palmCenter, offset);
    out.wristLeft = Vector3Add(out.wristLeft, offset);
    out.wristRight = Vector3Add(out.wristRight, offset);
    return out;
}

inline float EstimateTrackedDepthShift(const TrackedHandPacket& packet) {
    const float palm = LandmarkPalmNorm(packet.landmarks);
    const float relative = std::clamp((kTrackedDepthPalmRef / palm) - 1.0f, -0.55f, 0.90f);
    return -relative * kTrackedDepthRange;
}

inline float HandVisualScale(const HandGeometry& g) {
    return std::max(0.14f, g.radii[0] / 0.34f);
}

inline HandGeometry BuildTrackedGeometry(const TrackedHandPacket& packet, bool rightHand) {
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
    const Vector3 sideKnuckleCenterNorm = Vector3Scale(Vector3Add(indexNorm, pinkyNorm), 0.5f);
    const Vector3 rootAnchorNorm = Vector3Add(Vector3Scale(wristNorm, 0.88f), Vector3Scale(sideKnuckleCenterNorm, 0.12f));
    const float palm = LandmarkPalmNorm(packet.landmarks);

    const Vector3 origin = {
        (rootAnchorNorm.x - 0.50f) * 24.0f + (rightHand ? 0.4f : -0.4f),
        2.6f + (0.66f - rootAnchorNorm.y) * 8.2f,
        rightHand ? 0.45f : -0.45f,
    };

    const float xyScale = (3.8f / palm) * kTrackedHandModelScale;
    const float zScale = (5.2f / palm) * kTrackedHandModelScale;

    for (size_t i = 0; i < g.landmarks.size(); ++i) {
        const Vector3 rel = Vector3Subtract(packet.landmarks[i], rootAnchorNorm);
        const Vector3 local = {
            rel.x * xyScale,
            -rel.y * xyScale,
            -rel.z * zScale,
        };
        g.landmarks[i] = Vector3Add(origin, local);
        g.radii[i] = kRadii[i] * kTrackedHandModelScale;
    }

    const Vector3 across = Vector3Subtract(g.landmarks[17], g.landmarks[5]);
    const Vector3 wristAxis = SafeNormalize(across, {1.0f, 0.0f, 0.0f});
    const Vector3 forearmDir = SafeNormalize(Vector3Subtract(g.landmarks[0], Average({g.landmarks[5], g.landmarks[9], g.landmarks[13], g.landmarks[17]})), {0.0f, -1.0f, 0.0f});

    g.wristLeft = Vector3Add(Vector3Add(g.landmarks[0], Vector3Scale(wristAxis, -0.88f * kTrackedHandModelScale)), Vector3Scale(forearmDir, 0.18f * kTrackedHandModelScale));
    g.wristRight = Vector3Add(Vector3Add(g.landmarks[0], Vector3Scale(wristAxis, 0.88f * kTrackedHandModelScale)), Vector3Scale(forearmDir, 0.18f * kTrackedHandModelScale));
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

inline void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0032f;
        *pitch += d.y * 0.0030f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.7f;
    *distance = std::clamp(*distance, 6.0f, 34.0f);

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

inline HandVisualStyle StyleForHand(bool rightHand) {
    if (rightHand) {
        return HandVisualStyle{Color{236, 184, 146, 255}, Color{212, 155, 118, 255}, Color{248, 212, 176, 255},
                               Color{142, 96, 74, 255}, Color{255, 210, 126, 255}, Color{76, 44, 28, 120}};
    }
    return HandVisualStyle{Color{150, 190, 236, 255}, Color{112, 156, 214, 255}, Color{188, 220, 255, 255},
                           Color{74, 98, 148, 255}, Color{132, 224, 255, 255}, Color{28, 40, 76, 120}};
}

inline void DrawBone(Vector3 a, Vector3 b, float ra, float rb, Color color) {
    DrawCylinderEx(a, b, ra, rb, static_cast<int>(kBoneSides), color);
    DrawCylinderWiresEx(a, b, ra, rb, static_cast<int>(kBoneSides), Fade(BLACK, 0.25f));
}

inline void DrawPalmSurface(const HandGeometry& g, Color palmColor, Color highlightColor) {
    for (size_t i = 0; i + 1 < g.palmRim.size(); ++i) DrawTriangle3D(g.palmCenter, g.palmRim[i], g.palmRim[i + 1], palmColor);
    const Color webColor = Fade(palmColor, 0.92f);
    DrawTriangle3D(g.landmarks[1], g.landmarks[5], g.landmarks[6], webColor);
    DrawTriangle3D(g.landmarks[5], g.landmarks[9], g.landmarks[6], webColor);
    DrawTriangle3D(g.landmarks[9], g.landmarks[13], g.landmarks[10], webColor);
    DrawTriangle3D(g.landmarks[13], g.landmarks[17], g.landmarks[14], webColor);
    DrawTriangle3D(g.landmarks[1], g.landmarks[5], g.palmCenter, Fade(highlightColor, 0.35f));
    DrawTriangle3D(g.landmarks[5], g.landmarks[9], g.palmCenter, Fade(highlightColor, 0.18f));
}

inline void DrawPalmLines(const HandGeometry& g, Color color) {
    for (size_t i = 0; i + 1 < g.palmRim.size(); ++i) DrawLine3D(g.palmRim[i], g.palmRim[i + 1], color);
    DrawLine3D(g.wristLeft, g.wristRight, color);
    DrawLine3D(g.landmarks[0], g.landmarks[5], Fade(color, 0.85f));
    DrawLine3D(g.landmarks[0], g.landmarks[9], Fade(color, 0.90f));
    DrawLine3D(g.landmarks[0], g.landmarks[13], Fade(color, 0.85f));
    DrawLine3D(g.landmarks[0], g.landmarks[17], Fade(color, 0.80f));
}

inline void DrawTendonLines(const HandGeometry& g, Color color) {
    const std::array<int, 4> fingerStarts = {5, 9, 13, 17};
    for (int idx : fingerStarts) {
        DrawLine3D(g.landmarks[0], g.landmarks[idx + 1], Fade(color, 0.65f));
        DrawLine3D(g.landmarks[idx], g.landmarks[idx + 2], Fade(color, 0.50f));
    }
    DrawLine3D(g.landmarks[1], g.landmarks[3], Fade(color, 0.55f));
}

inline void DrawForearm(const HandGeometry& g, const HandVisualStyle& style) {
    const float s = HandVisualScale(g);
    const Vector3 wristMid = Vector3Scale(Vector3Add(g.wristLeft, g.wristRight), 0.5f);
    const Vector3 knuckleMid = Average({g.landmarks[5], g.landmarks[9], g.landmarks[13], g.landmarks[17]});
    const Vector3 dir = SafeNormalize(Vector3Subtract(wristMid, knuckleMid), {0.0f, -1.0f, 0.0f});
    const Vector3 forearmEnd = Vector3Add(wristMid, Vector3Scale(dir, 1.55f * s));
    DrawBone(wristMid, forearmEnd, 0.48f * s, 0.34f * s, Fade(style.bone, 0.82f));
    DrawSphere(forearmEnd, 0.34f * s, Fade(style.palm, 0.78f));
}

inline void DrawKnuckleBridge(const HandGeometry& g, const HandVisualStyle& style) {
    const std::array<int, 4> ridge = {5, 9, 13, 17};
    for (size_t i = 0; i + 1 < ridge.size(); ++i) DrawLine3D(g.landmarks[ridge[i]], g.landmarks[ridge[i + 1]], Fade(style.accent, 0.55f));
    for (int idx : ridge) DrawSphere(g.landmarks[idx], g.radii[idx] * 0.72f, Fade(style.accent, 0.55f));
}

inline void DrawPinchCue(const HandGeometry& g, const HandVisualStyle& style) {
    const float s = HandVisualScale(g);
    const Vector3 thumb = g.landmarks[4];
    const Vector3 index = g.landmarks[8];
    const Vector3 center = Vector3Scale(Vector3Add(thumb, index), 0.5f);
    DrawLine3D(thumb, index, style.accent);
    DrawSphere(center, 0.16f * s, Fade(style.accent, 0.92f));
    DrawSphere(thumb, 0.10f * s, style.accent);
    DrawSphere(index, 0.10f * s, style.accent);
}

inline void DrawPalmNormalCue(const HandGeometry& g, const HandVisualStyle& style) {
    const float s = HandVisualScale(g);
    const Vector3 across = Vector3Subtract(g.landmarks[17], g.landmarks[5]);
    const Vector3 fingers = Vector3Subtract(Average({g.landmarks[5], g.landmarks[9], g.landmarks[13], g.landmarks[17]}), g.landmarks[0]);
    Vector3 normal = SafeNormalize(Vector3CrossProduct(across, fingers), {0.0f, 0.0f, 1.0f});
    if (normal.z < 0.0f) normal = Vector3Scale(normal, -1.0f);
    const Vector3 tip = Vector3Add(g.palmCenter, Vector3Scale(normal, 0.72f * s));
    DrawLine3D(g.palmCenter, tip, style.accent);
    DrawSphere(tip, 0.10f * s, style.accent);
}

inline void DrawHandShadow(const HandGeometry& g, const HandVisualStyle& style) {
    const float s = HandVisualScale(g);
    for (const Vector3& point : g.palmRim) {
        const Vector3 shadow = {point.x, 0.021f, point.z};
        DrawSphere(shadow, 0.18f * s, Fade(style.shadow, 0.25f));
    }
}

inline void DrawHandModel(const HandGeometry& g, bool rightHand, bool showLandmarks, bool pinched) {
    const HandVisualStyle style = StyleForHand(rightHand);
    const float s = HandVisualScale(g);
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

    for (const auto& bone : bones) {
        const bool fingertipBone = (bone.second == 4 || bone.second == 8 || bone.second == 12 || bone.second == 16 || bone.second == 20);
        DrawBone(g.landmarks[bone.first], g.landmarks[bone.second], g.radii[bone.first] * 0.78f, g.radii[bone.second] * 0.82f, fingertipBone ? style.tip : style.bone);
    }

    const std::array<int, 4> metacarpalTargets = {5, 9, 13, 17};
    for (int idx : metacarpalTargets) DrawBone(g.landmarks[0], g.landmarks[idx], 0.14f * s, g.radii[idx] * 0.90f, Fade(style.bone, 0.75f));
    DrawBone(g.landmarks[0], g.landmarks[1], 0.16f * s, g.radii[1] * 0.95f, Fade(style.bone, 0.72f));

    DrawTendonLines(g, style.tendon);
    DrawPalmLines(g, Fade(style.tendon, 0.65f));
    DrawKnuckleBridge(g, style);

    for (size_t i = 0; i < g.landmarks.size(); ++i) {
        const bool tip = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20);
        DrawSphere(g.landmarks[i], g.radii[i] * kJointSphereScale, tip ? style.tip : style.palm);
    }

    DrawSphere(g.palmCenter, 0.33f * s, Fade(style.palm, 0.85f));
    DrawSphere(g.landmarks[1], 0.18f * s, Fade(style.accent, 0.30f));
    DrawPalmNormalCue(g, style);
    if (pinched) DrawPinchCue(g, style);

    if (showLandmarks) {
        for (size_t i = 0; i < g.landmarks.size(); ++i) DrawSphere(g.landmarks[i], g.radii[i] * 0.42f, style.accent);
    }
}

struct HandControlState {
    bool active = false;
    bool pinched = false;
    bool rightHand = false;
    float score = 0.0f;
    float palmSize = 0.0f;
    Vector3 wrist{};
    Vector3 palm{};
    Vector3 thumbTip{};
    Vector3 indexTip{};
    Vector3 pinchPoint{};
    Vector3 velocity{};
};

class HandSceneBridge {
  public:
    bool Start() {
        receiverOk_ = receiver_.Start(static_cast<uint16_t>(kUdpPort));
        frameReceiverOk_ = frameReceiver_.Start(static_cast<uint16_t>(kFrameUdpPort));
        return receiverOk_ || frameReceiverOk_;
    }

    void Shutdown() {
        receiver_.Close();
        frameReceiver_.Close();
        if (webcamTexture_.id > 0) {
            UnloadTexture(webcamTexture_);
            webcamTexture_ = Texture2D{};
        }
    }

    ~HandSceneBridge() {
        Shutdown();
    }

    void Update(const Camera3D& camera, float now, float dt) {
        int packetsRead = 0;
        if (receiver_.Poll(tracking_, packetsRead)) lastPacketWallClock_ = now;

        int framePacketsRead = 0;
        if (frameReceiver_.Poll(previewFrameBytes_, framePacketsRead)) {
            lastFramePacketWallClock_ = now;
            previewDirty_ = true;
        }
        if (previewDirty_ && !previewFrameBytes_.empty() &&
            (webcamTexture_.id == 0 || (now - lastPreviewTextureUpdateWallClock_) >= kPreviewUpdateInterval)) {
            if (UpdatePreviewTexture(webcamTexture_, previewFrameBytes_)) {
                lastPreviewTextureUpdateWallClock_ = now;
                previewDirty_ = false;
            }
        }

        linkLive_ = receiver_.ready() && ((now - lastPacketWallClock_) < kLinkTimeout);
        previewLive_ = frameReceiver_.ready() && webcamTexture_.id > 0 && ((now - lastFramePacketWallClock_) < kLinkTimeout);
        leftTracked_ = linkLive_ && tracking_.left.valid;
        rightTracked_ = linkLive_ && tracking_.right.valid;

        Vector3 depthAxis = Vector3Subtract(camera.position, camera.target);
        depthAxis.y *= 0.16f;
        depthAxis = SafeNormalize(depthAxis, {0.72f, 0.08f, 0.69f});
        const float liveBlend = 1.0f - std::exp(-12.0f * dt);

        if (leftTracked_) {
            HandGeometry target = BuildTrackedGeometry(tracking_.left, false);
            target = OffsetGeometry(target, Vector3Scale(depthAxis, EstimateTrackedDepthShift(tracking_.left)));
            liveGeometry_[0] = liveGeometryInit_[0] ? BlendGeometry(liveGeometry_[0], target, liveBlend) : target;
            liveGeometryInit_[0] = true;
        } else {
            liveGeometryInit_[0] = false;
        }

        if (rightTracked_) {
            HandGeometry target = BuildTrackedGeometry(tracking_.right, true);
            target = OffsetGeometry(target, Vector3Scale(depthAxis, EstimateTrackedDepthShift(tracking_.right)));
            liveGeometry_[1] = liveGeometryInit_[1] ? BlendGeometry(liveGeometry_[1], target, liveBlend) : target;
            liveGeometryInit_[1] = true;
        } else {
            liveGeometryInit_[1] = false;
        }

        UpdateControl(control_[0], liveGeometry_[0], leftTracked_, false, tracking_.left.score, tracking_.left.pinched, &prevAnchor_[0], &prevValid_[0], dt);
        UpdateControl(control_[1], liveGeometry_[1], rightTracked_, true, tracking_.right.score, tracking_.right.pinched, &prevAnchor_[1], &prevValid_[1], dt);
    }

    bool AnyTracked() const { return leftTracked_ || rightTracked_; }
    bool LeftTracked() const { return leftTracked_; }
    bool RightTracked() const { return rightTracked_; }
    bool PreviewLive() const { return previewLive_; }
    bool ReceiverOk() const { return receiverOk_; }
    bool FrameReceiverOk() const { return frameReceiverOk_; }
    const Texture2D& WebcamTexture() const { return webcamTexture_; }
    const HandGeometry& Geometry(bool rightHand) const { return liveGeometry_[rightHand ? 1 : 0]; }
    const HandControlState& Control(bool rightHand) const { return control_[rightHand ? 1 : 0]; }

    void DrawHands(bool showLandmarks = false) const {
        if (leftTracked_) DrawHandModel(liveGeometry_[0], false, showLandmarks, tracking_.left.pinched);
        if (rightTracked_) DrawHandModel(liveGeometry_[1], true, showLandmarks, tracking_.right.pinched);
    }

    void DrawPreviewPanel(Rectangle panel, const char* title) const {
        DrawRectangleRounded(panel, 0.06f, 10, Fade(BLACK, 0.38f));
        DrawRectangleLinesEx(panel, 1.5f, Color{92, 110, 138, 255});
        DrawText(title, static_cast<int>(panel.x) + 14, static_cast<int>(panel.y) + 12, 20, Color{222, 230, 244, 255});
        if (previewLive_) {
            const Rectangle src = {0.0f, 0.0f, static_cast<float>(webcamTexture_.width), static_cast<float>(webcamTexture_.height)};
            const Rectangle dst = {panel.x + 12.0f, panel.y + 42.0f, panel.width - 24.0f, panel.height - 54.0f};
            DrawTexturePro(webcamTexture_, src, dst, {0.0f, 0.0f}, 0.0f, WHITE);
        } else {
            DrawRectangle(static_cast<int>(panel.x) + 12, static_cast<int>(panel.y) + 42, static_cast<int>(panel.width) - 24, static_cast<int>(panel.height) - 54, Color{20, 24, 32, 255});
            const char* previewStatus = !frameReceiverOk_ ? "preview receiver failed" : "waiting for webcam preview";
            DrawText(previewStatus, static_cast<int>(panel.x) + 24, static_cast<int>(panel.y) + 122, 18, Color{180, 194, 214, 255});
        }
    }

  private:
    static void UpdateControl(
        HandControlState& out,
        const HandGeometry& g,
        bool active,
        bool rightHand,
        float score,
        bool pinched,
        Vector3* prevAnchor,
        bool* prevValid,
        float dt) {
        out.active = active;
        out.pinched = pinched;
        out.rightHand = rightHand;
        out.score = score;
        if (!active) {
            out.velocity = {0.0f, 0.0f, 0.0f};
            out.palmSize = 0.0f;
            *prevValid = false;
            return;
        }

        out.wrist = g.landmarks[0];
        out.palm = g.palmCenter;
        out.thumbTip = g.landmarks[4];
        out.indexTip = g.landmarks[8];
        out.pinchPoint = Vector3Scale(Vector3Add(out.indexTip, out.thumbTip), 0.5f);
        out.palmSize = Vector3Distance(g.landmarks[0], g.landmarks[9]);

        const Vector3 anchor = pinched ? out.pinchPoint : out.palm;
        if (*prevValid && dt > 1.0e-4f) {
            out.velocity = Vector3Scale(Vector3Subtract(anchor, *prevAnchor), 1.0f / dt);
        } else {
            out.velocity = {0.0f, 0.0f, 0.0f};
        }
        *prevAnchor = anchor;
        *prevValid = true;
    }

    UdpHandReceiver receiver_{};
    UdpFrameReceiver frameReceiver_{};
    bool receiverOk_ = false;
    bool frameReceiverOk_ = false;
    bool linkLive_ = false;
    bool previewLive_ = false;
    bool leftTracked_ = false;
    bool rightTracked_ = false;
    float lastPacketWallClock_ = -100.0f;
    float lastFramePacketWallClock_ = -100.0f;
    float lastPreviewTextureUpdateWallClock_ = -100.0f;
    TrackingPacket tracking_{};
    std::array<HandGeometry, 2> liveGeometry_{};
    std::array<bool, 2> liveGeometryInit_ = {false, false};
    std::array<HandControlState, 2> control_{};
    std::array<Vector3, 2> prevAnchor_ = {Vector3{0.0f, 0.0f, 0.0f}, Vector3{0.0f, 0.0f, 0.0f}};
    std::array<bool, 2> prevValid_ = {false, false};
    bool previewDirty_ = false;
    std::vector<unsigned char> previewFrameBytes_{};
    Texture2D webcamTexture_{};
};

inline void DrawBridgeStatus(const HandSceneBridge& bridge, int x, int y) {
    const char* bridgeStatus =
        !bridge.ReceiverOk() ? "bridge: UDP receiver failed to start"
        : bridge.AnyTracked() ? "bridge: tracking live on udp:50515"
        : "bridge: idle on udp:50515  run AstroPhysics/vision/hand_biomechanics_bridge.py";
    DrawText(bridgeStatus, x, y, 18, bridge.AnyTracked() ? Color{142, 255, 190, 255} : Color{188, 198, 220, 255});
}

inline void DrawStarfieldBackdrop(int count, unsigned int seed, float drift, Color baseColor) {
    for (int i = 0; i < count; ++i) {
        const float sx = static_cast<float>((seed * 1103515245u + static_cast<unsigned int>(i * 7919)) % 10000) / 10000.0f;
        const float sy = static_cast<float>((seed * 214013u + static_cast<unsigned int>(i * 4051)) % 10000) / 10000.0f;
        const float twinkle = 0.55f + 0.45f * std::sin(drift * (0.8f + 0.05f * static_cast<float>(i)) + 9.0f * sx);
        const float radius = 0.8f + 1.9f * sx;
        const int px = static_cast<int>(sx * static_cast<float>(GetScreenWidth()));
        const int py = static_cast<int>(sy * static_cast<float>(GetScreenHeight()));
        DrawCircle(px, py, radius, Fade(baseColor, 0.35f + 0.55f * twinkle));
    }
}

}  // namespace astro_hand
