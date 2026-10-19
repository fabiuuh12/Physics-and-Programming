#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

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

constexpr int kWindowW = 1320;
constexpr int kWindowH = 820;
constexpr int kUdpPort = 50505;
constexpr float kHandSmooth = 12.0f;
constexpr float kCursorSmooth = 8.0f;
constexpr float kLinkTimeout = 0.75f;

struct TrackingInput {
    bool leftValid = false;
    float leftX = 0.5f;
    float leftY = 0.5f;
    bool rightValid = false;
    bool rightPinch = false;
    float rightX = 0.5f;
    float rightY = 0.5f;
    bool leftPinch = false;
    double timestamp = 0.0;
};

struct HandPose {
    Vector3 pos = {0.0f, 1.7f, 0.0f};
    bool valid = false;
    bool pinched = false;
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
            int lp = 0;

            const int got = std::sscanf(
                buffer.data(),
                "%lf,%d,%f,%f,%d,%d,%f,%f,%d",
                &parsed.timestamp,
                &lv,
                &parsed.leftX,
                &parsed.leftY,
                &rv,
                &rp,
                &parsed.rightX,
                &parsed.rightY,
                &lp
            );
            if (got >= 6) {
                parsed.leftValid = (lv != 0);
                parsed.rightValid = (rv != 0);
                parsed.rightPinch = (rp != 0);
                if (got < 8) {
                    parsed.rightX = 0.5f;
                    parsed.rightY = 0.5f;
                }
                if (got >= 9) parsed.leftPinch = (lp != 0);

                parsed.leftX = std::clamp(parsed.leftX, 0.0f, 1.0f);
                parsed.leftY = std::clamp(parsed.leftY, 0.0f, 1.0f);
                parsed.rightX = std::clamp(parsed.rightX, 0.0f, 1.0f);
                parsed.rightY = std::clamp(parsed.rightY, 0.0f, 1.0f);

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

Vector3 HandSpaceToWorld(float x, float y, bool rightSide) {
    const float worldX = Lerp(-10.0f, 10.0f, x) + (rightSide ? 0.55f : -0.55f);
    const float worldY = Lerp(1.5f, 8.8f, 1.0f - y);
    const float worldZ = rightSide ? 1.8f : -1.8f;
    return {worldX, worldY, worldZ};
}

void DrawHandAvatar(const HandPose &hand, bool rightHand) {
    if (!hand.valid) return;

    const Color palmColor = rightHand ? Color{255, 190, 126, 255} : Color{125, 198, 255, 255};
    const Color fingerColor = rightHand ? Color{243, 162, 100, 255} : Color{98, 168, 238, 255};
    const Color pinchGlow = rightHand ? Color{255, 240, 130, 255} : Color{152, 255, 236, 255};

    Vector3 palm = hand.pos;
    DrawSphere(palm, 0.42f, palmColor);
    DrawSphereWires(palm, 0.44f, 8, 8, Fade(BLACK, 0.45f));

    const float side = rightHand ? 1.0f : -1.0f;
    std::array<Vector3, 5> tipOffsets = {
        Vector3{0.58f * side, 0.24f, 0.15f},
        Vector3{0.23f * side, 0.68f, -0.06f},
        Vector3{0.03f * side, 0.80f, -0.02f},
        Vector3{-0.20f * side, 0.72f, 0.06f},
        Vector3{-0.44f * side, 0.54f, 0.12f},
    };

    if (hand.pinched) {
        tipOffsets[0] = {0.28f * side, 0.38f, 0.05f};
        tipOffsets[1] = {0.18f * side, 0.42f, 0.00f};
    }

    for (int i = 0; i < 5; ++i) {
        const Vector3 tip = Vector3Add(palm, tipOffsets[static_cast<size_t>(i)]);
        DrawCylinderEx(palm, tip, 0.07f, 0.05f, 8, fingerColor);
        DrawSphere(tip, hand.pinched && (i == 0 || i == 1) ? 0.10f : 0.08f, fingerColor);
    }

    if (hand.pinched) {
        const Vector3 thumb = Vector3Add(palm, tipOffsets[0]);
        const Vector3 index = Vector3Add(palm, tipOffsets[1]);
        const Vector3 mid = Vector3Scale(Vector3Add(thumb, index), 0.5f);
        DrawSphere(mid, 0.14f, pinchGlow);
    }
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kWindowW, kWindowH, "Vision Two Hands (UDP Bridge)");
    SetTargetFPS(120);
    SetWindowMinSize(960, 620);

    Camera3D cam{};
    cam.position = {0.0f, 13.0f, 22.0f};
    cam.target = {0.0f, 4.5f, 0.0f};
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.fovy = 48.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    UdpBridgeReceiver receiver;
    const bool receiverOk = receiver.Start(static_cast<uint16_t>(kUdpPort));

    TrackingInput tracking{};
    float lastPacketWallClock = -100.0f;
    HandPose leftHand{};
    HandPose rightHand{};
    Vector3 mover = {0.0f, 0.7f, 0.0f};

    while (!WindowShouldClose()) {
        const float dt = std::max(1.0e-4f, GetFrameTime());
        const float now = static_cast<float>(GetTime());

        int packetsRead = 0;
        if (receiver.Poll(tracking, packetsRead)) {
            lastPacketWallClock = now;
        }
        const bool linkLive = receiver.ready() && ((now - lastPacketWallClock) < kLinkTimeout);

        const bool leftLive = linkLive && tracking.leftValid;
        const bool rightLive = linkLive && tracking.rightValid;

        leftHand.valid = leftLive;
        rightHand.valid = rightLive;
        leftHand.pinched = leftLive && tracking.leftPinch;
        rightHand.pinched = rightLive && tracking.rightPinch;

        const Vector3 leftTarget = HandSpaceToWorld(tracking.leftX, tracking.leftY, false);
        const Vector3 rightTarget = HandSpaceToWorld(tracking.rightX, tracking.rightY, true);

        const float handBlend = 1.0f - std::exp(-kHandSmooth * dt);
        leftHand.pos = Vector3Lerp(leftHand.pos, leftTarget, handBlend);
        rightHand.pos = Vector3Lerp(rightHand.pos, rightTarget, handBlend);

        Vector3 moverTarget = mover;
        bool haveMoverTarget = false;
        if (leftLive && rightLive) {
            moverTarget = Vector3Scale(Vector3Add(leftHand.pos, rightHand.pos), 0.5f);
            moverTarget.y = std::max(0.8f, moverTarget.y - 3.2f);
            moverTarget.z = 0.0f;
            haveMoverTarget = true;
        } else if (leftLive) {
            moverTarget = leftHand.pos;
            moverTarget.y = std::max(0.8f, moverTarget.y - 3.2f);
            moverTarget.z = 0.0f;
            haveMoverTarget = true;
        } else if (rightLive) {
            moverTarget = rightHand.pos;
            moverTarget.y = std::max(0.8f, moverTarget.y - 3.2f);
            moverTarget.z = 0.0f;
            haveMoverTarget = true;
        }

        if (haveMoverTarget) {
            float speedBoost = 1.0f;
            if (leftHand.pinched || rightHand.pinched) speedBoost = 1.85f;
            const float moverBlend = 1.0f - std::exp(-(kCursorSmooth * speedBoost) * dt);
            mover = Vector3Lerp(mover, moverTarget, moverBlend);
        }

        cam.target = Vector3Lerp(cam.target, {mover.x, 3.2f, 0.0f}, 0.065f);
        cam.position = Vector3Lerp(cam.position, {mover.x * 0.4f, 13.0f, 22.0f}, 0.045f);

        BeginDrawing();
        ClearBackground(Color{16, 18, 24, 255});
        BeginMode3D(cam);

        DrawPlane({0.0f, 0.0f, 0.0f}, {42.0f, 24.0f}, Color{38, 46, 64, 255});
        for (int i = -10; i <= 10; ++i) {
            const Color c = (i % 5 == 0) ? Color{58, 74, 102, 180} : Color{44, 56, 78, 140};
            DrawLine3D({static_cast<float>(i) * 2.0f, 0.001f, -10.0f}, {static_cast<float>(i) * 2.0f, 0.001f, 10.0f}, c);
        }

        DrawCube({mover.x, mover.y, mover.z}, 1.1f, 1.1f, 1.1f, Color{102, 250, 190, 255});
        DrawCubeWires({mover.x, mover.y, mover.z}, 1.1f, 1.1f, 1.1f, Color{20, 28, 30, 255});

        DrawHandAvatar(leftHand, false);
        DrawHandAvatar(rightHand, true);

        DrawCylinder({-12.0f, 2.5f, 0.0f}, 0.3f, 0.3f, 5.0f, 18, Color{90, 104, 136, 255});
        DrawCylinder({12.0f, 2.5f, 0.0f}, 0.3f, 0.3f, 5.0f, 18, Color{90, 104, 136, 255});

        EndMode3D();

        DrawRectangle(0, 0, GetScreenWidth(), 82, Fade(BLACK, 0.36f));
        DrawText("Vision two-hand bridge scene", 16, 14, 26, Color{222, 231, 248, 255});
        DrawText(
            "Move your hands in webcam view. Left/right hand positions drive two avatars and mover cube.",
            16,
            47,
            18,
            Color{180, 198, 226, 255}
        );

        DrawText(
            TextFormat(
                "%s udp:%d   left:%s (%.2f,%.2f) pinch:%s   right:%s (%.2f,%.2f) pinch:%s",
                receiverOk ? "bridge:on" : "bridge:failed",
                kUdpPort,
                leftLive ? "ok" : "none",
                tracking.leftX,
                tracking.leftY,
                leftHand.pinched ? "on" : "off",
                rightLive ? "ok" : "none",
                tracking.rightX,
                tracking.rightY,
                rightHand.pinched ? "on" : "off"
            ),
            16,
            GetScreenHeight() - 34,
            18,
            Color{208, 226, 248, 255}
        );

        DrawFPS(GetScreenWidth() - 100, 14);
        EndDrawing();
    }

    receiver.Close();
    CloseWindow();
    return 0;
}
