#pragma once
#include "model.h"
#include "../../../vision/hand_tracking_scene_shared.h"
namespace plant {
struct Pointer {Vector2 p{};bool active=false,down=false;Grab grab;};
class Controls {
    astro_hand::UdpHandReceiver receiver;
    astro_hand::TrackingPacket packet{};
    float lastPacket=-100;
public:
    Pointer pointers[3];bool connected=false,receiverOk=false;
    Controls() {receiverOk=receiver.Start(astro_hand::kUdpPort);}
    void update(Camera3D camera,Model& model,float now) {
        int read=0;if(receiver.Poll(packet,read))lastPacket=now;
        connected=receiverOk && now-lastPacket<.5f;
        const astro_hand::TrackedHandPacket* hands[]={&packet.left,&packet.right};
        for(int i=0;i<2;++i) {
            auto& p=pointers[i];const auto& hand=*hands[i];
            bool valid=connected&&hand.valid&&hand.score>.45f;
            Vector3 tip=Vector3Scale(Vector3Add(hand.landmarks[4],hand.landmarks[8]),.5f);
            valid=valid&&std::isfinite(tip.x)&&std::isfinite(tip.y)&&tip.x>=0&&tip.x<=1&&tip.y>=0&&tip.y<=1;
            Vector2 target{tip.x*GetScreenWidth(),tip.y*GetScreenHeight()};
            p.p=p.active?Vector2Lerp(p.p,target,.45f):target;p.active=valid;p.down=valid&&hand.pinched;
        }
        auto& mouse=pointers[2];mouse.active=true;mouse.p=GetMousePosition();mouse.down=IsMouseButtonDown(MOUSE_BUTTON_LEFT);
        Vector2 low=GetWorldToScreen({-3,1,6.15f},camera),high=GetWorldToScreen({-3,2.6f,6.15f},camera);
        Vector2 handle=Vector2Lerp(high,low,model.rods),dial=GetWorldToScreen({-1,1.65f,6.1f},camera),stop=GetWorldToScreen({1,1.5f,6.25f},camera);
        for(auto& p:pointers) {
            int hover=Vector2Distance(p.p,handle)<30?0:Vector2Distance(p.p,dial)<34?1:Vector2Distance(p.p,stop)<27?2:-1;
            for(auto& other:pointers)if(&other!=&p && other.grab.held==hover)hover=-1;
            bool pressed=p.grab.update(p.active,p.down,hover);
            if(pressed&&p.grab.held==2)model.shutdown();
            if(p.grab.held==0&&!model.scrammed) {
                Vector2 axis=Vector2Subtract(low,high);
                model.rods=std::clamp(Vector2DotProduct(Vector2Subtract(p.p,high),axis)/std::max(1.0f,Vector2LengthSqr(axis)),0.0f,1.0f);
            }
            if(p.grab.held==1) {
                Ray ray=GetScreenToWorldRay(p.p,camera);
                if(std::abs(ray.direction.z)>.001f) {
                    float t=(6.10f-ray.position.z)/ray.direction.z;
                    Vector3 point=Vector3Add(ray.position,Vector3Scale(ray.direction,t));
                    if(t>0) model.load=std::clamp((std::atan2(point.x+1,point.y-1.65f)*RAD2DEG+135)/270,0.0f,1.0f);
                }
            }
        }
    }
    bool mouseCaptured() const {return pointers[2].grab.held>=0;}
    void drawCursors() const {
        for(int i=0;i<2;++i)if(pointers[i].active) {
            auto& p=pointers[i];Color c=i?Color{100,217,238,255}:Color{240,186,109,255};
            DrawCircleLinesV(p.p,p.down?10:16,c);DrawCircleV(p.p,3,c);
            if(p.grab.held>=0)DrawCircleLinesV(p.p,21,Fade(c,.6f));
        }
    }
};
}
