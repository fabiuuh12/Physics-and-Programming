#pragma once

#include "studio.h"
#include "raymath.h"
#include "rlgl.h"

// Presentation helpers shared by the quasar and illustrative wormhole scenes.
namespace cosmic {
constexpr Color text{233,240,249,255};
constexpr Color muted{147,166,188,255};
constexpr Color motion{108,223,244,255};
constexpr Color warm{255,193,119,255};
constexpr Color purple{195,173,255,255};
constexpr int viewWidth=1074, viewHeight=654;
constexpr Rectangle view{16,130,float(viewWidth),float(viewHeight)};
constexpr float panelX=1110;

inline void button(Rectangle r,const char* label,bool active,Color color=motion) {
    DrawRectangleRounded(r,0.16f,8,active ? Color{34,49,68,255} : Color{20,30,45,255});
    DrawRectangleRoundedLinesEx(r,0.16f,8,1,active ? color : Color{48,64,83,255});
    studio::text(label,r.x+12,r.y+11,16,active ? color : muted);
}
inline bool front(Vector3 p,Camera3D camera) {
    return Vector3DotProduct(Vector3Subtract(p,camera.position),Vector3Subtract(camera.target,camera.position))>0;
}
struct MotionArrow { Vector3 origin,tip; Color color; };
inline std::vector<MotionArrow>& arrows() { static std::vector<MotionArrow> list; return list; }
inline void arrow(Vector3 origin,Vector3 velocity,float scale,Color color) {
    float speed=Vector3Length(velocity);
    if (speed<0.00001f) return;
    float length=std::min(speed*scale,2.6f);
    if (length<0.06f) return;
    Vector3 direction=Vector3Scale(velocity,1/speed);
    Vector3 tip=Vector3Add(origin,Vector3Scale(direction,length));
    arrows().push_back({origin,tip,color});
    float head=std::min(0.23f,length*0.3f);
    Vector3 base=Vector3Subtract(tip,Vector3Scale(direction,head));
    DrawCylinderEx(origin,base,0.022f,0.022f,7,color);
    DrawCylinderEx(base,tip,head*0.36f,0,9,color);
}
inline void drawMotion(Camera3D camera) {
    for (const auto& arrow:arrows()) {
        if (!front(arrow.origin,camera) || !front(arrow.tip,camera)) continue;
        Vector2 a=GetWorldToScreenEx(arrow.origin,camera,viewWidth,viewHeight);
        Vector2 b=GetWorldToScreenEx(arrow.tip,camera,viewWidth,viewHeight);
        float length=Vector2Distance(a,b);
        if (length<4) continue;
        Vector2 d=Vector2Scale(Vector2Subtract(b,a),1/length), side{-d.y,d.x};
        float head=std::min(9.0f,length*0.4f);
        Vector2 base=Vector2Subtract(b,Vector2Scale(d,head));
        DrawLineEx(a,base,2,Fade(arrow.color,0.9f));
        DrawTriangle(b,Vector2Subtract(base,Vector2Scale(side,head*0.45f)),
                       Vector2Add(base,Vector2Scale(side,head*0.45f)),arrow.color);
    }
}
inline void label(Camera3D camera,Vector3 point,const char* value,Color color,Vector2 offset={12,-12}) {
    if (!front(point,camera)) return;
    Vector2 p=GetWorldToScreenEx(point,camera,viewWidth,viewHeight);
    p=Vector2Add(p,offset);
    float width=studio::textWidth(value,14)+16;
    if (p.x<8 || p.x+width>viewWidth-8 || p.y<8 || p.y>viewHeight-30) return;
    studio::panel({p.x-6,p.y-4,width,25},Color{10,18,30,235});
    studio::text(value,p.x,p.y,14,color);
}
inline void beginScene(RenderTexture2D target) {
    arrows().clear();
    BeginTextureMode(target);
    ClearBackground(Color{6,10,19,255});
    DrawRectangleGradientV(0,0,viewWidth,viewHeight,Color{11,18,31,255},Color{3,6,13,255});
}
inline void compose(RenderTexture2D target,const char* category,const char* title,const char* subtitle) {
    EndTextureMode();
    BeginDrawing();
    ClearBackground(Color{7,12,22,255});
    DrawTextureRec(target.texture,{0,0,float(viewWidth),-float(viewHeight)},{view.x,view.y},WHITE);
    studio::text(category,28,24,13,motion);
    studio::text(title,27,46,32,text);
    studio::text(subtitle,29,94,17,muted);
    studio::panel({panelX,20,306,796},Color{12,21,34,250});
}
inline void help(const char* value) {
    studio::panel({24,835,1392,43},Color{14,23,37,255});
    studio::fit(value,{40,847,1360,24},15,muted);
}
inline void bar(float x,float y,float width,float amount,Color color) {
    DrawRectangleRounded({x,y,width,7},0.8f,8,Color{37,53,72,255});
    if (amount>0) DrawRectangleRounded({x,y,width*std::clamp(amount,0.0f,1.0f),7},0.8f,8,color);
}
inline void glow(Camera3D camera,Vector3 point,Color color,float alpha,float radius) {
    if (!front(point,camera)) return;
    Vector2 p=GetWorldToScreenEx(point,camera,viewWidth,viewHeight);
    if (p.x < -radius || p.x > viewWidth+radius || p.y < -radius || p.y > viewHeight+radius) return;
    DrawCircleGradient(int(p.x),int(p.y),radius,Fade(color,alpha),Fade(color,0));
}
} // namespace cosmic
