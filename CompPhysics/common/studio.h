#pragma once

#include "raylib.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

// Small, explicit presentation helpers. No draw-call macros or hidden input remapping.
namespace studio {
inline Font& fontStorage() { static Font font{}; return font; }
inline Font font() {
    Font& f=fontStorage();
    if (f.texture.id) return f;
    // Resolve from the executable first so launching from another cwd works.
    const char* resource="assets/fonts/Inter-Regular.ttf";
    std::vector<std::filesystem::path> roots={
        std::filesystem::path(GetApplicationDirectory())/"..",
        std::filesystem::path(__FILE__).parent_path().parent_path(),
        std::filesystem::path("CompPhysics"), std::filesystem::path(".")};
    for (const auto& root : roots) {
        auto path=root/resource;
        if (FileExists(path.string().c_str())) {
            f=LoadFontEx(path.string().c_str(),48,nullptr,0);
            if (f.texture.id) { SetTextureFilter(f.texture,TEXTURE_FILTER_BILINEAR); return f; }
        }
    }
    f=GetFontDefault();
    return f;
}
inline void unload() {
    Font& f=fontStorage();
    if (f.texture.id && f.texture.id!=GetFontDefault().texture.id) UnloadFont(f);
    f={};
}
inline float textWidth(const char* value, float size) { return MeasureTextEx(font(),value,size,0.3f).x; }
inline void text(const char* value, float x, float y, float size, Color color) {
    DrawTextEx(font(),value,{x,y},size,0.3f,color);
}
inline void fit(const char* value, Rectangle bounds, float size, Color color) {
    float width=textWidth(value,size);
    if (width>bounds.width) size*=bounds.width/width;
    text(value,bounds.x,bounds.y,size,color);
}
inline void panel(Rectangle bounds, Color color) {
    DrawRectangleRounded({bounds.x+1,bounds.y+4,bounds.width,bounds.height},0.08f,10,Color{0,0,0,35});
    DrawRectangleRounded(bounds,0.08f,10,color);
}
inline bool over(Rectangle bounds) { return CheckCollisionPointRec(GetMousePosition(),bounds); }
inline bool clicked(Rectangle bounds) { return over(bounds) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT); }
inline void button(Rectangle bounds, const char* label, bool selected, Color accent) {
    Color bg=selected ? accent : (over(bounds) ? Color{221,227,223,255} : Color{232,235,230,255});
    DrawRectangleRounded(bounds,0.18f,8,bg);
    text(label,bounds.x+13,bounds.y+(bounds.height-18)/2,18,selected ? WHITE : Color{38,54,53,255});
}

// Pan in the camera's screen plane; leaves every simulation's existing orbit
// limits and model-specific key bindings intact. Shift + left drag also works
// on trackpads that have no middle button.
inline bool panGesture() {
    return IsMouseButtonDown(MOUSE_BUTTON_MIDDLE) ||
           (IsKeyDown(KEY_LEFT_SHIFT) && IsMouseButtonDown(MOUSE_BUTTON_LEFT));
}
inline void pan(Camera3D* c, float yaw, float pitch, float distance) {
    if (!panGesture()) return;
    Vector2 d=GetMouseDelta();
    float scale=2*distance*std::tan(c->fovy*PI/360)/std::max(1,GetScreenHeight());
    Vector3 right{std::sin(yaw),0,-std::cos(yaw)};
    Vector3 up{-std::sin(pitch)*std::cos(yaw),std::cos(pitch),-std::sin(pitch)*std::sin(yaw)};
    c->target.x+=(-d.x*right.x+d.y*up.x)*scale;
    c->target.y+=d.y*up.y*scale;
    c->target.z+=(-d.x*right.z+d.y*up.z)*scale;
}
inline void orbit(Camera3D& c,float& yaw,float& pitch,float& distance) {
    pan(&c,yaw,pitch,distance);
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !panGesture()) {
        Vector2 d=GetMouseDelta(); yaw-=d.x*0.004f;
        pitch=std::clamp(pitch+d.y*0.004f,-1.4f,1.4f);
    }
    distance=std::clamp(distance*std::exp(-GetMouseWheelMove()*0.07f),2.0f,60.0f);
    c.position={c.target.x+distance*std::cos(pitch)*std::cos(yaw),
                c.target.y+distance*std::sin(pitch),
                c.target.z+distance*std::cos(pitch)*std::sin(yaw)};
}

enum class Style { Instrument, Observatory, Field, Quantum };
inline Style& currentStyle() { static Style style=Style::Instrument; return style; }
inline Color accent() {
    switch (currentStyle()) {
        case Style::Observatory: return {231,190,128,255};
        case Style::Field: return {121,214,177,255};
        case Style::Quantum: return {192,173,246,255};
        default: return {134,198,224,255};
    }
}
inline void title(const char* value, Style style) {
    currentStyle()=style;
    float w=std::min(float(GetScreenWidth()-48),textWidth(value,25)+40);
    if (style==Style::Observatory) {
        text("OBSERVATORY",26,18,12,accent());
        fit(value,{26,39,float(GetScreenWidth()-190),32},25,Color{239,239,234,255});
        DrawLine(26,79,130,79,accent());
    } else if (style==Style::Field) {
        panel({20,20,w,61},Color{21,39,37,242});
        DrawCircle(39,49,4,accent());
        fit(value,{53,34,w-65,33},24,Color{233,244,238,255});
    } else if (style==Style::Quantum) {
        DrawRectangle(24,24,3,49,accent());
        text("STATE / EXPERIMENT",40,20,12,accent());
        fit(value,{40,39,float(GetScreenWidth()-200),35},25,Color{241,235,252,255});
    } else {
        panel({20,20,w,59},Color{230,233,227,250});
        fit(value,{36,33,w-32,35},25,Color{35,49,53,255});
    }
}
inline void readout(const char* value) {
    float w=std::min(float(GetScreenWidth()-48),textWidth(value,18)+32);
    panel({24,91,w,39},Color{20,29,40,235});
    fit(value,{40,100,w-32,24},18,accent());
}
inline void note(const char* value) {
    fit(value,{26,140,float(GetScreenWidth()-52),25},16,Color{169,184,196,255});
}
inline void help(const char* value) {
    std::string controls=value;
    if (controls.find("orbit")!=std::string::npos || controls.find("Orbit")!=std::string::npos)
        controls+=" | Shift+drag / middle: pan";
    float w=std::min(float(GetScreenWidth()-48),textWidth(controls.c_str(),15)+32);
    panel({24,float(GetScreenHeight()-50),w,32},Color{20,29,40,235});
    fit(controls.c_str(),{40,float(GetScreenHeight()-44),w-32,22},15,Color{192,204,211,255});
}
inline void fps() {
    text(TextFormat("%d fps",GetFPS()),float(GetScreenWidth()-84),23,13,Color{137,156,171,255});
}

// Opt-in only: lets the validation runner exercise the actual graphics loop.
// Normal interactive sessions never take screenshots or close automatically.
inline bool smokeFrame(const char* source) {
    const char* option=std::getenv("COMPPHYSICS_SMOKE_FRAMES");
    if (!option) return false;
    int limit=std::clamp(std::atoi(option),1,1200);
    static int frames=0;
    if (++frames<limit) return false;
    std::string file="/tmp/"+std::filesystem::path(source).stem().string()+"_preview.png";
    Image capture=LoadImageFromScreen();
    bool saved=ExportImage(capture,file.c_str());
    UnloadImage(capture);
    if (!saved) TraceLog(LOG_ERROR,"Preview export failed: %s",file.c_str());
    return true;
}

// Graphs use a fixed physical vertical scale, clipped to the plot bounds.
// The caller determines sample cadence and supplies axis units in the title.
template<class Container>
inline void plot(Rectangle r, const char* label, const Container& values,
                 float lo, float hi, Color color) {
    panel(r,Color{17,26,38,244});
    text(label,r.x+16,r.y+12,16,Color{219,229,237,255});
    Rectangle p{r.x+48,r.y+43,r.width-64,r.height-67};
    for (int i=0;i<=4;++i) {
        float y=p.y+p.height*i/4;
        DrawLineEx({p.x,y},{p.x+p.width,y},1,Color{49,64,80,150});
        text(TextFormat("%.2f",hi-(hi-lo)*i/4),r.x+7,y-7,11,Color{141,162,177,255});
    }
    if (values.size()<2 || hi<=lo) return;
    BeginScissorMode(int(p.x),int(p.y),int(p.width),int(p.height));
    for (size_t i=1;i<values.size();++i) {
        float x0=p.x+p.width*(i-1)/(values.size()-1), x1=p.x+p.width*i/(values.size()-1);
        float y0=p.y+p.height*(hi-values[i-1])/(hi-lo), y1=p.y+p.height*(hi-values[i])/(hi-lo);
        DrawLineEx({x0,y0},{x1,y1},2,color);
    }
    EndScissorMode();
}
} // namespace studio
