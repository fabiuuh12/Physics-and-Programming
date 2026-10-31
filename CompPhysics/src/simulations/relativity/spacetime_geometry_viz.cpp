#include "../../common/studio.h"
#include "../../common/spacetime_geometry.h"
#include "raymath.h"
#include "rlgl.h"
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace {
constexpr int W=1440,H=900;
constexpr Color white{228,239,247,255},muted{140,163,181,255},cyan{92,218,231,255},gold{255,200,116,255};
constexpr Color border{45,65,81,255};
const char* names[]={"Flat space","Planet","Star","Black hole","Wormhole"};
const char* captions[]={"An undisturbed reference geometry.","Earth-like compactness. Spatial height magnified.","Sun-like compactness. Spatial height magnified.","Schwarzschild geometry around a nonrotating mass.","Ellis geometry: two spatial regions, one throat."};
struct Settings{
    int preset=3;float mass=1,warp=1,probe=3.5f,impact=3;
    bool lattice=false,surface=true,rays=true,cones=true,clocks=true,autoOrbit=false,paused=false,fan=true;
    double rs()const{return preset==0||preset==4?0:(preset==1?1.39e-9:(preset==2?4.24e-6:1.))*mass;}
    double radius()const{return preset==0?0:(preset==3?rs():1.);}
    double throat()const{return mass;}
    double gain()const{return (preset==1||preset==2)?.8/std::sqrt(rs()/mass)*warp:warp;}
    double inner()const{return preset==4?throat():radius();}
    double height(double r,int branch=1)const{
        if(preset==4)return branch*throat()*std::acosh(std::max(1.,r/throat()))*warp;
        return spacetime::height(r,rs())*gain();
    }
    Vector3 point(double x,double z,int branch=1,float lift=0)const{
        return {float(x),float(height(std::hypot(x,z),branch))+lift,float(z)};
    }
    double rate()const{
        if(preset==4||preset==0)return 1;
        return spacetime::staticRate(probe,rs(),preset==3?0:1.);
    }
};
Vector2 mouse{};
bool over(Rectangle r){return CheckCollisionPointRec(mouse,r);}
bool click(Rectangle r){return over(r)&&IsMouseButtonPressed(MOUSE_BUTTON_LEFT);}
void text(const char* s,float x,float y,float size=16,Color color=white){studio::text(s,x,y,size,color);}
void panel(Rectangle r){DrawRectangleRounded(r,.08f,8,{9,19,29,238});DrawRectangleRoundedLinesEx(r,.08f,8,1,border);}
void button(Rectangle r,const char* label,bool selected){
    DrawRectangleRounded(r,.18f,8,selected?Color{31,77,87,255}:(over(r)?Color{29,45,58,255}:Color{13,27,39,245}));
    DrawRectangleRoundedLinesEx(r,.18f,8,1,selected?cyan:border);
    text(label,r.x+12,r.y+(r.height-15)/2,15,selected?white:muted);
}
void wrap(const char* s,float x,float y,float width,float size,Color color=muted){
    std::istringstream words(s);std::string word,row;
    while(words>>word){std::string next=row.empty()?word:row+" "+word;
        if(studio::textWidth(next.c_str(),size)>width&&!row.empty()){text(row.c_str(),x,y,size,color);y+=size*1.45f;row=word;}else row=next;
    }text(row.c_str(),x,y,size,color);
}
// Pointer capture is shared by sliders so a drag cannot accidentally orbit the camera.
int activeSlider=-1;
bool slider(int id,float y,const char* label,float& value,float lo,float hi,const char* formatted,bool render=false){
    Rectangle hit{1123,y+22,271,28};
    float before=value;
    if(!render){
        if(click(hit))activeSlider=id;
        if(activeSlider==id&&IsMouseButtonDown(MOUSE_BUTTON_LEFT))value=lo+(hi-lo)*std::clamp((mouse.x-1125)/265,0.f,1.f);
        return before!=value;
    }
    text(label,1123,y,12,muted);float width=studio::textWidth(formatted,13);text(formatted,1393-width,y,13);
    DrawRectangle(1125,int(y+34),265,3,{51,72,88,255});
    DrawRectangle(1125,int(y+34),int(265*(value-lo)/(hi-lo)),3,cyan);
    DrawCircle(int(1125+265*(value-lo)/(hi-lo)),int(y+35),5,white);
    return before!=value;
}
Color fieldColor(double r,const Settings& s,bool minor=false){
    float proximity=float(std::exp(-r/4));
    Color col=ColorLerp(s.preset==4?Color{92,91,160,255}:Color{37,113,139,255},Color{112,238,227,255},proximity);
    col.a=minor?75:180;return col;
}
void line3(Vector3 a,Vector3 b,Color c){DrawLine3D(a,b,c);}
void ring(const Settings& s,float radius,int branch,Color color,float lift=0){
    Vector3 prev=s.point(radius,0,branch,lift);
    for(int i=1;i<=180;i++){float a=i*2*PI/180;Vector3 p=s.point(radius*cos(a),radius*sin(a),branch,lift);line3(prev,p,color);prev=p;}
}
void drawSurface(const Settings& s,int branch){
    // Polar tessellation removes the central domain cleanly, including the throat.
    constexpr int radial=58,angular=112;
    float inner=std::max(.025,double(s.inner()));
    for(int j=0;j<radial;j++){
        float u=float(j)/radial,v=float(j+1)/radial;
        float r0=inner+(16-inner)*u*u,r1=inner+(16-inner)*v*v;
        for(int i=0;i<angular;i++){
            float a=i*2*PI/angular,b=(i+1)*2*PI/angular;
            Vector3 p=s.point(r0*cos(a),r0*sin(a),branch),q=s.point(r1*cos(a),r1*sin(a),branch);
            Vector3 r=s.point(r1*cos(b),r1*sin(b),branch),t=s.point(r0*cos(b),r0*sin(b),branch);
            Color c=fieldColor((r0+r1)/2,s);c.a=branch>0?28:16;
            DrawTriangle3D(p,q,r,c);DrawTriangle3D(p,r,t,c);
        }
    }
}
void drawGrid(const Settings& s){
    // Translucent reference geometry must not occlude the second wormhole sheet.
    rlDisableDepthMask();
    if(s.surface){rlDisableBackfaceCulling();if(s.preset==4)drawSurface(s,-1);drawSurface(s,1);rlEnableBackfaceCulling();}
    int branches=s.preset==4?2:1;
    for(int branchIndex=0;branchIndex<branches;branchIndex++){
        int branch=branchIndex==0?1:-1;
        int layers=s.lattice?3:1;
        for(int layer=0;layer<layers;layer++){
            float offset=s.lattice?(layer-1)*2.3f:0;
            for(int axis=0;axis<2;axis++)for(int j=-28;j<=28;j++){
                float fixed=j*.5f;
                for(int k=0;k<144;k++){
                    float t0=-14+k*28.f/144,t1=-14+(k+1)*28.f/144;
                    double x0=axis==0?fixed:t0,z0=axis==0?t0:fixed;
                    double x1=axis==0?fixed:t1,z1=axis==0?t1:fixed;
                    double r0=std::hypot(x0,z0),r1=std::hypot(x1,z1);
                    if(std::min(r0,r1)<s.inner()*1.002||std::max(r0,r1)>16)continue;
                    Vector3 a=s.point(x0,z0,branch,offset+.025f),b=s.point(x1,z1,branch,offset+.025f);
                    Color c=fieldColor((r0+r1)/2,s,j%2!=0);if(layer!=1&&s.lattice)c.a=45;
                    line3(a,b,c);
                }
            }
        }
        if(s.lattice)for(int x=-12;x<=12;x+=2)for(int z=-12;z<=12;z+=2){
            if(std::hypot(x,z)<s.inner()||std::hypot(x,z)>15)continue;
            line3(s.point(x,z,branch,-2.3f),s.point(x,z,branch,2.3f),{43,140,162,45});
        }
    }
    if(s.inner()>0){ring(s,float(s.inner()*1.006),1,cyan,.03f);if(s.preset==4)ring(s,float(s.inner()*1.006),-1,cyan,.03f);}
    if(s.preset==3)ring(s,float(s.rs()*1.5),1,{242,178,102,180},.02f);
    rlEnableDepthMask();
}
void glow(Camera3D camera,Vector3 p,float radius,Color color,float alpha){
    if(Vector3DotProduct(Vector3Subtract(p,camera.position),Vector3Subtract(camera.target,camera.position))<=0)return;
    Vector2 a=GetWorldToScreenEx(p,camera,W,H);
    DrawCircleGradient(int(a.x),int(a.y),radius,Fade(color,alpha),Fade(color,0));
}
void label(Camera3D camera,Vector3 p,const char* value,Color color){
    if(Vector3DotProduct(Vector3Subtract(p,camera.position),Vector3Subtract(camera.target,camera.position))<=0)return;
    Vector2 a=GetWorldToScreenEx(p,camera,W,H);float w=studio::textWidth(value,13)+20;
    if(a.x<30||a.x+w>1090||a.y<140||a.y>700)return;
    DrawLineEx(a,{a.x+18,a.y-22},1,Fade(color,.7));
    DrawRectangleRounded({a.x+18,a.y-39,w,27},.1f,6,{8,18,29,220});text(value,a.x+28,a.y-33,13,color);
}
void causalInset(const Settings& s){
    Rectangle r{24,548,263,182};panel(r);
    text("LIGHT CONE / radial section",40,563,12,cyan);
    text(s.preset==4?"t / proper radial distance l":"T = v - r    (c = 1)",40,584,11,muted);
    Vector2 origin{150,672};float h=51;
    double outgoing=s.preset==4?1:spacetime::outgoingSlope(s.probe,s.rs());
    Vector2 left{origin.x-h,origin.y-h},right{origin.x+float(outgoing)*h,origin.y-h};
    DrawTriangle(origin,right,left,{74,205,216,48});
    DrawLineEx(origin,left,2,cyan);DrawLineEx(origin,right,2,gold);
    DrawLineEx({origin.x-80,origin.y},{origin.x+86,origin.y},1,border);
    DrawLineEx({origin.x,origin.y+26},{origin.x,origin.y-64},1,border);
    DrawCircleV(origin,3,white);text(s.preset==4?"l":"r",242,665,12,muted);text(s.preset==4?"t":"T",156,603,12,muted);
    text("in",86,608,10,cyan);text("out",right.x+5,right.y-8,10,gold);
    text(s.preset==4?"Ultrastatic throat: no horizon":(s.probe<=s.rs()?"Inside: both futures point inward":"Future-directed radial light"),40,708,11,muted);
}
}

int main(int argc,char** argv){
    Settings s;std::string capture;int captureFrames=3;bool impactExplicit=false;
    try{
        for(int i=1;i<argc;i++){
            std::string arg=argv[i];
            if(arg=="--help"){std::puts("--preset 0..4 --lattice 0|1 --mass .3..2.5 --warp .25..2 --probe .55..10 --impact -6..6 --capture file.png --frames N");return 0;}
            if(i+1>=argc)throw std::runtime_error("Missing argument value");
            std::string value=argv[++i];
            if(arg=="--preset")s.preset=std::stoi(value);
            else if(arg=="--lattice")s.lattice=std::stoi(value)!=0;
            else if(arg=="--mass")s.mass=std::stof(value);
            else if(arg=="--warp")s.warp=std::stof(value);
            else if(arg=="--probe")s.probe=std::stof(value);
            else if(arg=="--impact"){s.impact=std::stof(value);impactExplicit=true;}
            else if(arg=="--capture")capture=value;
            else if(arg=="--frames")captureFrames=std::stoi(value);
            else throw std::runtime_error("Unknown option: "+arg);
        }
        if(s.preset<0||s.preset>4||captureFrames<1||captureFrames>1200||
           !std::isfinite(s.mass)||!std::isfinite(s.warp)||!std::isfinite(s.probe)||!std::isfinite(s.impact)||
           s.mass<.3f||s.mass>2.5f||s.warp<.25f||s.warp>2||s.probe<.55f||s.probe>10||std::abs(s.impact)>6)throw std::runtime_error("Option outside supported range");
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}
    if(s.preset==4&&!impactExplicit)s.impact=.25f;
    bool smoke=std::getenv("COMPPHYSICS_SMOKE_FRAMES")!=nullptr;
    SetConfigFlags(FLAG_MSAA_4X_HINT|FLAG_WINDOW_RESIZABLE|FLAG_VSYNC_HINT|((smoke||!capture.empty())?FLAG_WINDOW_HIDDEN:0));
    InitWindow(W,H,"Spacetime Geometry | Interactive Field Lab");SetWindowMinSize(960,600);SetTargetFPS(60);
    RenderTexture2D canvas=LoadRenderTexture(W,H);SetTextureFilter(canvas.texture,TEXTURE_FILTER_BILINEAR);
    Camera3D camera{};camera.up={0,1,0};camera.fovy=46;camera.projection=CAMERA_PERSPECTIVE;
    float yaw=.85f,pitch=s.preset==4?.30f:.65f,distance=29;
    double time=0,properTime=0;bool dragging=false,dirty=true,showHelp=false;
    std::vector<spacetime::Path> paths;int captured=0,crossed=0,frames=0,result=0;
    std::vector<Vector3> stars;
    for(int i=0;i<650;i++){
        float theta=i*2.399963f,y=1-2*(i+.5f)/650,r=std::sqrt(1-y*y);
        stars.push_back({80*r*std::cos(theta),80*y,80*r*std::sin(theta)});
    }
    while(!WindowShouldClose()){
        float scale=std::min(float(GetScreenWidth())/W,float(GetScreenHeight())/H);
        Vector2 offset{(GetScreenWidth()-W*scale)/2,(GetScreenHeight()-H*scale)/2};
        Vector2 raw=GetMousePosition();mouse={(raw.x-offset.x)/scale,(raw.y-offset.y)/scale};
        if(IsMouseButtonReleased(MOUSE_BUTTON_LEFT)){activeSlider=-1;dragging=false;}
        bool changed=false;
        for(int i=0;i<5;i++)if(click({24+i*147.f,101,137,35})||IsKeyPressed(KEY_ONE+i)){
            s.preset=i;s.mass=1;s.warp=1;s.probe=3.5f;s.impact=i==4?.25f:3.f;pitch=i==4?.30f:.65f;changed=true;
        }
        if(IsKeyPressed(KEY_SPACE)||click({24,763,97,35}))s.paused=!s.paused;
        if(IsKeyPressed(KEY_R)||click({131,763,97,35})){time=properTime=0;}
        if(IsKeyPressed(KEY_G)||click({244,763,130,35}))s.lattice=!s.lattice;
        if(IsKeyPressed(KEY_S)||click({384,763,130,35}))s.surface=!s.surface;
        if(IsKeyPressed(KEY_L)||click({524,763,130,35}))s.rays=!s.rays;
        if(IsKeyPressed(KEY_C)||click({664,763,130,35}))s.cones=!s.cones;
        if(IsKeyPressed(KEY_K)||click({804,763,130,35}))s.clocks=!s.clocks;
        if(IsKeyPressed(KEY_F)||click({944,763,130,35})){s.fan=!s.fan;dirty=true;}
        if(IsKeyPressed(KEY_A)||click({1123,679,271,34}))s.autoOrbit=!s.autoOrbit;
        if(IsKeyPressed(KEY_H))showHelp=!showHelp;
        if(IsKeyPressed(KEY_O)||click({1123,723,271,34})){yaw=.85f;pitch=s.preset==4?.30f:.65f;distance=29;}
        changed|=slider(0,204,s.preset==4?"THROAT RADIUS":"MASS MULTIPLIER",s.mass,.3f,2.5f,TextFormat("%.2fx",s.mass));
        slider(1,270,"DISPLAY HEIGHT",s.warp,.25f,2.f,TextFormat("%.2fx",s.warp));
        bool probeChanged=slider(2,336,"PROBE RADIUS",s.probe,.55f,10.f,TextFormat("%.2f",s.probe));
        bool impactChanged=slider(3,402,"BEAM OFFSET",s.impact,-6.f,6.f,TextFormat("%.2f",s.impact));
        if(s.preset!=3&&s.preset!=4)s.probe=std::max(s.probe,1.01f);
        if(s.preset==4)s.probe=std::max(s.probe,float(s.throat()*1.01));
        if(changed||probeChanged){time=properTime=0;}
        dirty|=changed||impactChanged;
        if(dirty){
            paths.clear();captured=crossed=0;
            int count=s.fan?9:1;
            for(int i=0;i<count;i++){
                double b=s.impact+(count==1?0:(i-4)*.42);
                paths.push_back(s.preset==4?spacetime::ellisPath(b,s.throat()):spacetime::schwarzschildPath(b,s.rs(),s.radius()));
                captured+=paths.back().captured;crossed+=paths.back().crossed;
            }dirty=false;
        }
        bool ui=over({1107,153,309,615})||over({0,0,W,147})||over({0,742,W,158})||(s.cones&&over({24,548,263,182}));
        if(!ui&&activeSlider<0&&IsMouseButtonPressed(MOUSE_BUTTON_LEFT)){dragging=true;s.autoOrbit=false;}
        if(dragging&&activeSlider<0){Vector2 d=GetMouseDelta();yaw-=d.x*.004f/scale;pitch=std::clamp(pitch+d.y*.004f/scale,.08f,1.48f);}
        if(!ui)distance=std::clamp(distance*std::exp(-GetMouseWheelMove()*.07f),9.f,65.f);
        if(!s.paused){
            double dt=capture.empty()?std::min(GetFrameTime(),.1f):1./60;time+=dt;
            if(std::isfinite(s.rate()))properTime+=dt*s.rate();
            if(s.autoOrbit)yaw+=float(dt)*.04f;
        }
        float centerY=s.preset==4?0:float(s.height(4))*.55f;
        camera.target={0,centerY,0};
        camera.position={distance*std::cos(pitch)*std::cos(yaw),centerY+distance*std::sin(pitch),distance*std::cos(pitch)*std::sin(yaw)};
        BeginTextureMode(canvas);ClearBackground({4,9,17,255});
        DrawRectangleGradientV(0,0,W,H,{9,17,29,255},{3,8,15,255});
        BeginMode3D(camera);
        for(size_t i=0;i<stars.size();i++)DrawPoint3D(stars[i],Fade(i%3==0?gold:cyan,.16f));
        drawGrid(s);
        Vector3 center=s.point(0,0);
        if(s.preset==1||s.preset==2){
            center.y=float(s.height(1))+.38f;
            DrawSphereEx(center,1.f,32,48,s.preset==1?Color{46,117,157,255}:Color{253,204,124,255});
            if(s.preset==1){
                // Latitude lines emphasize the globe without texture assets.
                DrawSphereWires(center,1.008f,10,20,{118,203,198,100});
            }
        }
        if(s.preset==3){
            center.y+=.12f;
            DrawSphereEx(center,float(s.rs()*.96),32,48,{1,2,5,255});
            DrawSphereWires(center,float(s.rs()*1.002),16,32,{73,134,158,70});
        }
        if(s.rays)for(size_t i=0;i<paths.size();i++){
            const auto& p=paths[i].points;Color color=i==paths.size()/2?gold:Color{109,209,228,255};
            for(size_t j=1;j<p.size();j++){
                Vector3 a=s.point(p[j-1].x,p[j-1].z,int(p[j-1].branch),.055f),b=s.point(p[j].x,p[j].z,int(p[j].branch),.055f);
                if(std::abs(a.x)>16||std::abs(a.z)>16||std::abs(b.x)>16||std::abs(b.z)>16)continue;
                line3(a,b,Fade(color,i==paths.size()/2?.96f:.55f));
            }
            if(!p.empty()){
                size_t index=size_t(std::fmod(time*.12+i*.07,1.)*(p.size()-1));const auto& point=p[index];
                Vector3 dot=s.point(point.x,point.z,int(point.branch),.08f);DrawSphere(dot,.055f,color);
            }
        }
        Vector3 probe=s.point(s.probe,0,1,.12f);
        if(s.clocks){
            DrawCylinderEx(probe,Vector3Add(probe,{0,.7f,0}),.025f,.025f,6,white);
            DrawSphere(Vector3Add(probe,{0,.8f,0}),.10f,gold);
            if(s.probe>s.inner())ring(s,s.probe,1,{222,185,124,100},.025f);
        }
        EndMode3D();
        if(s.preset==2)glow(camera,center,100,gold,.18f);
        if(s.preset==3)glow(camera,center,100,cyan,.05f);
        if(s.clocks)label(camera,Vector3Add(probe,{0,.8f,0}),"CLOCK PROBE",gold);
        if(s.preset==3)label(camera,s.point(s.rs()*1.5,0,1,.1f),"PHOTON SPHERE  r = 1.5 rs",gold);
        if(s.preset==4)label(camera,{float(s.throat()),0,0},"THROAT / TWO SPATIAL REGIONS",cyan);
        // Floating controls keep the grid as the dominant visual surface.
        DrawRectangleGradientV(0,0,W,150,{4,10,18,255},{4,10,18,0});
        text("S P A C E T I M E    /    G E O M E T R Y",26,22,12,cyan);
        text("Bend the grid. Follow the light.",24,45,31);
        for(int i=0;i<5;i++)button({24+i*147.f,101,137,35},names[i],s.preset==i);
        text(captions[s.preset],26,154,14,muted);
        text("SPATIAL EMBEDDING  /  G = c = 1",26,184,11,cyan);
        if(s.cones)causalInset(s);
        panel({1107,153,309,615});
        text("GEOMETRY CONTROLS",1123,174,13,cyan);
        slider(0,204,s.preset==4?"THROAT RADIUS":"MASS MULTIPLIER",s.mass,.3f,2.5f,TextFormat("%.2fx",s.mass),true);
        slider(1,270,"DISPLAY HEIGHT",s.warp,.25f,2.f,TextFormat("%.2fx",s.warp),true);
        slider(2,336,"PROBE RADIUS",s.probe,.55f,10.f,TextFormat("%.2f",s.probe),true);
        slider(3,402,"BEAM OFFSET",s.impact,-6.f,6.f,TextFormat("%.2f",s.impact),true);
        text(s.preset==4?"Throat size sets the length scale.":"rs / reference radius",1123,463,12,muted);
        text(s.preset==4?TextFormat("b = %.3f",s.throat()):TextFormat("%.4g",s.rs()),1123,483,22);
        text("EXTRINSIC HEIGHT MAGNIFICATION",1123,520,11,muted);
        text(TextFormat("%.3g x",s.gain()),1123,541,22);
        wrap(s.preset==4?"Hypothetical ultrastatic Ellis wormhole. Exotic matter required; not an astrophysical prediction.":"Grid height depicts a spatial slice, not time. Height controls change the drawing, never the light paths.",1123,583,271,14);
        button({1123,679,271,34},"Auto orbit",s.autoOrbit);
        button({1123,723,271,34},"Reset camera",false);
        panel({24,748,1050,116});
        button({24,763,97,35},s.paused?"Play":"Pause",!s.paused);button({131,763,97,35},"Reset clocks",false);
        button({244,763,130,35},s.lattice?"3D lattice":"Single plane",s.lattice);
        button({384,763,130,35},"Surface",s.surface);button({524,763,130,35},"Light paths",s.rays);
        button({664,763,130,35},"Light cone",s.cones);button({804,763,130,35},"Clock probe",s.clocks);
        button({944,763,130,35},"Beam fan",s.fan);
        if(s.clocks){
            text("FAR CLOCK",42,815,11,muted);text(TextFormat("%.2f",time),42,835,20);
            text("PROBE CLOCK",208,815,11,muted);
            text(std::isfinite(s.rate())?TextFormat("%.2f",properTime):"No static observer",208,835,std::isfinite(s.rate())?20:15,gold);
            text("d tau / dt",426,815,11,muted);text(std::isfinite(s.rate())?TextFormat("%.10f",s.rate()):"undefined",426,835,18,gold);
        }
        text("RAY OUTCOMES",678,815,11,muted);
        text(TextFormat(s.preset==4?"%d / %d cross throat":"%d / %d intercepted",s.preset==4?crossed:captured,int(paths.size())),678,836,16,cyan);
        text("Guides: null geodesics / dots: affine playback",42,878,12,muted);
        text(TextFormat("%s  |  H controls  |  %d FPS",s.lattice?"Repeated spatial reference layers":"One spatial equatorial slice",GetFPS()),830,878,12,muted);
        if(showHelp){
            panel({350,238,640,362});text("EXPLORE THE GEOMETRY",376,263,20,cyan);
            wrap("Drag the scene to orbit. Scroll to zoom. 1-5 selects a geometry. Space pauses; R resets both clocks. O resets the camera; A toggles orbit. G changes plane/lattice. S surface, L light paths, C causal inset, K clocks, F beam fan. H closes this guide.",376,308,586,18);
            wrap("Mass and probe changes reset clocks. Changing display height leaves the numerical paths and clocks unchanged. The repeated lattice layers are a visual reference, not extra time dimensions. The causal inset has its own labeled time axis.",376,469,586,16,gold);
        }
        EndTextureMode();BeginDrawing();ClearBackground(BLACK);
        DrawTexturePro(canvas.texture,{0,0,W,-H},{offset.x,offset.y,W*scale,H*scale},{0,0},0,WHITE);EndDrawing();
        if(!capture.empty()&&++frames>=captureFrames){
            Image image=LoadImageFromTexture(canvas.texture);ImageFlipVertical(&image);
            if(!ExportImage(image,capture.c_str()))result=1;UnloadImage(image);break;
        }
        if(smoke&&studio::smokeFrame(__FILE__))break;
    }
    UnloadRenderTexture(canvas);studio::unload();CloseWindow();return result;
}
