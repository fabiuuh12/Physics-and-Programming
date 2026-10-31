#include "../../common/studio.h"
#include "../../common/spacetime_geometry.h"
#include "raymath.h"
#include "rlgl.h"
#include <cstdio>
#include <stdexcept>

namespace {
constexpr int W=1440,H=900;
constexpr Color ink{230,241,250,255},muted{146,168,192,255},cyan{105,224,241,255};
const char* names[]={"Planet","Star","Black hole","Wormhole"};
const Color colors[]={{85,178,227,255},{255,199,108,255},{248,153,98,255},{181,140,255,255}};
using spacetime::Body;using spacetime::Kind;
struct World {
    std::vector<Body> bodies;float gain=1;bool layers=false,surface=false;
    Vector3 point(double x,double z,float lift=0)const{return {float(x),float(spacetime::height(x,z,bodies,gain))+lift,float(z)};}
    Vector3 center(const Body& b)const{return point(b.x,b.z,float(spacetime::radius(b))*.72f);}
};
Vector2 mouse{};
void text(const char* s,float x,float y,float size=16,Color c=ink){studio::text(s,x,y,size,c);}
bool over(Rectangle r){return CheckCollisionPointRec(mouse,r);}
void ring(Vector3 p,float radius,Color color,float phase=0){
    for(int i=0;i<100;i++){
        float a=phase+i*2*PI/100,b=phase+(i+1)*2*PI/100;
        DrawLine3D(Vector3Add(p,{cosf(a)*radius,0,sinf(a)*radius}),Vector3Add(p,{cosf(b)*radius,0,sinf(b)*radius}),color);
    }
}
// Intersect the actual deformed grid, not an invisible flat placement plane.
bool gridHit(Ray ray,const World& world,Vector3& hit){
    BoundingBox box{{-14,-8.1f,-14},{14,.01f,14}};
    auto entry=GetRayCollisionBox(ray,box);if(!entry.hit)return false;
    float start=std::max(0.f,entry.distance-.01f),previous=start;
    for(int i=0;i<380;i++){
        float t=start+i*.25f;Vector3 p=Vector3Add(ray.position,Vector3Scale(ray.direction,t));
        if(p.x < -14.02f||p.x>14.02f||p.z < -14.02f||p.z>14.02f||p.y< -8.2f) {if(i>1)break;continue;}
        if(p.y<=spacetime::height(p.x,p.z,world.bodies,world.gain)){
            float lo=previous,hi=t;
            for(int j=0;j<18;j++){
                float mid=(lo+hi)/2;Vector3 q=Vector3Add(ray.position,Vector3Scale(ray.direction,mid));
                if(q.y>spacetime::height(q.x,q.z,world.bodies,world.gain))lo=mid;else hi=mid;
            }
            hit=Vector3Add(ray.position,Vector3Scale(ray.direction,(lo+hi)/2));return true;
        }previous=t;
    }return false;
}
int pick(Ray ray,const World& w){
    int selected=-1;float nearest=1e9;
    for(size_t i=0;i<w.bodies.size();i++){
        const auto& b=w.bodies[i];float radius=float(spacetime::radius(b))*(b.kind==Kind::Wormhole?1.65f:1.15f);
        auto hit=GetRayCollisionSphere(ray,w.center(b),radius);
        if(hit.hit&&hit.distance<nearest){nearest=hit.distance;selected=int(i);}
    }return selected;
}
void drawGrid(const World& w){
    constexpr int N=112;
    static std::vector<Vector3> vertices((N+1)*(N+1));
    for(int z=0;z<=N;z++)for(int x=0;x<=N;x++)vertices[z*(N+1)+x]=w.point(-14+x*.25,-14+z*.25);
    rlDisableDepthMask();
    if(w.surface){
        rlDisableBackfaceCulling();
        for(int z=0;z<N;z++)for(int x=0;x<N;x++){
            auto a=vertices[z*(N+1)+x],b=vertices[z*(N+1)+x+1],c=vertices[(z+1)*(N+1)+x+1],d=vertices[(z+1)*(N+1)+x];
            DrawTriangle3D(a,b,c,{23,67,91,30});DrawTriangle3D(a,c,d,{23,67,91,30});
        }
        rlEnableBackfaceCulling();
    }
    int layers=w.layers?3:1;
    for(int layer=0;layer<layers;layer++){
        float offset=w.layers?(layer-1)*2.f:0;
        for(int axis=0;axis<2;axis++)for(int line=0;line<=N;line+=2)for(int j=0;j<N;j++){
            int a=axis==0?line*(N+1)+j:j*(N+1)+line;
            int b=axis==0?a+1:a+N+1;
            auto p=vertices[a],q=vertices[b];p.y+=offset+.025f;q.y+=offset+.025f;
            float near=std::clamp(-vertices[a].y/8,0.f,1.f);
            Color c=ColorLerp(Color{40,113,169,255},Color{123,241,239,255},near);
            c.a=(w.layers&&layer!=1)?48:(line%8==0?200:110);
            DrawLine3D(p,q,c);
        }
    }
    if(w.layers)for(int x=-12;x<=12;x+=2)for(int z=-12;z<=12;z+=2)DrawLine3D(w.point(x,z,-2),w.point(x,z,2),{61,155,184,50});
    rlEnableDepthMask();
}
float noise(Vector3 n){return .5f+.5f*sinf(n.x*13+sinf(n.z*17)*2)*cosf(n.y*19+n.z*7);}
void sphere(const Body& b,Vector3 center,float time){
    int type=int(b.kind);float radius=float(spacetime::radius(b));
    if(b.kind==Kind::Wormhole){
        DrawSphereEx(center,radius*.75f,20,30,{8,4,19,255});
        for(int band=0;band<9;band++){
            float r=radius*(1.1f+band*.055f);Vector3 p=center;p.y+=sinf(time*1.5f+band*.4f)*.025f;
            ring(p,r,Fade(colors[type],.85f-band*.075f));
        }
        for(int i=0;i<24;i++){
            float a=i*2*PI/24+time*.5f;Vector3 p=Vector3Add(center,{cosf(a)*radius*1.38f,0,sinf(a)*radius*1.38f});
            DrawSphere(p,.035f,colors[type]);
        }return;
    }
    if(b.kind==Kind::BlackHole){
        DrawSphereEx(center,radius,24,36,{2,3,8,255});
        for(int band=0;band<18;band++){
            float r=radius*(1.15f+band*.06f);
            for(int i=0;i<100;i++){
                float a=i*2*PI/100,b=a+2*PI/100;
                Color c=ColorLerp(Color{255,231,169,255},Color{229,88,44,255},band/18.f);
                c=Fade(c,(.2f+.65f*(.5f+.5f*sinf(a*3-time*2+band)))*(1-band/22.f));
                DrawLine3D(Vector3Add(center,{cosf(a)*r,0,sinf(a)*r}),Vector3Add(center,{cosf(b)*r,0,sinf(b)*r}),c);
            }
        }return;
    }
    // A shaded procedural sphere keeps planets readable against the wire grid.
    rlDisableBackfaceCulling();
    constexpr int rows=22,columns=36;
    auto vertex=[&](int lat,int lon){
        float a=PI*lat/rows,phi=2*PI*lon/columns;
        Vector3 n{sinf(a)*cosf(phi),cosf(a),sinf(a)*sinf(phi)};
        float shade=.18f+.82f*std::max(0.f,Vector3DotProduct(n,Vector3Normalize({-1,2,1})));
        Vector3 local=n;float angle=time*.04f;
        local.x=n.x*cosf(angle)-n.z*sinf(angle);local.z=n.x*sinf(angle)+n.z*cosf(angle);
        Color color;
        if(b.kind==Kind::Star)color=ColorLerp(Color{244,128,39,255},Color{255,238,166,255},noise(local));
        else{
            float land=noise(local);
            color=land>.63f?Color{100,153,102,255}:Color{39,124,190,255};
            if(std::abs(n.y)>.88f)color={211,235,241,255};
            shade=std::max(.16f,shade);
        }
        if(b.kind==Kind::Star)shade=.85f+noise(local)*.15f;
        color.r=static_cast<unsigned char>(color.r*shade);color.g=static_cast<unsigned char>(color.g*shade);color.b=static_cast<unsigned char>(color.b*shade);
        return std::make_pair(Vector3Add(center,Vector3Scale(n,radius)),color);
    };
    rlBegin(RL_TRIANGLES);
    for(int y=0;y<rows;y++)for(int x=0;x<columns;x++){
        for(auto index: {std::pair<int,int>{y,x},{y+1,x},{y+1,x+1},{y,x},{y+1,x+1},{y,x+1}}){
            auto p=vertex(index.first,index.second);rlColor4ub(p.second.r,p.second.g,p.second.b,255);rlVertex3f(p.first.x,p.first.y,p.first.z);
        }
    }
    rlEnd();rlEnableBackfaceCulling();
}
void glow(Camera3D camera,Vector3 center,float radius,Color color,float alpha){
    if(Vector3DotProduct(Vector3Subtract(center,camera.position),Vector3Subtract(camera.target,camera.position))<=0)return;
    Vector2 p=GetWorldToScreenEx(center,camera,W,H);
    DrawCircleGradient(int(p.x),int(p.y),radius,Fade(color,alpha),Fade(color,0));
}
void populate(World& w,const std::string& scene){
    if(scene=="empty")return;
    if(scene=="mixed"){
        spacetime::add(w.bodies,Kind::Star,-3,-2);spacetime::add(w.bodies,Kind::Planet,3,-1);
        spacetime::add(w.bodies,Kind::BlackHole,4,5);spacetime::add(w.bodies,Kind::Wormhole,-5,4);return;
    }
    for(int i=0;i<4;i++)if(scene==std::string(i==2?"blackhole":(i==3?"wormhole":(i==1?"star":"planet")))){
        spacetime::add(w.bodies,Kind(i),0,0);return;
    }
    throw std::runtime_error("Unknown scene");
}
}

int main(int argc,char** argv){
    World w;std::string capture,scene="empty";int limit=3;
    try{
        for(int i=1;i<argc;i++){
            std::string arg=argv[i];
            if(arg=="--help"){std::puts("--scene empty|mixed|planet|star|blackhole|wormhole --lattice 0|1 --capture file.png --frames 1..1200");return 0;}
            if(i+1>=argc)throw std::runtime_error("Missing option value");
            std::string value=argv[++i];
            if(arg=="--scene")scene=value;else if(arg=="--capture")capture=value;
            else if(arg=="--lattice")w.layers=std::stoi(value)!=0;
            else if(arg=="--frames")limit=std::stoi(value);else throw std::runtime_error("Unknown option");
        }
        if(limit<1||limit>1200)throw std::runtime_error("Invalid frame count");populate(w,scene);
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}
    bool smoke=std::getenv("COMPPHYSICS_SMOKE_FRAMES")!=nullptr;
    SetConfigFlags(FLAG_WINDOW_RESIZABLE|FLAG_VSYNC_HINT|((smoke||!capture.empty())?FLAG_WINDOW_HIDDEN:0));
    InitWindow(W,H,"Spacetime Sandbox");SetWindowMinSize(900,560);SetTargetFPS(60);
    RenderTexture2D canvas=LoadRenderTexture(W,H);SetTextureFilter(canvas.texture,TEXTURE_FILTER_BILINEAR);
    Camera3D camera{};camera.up={0,1,0};camera.fovy=45;camera.projection=CAMERA_PERSPECTIVE;
    float yaw=.83f,pitch=.66f,distance=34,time=0,dragY=0;Vector3 target{0,-1,0};
    int tool=0,selected=-1,dragged=-1,frames=0,result=0;bool help=false,paused=false,orbiting=false,panning=false;
    while(!WindowShouldClose()){
        float scale=std::min(float(GetScreenWidth())/W,float(GetScreenHeight())/H);
        Vector2 offset{(GetScreenWidth()-W*scale)/2,(GetScreenHeight()-H*scale)/2};
        Vector2 raw=GetMousePosition();mouse={(raw.x-offset.x)/scale,(raw.y-offset.y)/scale};
        Rectangle palette{354,824,732,54};bool ui=over(palette);
        for(int i=0;i<4;i++)if(IsKeyPressed(KEY_ONE+i)||(ui&&IsMouseButtonPressed(MOUSE_BUTTON_LEFT)&&over({362+i*180.f,832,172,38})))tool=i;
        if(IsKeyPressed(KEY_TAB))tool=(tool+1)%4;
        if(IsKeyPressed(KEY_H))help=!help;
        if(IsKeyPressed(KEY_G))w.layers=!w.layers;
        if(IsKeyPressed(KEY_T))w.surface=!w.surface;
        if(IsKeyPressed(KEY_P)||IsKeyPressed(KEY_SPACE))paused=!paused;
        if(IsKeyPressed(KEY_R)){w.bodies.clear();selected=dragged=-1;}
        if(IsKeyPressed(KEY_O)){yaw=.83f;pitch=.66f;distance=34;target={0,-1,0};}
        if((IsKeyPressed(KEY_DELETE)||IsKeyPressed(KEY_BACKSPACE))&&selected>=0){w.bodies.erase(w.bodies.begin()+selected);selected=dragged=-1;}
        if(IsKeyPressed(KEY_LEFT_BRACKET))w.gain=std::max(.2f,w.gain*.85f);
        if(IsKeyPressed(KEY_RIGHT_BRACKET))w.gain=std::min(3.f,w.gain*1.18f);
        if(selected>=0){
            if(IsKeyPressed(KEY_EQUAL)||IsKeyPressed(KEY_KP_ADD))spacetime::resize(w.bodies[selected],1.2);
            if(IsKeyPressed(KEY_MINUS)||IsKeyPressed(KEY_KP_SUBTRACT))spacetime::resize(w.bodies[selected],1/1.2);
        }
        if(!ui&&IsMouseButtonPressed(MOUSE_BUTTON_RIGHT))orbiting=true;
        if(!ui&&IsMouseButtonPressed(MOUSE_BUTTON_MIDDLE))panning=true;
        if(IsMouseButtonReleased(MOUSE_BUTTON_RIGHT))orbiting=false;
        if(IsMouseButtonReleased(MOUSE_BUTTON_MIDDLE))panning=false;
        if(orbiting){auto d=GetMouseDelta();yaw-=d.x*.004f/scale;pitch=std::clamp(pitch+d.y*.004f/scale,.15f,1.48f);}
        if(panning){auto d=GetMouseDelta();float speed=distance*.001f/scale;
            target.x+=(-d.x*sinf(yaw)+d.y*cosf(yaw))*speed;
            target.z+=(d.x*cosf(yaw)+d.y*sinf(yaw))*speed;
        }
        if(!ui)distance=std::clamp(distance*expf(-GetMouseWheelMove()*.075f),10.f,65.f);
        camera.target=target;camera.position=Vector3Add(target,{distance*cosf(pitch)*cosf(yaw),distance*sinf(pitch),distance*cosf(pitch)*sinf(yaw)});
        Ray ray=GetScreenToWorldRayEx(mouse,camera,W,H);Vector3 hit{};bool valid=gridHit(ray,w,hit);
        int hovered=ui?-1:pick(ray,w);
        if(!ui&&!help&&IsMouseButtonPressed(MOUSE_BUTTON_LEFT)){
            if(hovered>=0){selected=dragged=hovered;dragY=w.center(w.bodies[selected]).y;}
            else if(valid&&spacetime::add(w.bodies,Kind(tool),hit.x,hit.z)){selected=int(w.bodies.size())-1;}
            else selected=-1;
        }
        if(IsMouseButtonReleased(MOUSE_BUTTON_LEFT))dragged=-1;
        if(dragged>=0&&IsMouseButtonDown(MOUSE_BUTTON_LEFT)&&std::abs(ray.direction.y)>.001){
            float t=(dragY-ray.position.y)/ray.direction.y;
            if(t>0){auto p=Vector3Add(ray.position,Vector3Scale(ray.direction,t));spacetime::move(w.bodies[dragged],p.x,p.z);}
        }
        if(!paused)time+=capture.empty()?std::min(GetFrameTime(),.1f):1.f/60;
        BeginTextureMode(canvas);ClearBackground({4,8,17,255});
        DrawRectangleGradientV(0,0,W,H,{7,13,26,255},{2,5,12,255});
        BeginMode3D(camera);drawGrid(w);
        for(size_t i=0;i<w.bodies.size();i++){
            const auto& body=w.bodies[i];auto p=w.center(body);sphere(body,p,time);
            if(int(i)==selected)ring(w.point(body.x,body.z,.07f),float(spacetime::radius(body))*1.65f,Fade(cyan,.7f));
        }
        if(valid&&!ui&&hovered<0&&dragged<0&&!help&&!orbiting){
            float r=.4f;Vector3 p=w.point(std::clamp(double(hit.x),-12.,12.),std::clamp(double(hit.z),-12.,12.),.06f);
            ring(p,r,Fade(colors[tool],.7f));DrawLine3D(Vector3Add(p,{-r,0,0}),Vector3Add(p,{r,0,0}),Fade(colors[tool],.4f));
            DrawLine3D(Vector3Add(p,{0,0,-r}),Vector3Add(p,{0,0,r}),Fade(colors[tool],.4f));
        }
        EndMode3D();
        BeginBlendMode(BLEND_ADDITIVE);
        for(const auto& body:w.bodies){
            float radius=float(spacetime::radius(body));auto p=w.center(body);
            float pixels=std::clamp(radius*2200/Vector3Distance(p,camera.position),12.f,170.f);
            if(body.kind==Kind::Star)glow(camera,p,pixels*1.65f,colors[1],.18f);
            if(body.kind==Kind::Wormhole)glow(camera,p,pixels*1.5f,colors[3],.12f);
        }EndBlendMode();
        text("Spacetime Sandbox",24,20,25);
        text("Click to place  /  Drag objects  /  Right-drag to orbit  /  H help",25,55,14,muted);
        if(selected>=0){const auto& body=w.bodies[selected];text(TextFormat("%s  /  mass %.1f  /  +/- resize  /  Delete remove",names[int(body.kind)],body.mass),25,82,14,colors[int(body.kind)]);}
        DrawRectangleRounded(palette,.2f,8,{9,17,30,230});
        for(int i=0;i<4;i++){
            Rectangle r{362+i*180.f,832,172,38};
            if(tool==i)DrawRectangleRounded(r,.2f,8,{29,49,66,255});
            DrawCircle(int(r.x+18),int(r.y+19),4,colors[i]);text(TextFormat("%d  %s",i+1,names[i]),r.x+30,r.y+11,15,tool==i?ink:muted);
        }
        text(TextFormat("%d / %d objects",int(w.bodies.size()),spacetime::maxBodies),24,865,13,muted);
        text("Illustrative grid deformation",1204,865,12,muted);
        if(w.bodies.empty())text("Choose an object below, then click anywhere on the grid.",465,785,16,muted);
        if(help){
            DrawRectangleRounded({420,247,600,350},.08f,8,{9,18,32,248});
            text("Build your own little universe",447,272,25,cyan);
            const char* lines[]={"1-4 or Tab: choose an object", "Left click: place or select; drag an existing object to move it", "Right drag: orbit  /  Wheel: zoom  /  Middle drag: pan", "+/-: change selected mass  /  Delete: remove selected", "R: clear grid  /  O: reset camera  /  P: pause animation", "G: layered grid  /  T: translucent surface  /  [ ]: grid depth", "H: close help", "Objects stay where you put them. Wells blend visually."};
            for(int i=0;i<8;i++)text(lines[i],447,321+i*30,15,i==7?muted:ink);
        }
        EndTextureMode();BeginDrawing();ClearBackground(BLACK);
        DrawTexturePro(canvas.texture,{0,0,W,-H},{offset.x,offset.y,W*scale,H*scale},{0,0},0,WHITE);EndDrawing();
        if(!capture.empty()&&++frames>=limit){Image img=LoadImageFromTexture(canvas.texture);ImageFlipVertical(&img);if(!ExportImage(img,capture.c_str()))result=1;UnloadImage(img);break;}
        if(smoke&&studio::smokeFrame(__FILE__))break;
    }
    UnloadRenderTexture(canvas);studio::unload();CloseWindow();return result;
}
