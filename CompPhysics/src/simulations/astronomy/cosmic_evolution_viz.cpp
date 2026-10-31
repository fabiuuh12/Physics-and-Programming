#include "../../common/studio.h"
#include "cosmic_evolution_model.h"
#include <cstdio>
#include <sstream>
#include <string>

namespace {
constexpr int W=1440,H=900,SW=1000,SH=530;
constexpr Color ink{232,239,250,255},muted{143,160,184,255},accent{149,205,237,255};
constexpr Color line{37,48,68,255},panel{14,21,34,255};
const char* chapters[]={"PLANET FORMATION","STELLAR LIFECYCLE","BLACK HOLE GROWTH","GALAXY ASSEMBLY"};
const char* titles[]={"From dust to distant worlds.","Every star has a story.","Where gravity takes over.","Islands of a billion suns."};
const char* compact[]={"01   Planets","02   Stars","03   Black holes","04   Galaxies"};
Vector2 mouse{};
bool over(Rectangle r){return CheckCollisionPointRec(mouse,r);}
bool click(Rectangle r){return over(r)&&IsMouseButtonPressed(MOUSE_BUTTON_LEFT);}
void text(const char* s,float x,float y,float size=16,Color c=ink){studio::text(s,x,y,size,c);}
void wrap(const char* value,float x,float y,float width,float size,Color c){
    std::istringstream words(value);std::string word,row;
    while(words>>word){
        std::string next=row.empty()?word:row+" "+word;
        if(studio::textWidth(next.c_str(),size)>width&&!row.empty()){
            text(row.c_str(),x,y,size,c);y+=size*1.48f;row=word;
        }else row=next;
    }
    text(row.c_str(),x,y,size,c);
}
void button(Rectangle r,const char* label,bool selected){
    DrawRectangleRounded(r,.2f,8,selected?Color{35,57,76,255}:(over(r)?Color{26,37,54,255}:panel));
    DrawRectangleRoundedLinesEx(r,.2f,8,1,selected?Color{105,153,185,255}:line);
    text(label,r.x+15,r.y+(r.height-16)/2,16,selected?ink:muted);
}
std::string years(double y){
    char out[96];
    if(y>=1e12)std::snprintf(out,sizeof out,"%.2f trillion years",y/1e12);
    else if(y>=1e9)std::snprintf(out,sizeof out,"%.2f billion years",y/1e9);
    else if(y>=1e6)std::snprintf(out,sizeof out,"%.2f million years",y/1e6);
    else if(y>=1000)std::snprintf(out,sizeof out,"%.1f thousand years",y/1000);
    else std::snprintf(out,sizeof out,"%.1f years",y);
    return out;
}
void metric(float y,const char* label,const std::string& value){text(label,1072,y,12,muted);text(value.c_str(),1072,y+21,22);}
struct Uniforms {
    Shader shader;
    void f(const char* name,float value){SetShaderValue(shader,GetShaderLocation(shader,name),&value,SHADER_UNIFORM_FLOAT);}
    void i(const char* name,int value){SetShaderValue(shader,GetShaderLocation(shader,name),&value,SHADER_UNIFORM_INT);}
};
}
int main(int argc,char** argv){
    int chapter=0,variant=0;float progress=.12f;bool capture=false;
    std::string capturePath;
    try{
        for(int i=1;i<argc;i++){
            std::string arg=argv[i];
            if(arg=="--help"){
                std::puts("Cosmic Atlas: --chapter 0..3 --variant 0..2 --progress 0..1 --capture output.png");return 0;
            }
            if(i+1>=argc)throw std::runtime_error("Missing option value");
            std::string value=argv[++i];
            if(arg=="--chapter")chapter=std::stoi(value);
            else if(arg=="--variant")variant=std::stoi(value);
            else if(arg=="--progress")progress=std::stof(value);
            else if(arg=="--capture"){capture=true;capturePath=value;}
            else throw std::runtime_error("Unknown option: "+arg);
        }
        if(chapter<0||chapter>3||variant<0||variant>2||!std::isfinite(progress)||progress<0||progress>1)
            throw std::runtime_error("Chapter, variant, or progress outside supported range");
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}
    bool smoke=std::getenv("COMPPHYSICS_SMOKE_FRAMES")!=nullptr;
    SetConfigFlags(FLAG_WINDOW_RESIZABLE|FLAG_VSYNC_HINT|((capture||smoke)?FLAG_WINDOW_HIDDEN:0));
    InitWindow(W,H,"Cosmic Atlas | Formation & Evolution");
    SetWindowMinSize(960,600);SetTargetFPS(60);
    auto shaderPath=project_paths::root()/"resources/shaders/cosmic_evolution.fs";
    // Treat shader failure as an error; a blank default-shader preview is not a pass.
    Shader shader=LoadShader(nullptr,shaderPath.string().c_str());
    if(shader.id==0||GetShaderLocation(shader,"progress")<0){
        TraceLog(LOG_ERROR,"Cosmic Atlas shader failed to load");CloseWindow();return 1;
    }
    Uniforms u{shader};float resolution[]={float(SW),float(SH)};
    SetShaderValue(shader,GetShaderLocation(shader,"resolution"),resolution,SHADER_UNIFORM_VEC2);
    RenderTexture2D scene=LoadRenderTexture(SW,SH),canvas=LoadRenderTexture(W,H);
    SetTextureFilter(scene.texture,TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(canvas.texture,TEXTURE_FILTER_BILINEAR);
    float yaw=1.15f,pitch=.48f,zoom=9.0f,clock=0,speed=1;
    bool paused=capture||smoke,autoOrbit=true,labels=true,dragging=false,scrubbing=false;
    int frames=0,result=0;
    while(!WindowShouldClose()){
        float scale=std::min(float(GetScreenWidth())/W,float(GetScreenHeight())/H);
        Vector2 offset{(GetScreenWidth()-W*scale)/2,(GetScreenHeight()-H*scale)/2};
        Vector2 raw=GetMousePosition();mouse={(raw.x-offset.x)/scale,(raw.y-offset.y)/scale};
        Rectangle viewport{24,158,SW,SH},timeline{48,772,952,30};
        auto selectChapter=[&](int c){chapter=c;progress=.04f;variant=chapter==1?1:0;};
        for(int i=0;i<4;i++){
            if(click({24+i*252.f,105,238,37})||IsKeyPressed(KEY_ONE+i))selectChapter(i);
        }
        for(int i=0;i<3;i++)if(click({1072,193+i*44.f,312,35})){variant=i;progress=.04f;}
        if(IsKeyPressed(KEY_TAB)){variant=(variant+1)%3;progress=.04f;}
        if(IsKeyPressed(KEY_SPACE)||click({48,719,100,35})){
            if(paused&&progress>=1)progress=0;
            paused=!paused;
        }
        if(IsKeyPressed(KEY_R)||click({158,719,100,35})){progress=0;clock=0;}
        if(IsKeyPressed(KEY_A)||click({1072,766,148,35}))autoOrbit=!autoOrbit;
        if(IsKeyPressed(KEY_L)||click({1230,766,154,35}))labels=!labels;
        if(IsKeyPressed(KEY_C)){yaw=1.15f;pitch=.48f;zoom=9;}
        const float speeds[]={.25f,1,4};
        for(int i=0;i<3;i++)if(click({683+i*107.f,719,97,35}))speed=speeds[i];
        if(click(timeline)){scrubbing=true;paused=true;}
        if(scrubbing&&IsMouseButtonDown(MOUSE_BUTTON_LEFT))progress=std::clamp((mouse.x-48)/952,0.f,1.f);
        if(IsMouseButtonReleased(MOUSE_BUTTON_LEFT)){scrubbing=false;dragging=false;}
        if(click(viewport)){dragging=true;autoOrbit=false;}
        if(dragging){Vector2 d=GetMouseDelta();yaw-=d.x*.005f/scale;pitch=std::clamp(pitch+d.y*.005f/scale,.08f,1.4f);}
        if(over(viewport))zoom=std::clamp(zoom*std::exp(-GetMouseWheelMove()*.08f),5.6f,15.f);
        if(IsKeyPressed(KEY_RIGHT)){progress=std::min(1.f,progress+.01f);paused=true;}
        if(IsKeyPressed(KEY_LEFT)){progress=std::max(0.f,progress-.01f);paused=true;}
        for(int i=0;i<5;i++)if(click({48+i*192.f,817,181,27})){progress=i*.2f;paused=true;}
        if(!paused){
            float dt=std::min(GetFrameTime(),.1f);clock+=dt*speed;
            progress=std::min(1.f,progress+dt*speed/100.f);
            if(autoOrbit)yaw+=dt*.025f;
            if(progress>=1)paused=true;
        }
        const auto& track=evolution::tracks[chapter][variant];int stage=evolution::stage(progress);
        u.f("progress",progress);u.f("clockTime",clock);u.f("yaw",yaw);u.f("pitch",pitch);u.f("zoom",zoom);
        u.i("chapter",chapter);u.i("variant",variant);
        BeginTextureMode(scene);ClearBackground(BLACK);
        BeginShaderMode(shader);DrawRectangle(0,0,SW,SH,WHITE);EndShaderMode();EndTextureMode();
        BeginTextureMode(canvas);ClearBackground({8,13,23,255});
        text("C O S M I C   /   A T L A S",25,22,13,accent);
        text(titles[chapter],24,46,34);
        text("FORMATION  /  EVOLUTION  /  DEEP TIME",1055,30,12,muted);
        text("An interactive cosmic field guide",1055,53,16);
        for(int i=0;i<4;i++)button({24+i*252.f,105,238,37},compact[i],chapter==i);
        DrawTextureRec(scene.texture,{0,0,SW,-SH},{24,158},WHITE);
        DrawRectangleLinesEx(viewport,1,line);
        DrawRectangleGradientV(25,159,998,94,{5,9,18,215},{5,9,18,0});
        text(chapters[chapter],47,181,12,accent);
        text(track.stages[stage],46,205,26);
        text(TextFormat("0%d / 05",stage+1),927,184,15,muted);
        if(labels){
            DrawRectangleRounded({47,628,390,37},.2f,8,{8,15,27,205});
            text(track.scale,61,639,13,muted);
            text("DRAG TO ORBIT    /    SCROLL TO ZOOM",691,650,12,muted);
        }
        DrawRectangleRounded({1048,105,360,747},.03f,12,panel);
        text("CHOOSE A PATH",1072,128,12,accent);
        text("Same universe. Different outcomes.",1072,152,15,muted);
        for(int i=0;i<3;i++)button({1072,193+i*44.f,312,35},evolution::tracks[chapter][i].name,variant==i);
        text(track.subtitle,1072,335,15,accent);
        DrawLine(1072,367,1383,367,line);
        text("WHAT IS HAPPENING",1072,391,12,muted);
        text(track.stages[stage],1072,416,22);
        wrap(track.notes[stage],1072,454,307,17,ink);
        DrawLine(1072,584,1383,584,line);
        if(chapter==0){
            metric(603,"REFERENCE FINAL MASS",TextFormat("%.0f Earth masses",track.mass));
            metric(671,"REFERENCE TEMPERATURE",TextFormat("%.0f K  /  mature world",track.temperature));
        }else if(chapter==1){
            metric(603,"INITIAL STELLAR MASS",TextFormat("%.1f solar masses",track.mass));
            metric(671,"MAIN-SEQUENCE SURFACE",TextFormat("~ %.0f K",track.temperature));
        }else if(chapter==2){
            double mass=evolution::blackHoleMass(variant,progress);
            metric(603,variant==0?"CHOSEN FINAL REMNANT":"ILLUSTRATIVE MASS",TextFormat("%.3g solar masses",mass));
            metric(671,"NON-SPINNING HORIZON RADIUS",TextFormat("%.3g km",evolution::horizonKm(mass)));
        }else{
            metric(603,"REFERENCE STELLAR MASS",TextFormat("%.0e solar masses",track.mass));
            metric(671,"FORMATION HISTORY","One possible pathway");
        }
        button({1072,766,148,35},"Auto orbit",autoOrbit);button({1230,766,154,35},"Scene labels",labels);
        text(TextFormat("%d FPS  /  GPU PROCEDURAL SCENE",GetFPS()),1072,822,11,muted);
        DrawRectangleRounded({24,704,1000,148},.05f,10,panel);
        button({48,719,100,35},paused?"Play":"Pause",!paused);button({158,719,100,35},"Restart",false);
        std::string age=years(evolution::age(track,progress));text(age.c_str(),284,720,24);
        const char* speedNames[]={"0.25x","1x","4x"};
        for(int i=0;i<3;i++)button({683+i*107.f,719,97,35},speedNames[i],speed==speeds[i]);
        DrawRectangleRounded({48,784,952,4},1,8,line);
        if(progress>0)DrawRectangleRounded({48,784,952*progress,4},1,8,accent);
        for(int i=0;i<5;i++){
            float x=48+i*190.4f;DrawCircle(int(x),786,4,i<=stage?accent:muted);
            studio::fit(track.stages[i],{48+i*192.f,820,178,22},13,i==stage?ink:muted);
        }
        DrawCircle(int(48+952*progress),786,7,ink);
        text("Years since this track began  /  Each stage has its own time scale",48,859,12,muted);
        text("SPACE pause   1-4 journey   TAB type   R restart   C camera   A orbit   L labels",24,881,12,muted);
        text("Illustrative tracks / sizes & brightness rescaled",1048,875,12,muted);
        EndTextureMode();
        BeginDrawing();ClearBackground(BLACK);
        DrawTexturePro(canvas.texture,{0,0,W,-H},{offset.x,offset.y,W*scale,H*scale},{0,0},0,WHITE);EndDrawing();
        if(capture&&++frames>=3){
            Image img=LoadImageFromTexture(canvas.texture);ImageFlipVertical(&img);
            if(!ExportImage(img,capturePath.c_str()))result=1;
            UnloadImage(img);break;
        }
        if(smoke&&studio::smokeFrame(__FILE__))break;
    }
    UnloadRenderTexture(canvas);UnloadRenderTexture(scene);UnloadShader(shader);studio::unload();CloseWindow();
    return result;
}
