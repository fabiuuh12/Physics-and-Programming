#include "plant/renderer.h"
#include "plant/controls.h"

namespace {
constexpr Color ink{221,231,239,255},muted{143,164,183,255};
void annotation(Camera3D camera,Vector3 p,const char* text,Color color) {
    Vector2 s=GetWorldToScreen(p,camera);
    if(s.y<115||s.y>GetScreenHeight()-75||s.x<10||s.x>GetScreenWidth()-150)return;
    DrawCircleV(s,2.5f,color);DrawLineEx(s,{s.x+14,s.y-14},1,Fade(color,.65f));
    float width=studio::textWidth(text,13);
    DrawRectangleRounded({s.x+12,s.y-35,width+14,23},.2f,6,{8,17,28,220});
    studio::text(text,s.x+19,s.y-31,13,color);
}
}
int main() {
    bool smoke=std::getenv("COMPPHYSICS_SMOKE_FRAMES")!=nullptr;
    SetConfigFlags(FLAG_MSAA_4X_HINT|FLAG_WINDOW_RESIZABLE|(smoke?FLAG_WINDOW_HIDDEN:0));
    InitWindow(1400,880,"Nuclear power | From fission to electricity");
    SetWindowMinSize(1100,760);SetTargetFPS(60);
    int result=0;
    {
        plant::Renderer scene;plant::Controls controls;plant::Model model;
        if(!scene.ready){CloseWindow();return 1;}
        Camera3D camera{{19,18,26},{0,1.5f,0},{0,1,0},40,CAMERA_PERSPECTIVE};
        float yaw=1.05f,pitch=.56f,distance=31;
        bool paused=false,cutaway=true,labels=true,flows=true,orbit=false;
        int selected=-1;
        const char* stages[]={"01  Fission","02  Heat exchange","03  Turbine","04  Electricity","05  Cooling"};
        const char* descriptions[]={
            "Fuel fission releases heat. Lowering the control rods absorbs neutrons and reduces the reaction.",
            "Hot pressurized primary water heats separate secondary water through the steam-generator tubes.",
            "Secondary steam expands through the turbine blades, spinning the shaft attached to the generator.",
            "The rotor turns inside copper stator windings. The transformer feeds electricity to the grid.",
            "The condenser turns steam back into water. A third water circuit carries waste heat to the cooling tower."};
        while(!WindowShouldClose()) {
            float dt=std::min(GetFrameTime(),.05f);int w=GetScreenWidth(),h=GetScreenHeight();
            controls.update(camera,model,float(GetTime()));
            if(IsKeyPressed(KEY_SPACE)||IsKeyPressed(KEY_P))paused=!paused;
            if(IsKeyPressed(KEY_C))cutaway=!cutaway;
            if(IsKeyPressed(KEY_L))labels=!labels;
            if(IsKeyPressed(KEY_F))flows=!flows;
            if(IsKeyPressed(KEY_X))model.shutdown();
            if(IsKeyPressed(KEY_R)) {model.reset();paused=false;}
            if(!model.scrammed) {
                if(IsKeyDown(KEY_UP))model.rods=std::max(0.0f,model.rods-dt*.3f);
                if(IsKeyDown(KEY_DOWN))model.rods=std::min(1.0f,model.rods+dt*.3f);
            }
            if(IsKeyDown(KEY_LEFT))model.load=std::max(0.0f,model.load-dt*.3f);
            if(IsKeyDown(KEY_RIGHT))model.load=std::min(1.0f,model.load+dt*.3f);
            if(IsMouseButtonPressed(MOUSE_BUTTON_LEFT))orbit=GetMouseY()>140&&GetMouseY()<h-120&&!controls.mouseCaptured();
            if(!IsMouseButtonDown(MOUSE_BUTTON_LEFT))orbit=false;
            if(orbit&&!controls.mouseCaptured()) {Vector2 d=GetMouseDelta();yaw-=d.x*.004f;pitch=std::clamp(pitch+d.y*.004f,.15f,1.2f);}
            distance=std::clamp(distance*std::exp(-GetMouseWheelMove()*.07f),15.0f,48.0f);
            camera.position=Vector3Add(camera.target,{distance*std::cos(pitch)*std::cos(yaw),distance*std::sin(pitch),distance*std::cos(pitch)*std::sin(yaw)});
            if(!paused)model.step(dt);
            if(smoke) {
                if(const char* rods=std::getenv("COMPPHYSICS_PLANT_RODS"))model.rods=std::clamp(float(std::atof(rods)),0.0f,1.0f);
            }
            BeginDrawing();ClearBackground({8,16,27,255});
            BeginMode3D(camera);scene.render(camera,model,cutaway,flows);EndMode3D();
            if(labels) {
                annotation(camera,{-7,4.5f,0},"Reactor + control rods",plant::orange);
                annotation(camera,{-3,5,0},"Steam generator",plant::gold);
                annotation(camera,{-5,3,-1.7f},"Pressurizer",muted);
                annotation(camera,{2,3.9f,0},"Steam turbine",ink);
                annotation(camera,{6,4.1f,0},"Generator",plant::gold);
                annotation(camera,{2,1.6f,2.8f},"Condenser",plant::mint);
                annotation(camera,{0,5.3f,-6},"Cooling tower",plant::mint);
                annotation(camera,{9,2.5f,0},"Transformer / grid",plant::gold);
                annotation(camera,{-5,1.5f,2},"Primary pump",plant::blue);
            }
            // Console annotations remain visible with plant labels off.
            annotation(camera,{-3,2.9f,6.15f},TextFormat("RODS  %.0f%%",model.rods*100),plant::orange);
            annotation(camera,{-1,2.25f,6.1f},TextFormat("LOAD  %.0f%%",model.load*100),plant::gold);
            annotation(camera,{1,2.1f,6.25f},model.scrammed?"SHUT DOWN":"SCRAM",{255,109,104,255});
            DrawRectangleGradientV(0,0,w,145,Color{8,16,27,255},Color{8,16,27,235});
            studio::text("NUCLEAR / POWER",36,22,24,ink);
            studio::text("A working pressurized-water plant",36,54,14,muted);
            studio::text(TextFormat("%.0f MW",model.electric()),w-185,21,27,plant::gold);
            studio::text("net electricity",w-184,55,13,muted);
            float tabWidth=(w-72)/5.0f;
            for(int i=0;i<5;++i) {
                Rectangle tab{36+i*tabWidth,91,tabWidth-8,34};
                if(studio::clicked(tab))selected=selected==i?-1:i;
                DrawRectangleRounded(tab,.2f,8,selected==i?Color{47,67,82,255}:Color{22,36,51,255});
                studio::text(stages[i],tab.x+12,tab.y+9,14,selected==i?plant::gold:muted);
            }
            if(selected>=0) {
                studio::fit(descriptions[selected],{36,148,float(w-72),24},16,ink);
                Vector3 anchors[]={{-7,3,0},{-3,3,0},{2,3.2f,0},{6,3.2f,0},{0,4,-6}};
                Vector2 p=GetWorldToScreen(anchors[selected],camera);DrawCircleLinesV(p,29,plant::gold);
            }
            DrawRectangleGradientV(0,h-110,w,110,Color{8,16,27,0},Color{8,16,27,255});
            studio::text(TextFormat("%.0f MW heat   /   %.0f MW electricity   /   %.0f MW rejected",model.thermal(),model.electric(),model.rejected()),36,h-88,15,muted);
            const char* handStatus=!controls.receiverOk?"Hand port unavailable - mouse ready":controls.connected?"Hands connected - pinch a lever, dial or button":"Mouse ready / run nuclear_plant_hands.py for hands";
            studio::text(handStatus,36,h-59,14,controls.connected?plant::mint:muted);
            studio::text("Drag orbit / Scroll zoom / Space pause / C cutaway / L labels / F flow / R restart / X shutdown",36,h-30,12,muted);
            studio::text(paused?"PAUSED":model.scrammed?"SHUTDOWN / cooling":"ILLUSTRATIVE MODEL",w-205,h-30,12,model.scrammed?Color{255,109,104,255}:muted);
            controls.drawCursors();EndDrawing();
            if(studio::smokeFrame(__FILE__)) {
                TraceLog(LOG_INFO,"PLANT: rods=%.3f load=%.3f shutdown=%d net=%.1f",model.rods,model.load,int(model.scrammed),model.electric());
                break;
            }
        }
    }
    studio::unload();CloseWindow();return result;
}
