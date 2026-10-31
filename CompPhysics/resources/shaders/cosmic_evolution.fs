#version 330
in vec2 fragTexCoord;
out vec4 finalColor;
uniform vec2 resolution;
uniform float progress;
uniform float clockTime;
uniform float yaw;
uniform float pitch;
uniform float zoom;
uniform int chapter;
uniform int variant;
const float PI=3.14159265;
float hash(vec3 p){p=fract(p*.3183099+vec3(.1,.2,.3));p*=17.;return fract(p.x*p.y*p.z*(p.x+p.y+p.z));}
float noise(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(mix(hash(i),hash(i+vec3(1,0,0)),f.x),mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),mix(hash(i+vec3(0,1,1)),hash(i+vec3(1)),f.x),f.y),f.z);}
float fbm(vec3 p){float n=0.,a=.5;for(int i=0;i<4;i++){n+=a*noise(p);p=p*2.03+vec3(7.1,1.7,3.2);a*=.5;}return n;}
mat2 rot(float a){return mat2(cos(a),-sin(a),sin(a),cos(a));}
float hitSphere(vec3 ro,vec3 rd,float r){float b=dot(ro,rd),d=b*b-dot(ro,ro)+r*r;return d>0.?-b-sqrt(d):-1.;}
vec3 stars(vec3 d){
    vec3 c=vec3(0);vec2 sky=vec2(atan(d.z,d.x),asin(d.y));
    for(int i=0;i<3;i++){
        float scale=150.+float(i)*130.;vec2 grid=sky*scale;vec2 cell=floor(grid);vec2 f=fract(grid)-.5;
        float h=hash(vec3(cell,float(i)));vec2 offset=vec2(h,hash(vec3(cell,7.)))*.65-.325;
        float r=length(f-offset);float core=exp(-r*r*(140.+float(i)*100.));
        c+=mix(vec3(.48,.68,1),vec3(1,.73,.45),h)*core*step(.974,h)*(1.+float(i)*.3);
    }
    float n=fbm(d*3.+vec3(8,0,3));
    float cloud=pow(max(0.,n-.23),3.);
    c+=cloud*mix(vec3(.07,.13,.4),vec3(.5,.10,.27),fbm(d*7.))*1.7;
    return c+vec3(.003,.005,.014);
}
vec3 starColor(){return variant==0?vec3(1,.25,.07):(variant==1?vec3(1,.76,.37):vec3(.45,.7,1));}
vec3 surface(vec3 n,float phase){
    vec3 p=n;p.xz=rot(clockTime*.035)*p.xz;
    float f=fbm(p*5.);float detail=fbm(p*23.);
    if(chapter==0){
        float mature=smoothstep(.38,.93,progress);
        vec3 lava=mix(vec3(.08,.006,.003),vec3(2.5,.32,.015),smoothstep(.47,.61,fbm(p*8.)));
        vec3 terrain;
        if(variant==0){
            float land=smoothstep(.46,.51,f);
            terrain=mix(vec3(.015,.12,.26),mix(vec3(.12,.28,.13),vec3(.43,.32,.18),detail),land);
            terrain=mix(terrain,vec3(.8,.91,.99),smoothstep(.79,.96,abs(p.y)+.13*f));
            float clouds=smoothstep(.56,.7,fbm(p*8.+vec3(clockTime*.012,0,0)));
            terrain=mix(terrain,vec3(.9,.95,1),clouds*.85);
        }else if(variant==1){
            float bands=sin(p.y*45.+fbm(p*7.)*9.);
            terrain=mix(vec3(.35,.16,.07),vec3(.88,.73,.5),bands*.5+.5);
            float storm=exp(-length((vec2(atan(p.z,p.x),p.y)-vec2(.4,-.24))*vec2(2.,6.))*5.);
            terrain=mix(terrain,vec3(.65,.17,.07),storm);
        }else{terrain=mix(vec3(.03,.22,.53),vec3(.25,.73,.89),.5+.35*sin(p.y*27.+f*5.));}
        return mix(lava,terrain,mature);
    }
    vec3 col=chapter==2?vec3(.5,.72,1.):starColor();
    if(phase>=3.&&phase<4.&&variant>0)col=vec3(1,.18,.025);
    if(phase>=4.)col=variant==2?vec3(.008):vec3(.65,.82,1);
    return col*(.65+f*.75+pow(detail,5.)*1.6);
}
void main(){
    vec2 screen=gl_FragCoord.xy/resolution;
    vec2 uv=(screen-.5)*vec2(resolution.x/resolution.y,1.);
    // Reconstruct a perspective camera ray at the fixed scene render resolution.
    vec3 ro=vec3(cos(yaw)*cos(pitch),sin(pitch),sin(yaw)*cos(pitch))*zoom;
    vec3 forward=normalize(-ro),right=normalize(cross(forward,vec3(0,1,0))),up=cross(right,forward);
    vec3 rd=normalize(forward*1.7+right*uv.x+up*uv.y);
    vec3 col=stars(rd);float phase=progress*5.;
    if(chapter==3){
        // Optically thin emissive volume: a schematic stellar distribution.
        float entry=hitSphere(ro,rd,4.8);
        if(entry>0.){
            vec3 light=vec3(0);float dt=9.6/64.;
            for(int i=0;i<64;i++){
                vec3 p=ro+rd*(entry+(float(i)+.5)*dt);
                p.xz=rot(clockTime*.018)*p.xz;
                float r=length(p.xz),a=atan(p.z,p.x);
                float settled=smoothstep(.05,.8,progress);
                float thick=mix(1.2,.15,settled);
                float disk=exp(-r*.85)*exp(-p.y*p.y/(thick*thick));
                float arm=pow(.5+.5*cos(a*3.-log(r+.15)*5.+clockTime*.035),7.);
                float dust=fbm(p*3.);
                float density;vec3 tint;
                if(variant==1){
                    density=exp(-length(p*vec3(.75,1.3,1.))*1.5)*(1.+noise(p*23.)*2.);
                    density*=mix(pow(fbm(p*2.2)*1.6,3.),1.,settled);
                    tint=mix(vec3(.45,.62,1),vec3(1,.65,.31),settled);
                }else if(variant==2){
                    density=exp(-length(p)*.8)*pow(max(0.,fbm(p*1.7)-.3),2.)*6.;
                    tint=mix(vec3(.3,.55,1.3),vec3(1,.23,.43),dust);
                }else{
                    density=disk*(.1+arm*2.*settled)*(.3+dust);
                    density=mix(exp(-length(p)*.7)*pow(max(0.,dust-.25),2.)*2.,density,settled);
                    density+=exp(-length(p)*4.)*.8;
                    tint=mix(vec3(.2,.44,1.2),vec3(1,.67,.35),exp(-r*1.4));
                    tint+=vec3(.8,.06,.25)*pow(dust,6.)*12.;
                    density*=1.-smoothstep(2.9,4.5,r);
                }
                float clump=pow(noise(p*65.),24.)*45.;
                light+=tint*density*(.65+clump)*dt*(.35+settled*2.);
            }
            col+=light;
        }
    }else{
        bool bh=(chapter==2&&(variant>0||phase>=2.)) || (chapter==1&&variant==2&&phase>=4.);
        float formed=smoothstep(.04,.57,progress);
        float radius=chapter==0?mix(.12,1.12,formed):mix(.08,.91,formed);
        if(chapter==1){
            if(variant>0&&phase>3.&&phase<4.)radius=mix(.91,1.9,smoothstep(3.,3.8,phase));
            if(phase>=4.)radius=variant==0?.24:(variant==1?.3:.48);
        }
        if(chapter==2)radius=variant==0?(phase<2.?mix(1.5,.48,smoothstep(1.,2.,phase)):.48):mix(.32,.72,progress);
        float tSurface=hitSphere(ro,rd,radius);
        float closest=length(cross(ro,rd));
        vec3 emission=chapter==0?vec3(1,.32,.07):starColor();
        if(chapter==1&&variant>0&&phase>=3.&&phase<4.)emission=vec3(1,.2,.04);
        if(chapter==1&&phase>=4.)emission=vec3(.6,.8,1);
        float halo=exp(-max(0.,closest-radius)*4.)*.2;
        if(chapter==0)halo*=1.-smoothstep(.5,.85,progress);
        if(!bh)col+=emission*halo*formed;
        if(tSurface>0.){
            vec3 n=normalize(ro+rd*tSurface);
            if(bh){
                col=vec3(.0003,.0002,.001);
            }else{
                vec3 tex=surface(n,phase);
                if(chapter==0){
                    float lambert=max(0.,dot(n,normalize(vec3(-3,2,4))));
                    float hot=1.-smoothstep(.3,.83,progress);
                    col=tex*(.035+lambert*.95+hot*.65);
                    float rim=pow(1.-max(0.,dot(n,-rd)),3.5);
                    col+=vec3(.15,.48,1.)*rim*.5*smoothstep(.5,.9,progress);
                    col+=vec3(1,.45,.16)*pow(max(0.,dot(reflect(normalize(vec3(3,-2,-4)),n),-rd)),70.)*.3;
                }else{col=tex*(.6+.5*pow(max(0.,dot(n,-rd)),.4));}
            }
        }
        // Orbiting solids become fewer and larger before merging into the world.
        if(chapter==0&&progress<.76){
            for(int i=0;i<28;i++){
                float fi=float(i),seed=hash(vec3(fi,4,9));
                float orbit=1.5+seed*2.7;
                orbit*=1.-.5*smoothstep(.35,.76,progress);
                float angle=fi*2.39996+clockTime*.32/pow(orbit,1.5)+progress*5.;
                vec3 center=vec3(cos(angle)*orbit,sin(fi*3.7)*.13,sin(angle)*orbit);
                float rockSize=mix(.022,.095,smoothstep(.12,.48,progress))*(.5+seed);
                rockSize*=1.-smoothstep(.6,.76,progress);
                float rockHit=hitSphere(ro-center,rd,rockSize);
                if(rockHit>0.&&(tSurface<0.||rockHit<tSurface)){
                    vec3 n=normalize(ro+rd*rockHit-center);
                    col=mix(vec3(.18,.13,.1),vec3(.75,.45,.22),seed)*(.15+max(0.,dot(n,normalize(vec3(-3,2,4)))));
                    tSurface=rockHit;
                }
            }
        }
        // Gas disks are sampled as a thin volume; surface depth occludes the far side.
        float diskStrength=chapter==0?1.-smoothstep(.25,.9,progress):1.-smoothstep(.25,.6,progress);
        if(chapter==2)diskStrength=(variant==0?smoothstep(.35,.65,progress):mix(.12,1.,smoothstep(.05,.7,progress)))*(variant==2?1.-.8*smoothstep(.8,1.,progress):1.);
        if(chapter==1&&bh)diskStrength=.8;
        if(diskStrength>.001){
            for(int i=0;i<9;i++){
                float plane=(float(i)-4.)*.032;
                if(abs(rd.y)<.001)continue;
                float t=(plane-ro.y)/rd.y;
                if(t<0.||(tSurface>0.&&t>tSurface))continue;
                vec3 p=ro+rd*t;float r=length(p.xz),a=atan(p.z,p.x);
                float inner=bh?radius*1.42:radius*1.3;
                float outer=bh?3.4:4.4;
                float band=smoothstep(inner,inner+.18,r)*(1.-smoothstep(outer*.7,outer,r));
                float swirl=fbm(vec3(r*5.,a*2.+clockTime*.13/(r+.3),plane*4.));
                float rings=.6+.4*sin(r*35.+swirl*12.-clockTime*.5);
                rings*=.4+noise(vec3(p.xz*22.,clockTime*.1))*.9;
                vec3 tint=mix(vec3(1,.19,.025),vec3(.4,.65,1.),smoothstep(.8,3.3,r));
                if(!bh)tint=mix(vec3(.38,.16,.1),vec3(.64,.42,.65),swirl);
                float beam=bh?pow(1.+.35*cos(a+yaw),3.):1.;
                col+=tint*band*(.025+swirl*.075)*rings*diskStrength*beam;
            }
        }
        if(bh){
            // Stylized photon ring / secondary disk image, not GR ray tracing.
            float ring=exp(-pow((closest-radius*1.12)/.035,2.));
            col+=vec3(1,.55,.18)*ring*1.5;
            float arc=exp(-pow((closest-radius*1.5)/.09,2.));
            if(tSurface<0.)col+=vec3(1,.39,.07)*arc*.28;
        }
        // Birth cloud and shed envelopes; deterministic, reversible with the timeline.
        float cloudAmount=chapter==2?0.:1.-smoothstep(.08,.45,progress);
        float shell=chapter==1&&variant>0?smoothstep(.78,.82,progress):0.;
        if(cloudAmount>.001||shell>0.){
            float entry=hitSphere(ro,rd,4.5);
            if(entry>0.)for(int i=0;i<24;i++){
                float t=entry+(float(i)+.5)*.36;
                if(tSurface>0.&&t>tSurface)break;
                vec3 p=ro+rd*t;float r=length(p);
                float n=fbm(p*1.6+vec3(clockTime*.012,0,0));
                float density=pow(max(0.,n-.31),2.)*exp(-r*r*.14)*cloudAmount;
                float shellR=mix(1.4,3.5,smoothstep(.8,1.,progress));
                density+=exp(-pow((r-shellR)/.3,2.))*shell*n*.13;
                vec3 tint=mix(vec3(.13,.38,.9),vec3(1,.16,.29),n);
                col+=tint*density*.52;
            }
        }
        if(chapter==2&&variant==0&&phase>2.&&phase<3.){
            float shockR=.7+(phase-2.)*3.;
            col+=vec3(.5,.65,1.)*exp(-pow((closest-shockR)/.12,2.))*(3.-phase)*.65;
        }
    }
    col=vec3(1.)-exp(-col*1.45);
    col=pow(col,vec3(.82));
    float vignette=1.-.25*dot(screen-.5,screen-.5);
    finalColor=vec4(col*vignette,1);
}
