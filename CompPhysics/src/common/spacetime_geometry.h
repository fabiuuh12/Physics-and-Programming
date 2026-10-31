#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

// Artistic multi-source grid deformation, in arbitrary scene units.
// These softened wells are not superposed solutions of Einstein's equations.
namespace spacetime {
constexpr double extent=14;
constexpr int maxBodies=24;
enum class Kind { Planet,Star,BlackHole,Wormhole };
struct Body {Kind kind;double x,z,mass;};
inline double defaultMass(Kind k){return k==Kind::Planet?1:(k==Kind::Star?3.5:(k==Kind::BlackHole?7:5));}
inline double radius(const Body& b){return (b.kind==Kind::Star?.52:(b.kind==Kind::Planet?.38:.42))*std::cbrt(b.mass);}
inline double smooth(double x){x=std::clamp(x,0.,1.);return x*x*(3-2*x);}
inline double height(double x,double z,const std::vector<Body>& bodies,double gain=1){
    double strength=0;
    for(const auto& b:bodies){
        double dx=x-b.x,dz=z-b.z,soft=.75+radius(b)*.65;
        strength+=1.35*b.mass/std::sqrt(dx*dx+dz*dz+soft*soft);
    }
    double edge=smooth((extent-std::max(std::abs(x),std::abs(z)))/2);
    return -8*std::tanh(strength*gain/8)*edge;
}
inline bool add(std::vector<Body>& bodies,Kind kind,double x,double z){
    if(bodies.size()>=maxBodies)return false;
    bodies.push_back({kind,std::clamp(x,-12.,12.),std::clamp(z,-12.,12.),defaultMass(kind)});return true;
}
inline void move(Body& b,double x,double z){b.x=std::clamp(x,-12.,12.);b.z=std::clamp(z,-12.,12.);}
inline void resize(Body& b,double multiplier){b.mass=std::clamp(b.mass*multiplier,.2,20.);}
} // namespace spacetime
