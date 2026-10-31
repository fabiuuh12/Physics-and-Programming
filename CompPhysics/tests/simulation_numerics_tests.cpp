#include "../src/common/physics_models.h"
#include "../src/common/project_paths.h"
#include "../src/common/quantum_solver.h"
#include "../src/common/spacetime_geometry.h"
#include <cstdio>
#include <cstdlib>

void require(bool condition,const char* message) {
    if (!condition) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); }
}
int main() {
    require(std::filesystem::exists(project_paths::root()/"resources/fonts/Inter-Regular.ttf"),"shared resource root resolves the bundled font");
    physics::Flight vacuum{0,0,12,10};
    auto flight=vacuum;
    for (int i=0;i<240;++i) flight=physics::projectileStep(flight,0,1.0/240);
    require(std::abs(flight[0]-12)<1e-10 && std::abs(flight[1]-5.095)<1e-10,"vacuum trajectory matches analytic solution");
    auto drag=vacuum;
    for (int i=0;i<240;++i) drag=physics::projectileStep(drag,0.08,1.0/240);
    require(drag[0]<flight[0] && drag[2]<flight[2],"drag reduces horizontal travel and speed");
    double x=1,v=0;
    for (int i=0;i<2400;++i) physics::oscillatorStep(x,v,i/240.0,1.0/240,1,10,0,0,0);
    require(std::abs(x-std::cos(std::sqrt(10.0)*10))<1e-7,"oscillator phase matches analytic solution");
    require(std::abs(0.5*v*v+5*x*x-5)<1e-7,"undriven oscillator conserves energy");
    auto runClock=[](int fps) {
        physics::Clock c; double position=1,velocity=0,time=0;
        for (int i=0;i<fps*4;++i) c.advance(1.0/fps,[&](double dt) {
            physics::oscillatorStep(position,velocity,time,dt,1,10,0.5,2,2.4); time+=dt;
        });
        return position;
    };
    require(std::abs(runClock(30)-runClock(144))<1e-12,"fixed-step dynamics independent of rendering rate");
    require(physics::circleOverlap(3,1,1)==0,"separated disks do not occult");
    require(std::abs(physics::circleOverlap(0,1,0.2)-physics::pi*0.04)<1e-12,"central transit has area-ratio depth");
    require(std::abs(physics::circleOverlap(1,1,1)-(2*physics::pi/3-std::sqrt(3.0)/2))<1e-12,"partial overlap matches equal-circle geometry");
    double anomaly=physics::eccentricAnomaly(1.2,0.7);
    require(std::abs(anomaly-0.7*std::sin(anomaly)-1.2)<1e-12,"Kepler solver satisfies mean-anomaly equation");
    require(physics::binaryEntropy(20,0)==0 && physics::binaryEntropy(10,10)==1,"mixing distinguishes separated and balanced colors");
    double p=0.2,speed=3;
    double impulse=physics::reflect(p,speed,-1,1,2);
    require(std::abs(p+0.2)<1e-12 && speed==-3 && impulse==18,"multiple wall impacts preserve speed and count impulse");
    require(std::abs(physics::packetSigma(0.5,0)-0.5)<1e-12,"initial Gaussian width");
    require(physics::packetSigma(0.5,2)>0.5,"free position width increases with fixed momentum width");
    double integral=0;
    for (int i=0;i<20000;++i) integral+=physics::gaussianDensity(-10+(i+0.5)*0.001,0,0.7)*0.001;
    require(std::abs(integral-1)<1e-12,"Gaussian probability density is normalized");
    physics::WavePacket free,barrier;
    free.reset(0.8,0); barrier.reset(0.8,3);
    for (int i=0;i<480;++i) { free.step(); barrier.step(); }
    double center=0;
    for (int i=0;i<free.count;++i) center+=free.x(i)*std::norm(free.psi[i])*free.dx;
    require(std::abs(center-(-7+std::sqrt(1.6)))<0.004,"free packet propagates at expected group velocity");
    for (int i=480;i<3360;++i) { free.step(); barrier.step(); }
    require(std::abs(free.norm()-1)<1e-10 && std::abs(barrier.norm()-1)<1e-10,"Schrodinger solver conserves probability");
    require(barrier.regions()[2]<free.regions()[2]*0.6,"barrier scatters the packet and reduces transmission");
    require(spacetime::height(16,1)==0,"Schwarzschild embedding is anchored at the reference rim");
    double derivative=(spacetime::height(4.00001,1)-spacetime::height(3.99999,1))/.00002;
    require(std::abs(derivative-std::sqrt(1./3))<1e-8,"Flamm slope reproduces exterior radial spatial metric");
    require(std::abs(spacetime::staticRate(4,1)-std::sqrt(.75))<1e-12,"static clock uses Schwarzschild lapse");
    require(std::isnan(spacetime::staticRate(.9,1)),"no static observer exists inside the horizon");
    require(std::isnan(spacetime::staticRate(.9,.01,1)),"exterior clock model excludes the stellar interior");
    require(spacetime::outgoingSlope(1,1)==0&&spacetime::outgoingSlope(.8,1)<0,"outgoing radial light stops moving outward at the horizon");
    auto straight=spacetime::schwarzschildPath(3,0,0);
    for(auto p:straight.points)require(std::abs(p.z-3)<1e-12,"flat-space null trajectories are straight");
    auto upper=spacetime::schwarzschildPath(5,1,1),lower=spacetime::schwarzschildPath(-5,1,1);
    require(!upper.captured&&upper.points.size()==lower.points.size(),"distant rays escape symmetrically");
    for(size_t i=0;i<upper.points.size();i++)
        require(std::abs(upper.points[i].x-lower.points[i].x)<1e-12&&std::abs(upper.points[i].z+lower.points[i].z)<1e-12,"spherical lens has reflection symmetry");
    require(spacetime::schwarzschildPath(1,1,1).captured,"low-impact light crosses the capture boundary");
    spacetime::State ray{-200,20,1,0};
    double L=-20,E=spacetime::photonEnergySquared(ray,.05);
    for(int i=0;i<8100;i++)ray=spacetime::rk4(ray,.05,[&](spacetime::State p){return spacetime::photonDerivative(p,.05,L);});
    require(std::abs(spacetime::photonEnergySquared(ray,.05)-E)<1e-10,"null solver conserves energy");
    require(std::abs(std::atan2(-ray[3],ray[2])-.005)<.005*.03,"weak deflection approaches 2rs / impact");
    spacetime::State orbit{1.5,0,0,1};
    for(int i=0;i<9000;i++)orbit=spacetime::rk4(orbit,.001,[](spacetime::State p){return spacetime::photonDerivative(p,1,1.5);});
    require(std::abs(std::hypot(orbit[0],orbit[1])-1.5)<1e-7,"photon sphere supports the unstable circular null orbit");
    require(spacetime::ellisPath(0,1).crossed,"radial Ellis ray passes through the throat");
    require(!spacetime::ellisPath(4,1).crossed,"large-impact Ellis ray stays in its original region");
    spacetime::State worm{5,.2,-.9,0};
    double wormEnergy=worm[2]*worm[2]+.25/(worm[0]*worm[0]+1);
    for(int i=0;i<2000;i++)worm=spacetime::rk4(worm,.005,[](spacetime::State p){return spacetime::ellisDerivative(p,1,.5);});
    require(std::abs(worm[2]*worm[2]+.25/(worm[0]*worm[0]+1)-wormEnergy)<1e-9,"Ellis null solver conserves ultrastatic energy");
    std::puts("PASS: trajectories, oscillator, frame-rate independence, overlap, mixing, wall impacts, Gaussian and quantum evolution");
}
