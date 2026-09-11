#include "../common/physics_models.h"
#include "../common/quantum_solver.h"
#include <cstdio>
#include <cstdlib>

void require(bool condition,const char* message) {
    if (!condition) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); }
}
int main() {
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
    std::puts("PASS: trajectories, oscillator, frame-rate independence, overlap, mixing, wall impacts, Gaussian and quantum evolution");
}
