#pragma once

#include <array>
#include <string>
#include <vector>

namespace ai_space {

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct State {
    Vec2 position;
    Vec2 velocity;
};

struct SimConfig {
    double dt = 10.0;
    double duration = 7200.0;
    double decision_interval = 60.0;
    double lookahead = 900.0;
    double max_thrust_accel = 2.0e-5;
    double fuel_cost_per_burn = 0.0012;
    double success_distance = 5.0;
    double success_relative_speed = 0.020;
};

struct Sample {
    double time_min = 0.0;
    double distance_km = 0.0;
    double relative_speed_km_s = 0.0;
    double fuel_delta_v_km_s = 0.0;
    State chaser;
    State target;
    std::string action;
};

struct RunResult {
    bool success = false;
    std::vector<Sample> samples;
};

constexpr double kMuEarth = 398600.4418;
constexpr double kEarthRadius = 6378.137;

Vec2 operator+(Vec2 lhs, Vec2 rhs);
Vec2 operator-(Vec2 lhs, Vec2 rhs);
Vec2 operator*(double scalar, Vec2 value);
Vec2 operator*(Vec2 value, double scalar);
Vec2 operator/(Vec2 value, double scalar);

double norm(Vec2 value);
State circular_state(double radius_km, double angle_rad);
State rk4_step(State state, const std::string& action, const SimConfig& cfg);
std::string choose_action(State chaser, State target, const SimConfig& cfg);
RunResult run_rendezvous(const SimConfig& cfg);
void write_csv(const RunResult& result, const std::string& path);

}  // namespace ai_space
