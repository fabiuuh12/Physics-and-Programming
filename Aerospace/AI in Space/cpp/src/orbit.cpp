#include "orbit.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ai_space {

namespace {

const std::array<std::string, 5> kActions = {
    "coast",
    "prograde",
    "retrograde",
    "radial_out",
    "radial_in",
};

Vec2 gravity(Vec2 position) {
    const double radius = norm(position);
    return (-kMuEarth / (radius * radius * radius)) * position;
}

Vec2 thrust_direction(State state, const std::string& action) {
    if (action == "coast") {
        return {};
    }

    const Vec2 radial = state.position / norm(state.position);
    const Vec2 prograde = state.velocity / norm(state.velocity);

    if (action == "prograde") {
        return prograde;
    }
    if (action == "retrograde") {
        return -1.0 * prograde;
    }
    if (action == "radial_out") {
        return radial;
    }
    if (action == "radial_in") {
        return -1.0 * radial;
    }

    throw std::runtime_error("Unknown action: " + action);
}

Vec2 acceleration(State state, const std::string& action, const SimConfig& cfg) {
    return gravity(state.position) + cfg.max_thrust_accel * thrust_direction(state, action);
}

double rollout_cost(State chaser, State target, const std::string& action, const SimConfig& cfg) {
    const int steps = static_cast<int>(cfg.lookahead / cfg.dt);
    double min_distance = std::numeric_limits<double>::infinity();

    for (int i = 0; i < steps; ++i) {
        chaser = rk4_step(chaser, action, cfg);
        target = rk4_step(target, "coast", cfg);
        min_distance = std::min(min_distance, norm(chaser.position - target.position));
    }

    const double distance = norm(chaser.position - target.position);
    const double relative_speed = norm(chaser.velocity - target.velocity);
    const double fuel_penalty = action == "coast" ? 0.0 : cfg.fuel_cost_per_burn * cfg.lookahead;

    return distance + 900.0 * relative_speed + 0.2 * min_distance + fuel_penalty;
}

}  // namespace

Vec2 operator+(Vec2 lhs, Vec2 rhs) {
    return {lhs.x + rhs.x, lhs.y + rhs.y};
}

Vec2 operator-(Vec2 lhs, Vec2 rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y};
}

Vec2 operator*(double scalar, Vec2 value) {
    return {scalar * value.x, scalar * value.y};
}

Vec2 operator*(Vec2 value, double scalar) {
    return scalar * value;
}

Vec2 operator/(Vec2 value, double scalar) {
    return {value.x / scalar, value.y / scalar};
}

double norm(Vec2 value) {
    return std::sqrt(value.x * value.x + value.y * value.y);
}

State circular_state(double radius_km, double angle_rad) {
    const double speed = std::sqrt(kMuEarth / radius_km);
    return {
        {radius_km * std::cos(angle_rad), radius_km * std::sin(angle_rad)},
        {-speed * std::sin(angle_rad), speed * std::cos(angle_rad)},
    };
}

State rk4_step(State state, const std::string& action, const SimConfig& cfg) {
    const double dt = cfg.dt;

    const Vec2 k1_r = state.velocity;
    const Vec2 k1_v = acceleration(state, action, cfg);

    const State s2{state.position + 0.5 * dt * k1_r, state.velocity + 0.5 * dt * k1_v};
    const Vec2 k2_r = s2.velocity;
    const Vec2 k2_v = acceleration(s2, action, cfg);

    const State s3{state.position + 0.5 * dt * k2_r, state.velocity + 0.5 * dt * k2_v};
    const Vec2 k3_r = s3.velocity;
    const Vec2 k3_v = acceleration(s3, action, cfg);

    const State s4{state.position + dt * k3_r, state.velocity + dt * k3_v};
    const Vec2 k4_r = s4.velocity;
    const Vec2 k4_v = acceleration(s4, action, cfg);

    return {
        state.position + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r),
        state.velocity + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v),
    };
}

std::string choose_action(State chaser, State target, const SimConfig& cfg) {
    std::string best_action = "coast";
    double best_cost = std::numeric_limits<double>::infinity();

    for (const std::string& action : kActions) {
        const double cost = rollout_cost(chaser, target, action, cfg);
        if (cost < best_cost) {
            best_cost = cost;
            best_action = action;
        }
    }

    return best_action;
}

RunResult run_rendezvous(const SimConfig& cfg) {
    State target = circular_state(kEarthRadius + 500.0, 0.0);
    State chaser = circular_state(kEarthRadius + 485.0, -0.045);

    const int steps = static_cast<int>(cfg.duration / cfg.dt);
    const int decision_steps = std::max(1, static_cast<int>(cfg.decision_interval / cfg.dt));

    RunResult result;
    std::string action = "coast";
    double fuel_delta_v = 0.0;

    for (int step = 0; step <= steps; ++step) {
        if (step % decision_steps == 0) {
            action = choose_action(chaser, target, cfg);
        }

        const double distance = norm(chaser.position - target.position);
        const double relative_speed = norm(chaser.velocity - target.velocity);

        result.samples.push_back({
            step * cfg.dt / 60.0,
            distance,
            relative_speed,
            fuel_delta_v,
            chaser,
            target,
            action,
        });

        if (distance <= cfg.success_distance && relative_speed <= cfg.success_relative_speed) {
            result.success = true;
            break;
        }

        if (action != "coast") {
            fuel_delta_v += cfg.max_thrust_accel * cfg.dt;
        }

        chaser = rk4_step(chaser, action, cfg);
        target = rk4_step(target, "coast", cfg);
    }

    return result;
}

void write_csv(const RunResult& result, const std::string& path) {
    std::ofstream out(path);
    out << "time_min,distance_km,relative_speed_km_s,fuel_delta_v_m_s,"
        << "chaser_x_km,chaser_y_km,target_x_km,target_y_km,action\n";

    for (const Sample& sample : result.samples) {
        out << sample.time_min << ','
            << sample.distance_km << ','
            << sample.relative_speed_km_s << ','
            << sample.fuel_delta_v_km_s * 1000.0 << ','
            << sample.chaser.position.x << ','
            << sample.chaser.position.y << ','
            << sample.target.position.x << ','
            << sample.target.position.y << ','
            << sample.action << '\n';
    }
}

}  // namespace ai_space
