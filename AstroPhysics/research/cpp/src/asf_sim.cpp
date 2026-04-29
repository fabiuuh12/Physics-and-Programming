#include "asf_sim.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace asf {

namespace {

double number_after(const std::string &text, const std::string &key, double fallback) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
        return std::stod(match[1].str());
    }
    return fallback;
}

int int_after(const std::string &text, const std::string &key, int fallback) {
    return static_cast<int>(number_after(text, key, static_cast<double>(fallback)));
}

std::string string_after(const std::string &text, const std::string &key, const std::string &fallback) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
        return match[1].str();
    }
    return fallback;
}

double clamp_floor(double value, double floor) {
    return std::max(value, floor);
}

}  // namespace

Grid::Grid(int nx_value, int ny_value, double fill) : nx(nx_value), ny(ny_value), values(nx_value * ny_value, fill) {}

double &Grid::operator()(int x, int y) {
    return values[static_cast<std::size_t>(y * nx + x)];
}

double Grid::operator()(int x, int y) const {
    return values[static_cast<std::size_t>(y * nx + x)];
}

Config load_config(const std::string &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open config: " + path);
    }
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    Config config;
    config.experiment_name = string_after(text, "experiment_name", config.experiment_name);
    config.seed = int_after(text, "seed", config.seed);
    config.nx = int_after(text, "nx", config.nx);
    config.ny = int_after(text, "ny", config.ny);
    config.dx = number_after(text, "dx", config.dx);
    config.dy = number_after(text, "dy", config.dy);
    config.G = number_after(text, "G", config.G);
    config.mu0 = number_after(text, "mu0", config.mu0);
    config.rho0 = number_after(text, "rho0", config.rho0);
    config.omega0 = number_after(text, "omega0", config.omega0);
    config.temperature0 = number_after(text, "temperature0", config.temperature0);
    config.length0 = number_after(text, "length0", config.length0);
    config.lambda_A = number_after(text, "lambda_A", config.lambda_A);
    config.chi_B = number_after(text, "chi_B", config.chi_B);
    config.chi_omega = number_after(text, "chi_omega", config.chi_omega);
    config.chi_T = number_after(text, "chi_T", config.chi_T);
    config.pressure_floor = number_after(text, "pressure_floor", config.pressure_floor);
    config.rho_floor = number_after(text, "rho_floor", config.rho_floor);
    config.iterations = int_after(text, "iterations", config.iterations);
    config.tolerance = number_after(text, "tolerance", config.tolerance);
    config.smooth_passes = int_after(text, "smooth_passes", config.smooth_passes);
    return config;
}

FieldState build_controlled_fields(const Config &config, bool structured) {
    FieldState state{
        Grid(config.nx, config.ny), Grid(config.nx, config.ny), Grid(config.nx, config.ny),
        Grid(config.nx, config.ny), Grid(config.nx, config.ny), Grid(config.nx, config.ny),
        Grid(config.nx, config.ny), Grid(config.nx, config.ny)};

    for (int y = 0; y < config.ny; ++y) {
        const double yy = -1.0 + 2.0 * static_cast<double>(y) / static_cast<double>(config.ny - 1);
        for (int x = 0; x < config.nx; ++x) {
            const double xx = -1.0 + 2.0 * static_cast<double>(x) / static_cast<double>(config.nx - 1);
            const double r2 = xx * xx + yy * yy;
            state.rho(x, y) = 0.2 + 2.0 * std::exp(-r2 / 0.12) +
                              0.45 * std::exp(-((xx - 0.28) * (xx - 0.28) + (yy + 0.18) * (yy + 0.18)) / 0.035);
            state.pressure(x, y) = 0.8 + 0.3 * std::exp(-r2 / 0.25);
            state.temperature(x, y) = 1.0 + 0.2 * std::exp(-((xx + 0.18) * (xx + 0.18) + (yy - 0.22) * (yy - 0.22)) / 0.08);
            if (structured) {
                const double phase = 3.0 * M_PI * xx + 2.0 * M_PI * yy;
                state.bx(x, y) = 0.10 + 0.08 * std::sin(phase);
                state.by(x, y) = 0.08 * std::cos(2.0 * M_PI * xx);
                state.vx(x, y) = -yy * std::exp(-r2 / 0.35);
                state.vy(x, y) = xx * std::exp(-r2 / 0.35);
                state.temperature(x, y) += 0.12 * std::sin(4.0 * M_PI * xx) * std::cos(3.0 * M_PI * yy);
            } else {
                state.bx(x, y) = 0.1;
                state.by(x, y) = 0.0;
                state.vx(x, y) = 0.0;
                state.vy(x, y) = 0.0;
            }
        }
    }
    state.structure = compute_structure_field(state, config);
    return state;
}

Grid compute_structure_field(const FieldState &state, const Config &config) {
    Grid result(config.nx, config.ny);
    auto grad_t = gradient(state.temperature, config.dx, config.dy);
    for (int y = 0; y < config.ny; ++y) {
        for (int x = 0; x < config.nx; ++x) {
            const double rho = clamp_floor(state.rho(x, y), config.rho_floor);
            const double pressure = clamp_floor(state.pressure(x, y), config.pressure_floor);
            const double density_term = std::log(rho / config.rho0);
            const double magnetic = (state.bx(x, y) * state.bx(x, y) + state.by(x, y) * state.by(x, y)) /
                                    (2.0 * config.mu0 * pressure);
            const int xm = std::max(0, x - 1);
            const int xp = std::min(config.nx - 1, x + 1);
            const int ym = std::max(0, y - 1);
            const int yp = std::min(config.ny - 1, y + 1);
            const double dvy_dx = (state.vy(xp, y) - state.vy(xm, y)) / (static_cast<double>(xp - xm) * config.dx);
            const double dvx_dy = (state.vx(x, yp) - state.vx(x, ym)) / (static_cast<double>(yp - ym) * config.dy);
            const double vort = dvy_dx - dvx_dy;
            const double vort_term = vort * vort / (config.omega0 * config.omega0);
            const double thermal = config.length0 * config.length0 *
                                   (grad_t.first(x, y) * grad_t.first(x, y) + grad_t.second(x, y) * grad_t.second(x, y)) /
                                   (config.temperature0 * config.temperature0);
            result(x, y) = density_term + config.chi_B * magnetic + config.chi_omega * vort_term + config.chi_T * thermal;
        }
    }
    return smooth_box(result, config.smooth_passes);
}

Grid smooth_box(const Grid &field, int passes) {
    Grid current = field;
    for (int pass = 0; pass < passes; ++pass) {
        Grid next(field.nx, field.ny);
        for (int y = 0; y < field.ny; ++y) {
            for (int x = 0; x < field.nx; ++x) {
                double sum = 0.0;
                int count = 0;
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        const int sx = std::clamp(x + ox, 0, field.nx - 1);
                        const int sy = std::clamp(y + oy, 0, field.ny - 1);
                        sum += current(sx, sy);
                        ++count;
                    }
                }
                next(x, y) = sum / static_cast<double>(count);
            }
        }
        current = next;
    }
    return current;
}

Grid laplacian(const Grid &field, double dx, double dy) {
    Grid out(field.nx, field.ny);
    for (int y = 1; y < field.ny - 1; ++y) {
        for (int x = 1; x < field.nx - 1; ++x) {
            out(x, y) = (field(x + 1, y) - 2.0 * field(x, y) + field(x - 1, y)) / (dx * dx) +
                        (field(x, y + 1) - 2.0 * field(x, y) + field(x, y - 1)) / (dy * dy);
        }
    }
    return out;
}

std::pair<Grid, Grid> gradient(const Grid &field, double dx, double dy) {
    Grid gx(field.nx, field.ny);
    Grid gy(field.nx, field.ny);
    for (int y = 0; y < field.ny; ++y) {
        for (int x = 0; x < field.nx; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(field.nx - 1, x + 1);
            const int ym = std::max(0, y - 1);
            const int yp = std::min(field.ny - 1, y + 1);
            gx(x, y) = (field(xp, y) - field(xm, y)) / (static_cast<double>(xp - xm) * dx);
            gy(x, y) = (field(x, yp) - field(x, ym)) / (static_cast<double>(yp - ym) * dy);
        }
    }
    return {gx, gy};
}

std::pair<Grid, std::vector<double>> poisson_jacobi(const Grid &source, const Config &config) {
    Grid phi(source.nx, source.ny);
    std::vector<double> residuals;
    const double dx2 = config.dx * config.dx;
    const double dy2 = config.dy * config.dy;
    const double denom = 2.0 * (dx2 + dy2);
    for (int iteration = 0; iteration < config.iterations; ++iteration) {
        Grid old = phi;
        for (int y = 1; y < source.ny - 1; ++y) {
            for (int x = 1; x < source.nx - 1; ++x) {
                phi(x, y) = (dy2 * (old(x + 1, y) + old(x - 1, y)) +
                             dx2 * (old(x, y + 1) + old(x, y - 1)) -
                             source(x, y) * dx2 * dy2) /
                            denom;
            }
        }
        const double residual = poisson_residual(phi, source, config.dx, config.dy);
        residuals.push_back(residual);
        if (residual <= config.tolerance) {
            break;
        }
    }
    return {phi, residuals};
}

double poisson_residual(const Grid &phi, const Grid &source, double dx, double dy) {
    Grid diff = laplacian(phi, dx, dy);
    double sum = 0.0;
    int count = 0;
    for (int y = 1; y < phi.ny - 1; ++y) {
        for (int x = 1; x < phi.nx - 1; ++x) {
            const double r = diff(x, y) - source(x, y);
            sum += r * r;
            ++count;
        }
    }
    return std::sqrt(sum / static_cast<double>(std::max(1, count)));
}

Grid effective_potential(const Grid &phi_n, const Grid &structure, const Config &config) {
    Grid out(phi_n.nx, phi_n.ny);
    for (int y = 0; y < phi_n.ny; ++y) {
        for (int x = 0; x < phi_n.nx; ++x) {
            out(x, y) = phi_n(x, y) + config.lambda_A * structure(x, y);
        }
    }
    return out;
}

std::pair<Grid, Grid> acceleration(const Grid &phi, double dx, double dy) {
    auto grad = gradient(phi, dx, dy);
    for (double &value : grad.first.values) {
        value = -value;
    }
    for (double &value : grad.second.values) {
        value = -value;
    }
    return grad;
}

void write_csv(const std::string &path, const FieldState &state, const Grid &phi_n, const Grid &phi_eff) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not write CSV: " + path);
    }
    output << "x,y,rho,structure,phi_n,phi_eff\n";
    for (int y = 0; y < state.rho.ny; ++y) {
        for (int x = 0; x < state.rho.nx; ++x) {
            output << x << ',' << y << ',' << state.rho(x, y) << ',' << state.structure(x, y) << ','
                   << phi_n(x, y) << ',' << phi_eff(x, y) << '\n';
        }
    }
}

}  // namespace asf
