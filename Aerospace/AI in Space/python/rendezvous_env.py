from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from rendezvous_sim import (
    ACTIONS,
    EARTH_RADIUS,
    SimConfig,
    circular_state,
    norm,
    rk4_step,
)


Difficulty = Literal["easy", "medium", "full"]


@dataclass(frozen=True)
class EnvConfig:
    sim: SimConfig = SimConfig()
    max_delta_v: float = 0.120  # km/s, equal to 120 m/s
    unsafe_distance: float = 1.0
    distance_scale: float = 500.0
    speed_scale: float = 0.1


@dataclass(frozen=True)
class Scenario:
    target_altitude_km: float = 500.0
    chaser_altitude_km: float = 485.0
    target_angle_rad: float = 0.0
    chaser_angle_rad: float = -0.045


class RendezvousEnv:
    """Small RL-style environment for planar autonomous rendezvous."""

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.action_names = ACTIONS
        self.n_actions = len(self.action_names)
        self.reset()

    def sample_scenario(self, seed: int | None = None, difficulty: Difficulty = "full") -> Scenario:
        rng = np.random.default_rng(seed)

        if difficulty == "easy":
            target_altitude = float(rng.uniform(495.0, 505.0))
            chaser_offset = float(rng.uniform(-20.0, -12.0))
            phase_angle = float(rng.uniform(-0.052, -0.038))
        elif difficulty == "medium":
            target_altitude = float(rng.uniform(490.0, 520.0))
            chaser_offset = float(rng.uniform(-28.0, -10.0))
            phase_angle = float(rng.uniform(-0.065, -0.032))
        elif difficulty == "full":
            target_altitude = float(rng.uniform(480.0, 540.0))
            chaser_offset = float(rng.uniform(-35.0, -8.0))
            phase_angle = float(rng.uniform(-0.08, -0.025))
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        return Scenario(
            target_altitude_km=target_altitude,
            chaser_altitude_km=target_altitude + chaser_offset,
            target_angle_rad=0.0,
            chaser_angle_rad=phase_angle,
        )

    def reset(
        self,
        scenario: Scenario | None = None,
        *,
        randomize: bool = False,
        seed: int | None = None,
        difficulty: Difficulty = "full",
    ) -> np.ndarray:
        if scenario is not None and randomize:
            raise ValueError("Pass either scenario or randomize=True, not both")

        self.difficulty = difficulty
        self.scenario = scenario or (self.sample_scenario(seed, difficulty) if randomize else Scenario())
        self.target_r, self.target_v = circular_state(
            EARTH_RADIUS + self.scenario.target_altitude_km,
            self.scenario.target_angle_rad,
        )
        self.chaser_r, self.chaser_v = circular_state(
            EARTH_RADIUS + self.scenario.chaser_altitude_km,
            self.scenario.chaser_angle_rad,
        )
        self.elapsed = 0.0
        self.fuel_delta_v = 0.0
        self.previous_distance = norm(self.chaser_r - self.target_r)
        return self.observation()

    def observation(self) -> np.ndarray:
        relative_r = self.chaser_r - self.target_r
        relative_v = self.chaser_v - self.target_v
        fuel_remaining = max(0.0, self.cfg.max_delta_v - self.fuel_delta_v)

        return np.array(
            [
                relative_r[0] / self.cfg.distance_scale,
                relative_r[1] / self.cfg.distance_scale,
                relative_v[0] / self.cfg.speed_scale,
                relative_v[1] / self.cfg.speed_scale,
                self.chaser_r[0] / (EARTH_RADIUS + 500.0),
                self.chaser_r[1] / (EARTH_RADIUS + 500.0),
                self.target_r[0] / (EARTH_RADIUS + 500.0),
                self.target_r[1] / (EARTH_RADIUS + 500.0),
                fuel_remaining / self.cfg.max_delta_v,
            ],
            dtype=np.float64,
        )

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if action_index < 0 or action_index >= self.n_actions:
            raise ValueError(f"action_index must be in [0, {self.n_actions - 1}]")

        action = self.action_names[action_index]
        sim = self.cfg.sim

        if action != "coast":
            self.fuel_delta_v += sim.max_thrust_accel * sim.dt

        self.chaser_r, self.chaser_v = rk4_step(self.chaser_r, self.chaser_v, action, sim)
        self.target_r, self.target_v = rk4_step(self.target_r, self.target_v, "coast", sim)
        self.elapsed += sim.dt

        distance = norm(self.chaser_r - self.target_r)
        relative_speed = norm(self.chaser_v - self.target_v)
        distance_progress = self.previous_distance - distance
        self.previous_distance = distance

        success = distance <= sim.success_distance and relative_speed <= sim.success_relative_speed
        earth_collision = norm(self.chaser_r) <= EARTH_RADIUS
        fuel_empty = self.fuel_delta_v >= self.cfg.max_delta_v
        timed_out = self.elapsed >= sim.duration
        unsafe_approach = distance <= self.cfg.unsafe_distance and not success

        reward = 0.25 * distance_progress
        reward -= 8.0 * relative_speed
        if action != "coast":
            reward -= 0.05

        if success:
            reward += 100.0
        if unsafe_approach:
            reward -= 50.0
        if earth_collision:
            reward -= 100.0
        if fuel_empty and not success:
            reward -= 20.0
        if timed_out and not success:
            reward -= 10.0

        done = success or unsafe_approach or earth_collision or fuel_empty or timed_out
        info = {
            "action": action,
            "distance_km": distance,
            "relative_speed_km_s": relative_speed,
            "fuel_delta_v_km_s": self.fuel_delta_v,
            "elapsed_s": self.elapsed,
            "success": success,
            "unsafe_approach": unsafe_approach,
            "earth_collision": earth_collision,
            "fuel_empty": fuel_empty,
            "timed_out": timed_out,
            "scenario": self.scenario,
        }

        return self.observation(), reward, done, info
