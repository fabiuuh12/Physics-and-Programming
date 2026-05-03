from __future__ import annotations

import numpy as np
import pytest

from rendezvous_env import EnvConfig, RendezvousEnv
from rendezvous_sim import EARTH_RADIUS, SimConfig, norm


SCENARIO_LIMITS = {
    "easy": {
        "target": (495.0, 505.0),
        "offset": (-20.0, -12.0),
        "phase": (-0.052, -0.038),
    },
    "medium": {
        "target": (490.0, 520.0),
        "offset": (-28.0, -10.0),
        "phase": (-0.065, -0.032),
    },
    "full": {
        "target": (480.0, 540.0),
        "offset": (-35.0, -8.0),
        "phase": (-0.08, -0.025),
    },
}


def assert_between(value: float, bounds: tuple[float, float]) -> None:
    low, high = bounds
    assert low <= value <= high


def test_reset_creates_valid_default_state() -> None:
    env = RendezvousEnv()
    observation = env.reset()

    assert observation.shape == (9,)
    assert env.elapsed == 0.0
    assert env.fuel_delta_v == 0.0
    assert norm(env.target_r) > EARTH_RADIUS
    assert norm(env.chaser_r) > EARTH_RADIUS
    assert env.previous_distance == pytest.approx(norm(env.chaser_r - env.target_r))


@pytest.mark.parametrize("difficulty", ("easy", "medium", "full"))
def test_randomized_scenarios_stay_in_expected_ranges(difficulty: str) -> None:
    env = RendezvousEnv()
    limits = SCENARIO_LIMITS[difficulty]

    for seed in range(20):
        scenario = env.sample_scenario(seed=seed, difficulty=difficulty)
        chaser_offset = scenario.chaser_altitude_km - scenario.target_altitude_km

        assert_between(scenario.target_altitude_km, limits["target"])
        assert_between(chaser_offset, limits["offset"])
        assert_between(scenario.chaser_angle_rad, limits["phase"])
        assert scenario.target_angle_rad == 0.0


def test_step_advances_time_and_only_thrust_actions_consume_fuel() -> None:
    env = RendezvousEnv()
    coast_index = env.action_names.index("coast")
    prograde_index = env.action_names.index("prograde")

    env.reset()
    env.step(coast_index)
    assert env.elapsed == pytest.approx(env.cfg.sim.dt)
    assert env.fuel_delta_v == 0.0

    env.reset()
    env.step(prograde_index)
    assert env.elapsed == pytest.approx(env.cfg.sim.dt)
    assert env.fuel_delta_v == pytest.approx(env.cfg.sim.max_thrust_accel * env.cfg.sim.dt)


def test_success_termination_when_chaser_matches_target() -> None:
    env = RendezvousEnv()
    env.reset()
    env.chaser_r = env.target_r.copy()
    env.chaser_v = env.target_v.copy()
    env.previous_distance = 0.0

    _, _, done, info = env.step(env.action_names.index("coast"))

    assert done
    assert info["success"]
    assert not info["unsafe_approach"]


def test_timeout_termination() -> None:
    cfg = EnvConfig(sim=SimConfig(duration=10.0))
    env = RendezvousEnv(cfg)
    env.reset()

    _, _, done, info = env.step(env.action_names.index("coast"))

    assert done
    assert info["timed_out"]
    assert not info["success"]


def test_fuel_empty_termination() -> None:
    cfg = EnvConfig(max_delta_v=1.0e-5)
    env = RendezvousEnv(cfg)
    env.reset()

    _, _, done, info = env.step(env.action_names.index("prograde"))

    assert done
    assert info["fuel_empty"]
    assert not info["success"]


def test_unsafe_approach_termination() -> None:
    env = RendezvousEnv()
    env.reset()
    env.chaser_r = env.target_r + np.array([0.25, 0.0])
    env.chaser_v = env.target_v + np.array([0.0, 0.03])
    env.previous_distance = norm(env.chaser_r - env.target_r)

    _, _, done, info = env.step(env.action_names.index("coast"))

    assert done
    assert info["unsafe_approach"]
    assert not info["success"]
