from __future__ import annotations

import random

from rendezvous_env import RendezvousEnv


def main() -> None:
    env = RendezvousEnv()
    rng = random.Random(7)
    env.reset()

    total_reward = 0.0
    done = False
    info = {}
    steps = 0

    while not done:
        action_index = rng.randrange(env.n_actions)
        _, reward, done, info = env.step(action_index)
        total_reward += reward
        steps += 1

    print("Random policy rendezvous baseline")
    print(f"success: {info['success']}")
    print(f"steps: {steps}")
    print(f"simulated time: {info['elapsed_s'] / 60.0:.1f} min")
    print(f"final distance: {info['distance_km']:.2f} km")
    print(f"final relative speed: {info['relative_speed_km_s']:.4f} km/s")
    print(f"fuel proxy, delta-v used: {info['fuel_delta_v_km_s'] * 1000.0:.2f} m/s")
    print(f"total reward: {total_reward:.2f}")
    print(
        "termination:"
        f" timed_out={info['timed_out']}"
        f" fuel_empty={info['fuel_empty']}"
        f" unsafe_approach={info['unsafe_approach']}"
        f" earth_collision={info['earth_collision']}"
    )


if __name__ == "__main__":
    main()
