# First Project: Autonomous Orbital Rendezvous

## Question

Can an autonomous agent learn useful maneuver choices for spacecraft rendezvous using only simulation feedback?

## Environment

Start with a planar two-body Earth orbit model.

State:

- spacecraft position and velocity
- target position and velocity
- relative position
- relative velocity
- fuel remaining

Actions:

- coast
- thrust prograde
- thrust retrograde
- thrust radial outward
- thrust radial inward

Physics:

```text
a = -mu * r / |r|^3 + thrust_acceleration
```

## Success

The spacecraft succeeds if it reaches a small distance from the target with low relative velocity before fuel or time runs out.

## First AI Method

Begin with a simple planner or Q-learning agent before moving to neural reinforcement learning.

Current baseline:

- `python/rendezvous_sim.py` runs a greedy lookahead planner.
- `python/rendezvous_env.py` exposes the same physics as `reset()` and `step(action_index)`.
- `python/random_policy.py` runs a random baseline through the environment.
- `python/episode_viz.py` exports a GIF replay of the greedy rendezvous episode.

Next coding target:

- discretize relative distance, relative speed, and fuel remaining
- train a small Q-learning agent
- compare success rate, fuel use, and final miss distance against the greedy planner and random policy

Good later tools:

- Gymnasium
- Stable-Baselines3
- SciPy
- NumPy
- Matplotlib
