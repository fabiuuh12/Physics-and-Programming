# First Project: Autonomous Orbital Rendezvous

## Question

Can an autonomous agent learn useful maneuver choices for spacecraft rendezvous using only simulation feedback?

## Environment

Start with a planar two-body Earth orbit model.

The environment now supports both a fixed starter scenario and randomized scenarios. The randomized reset varies target altitude, chaser altitude offset, and initial phase angle.

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
- `python/q_learning.py` trains a first tabular Q-learning policy.
- `python/episode_viz.py` exports GIF replays for greedy, random, and Q-learning policies.

The first Q-learning policy is warm-started from the greedy planner. Treat it as the starting scaffold for learning experiments, not as proof that tabular Q-learning has independently solved rendezvous yet.

The fixed-scenario and randomized-scenario Q policies are stored separately so replays use the right table:

- `simulations/q_learning/q_policy_fixed.json`
- `simulations/q_learning/q_policy_randomized.json`

## Observation From Randomized Replay

The Q-learning randomized replay for seed `10024` is the first case where the learned table looks like it is doing something useful instead of only copying the fixed scenario.

Result for `qlearn_randomized_seed_10024.gif`:

- success: true
- simulated time: 91.3 min
- final distance: 4.95 km
- final relative speed: 0.0146 km/s
- delta-v used: 76.80 m/s
- action counts: coast 165, radial-in 300, retrograde 30, radial-out 54

The important visual behavior is that the chaser catches up to the target path instead of just diving inward. The radial-in actions appear to help phase the orbit, while radial-out and retrograde actions shape the late approach. That is a promising sign because it looks closer to reusable rendezvous behavior.

But the full randomized evaluation is still weak:

- randomized Q-learning evaluation: 1/24 successes
- mean final distance: 136.33 km
- best final distance: 4.95 km
- mean relative speed: 0.1523 km/s
- mean delta-v: 84.20 m/s

So seed `10024` is evidence that the setup can work, not evidence that the policy is robust yet. The next learning problem is to make more seeds behave like `10024`.

Next coding target:

- compare success rate, fuel use, and final miss distance against the greedy planner and random policy
- improve Q-learning state bins and reward shaping until it reliably beats random
- expand randomized starting conditions so the agent learns a more general rendezvous strategy

Good later tools:

- Gymnasium
- Stable-Baselines3
- SciPy
- NumPy
- Matplotlib
