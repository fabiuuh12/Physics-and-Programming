# First Project: Autonomous Orbital Rendezvous

## Question

Can an autonomous agent learn useful maneuver choices for spacecraft rendezvous using only simulation feedback?

## Environment

Start with a planar two-body Earth orbit model.

The environment now supports both a fixed starter scenario and randomized scenarios. The randomized reset varies target altitude, chaser altitude offset, and initial phase angle.

Randomized scenarios are staged by difficulty:

- `easy`: small variation around the starter case
- `medium`: wider variation, but not the full spread
- `full`: the current widest randomized range

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
- `simulations/q_learning/q_policy_easy.json`
- `simulations/q_learning/q_policy_medium.json`
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

## Curriculum Check

Added staged randomized difficulties:

- `easy`: target altitude 495-505 km, chaser offset -20 to -12 km, phase angle -0.052 to -0.038 rad
- `medium`: target altitude 490-520 km, chaser offset -28 to -10 km, phase angle -0.065 to -0.032 rad
- `full`: target altitude 480-540 km, chaser offset -35 to -8 km, phase angle -0.080 to -0.025 rad

First easy-difficulty Q-learning run, before reward/state changes:

- command: `python3 python/q_learning.py --randomized --difficulty easy --episodes 1200 --eval-episodes 24`
- training successes: 5
- evaluation success rate: 1/24
- mean final distance: 82.58 km
- median final distance: 75.12 km
- best final distance: 4.98 km
- mean relative speed: 0.0929 km/s
- mean delta-v: 78.20 m/s

Easy policy comparison over the same 24 evaluation seeds:

- random: 0/24 successes, mean final distance 118.80 km
- greedy: 20/24 successes, mean final distance 4.91 km
- qlearn: 1/24 successes, mean final distance 82.58 km

This tells us the curriculum idea is useful as a diagnostic, but it does not solve the learning problem by itself. Q-learning is better than random on mean distance, but far worse than the greedy planner. Since greedy solves most easy cases, the environment is not the blocker. The likely blockers are the tabular state representation and reward shaping.

After improving the tabular state and reward:

- added closing-speed and relative-speed buckets
- replaced exact decision time with coarse mission-time buckets
- added a shaped decision reward that favors controlled closing and penalizes fast near-target approaches

New easy-difficulty Q-learning result:

- command: `python3 python/q_learning.py --randomized --difficulty easy --episodes 1200 --eval-episodes 24`
- evaluation success rate: 11/24
- mean final distance: 29.00 km
- median final distance: 5.60 km
- best final distance: 0.98 km
- mean relative speed: 0.0399 km/s
- mean delta-v: 86.59 m/s

Updated easy policy comparison over the same 24 evaluation seeds:

- random: 0/24 successes, mean final distance 118.80 km
- greedy: 20/24 successes, mean final distance 4.91 km
- qlearn: 11/24 successes, mean final distance 29.00 km

This is the first clear learning improvement. Q-learning is still behind greedy, but it moved from barely working to solving nearly half of the easy randomized cases.

Useful easy replay files after the reward/state update:

- `simulations/episode_viz/qlearn_easy_seed_10013.gif`: Q-learning success
- `simulations/episode_viz/qlearn_easy_seed_10011.gif`: Q-learning near miss, 9.85 km final range with low relative speed
- `simulations/episode_viz/greedy_easy_seed_10011.gif`: greedy success on the same failure seed

Next coding target:

- train and evaluate the Q table on `easy`, then move to `medium`, then `full`
- compare success rate, fuel use, and final miss distance against the greedy planner and random policy at each difficulty
- improve Q-learning state bins and reward shaping until it reliably beats random

Good later tools:

- Gymnasium
- Stable-Baselines3
- SciPy
- NumPy
- Matplotlib
