# arm_muscles_direct

Two-link arm driven **directly** by the six Lillicrap & Scott muscles (no
network), solved with iLQG, to compare the optimal control **with and without
the "muscles can only pull" constraint** on the WP1 center-out task.

| file | role |
|---|---|
| `wp1_pull_vs_pushpull.py` | **the WP1 iLQG itself** (`WP1/Controllers/ILQG.py`, muscle force commands, no activation dynamics, joint-space cost), wrapped so that `u = 8 sigma_u(v + v*)` (pull-only) or `u = 8·2e^-4 v` (push-pull = WP1 exactly, checked to 1e-8 cm with noise). WP1 center-out: 8 targets, 15 noisy trials + noise-free, delay 60 ms. Writes `wp1_sims.npz`, `wp1_table.txt`, `fig_wp1_pull_vs_pushpull.png` |
| `arm_muscles.py` | plant (arm, moment arms, F-L/V, activation of `../iLQG_Muscle.py`), the two activation laws, the static-equilibrium start, an iLQG whose control enters nonlinearly, `python arm_muscles.py` checks the Jacobians |
| `centerout_pull_vs_pushpull.py` | 16-target center-out, start (0, 40) cm, 15 cm, 0.4 s (as `WP1/Final_Code`); writes `centerout_sims.npz`, `centerout_table.txt`, `fig_centerout_pull_vs_pushpull.png` |

Activation laws (control `u`, one command per muscle):

    pull-only : a_dot = (sigma_u(u) - a) / tau_act
    push-pull : a_dot = (g u - a) / tau_act,   g = 2 e^-4

`sigma_u(x) = exp(2x - 5)` for `x < 1/2`, `2x exp(-4)` for `x >= 1/2`
(Lillicrap & Scott Supplemental, breakpoint 1/2) is strictly positive, so a
muscle can never push. `g` is the slope of its linear branch, which passes
through 0, so both laws have the same gain in the linear regime.

Both models start at rest: zero activation (push-pull), or the smallest command
`u*` whose resting activations `sigma_u(u*)` give zero net torque (pull-only;
`sigma_u(0)` alone does not). Effort cost `r1/2 |u - u*|^2`; terminal cost on
hand position and velocity. Deterministic: no motor noise, no sensory delay.

    python arm_muscles_direct/centerout_pull_vs_pushpull.py [--force]
