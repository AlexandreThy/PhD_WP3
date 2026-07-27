# Nonlinear-arm + random-network motor control (iLQG)

A reimplementation of the combined **random recurrent network + biomechanical
limb** model of Kalidindi & Crevecoeur (2025), but with a **nonlinear two-link
arm** and an **iLQG** controller (instead of the paper's linear plant + LQG).
The network transition matrix `J` and readout `C` are taken from the trained
Gaussian networks in `gaussian_networks.hdf5` (spectral radius 0.8, network #2).

## Model

Combined state `x = [r ; q]` (dimension `N+6`, here `N = 100`):

* `r ∈ R^N` — network firing rates. Linear dynamics `ṙ = J·r + u`, where the
  loaded `W` **is** the paper's stable transition matrix `J = (W_adj − I)/τ_net`
  (τ_net ≈ 20 ms). The control `u` acts on every node.
* `q = [θ_s, θ_e, ω_s, ω_e, τ_s, τ_e]` — two-link arm: shoulder/elbow angles,
  velocities and joint torques. Nonlinear rigid-body dynamics with
  configuration-dependent inertia `M(θ_e)`, Coriolis/centripetal terms, and
  viscous joint friction.
* Coupling: the network readout drives the muscle/actuation dynamics,
  `τ̇ = (C·r − τ)/τ_act` with `C = Wout`, `τ_act = 60 ms`.

Control is solved by **iLQG** over a finite horizon that **starts at movement
onset** (no preparatory epoch, per request). Cost = terminal joint-angle error
to target + terminal joint velocity + quadratic control effort. The network
starts at rest (`r = 0`).

## Files

| file | purpose |
|------|---------|
| `iLQG_Combined.py` | dynamics `f`, analytic Jacobians `fx`/`fu`, iLQG passes (`step1`–`step4`), `simulate_ILQG`. Also a `LINEARIZE` flag that freezes the arm to a linear plant (baseline). |
| `hdf.py` | loads `J` (=`W`) and `C` (=`Wout`) from `gaussian_networks.hdf5`. |
| `centerout.py` | solves the N-target center-out task and caches trajectories. `--linear` uses the linear-arm baseline. |
| `figures.py` | reproduces Fig 2a (kinematics), 2b (node activity), 2d (movement-epoch PCA / rotations). |
| `analysis.py` | preferred-direction distributions (Lillicrap & Scott 2013 style) and nonlinearity probes. |
| `iLQG_Muscle.py` | the same combined model with the **joint-torque actuator replaced by the six lumped muscles** of Lillicrap & Scott (2013), plus the abstraction switches for their Figure 5 ladder. |
| `fig5_centreout.py` | solves the ladder and reproduces the centre-out row of their Figure 5 with our iLQG methodology. |

## Muscle model (`iLQG_Muscle.py`)

The actuator block `a` is now muscle *activation* rather than joint torque:

```
a_dot = (sigma_u(Wout . r) - a) / tau_act        muscles can only pull
l     = 1 + sum_j M[j,i] (theta0[j,i] - theta_j) / L0[i]     (their Eq 2)
tau   = M . (F_max * a * f_l(l) * f_fv(l, l_dot))            (their Eq 1, 6, 7)
```

with the paper's moment-arm matrix, optimal lengths/angles and force-length /
force-velocity parameters (their Supplemental Eq. 1-7). The force-length and
force-velocity curves are evaluated **directly from their Eq 6 and Eq 7** --
each branch of Eq 7 only on its own side of zero velocity, so the lengthening
branch never approaches its pole at `l_dot = -b_V` (which lies inside the
velocity range our reaches visit).

The paper additionally replaced the product `f_l * f_fv` with a fitted
5-hidden-node sigmoidal network to remove the derivative kink at zero velocity.
That step is **not** taken here: the plant sees the equations themselves. iLQG
converges the same either way (10-14 iterations per reach) and the preferred
directions differ by ~1 degree, so the kink is not a practical problem at this
step size. `FV_EPS > 0` cross-fades the two branches over `|l_dot| < FV_EPS` if
a `C^1` plant is ever wanted; `FV_EPS = 0` (the default) is the paper verbatim.

Two deviations remain, both forced by errors in the printed supplement:

* their `sigma_u` breakpoint is garbled; we use the value (`1/2`) that makes
  the two branches match in value *and* slope,
* their force-length equation as printed drops the minus sign and the exponent
  `rho`; we use Brown's `exp(-|(l^beta-1)/omega|^rho)`. `rho = 2.12` is listed
  among their parameters but appears nowhere in the equation as typeset, and
  the printed form `exp(+|...|)` would make force *minimal* at optimal length
  and grow without bound at the extremes.

The arm itself (segment lengths, masses, inertias) is unchanged from
`iLQG_Combined.py`, so it is longer than the monkey arm the muscle parameters
were measured on; `Plant(..., recenter_lengths=True)` shifts the optimal angles
so every muscle sits at `l = 1` at the centre posture, as a robustness check.

Two cost changes relative to `iLQG_Combined.py`, both required by the ladder:

* the terminal cost is on **hand** position/velocity (Gauss-Newton residual
  form) rather than joint angles -- otherwise "removing limb geometry" would
  have no effect and rung 1 of the ladder would be meaningless;
* a running penalty on lateral deviation from the start->target line (`qp`),
  which is the term the paper adds "in place of" the kinematic error during the
  reach. Without it the effort-optimal handpaths bow by ~5-7% of the reach
  length; `qp = 0.5` brings that to ~0.3% with no change to the bell-shaped
  speed profile. `qp = 0` reproduces the free-path solution
  (`fig5_ladder_freepath.npz`), and the preferred-direction result is
  essentially the same either way.

`python iLQG_Muscle.py` finite-difference-checks every analytic Jacobian in the
ladder (relative error ~1e-10).

## Reproducing Figure 5, centre-out row

```bash
python fig5_centreout.py run --n_net 10 --jobs 7   # 960 reaches, ~5 min
python fig5_centreout.py figure                    # stats + PNGs
```

Six plants, matching the paper's ladder: 2-D point mass -> + geometry ->
+ intersegmental dynamics -> + monoarticular muscles -> + biarticular muscles
-> + force-length/velocity. Ten instantiations each (a different recurrent
network `W` and a fresh random readout `Wout`), 16 targets, so 1,000 unit
preferred directions per plant. Unit activity is averaged over the epoch the
paper used for real neurons (movement onset, i.e. 10% of peak speed, to peak
hand speed) and fitted by the same planar regression; only significant fits
(`p<0.05`) enter the polar histograms. Outputs `fig5_centreout.png` (unit
PMDs), `fig5_muscles.png` (muscle PMDs) and `fig5_behaviour.png` (handpaths and
speed profiles -- the paper's precondition for the analysis to mean anything).

## Reproduce

```bash
python centerout.py --n_dir 8                                   # Fig 2 data
python figures.py                                               # Fig 2a/2b/2d PNGs

python centerout.py --n_dir 24 --out centerout_nl24.npz --tol 1e-4
python centerout.py --n_dir 24 --out centerout_lin24.npz --tol 1e-4 --linear
python analysis.py                                             # PD + nonlinearity PNGs
```

## Key fixes to the original `iLQG_Combined.py`

* `fx`: added the missing torque-row Jacobian terms — actuation self-decay
  `∂τ̇/∂τ = −1/τ_act` and the correctly scaled readout coupling
  `∂τ̇/∂r = Wout/τ_act`. Verified against finite differences (err ~1e-9).
* `fu`: fixed the call signature used by `step2`.
* `step3`: symmetric (`eigh`) regularized inverse of the control Hessian, and a
  fast deterministic path that skips the multiplicative-noise loop (which would
  otherwise run `N` times per step) and exploits the structure of `B`.
* `simulate_ILQG`: rewritten for the full `N+6` combined state (was initialising
  only a 4-D arm state with a 6-D control); network control `u ∈ R^N`; nonlinear
  forward pass with backtracking line search; returns the optimal nominal
  trajectory.

## Traces of nonlinearity

Because the network is linear, all nonlinearity in the neural activity is
inherited from the arm through the optimal feedback. Compared with the
linearized-arm baseline (same network, same targets), the nonlinear arm yields:

* direction-dependent asymmetry of opposite reaches in network space,
* a measurable 2nd harmonic in single-node directional tuning (pure cosine for a
  linear plant),
* a biased (bimodal) preferred-direction distribution, as in Lillicrap & Scott.
