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
