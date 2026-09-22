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
inherited from the arm through the iLQG backward pass. Compared with the
linearized-arm baseline (same network, same targets), the nonlinear arm yields:

* **opposite-direction asymmetry** in network space: 0.36 vs 0.22 for the linear
  baseline (peak of `opposite_asymmetry`) — a difference of degree, not of kind;
* **~30 % of the network activity reshaped** (`node_divergence`), which is a
  large effect;
* a **PD distribution that only separates for large, fast reaches** (see below).

### What the linear baseline does *not* remove (important)

`LINEARIZE` freezes inertia at `M(THETA2_REF)` and drops Coriolis. It does **not**
make the arm isotropic and does **not** linearize the task:

* `M(THETA2_REF)` is still anisotropic (eigenvalue ratio ≈ 4.6);
* the hand Jacobian `J` is untouched — pure kinematics, identical in both models;
* `xtarg(θ) = IK(center + R·[cos θ, sin θ])` is nonlinear in **both** models.

`fx` is block-triangular (the arm never enters `ṙ = W r + u`), so for the linear
arm the optimal activity is *affine* in the joint target:
`r_i(θ) ≈ c_i + G[i,:]·[cos θ, sin θ]` with `G = pinv(Wout) · M · inv(J)`.
The PD distribution is then the pushforward of the readout directions through the
anisotropic map `J⁻ᵀM` — built from kinematics and frozen inertia alone. That
closed form predicts the observed bimodal axis to **within ~2°** with no
nonlinearity whatsoever (−28.2° predicted vs −30.2° nonlinear / −29.6° linear).

**So bimodal PDs and a nonzero 2nd harmonic are not signatures of nonlinearity
here** — the linear baseline shows both. At 12 cm / 0.5 s: bimodal `r` = 0.67
(linear) vs 0.68 (nonlinear); population `P2/P1` = 0.018 vs 0.019.

### Why the PD is nearly blind to the nonlinearity

Coriolis `C ∝ v²` is invariant under reversing the movement, so `C(θ+π) ≈ C(θ)`:
it is nearly **even** in reach direction (measured ~4× more even than odd), and
so is the inertia-variation term to leading order. Even functions have only even
harmonics. But `PD = arctan2(b1, a1)` is read off the **first** harmonic, and on
a uniform direction grid `cos/sin(θ)` are *exactly* orthogonal to `cos/sin(2θ)`.
A 2nd harmonic 10× the size of the 1st shifts the fitted PD by **exactly zero**.
Hence: 30 % of the activity reshaped, ~2.6° median PD shift.

### jPCA rotations, and why they don't look like Kalidindi & Crevecoeur Fig 2F

`figures.jpca` / `figures.fig_jpca_rotations` implement jPCA to the Kalidindi &
Crevecoeur (2026) STAR-Methods recipe (their Fig 2F/2J): soft-normalise by
`range + 0.005`, PCA to the **top 6 movement-epoch PCs over the first 300 ms**,
fit `dX/dt = M X` with `M` skew-symmetric, project onto the top eigenpair's plane.
(We also subtract the cross-condition mean, as in the Churchland code they cite;
it changes little here.) On 8 targets, 20 cm / 0.35 s (`fig_jpca_rotations.png`):

| | rotation | turns / target | skew-fit R² | unconstrained R² |
|---|---|---|---|---|
| nonlinear | 2.92 Hz | 1.05 | 0.586 | 0.811 |
| linear | 2.93 Hz | 1.01 | 0.591 | 0.804 |

So **one rotation per target is present**, with a skew-fit R² (~0.59) in the same
ballpark the paper reports for model and M1 (their Fig 2J). Note the two plants
are indistinguishable here too (2.92 vs 2.93 Hz), like every other
population-geometry measure in this file.

**But the geometry is a rosette, not the paper's spatially separated loops.**
Every condition starts at **exactly `r = 0`** (verified: `max |r(t=0)| = 0.0`;
jPC start radius ~1e-17), so all targets launch from one point and spiral outward
instead of occupying different regions of the plane.

That separation *is* preparatory activity, and this model has none by
construction — the horizon starts at movement onset and `simulate_ILQG` defaults
`r0 = np.zeros(N)` for every target. **This is a property of the model, not of
the analysis**: no jPCA variant can recover a spread that isn't in the data.

The paper supplies the exact fix — its prepare-to-reach cost (their Eqs. 19-20):
a two-phase running cost that penalises deviation from the **start** over a short
prep horizon (`Δh_prep = 50 ms`), then from the **target** over the move horizon.
This builds a target-dependent preparatory state — necessarily in the
**output-null subspace** (`Wout·r ≈ 0`, Kaufman et al. 2014) so the arm stays put
— which offsets each condition's loop and yields the separated-loop picture. It
is a real modelling change (a two-phase cost + re-solve), not a plotting tweak.

### Does the nonlinear plant need more dimensions? (yes — but not variance ones)

**By variance-based measures, no.** Effective rank 2.51 (nonlinear) vs 2.46
(linear); participation ratio 1.96 vs 1.92; dimensions for 80/95/99 % variance
identical (2/3/5). PCA says the two controllers are the same size.

**By the rank of the condition-dependence, decisively yes.** `fx` is
block-triangular (the arm never enters `ṙ = W r + u`), so for a linear plant with
quadratic cost the optimal control is *affine in the target* — and therefore

```
r(θ, t) = c(t) + G(t)·xtarg(θ)        =>   rank( r[:, t, :] ) ≤ 1 + 2 = 3
```

**at every time, for any number of targets.** This is a theorem, not a fit.
Measured (`analysis.target_affine_r2`, `analysis.condition_rank`):

| | affine-in-target R² | median across-direction rank |
|---|---|---|
| linear, 20 cm / 0.35 s | **0.99999982** | 4 |
| nonlinear, 20 cm / 0.35 s | 0.97519534 | **15** |
| linear, 12 cm / 0.50 s | 0.99999994 | 4 |
| nonlinear, 12 cm / 0.50 s | 0.98849153 | 15 |

(the linear 4th singular value sits at ~1e-4 = the iLQG `tol`, i.e. solver
residue). The cap is real: subsampling targets, the **linear rank stays at 4**
for 6/8/12/24 targets, while the **nonlinear rank tracks the number of
conditions** (6/8/12/15) — it is not a fixed number but an uncapped one.

The escape from the rank-3 model **grows with the regime** (1.2 % of variance at
12 cm/0.5 s → 2.5 % at 20 cm/0.35 s) and is **88.8 % direction-EVEN** (vs 5.7 %
for the raw activity). So the extra dimensions *are* the interaction-torque
subspace — the same one the PD cannot see and `predictions.term_dr2(even=True)`
targets. They carry only ~2.5 % of the variance, which is exactly why PCA
effective rank is blind to them.

> **Interpretation.** The plant is 2-DOF in both cases; what needs more
> dimensions is the *representation of the optimal policy as a function of task
> condition*. A linear plant's policy compresses into a rank-3 affine map
> forever; a nonlinear plant's does not compress at all. Note this also means
> **all** of the linear model's apparent directional nonlinearity (2nd harmonic,
> bimodal PDs) comes from the IK curve `xtarg(θ)` — not from the controller.

### Where the difference sits in PC space

`figures.pca_divergence` projects both runs into a **common** basis (pooled PCs
and mean — legitimate because the network is node-matched across runs) and asks
where the nonlinear/linear difference lives. At 20 cm / 0.35 s
(`fig_pca_nl_vs_lin.png`):

| | PC1 | PC2 | PC3 | PC4 | PC5 |
|---|---|---|---|---|---|
| variance explained | 67.4 % | 24.2 % | 5.2 % | 1.5 % | 1.1 % |
| share of the nl−lin difference | 22.5 % | 34.3 % | **30.0 %** | 4.0 % | 5.1 % |
| relative divergence ‖nl−lin‖/‖nl‖ | 13 % | 25 % | **53 %** | 35 % | 45 % |
| direction-**even** share, nonlinear | 3.2 % | 7.8 % | **21.5 %** | 10.6 % | 15.7 % |
| direction-**even** share, linear | 6.0 % | 6.6 % | **8.3 %** | 6.7 % | 6.3 % |

Read this carefully — the naive summaries all say "no difference":

* the two models' **variance spectra are near-identical** (67/24/5 vs 68/24/5),
* their **top-3 subspaces coincide** (principal angles 0.8°/1.6°/2.8°),
* effective rank 2.51 vs 2.46.

Yet **87 % of the difference lies inside PC1–3**, concentrated in **PC3**: 5 % of
the variance but 30 % of the difference and a 53 % relative divergence. PC3 is
where the nonlinearity injects **direction-even** structure — it triples its even
content (8.3 % → 21.5 %) — while PC1–2 stay dominated by the shared odd /
direction-tuned component. Relative divergence *grows* with PC index (PC5–6 also
35–46 %), but those components are too small to matter.

The cleanest single readout is the **condition-invariant (direction-averaged)
PC3 time course**, which *flips sign* between the plants in the common basis:
correlation with v² is **+0.57 (nonlinear) vs −0.59 (linear)**. The signature is
present but weaker at the default reach (PC3 divergence 34 %, even share
11.0 % vs 2.9 %), and strengthens in the large+fast regime.

### Making the PD distribution separate

The PD moves only once the reach is big/fast enough to break direction-reversal
symmetry. Sweep (24 targets, paired per node, `r2 ≥ 0.5`):

| center (cm) | radius | T (s) | peak speed | Δθ₂ | ΔM | \|C\|/\|τ\| | median ΔPD | bimodal r (nl / lin) |
|---|---|---|---|---|---|---|---|---|
| (0,40) | 12 | 0.50 | 0.45 | 33° | 38 % | 12 % | 2.6° | 0.68 / 0.67 |
| (0,40) | 20 | 0.50 | 0.81 | 66° | 69 % | 22 % | 6.5° | 0.69 / 0.66 |
| **(0,40)** | **20** | **0.35** | **1.17** | **66°** | **69 %** | **23 %** | **5.3°** | **0.79 / 0.64** |
| (0,50) | 12 | 0.35 | 0.71 | 55° | 35 % | 13 % | 2.8° | 0.82 / 0.73 |
| (0,30) | 18 | 0.35 | 1.01 | 42° | 64 % | 21 % | 4.8° | 0.73 / 0.61 |

The **axis** never rotates by more than ~2° — as the `J⁻ᵀM` argument predicts,
it is set by shared geometry. What changes is the **concentration** `r`: the
nonlinear arm clusters PDs more tightly along that axis. Best setting:

```bash
python centerout.py --n_dir 24 --radius 20 --duration 0.35 --center 0 40 \
       --out centerout_nl24_big.npz  --tol 1e-4
python centerout.py --n_dir 24 --radius 20 --duration 0.35 --center 0 40 \
       --out centerout_lin24_big.npz --tol 1e-4 --linear
python analysis.py --nl centerout_nl24_big.npz --lin centerout_lin24_big.npz
```
