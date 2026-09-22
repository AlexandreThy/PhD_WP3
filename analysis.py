"""Directional-preference distributions (Lillicrap & Scott 2013 style) and
probes for *traces of nonlinearity* in the network activity of the combined
nonlinear-arm + iLQG model.

What the linear baseline does and does NOT remove
-------------------------------------------------
`LINEARIZE` freezes the inertia at M(THETA2_REF) and drops the Coriolis terms.
It does *not* make the plant isotropic, and it does *not* linearize the task:

  * M(THETA2_REF) is still a full anisotropic matrix (eigenvalue ratio ~4.6),
  * the hand Jacobian J is untouched (pure kinematics, identical in both models),
  * the joint target xtarg(theta) = IK(center + R*[cos, sin]) is a *nonlinear*
    function of reach direction in both models.

Because the network never receives arm state (fx is block-triangular: the plant
is a one-way cascade network -> torque -> arm), for the linear arm the optimal
activity is affine in the joint target, r_i(theta) ~ c_i + G[i,:] . [cos, sin]
with G = pinv(Wout) @ M @ inv(J). The PD distribution is therefore the pushforward
of the readout directions through J^-T M -- an anisotropic map built from
kinematics and frozen inertia alone. This predicts the observed bimodal axis to
within ~2 deg *with no nonlinearity at all*.

So bimodal PDs and a nonzero 2nd harmonic are NOT signatures of nonlinearity
here: the linearized baseline shows them too (measured at 12 cm / 0.5 s:
bimodal r = 0.67 linear vs 0.68 nonlinear; P2/P1 = 0.018 vs 0.019).

Where the nonlinearity actually shows up
----------------------------------------
Coriolis ~ v^2 is invariant under reversing the movement, so C(theta+pi) ~
C(theta): it is nearly EVEN in reach direction, hence lives in even harmonics.
The inertia-variation term is even to leading order too. But PD = arctan2(b1,a1)
is read off the FIRST harmonic, and on a uniform direction grid cos/sin(theta)
are exactly orthogonal to cos/sin(2*theta) -- so the PD estimate is blind to the
even harmonics by construction (a 2nd harmonic 10x the 1st shifts the fitted PD
by exactly zero). This is why the nonlinearity reshapes activity by ~30%
(`node_divergence`) while barely moving the PD.

Probes that DO separate the two plants: `opposite_asymmetry` (the even/odd ratio
-- exactly where the nonlinearity lives) and `node_divergence`. The PD only
separates once the reach is large and fast enough to break the
direction-reversal symmetry (see `fig_pd`'s docstring for a working setting).

Run `centerout.py --linear` to produce the linearized-arm baseline cache.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import centerout as CO


# -----------------------------------------------------------------------------
# per-neuron directional tuning
# -----------------------------------------------------------------------------
def neuron_feature(states, N, kind="mean", tslice=None):
    """Scalar activity per (direction, neuron). Default = time-mean rate."""
    r = states[:, :, :N]
    if tslice is not None:
        r = r[:, tslice, :]
    if kind == "mean":
        return r.mean(axis=1)          # (n_dir, N)
    if kind == "integral":
        return r.sum(axis=1)
    if kind == "peak":
        j = np.argmax(np.abs(r), axis=1)
        return np.take_along_axis(r, j[:, None, :], axis=1)[:, 0, :]
    raise ValueError(kind)


def tuning_fit(feature, ang, n_harm=2):
    """Least-squares harmonic fit of activity vs direction, per neuron.

    activity(theta) ~ b0 + sum_h [a_h cos(h theta) + b_h sin(h theta)]

    Returns dict with preferred direction, first/second harmonic amplitudes,
    R^2 of the pure-cosine (first-harmonic) fit, and the harmonic-power ratio
    P2/P1 (a per-neuron index of tuning nonlinearity).
    """
    ang = np.asarray(ang)
    cols = [np.ones_like(ang)]
    for h in range(1, n_harm + 1):
        cols += [np.cos(h * ang), np.sin(h * ang)]
    Xmat = np.stack(cols, axis=1)                 # (n_dir, 1+2H)
    beta, *_ = np.linalg.lstsq(Xmat, feature, rcond=None)  # (1+2H, N)

    a1, b1 = beta[1], beta[2]
    pd = np.arctan2(b1, a1)
    amp1 = np.hypot(a1, b1)
    amp2 = np.hypot(beta[3], beta[4]) if n_harm >= 2 else np.zeros_like(amp1)

    # R^2 of first-harmonic-only model, per neuron
    X1 = Xmat[:, :3]
    b1fit, *_ = np.linalg.lstsq(X1, feature, rcond=None)
    pred = X1 @ b1fit
    ss_res = ((feature - pred) ** 2).sum(axis=0)
    ss_tot = ((feature - feature.mean(axis=0)) ** 2).sum(axis=0) + 1e-15
    r2 = 1 - ss_res / ss_tot

    p_ratio = (amp2 ** 2) / (amp1 ** 2 + 1e-15)
    return {"pd": pd, "amp1": amp1, "amp2": amp2, "r2": r2,
            "p_ratio": p_ratio, "beta": beta}


def bimodal_rayleigh(pd):
    """Bimodal (axial) Rayleigh vector: doubles the angles. Returns (r, orient)."""
    z = np.mean(np.exp(1j * 2 * pd))
    return np.abs(z), 0.5 * np.angle(z)


# -----------------------------------------------------------------------------
# Preferred-direction distribution (Lillicrap & Scott Fig 3 analogue)
# -----------------------------------------------------------------------------
def _pd_polar(ax, pd, N, color, title):
    r_bi, orient = bimodal_rayleigh(pd)
    nb = 16
    bins = np.linspace(-np.pi, np.pi, nb + 1)
    counts, _ = np.histogram(pd, bins=bins)
    width = 2 * np.pi / nb
    ax.bar(bins[:-1] + width / 2, counts, width=width, color=color,
           edgecolor="k", alpha=0.8, linewidth=0.6)
    rmax = counts.max() * 1.05
    for o in (orient, orient + np.pi):
        ax.plot([o, o], [0, rmax], color="crimson", lw=2)
    ax.set_yticklabels([])
    ax.set_title(f"{title}\n{len(pd)}/{N} tuned, bimodal r = {r_bi:.2f}, "
                 f"axis = {np.degrees(orient):.0f}°", fontsize=9)
    return r_bi, orient


def fig_pd(d, dlin=None, fname="fig_pd_distribution.png", r2_min=0.5):
    """Preferred-direction polar histogram (Lillicrap & Scott Fig. 3 analogue).

    If dlin (linear-arm baseline) is given, plot it alongside for comparison.

    Note this panel barely separates the two plants at the default reach
    (12 cm / 0.5 s: bimodal r = 0.68 nonlinear vs 0.67 linear) -- see the module
    docstring for why (the PD reads the 1st harmonic; the nonlinearity is even,
    hence 2nd-harmonic). It separates for large, fast reaches, which break the
    direction-reversal symmetry:

        python centerout.py --n_dir 24 --radius 20 --duration 0.35 \
               --center 0 40 --out centerout_nl24_big.npz  --tol 1e-4
        python centerout.py --n_dir 24 --radius 20 --duration 0.35 \
               --center 0 40 --out centerout_lin24_big.npz --tol 1e-4 --linear

    giving bimodal r = 0.79 (nonlinear) vs 0.64 (linear), median per-node PD
    shift 5.3 deg. The bimodal *axis* stays put (~2 deg) in every setting: it is
    fixed by J^-T M, which both plants share.
    """
    states, ang, N = d["states"], d["ang"], int(d["N"])
    fit = tuning_fit(neuron_feature(states, N, "mean"), ang)
    keep = fit["r2"] >= r2_min

    npan = 2 if dlin is not None else 1
    fig = plt.figure(figsize=(5 * npan, 5))
    ax = fig.add_subplot(1, npan, 1, projection="polar")
    r_bi, orient = _pd_polar(ax, fit["pd"][keep], N, "mediumseagreen",
                             "Nonlinear arm")
    if dlin is not None:
        fl = tuning_fit(neuron_feature(dlin["states"], int(dlin["N"]), "mean"),
                        dlin["ang"])
        kl = fl["r2"] >= r2_min
        ax2 = fig.add_subplot(1, npan, 2, projection="polar")
        _pd_polar(ax2, fl["pd"][kl], int(dlin["N"]), "0.6", "Linear arm")

    fig.suptitle("Preferred-direction distribution", y=1.02)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}  (nonlinear bimodal r={r_bi:.3f}, tuned={keep.sum()}/{N})")
    return fit, keep


# -----------------------------------------------------------------------------
# Nonlinearity probe 0 -- rank of the condition-dependence (the sharpest test)
# -----------------------------------------------------------------------------
def target_affine_r2(d):
    """Is the activity an AFFINE function of the 2-D joint target?

        r(theta, t)  ~  c(t) + G(t) . xtarg(theta)

    For a **linear** plant with quadratic cost this is exact, not approximate:
    LQR makes the optimal control affine in the target, and the network never
    receives arm state (fx is block-triangular), so across conditions the
    activity matrix r[:, t, :] has **rank <= 1 + dim(xtarg) = 3 at every time**,
    no matter how many targets are probed. A nonlinear plant has no such bound.

    Measured (20 cm / 0.35 s, 24 targets):

        linear     R^2 = 0.99999982   median across-direction rank 4  (the 4th
                                      singular value sits at ~1e-4 = the iLQG
                                      tol, i.e. solver residue, not signal)
        nonlinear  R^2 = 0.97519534   median across-direction rank 15

    and the linear rank stays at 4 whether you probe 6, 8, 12 or 24 targets,
    while the nonlinear rank tracks the number of conditions (6/8/12/15).

    This is the precise sense in which the nonlinear plant "needs more
    dimensions": not more *variance* dimensions -- PCA effective rank is 2.51 vs
    2.46, blind to this -- but a higher-rank dependence on the task condition.
    The extra dimensions carry only ~2.5% of the variance and are ~89%
    direction-EVEN, i.e. they are exactly the interaction-torque subspace that
    the preferred direction cannot see (see module docstring).

    Returns (r2, residual) with residual shaped like the activity.
    """
    import iLQG_Combined as _M
    N = int(d["N"])
    st = d["states"]
    nd, T = st.shape[:2]
    xt = np.array([_M.compute_angles_from_cartesian(t[0], t[1]) for t in d["targ"]])
    X = np.concatenate([np.ones((nd, 1)), xt], axis=1)      # [1, xtarg]  (nd, 3)
    r = st[:, :, :N]
    res = np.zeros_like(r)
    for t in range(T):
        beta, *_ = np.linalg.lstsq(X, r[:, t, :], rcond=None)
        res[:, t, :] = r[:, t, :] - X @ beta
    ss_res = (res ** 2).sum()
    ss_tot = ((r - r.mean(axis=0, keepdims=True)) ** 2).sum()
    return 1 - ss_res / ss_tot, res


def condition_rank(d, tol=1e-6):
    """Median numerical rank over time of the across-direction activity matrix
    r[:, t, :]. Capped at 3 (+solver residue) for a linear plant; unbounded for
    a nonlinear one. See `target_affine_r2`."""
    N = int(d["N"])
    r = d["states"][:, :, :N]
    rk = []
    for t in range(1, r.shape[1]):
        s = np.linalg.svd(r[:, t, :], compute_uv=False)
        rk.append(int((s > tol * s[0]).sum()))
    return float(np.median(rk))


# -----------------------------------------------------------------------------
# Nonlinearity probe 1 -- opposite-direction asymmetry over time
# -----------------------------------------------------------------------------
def opposite_asymmetry(states, ang, N):
    """For each opposite target pair, antisymmetry index over time:
        A(t) = ||r(theta,t) + r(theta+pi,t)|| / ||r(theta,t) - r(theta+pi,t)||

    This is the even/odd ratio of the directional response, i.e. exactly the
    subspace the arm nonlinearity occupies (Coriolis ~ v^2 is even under
    direction reversal). It is the most sensitive probe here.

    Note A > 0 even for the *linear* baseline (measured ~0.22 at 12 cm / 0.5 s),
    because the inverse kinematics theta -> joint target is nonlinear regardless
    of the plant. The nonlinear arm raises it to ~0.36. Compare against the
    linearized run rather than against zero.
    """
    r = states[:, :, :N]
    n_dir = len(ang)
    assert n_dir % 2 == 0
    half = n_dir // 2
    A = []
    for i in range(half):
        j = i + half                     # opposite direction
        num = np.linalg.norm(r[i] + r[j], axis=1)
        den = np.linalg.norm(r[i] - r[j], axis=1) + 1e-12
        A.append(num / den)
    return np.array(A)                    # (half, T)


def node_divergence(states_nl, states_lin, N):
    """Node-matched divergence between nonlinear and linear activity.

    Same network and targets in both runs, so node i corresponds across them.
    Returns the fraction of activity reshaped by the plant nonlinearity,
        D(t) = ||r_nl(.,t) - r_lin(.,t)||_F / ||r_nl(.,t)||_F
    computed across (direction x node) at each time, and the per-node total
    L2 difference (to pick the most-affected exemplar node).
    """
    rn = states_nl[:, :, :N]
    rl = states_lin[:, :, :N]
    diff = rn - rl
    num = np.linalg.norm(diff.reshape(-1, diff.shape[1], N), axis=(0, 2))  # per t
    den = np.linalg.norm(rn.reshape(-1, rn.shape[1], N), axis=(0, 2)) + 1e-12
    D_t = num / den
    per_node = np.linalg.norm(diff, axis=(0, 1))  # (N,)
    return D_t, per_node


def fig_nonlinearity(d, dlin=None, fname="fig_nonlinearity.png"):
    """Traces of *dynamic* nonlinearity in the network activity.

    The network is linear, so all nonlinearity is inherited from the arm.
    Comparing to the linear-arm baseline (identical network, identical targets)
    isolates the effect of configuration-dependent inertia + Coriolis terms.
    """
    states, ang, N, t = d["states"], d["ang"], int(d["N"]), d["t"]
    tc = (t - t[0]) * 1000
    A = opposite_asymmetry(states, ang, N)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))

    # (1) opposite-direction asymmetry over time
    ax = axes[0]
    ax.plot(tc, A.mean(0), color="crimson", lw=2, label="nonlinear arm")
    ax.fill_between(tc, A.mean(0) - A.std(0), A.mean(0) + A.std(0),
                    color="crimson", alpha=0.2)
    if dlin is not None:
        Al = opposite_asymmetry(dlin["states"], dlin["ang"], int(dlin["N"]))
        ax.plot(tc, Al.mean(0), color="0.4", lw=2, ls="--", label="linear arm")
        ax.fill_between(tc, Al.mean(0) - Al.std(0), Al.mean(0) + Al.std(0),
                        color="0.4", alpha=0.15)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("antisymmetry index")
    ax.set_title("Opposite-direction asymmetry")
    ax.legend(fontsize=8)

    node = None
    if dlin is not None:
        D_t, per_node = node_divergence(states, dlin["states"], N)
        node = int(np.argmax(per_node))

        # (2) node-matched nonlinear/linear divergence over time
        ax = axes[1]
        ax.plot(tc, D_t * 100, color="darkviolet", lw=2)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("activity reshaped by\nnonlinearity (%)")
        ax.set_title("Nonlinear vs linear network activity")

        # (3) most-affected node: activity time course, nonlinear vs linear
        ax = axes[2]
        rn = states[:, :, node]
        rl = dlin["states"][:, :, node]
        cols = plt.cm.hsv(ang / (2 * np.pi))
        for i in range(0, len(ang), max(1, len(ang) // 8)):
            ax.plot(tc, rn[i], color=cols[i], lw=1.6)
            ax.plot(tc, rl[i], color=cols[i], lw=1.2, ls=":")
        ax.plot([], [], color="k", lw=1.6, label="nonlinear")
        ax.plot([], [], color="k", lw=1.2, ls=":", label="linear")
        ax.set_xlabel("time (ms)"); ax.set_ylabel("rate (a.u.)")
        ax.set_title(f"Most-affected node ({node})")
        ax.legend(fontsize=8)
    else:
        for ax in axes[1:]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)

    print("wrote", fname)
    print(f"  opposite-direction asymmetry (peak): nonlinear={A.mean(0).max():.3f}",
          end="")
    if dlin is not None:
        print(f"  linear={Al.mean(0).max():.3f}")
        print(f"  peak activity reshaped by nonlinearity: {D_t.max()*100:.1f}%  "
              f"(most-affected node {node})")
    else:
        print()


if __name__ == "__main__":
    import os
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--nl", default="centerout_nl24.npz",
                   help="nonlinear-arm cache")
    p.add_argument("--lin", default="centerout_lin24.npz",
                   help="linear-arm baseline cache")
    args = p.parse_args()

    d = CO.load(args.nl) if os.path.exists(args.nl) else CO.load()
    dlin = CO.load(args.lin) if os.path.exists(args.lin) else None
    fig_pd(d, dlin)
    fig_nonlinearity(d, dlin)
