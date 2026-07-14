"""Directional-preference distributions (Lillicrap & Scott 2013 style) and
probes for *traces of nonlinearity* in the network activity of the combined
nonlinear-arm + iLQG model.

Key idea for the nonlinearity probes
------------------------------------
If the plant were linear (linearized arm), optimal control about a fixed start
posture would make network activity essentially *antisymmetric* across opposite
reach directions (r(theta+pi) = -r(theta)), directional tuning would be a pure
cosine (first harmonic only), and preferred directions would be ~uniform. The
nonlinear two-link arm (configuration-dependent inertia, Coriolis / centripetal
terms, intersegmental coupling) breaks each of these:
  * opposite-direction asymmetry  > 0,
  * a measurable 2nd-harmonic in directional tuning,
  * a biased (bimodal) preferred-direction distribution.
`centerout_linear.py` produces the linearized-arm control to serve as the
reference "no-nonlinearity" baseline for comparison.
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
# Nonlinearity probe 1 -- opposite-direction asymmetry over time
# -----------------------------------------------------------------------------
def opposite_asymmetry(states, ang, N):
    """For each opposite target pair, antisymmetry index over time:
        A(t) = ||r(theta,t) + r(theta+pi,t)|| / ||r(theta,t) - r(theta+pi,t)||
    A == 0 for a linear plant; > 0 signals nonlinearity."""
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
