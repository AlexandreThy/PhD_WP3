"""Reproduce Kalidindi & Crevecoeur (2025) Fig. 2 (movement-related panels) plus
directional-preference and nonlinearity analyses, for the nonlinear-arm + iLQG
model. Reads the cache produced by `centerout.py`.

    python centerout.py --n_dir 8      # produce centerout_data.npz
    python figures.py                  # produce the PNGs
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

import centerout as CO

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def dir_colors(ang):
    """Distinct hue per movement direction (blue->red diverging by angle)."""
    return plt.cm.hsv(ang / (2 * np.pi))


def hand_velocity(hand, t):
    """Cartesian hand velocity [m/s] from position [cm]; central differences."""
    dt = t[1] - t[0]
    v = np.gradient(hand, dt, axis=1) / 100.0  # cm/s -> m/s
    return v  # (n_dir, K+1, 2)


# =============================================================================
# Figure 2a -- hand kinematics
# =============================================================================
def fig2a(d, fname="fig2a_kinematics.png"):
    hand, t, ang, targ = d["hand"], d["t"], d["ang"], d["targ"]
    cols = dir_colors(ang)
    v = hand_velocity(hand, t)
    tc = t - t[0]

    fig = plt.figure(figsize=(9, 4.2))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.1, 1.4], hspace=0.5, wspace=0.3)

    # --- hand paths ---
    ax = fig.add_subplot(gs[:, 0])
    for i in range(len(ang)):
        ax.plot(hand[i, :, 0], hand[i, :, 1], color=cols[i], lw=2)
        ax.plot(targ[i, 0], targ[i, 1], "s", color=cols[i], ms=9,
                mec="k", mew=0.6)
    ax.plot(hand[0, 0, 0], hand[0, 0, 1], "o", color="k", ms=7)
    ax.set_aspect("equal")
    ax.set_title("Hand kinematics")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # 5 cm scale bar
    x0, y0 = targ[:, 0].min() - 3, targ[:, 1].min() - 4
    ax.plot([x0, x0 + 5], [y0, y0], "k", lw=2)
    ax.text(x0 + 2.5, y0 - 1.6, "5 cm", ha="center", va="top", fontsize=8)

    # --- x velocity ---
    ax = fig.add_subplot(gs[0, 1])
    for i in range(len(ang)):
        ax.plot(tc, v[i, :, 0], color=cols[i], lw=1.5)
    ax.set_ylabel("x velocity\n(m/s)")
    ax.set_xticklabels([])
    ax.axhline(0, color="0.7", lw=0.6)

    # --- y velocity ---
    ax = fig.add_subplot(gs[1, 1])
    for i in range(len(ang)):
        ax.plot(tc, v[i, :, 1], color=cols[i], lw=1.5)
    ax.set_ylabel("y velocity\n(m/s)")
    ax.set_xlabel("Time (s)")
    ax.axhline(0, color="0.7", lw=0.6)

    fig.suptitle("Fig. 2a  —  point-to-point reaches (nonlinear arm, iLQG)",
                 fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


# =============================================================================
# Figure 2b -- exemplar node (neuron) activity
# =============================================================================
def fig2b(d, nodes=None, fname="fig2b_nodes.png"):
    states, t, ang, N = d["states"], d["t"], d["ang"], int(d["N"])
    cols = dir_colors(ang)
    tc = (t - t[0]) * 1000  # ms
    r = states[:, :, :N]  # (n_dir, K+1, N)

    if nodes is None:
        # exemplar = most strongly modulated nodes (largest range across the
        # whole dataset), which show the clearest movement-related responses.
        rng = r.max(axis=(0, 1)) - r.min(axis=(0, 1))
        nodes = np.argsort(rng)[::-1][:4]

    fig, axes = plt.subplots(len(nodes), 1, figsize=(4.4, 6.4), sharex=True)
    for k, node in enumerate(nodes):
        ax = axes[k]
        for i in range(len(ang)):
            ax.plot(tc, r[i, :, node], color=cols[i], lw=1.5)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_ylabel(f"node {node}\nrate (a.u.)")
        if k == 0:
            ax.set_title("Fig. 2b  —  exemplar node activity")
    axes[-1].set_xlabel("Time from movement onset (ms)")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


# =============================================================================
# Figure 2d -- movement-epoch population dynamics (PCA, rotations)
# =============================================================================
def movement_pca(states, N, n_comp=6, tslice=None):
    """PCA of movement-epoch network activity across conditions.

    Returns projected trajectories (n_dir, T, n_comp), explained-variance ratio,
    and the components. Each unit is mean-centred across (condition x time).
    """
    r = states[:, :, :N]
    if tslice is not None:
        r = r[:, tslice, :]
    n_dir, T, _ = r.shape
    Xflat = r.reshape(n_dir * T, N)          # samples x units
    mean = Xflat.mean(axis=0, keepdims=True)
    Xc = Xflat - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    evr = S ** 2 / np.sum(S ** 2)
    comps = Vt[:n_comp]                       # (n_comp, N)
    proj = (Xc @ comps.T).reshape(n_dir, T, n_comp)
    return proj, evr, comps


def fig2d(d, fname="fig2d_rotations.png"):
    states, ang, N = d["states"], d["ang"], int(d["N"])
    cols = dir_colors(ang)
    proj, evr, _ = movement_pca(states, N, n_comp=6)

    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for i in range(len(ang)):
        ax.plot(proj[i, :, 0], proj[i, :, 1], proj[i, :, 2],
                color=cols[i], lw=1.8)
        ax.scatter(*proj[i, 0, :3], color=cols[i], s=25)          # onset
        ax.scatter(*proj[i, -1, :3], color=cols[i], s=40, marker="o",
                   edgecolor="k")                                  # end
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_zlabel("PC 3")
    ax.set_title("Fig. 2d  —  movement-epoch population trajectories")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(np.arange(1, 11), evr[:10] * 100, color="0.4")
    ax2.set_xlabel("PC index")
    ax2.set_ylabel("variance explained (%)")
    er = np.exp(-np.sum((evr) * np.log(evr + 1e-12)))
    ax2.set_title(f"scree (effective rank = {er:.1f})")

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)
    return evr


if __name__ == "__main__":
    d = CO.load()
    fig2a(d)
    fig2b(d)
    fig2d(d)
