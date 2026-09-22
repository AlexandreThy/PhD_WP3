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


def jpca(states, N, dur, k=6, kappa=0.1, subtract_cc_mean=True, tslice=None):
    """jPCA (Churchland et al. 2012) -- the plane of fastest rotation.

    Steps, canonical order:
      1. soft-normalise each unit:  r / (range(r) + kappa)
      2. subtract the **cross-condition mean at each time point** (removes the
         condition-invariant signal). Note `movement_pca` does NOT do this: it
         removes one grand mean per unit, leaving the condition-invariant time
         course in.
      3. PCA -> top k components
      4. fit dX/dt = X M with M constrained SKEW-SYMMETRIC (M = -M^T)
      5. the top conjugate eigenpair of M spans the jPC plane

    Returns dict: J (n_cond, T, 2) jPC projection, M, lam (top eigenvalue,
    purely imaginary -> rotation frequency = |Im(lam)|/2pi), r2_skew and r2_full
    (skew-constrained vs unconstrained fit of the derivative), evr, comps, P.

    CAVEAT for this model. The classic jPCA picture -- conditions spread around
    the plane, each tracing one clean rotation -- needs condition-dependent
    **preparatory** states. This model starts every condition at r = 0 exactly
    (`simulate_ILQG(r0=None)`, no preparatory epoch), so all conditions start at
    the origin of the jPC plane and spiral outward. Rotation is present (~1.1
    turns/target at 20 cm / 0.35 s, 2.5 Hz) but the figure will not look like
    Kalidindi & Crevecoeur Fig 2d without adding a preparatory epoch.
    """
    r = states[:, :, :N]
    if tslice is not None:
        r = r[:, tslice, :]
    nd, T, _ = r.shape
    dt = dur / (states.shape[1] - 1)

    rng = r.max(axis=(0, 1)) - r.min(axis=(0, 1))
    rn = r / (rng + kappa)
    if subtract_cc_mean:
        rn = rn - rn.mean(axis=0, keepdims=True)

    flat = rn.reshape(-1, N)
    mu = flat.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(flat - mu, full_matrices=False)
    comps = Vt[:k]
    X = ((flat - mu) @ comps.T).reshape(nd, T, k)

    Xs = X[:, :-1, :].reshape(-1, k)
    dX = (np.diff(X, axis=1) / dt).reshape(-1, k)

    iu = np.triu_indices(k, 1)
    A = np.zeros((Xs.shape[0] * k, len(iu[0])))
    for p in range(len(iu[0])):
        i, j = iu[0][p], iu[1][p]
        col = np.zeros((Xs.shape[0], k))
        col[:, j] += Xs[:, i]
        col[:, i] -= Xs[:, j]
        A[:, p] = col.ravel()
    m, *_ = np.linalg.lstsq(A, dX.ravel(), rcond=None)
    M = np.zeros((k, k))
    M[iu] = m
    M = M - M.T

    denom = ((dX - dX.mean(0)) ** 2).sum()
    r2_skew = 1 - ((dX - Xs @ M) ** 2).sum() / denom
    Mfull, *_ = np.linalg.lstsq(Xs, dX, rcond=None)
    r2_full = 1 - ((dX - Xs @ Mfull) ** 2).sum() / denom

    w, v = np.linalg.eig(M)
    top = np.argsort(-np.abs(w.imag))[0]
    v1 = v[:, top]
    u1 = np.real(v1); u2 = np.imag(v1)
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 - (u2 @ u1) * u1
    u2 = u2 / np.linalg.norm(u2)
    P = np.stack([u1, u2], axis=1)
    return {"J": X @ P, "M": M, "lam": w[top], "r2_skew": r2_skew,
            "r2_full": r2_full, "evr": S ** 2 / (S ** 2).sum(), "comps": comps,
            "P": P, "X": X}


def jpca_turns(J):
    """Full turns swept in the jPC plane, per condition."""
    out = []
    for i in range(J.shape[0]):
        a = np.unwrap(np.arctan2(J[i, :, 1], J[i, :, 0]))
        out.append((a[-1] - a[0]) / (2 * np.pi))
    return np.array(out)


def fig_jpca_rotations(d, dlin=None, idx=None, kappa=0.005, win_ms=300,
                       fname="fig_jpca_rotations.png"):
    """jPCA rotations to the Kalidindi & Crevecoeur (2026) STAR-Methods recipe:
    soft-normalise by (range + kappa), top-6 movement-epoch PCs over the first
    `win_ms`, skew-symmetric fit, project onto the top eigenpair. One discrete
    colour per target.

    `idx` selects which of the cached targets to show (default 8, evenly spaced,
    matching the paper's c = 8). If `dlin` is given, the linear arm is drawn in a
    second panel with shared axes.

    Reminder (see the module and README notes): because this model has no
    preparatory epoch, every target starts at the jPC origin and the loops form a
    ROSETTE rather than the paper's spatially separated loops. That separation is
    supplied by preparatory activity (paper Eqs. 19-20), absent here.
    """
    import matplotlib.pyplot as plt
    N = int(d["N"])
    dur = float(d["duration"])
    T = d["states"].shape[1]
    if idx is None:
        nd = d["states"].shape[0]
        step = max(1, nd // 8)
        idx = np.arange(0, nd, step)[:8]
    win = slice(0, int(round((win_ms / 1000.0) / (dur / (T - 1)))) + 1)
    ang = np.asarray(d["ang"])[idx]
    cols = plt.cm.hsv(np.linspace(0, 1, len(idx), endpoint=False))

    runs = [(d, "Nonlinear arm")]
    if dlin is not None:
        runs.append((dlin, "Linearized arm"))
    res = [jpca(r["states"][idx], N, dur, k=6, kappa=kappa,
               subtract_cc_mean=True, tslice=win) for r, _ in runs]
    lim = 1.08 * max(np.abs(r["J"]).max() for r in res)

    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 6.4),
                             squeeze=False)
    for ax, r, (_, name) in zip(axes[0], res, runs):
        J = r["J"]
        for i in range(len(idx)):
            ax.plot(J[i, :, 0], J[i, :, 1], color=cols[i], lw=2.6,
                    solid_capstyle="round", zorder=2)
            ax.plot(J[i, 0, 0], J[i, 0, 1], "o", color=cols[i], ms=7, mec="k",
                    mew=0.7, zorder=3)
            ax.annotate("", xy=(J[i, -1, 0], J[i, -1, 1]),
                        xytext=(J[i, -5, 0], J[i, -5, 1]),
                        arrowprops=dict(arrowstyle="-|>", color=cols[i], lw=2.4,
                                        shrinkA=0, shrinkB=0), zorder=3)
        ax.axhline(0, color="0.9", lw=0.6); ax.axvline(0, color="0.9", lw=0.6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("jPC1"); ax.set_ylabel("jPC2")
        ax.set_title("%s\nskew-fit $R^2$ = %.2f   |   %.2f Hz"
                     % (name, r["r2_skew"], abs(r["lam"].imag) / (2 * np.pi)),
                     fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)
    for (_, name), r in zip(runs, res):
        print("  %-14s R2skew %.3f (uncon %.3f)  %.2f Hz  turns %.2f"
              % (name, r["r2_skew"], r["r2_full"],
                 abs(r["lam"].imag) / (2 * np.pi), jpca_turns(r["J"]).mean()))
    return res


def fig_prep_rotations(d_nl, d_lin, fname="fig_prep_rotations.png"):
    """Prepare-to-reach model (iLQG_prep): a 2x2 panel comparing the nonlinear
    and linearized arm on (top) preparatory-epoch activity and (bottom)
    movement-epoch jPCA rotations. Expects caches from `iLQG_prep.run` (which
    carry `k_prep`).

    Prep panels use a COMMON PCA basis (pooled nl+lin prep activity -- the network
    is node-matched) so the two arms are directly comparable; movement panels fit
    jPCA per arm over the first 300 ms after movement onset (paper method).
    """
    import matplotlib.pyplot as plt
    import iLQG_Combined as B
    N = int(d_nl["N"])
    kp = int(d_nl["k_prep"])
    ang = d_nl["ang"]
    dur = float(d_nl["duration"])
    K = int(d_nl["K"])
    dt = dur / K
    nd = len(ang)
    cols = plt.cm.hsv(np.linspace(0, 1, nd, endpoint=False))
    prep_win = slice(0, kp)
    move_win = slice(kp, kp + int(round(0.300 / dt)) + 1)

    rp_n = d_nl["states"][:, prep_win, :N]
    rp_l = d_lin["states"][:, prep_win, :N]
    Xp = np.concatenate([rp_n.reshape(-1, N), rp_l.reshape(-1, N)], 0)
    mu = Xp.mean(0, keepdims=True)
    Pp = np.linalg.svd(Xp - mu, full_matrices=False)[2][:2]
    pp_n = ((rp_n.reshape(-1, N) - mu) @ Pp.T).reshape(nd, -1, 2)
    pp_l = ((rp_l.reshape(-1, N) - mu) @ Pp.T).reshape(nd, -1, 2)

    jn = jpca(d_nl["states"], N, dur, k=6, kappa=0.005, tslice=move_win)
    jl = jpca(d_lin["states"], N, dur, k=6, kappa=0.005, tslice=move_win)

    def leak(d):
        m = d["states"][:, :, :N] @ B.Wout.T
        return (np.linalg.norm(m[:, prep_win], axis=2).mean(),
                np.linalg.norm(m[:, move_win], axis=2).max())

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 11.0))
    plim = 1.08 * max(np.abs(pp_n).max(), np.abs(pp_l).max())
    for col, (pp, d, name) in enumerate([(pp_n, d_nl, "Nonlinear arm"),
                                         (pp_l, d_lin, "Linearized arm")]):
        ax = axes[0, col]
        for i in range(nd):
            ax.plot(pp[i, :, 0], pp[i, :, 1], color=cols[i], lw=2.0, zorder=2)
            ax.plot(pp[i, -1, 0], pp[i, -1, 1], "o", color=cols[i], ms=9,
                    mec="k", mew=0.8, zorder=3)
        ax.plot(pp[0, 0, 0], pp[0, 0, 1], "o", color="k", ms=7, zorder=4)
        ax.set_xlim(-plim, plim); ax.set_ylim(-plim, plim); ax.set_aspect("equal")
        ax.axhline(0, color="0.9", lw=0.6); ax.axvline(0, color="0.9", lw=0.6)
        ax.set_xlabel("prep PC1"); ax.set_ylabel("prep PC2")
        lk = leak(d)
        ax.set_title("%s — PREPARATORY activity\n(dots = states at GO; torque leak "
                     "%.2f vs move-peak %.1f)" % (name, lk[0], lk[1]),
                     fontsize=10.5, fontweight="bold", pad=8)

    jlim = 1.08 * max(np.abs(jn["J"]).max(), np.abs(jl["J"]).max())
    for col, (jr, name) in enumerate([(jn, "Nonlinear arm"), (jl, "Linearized arm")]):
        ax = axes[1, col]
        J = jr["J"]
        for i in range(nd):
            ax.plot(J[i, :, 0], J[i, :, 1], color=cols[i], lw=2.6,
                    solid_capstyle="round", zorder=2)
            ax.plot(J[i, 0, 0], J[i, 0, 1], "o", color=cols[i], ms=7, mec="k",
                    mew=0.7, zorder=3)
            ax.annotate("", xy=(J[i, -1, 0], J[i, -1, 1]),
                        xytext=(J[i, -5, 0], J[i, -5, 1]),
                        arrowprops=dict(arrowstyle="-|>", color=cols[i], lw=2.4,
                                        shrinkA=0, shrinkB=0), zorder=3)
        ax.set_xlim(-jlim, jlim); ax.set_ylim(-jlim, jlim); ax.set_aspect("equal")
        ax.axhline(0, color="0.9", lw=0.6); ax.axvline(0, color="0.9", lw=0.6)
        ax.set_xlabel("jPC1"); ax.set_ylabel("jPC2")
        ax.set_title("%s — MOVEMENT rotations\nskew-fit $R^2$ = %.2f | %.2f Hz"
                     % (name, jr["r2_skew"], abs(jr["lam"].imag) / (2 * np.pi)),
                     fontsize=10.5, fontweight="bold", pad=8)

    # self-describing condition line (amplitude, move duration, peak speed)
    move_ms = (dur - kp * dt) * 1000.0
    if "vpk" in d_nl:
        vpk = float(d_nl["vpk"])
    else:
        v = np.gradient(d_nl["hand"][:, kp:], dt, axis=1) / 100.0
        vpk = float(np.linalg.norm(v, axis=2).max())
    cond = "%g cm reach  |  %.0f ms prep + %.0f ms move  |  peak speed %.2f m/s" % (
        float(d_nl["radius"]), kp * dt * 1000.0, move_ms, vpk)
    fig.suptitle("Prepare-to-reach model (iLQG_prep): preparatory encoding and "
                 "movement rotations\n" + cond, fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # center-out target-colour key: 8 targets at their reach directions, start
    # at centre, each marker in the colour used for that target throughout.
    kax = fig.add_axes([0.455, 0.475, 0.09, 0.09])
    for i, a in enumerate(ang):
        kax.plot([0, np.cos(a)], [0, np.sin(a)], color="0.75", lw=0.8, zorder=1)
        kax.plot(np.cos(a), np.sin(a), "s", color=cols[i], ms=10, mec="k",
                 mew=0.6, zorder=2)
    kax.plot(0, 0, "o", color="k", ms=6, zorder=3)
    kax.set_xlim(-1.5, 1.5); kax.set_ylim(-1.5, 1.5); kax.set_aspect("equal")
    kax.axis("off")
    kax.set_title("target\n(colour = reach direction)", fontsize=7.5, pad=1)

    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)
    print("  movement jPCA  nonlinear R2=%.3f %.2fHz | linear R2=%.3f %.2fHz"
          % (jn["r2_skew"], abs(jn["lam"].imag) / (2 * np.pi),
             jl["r2_skew"], abs(jl["lam"].imag) / (2 * np.pi)))
    return jn, jl


def common_pca(states_a, states_b, N, n_comp=None):
    """PCA basis from the POOLED activity of two runs, with both projected into
    it (common mean, common components).

    The two runs share the network, so units are matched across them and this is
    well defined. Projecting each run into its *own* basis instead would make the
    trajectories incomparable -- the interesting quantity is the difference at
    matched (condition, time, unit), which needs one basis.

    Returns (proj_a, proj_b, evr, comps).
    """
    ra, rb = states_a[:, :, :N], states_b[:, :, :N]
    nd, T = ra.shape[:2]
    X = np.concatenate([ra.reshape(-1, N), rb.reshape(-1, N)], 0)
    mu = X.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    evr = S ** 2 / (S ** 2).sum()
    k = n_comp or N
    comps = Vt[:k]
    pa = ((ra.reshape(-1, N) - mu) @ comps.T).reshape(nd, T, k)
    pb = ((rb.reshape(-1, N) - mu) @ comps.T).reshape(nd, T, k)
    return pa, pb, evr, comps


def pca_divergence(d, dlin, n_comp=6):
    """How the nonlinear/linear difference distributes over the leading PCs.

    Returns a dict with, per PC: variance explained, share of the total squared
    difference, relative divergence ||nl-lin||/||nl||, and the direction-EVEN
    share of the PC in each model (the even component is where the arm
    nonlinearity lives -- see analysis.py).

    Measured at 20 cm / 0.35 s: PC1-3 span the same subspace in both models
    (principal angles < 3 deg) and have near-identical variance spectra, yet PC3
    -- only ~5% of the variance -- carries ~30% of the difference, diverges by
    53%, and triples its even content (21.5% vs 8.3%).
    """
    N = int(d["N"])
    pn, pl, evr, _ = common_pca(d["states"], dlin["states"], N, n_comp)
    nd = pn.shape[0]
    half = nd // 2
    dp = pn - pl
    # Normalise by the FULL difference across all N dims, not just the retained
    # PCs, so diff_share is a true share of the total (it will not sum to 1).
    # The basis is orthonormal, so the full projected difference has the same
    # norm as the raw difference.
    full = ((d["states"][:, :, :N] - dlin["states"][:, :, :N]) ** 2).sum()
    pw = (dp ** 2).sum((0, 1)) / full

    def even_share(p, k):
        e = (p[:half, :, k] + p[half:, :, k]) / 2.0
        o = (p[:half, :, k] - p[half:, :, k]) / 2.0
        return (e ** 2).sum() / ((e ** 2).sum() + (o ** 2).sum())

    return {
        "evr": evr[:n_comp],
        "diff_share": pw,
        "rel": np.array([np.linalg.norm(dp[:, :, k]) / np.linalg.norm(pn[:, :, k])
                         for k in range(n_comp)]),
        "even_nl": np.array([even_share(pn, k) for k in range(n_comp)]),
        "even_lin": np.array([even_share(pl, k) for k in range(n_comp)]),
        "proj_nl": pn, "proj_lin": pl,
    }


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
