"""erank and dR2_even for the muscle model: F-L/V variants x linear/nonlinear arm.

Reads analyses_claude/sims/muscle_*.npz (from muscle_centerout.py).

dR2_even: cross-validated gain (leave-one-opposite-pair-out) from adding a
family of nonlinear regressors to the linear-kinematics model
[1, theta, theta_dot, theta_ddot], on the direction-even component
r_e = [r(th) + r(th + pi)] / 2 (regressors averaged the same way), as in
predictions.term_dr2. Regressor families, all computed from kinematics:

    coriolis   C(theta, theta_dot)                       (2)
    Mvar       dM(theta2) theta_ddot, ref = centre elbow  (2)
    FL         f_l(l_i) - 1, per muscle                  (6)
    FV         f_fv(l_i, ldot_i) - 1, per muscle         (6)

The null for a family is the plant that never had to produce it (linear arm
for coriolis/Mvar, flv = none for FL/FV).

    python analyses_claude/muscle_analysis.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import muscle_plant as MP  # noqa: E402
import muscle_centerout as MC  # noqa: E402
from erank import measures, even_odd  # noqa: E402

M = MP.M
SIMS = os.path.join(HERE, "sims")
REGIMES = [("12cm_050s", "12 cm / 0,50 s"), ("12cm_030s", "12 cm / 0,30 s")]
ARMS = [("lin", "bras linéarisé"), ("nl", "bras non linéaire")]
FLVS = ["none", "length", "velocity", "both"]
FLV_LAB = {"none": "sans FLV", "length": "F-L", "velocity": "F-V", "both": "F-L + F-V"}
FAMILIES = ["coriolis", "Mvar", "FL", "FV"]
FAM_LAB = {"coriolis": "Coriolis C", "Mvar": "inertie dM·θ̈",
           "FL": "force–longueur", "FV": "force–vitesse"}

_GEOM = MP.make_plant(False, True)          # muscle geometry (same for all plants)


def load(regime, arm, flv):
    return np.load(os.path.join(SIMS, f"muscle_{regime}_{arm}_{flv}.npz"), allow_pickle=True)


def kept_targets(regime):
    """Indices of the opposite-target pairs whose reaches converged in ALL eight
    plants of the regime, ordered [first halves, second halves] so that
    even_odd() still pairs target k with target k + n/2."""
    oks = []
    for a, _ in ARMS:
        for f in FLVS:
            d = load(regime, a, f)
            oks.append(MC.valid(d["hand"], d["targ"], d["endpoint_error"]))
    oks = np.array(oks)
    half = oks.shape[1] // 2
    ok = oks[:, :half].all(0) & oks[:, half:].all(0)
    keep = np.where(ok)[0]
    return np.concatenate([keep, keep + half])


def regressors(d, idx):
    st, N, dur = d["states"][idx], int(d["N"]), float(d["duration"])
    nd, T, _ = st.shape
    q = st[:, :, N:N + 4]
    th, w = q[..., 0:2], q[..., 2:4]
    acc = np.gradient(w, dur / (T - 1), axis=1)
    th2, w1, w2 = th[..., 1], w[..., 0], w[..., 1]

    C = np.stack([-w2 * (2 * w1 + w2) * M.a2 * np.sin(th2),
                  w1 ** 2 * M.a2 * np.sin(th2)], -1)
    dc = M.a2 * (np.cos(th2) - np.cos(M.CENTER_JOINTS[1]))
    Mv = np.stack([dc * (2 * acc[..., 0] + acc[..., 1]), dc * acc[..., 0]], -1)

    l = _GEOM.l_const - th @ _GEOM.A_len          # (nd, T, 6)
    v = -w @ _GEOM.A_len
    FL = M.force_length(l) - 1.0
    FV = M.force_velocity(l, v)[0] - 1.0

    Xlin = np.concatenate([np.ones((nd, T, 1)), th, w, acc], 2)
    return Xlin, {"coriolis": C, "Mvar": Mv, "FL": FL, "FV": FV}


def dr2_even(r, Xlin, extra):
    Xful = np.concatenate([Xlin, extra], 2)
    r, Xlin, Xful = even_odd(r)[0], even_odd(Xlin)[0], even_odd(Xful)[0]
    nf, N = r.shape[0], r.shape[2]

    def cv(X):
        sse, sst = 0.0, 0.0
        for h in range(nf):
            tr = [k for k in range(nf) if k != h]
            b, *_ = np.linalg.lstsq(X[tr].reshape(-1, X.shape[2]),
                                    r[tr].reshape(-1, N), rcond=None)
            sse += ((r[h] - X[h] @ b) ** 2).sum()
            sst += ((r[h] - r[tr].reshape(-1, N).mean(0)) ** 2).sum()
        return sse, sst

    sl, sst = cv(Xlin)
    sf, _ = cv(Xful)
    return (sl - sf) / (sst + 1e-12)


def main():
    res, npairs = {}, {}
    for rg, _ in REGIMES:
        idx = kept_targets(rg)
        npairs[rg] = len(idx) // 2
        for arm, _ in ARMS:
            for fl in FLVS:
                d = load(rg, arm, fl)
                N = int(d["N"])
                r = d["states"][idx, :, :N]
                Xlin, ex = regressors(d, idx)
                ev, od = even_odd(r)
                hv = np.linalg.norm(np.gradient(d["hand"][idx], float(d["duration"]) / int(d["K"]),
                                                axis=1), axis=-1).max() / 100
                res[rg, arm, fl] = dict(
                    erank=measures(r), even=measures(ev)[0], odd=measures(od)[0],
                    dr2={f: dr2_even(r, Xlin, ex[f]) for f in FAMILIES},
                    dr2_all=dr2_even(r, Xlin, np.concatenate([ex[f] for f in FAMILIES], 2)),
                    err=float(d["endpoint_error"][idx].max()), vpk=hv)

    lines = []
    for rg, rl in REGIMES:
        vpk = res[rg, "nl", "both"]["vpk"]
        lines.append(f"\n=== {rl}  (vitesse de pic {vpk:.2f} m/s, {npairs[rg]} paires de cibles "
                     f"opposées sur 12 conservées) ===")
        hdr = (f"{'bras':5s} {'FLV':9s} | {'erank_RV':>8s} {'erank_var':>9s} {'pair':>6s} {'impair':>6s} | "
               + " ".join(f"{f:>9s}" for f in FAMILIES) + f" {'tous':>7s} | {'err (cm)':>8s}")
        lines += [hdr, "-" * len(hdr)]
        for arm, _ in ARMS:
            for fl in FLVS:
                w = res[rg, arm, fl]
                lines.append(
                    f"{arm:5s} {fl:9s} | {w['erank'][0]:8.2f} {w['erank'][1]:9.2f} "
                    f"{w['even']:6.2f} {w['odd']:6.2f} | "
                    + " ".join(f"{w['dr2'][f]:9.3f}" for f in FAMILIES)
                    + f" {w['dr2_all']:7.3f} | {w['err']:8.1e}")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(HERE, "muscle_table.txt"), "w", encoding="utf-8") as f:
        f.write("dR2_even columns: cross-validated gain over linear kinematics\n" + table + "\n")

    # leave-one-pair-out CV on fewer than 8 pairs over-fits wildly; table only
    shown = [(rg, rl) for rg, rl in REGIMES if npairs[rg] >= 8]
    fig, ax = plt.subplots(len(shown), 2, figsize=(16, 4.6 * len(shown)), squeeze=False)
    cols = {("lin", "none"): "0.75", ("lin", "length"): "#9ecae1", ("lin", "velocity"): "#a1d99b",
            ("lin", "both"): "#bcbddc", ("nl", "none"): "0.35", ("nl", "length"): "#3182bd",
            ("nl", "velocity"): "#31a354", ("nl", "both"): "#756bb1"}
    plants = [(a, f) for a, _ in ARMS for f in FLVS]
    wbar = 0.8 / len(plants)
    for i, (rg, rl) in enumerate(shown):
        a0, a1 = ax[i, 0], ax[i, 1]
        x = np.arange(len(FAMILIES))
        for j, (arm, fl) in enumerate(plants):
            vals = [res[rg, arm, fl]["dr2"][f] for f in FAMILIES]
            a0.bar(x - 0.4 + (j + 0.5) * wbar, vals, wbar, color=cols[arm, fl],
                   hatch=None if arm == "nl" else "//", edgecolor="w", lw=0.3,
                   label=f"{'non lin.' if arm == 'nl' else 'linéarisé'}, {FLV_LAB[fl]}")
        a0.axhline(0, color="k", lw=0.6)
        a0.set_xticks(x, [FAM_LAB[f] for f in FAMILIES])
        a0.set_ylabel("ΔR² validé, composante paire")
        a0.set_title(f"A. Gain de chaque famille de régresseurs — {rl}")
        if i == 0:
            a0.legend(fontsize=7, ncol=2)

        x = np.arange(len(plants))
        a1.bar(x - 0.2, [res[rg, a, f]["odd"] for a, f in plants], 0.4, color="0.7", label="impair")
        a1.bar(x + 0.2, [res[rg, a, f]["even"] for a, f in plants], 0.4,
               color=[cols[p] for p in plants], label="pair")
        a1.set_xticks(x, [f"{'NL' if a == 'nl' else 'lin'}\n{FLV_LAB[f]}" for a, f in plants], fontsize=8)
        a1.set_ylabel("erank (Roy & Vetterli)")
        a1.set_ylim(min(res[rg, a, f]["odd"] for a, f in plants) - 1.0, None)
        a1.set_title(f"B. erank des composantes impaire (gris) et paire — {rl}")
    fig.suptitle("Modèle musculaire (6 muscles, Lillicrap & Scott) — même réseau, même coût, mêmes cibles",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_muscle_flv.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
