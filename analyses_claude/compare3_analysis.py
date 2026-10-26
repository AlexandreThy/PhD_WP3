"""Analysis of compare3.py: lin2 / nl2 / nl6 (see that module).

Per regime and model:
    erank (Roy & Vetterli) of the activity, and of its odd / even components
    non-affine fraction 1 - R2 of  r(theta, t) ~ c(t) + G(t) x_targ_joint(theta)
    dR2_even of all nonlinear regressors (Coriolis, dM theta_ddot, F-L, F-V)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from erank import measures, even_odd  # noqa: E402
import muscle_analysis as MA  # noqa: E402

SIMS = os.path.join(HERE, "sims")
REGIMES = [("12cm_030s", "12 cm / 0,30 s"), ("12cm_050s", "12 cm / 0,50 s")]
MODELS = [("lin2", "linéaire,\n2 actionneurs", "0.6"),
          ("nl2", "non linéaire,\n2 actionneurs", "#3182bd"),
          ("nl6", "non linéaire,\n6 muscles + FLV", "crimson")]


def affine_residual(r, xt):
    """1 - R2 of the across-condition affine map from the joint target, all times."""
    X = np.c_[np.ones(len(xt)), xt]
    sse = sst = 0.0
    for k in range(1, r.shape[1]):
        Y = r[:, k, :]
        b, *_ = np.linalg.lstsq(X, Y, rcond=None)
        sse += ((Y - X @ b) ** 2).sum()
        sst += ((Y - Y.mean(0)) ** 2).sum()
    return sse / sst


def main():
    res = {}
    for rg, _ in REGIMES:
        for m, _, _ in MODELS:
            d = np.load(os.path.join(SIMS, f"cmp_{rg}_{m}.npz"))
            N = int(d["N"])
            r = d["states"][:, :, :N]
            ev, od = even_odd(r)
            idx = np.arange(r.shape[0])
            Xlin, ex = MA.regressors(d, idx)
            res[rg, m] = dict(
                glob=measures(r)[0], odd=measures(od)[0], even=measures(ev)[0],
                nonaff=affine_residual(r, d["targ_joint"]),
                dr2=MA.dr2_even(r, Xlin, np.concatenate([ex[f] for f in MA.FAMILIES], 2)),
                err=float(d["endpoint_error"].max()))

    lines = []
    for rg, rl in REGIMES:
        lines.append(f"\n=== {rl} (24 cibles, coût en angles articulaires) ===")
        lines.append(f"{'modèle':6s} | {'erank':>6s} {'impair':>7s} {'pair':>6s} | "
                     f"{'non affine (%)':>14s} | {'dR2_even':>8s} | {'err (cm)':>8s}")
        for m, _, _ in MODELS:
            w = res[rg, m]
            lines.append(f"{m:6s} | {w['glob']:6.2f} {w['odd']:7.2f} {w['even']:6.2f} | "
                         f"{100 * w['nonaff']:14.5f} | {w['dr2']:8.3f} | {w['err']:8.1e}")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(HERE, "compare3_table.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")

    labels = [m[1] for m in MODELS]
    colors = [m[2] for m in MODELS]
    x = np.arange(len(MODELS))
    fig, ax = plt.subplots(len(REGIMES), 4, figsize=(19, 4.3 * len(REGIMES)), squeeze=False)
    for i, (rg, rl) in enumerate(REGIMES):
        w = [res[rg, m] for m, _, _ in MODELS]

        a = ax[i, 0]
        g = [v["glob"] for v in w]
        a.bar(x, g, 0.6, color=colors)
        for xi, v in zip(x, g):
            a.text(xi, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
        a.set_ylim(0, max(g) * 1.15)
        a.set_title(f"A. erank global — {rl}")

        a = ax[i, 1]
        o, e = [v["odd"] for v in w], [v["even"] for v in w]
        a.bar(x - 0.2, o, 0.4, color=colors, alpha=0.45)
        a.bar(x + 0.2, e, 0.4, color=colors)
        for xi, oo, ee in zip(x, o, e):
            a.text(xi - 0.2, oo + 0.1, f"{oo:.2f}", ha="center", fontsize=8)
            a.text(xi + 0.2, ee + 0.1, f"{ee:.2f}", ha="center", fontsize=8)
        a.set_ylim(0, max(o + e) * 1.15)
        a.set_title("B. erank impair (clair) / pair (foncé)")

        a = ax[i, 2]
        na = [100 * v["nonaff"] for v in w]
        a.bar(x, na, 0.6, color=colors)
        a.set_yscale("log")
        for xi, v in zip(x, na):
            a.text(xi, v * 1.3, f"{v:.2g} %", ha="center", fontsize=9)
        a.set_ylim(min(na) / 5, max(na) * 8)
        a.set_ylabel("% de variance")
        a.set_title("C. Part non affine en la cible articulaire")

        a = ax[i, 3]
        dr = [v["dr2"] for v in w]
        a.bar(x, dr, 0.6, color=colors)
        for xi, v in zip(x, dr):
            a.text(xi, v + 0.004 if v >= 0 else v - 0.012, f"{v:.3f}", ha="center", fontsize=9)
        a.axhline(0, color="k", lw=0.6)
        a.set_title("D. ΔR²_even (tous régresseurs non linéaires)")

        for a in ax[i]:
            a.set_xticks(x, labels, fontsize=8)
        ax[i, 0].set_ylabel("erank (Roy & Vetterli)")
        ax[i, 1].set_ylabel("erank (Roy & Vetterli)")
        ax[i, 3].set_ylabel("ΔR² validé")

    fig.suptitle("Même réseau, même tâche, même coût (angles articulaires) ; seul le corps change",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_compare3.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
