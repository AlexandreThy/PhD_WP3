"""Figure: nonlinearity signatures measurable in monkey M1 (lin6 vs nl6).

Needs sims/cmp_*.npz (compare3.py, models lin6 nl6 nl6x lin6L lin6V lin6f)
and sims/netvar.npz (net_variability.py).

    python analyses_claude/fig_signatures.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import signatures as S  # noqa: E402
from net_variability import NETS  # noqa: E402

COL = {"lin6": "0.5", "nl6": "crimson"}
LAB = {"lin6": "bras linéarisé, muscles sans FLV", "nl6": "complet (bras non lin. + FLV)"}
RG = [("12cm_030s", "0,30 s"), ("12cm_050s", "0,50 s")]
LS_AXIS = (-52.1, -80.3)        # Lillicrap & Scott 2013, rung 5 -> 6 (127.9 -> 99.7 deg)


def wrap_axis(a):
    """Axial angle into (-135, 45] so that axes near -90 do not split at +-90."""
    return (np.asarray(a) + 135.0) % 180.0 - 135.0


def main():
    nv = np.load(os.path.join(HERE, "sims", "netvar.npz"))
    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(2, 4)

    # A. PD distributions (one network, 24 targets, 0.5 s)
    for j, m in enumerate(("lin6", "nl6")):
        ax = fig.add_subplot(gs[0, j], projection="polar")
        d = S.load("12cm_050s", m)
        y = S.epoch_mean(d["r"], 1, d["k_peak"] + 1)
        pd, _, r2 = S.cosine_fit(y, d["ang"])
        pd = pd[r2 > 0.5]
        R, axis = S.bimodal(pd)
        ax.hist(pd, bins=24, range=(-np.pi, np.pi), color=COL[m], alpha=0.85)
        for a in (axis, axis + 180):
            ax.plot([np.radians(a)] * 2, [0, ax.get_ylim()[1]], "k--", lw=1.2)
        ax.set_title(f"A{j + 1}. PD — {LAB[m]}\naxe {wrap_axis(axis):.0f}°, bimodalité {R:.2f}",
                     fontsize=9)
        ax.set_yticklabels([])

    # B. bimodal axis across networks
    ax = fig.add_subplot(gs[0, 2])
    for i, (rg, rl) in enumerate(RG):
        for k, m in enumerate(("lin6", "nl6")):
            v = wrap_axis([float(nv[f"{rg}|{m}|{n}|axis"]) for n, _ in NETS])
            xx = i + (k - 0.5) * 0.3 + np.linspace(-0.05, 0.05, len(v))
            ax.scatter(xx, v, color=COL[m], s=30, label=LAB[m] if i == 0 else None, zorder=3)
    for y, t in zip(LS_AXIS, ("L&S sans FLV", "L&S avec FLV")):
        ax.axhline(y, color="k", ls=":", lw=1)
        ax.text(1.55, y, t, fontsize=7, va="center")
    ax.set_xticks([0, 1], [r[1] for r in RG])
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylabel("axe bimodal des PD (°)")
    ax.set_title("B. Axe bimodal, 6 réseaux, 8 cibles", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")

    # C. PD reversal accel -> decel across networks
    ax = fig.add_subplot(gs[0, 3])
    for i, (rg, rl) in enumerate(RG):
        for k, m in enumerate(("lin6", "nl6")):
            v = [float(nv[f"{rg}|{m}|{n}|pd_drift"]) for n, _ in NETS]
            xx = i + (k - 0.5) * 0.3 + np.linspace(-0.05, 0.05, len(v))
            ax.scatter(xx, v, color=COL[m], s=30, zorder=3)
    ax.axhline(180, color="k", ls=":", lw=1)
    ax.text(0.5, 176, "inversion complète", fontsize=7, ha="center", va="top")
    ax.set_xticks([0, 1], [r[1] for r in RG])
    ax.set_ylabel("|PD décélération − PD accélération| (°, médiane)")
    ax.set_title("C. Inversion des PD au pic de vitesse, 6 réseaux", fontsize=10)

    # D. which part of the body causes it (one network)
    ax = fig.add_subplot(gs[1, 0:2])
    models = [("lin6", "linéaire,\nsans FLV"), ("nl6x", "squelette\nnon lin. seul"),
              ("lin6L", "F-L seule"), ("lin6V", "F-V seule"), ("lin6f", "F-L + F-V"),
              ("nl6", "complet")]
    x = np.arange(len(models))
    ax2 = ax.twinx()
    for i, (rg, rl) in enumerate(RG):
        ax_v, dr_v = [], []
        for m, _ in models:
            d = S.load(rg, m, 3)
            p = S.pd_measures(d)
            ax_v.append(wrap_axis(p["axis"]))
            dr_v.append(p["pd_drift"])
        ax.bar(x + (i - 0.5) * 0.38, ax_v, 0.36, color=["#9ecae1", "#3182bd"][i],
               label=f"axe bimodal, {rl}")
        ax2.plot(x, dr_v, "o-", color=["#fdae6b", "#e6550d"][i], label=f"inversion des PD, {rl}")
    ax.set_xticks(x, [m[1] for m in models], fontsize=8)
    ax.set_ylabel("axe bimodal (°) — barres")
    ax.set_ylim(-110, 0)
    ax2.set_ylabel("inversion des PD (°) — points")
    ax2.set_ylim(60, 185)
    ax.set_title("D. Quelle partie du corps produit la signature ? (réseau n° 2, 8 cibles)",
                 fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left", ncol=2)

    # E. detectability under Poisson noise
    keys = [("axis", "axe bimodal"), ("bimodal_R", "force de bimodalité"),
            ("pd_drift", "inversion des PD"), ("acc_dec", "accél./décél."),
            ("even_frac", "part paire"), ("oop_mean", "hors du plan")]
    trials = (5, 10, 20)
    for i, (rg, rl) in enumerate(RG):
        ax = fig.add_subplot(gs[1, 2 + i])
        out = S.noise_test(rg, every=3, trials=trials, n_rep=100)
        for k, lab in keys:
            dp = []
            for tr in trials:
                a, b = out["lin6", tr][k], out["nl6", tr][k]
                dp.append(abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12))
            ax.plot(trials, dp, "o-", label=lab)
        ax.axhline(2, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(trials, [str(t) for t in trials])
        ax.set_xlabel("essais par cible")
        ax.set_ylabel("d′ (lin6 contre nl6)")
        ax.set_title(f"E{i + 1}. Détectabilité, Poisson, 8 cibles, 100 neurones — {rl}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Signatures de la non-linéarité du corps mesurables dans M1 — même réseau, "
                 "même tâche, même coût", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_signatures.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
