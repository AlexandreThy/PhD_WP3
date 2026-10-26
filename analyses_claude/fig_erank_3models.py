"""erank of the muscle model: linearized arm vs nonlinear arm without / with F-L/V.

Muscle model (iLQG_Muscle, 6 muscles), 12 cm / 0.30 s, 24 targets
(sims/muscle_12cm_030s_*.npz from muscle_centerout.py).

    1. global erank (Roy & Vetterli) of the network activity
    2. the same, split into direction-odd and direction-even components

    python analyses_claude/fig_erank_3models.py
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

MODELS = [("lin_none", "bras linéarisé", "0.6"),
          ("nl_none", "non linéaire,\nsans FLV", "#3182bd"),
          ("nl_both", "non linéaire,\navec FLV", "crimson")]


def main():
    glob_, odd, even = [], [], []
    for key, _, _ in MODELS:
        d = np.load(os.path.join(HERE, "sims", f"muscle_12cm_030s_{key}.npz"))
        r = d["states"][:, :, :int(d["N"])]
        ev, od = even_odd(r)
        glob_.append(measures(r)[0])
        even.append(measures(ev)[0])
        odd.append(measures(od)[0])

    labels = [m[1] for m in MODELS]
    colors = [m[2] for m in MODELS]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))

    x = np.arange(len(MODELS))
    ax[0].bar(x, glob_, 0.6, color=colors)
    for xi, v in zip(x, glob_):
        ax[0].text(xi, v + 0.1, f"{v:.2f}", ha="center")
    ax[0].set_xticks(x, labels)
    ax[0].set_ylabel("erank (Roy & Vetterli)")
    ax[0].set_ylim(0, max(glob_) * 1.15)
    ax[0].set_title("A. erank global de l'activité")

    ax[1].bar(x - 0.2, odd, 0.4, color=colors, alpha=0.45, label="impair")
    ax[1].bar(x + 0.2, even, 0.4, color=colors, label="pair")
    for xi, o, e in zip(x, odd, even):
        ax[1].text(xi - 0.2, o + 0.1, f"{o:.2f}", ha="center", fontsize=9)
        ax[1].text(xi + 0.2, e + 0.1, f"{e:.2f}", ha="center", fontsize=9)
    ax[1].set_xticks(x, labels)
    ax[1].set_ylabel("erank (Roy & Vetterli)")
    ax[1].set_ylim(0, max(even) * 1.15)
    ax[1].set_title("B. erank des composantes impaire (clair) et paire (foncé)")

    fig.suptitle("Modèle musculaire, 12 cm / 0,30 s, 24 cibles — même réseau, même coût", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_erank_3models.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    for (k, _, _), g, o, e in zip(MODELS, glob_, odd, even):
        print(f"{k:9s} global {g:.2f}  impair {o:.2f}  pair {e:.2f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
