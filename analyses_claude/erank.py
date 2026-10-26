"""Effective rank (erank) of network activity: linearized-arm null vs nonlinear arm.

Three measures on the same activity matrix X (samples x units, each unit
mean-centred across condition x time, as in figures.movement_pca):

    erank_RV  = exp(H(p)),  p_i = s_i / sum(s)        Roy & Vetterli (2007)
    erank_var = exp(H(p)),  p_i = s_i^2 / sum(s^2)    the "PCA effective rank" of README_iLQG.md
    PR        = (sum s^2)^2 / sum s^4                 participation ratio

Also: erank_RV of the across-condition matrix r[:, t, :] at each time, and of
the direction-even / direction-odd components (opposite targets averaged /
differenced).

    python analyses_claude/erank.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

# (label, nonlinear file, linear file, epoch) -- epoch 'move' drops the prep phase
REGIMES = [
    ("12 cm / 0,50 s", "centerout_nl24.npz", "centerout_lin24.npz", "move"),
    ("20 cm / 0,35 s", "centerout_nl24_big.npz", "centerout_lin24_big.npz", "move"),
    ("10 cm / 0,65 s", "prep_small_nl8.npz", "prep_small_lin8.npz", "move"),
    ("20 cm / 0,75 s", "prep_slow_nl8.npz", "prep_slow_lin8.npz", "move"),
    ("20 cm / 0,65 s", "prep_nl8.npz", "prep_lin8.npz", "move"),
    ("20 cm / 0,55 s", "prep_fast_nl8.npz", "prep_fast_lin8.npz", "move"),
    ("22 cm / 0,65 s", "prep_large_nl8.npz", "prep_large_lin8.npz", "move"),
]


def entropy_rank(p):
    p = p[p > 0]
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def measures(r):
    """r: (n_dir, T, N) -> erank_RV, erank_var, PR on the pooled matrix."""
    X = r.reshape(-1, r.shape[2])
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    lam = s ** 2
    return entropy_rank(s), entropy_rank(lam), float(lam.sum() ** 2 / (lam ** 2).sum())


def activity(fname, epoch):
    d = np.load(os.path.join(PROJ, "data", fname), allow_pickle=True)
    N = int(d["N"])
    r = d["states"][:, :, :N]
    if epoch == "move" and "k_prep" in d.files:
        r = r[:, int(d["k_prep"]):, :]
    return r, d


def even_odd(r):
    half = r.shape[0] // 2
    return (r[:half] + r[half:]) / 2.0, (r[:half] - r[half:]) / 2.0


def erank_t(r):
    out = []
    for k in range(1, r.shape[1]):
        X = r[:, k, :] - r[:, k, :].mean(0, keepdims=True)
        out.append(entropy_rank(np.linalg.svd(X, compute_uv=False)))
    return np.array(out)


def main():
    rows = []
    for label, fnl, flin, ep in REGIMES:
        rnl, dnl = activity(fnl, ep)
        rli, _ = activity(flin, ep)
        enl, onl = even_odd(rnl)
        eli, oli = even_odd(rli)
        rows.append(dict(
            label=label, n_dir=rnl.shape[0],
            nl=measures(rnl), lin=measures(rli),
            even_nl=measures(enl)[0], even_lin=measures(eli)[0],
            odd_nl=measures(onl)[0], odd_lin=measures(oli)[0]))

    hdr = (f"{'régime':16s} {'cibles':>6s} | {'erank_RV nl':>11s} {'lin':>6s} | "
           f"{'erank_var nl':>12s} {'lin':>6s} | {'PR nl':>6s} {'lin':>6s} | "
           f"{'pair nl':>7s} {'lin':>6s} | {'impair nl':>9s} {'lin':>6s}")
    lines = [hdr, "-" * len(hdr)]
    for w in rows:
        lines.append(
            f"{w['label']:16s} {w['n_dir']:6d} | {w['nl'][0]:11.2f} {w['lin'][0]:6.2f} | "
            f"{w['nl'][1]:12.2f} {w['lin'][1]:6.2f} | {w['nl'][2]:6.2f} {w['lin'][2]:6.2f} | "
            f"{w['even_nl']:7.2f} {w['even_lin']:6.2f} | {w['odd_nl']:9.2f} {w['odd_lin']:6.2f}")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(HERE, "erank_table.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    x = np.arange(len(rows))
    lab = [w["label"] for w in rows]

    ax[0].bar(x - 0.2, [w["lin"][0] for w in rows], 0.4, color="0.6", label="bras linéarisé (nul)")
    ax[0].bar(x + 0.2, [w["nl"][0] for w in rows], 0.4, color="crimson", label="bras non linéaire")
    ax[0].plot(x - 0.2, [w["lin"][1] for w in rows], "k_", ms=14, mew=2, label="erank_var (README)")
    ax[0].plot(x + 0.2, [w["nl"][1] for w in rows], "k_", ms=14, mew=2)
    ax[0].set_xticks(x, lab, rotation=35, ha="right", fontsize=8)
    ax[0].set_ylabel("erank")
    ax[0].set_title("A. erank de l'activité (barres : Roy & Vetterli)")
    ax[0].legend(fontsize=8)

    rnl, _ = activity("centerout_nl24_big.npz", "move")
    rli, d = activity("centerout_lin24_big.npz", "move")
    t = d["t"][1:] * 1000
    ax[1].plot(t, erank_t(rli), "--", color="0.4", label="bras linéarisé (nul)")
    ax[1].plot(t, erank_t(rnl), color="crimson", label="bras non linéaire")
    ax[1].set_xlabel("temps (ms)")
    ax[1].set_ylabel("erank à travers les 24 cibles")
    ax[1].set_title("B. erank instantané (20 cm / 0,35 s)")
    ax[1].legend(fontsize=8)

    ax[2].bar(x - 0.3, [w["odd_lin"] for w in rows], 0.2, color="0.75", label="impair, nul")
    ax[2].bar(x - 0.1, [w["odd_nl"] for w in rows], 0.2, color="salmon", label="impair, non linéaire")
    ax[2].bar(x + 0.1, [w["even_lin"] for w in rows], 0.2, color="0.4", label="pair, nul")
    ax[2].bar(x + 0.3, [w["even_nl"] for w in rows], 0.2, color="darkred", label="pair, non linéaire")
    ax[2].set_xticks(x, lab, rotation=35, ha="right", fontsize=8)
    ax[2].set_ylabel("erank (Roy & Vetterli)")
    ax[2].set_title("C. erank des composantes paire et impaire")
    ax[2].legend(fontsize=7, ncol=2)

    fig.suptitle("Même réseau, même coût, mêmes cibles ; seul le bras change", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_erank.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
