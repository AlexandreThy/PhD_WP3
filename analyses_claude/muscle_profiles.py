"""Why does 'muscles can only pull' double the erank of network activity?

Compares the muscle activations a(theta, t) and the network drive to the muscles
d = Wout r between lin6u (push-pull) and lin6c (pull-only), which differ ONLY by
the rectification sigma_u (linearized arm, F-L/V frozen, same network, readout,
cost and targets). Reads the caches of fig_erank_muscles.py (12 cm / 0.30 s).

Measures
    erank across targets at each time (median over the movement) along the chain
        joint torque -> muscle activation -> drive Wout r -> network activity r
    temporal complexity of each muscle: erank of its (targets x time) matrix
    tuning: harmonic content of the onset -> peak-speed activation
    co-activation: summed |activation| per unit of net torque

    python analyses_claude/muscle_profiles.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fig_erank_muscles as F  # noqa: E402
from erank import entropy_rank  # noqa: E402

RG = "12cm_030s"
NAMES = ["épaule fléch.", "épaule ext.", "coude fléch.", "coude ext.",
         "biart. fléch.", "biart. ext."]
SHOW = np.arange(0, 24, 3)                      # 8 of the 24 directions


def unpack(m):
    p = F.make(m)
    r, d = F.load(RG, m)
    X = d["states"]
    a = X[:, :, p.i_a]
    tau = np.array([[p.torque(X[i, k, p.i_s], X[i, k, p.i_a])[0]
                     for k in range(X.shape[1])] for i in range(X.shape[0])])
    speed = np.linalg.norm(np.gradient(d["hand"], axis=1), axis=2)
    return dict(r=r, a=a, drive=r @ p.Wout.T, tau=tau, speed=speed, ang=d["ang"],
                t=np.linspace(0, float(d["duration"]), X.shape[1]) * 1000)


def erank_across(Y):
    """Median over time of the erank of the (targets x channels) matrix."""
    return float(np.median([entropy_rank(np.linalg.svd(Y[:, k] - Y[:, k].mean(0),
                                                       compute_uv=False))
                            for k in range(1, Y.shape[1])]))


def temporal_erank(a):
    """Per muscle: erank of the (targets x time) matrix of its activation."""
    return np.array([entropy_rank(np.linalg.svd(a[:, :, i] - a[:, :, i].mean(),
                                                compute_uv=False)) for i in range(a.shape[2])])


def tuning(z):
    """Onset -> peak-speed mean activation per (target, muscle)."""
    out = []
    for i in range(len(z["ang"])):
        sp = z["speed"][i]
        kp = int(sp.argmax())
        k0 = int(np.argmax(sp > 0.1 * sp[kp]))
        out.append(z["a"][i, k0:kp + 1].mean(0))
    return np.array(out)


def harmonics(feat, ang):
    """Share (%) of the non-constant tuning power in harmonics 1, 2, 3+."""
    X = np.stack([np.ones_like(ang)] + [f(h * ang) for h in range(1, 6)
                                        for f in (np.cos, np.sin)], 1)
    b, *_ = np.linalg.lstsq(X, feat, rcond=None)
    P = np.array([b[2 * h - 1] ** 2 + b[2 * h] ** 2 for h in range(1, 6)])
    P = P / P.sum(0)
    return 100 * np.stack([P[0], P[1], P[2:].sum(0)])


def main():
    models = {"lin6u": "poussent et tirent", "lin6c": "tirent seulement"}
    Z = {m: unpack(m) for m in models}
    chain = {m: [erank_across(Z[m][k]) for k in ("tau", "a", "drive", "r")] for m in models}
    for extra in ("nl6c", "nl6"):
        z = unpack(extra)
        chain[extra] = [erank_across(z[k]) for k in ("tau", "a", "drive", "r")]

    lines = ["erank à travers les 24 cibles (médiane sur le mouvement), 12 cm / 0,30 s",
             f"{'modèle':7s} {'couple':>7s} {'activation':>10s} {'commande':>9s} {'réseau':>7s}"]
    for m, c in chain.items():
        lines.append(f"{m:7s} " + " ".join(f"{v:9.2f}" for v in c))
    for m in models:
        z = Z[m]
        h = harmonics(tuning(z), z["ang"])
        coc = np.abs(z["a"]).sum(2).max() / np.linalg.norm(z["tau"], axis=2).max()
        lines.append(f"\n{m} ({models[m]}): fraction a<0 = {(z['a'] < 0).mean():.2f}, "
                     f"somme des |activations| / |couple| au pic = {coc:.3f}")
        lines.append("  complexité temporelle (erank cibles x temps) par muscle: "
                     + " ".join(f"{v:.2f}" for v in temporal_erank(z["a"])))
        lines.append("  accord (début -> pic de vitesse), % de puissance harmonique 1 / 2 / 3+ :")
        for i, n in enumerate(NAMES):
            lines.append(f"    {n:14s} {h[0, i]:5.1f} {h[1, i]:5.1f} {h[2, i]:5.1f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "muscle_profiles_table.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 1.1], hspace=0.55, wspace=0.35)
    cols = plt.cm.hsv(np.linspace(0, 1, len(SHOW), endpoint=False))
    for row, m in enumerate(models):
        z = Z[m]
        for i in range(6):
            ax = fig.add_subplot(gs[row, i])
            for c, k in zip(cols, SHOW):
                ax.plot(z["t"], z["a"][k, :, i], color=c, lw=1.4)
            ax.axhline(0, color="0.6", lw=0.6)
            kp = int(z["speed"].mean(0).argmax())
            ax.axvline(z["t"][kp], color="0.4", ls=":", lw=0.8)
            ax.set_title(NAMES[i], fontsize=9)
            if i == 0:
                ax.set_ylabel(f"activation\n({models[m]})", fontsize=9)
            if row == 1:
                ax.set_xlabel("temps (ms)", fontsize=8)
            ax.tick_params(labelsize=7)

    for i in range(6):
        ax = fig.add_subplot(gs[2, i], projection="polar")
        for m, col in (("lin6u", "0.15"), ("lin6c", "0.55")):
            z = Z[m]
            f = tuning(z)[:, i]
            ang = np.append(z["ang"], z["ang"][0])
            ax.plot(ang, np.append(f, f[0]) / np.abs(f).max(), color=col, lw=1.8,
                    label=models[m] if i == 0 else None)
        ax.plot(np.linspace(0, 2 * np.pi, 100), np.zeros(100), color="crimson", lw=0.6)
        ax.set_ylim(-1, 1.05)
        ax.set_yticks([])
        ax.set_xticks(np.radians([0, 90, 180, 270]), ["0°", "90°", "180°", "270°"], fontsize=7)
        ax.set_title(f"accord : {NAMES[i]}", fontsize=8.5, pad=12)
    fig.legend(*fig.axes[-6].get_legend_handles_labels(), loc="center left",
               bbox_to_anchor=(0.005, 0.40), fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[3, 0:3])
    labels = ["couple\narticulaire", "activation\nmusculaire", "commande\nWout·r", "activité\ndu réseau r"]
    x = np.arange(4)
    style = {"lin6u": ("0.15", "linéaire, poussent et tirent"),
             "lin6c": ("0.55", "linéaire, tirent seulement"),
             "nl6c": ("#3182bd", "bras non linéaire, F-L/V figée"),
             "nl6": ("crimson", "bras non linéaire, F-L/V complète")}
    w = 0.2
    for j, (m, (col, lab)) in enumerate(style.items()):
        ax.bar(x + (j - 1.5) * w, chain[m], w, color=col, label=lab)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("erank à travers les cibles\n(médiane sur le mouvement)")
    ax.set_title("La dimensionnalité naît entre le couple et l'activation musculaire, "
                 "puis passe telle quelle au réseau", fontsize=10)
    ax.legend(fontsize=8, ncol=2)

    ax = fig.add_subplot(gs[3, 3:6])
    w = 0.38
    for j, (m, col) in enumerate((("lin6u", "0.15"), ("lin6c", "0.55"))):
        ax.bar(np.arange(6) + (j - 0.5) * w, temporal_erank(Z[m]["a"]), w, color=col,
               label=models[m])
    ax.axhline(2, color="crimson", ls=":", lw=1, label="2 = profils séparables (théorie linéaire)")
    ax.set_xticks(np.arange(6), NAMES, fontsize=8, rotation=20)
    ax.set_ylabel("erank (cibles x temps)")
    ax.set_title("Complexité temporelle de chaque muscle", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Activations musculaires, 12 cm / 0,30 s : bras linéarisé et F-L/V figée, "
                 "seule change la contrainte de tirer seulement\n(lignes 1-2 : 8 directions, "
                 "couleur = direction, pointillé = pic de vitesse ; ligne 3 : accord début → pic, "
                 "normalisé, cercle rouge = 0)", fontsize=11.5)
    out = os.path.join(HERE, "fig_muscle_profiles.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
