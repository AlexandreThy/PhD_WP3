"""Linearized-arm null vs nonlinear arm: condition rank and dR2_even.

    python analyses_claude/fig_null_model.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

proj = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj)
import predictions

nl = np.load(f"{proj}/centerout_nl24_big.npz", allow_pickle=True)
li = np.load(f"{proj}/centerout_lin24_big.npz", allow_pickle=True)
N = int(nl["N"])
t = nl["t"] * 1000

def svals(d, k):
    r = d["states"][:, :, :N]
    s = np.linalg.svd(r[:, k, :], compute_uv=False)
    return s / s[0]

def rank_t(d, tol=1e-3):
    r = d["states"][:, :, :N]
    return np.array([(lambda s: (s > tol * s[0]).sum())(np.linalg.svd(r[:, k, :], compute_uv=False))
                     for k in range(1, r.shape[1])])

k_mid = len(t) // 2
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

i = np.arange(1, 25)
ax[0].semilogy(i, svals(li, k_mid), "o--", color="0.4", label="bras linéarisé (modèle nul)")
ax[0].semilogy(i, svals(nl, k_mid), "o-", color="crimson", label="bras non linéaire")
ax[0].axvline(3.5, color="k", lw=0.8, ls=":")
ax[0].text(3.7, 1e-1, "plafond\nthéorique = 3", fontsize=9)
ax[0].set_xlabel("valeur singulière n°")
ax[0].set_ylabel("σ_i / σ_1")
ax[0].set_title(f"A. Activité à travers les 24 cibles (t = {t[k_mid]:.0f} ms)")
ax[0].legend(fontsize=8)

ax[1].plot(t[1:], rank_t(li), color="0.4", ls="--", label="bras linéarisé")
ax[1].plot(t[1:], rank_t(nl), color="crimson", label="bras non linéaire")
ax[1].set_xlabel("temps (ms)")
ax[1].set_ylabel("rang de condition (seuil 1e-3)")
ax[1].set_title("B. Rang de condition au cours du mouvement")
ax[1].legend(fontsize=8)

dur = float(nl["duration"])
terms = [("coriolis", "Coriolis C"), ("Mvar", "variation d'inertie\ndM·θ̈"), ("both", "les deux")]
v_nl = [predictions.term_dr2(nl["states"], N, dur, term=k, even=True) for k, _ in terms]
v_li = [predictions.term_dr2(li["states"], N, dur, term=k, even=True) for k, _ in terms]
v_nl = [float(v[0]) for v in v_nl]
v_li = [float(v[0]) for v in v_li]
x = np.arange(3)
ax[2].bar(x - 0.2, v_li, 0.4, color="0.6", label="bras linéarisé (modèle nul)")
ax[2].bar(x + 0.2, v_nl, 0.4, color="crimson", label="bras non linéaire")
for xi, a, b in zip(x, v_li, v_nl):
    ax[2].text(xi - 0.2, a + 0.003, f"{a:.3f}", ha="center", fontsize=8)
    ax[2].text(xi + 0.2, b + 0.003, f"{b:.3f}", ha="center", fontsize=8)
ax[2].set_xticks(x, [l for _, l in terms])
ax[2].set_ylabel("ΔR² validé (composante paire)")
ax[2].set_title("C. Gain des régresseurs non linéaires\nau-delà de la cinématique")
ax[2].legend(fontsize=8)

fig.suptitle("20 cm / 0,35 s, 24 cibles — même réseau, même coût ; seul le bras change", fontsize=11)
fig.tight_layout()
out = f"{proj}/analyses_claude/fig_null_model.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(out, "ranks lin/nl median:", np.median(rank_t(li)), np.median(rank_t(nl)), "dR2", v_li, v_nl)
