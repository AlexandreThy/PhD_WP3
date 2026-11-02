"""Minimal version of the macaque Fig. 5 overlay (Cheng & Scott arm, 20 networks):
unit-PMD row only, no title, no r, reference axis labelled 'ref'. Reads the caches
of fig5_macaque.py and writes to WP3/meeting fred/.

    python analyses_claude/fig5_minimal.py          # todorov and kalidindi
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "meeting fred")


def draw():
    sys.path.insert(0, HERE)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import macaque
    import compare3 as C
    import fig5_overlay as F
    import net_variability as NV

    rg, tag = macaque.REGIME, f"{macaque.VISCOSITY}_{macaque.PARAMS}_n20"
    F.NETS = NV.NETS20
    C.REGIMES = {rg: C.DURATIONS[rg]}
    z = dict(np.load(os.path.join(HERE, "sims", f"fig5_overlay_macaque_{tag}.npz")))
    rungs = F.RUNGS
    paper = F.PAPER_UNIT

    fig, axs = plt.subplots(1, len(rungs), figsize=(16 * len(rungs) / 6, 3.4),
                            subplot_kw=dict(projection="polar"))
    for j, ((m, lab), ax) in enumerate(zip(rungs, axs)):
        pd, _ = F.pmds(z, rg, m, "r")
        _, th, p = F.bimodal_test(pd)
        F.star(ax, pd, p < 1e-3)
        if paper[m] is not None:
            F.axis_line(ax, paper[m][0], "red", "-", 2.2, "ref" if j == 1 else None)
        if p < 1e-3:
            F.axis_line(ax, th, "#2171b5", "--", 2.2, "iLQG" if j == 1 else None)
        F.polar_clean(ax)
        ref = f"ref θ={paper[m][0]:.0f}°" if paper[m] else "ref : n.s."
        ax.set_title(f"{lab}\n{ref}\niLQG " + (f"θ={th:.0f}°" if p < 1e-3 else "n.s."), fontsize=8)
    fig.legend(loc="upper left", bbox_to_anchor=(0.005, 0.99), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0.04, 0, 1, 1))
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"fig5_overlay_6cm_040s_macaque_{macaque.VISCOSITY}_cheng_scott.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    if "MACAQUE_VISCOSITY" in os.environ:
        draw()
    else:   # macaque.py patches at import: one process per viscosity
        for visc in ("todorov", "kalidindi"):
            env = {**os.environ, "MACAQUE_VISCOSITY": visc, "MACAQUE_PARAMS": "cheng_scott",
                   "PYTHONIOENCODING": "utf-8"}
            subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, check=True)
