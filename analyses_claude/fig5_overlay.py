"""Lillicrap & Scott (2013) Figure 5, centre-out rows, redrawn from the closed-loop
iLQG model and overlaid with the paper's bimodal axes.

Row A: unit PMDs for the six rungs (point mass -> + F-L/V), 16-bin polar 'star'
histograms as in the paper (black = significant bimodal distribution, grey =
not), with the paper's axis (red) and ours (blue).
Row B: muscle PMDs for the three muscle rungs (grey circles: monoarticular
shoulder; black diamonds: monoarticular elbow; open triangles: biarticular).

Model: random fixed network + iLQG, joint-space cost of compare3.py (hand-space
for the point mass), 6 networks pooled, 16 targets, 12 cm.

    python analyses_claude/fig5_overlay.py
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compare3 as C  # noqa: E402
import signatures as S  # noqa: E402
from net_variability import NETS, netkey  # noqa: E402

RUNGS = [("pm", "1  2D point mass"), ("geo", "2  + geometry"),
         ("nl2", "3  + intersegmental\ndynamics"), ("mono", "4  + monoarticular\nmuscles"),
         ("nl6x", "5  + biarticular\nmuscles"), ("nl6", "6  + muscle\nmechanics")]
# paper, unit PMDs (theta, r) and muscle PMDs; None = not significant
PAPER_UNIT = {"pm": None, "geo": (108.8, 0.61), "nl2": (131.4, 0.67), "mono": (131.4, 0.67),
              "nl6x": (127.9, 0.65), "nl6": (99.7, 0.44)}
PAPER_MUSC = {"mono": (165.0, 0.27), "nl6x": (122.9, 0.58), "nl6": (103.8, 0.71)}
N_T = 16
ANG = np.linspace(0, 2 * np.pi, N_T, endpoint=False)
CACHE = os.path.join(HERE, "sims", "fig5_overlay.npz")
TITLE_EXTRA = ""


def solve_one(job):
    regime, model, net, seed, i = job
    plant = C.make(model, network=net, seed=seed)
    c = np.array(C.M.CENTER_XY)
    xy = c + C.RADIUS * np.array([np.cos(ANG[i]), np.sin(ANG[i])])
    target = xy if model == "pm" else np.array(C.M.inverse_kinematics(*xy))
    x, _, _ = C.M.solve(plant, C.M.CENTER_XY, target, duration=C.DURATIONS[regime], K=C.K,
                        **C.SOLVER)
    hand = plant.hand(x)
    return job, x[:, :plant.N], x[:, plant.i_a], hand, float(np.hypot(*(hand[-1] - xy)))


def simulate(jobs=14):
    work = [(rg, m, n, s, i) for rg in C.REGIMES for m, _ in RUNGS for n, s in NETS
            for i in range(N_T)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(solve_one, work, chunksize=1))
    out = {}
    for rg in C.REGIMES:
        for m, _ in RUNGS:
            for n, s in NETS:
                sel = sorted((r for r in res if r[0][:4] == (rg, m, n, s)), key=lambda r: r[0][4])
                k = netkey(n, s)
                out[f"{rg}|{m}|{k}|r"] = np.stack([x[1] for x in sel]).astype(np.float32)
                out[f"{rg}|{m}|{k}|a"] = np.stack([x[2] for x in sel]).astype(np.float32)
                out[f"{rg}|{m}|{k}|hand"] = np.stack([x[3] for x in sel])
                out[f"{rg}|{m}|{k}|err"] = max(x[4] for x in sel)
    np.savez_compressed(CACHE, **out)


def pmds(z, rg, m, key, nets=None):
    """PMDs (rad) of units ('r') or muscles ('a') pooled over networks (default:
    all of NETS), plus the muscle index of each PMD for 'a'."""
    dur = C.DURATIONS[rg]
    pds, idx = [], []
    for n, s in (NETS if nets is None else nets):
        y = z[f"{rg}|{m}|{netkey(n, s)}|{key}"]
        hand = z[f"{rg}|{m}|{netkey(n, s)}|hand"]
        v = np.gradient(hand, dur / (y.shape[1] - 1), axis=1)
        kp = int(np.argmax(np.linalg.norm(v, axis=-1).mean(0)))
        pd, _, r2 = S.cosine_fit(S.epoch_mean(y, 1, kp + 1), ANG)
        keep = r2 > 0.5 if key == "r" else np.ones_like(r2, bool)
        pds.append(pd[keep])
        idx.append(np.arange(len(pd))[keep])
    return np.concatenate(pds), np.concatenate(idx)


def bimodal_test(pd, n_boot=2000, seed=0):
    R, ax = S.bimodal(pd)
    rng = np.random.default_rng(seed)
    null = np.abs(np.exp(2j * rng.uniform(-np.pi, np.pi, (n_boot, len(pd)))).mean(1))
    return R, ax % 180.0, float((null >= R).mean())


def axis_line(ax, theta_deg, color, ls, lw, label=None):
    t = np.radians(theta_deg)
    ax.plot([t, t + np.pi], [1.05, 1.05], color=color, ls=ls, lw=lw, label=label, zorder=5)


def star(ax, pd, sig):
    counts, edges = np.histogram(np.mod(pd, 2 * np.pi), bins=N_T, range=(0, 2 * np.pi))
    centres = edges[:-1] + np.pi / N_T
    col = "k" if sig else "0.6"
    for c, n in zip(centres, counts / max(counts.max(), 1)):
        ax.plot([c, c], [0, n], color=col, lw=1.6)


def muscle_markers(ax, pd, idx):
    """Stack one marker per muscle PMD along its 16-bin direction, as in the paper."""
    style = {0: dict(marker="o", mfc="0.55", mec="0.55"), 1: dict(marker="o", mfc="0.55", mec="0.55"),
             2: dict(marker="D", mfc="k", mec="k"), 3: dict(marker="D", mfc="k", mec="k"),
             4: dict(marker="^", mfc="w", mec="k"), 5: dict(marker="^", mfc="w", mec="k")}
    b = np.floor(np.mod(pd, 2 * np.pi) / (2 * np.pi / N_T)).astype(int)
    height = np.zeros(N_T)
    for k in np.argsort(idx, kind="stable"):
        height[b[k]] += 1
        ax.plot((b[k] + 0.5) * 2 * np.pi / N_T, 0.12 * height[b[k]], ls="none", ms=4,
                **style[int(idx[k])])


def polar_clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1.1)
    ax.spines["polar"].set_visible(False)


def figure(z, rg, suffix=""):
    fig = plt.figure(figsize=(16 * len(RUNGS) / 6, 6.4))
    gs = fig.add_gridspec(2, len(RUNGS), height_ratios=[1, 1])
    for j, (m, lab) in enumerate(RUNGS):
        ax = fig.add_subplot(gs[0, j], projection="polar")
        pd, _ = pmds(z, rg, m, "r")
        R, th, p = bimodal_test(pd)
        star(ax, pd, p < 1e-3)
        if PAPER_UNIT[m] is not None:
            axis_line(ax, PAPER_UNIT[m][0], "red", "-", 2.2, "Lillicrap & Scott" if j == 1 else None)
        if p < 1e-3:
            axis_line(ax, th, "#2171b5", "--", 2.2, "iLQG (ce modèle)" if j == 1 else None)
        polar_clean(ax)
        pap = PAPER_UNIT[m]
        ptxt = f"L&S θ={pap[0]:.0f}° r={pap[1]:.2f}" if pap else "L&S : n.s."
        ax.set_title(f"{lab}\n{ptxt}\niLQG θ={th:.0f}° r={R:.2f}" + ("" if p < 1e-3 else " (n.s.)"),
                     fontsize=8)
        if j == 1:
            fig.legend(loc="upper left", bbox_to_anchor=(0.005, 0.99), fontsize=8, frameon=False)
    fig.text(0.005, 0.72, "A  centre-out\nPMD des unités", fontsize=9, fontweight="bold")

    for j, (m, lab) in enumerate(RUNGS):
        if m not in PAPER_MUSC:
            continue
        ax = fig.add_subplot(gs[1, j], projection="polar")
        pd, idx = pmds(z, rg, m, "a")
        R, th, _ = bimodal_test(pd)
        muscle_markers(ax, pd, idx)
        ax.set_ylim(0, max(1.1, ax.get_ylim()[1]))
        top = ax.get_ylim()[1]
        for t, col, ls in ((PAPER_MUSC[m][0], "red", "-"), (th, "#2171b5", "--")):
            tt = np.radians(t)
            ax.plot([tt, tt + np.pi], [top] * 2, color=col, ls=ls, lw=2.2, zorder=5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)
        ax.set_title(f"L&S θ={PAPER_MUSC[m][0]:.0f}° r={PAPER_MUSC[m][1]:.2f}\n"
                     f"iLQG θ={th:.0f}° r={R:.2f}", fontsize=8)
    fig.text(0.005, 0.30, "B  centre-out\nPMD des muscles\n\n● mono épaule\n◆ mono coude\n△ biarticulaires",
             fontsize=9)
    fig.suptitle(f"Fig. 5 de Lillicrap & Scott (centre-out) refaite avec iLQG en boucle fermée — "
                 f"{C.RADIUS:.0f} cm / {C.DURATIONS[rg]:.2f} s, {len(NETS)} réseaux, 16 cibles{TITLE_EXTRA}",
                 fontsize=11)
    fig.tight_layout(rect=(0.06, 0, 1, 0.95))
    out = os.path.join(HERE, f"fig5_overlay_{rg}{suffix}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def main():
    if not os.path.exists(CACHE):
        simulate()
    z = np.load(CACHE)
    for rg in C.REGIMES:
        print(rg, "max endpoint err (cm):",
              {m: round(max(float(z[f'{rg}|{m}|{netkey(n, s)}|err']) for n, s in NETS), 4) for m, _ in RUNGS})
        figure(z, rg)


if __name__ == "__main__":
    main()
