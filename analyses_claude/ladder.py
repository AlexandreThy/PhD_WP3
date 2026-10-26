"""Lillicrap & Scott (2013) Figure 5 ladder, redone with the closed-loop iLQG model.

Rungs (all: nonlinear arm, joint-space cost of compare3.py, same network/readout):
    geometry        2 torque actuators, diagonal inertia, no Coriolis
    intersegmental  2 torque actuators, full arm dynamics           (= nl2)
    mono            4 monoarticular muscles, no F-L/V
    bi              6 muscles, no F-L/V                              (= nl6x)
    flv             6 muscles + F-L/V                                (= nl6)
Unit PMDs: cosine fit of activity averaged from movement onset to peak speed
(the paper's epoch), units with R2 > 0.5, pooled over networks as in the paper.
Axis in the paper's convention (0-180 deg).

    python analyses_claude/ladder.py
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
from net_variability import NETS  # noqa: E402

RUNGS = [("geo", "+ géométrie"), ("nl2", "+ dynamique\nintersegmentaire"),
         ("mono", "+ muscles\nmonoarticulaires"), ("nl6x", "+ muscles\nbiarticulaires"),
         ("nl6", "+ FLV")]
PAPER = {"geo": (108.8, 0.61), "nl2": (131.4, 0.67), "mono": (131.4, 0.67),
         "nl6x": (127.9, 0.65), "nl6": (99.7, 0.44)}
M1_DATA = (98.4, 0.37)                 # their Fig 1C, n = 395
N_T = 16
ANG = np.linspace(0, 2 * np.pi, N_T, endpoint=False)
CACHE = os.path.join(HERE, "sims", "ladder.npz")


def solve_one(job):
    regime, model, net, seed, i = job
    plant = C.make(model, network=net, seed=seed)
    c = np.array(C.M.CENTER_XY)
    xy = c + C.RADIUS * np.array([np.cos(ANG[i]), np.sin(ANG[i])])
    jt = np.array(C.M.inverse_kinematics(*xy))
    x, _, _ = C.M.solve(plant, C.M.CENTER_XY, jt, duration=C.REGIMES[regime], K=C.K, **C.SOLVER)
    hand = plant.hand(x)
    return job, x[:, :plant.N], hand, float(np.hypot(*(hand[-1] - xy)))


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
                out[f"{rg}|{m}|{n}|r"] = np.stack([x[1] for x in sel])
                out[f"{rg}|{m}|{n}|hand"] = np.stack([x[2] for x in sel])
                out[f"{rg}|{m}|{n}|err"] = max(x[3] for x in sel)
    np.savez_compressed(CACHE, **out)


def paper_axis(a_deg):
    return np.asarray(a_deg) % 180.0


def pooled_pmds(z, rg, m):
    dur = C.REGIMES[rg]
    pds, per_net = [], []
    for n, _ in NETS:
        r, hand = z[f"{rg}|{m}|{n}|r"], z[f"{rg}|{m}|{n}|hand"]
        v = np.gradient(hand, dur / (r.shape[1] - 1), axis=1)
        kp = int(np.argmax(np.linalg.norm(v, axis=-1).mean(0)))
        pd, _, r2 = S.cosine_fit(S.epoch_mean(r, 1, kp + 1), ANG)
        pds.append(pd[r2 > 0.5])
        R, ax = S.bimodal(pd[r2 > 0.5])
        per_net.append((paper_axis(ax), R))
    R, ax = S.bimodal(np.concatenate(pds))
    return paper_axis(ax), R, np.array(per_net), float(max(z[f"{rg}|{m}|{n}|err"] for n, _ in NETS))


def main():
    if not os.path.exists(CACHE):
        simulate()
    z = np.load(CACHE)
    res = {(rg, m): pooled_pmds(z, rg, m) for rg in C.REGIMES for m, _ in RUNGS}

    lines = [f"{'échelon':18s} | {'papier θ':>8s} {'r':>5s} | " + " | ".join(
        f"{rg}: θ  r  (θ min-max sur réseaux)" for rg in C.REGIMES)]
    for m, lab in RUNGS:
        p = PAPER[m]
        cells = []
        for rg in C.REGIMES:
            ax, R, pn, err = res[rg, m]
            cells.append(f"{ax:6.1f} {R:4.2f} ({pn[:, 0].min():5.1f}-{pn[:, 0].max():5.1f})")
        lines.append(f"{lab.replace(chr(10), ' '):18s} | {p[0]:8.1f} {p[1]:5.2f} | " + " | ".join(cells))
    lines.append(f"{'M1 (leur Fig 1C)':18s} | {M1_DATA[0]:8.1f} {M1_DATA[1]:5.2f} |")
    table = "\n".join(lines)
    sys.stdout.reconfigure(encoding="utf-8")
    print(table)
    with open(os.path.join(HERE, "ladder_table.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(RUNGS))
    ax[0].plot(x, [PAPER[m][0] for m, _ in RUNGS], "ks--", label="Lillicrap & Scott, Fig. 5")
    ax[1].plot(x, [PAPER[m][1] for m, _ in RUNGS], "ks--", label="Lillicrap & Scott, Fig. 5")
    for rg, col in zip(C.REGIMES, ("crimson", "#3182bd")):
        axs = [res[rg, m][0] for m, _ in RUNGS]
        Rs = [res[rg, m][1] for m, _ in RUNGS]
        lo = [res[rg, m][2][:, 0].min() for m, _ in RUNGS]
        hi = [res[rg, m][2][:, 0].max() for m, _ in RUNGS]
        lab = f"iLQG boucle fermée, {C.REGIMES[rg]:.2f} s"
        ax[0].plot(x, axs, "o-", color=col, label=lab)
        ax[0].fill_between(x, lo, hi, color=col, alpha=0.15)
        ax[1].plot(x, Rs, "o-", color=col, label=lab)
    for a in ax:
        a.axhline(M1_DATA[0] if a is ax[0] else M1_DATA[1], color="green", ls=":", lw=1.5,
                  label="M1 enregistré (Fig. 1C)")
        a.set_xticks(x, [r[1] for r in RUNGS], fontsize=8)
        a.legend(fontsize=8)
    ax[0].set_ylabel("axe bimodal des PMD (°, 0–180)")
    ax[0].set_title("A. Axe (bande : min–max sur 6 réseaux)")
    ax[1].set_ylabel("force de bimodalité r (PMD regroupées)")
    ax[1].set_title("B. Force de bimodalité")
    fig.suptitle("Échelle de la Fig. 5 de Lillicrap & Scott refaite en boucle fermée "
                 "(6 réseaux, 16 cibles, 12 cm)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_ladder.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
