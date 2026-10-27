"""Do the lin6 / nl6 differences survive a change of network and readout?

Re-solves both models for several (network, Wout seed) pairs, 8 targets, and
recomputes the PD and population measures of signatures.py. Saves
sims/netvar.npz.

    python analyses_claude/net_variability.py
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compare3 as C  # noqa: E402
import signatures as S  # noqa: E402

NETS = [(0, 10), (1, 11), (3, 13), (4, 14), (5, 15), (6, 16)]   # (network, Wout seed)
# all 10 g=0.8 networks of the HDF5, each with two readouts (contains NETS)
NETS20 = [(n, 10 + n) for n in range(10)] + [(n, 20 + n) for n in range(10)]


def netkey(n, s):
    """Cache key of a (network, Wout seed) pair; the original pairs keep the bare
    network index so that existing caches stay readable."""
    return f"{n}" if s == 10 + n else f"{n}-{s}"
MODELS = ("lin6", "nl6")
TARGETS = np.arange(0, C.N_DIR, 3)                               # 8 targets


def solve_one(job):
    regime, model, net, seed, i = job
    plant = C.make(model, network=net, seed=seed)
    _, xy, jt = C.targets()
    x, _, _ = C.M.solve(plant, C.M.CENTER_XY, jt[i], duration=C.REGIMES[regime], K=C.K, **C.SOLVER)
    hand = plant.hand(x)
    return job, x[:, :plant.N], hand, float(np.hypot(*(hand[-1] - xy[i])))


def main(jobs=14):
    work = [(rg, m, n, s, i) for rg in C.REGIMES for m in MODELS for n, s in NETS for i in TARGETS]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(solve_one, work, chunksize=1))
    ang = np.linspace(0, 2 * np.pi, C.N_DIR, endpoint=False)[TARGETS]
    out = {}
    for rg in C.REGIMES:
        dur = C.REGIMES[rg]
        for m in MODELS:
            for n, s in NETS:
                sel = [r for r in res if r[0][:4] == (rg, m, n, s)]
                sel.sort(key=lambda r: r[0][4])
                r = np.stack([x[1] for x in sel])
                hand = np.stack([x[2] for x in sel])
                err = max(x[3] for x in sel)
                v = np.gradient(hand, dur / (r.shape[1] - 1), axis=1)
                speed = np.linalg.norm(v, axis=-1).mean(0)
                d = dict(r=r, ang=ang, speed=speed, t=np.linspace(0, dur, r.shape[1]),
                         dur=dur, k_peak=int(np.argmax(speed)))
                vals = {**S.pd_measures(d), **S.population_measures(d)}
                for k, v in vals.items():
                    out[f"{rg}|{m}|{n}|{k}"] = v
                out[f"{rg}|{m}|{n}|err"] = err
    np.savez(os.path.join(HERE, "sims", "netvar.npz"), **out)
    keys = ["bimodal_R", "axis", "pd_drift", "cis_frac", "acc_dec", "err"]
    for rg in C.REGIMES:
        print("==", rg)
        for m in MODELS:
            print(f"  {m:5s} " + " | ".join(
                f"{k}: " + " ".join(f"{out[f'{rg}|{m}|{n}|{k}']:.2f}" for n, _ in NETS) for k in keys))


if __name__ == "__main__":
    main()
