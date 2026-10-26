"""Center-out reaching with the 6-muscle model: F-L/V variants x linear/nonlinear arm.

Saves one file per condition in analyses_claude/sims/:
    muscle_{regime}_{arm}_{flv}.npz     arm in {nl, lin}, flv in {none, length, velocity, both}

    python analyses_claude/muscle_centerout.py [--jobs 14]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import muscle_plant as MP  # noqa: E402

OUT = os.path.join(HERE, "sims")

N_DIR = 24
K = 100
RADIUS = 12.0                      # cm; 20 cm leaves the workspace from CENTER_XY
REGIMES = {"12cm_050s": 0.5, "12cm_030s": 0.3}
FLV = {"none": False, "length": "length", "velocity": "velocity", "both": True}
SOLVER = dict(w1=1e4, w2=10.0, r1=1e-5, qp=0.5, tol=1e-4, max_iter=150)


def targets():
    ang = np.linspace(0, 2 * np.pi, N_DIR, endpoint=False)
    c = np.array(MP.M.CENTER_XY)
    return ang, c + RADIUS * np.stack([np.cos(ang), np.sin(ang)], 1)


def solve_one(job):
    regime, arm, fname, i = job
    plant = MP.make_plant(arm == "lin", FLV[fname])
    _, targ = targets()
    x, u, info = MP.M.solve(plant, MP.M.CENTER_XY, targ[i],
                            duration=REGIMES[regime], K=K, **SOLVER)
    return job, x, u, plant.hand(x), info


def main(jobs):
    os.makedirs(OUT, exist_ok=True)
    ang, targ = targets()
    conds = [(rg, arm, fl) for rg in REGIMES for arm in ("nl", "lin") for fl in FLV]
    work = [(rg, arm, fl, i) for rg, arm, fl in conds for i in range(N_DIR)]
    res = {c: [None] * N_DIR for c in conds}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for k, (job, x, u, hand, info) in enumerate(ex.map(solve_one, work, chunksize=1)):
            res[job[:3]][job[3]] = (x, u, hand, info)
            if (k + 1) % 48 == 0:
                print(f"  {k + 1}/{len(work)}  {(time.time() - t0) / 60:.1f} min", flush=True)

    for (rg, arm, fl), lst in res.items():
        err = np.array([r[3]["endpoint_error"] for r in lst])
        np.savez_compressed(
            os.path.join(OUT, f"muscle_{rg}_{arm}_{fl}.npz"),
            states=np.stack([r[0] for r in lst]), controls=np.stack([r[1] for r in lst]),
            hand=np.stack([r[2] for r in lst]), ang=ang, targ=targ,
            t=np.linspace(0, REGIMES[rg], K + 1), N=lst[0][0].shape[1] - 4 - 6,
            duration=REGIMES[rg], K=K, center=np.array(MP.M.CENTER_XY), radius=RADIUS,
            linear=(arm == "lin"), flv=fl, endpoint_error=err,
            iters=np.array([r[3]["iters"] for r in lst]))
        print(f"{rg} {arm:3s} {fl:8s} max endpoint err {err.max():.2e} cm")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


BAD = 0.1                          # cm; endpoint error above this = failed reach
MAX_LAT = 2.0                      # cm; lateral deviation / overshoot above this = aberrant path


def valid(hand, targ, err):
    """Reach converged AND took a sane path: some 'converged' slow reaches
    without F-V reach the target only after flailing far from it."""
    hand = np.atleast_3d(hand) if hand.ndim == 2 else hand
    hand, targ, err = np.reshape(hand, (-1,) + hand.shape[-2:]), np.atleast_2d(targ), np.atleast_1d(err)
    c = np.array(MP.M.CENTER_XY)
    u = (targ - c) / np.linalg.norm(targ - c, axis=1, keepdims=True)
    nrm = np.stack([-u[:, 1], u[:, 0]], 1)
    lat = np.abs(np.einsum("ktj,kj->kt", hand - c, nrm)).max(1)
    far = np.linalg.norm(hand - c, axis=-1).max(1)
    return (err < BAD) & (lat < MAX_LAT) & (far < RADIUS + MAX_LAT)


def repair_one(job):
    """Re-solve one failed reach, warm-started from each converged F-L/V variant
    of the same arm/regime/target; keep the lowest-cost converged solution."""
    regime, arm, fl, i = job
    plant = MP.make_plant(arm == "lin", FLV[fl])
    _, targ = targets()
    best = None
    for src in ("both", "velocity", "none", "length"):
        if src == fl:
            continue
        d = np.load(os.path.join(OUT, f"muscle_{regime}_{arm}_{src}.npz"))
        if not valid(d["hand"][i], targ[i], d["endpoint_error"][i])[0]:
            continue
        x, u, info = MP.solve_warm(plant, MP.M.CENTER_XY, targ[i], d["controls"][i],
                                   duration=REGIMES[regime], K=K, **SOLVER)
        ok = valid(plant.hand(x), targ[i], info["endpoint_error"])[0]
        if ok and (best is None or info["cost"] < best[3]["cost"]):
            best = (x, u, plant.hand(x), info)
    return job, best


def repair(jobs):
    files = sorted(f for f in os.listdir(OUT) if f.startswith("muscle_"))
    work = []
    for f in files:
        _, rg_a, rg_b, arm, fl = f[:-4].split("_")
        d = np.load(os.path.join(OUT, f))
        ok = valid(d["hand"], d["targ"], d["endpoint_error"])
        work += [(f"{rg_a}_{rg_b}", arm, fl, int(i)) for i in np.where(~ok)[0]]
    print(f"repairing {len(work)} reaches")
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        fixed = list(ex.map(repair_one, work, chunksize=1))
    for (rg, arm, fl, i), best in fixed:
        path = os.path.join(OUT, f"muscle_{rg}_{arm}_{fl}.npz")
        d = dict(np.load(path, allow_pickle=True))
        if best is None:
            print(f"  {rg} {arm} {fl} target {i}: still failed")
            continue
        x, u, hand, info = best
        d["states"][i], d["controls"][i], d["hand"][i] = x, u, hand
        d["endpoint_error"][i], d["iters"][i] = info["endpoint_error"], info["iters"]
        d.setdefault("warm_started", np.zeros(N_DIR, bool))
        d["warm_started"] = np.asarray(d["warm_started"]).copy()
        d["warm_started"][i] = True
        np.savez_compressed(path, **d)
        print(f"  {rg} {arm} {fl} target {i}: err {info['endpoint_error']:.1e} cm")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=14)
    p.add_argument("--repair", action="store_true",
                   help="only re-solve failed reaches, warm-started")
    a = p.parse_args()
    if not a.repair:
        main(a.jobs)
    repair(a.jobs)
