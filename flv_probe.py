"""Why does the force-length/velocity rung rotate the PMD distribution far less
here than in Lillicrap & Scott (2013)?

Rungs 1-5 of `fig5_centreout.py` land within a few degrees of the paper, but
rung 6 rotates the distribution by only about -8 deg where the paper reports
-28 deg.  This script splits the F-L/V rung into its two factors and reruns each
on its own, plus a variant in which the muscles' optimal angles are shifted so
that every muscle sits at l = 1 at the centre posture (our arm segments are much
longer than the monkey forelimb the L0/theta0 values were measured on, so by
default several muscles operate far off their optimal length).

    python flv_probe.py [--n_net 10] [--jobs 7]
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time

import numpy as np

import iLQG_Muscle as M
import fig5_centreout as F

BASE = dict(geometry=True, intersegmental=True, muscles="bi")

VARIANTS = [
    ("5. biarticular muscles (no F-L/V)", dict(flv=False)),
    ("6. + F-L/V  (both factors)", dict(flv=True)),
    ("6a. + force-LENGTH only", dict(flv="length")),
    ("6b. + force-VELOCITY only", dict(flv="velocity")),
    ("6c. + F-L/V, muscles recentred to l=1", dict(flv=True,
                                                   recenter_lengths=True)),
]


def solve_one(args):
    vi, ni, ti, cfg = args
    W = M.load_network(ni % 10)
    rng = np.random.default_rng(1000 + ni)
    plant = M.Plant(W, M.random_readout(6, W.shape[0], rng), **cfg)
    centre = np.array(M.CENTER_XY)
    _, targ = F.targets(centre)
    x, _, info = M.solve(plant, centre, targ[ti], duration=F.DURATION,
                         K=F.K_STEPS, w1=F.W1, w2=F.W2, r1=F.R1, qr=F.QR,
                         qa=F.QA, qp=F.QP, tol=F.TOL)
    return (vi, ni, ti, x[:, :W.shape[0]].copy(), plant.hand_velocity(x),
            info["endpoint_error"])


def main(n_net, jobs):
    N = M.load_network(0).shape[0]
    units = np.zeros((len(VARIANTS), n_net, F.N_TARGETS, F.K_STEPS + 1, N),
                     dtype=np.float32)
    hvel = np.zeros((len(VARIANTS), n_net, F.N_TARGETS, F.K_STEPS + 1, 2),
                    dtype=np.float32)
    err = np.zeros((len(VARIANTS), n_net, F.N_TARGETS))

    jobs_list = [(vi, ni, ti, {**BASE, **extra})
                 for vi, (_, extra) in enumerate(VARIANTS)
                 for ni in range(n_net) for ti in range(F.N_TARGETS)]

    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for k, (vi, ni, ti, r, v, e) in enumerate(
                ex.map(solve_one, jobs_list, chunksize=1)):
            units[vi, ni, ti], hvel[vi, ni, ti], err[vi, ni, ti] = r, v, e
            if (k + 1) % 80 == 0:
                print(f"  {k + 1}/{len(jobs_list)}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)

    targ_rel = F.targets(np.array(M.CENTER_XY))[1] - np.array(M.CENTER_XY)
    print(f"\n{'variant':40s} {'theta':>8s} {'r':>7s} {'shift vs rung 5':>16s}")
    base = None
    for vi, (name, _) in enumerate(VARIANTS):
        pds = []
        for ni in range(n_net):
            feat = F.unit_features(units[vi, ni], hvel[vi, ni])
            pd, keep, _, _ = F.planar_regression(feat, targ_rel)
            pds.append(pd[keep])
        r, th = F.bimodal_rayleigh(np.concatenate(pds))
        th = np.degrees(th)
        if base is None:
            base, shift = th, ""
        else:
            shift = f"{(th - base + 90) % 180 - 90:+.1f} deg"
        print(f"{name:40s} {th:8.1f} {r:7.3f} {shift:>16s}")
    print(f"\npaper: rung 5 -> 6 shifts -28.2 deg (127.9 -> 99.7), r 0.65 -> 0.44")
    print(f"median endpoint error {np.median(err):.4f} cm")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_net", type=int, default=10)
    p.add_argument("--jobs", type=int, default=7)
    a = p.parse_args()
    main(a.n_net, a.jobs)
