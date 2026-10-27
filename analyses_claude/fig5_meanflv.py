"""Rung 6 the way Lillicrap & Scott built it: F-L/V as a fixed scaling factor.

Their static model (Experimental Procedures, "Static Model Specification")
replaces F-L/V by a factor H[muscle, direction] = f_l * f_fv averaged "from
movement onset to 100 ms after peak velocity" over the optimal solution of the
full dynamic model. Here:

  1. H is computed per network, target and muscle from the full nl6 solutions
     (sims/fig5_overlay.npz, joint angles recovered from the hand path by IK);
  2. each reach is re-solved with a plant whose tension is F_max * a * H[:, target]
     -- constant along the movement, no dependence on instantaneous length or
     velocity (nonlinear arm, 6 rectified muscles, same cost);
  3. Fig. 5 is redrawn with both rung-6 versions side by side.

    python analyses_claude/fig5_meanflv.py
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
import fig5_overlay as F  # noqa: E402
from net_variability import NETS, netkey  # noqa: E402

M = C.M
CACHE = os.path.join(HERE, "sims", "fig5_meanflv.npz")
AFTER_PEAK = 0.100            # s, end of the averaging window after peak velocity


class MeanFLVPlant(C.JointPlant):
    """6 muscles, tension = F_max * a * h with h fixed for the whole reach."""

    def __init__(self, *args, h, **kw):
        super().__init__(*args, **kw)
        self.h_flv = np.asarray(h, dtype=float)

    def torque(self, s, a):
        tension = M.F_MAX * a * self.h_flv
        z = np.zeros((2, 2))
        return self.moment_m @ tension, (z, z, self.moment_m * (M.F_MAX * self.h_flv))


def mean_flv(z, rg, k):
    """H (16 targets, 6 muscles) from the full-model (nl6) solution of network key k."""
    dur = C.DURATIONS[rg]
    hand = z[f"{rg}|nl6|{k}|hand"]                           # (16, K+1, 2)
    dt = dur / (hand.shape[1] - 1)
    th = np.stack(M.inverse_kinematics(hand[..., 0], hand[..., 1]), -1)
    om = np.gradient(th, dt, axis=1)
    geom = C.make("nl6")
    l = geom.l_const - th @ geom.A_len
    v = -om @ geom.A_len
    phi = M.flv(l, v)[0]                                     # (16, K+1, 6)
    speed = np.linalg.norm(np.gradient(hand, dt, axis=1), axis=-1)
    H = np.zeros((hand.shape[0], 6))
    for i in range(hand.shape[0]):
        k_end = min(int(np.argmax(speed[i])) + int(round(AFTER_PEAK / dt)), hand.shape[1] - 1)
        H[i] = phi[i, :k_end + 1].mean(0)
    return H


def solve_one(job):
    regime, net, seed, i, h = job
    base = C.make("nl6x", network=net, seed=seed)            # same W, Wout as the other rungs
    plant = MeanFLVPlant(base.W, base.Wout, geometry=True, intersegmental=True,
                         muscles="bi", flv=False, h=h)
    c = np.array(M.CENTER_XY)
    xy = c + C.RADIUS * np.array([np.cos(F.ANG[i]), np.sin(F.ANG[i])])
    x, _, _ = M.solve(plant, M.CENTER_XY, np.array(M.inverse_kinematics(*xy)),
                      duration=C.DURATIONS[regime], K=C.K, **C.SOLVER)
    hand = plant.hand(x)
    return job[:4], x[:, :plant.N], x[:, plant.i_a], hand, float(np.hypot(*(hand[-1] - xy)))


def simulate(z, jobs=14):
    Hs = {(rg, n, s): mean_flv(z, rg, netkey(n, s)) for rg in C.REGIMES for n, s in NETS}
    work = [(rg, n, s, i, Hs[rg, n, s][i]) for rg in C.REGIMES for n, s in NETS
            for i in range(F.N_T)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(solve_one, work, chunksize=1))
    out = {}
    for rg in C.REGIMES:
        for n, s in NETS:
            sel = sorted((r for r in res if r[0][:3] == (rg, n, s)), key=lambda r: r[0][3])
            k = netkey(n, s)
            out[f"{rg}|nl6m|{k}|r"] = np.stack([x[1] for x in sel]).astype(np.float32)
            out[f"{rg}|nl6m|{k}|a"] = np.stack([x[2] for x in sel]).astype(np.float32)
            out[f"{rg}|nl6m|{k}|hand"] = np.stack([x[3] for x in sel])
            out[f"{rg}|nl6m|{k}|err"] = max(x[4] for x in sel)
            out[f"{rg}|nl6m|{k}|H"] = Hs[rg, n, s]
    np.savez_compressed(CACHE, **out)


def main():
    z0 = dict(np.load(F.CACHE))
    if not os.path.exists(CACHE):
        simulate(z0)
    z = {**z0, **dict(np.load(CACHE))}

    for rg in C.REGIMES:
        H = np.stack([z[f"{rg}|nl6m|{netkey(n, s)}|H"] for n, s in NETS])
        print(f"{rg}: H range {H.min():.2f}-{H.max():.2f}, mean per muscle "
              f"{np.round(H.mean((0, 1)), 2)}, max endpoint err "
              f"{max(float(z[f'{rg}|nl6m|{netkey(n, s)}|err']) for n, s in NETS):.1e} cm")

    F.RUNGS = F.RUNGS[:5] + [("nl6", "6  + muscle mechanics\n(FLV instantanée)"),
                             ("nl6m", "6'  + muscle mechanics\n(FLV moyenne, comme L&S)")]
    F.PAPER_UNIT["nl6m"] = F.PAPER_UNIT["nl6"]
    F.PAPER_MUSC["nl6m"] = F.PAPER_MUSC["nl6"]
    for rg in C.REGIMES:
        F.figure(z, rg, suffix="_meanflv")
        for m in ("nl6", "nl6m"):
            pu, _ = F.pmds(z, rg, m, "r")
            pa, _ = F.pmds(z, rg, m, "a")
            Ru, tu, _ = F.bimodal_test(pu)
            Ra, ta, _ = F.bimodal_test(pa)
            print(f"  {rg} {m:5s} units θ={tu:6.1f} r={Ru:.2f} | muscles θ={ta:6.1f} r={Ra:.2f}")


if __name__ == "__main__":
    main()
