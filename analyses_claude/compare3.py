"""Three bodies, one controller, one task, one cost:

    lin2   linearized arm (frozen inertia, no Coriolis) + 2 joint-torque actuators
    nl2    nonlinear arm + 2 joint-torque actuators
    nl6    nonlinear arm + 6 Lillicrap & Scott muscles (rectified) + F-L/V

All actuators have the same first-order filter (tau_act = 60 ms). All three use
the cost of iLQG_Combined -- terminal cost on JOINT angles/velocities, no
straight-path term -- so lin2 is exactly linear-quadratic and its activity is
affine in the joint target (the rank-3 theorem).

    python analyses_claude/compare3.py            # simulate + analyse
    python analyses_claude/compare3.py --analyse  # analyse cached sims only
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

M = MP.M
OUT = os.path.join(HERE, "sims")
N_DIR, K, RADIUS = 24, 100, 12.0
REGIMES = {"12cm_030s": 0.3, "12cm_050s": 0.5}
# every duration a solver may be asked for (worker processes re-import this module)
DURATIONS = {**REGIMES, "12cm_040s": 0.4}
MODELS = {"lin2": dict(linear=True, muscles="none", flv=False),
          "nl2": dict(linear=False, muscles="none", flv=False),
          "nl6": dict(linear=False, muscles="bi", flv=True),
          "lin6": dict(linear=True, muscles="bi", flv=False),
          "nl6x": dict(linear=False, muscles="bi", flv=False),     # skeleton only
          "lin6f": dict(linear=True, muscles="bi", flv=True),      # F-L/V only
          "lin6L": dict(linear=True, muscles="bi", flv="length"),  # F-L only
          "lin6V": dict(linear=True, muscles="bi", flv="velocity"),  # F-V only
          # Lillicrap & Scott Fig 5 rungs not covered above (nl2 = + intersegmental,
          # nl6x = + biarticular, nl6 = + F-L/V)
          "pm": dict(linear=False, muscles="none", flv=False, intersegmental=False,
                     geometry=False),        # 2-D point mass: target given in hand space
          "geo": dict(linear=False, muscles="none", flv=False, intersegmental=False),
          "mono": dict(linear=False, muscles="mono", flv=False)}
SOLVER = dict(w1=1e4, w2=10.0, r1=1e-5, qp=0.0, tol=1e-6, max_iter=200)


class JointCost:
    """Terminal cost 0.5 w1 |theta - theta*|^2 + 0.5 w2 |omega|^2 (iLQG_Combined)."""

    def terminal_residual(self, x, ptarg, w1, w2):
        s = x[self.i_s]
        J = np.zeros((4, self.n))
        R = np.concatenate([np.sqrt(w1) * (s[:2] - ptarg), np.sqrt(w2) * s[2:]])
        J[0:2, self.N:self.N + 2] = np.sqrt(w1) * np.eye(2)
        J[2:4, self.N + 2:self.N + 4] = np.sqrt(w2) * np.eye(2)
        return R, J


class JointPlant(JointCost, M.Plant):
    pass


class JointLinearPlant(JointCost, MP.LinearArmPlant):
    pass


def make(model, network=2, seed=0):
    cfg = MODELS[model]
    W = M.load_network(network, path=os.path.join(MP.PROJ, "gaussian_networks.hdf5"))
    n_act = {"none": 2, "mono": 4, "bi": 6}[cfg["muscles"]]
    Wout = M.random_readout(n_act, W.shape[0], np.random.default_rng(seed))
    cls = JointLinearPlant if cfg["linear"] else JointPlant
    return cls(W, Wout, geometry=cfg.get("geometry", True),
               intersegmental=cfg.get("intersegmental", True),
               muscles=cfg["muscles"], flv=cfg["flv"])


def targets():
    ang = np.linspace(0, 2 * np.pi, N_DIR, endpoint=False)
    c = np.array(M.CENTER_XY)
    xy = c + RADIUS * np.stack([np.cos(ang), np.sin(ang)], 1)
    return ang, xy, np.array([M.inverse_kinematics(*p) for p in xy])


def solve_one(job):
    regime, model, i = job
    plant = make(model)
    _, xy, jt = targets()
    x, u, info = M.solve(plant, M.CENTER_XY, jt[i], duration=REGIMES[regime], K=K, **SOLVER)
    hand = plant.hand(x)
    return job, x, u, hand, float(np.hypot(*(hand[-1] - xy[i]))), info["iters"]


def simulate(jobs, models=None):
    os.makedirs(OUT, exist_ok=True)
    ang, xy, jt = targets()
    models = list(MODELS) if models is None else models
    work = [(rg, m, i) for rg in REGIMES for m in models for i in range(N_DIR)]
    res = {(rg, m): [None] * N_DIR for rg in REGIMES for m in models}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for job, x, u, hand, err, it in ex.map(solve_one, work, chunksize=1):
            res[job[:2]][job[2]] = (x, u, hand, err, it)
    for (rg, m), lst in res.items():
        err = np.array([r[3] for r in lst])
        np.savez_compressed(
            os.path.join(OUT, f"cmp_{rg}_{m}.npz"),
            states=np.stack([r[0] for r in lst]), controls=np.stack([r[1] for r in lst]),
            hand=np.stack([r[2] for r in lst]), ang=ang, targ=xy, targ_joint=jt,
            t=np.linspace(0, REGIMES[rg], K + 1),
            N=lst[0][0].shape[1] - 4 - {"none": 2, "mono": 4, "bi": 6}[MODELS[m]["muscles"]],
            duration=REGIMES[rg], K=K, center=np.array(M.CENTER_XY), radius=RADIUS,
            model=m, endpoint_error=err, iters=np.array([r[4] for r in lst]))
        print(f"{rg} {m:4s} max endpoint err {err.max():.2e} cm, iters {min(r[4] for r in lst)}-"
              f"{max(r[4] for r in lst)}")
    print(f"simulated in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=14)
    p.add_argument("--analyse", action="store_true")
    p.add_argument("--models", nargs="+", default=["lin2", "nl2", "nl6"],
                   help=f"models to simulate, among {list(MODELS)}")
    a = p.parse_args()
    if not a.analyse:
        simulate(a.jobs, a.models)
    if set(a.models) == {"lin2", "nl2", "nl6"}:
        import compare3_analysis
        compare3_analysis.main()
