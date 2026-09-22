"""Center-out reaching with the network + 6-muscle arm model. Solves N_DIR
radial targets in parallel and caches the trajectories for analysis."""
import os
import sys
import time
import numpy as np
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

CENTER = (0.0, 40.0)
RADIUS = 12.0
DURATION = 0.5
K = 100
N_DIR = 8
CACHE = os.path.join(_HERE, "centerout_muscle.npz")


def _solve(args):
    tgt, start, dur, K = args
    import ilqg_muscle as M
    X, Y, x, u = M.simulate_ILQG(targets=list(tgt), start=list(start),
                                 Duration=dur, K=K, tol=1e-4, max_iter=120)
    return x, u, np.stack([X, Y], 1)


def run(n_dir=N_DIR, center=CENTER, radius=RADIUS, dur=DURATION, K=K):
    import ilqg_muscle as M
    ang = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    targ = np.stack([center[0] + radius * np.cos(ang),
                     center[1] + radius * np.sin(ang)], 1)
    Nn = M.N
    t0 = time.time()
    with Pool(processes=min(os.cpu_count(), 12)) as pool:
        res = pool.map(_solve, [(targ[i], center, dur, K) for i in range(n_dir)])
    states = np.stack([r[0] for r in res])
    controls = np.stack([r[1] for r in res])
    hand = np.stack([r[2] for r in res])
    err = np.hypot(hand[:, -1, 0] - targ[:, 0], hand[:, -1, 1] - targ[:, 1])
    np.savez_compressed(CACHE, states=states, controls=controls, hand=hand,
                        ang=ang, targ=targ, N=Nn, K=K, duration=dur,
                        center=np.array(center), radius=radius)
    print("saved %s : %d targets, max endpoint err %.4f cm  (%.0fs)"
          % (CACHE, n_dir, err.max(), time.time() - t0))
    return err


if __name__ == "__main__":
    run()
