"""Center-out reaching with the combined random-network + nonlinear-arm model,
controlled by iLQG (finite horizon, starting at movement onset -- no
preparatory epoch). Reproduces the movement-related analyses of Kalidindi &
Crevecoeur (2025), Fig. 2, with a nonlinear two-link arm.

Running this module solves the reaches for `N_DIR` radially arranged targets and
caches the full network + limb trajectories to an .npz file, so that the
plotting scripts can be re-run instantly without re-solving.
"""

import os
import time

import numpy as np

import iLQG_Combined as M

# -----------------------------------------------------------------------------
# Task / solver configuration
# -----------------------------------------------------------------------------
CENTER = (0.0, 45.0)   # hand start position [cm]
RADIUS = 12.0          # reach distance [cm]
DURATION = 0.5         # movement duration [s]
K = 100                # time steps (dt = 5 ms)
W1, W2, R1 = 1e4, 10.0, 1e-5   # terminal-pos, terminal-vel, effort costs
N_DIR = 8              # number of radial targets

CACHE = os.path.join(
    os.environ.get("CENTEROUT_CACHE_DIR", os.path.dirname(os.path.abspath(__file__))),
    "centerout_data.npz",
)


def target_positions(n_dir=N_DIR, center=CENTER, radius=RADIUS):
    """Radial target hand positions (cm), starting at 0 rad (rightward, +x)."""
    ang = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    xs = center[0] + radius * np.cos(ang)
    ys = center[1] + radius * np.sin(ang)
    return ang, np.stack([xs, ys], axis=1)


def run(n_dir=N_DIR, duration=DURATION, K=K, w1=W1, w2=W2, r1=R1,
        center=CENTER, radius=RADIUS, verbose=True, linearize=False, tol=1e-6):
    """Solve the center-out task; return a dict of trajectories/arrays.

    If linearize=True the arm is frozen to a linear plant (inertia fixed at the
    start elbow angle, no Coriolis) -- the "no-nonlinearity" baseline.
    """
    N = M.W.shape[0]
    ang, targ = target_positions(n_dir, center, radius)

    M.LINEARIZE = linearize
    if linearize:
        M.THETA2_REF = M.compute_angles_from_cartesian(center[0], center[1])[1]

    states = np.zeros((n_dir, K + 1, N + 6))   # full state per target
    controls = np.zeros((n_dir, K, N))         # network inputs
    hand = np.zeros((n_dir, K + 1, 2))         # cartesian hand position

    t0 = time.time()
    for i in range(n_dir):
        X, Y, x, u = M.simulate_ILQG(
            Duration=duration, w1=w1, w2=w2, r1=r1,
            targets=list(targ[i]), start=list(center),
            K=K, tol=tol, print_iterations=False,
        )
        states[i], controls[i] = x, u
        hand[i, :, 0], hand[i, :, 1] = X, Y
        if verbose:
            err = np.hypot(X[-1] - targ[i, 0], Y[-1] - targ[i, 1])
            print(f"  target {i:2d}  dir {np.degrees(ang[i]):6.1f} deg  "
                  f"endpoint err {err:.4f} cm  ({time.time()-t0:5.1f}s)")

    t = np.linspace(0, duration, K + 1)
    return {
        "states": states, "controls": controls, "hand": hand,
        "t": t, "ang": ang, "targ": targ, "N": N,
        "duration": duration, "K": K, "center": np.array(center),
        "radius": radius,
    }


def save(data, path=CACHE):
    np.savez_compressed(path, **data)
    print("saved", path)


def load(path=CACHE):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n_dir", type=int, default=N_DIR)
    p.add_argument("--out", type=str, default=CACHE)
    p.add_argument("--linear", action="store_true",
                   help="use the linearized-arm baseline plant")
    p.add_argument("--tol", type=float, default=1e-6)
    args = p.parse_args()

    kind = "linear" if args.linear else "nonlinear"
    print(f"Solving center-out task, {args.n_dir} targets "
          f"({M.W.shape[0]} nodes, {kind} arm, iLQG)...")
    data = run(n_dir=args.n_dir, linearize=args.linear, tol=args.tol)
    save(data, args.out)
