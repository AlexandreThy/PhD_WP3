"""Preparatory-epoch variant of the combined random-network + nonlinear-arm iLQG
model.

This file leaves `iLQG_Combined.py` untouched: the plant -- network `W`, readout
`Wout`, the two-link arm, the `f`/`fx`/`fu` dynamics, and the `LINEARIZE` flag --
is imported unchanged, so the physics here is provably identical to the main
model. What differs is only the COST, which becomes a two-phase prepare-to-reach
objective in the spirit of Kalidindi & Crevecoeur (2026), Eqs. 19-20:

  * PREP phase (first `k_prep` steps): a running cost holds the arm at the START
    posture (penalises joint-angle deviation from start and joint velocity). The
    network is free to build up activity, but any activity that reads out to
    torque (`Wout r != 0`) moves the arm and is penalised -- so preparation is
    pushed into the OUTPUT-NULL subspace (Kaufman et al. 2014).
  * MOVE phase (remaining steps): running control cost only; a TERMINAL cost
    pulls the arm to the target with zero terminal velocity.

A single finite-horizon iLQG solve over [prep + move] lets the network anticipate
the target during preparation, producing target-specific preparatory states that
spread the conditions before movement onset -- the ingredient the main model
lacks (there every condition starts at r = 0, giving a jPCA rosette).
"""
import numpy as np

import iLQG_Combined as base
from iLQG_Combined import (
    W, Wout, f, fx, fu, hand_xy, compute_angles_from_cartesian,
)

N = W.shape[0]

# -----------------------------------------------------------------------------
# prepare-to-reach schedule and cost weights (uniform dt across both phases)
# -----------------------------------------------------------------------------
K_PREP = 0            # preparatory steps
K_MOVE = 70            # movement steps
DURATION = 0.65        # total horizon [s]  -> dt = 5 ms
W_HOLD = 1e4           # running cost holding joint angle at start during prep
W_VHOLD = 1e2          # running cost on joint velocity during prep
W1 = 1e4              # terminal target-angle cost
W2 = 10.0             # terminal target-velocity cost
R1 = 1e-5             # control (network input) cost


# -----------------------------------------------------------------------------
# two-phase running cost  (control cost everywhere; posture-hold during prep)
# -----------------------------------------------------------------------------
def running_cost(x, u, k, r1, w_hold, w_vhold, xstart, k_prep):
    c = r1 * np.sum(u ** 2) / 2.0
    if k < k_prep:
        q = x[-6:]
        c += (w_hold / 2.0) * ((q[0] - xstart[0]) ** 2 + (q[1] - xstart[1]) ** 2)
        c += (w_vhold / 2.0) * (q[2] ** 2 + q[3] ** 2)
    return c


def running_lx(x, k, w_hold, w_vhold, xstart, k_prep):
    g = np.zeros(len(x))
    if k < k_prep:
        q = x[-6:]
        g[-6] = w_hold * (q[0] - xstart[0])
        g[-5] = w_hold * (q[1] - xstart[1])
        g[-4] = w_vhold * q[2]
        g[-3] = w_vhold * q[3]
    return g


def running_lxx(n, k, w_hold, w_vhold, k_prep):
    Q = np.zeros((n, n))
    if k < k_prep:
        Q[-6, -6] = w_hold
        Q[-5, -5] = w_hold
        Q[-4, -4] = w_vhold
        Q[-3, -3] = w_vhold
    return Q


# -----------------------------------------------------------------------------
# iLQG backward-model build (mirrors base.step2 but with the two-phase cost)
# -----------------------------------------------------------------------------
def step2_prep(x, u, Duration, k_prep, w1, w2, r1, w_hold, w_vhold, xstart, xtarg):
    K = u.shape[0]
    dt = Duration / K
    n, m = len(x[0]), len(u[0])
    A, B = np.zeros((K, n, n)), np.zeros((K, n, m))
    q, qbold = np.zeros(K + 1), np.zeros((K + 1, n))
    r, Q, R = np.zeros((K, m)), np.zeros((K + 1, n, n)), np.zeros((K, m, m))
    for i in range(K):
        A[i] = np.identity(n) + dt * fx(x[i], u[i])
        B[i] = dt * fu(x[i], u[i])
        q[i] = dt * running_cost(x[i], u[i], i, r1, w_hold, w_vhold, xstart, k_prep)
        qbold[i] = dt * running_lx(x[i], i, w_hold, w_vhold, xstart, k_prep)
        r[i] = dt * base.lu(x[i], u[i], r1)
        Q[i] = dt * running_lxx(n, i, w_hold, w_vhold, k_prep)
        R[i] = dt * base.luu(x[i], u[i], r1)
    q[-1] = base.h(x[-1], w1, w2, xtarg)
    qbold[-1] = base.hx(x[-1], w1, w2, xtarg)
    Q[-1] = base.hxx(x[-1], w1, w2)
    return A, B, q, qbold, r, Q, R


def total_cost_prep(x, u, Duration, k_prep, w1, w2, r1, w_hold, w_vhold,
                    xstart, xtarg):
    K = u.shape[0]
    dt = Duration / K
    J = base.h(x[-1], w1, w2, xtarg)
    for k in range(K):
        J += dt * running_cost(x[k], u[k], k, r1, w_hold, w_vhold, xstart, k_prep)
    return J


# -----------------------------------------------------------------------------
# solver
# -----------------------------------------------------------------------------
def simulate_prep(targets, start, duration=DURATION, k_prep=K_PREP, k_move=K_MOVE,
                  w1=W1, w2=W2, r1=R1, w_hold=W_HOLD, w_vhold=W_VHOLD,
                  eps=1e-6, max_iter=300, tol=1e-4, print_iterations=False):
    """Solve the prepare-to-reach problem. Returns X, Y (hand), x (full state
    trajectory, K+1 x N+6), u (inputs), and k_prep (the prep/move split index)."""
    K = k_prep + k_move
    obj1, obj2 = compute_angles_from_cartesian(targets[0], targets[1])
    st1, st2 = compute_angles_from_cartesian(start[0], start[1])
    xtarg = np.array([obj1, obj2])
    xstart = np.array([st1, st2])

    x0 = np.concatenate([np.zeros(N), [st1, st2, 0, 0, 0, 0]])
    u = np.zeros((K, N))
    x = base.step1(x0, u, duration)
    Jcost = total_cost_prep(x, u, duration, k_prep, w1, w2, r1, w_hold, w_vhold,
                            xstart, xtarg)

    alphas = 0.5 ** np.arange(12)
    for iterate in range(max_iter):
        A, B, q, qbold, r, Q, R = step2_prep(
            x, u, duration, k_prep, w1, w2, r1, w_hold, w_vhold, xstart, xtarg)
        l_gain, L_gain = base.step3(A, B, None, None, q, qbold, r, Q, R, eps)

        improved = False
        for alpha in alphas:
            xnew, unew = base.forward_pass(x0, x, u, l_gain, L_gain, duration, alpha)
            Jnew = total_cost_prep(xnew, unew, duration, k_prep, w1, w2, r1,
                                   w_hold, w_vhold, xstart, xtarg)
            if np.isfinite(Jnew) and Jnew < Jcost:
                improved = True
                break
        if not improved:
            if print_iterations:
                print(f"converged (no improvement) at iter {iterate}, cost {Jcost:.6g}")
            break

        du = np.max(np.abs(unew - u))
        rel = (Jcost - Jnew) / max(abs(Jcost), 1e-12)
        x, u, Jcost = xnew, unew, Jnew
        if du < 1e-10 or rel < tol:
            if print_iterations:
                print(f"converged at iter {iterate}, cost {Jcost:.6g}")
            break

    X, Y = hand_xy(x)
    return X, Y, x, u, k_prep


# -----------------------------------------------------------------------------
# batch solve over radial targets + caching  (mirrors centerout.run/save/load)
# -----------------------------------------------------------------------------
def target_positions(n_dir=8, center=(0.0, 40.0), radius=20.0):
    ang = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    xs = center[0] + radius * np.cos(ang)
    ys = center[1] + radius * np.sin(ang)
    return ang, np.stack([xs, ys], axis=1)


def run(n_dir=8, center=(0.0, 40.0), radius=20.0, linearize=False, tol=1e-4,
        verbose=True):
    """Solve the prepare-to-reach task for `n_dir` radial targets (serial).

    For speed, callers can instead parallelise `simulate_prep` across targets in
    a process pool (each worker sets base.LINEARIZE / base.THETA2_REF); this
    serial version is the reference implementation.
    """
    base.LINEARIZE = linearize
    base.THETA2_REF = compute_angles_from_cartesian(center[0], center[1])[1]
    ang, targ = target_positions(n_dir, center, radius)
    K = K_PREP + K_MOVE
    states = np.zeros((n_dir, K + 1, N + 6))
    hand = np.zeros((n_dir, K + 1, 2))
    kp = K_PREP
    for i in range(n_dir):
        X, Y, x, u, kp = simulate_prep(list(targ[i]), list(center), tol=tol)
        states[i] = x
        hand[i, :, 0], hand[i, :, 1] = X, Y
        if verbose:
            err = np.hypot(X[-1] - targ[i, 0], Y[-1] - targ[i, 1])
            print(f"  target {i} dir {np.degrees(ang[i]):6.1f} deg  endpoint err {err:.4f} cm")
    return {"states": states, "hand": hand, "ang": ang, "targ": targ, "N": N,
            "k_prep": kp, "K": K, "duration": DURATION,
            "center": np.array(center), "radius": radius}


def save(data, path):
    np.savez_compressed(path, **data)
    print("saved", path)


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    import time
    t0 = time.time()
    X, Y, x, u, kp = simulate_prep(targets=[20.0, 40.0], start=[0.0, 40.0],
                                   print_iterations=True)
    st1, st2 = compute_angles_from_cartesian(0.0, 40.0)
    th = x[:, N:N + 2]
    om = x[:, N + 2:N + 4]
    tau = x[:, N + 4:N + 6]
    r = x[:, :N]
    dev_prep = np.degrees(np.hypot(th[:kp, 0] - st1, th[:kp, 1] - st2)).max()
    print(f"solve {time.time()-t0:.1f}s  endpoint err {np.hypot(X[-1]-20, Y[-1]-40):.4f} cm")
    print(f"max joint deviation from start DURING PREP: {dev_prep:.3f} deg")
    print(f"prep network activity ||r|| at end of prep: {np.linalg.norm(r[kp]):.3f}  "
          f"(||r|| peak during move: {np.linalg.norm(r[kp:], axis=1).max():.3f})")
    print(f"||Wout r|| during prep (torque leak): mean {np.linalg.norm(r[:kp] @ Wout.T, axis=1).mean():.4f}  "
          f"peak move {np.linalg.norm(r[kp:] @ Wout.T, axis=1).max():.4f}")
