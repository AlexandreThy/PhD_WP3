"""Helpers to run iLQG_Muscle for the FLV x (linear / nonlinear arm) study.

1. `sigma_u` / `dsigma_u` are referenced by iLQG_Muscle.py but not defined there
   (nor in its git history). They are restored here from Lillicrap & Scott's
   Supplemental ("Muscle activation"):

        sigma_u(x) = exp(2x - 5)          x <  1/2
                     2x exp(2 * 1/2 - 5)  x >= 1/2

   The supplement prints the breakpoint as 1/10, where the two branches match in
   slope but not in value; 1/2 is the value at which they match in both, as
   README_iLQG.md states it was used. Injected into the iLQG_Muscle namespace;
   iLQG_Muscle.py is not modified.

2. `LinearArmPlant` is the linearized-arm null of iLQG_Combined (LINEARIZE):
   inertia frozen at M(theta2_ref), Coriolis/centripetal dropped. Muscles,
   rectification and the chosen F-L/V variant are unchanged.
"""
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import iLQG_Muscle as M  # noqa: E402


SIGMA_BREAK = 0.5


def sigma_u(x):
    x = np.asarray(x, dtype=float)
    lin = 2.0 * x * np.exp(2.0 * SIGMA_BREAK - 5.0)
    return np.where(x < SIGMA_BREAK, np.exp(2.0 * np.minimum(x, SIGMA_BREAK) - 5.0), lin)


def dsigma_u(x):
    x = np.asarray(x, dtype=float)
    return np.where(x < SIGMA_BREAK, 2.0 * np.exp(2.0 * np.minimum(x, SIGMA_BREAK) - 5.0),
                    2.0 * np.exp(2.0 * SIGMA_BREAK - 5.0))


M.sigma_u = sigma_u
M.dsigma_u = dsigma_u


def coriolis(theta, omega):
    st2, ct2 = np.sin(theta[1]), np.cos(theta[1])
    C = np.array([-omega[1] * (2 * omega[0] + omega[1]) * M.a2 * st2,
                  omega[0] ** 2 * M.a2 * st2])
    dC_dth2 = np.array([-omega[1] * (2 * omega[0] + omega[1]) * M.a2 * ct2,
                        omega[0] ** 2 * M.a2 * ct2])
    dC_domega = np.array([[-2 * omega[1] * M.a2 * st2,
                           -(2 * omega[0] + 2 * omega[1]) * M.a2 * st2],
                          [2 * omega[0] * M.a2 * st2, 0.0]])
    return C, dC_dth2, dC_domega


class LinearArmPlant(M.Plant):
    """Frozen inertia M(theta2_ref), no Coriolis; everything else as M.Plant."""

    def __init__(self, *args, theta2_ref=M.CENTER_JOINTS[1], **kw):
        super().__init__(*args, **kw)
        self.theta2_ref = theta2_ref

    def inertia(self, th2):
        Mref, _ = super().inertia(self.theta2_ref)
        return Mref, np.zeros((2, 2))

    def f(self, x, u):
        out = super().f(x, u)           # uses M(theta2_ref) and subtracts C
        s = x[self.i_s]
        C, _, _ = coriolis(s[:2], s[2:])
        Mref, _ = self.inertia(None)
        out[self.N + 2:self.N + 4] += np.linalg.solve(Mref, C)
        return out

    def fx(self, x, u):
        A = super().fx(x, u)            # dM = 0 already; remove the dC terms
        s = x[self.i_s]
        _, dC_dth2, dC_domega = coriolis(s[:2], s[2:])
        Minv = np.linalg.inv(self.inertia(None)[0])
        i0 = self.N
        A[i0 + 2:i0 + 4, i0 + 1] += Minv @ dC_dth2
        A[i0 + 2:i0 + 4, i0 + 2:i0 + 4] += Minv @ dC_domega
        return A


def solve_warm(plant, start_xy, target_xy, u_init, duration=0.5, K=100,
               w1=1e4, w2=10.0, r1=1e-5, qr=0.0, qa=0.0, qp=0.0,
               eps=1e-14, max_iter=100, tol=1e-6):
    """M.solve, but starting iLQG from the control sequence `u_init` instead of
    zeros. Used for reaches where the zero-control rollout drifts so far that
    iLQG stalls in a poor local minimum (F-L without F-V, slow reaches)."""
    dt = duration / K
    x0 = plant.initial_state(start_xy)
    ptarg = np.asarray(target_xy, dtype=float)
    p0 = np.asarray(start_xy, dtype=float)
    path = None
    if qp > 0:
        v = ptarg - p0
        path = (qp, p0, np.array([-v[1], v[0]]) / np.linalg.norm(v))

    u = np.array(u_init, dtype=float)
    x = M.rollout(plant, x0, u, dt)
    J = M.total_cost(plant, x, u, dt, ptarg, w1, w2, r1, qr, qa, path)
    alphas = 0.5 ** np.arange(12)
    for it in range(max_iter):
        lg, Lg = M.backward_pass(plant, x, u, dt, ptarg, w1, w2, r1, qr, qa, path, eps)
        improved = False
        for alpha in alphas:
            xn, un = M.forward_pass(plant, x0, x, u, lg, Lg, dt, alpha)
            Jn = M.total_cost(plant, xn, un, dt, ptarg, w1, w2, r1, qr, qa, path)
            if np.isfinite(Jn) and Jn < J:
                improved = True
                break
        if not improved:
            break
        rel = (J - Jn) / max(abs(J), 1e-12)
        x, u, J = xn, un, Jn
        if rel < tol:
            break
    p = plant.hand(x)
    return x, u, {"cost": J, "iters": it + 1,
                  "endpoint_error": float(np.hypot(*(p[-1] - ptarg)))}


def make_plant(linear, flv, network=2, seed=0):
    W = M.load_network(network, path=os.path.join(PROJ, "gaussian_networks.hdf5"))
    Wout = M.random_readout(6, W.shape[0], np.random.default_rng(seed))
    cls = LinearArmPlant if linear else M.Plant
    return cls(W, Wout, geometry=True, intersegmental=True, muscles="bi", flv=flv)


def check_jacobian(plant, seed=0):
    rng = np.random.default_rng(seed)
    N = plant.N
    x = plant.initial_state(M.CENTER_XY)
    x[:N] = rng.normal(0, 0.3, N)
    x[plant.i_s][2:] += rng.normal(0, 0.5, 2)
    x[plant.i_a] += np.abs(rng.normal(0, 0.05, plant.n_act))
    u = rng.normal(0, 0.1, N)
    A = plant.fx(x, u)
    Afd = np.zeros_like(A)
    h = 1e-6
    for j in range(plant.n):
        e = np.zeros(plant.n)
        e[j] = h
        Afd[:, j] = (plant.f(x + e, u) - plant.f(x - e, u)) / (2 * h)
    return np.abs(A - Afd).max() / max(1.0, np.abs(Afd).max())
