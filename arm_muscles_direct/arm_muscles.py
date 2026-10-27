"""Two-link arm driven DIRECTLY by six muscles (no network), solved with iLQG.

State x = [theta_s, theta_e, omega_s, omega_e, a_1..a_6], control u in R^6 (one
command per muscle). Arm, moment arms, F-L/V and activation time constant are
those of ../iLQG_Muscle.py (Lillicrap & Scott 2013 muscles, corrected
equations). Two activation laws:

    pull-only : a_dot = (sigma_u(u) - a) / tau_act      (muscles can only pull)
    push-pull : a_dot = (g u - a) / tau_act,  g = 2 e^-4  (the slope of sigma_u's
                linear branch, which passes through 0, extended to u < 0)

sigma_u(x) = exp(2x - 5) for x < 1/2,  2x exp(2 * 1/2 - 5) for x >= 1/2
(Lillicrap & Scott Supplemental "Muscle activation"; breakpoint 1/2, see
../analyses_claude/muscle_plant.py). It is strictly positive, so the tension
F_max a f_l f_fv >= 0 and a muscle can never push.

Both models start at a static equilibrium: zero activation for push-pull; for
pull-only the smallest command u* whose resting activations sigma_u(u*) give
zero net torque at the start posture (sigma_u(0) alone leaves a net torque).
The effort cost is r1/2 |u - u*|^2, so holding still costs nothing in either.

The iLQG (deterministic, terminal cost on hand position and velocity, as in
iLQG_Muscle.solve) accepts a control that enters nonlinearly: B_k = dt fu(x, u).
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "analyses_claude"))
sys.path.insert(0, PROJ)

import muscle_plant as MP  # noqa: E402  (defines sigma_u inside iLQG_Muscle)

M = MP.M
GAIN = 2.0 * np.exp(2.0 * 0.5 - 5.0)


class ArmMuscles:
    def __init__(self, pull_only=True, flv=True, start_xy=(0.0, 40.0)):
        self.pull_only = pull_only
        self.p = M.Plant(np.zeros((0, 0)), np.zeros((6, 0)), muscles="bi", flv=flv)
        self.n, self.m = 10, 6
        self.i_s, self.i_a = slice(0, 4), slice(4, 10)
        self.start_xy = tuple(start_xy)
        self.u_rest = self._rest_command() if pull_only else np.zeros(6)

    # -- activation law --------------------------------------------------------
    def act(self, u):
        return M.sigma_u(u) if self.pull_only else GAIN * u

    def dact(self, u):
        return M.dsigma_u(u) if self.pull_only else np.full(6, GAIN)

    def _rest_command(self):
        """Smallest |u| with zero net torque at the start posture (pull-only)."""
        s0 = self.initial_state(rest=False)[self.i_s]
        tq = lambda u: self.p.torque(s0, M.sigma_u(u))[0]
        res = minimize(lambda u: u @ u, np.zeros(6), method="SLSQP",
                       constraints={"type": "eq", "fun": tq}, options={"ftol": 1e-14})
        assert res.success and np.abs(tq(res.x)).max() < 1e-9, res
        return res.x

    # -- dynamics --------------------------------------------------------------
    def f(self, x, u):
        s, a = x[self.i_s], x[self.i_a]
        th, om = s[:2], s[2:]
        tau, _ = self.p.torque(s, a)
        Mi, _ = self.p.inertia(th[1])
        C, _, _ = MP.coriolis(th, om)
        acc = np.linalg.solve(Mi, tau - C - self.p.viscous() @ om)
        return np.concatenate([om, acc, (self.act(u) - a) / M.TAU_ACT])

    def fx(self, x, u):
        return self.p.fx(x, None)                  # N = 0: arm + activation block

    def fu(self, x, u):
        B = np.zeros((self.n, self.m))
        B[self.i_a, :] = np.diag(self.dact(u)) / M.TAU_ACT
        return B

    def torque(self, x):
        return self.p.torque(x[self.i_s], x[self.i_a])[0]

    # -- states ----------------------------------------------------------------
    def initial_state(self, rest=True):
        th = np.array(M.inverse_kinematics(*self.start_xy))
        a0 = self.act(self.u_rest) if rest else np.zeros(6)
        return np.concatenate([th, [0.0, 0.0], a0])

    def hand(self, x):
        return M.hand_position(np.asarray(x)[..., :2])

    def hand_velocity(self, x):
        return self.p.hand_velocity(x)


# -----------------------------------------------------------------------------
# iLQG
# -----------------------------------------------------------------------------
def _cost(arm, x, u, dt, ptarg, w1, w2, r1):
    R, _ = arm.p.terminal_residual(x[-1], ptarg, w1, w2)
    du = u - arm.u_rest
    return 0.5 * R @ R + 0.5 * dt * r1 * np.sum(du * du)


def _rollout(arm, x0, u, dt):
    x = np.zeros((len(u) + 1, arm.n))
    x[0] = x0
    for k in range(len(u)):
        x[k + 1] = x[k] + dt * arm.f(x[k], u[k])
    return x


def solve(arm, target_xy, duration=0.4, K=80, w1=1e4, w2=10.0, r1=1e-3,
          max_iter=300, tol=1e-8):
    """Deterministic iLQG. Returns x (K+1, 10), u (K, 6), info."""
    dt = duration / K
    ptarg = np.asarray(target_xy, float)
    x0 = arm.initial_state()
    u = np.tile(arm.u_rest, (K, 1))
    x = _rollout(arm, x0, u, dt)
    J = _cost(arm, x, u, dt, ptarg, w1, w2, r1)
    for it in range(max_iter):
        R, Jr = arm.p.terminal_residual(x[-1], ptarg, w1, w2)
        S, sv = Jr.T @ Jr, Jr.T @ R
        lg = np.zeros((K, arm.m))
        Lg = np.zeros((K, arm.m, arm.n))
        for k in range(K - 1, -1, -1):
            A = np.eye(arm.n) + dt * arm.fx(x[k], u[k])
            B = dt * arm.fu(x[k], u[k])
            Qu = dt * r1 * (u[k] - arm.u_rest) + B.T @ sv
            Quu = dt * r1 * np.eye(arm.m) + B.T @ S @ B
            Qux = B.T @ S @ A
            Qx = A.T @ sv
            Quu = 0.5 * (Quu + Quu.T)
            lg[k] = -np.linalg.solve(Quu, Qu)
            Lg[k] = -np.linalg.solve(Quu, Qux)
            sv = Qx + Lg[k].T @ Quu @ lg[k] + Lg[k].T @ Qu + Qux.T @ lg[k]
            S = A.T @ S @ A + Lg[k].T @ Quu @ Lg[k] + Lg[k].T @ Qux + Qux.T @ Lg[k]
            S = 0.5 * (S + S.T)
        improved = False
        for alpha in 0.5 ** np.arange(14):
            xn = np.zeros_like(x)
            un = np.zeros_like(u)
            xn[0] = x0
            for k in range(K):
                un[k] = u[k] + alpha * lg[k] + Lg[k] @ (xn[k] - x[k])
                xn[k + 1] = xn[k] + dt * arm.f(xn[k], un[k])
            Jn = _cost(arm, xn, un, dt, ptarg, w1, w2, r1)
            if np.isfinite(Jn) and Jn < J:
                improved = True
                break
        if not improved:
            break
        rel = (J - Jn) / max(abs(J), 1e-12)
        x, u, J = xn, un, Jn
        if rel < tol:
            break
    err = float(np.hypot(*(arm.hand(x[-1]) - ptarg)))
    return x, u, dict(cost=J, iters=it + 1, endpoint_error=err)


def check_jacobians(seed=0):
    """Finite-difference check of fx and fu for both activation laws."""
    rng = np.random.default_rng(seed)
    for pull in (True, False):
        arm = ArmMuscles(pull_only=pull)
        x = arm.initial_state() + np.concatenate([rng.normal(0, 0.05, 4), rng.uniform(0, 0.05, 6)])
        u = rng.normal(0.3, 0.5, 6)
        h = 1e-6
        Ax = np.array([(arm.f(x + h * e, u) - arm.f(x - h * e, u)) / (2 * h) for e in np.eye(10)]).T
        Bu = np.array([(arm.f(x, u + h * e) - arm.f(x, u - h * e)) / (2 * h) for e in np.eye(6)]).T
        ex = np.abs(arm.fx(x, u) - Ax).max() / np.abs(Ax).max()
        eu = np.abs(arm.fu(x, u) - Bu).max() / np.abs(Bu).max()
        print(f"{'pull-only' if pull else 'push-pull':9s}: fx rel err {ex:.1e}, fu rel err {eu:.1e}, "
              f"rest command {np.round(arm.u_rest, 3)}, rest torque "
              f"{np.round(arm.torque(arm.initial_state()), 9)}")


if __name__ == "__main__":
    check_jacobians()
