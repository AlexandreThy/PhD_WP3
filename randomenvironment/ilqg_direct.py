"""Direct-muscle-control version of the uncertain-environment experiment (no
random network).

Same 6-muscle arm, same M-worlds uncertain curl field, same lateral-deviation
cost -- but the control u is now the 6 muscle commands directly (a_dot =
(rect(u) - a)/tau_act), with no network in between. State z = [a(6),
(th,om)_1..M], dim 6 + 4M (vs N + 6 + 4M with the network).

Key consequence: here the effort cost r1*|u|^2 penalises MUSCLE activation
directly, so cocontraction is no longer "free" (in the network model only the
network input was penalised). This is the "penalise muscle effort" regime -- a
stronger test of whether cocontraction is a genuine response to uncertainty.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "cocontraction"))
sys.path.insert(0, _HERE)
import ilqg_muscle as MU
import ilqg_robust as RB          # reuse _arm_dot + curl field (pure functions)

NM = 6
WORLD_OFF = 6
M_WORLDS = 3
BETAS = np.zeros(M_WORLDS)
W_LAT = 300.0


def set_env(mu=0.0, sigma=0.0, m=M_WORLDS):
    global BETAS, M_WORLDS
    M_WORLDS = m
    BETAS = mu + sigma * np.linspace(-1.0, 1.0, m)
    return BETAS


def _dim():
    return WORLD_OFF + 4 * M_WORLDS


# -----------------------------------------------------------------------------
# dynamics
# -----------------------------------------------------------------------------
def f(z, u):
    a = z[:NM]
    out = np.empty_like(z)
    out[:NM] = (MU._rect(u) - a) / MU.tau_act
    for m in range(M_WORLDS):
        o = WORLD_OFF + 4 * m
        out[o:o + 4] = RB._arm_dot(z[o], z[o + 1], z[o + 2], z[o + 3], a, BETAS[m])
    return out


def fx(z, u):
    a = z[:NM]
    n = _dim()
    A = np.zeros((n, n))
    A[:NM, :NM] = -np.eye(NM) / MU.tau_act
    e = 1e-6
    for m in range(M_WORLDS):
        o = WORLD_OFF + 4 * m
        inp0 = np.concatenate([z[o:o + 4], a])
        base = RB._arm_dot(inp0[0], inp0[1], inp0[2], inp0[3], inp0[4:10], BETAS[m])
        for k in range(10):                        # 4 arm states + 6 activations
            inp = inp0.copy(); inp[k] += e
            col = (RB._arm_dot(inp[0], inp[1], inp[2], inp[3], inp[4:10], BETAS[m]) - base) / e
            A[o:o + 4, (o + k) if k < 4 else (k - 4)] = col
    return A


def fu(z, u):
    B = np.zeros((_dim(), NM))
    B[:NM, :NM] = np.diag(MU._rect_prime(u)) / MU.tau_act
    return B


# -----------------------------------------------------------------------------
# costs (effort on the muscle command; lateral-deviation hold; terminal target)
# -----------------------------------------------------------------------------
def l_cost(x, u, r1):
    c = r1 * np.sum(u ** 2) / 2
    for m in range(M_WORLDS):
        o = WORLD_OFF + 4 * m
        c += W_LAT / M_WORLDS * RB._hand_x_cm(x[o], x[o + 1]) ** 2 / 2
    return c


def l_lx(x):
    g = np.zeros(len(x))
    for m in range(M_WORLDS):
        o = WORLD_OFF + 4 * m
        X = RB._hand_x_cm(x[o], x[o + 1])
        g[o:o + 2] = W_LAT / M_WORLDS * X * RB._hand_x_grad(x[o], x[o + 1])
    return g


def l_lxx(x):
    Q = np.zeros((len(x), len(x)))
    for m in range(M_WORLDS):
        o = WORLD_OFF + 4 * m
        gx = RB._hand_x_grad(x[o], x[o + 1])
        Q[o:o + 2, o:o + 2] = W_LAT / M_WORLDS * np.outer(gx, gx)
    return Q


def lu(x, u, r1):
    return r1 * u


def luu(x, u, r1):
    return np.diag(np.ones(len(u))) * r1


def h(z, w1, w2, xtarg):
    tot = 0.0
    for m in range(M_WORLDS):
        o = WORLD_OFF + 4 * m
        tot += w1 / 2 * ((z[o] - xtarg[0]) ** 2 + (z[o + 1] - xtarg[1]) ** 2) \
            + w2 / 2 * (z[o + 2] ** 2 + z[o + 3] ** 2)
    return tot / M_WORLDS


def hx(z, w1, w2, xtarg):
    g = np.zeros(len(z))
    for m in range(M_WORLDS):
        o = WORLD_OFF + 4 * m
        g[o] = w1 * (z[o] - xtarg[0]) / M_WORLDS
        g[o + 1] = w1 * (z[o + 1] - xtarg[1]) / M_WORLDS
        g[o + 2] = w2 * z[o + 2] / M_WORLDS
        g[o + 3] = w2 * z[o + 3] / M_WORLDS
    return g


def hxx(z, w1, w2):
    Q = np.zeros((len(z), len(z)))
    for m in range(M_WORLDS):
        o = WORLD_OFF + 4 * m
        Q[o, o] = w1 / M_WORLDS; Q[o + 1, o + 1] = w1 / M_WORLDS
        Q[o + 2, o + 2] = w2 / M_WORLDS; Q[o + 3, o + 3] = w2 / M_WORLDS
    return Q


# -----------------------------------------------------------------------------
# iLQG solver (control dim = 6; control-affected states are the first 6)
# -----------------------------------------------------------------------------
def _step1(x0, u, Dur):
    K = u.shape[0]; dt = Dur / K
    xs = np.zeros((K + 1, len(x0))); xs[0] = x0
    for i in range(K):
        xs[i + 1] = xs[i] + dt * f(xs[i], u[i])
    return xs


def _step2(x, u, Dur, w1, w2, r1, xtarg):
    K = u.shape[0]; dt = Dur / K
    n, m = len(x[0]), len(u[0])
    A = np.zeros((K, n, n)); B = np.zeros((K, n, m))
    q = np.zeros(K + 1); qbold = np.zeros((K + 1, n))
    r = np.zeros((K, m)); Q = np.zeros((K + 1, n, n)); R = np.zeros((K, m, m))
    for i in range(K):
        A[i] = np.identity(n) + dt * fx(x[i], u[i])
        B[i] = dt * fu(x[i], u[i])
        q[i] = dt * l_cost(x[i], u[i], r1)
        qbold[i] = dt * l_lx(x[i])
        r[i] = dt * lu(x[i], u[i], r1)
        Q[i] = dt * l_lxx(x[i])
        R[i] = dt * luu(x[i], u[i], r1)
    q[-1] = h(x[-1], w1, w2, xtarg)
    qbold[-1] = hx(x[-1], w1, w2, xtarg)
    Q[-1] = hxx(x[-1], w1, w2)
    return A, B, q, qbold, r, Q, R


def _total_cost(x, u, Dur, w1, w2, r1, xtarg):
    K = u.shape[0]; dt = Dur / K
    J = h(x[-1], w1, w2, xtarg)
    for k in range(K):
        J += dt * l_cost(x[k], u[k], r1)
    return J


def _forward(x0, x_nom, u_nom, lg, Lg, Dur, alpha):
    K = u_nom.shape[0]; dt = Dur / K
    xnew = np.zeros_like(x_nom); unew = np.zeros_like(u_nom); xnew[0] = x0
    for k in range(K):
        unew[k] = u_nom[k] + alpha * lg[k] + Lg[k] @ (xnew[k] - x_nom[k])
        xnew[k + 1] = xnew[k] + dt * f(xnew[k], unew[k])
    return xnew, unew


def simulate(targets, start, Duration=0.5, w1=1e4, w2=10.0, r1=1e-5, K=80,
             eps=1e-6, max_iter=140, tol=1e-4, u0=None):
    obj1, obj2 = MU.compute_angles_from_cartesian(targets[0], targets[1])
    st1, st2 = MU.compute_angles_from_cartesian(start[0], start[1])
    xtarg = np.array([obj1, obj2])
    x0 = np.concatenate([np.zeros(NM), np.tile([st1, st2, 0, 0], M_WORLDS)])
    u = np.zeros((K, NM)) if u0 is None else np.array(u0, dtype=float)
    x = _step1(x0, u, Duration)
    J = _total_cost(x, u, Duration, w1, w2, r1, xtarg)
    alphas = 0.5 ** np.arange(12)
    for it in range(max_iter):
        A, B, q, qbold, r, Q, R = _step2(x, u, Duration, w1, w2, r1, xtarg)
        lg, Lg = RB.step3(A, B, q, qbold, r, Q, R, eps, NM)
        improved = False
        for alpha in alphas:
            xnew, unew = _forward(x0, x, u, lg, Lg, Duration, alpha)
            Jnew = _total_cost(xnew, unew, Duration, w1, w2, r1, xtarg)
            if np.isfinite(Jnew) and Jnew < J:
                improved = True; break
        if not improved:
            break
        rel = (J - Jnew) / max(abs(J), 1e-12)
        x, u, J = xnew, unew, Jnew
        if rel < tol:
            break
    return x, u, xtarg


# -----------------------------------------------------------------------------
# read-outs
# -----------------------------------------------------------------------------
def activations(states):
    return states[..., :NM]


def cocontraction_pairs(states):
    a = np.clip(activations(states), 0, None)
    return np.stack([2 * np.minimum(a[..., i], a[..., j]) / (a[..., i] + a[..., j] + 1e-9)
                     for i, j in MU.PAIR_IDX], axis=-1)


def hands(states):
    xs = []
    for m in range(M_WORLDS):
        o = WORLD_OFF + 4 * m
        th1, th2 = states[..., o], states[..., o + 1]
        X = np.cos(th1 + th2) * 33 + np.cos(th1) * 30
        Y = np.sin(th1 + th2) * 33 + np.sin(th1) * 30
        xs.append(np.stack([X, Y], -1))
    return np.stack(xs, axis=-2)


if __name__ == "__main__":
    import time
    # Jacobian check
    set_env(0.0, 3.0, 3)
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(0, 0.1, NM), np.tile([0.7, 1.6, 0.3, -0.2], M_WORLDS)])
    ut = rng.normal(0, 0.1, NM)
    Ja = fx(z, ut); f0 = f(z, ut); Jn = np.zeros_like(Ja); e = 1e-6
    for i in range(len(z)):
        zp = z.copy(); zp[i] += e; Jn[:, i] = (f(zp, ut) - f0) / e
    print("fx vs finite-diff max abs err = %.2e" % np.abs(Ja - Jn).max())
    for sig in (0.0, 4.0):
        set_env(0.0, sig, 3)
        t0 = time.time()
        x, u, xt = simulate([0, 52], [0, 40])
        print("sigma=%.1f  cocontraction=%.3f  (%.1fs)"
              % (sig, cocontraction_pairs(x).mean(), time.time() - t0))
