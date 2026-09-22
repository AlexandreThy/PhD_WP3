"""Cocontraction as an optimal response to UNCERTAIN environment dynamics.

Reuses the network + 6-muscle plant from ../cocontraction/ilqg_muscle.py and adds
a lateral divergent force field (Burdet et al. 2001) whose gain `beta` is
uncertain. Uncertainty is modelled by the "parallel worlds" / scenario approach:

  M copies of the arm are driven by the SAME network and the SAME muscle
  activations (a single feedforward policy that cannot sense which world it is
  in), but each copy m feels a different field gain beta_m. A single iLQG solve
  minimises the AVERAGE endpoint cost across the M worlds. Because one activation
  trajectory must land every copy on target despite different (unknown) fields,
  the optimiser stiffens the arm -- it co-contracts -- and stiffens more as the
  spread of beta (the uncertainty) grows.

Field: reaching along x = 0 (from (0,40) to (0,52) cm), the field pushes the hand
laterally,  F_hand = [beta * x_hand, 0]  (beta>0 divergent / destabilising).

State  z = [ r(N), a(6), (th1,th2,om1,om2)_1, ..., (...)_M ]   (dim N + 6 + 4M).
Muscle stiffness (imported force-length) is restoring, so cocontraction reduces
the endpoint spread across worlds -- verified before building this.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "cocontraction"))
import ilqg_muscle as MU

N = MU.N
W, Wout = MU.W, MU.Wout
L1M, L2M = 0.30, 0.33            # link lengths in metres (for the field)

# --- environment: the M field gains for the current solve (set via set_env) ---
M_WORLDS = 5
BETAS = np.zeros(M_WORLDS)


def set_env(mu=0.0, sigma=0.0, m=M_WORLDS):
    """Set the M field gains to mu + sigma * linspace(-1, 1, m). sigma = the
    uncertainty level (sigma = 0 -> all worlds identical = certain field)."""
    global BETAS, M_WORLDS
    M_WORLDS = m
    BETAS = mu + sigma * np.linspace(-1.0, 1.0, m)
    return BETAS


# -----------------------------------------------------------------------------
# hand / field
# -----------------------------------------------------------------------------
def _hand_m(th1, th2):
    return (np.cos(th1 + th2) * L2M + np.cos(th1) * L1M,
            np.sin(th1 + th2) * L2M + np.sin(th1) * L1M)


def _hand_jac_m(th1, th2):
    return np.array([[-np.sin(th1 + th2) * L2M - np.sin(th1) * L1M, -np.sin(th1 + th2) * L2M],
                     [np.cos(th1 + th2) * L2M + np.cos(th1) * L1M, np.cos(th1 + th2) * L2M]])


def field_torque(th1, th2, om1, om2, beta):
    """Velocity-dependent CURL force field (perpendicular to hand velocity):
    F_hand = beta * [v_y, -v_x]. Acts throughout the movement (wherever v != 0),
    so each world deviates differently -- unlike a divergent field, which is null
    on the straight path and would leave a deterministic feedforward untouched."""
    J = _hand_jac_m(th1, th2)
    v = J @ np.array([om1, om2])
    return J.T @ (beta * np.array([v[1], -v[0]]))


def _arm_dot(th1, th2, om1, om2, a, beta):
    _, _, fl, fv = MU.muscle_flfv(th1, th2, om1, om2)
    tau_m = MU.A_mom @ (a * fl * fv)
    C = np.array([-om2 * (2 * om1 + om2) * MU.a2 * np.sin(th2),
                  om1 ** 2 * MU.a2 * np.sin(th2)])
    tau_e = field_torque(th1, th2, om1, om2, beta)
    omdot = MU._Minv(th2) @ (tau_m - MU.Viscous @ np.array([om1, om2]) - C + tau_e)
    return np.array([om1, om2, omdot[0], omdot[1]])


# -----------------------------------------------------------------------------
# dynamics and (hybrid) Jacobian
# -----------------------------------------------------------------------------
def _dim():
    return N + 6 + 4 * M_WORLDS


def f(z, u):
    r, a = z[:N], z[N:N + 6]
    out = np.empty_like(z)
    out[:N] = W @ r + u
    out[N:N + 6] = (MU._rect(Wout @ r) - a) / MU.tau_act
    for m in range(M_WORLDS):
        o = N + 6 + 4 * m
        out[o:o + 4] = _arm_dot(z[o], z[o + 1], z[o + 2], z[o + 3], a, BETAS[m])
    return out


def fx(z, u):
    r, a = z[:N], z[N:N + 6]
    n = _dim()
    A = np.zeros((n, n))
    A[:N, :N] = W
    A[N:N + 6, :N] = (MU._rect_prime(Wout @ r)[:, None] * Wout) / MU.tau_act
    A[N:N + 6, N:N + 6] = -np.eye(6) / MU.tau_act
    e = 1e-6
    for m in range(M_WORLDS):
        o = N + 6 + 4 * m
        inp0 = np.concatenate([z[o:o + 4], a])
        base = _arm_dot(inp0[0], inp0[1], inp0[2], inp0[3], inp0[4:10], BETAS[m])
        for k in range(10):                       # 4 arm states + 6 activations
            inp = inp0.copy(); inp[k] += e
            col = (_arm_dot(inp[0], inp[1], inp[2], inp[3], inp[4:10], BETAS[m]) - base) / e
            A[o:o + 4, (o + k) if k < 4 else (N + k - 4)] = col
    return A


def fu(z, u=None):
    B = np.zeros((_dim(), N))
    B[:N, :N] = np.identity(N)
    return B


# -----------------------------------------------------------------------------
# costs
#   * effort on the network input,
#   * RUNNING penalty on each world's lateral (x) hand deviation from the
#     straight reach line (Burdet-style: hold the hand on the path against the
#     field). This is what turns the uncertain field into a cocontraction demand:
#     the worlds are pushed off the line differently, and stiffness (cocontraction)
#     keeps them on it -- more so as the uncertainty grows.
#   * average terminal target over worlds.
# -----------------------------------------------------------------------------
W_LAT = 300.0            # weight on lateral deviation (hand x, in cm)


def _hand_x_cm(th1, th2):
    return np.cos(th1 + th2) * 33 + np.cos(th1) * 30


def _hand_x_grad(th1, th2):
    return np.array([-np.sin(th1 + th2) * 33 - np.sin(th1) * 30,
                     -np.sin(th1 + th2) * 33])


def l_cost(x, u, r1):
    c = r1 * np.sum(u ** 2) / 2
    for m in range(M_WORLDS):
        o = N + 6 + 4 * m
        c += W_LAT / M_WORLDS * _hand_x_cm(x[o], x[o + 1]) ** 2 / 2
    return c


def l_lx(x):
    g = np.zeros(len(x))
    for m in range(M_WORLDS):
        o = N + 6 + 4 * m
        X = _hand_x_cm(x[o], x[o + 1])
        g[o:o + 2] = W_LAT / M_WORLDS * X * _hand_x_grad(x[o], x[o + 1])
    return g


def l_lxx(x):
    Q = np.zeros((len(x), len(x)))
    for m in range(M_WORLDS):
        o = N + 6 + 4 * m
        gx = _hand_x_grad(x[o], x[o + 1])                # Gauss-Newton (PSD)
        Q[o:o + 2, o:o + 2] = W_LAT / M_WORLDS * np.outer(gx, gx)
    return Q


def lu(x, u, r1):
    return r1 * u


def luu(x, u, r1):
    return np.diag(np.ones(len(u))) * r1


def h(z, w1, w2, xtarg):
    tot = 0.0
    for m in range(M_WORLDS):
        o = N + 6 + 4 * m
        tot += w1 / 2 * ((z[o] - xtarg[0]) ** 2 + (z[o + 1] - xtarg[1]) ** 2) \
            + w2 / 2 * (z[o + 2] ** 2 + z[o + 3] ** 2)
    return tot / M_WORLDS


def hx(z, w1, w2, xtarg):
    g = np.zeros(len(z))
    for m in range(M_WORLDS):
        o = N + 6 + 4 * m
        g[o] = w1 * (z[o] - xtarg[0]) / M_WORLDS
        g[o + 1] = w1 * (z[o + 1] - xtarg[1]) / M_WORLDS
        g[o + 2] = w2 * z[o + 2] / M_WORLDS
        g[o + 3] = w2 * z[o + 3] / M_WORLDS
    return g


def hxx(z, w1, w2):
    Q = np.zeros((len(z), len(z)))
    for m in range(M_WORLDS):
        o = N + 6 + 4 * m
        Q[o, o] = w1 / M_WORLDS; Q[o + 1, o + 1] = w1 / M_WORLDS
        Q[o + 2, o + 2] = w2 / M_WORLDS; Q[o + 3, o + 3] = w2 / M_WORLDS
    return Q


# -----------------------------------------------------------------------------
# iLQG solver (deterministic; mirrors ilqg_muscle)
# -----------------------------------------------------------------------------
def step1(x0, u, Duration):
    K = u.shape[0]; dt = Duration / K
    xs = np.zeros((K + 1, len(x0))); xs[0] = x0
    for i in range(K):
        xs[i + 1] = xs[i] + dt * f(xs[i], u[i])
    return xs


def step2(x, u, Duration, w1, w2, r1, xtarg):
    K = u.shape[0]; dt = Duration / K
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


def step3(A, B, q, qbold, r, Q, R, eps, m):
    K = A.shape[0]; n = A.shape[1]
    S = np.zeros((K + 1, n, n)); sbold = np.zeros((K + 1, n))
    lg = np.zeros((K, m)); Lg = np.zeros((K, m, n))
    S[-1] = Q[-1]; sbold[-1] = qbold[-1]
    for k in range(K - 1, -1, -1):
        Snext = S[k + 1]; Bblk = B[k][:m, :]
        SA = Snext @ A[k]
        G = Bblk.T @ SA[:m, :]
        gbold = r[k] + Bblk.T @ sbold[k + 1][:m]
        H = R[k] + Bblk.T @ Snext[:m, :m] @ Bblk
        w_eig, V_eig = np.linalg.eigh(0.5 * (H + H.T))
        w_eig = np.where(w_eig < eps, eps, w_eig)
        Hinv = (V_eig * (1.0 / w_eig)) @ V_eig.T
        S[k] = Q[k] + A[k].T @ SA - G.T @ (Hinv @ G)
        sbold[k] = qbold[k] + A[k].T @ sbold[k + 1] - G.T @ (Hinv @ gbold)
        lg[k] = -Hinv @ gbold; Lg[k] = -Hinv @ G
    return lg, Lg


def total_cost(x, u, Duration, w1, w2, r1, xtarg):
    K = u.shape[0]; dt = Duration / K
    J = h(x[-1], w1, w2, xtarg)
    for k in range(K):
        J += dt * l_cost(x[k], u[k], r1)
    return J


def forward_pass(x0, x_nom, u_nom, lg, Lg, Duration, alpha):
    K = u_nom.shape[0]; dt = Duration / K
    xnew = np.zeros_like(x_nom); unew = np.zeros_like(u_nom); xnew[0] = x0
    for k in range(K):
        unew[k] = u_nom[k] + alpha * lg[k] + Lg[k] @ (xnew[k] - x_nom[k])
        xnew[k + 1] = xnew[k] + dt * f(xnew[k], unew[k])
    return xnew, unew


def simulate(targets, start, Duration=0.5, w1=1e4, w2=10.0, r1=1e-5, K=100,
             eps=1e-6, max_iter=200, tol=1e-4, print_iterations=False, u0=None):
    """u0: optional warm-start control sequence (K, N). Warm-starting from the
    previous uncertainty level makes the solution continuous in sigma and removes
    convergence noise from the cocontraction-vs-uncertainty curve."""
    obj1, obj2 = MU.compute_angles_from_cartesian(targets[0], targets[1])
    st1, st2 = MU.compute_angles_from_cartesian(start[0], start[1])
    xtarg = np.array([obj1, obj2])
    x0 = np.concatenate([np.zeros(N), np.zeros(6),
                         np.tile([st1, st2, 0, 0], M_WORLDS)])
    u = np.zeros((K, N)) if u0 is None else np.array(u0, dtype=float)
    x = step1(x0, u, Duration)
    J = total_cost(x, u, Duration, w1, w2, r1, xtarg)
    alphas = 0.5 ** np.arange(12)
    for it in range(max_iter):
        A, B, q, qbold, r, Q, R = step2(x, u, Duration, w1, w2, r1, xtarg)
        lg, Lg = step3(A, B, q, qbold, r, Q, R, eps, N)
        improved = False
        for alpha in alphas:
            xnew, unew = forward_pass(x0, x, u, lg, Lg, Duration, alpha)
            Jnew = total_cost(xnew, unew, Duration, w1, w2, r1, xtarg)
            if np.isfinite(Jnew) and Jnew < J:
                improved = True; break
        if not improved:
            if print_iterations:
                print(f"converged (no improvement) at {it}, cost {J:.6g}")
            break
        rel = (J - Jnew) / max(abs(J), 1e-12)
        x, u, J = xnew, unew, Jnew
        if rel < tol:
            if print_iterations:
                print(f"converged at {it}, cost {J:.6g}")
            break
    return x, u, xtarg


# -----------------------------------------------------------------------------
# read-outs
# -----------------------------------------------------------------------------
def activations(states):
    return states[..., N:N + 6]


def cocontraction_pairs(states):
    a = np.clip(activations(states), 0, None)
    return np.stack([2 * np.minimum(a[..., i], a[..., j]) / (a[..., i] + a[..., j] + 1e-9)
                     for i, j in MU.PAIR_IDX], axis=-1)


def hands(states):
    """Per-world hand path (cm), shape (..., M, 2)."""
    xs = []
    for m in range(M_WORLDS):
        o = N + 6 + 4 * m
        th1, th2 = states[..., o], states[..., o + 1]
        X = np.cos(th1 + th2) * 33 + np.cos(th1) * 30
        Y = np.sin(th1 + th2) * 33 + np.sin(th1) * 30
        xs.append(np.stack([X, Y], -1))
    return np.stack(xs, axis=-2)


if __name__ == "__main__":
    import time
    # Jacobian check
    set_env(mu=0.0, sigma=60.0, m=3)
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(0, 0.3, N), rng.normal(0, 0.1, 6),
                        np.tile([0.7, 1.6, 0.3, -0.2], M_WORLDS)])
    ut = rng.normal(0, 0.1, N)
    Ja = fx(z, ut); f0 = f(z, ut); Jn = np.zeros_like(Ja); e = 1e-6
    for i in range(len(z)):
        zp = z.copy(); zp[i] += e; Jn[:, i] = (f(zp, ut) - f0) / e
    print("fx vs finite-diff max abs err = %.2e" % np.abs(Ja - Jn).max())

    # certain (sigma=0) vs uncertain field, one reach
    for sig in (0.0, 120.0):
        set_env(mu=0.0, sigma=sig, m=5)
        t0 = time.time()
        x, u, xt = simulate([0, 52], [0, 40], print_iterations=False)
        cc = cocontraction_pairs(x).mean()
        H = hands(x)
        spread = np.abs(H[-1, :, 0]).max()
        print("sigma=%6.1f  cocontraction=%.3f  endpoint lateral spread=%.3f cm  (%.0fs)"
              % (sig, cc, spread, time.time() - t0))
