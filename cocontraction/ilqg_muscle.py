"""Combined random-network + 6-muscle two-link arm, controlled by iLQG.

This replaces the 2-torque actuator of `iLQG_Combined.py` with the 6-muscle model
of `ILQGbasic.py` (3 antagonist pairs: shoulder mono, elbow mono, bi-articular).

Plant, as a cascade:
    network  ->  6 muscle activations  ->  muscle forces  ->  joint torques  ->  arm

  * network:      r_dot = W r + u                       (W: 100x100 from the HDF5)
  * activations:  a_dot = (Wout r - a) / tau_act        (Wout: 6 x N, see below)
  * muscle force: F_m = a * fl(l) * fv(l, v)            (per muscle, 6-vector)
  * joint torque: tau = A_mom @ F_m                     (A_mom: 2x6 moment arms)
  * arm:          M(theta) omega_dot + C + B omega = tau

State  x = [ r (N),  theta1, theta2, omega1, omega2,  a1..a6 ]   (dim N + 10).
Control u acts on the N network nodes (as in the base model).

Wout is generated following Kalidindi & Crevecoeur (2026), STAR Methods Eq. 8:
entries drawn i.i.d. from a normal distribution N(0, 0.5) (variance 0.5), now
shaped 6 x N so that `Wout @ r` is 6-dimensional -- the input to the muscle model.
"""
import os
from math import pi, sqrt

import numpy as np
import h5py

# -----------------------------------------------------------------------------
# network W (from the HDF5, spectral radius 0.8, network #2 -- same as the base
# model) and a freshly generated 6 x N readout Wout.
# -----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_HDF = os.path.join(_HERE, os.pardir, "gaussian_networks.hdf5")
N_MUSCLES = 6
WOUT_VAR = 0.5          # paper: entries ~ N(0, 0.5)
WOUT_SEED = 0


def load_W(spectral_radius=0.8, index=2):
    with h5py.File(_HDF, "r") as f:
        return f[f"{spectral_radius}/networks/network_{index}"][()]


def make_Wout(N, n_muscles=N_MUSCLES, var=WOUT_VAR, seed=WOUT_SEED):
    """6 x N readout, entries ~ N(0, var) (paper Eq. 8, extended to 6 muscles)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sqrt(var), size=(n_muscles, N))


W = load_W()
N = W.shape[0]
Wout = make_Wout(N)

# Muscle activations must be non-negative (a muscle can only pull). ILQGbasic.py
# leaves them signed; with RECTIFY the excitation Wout@r is passed through a
# smooth rectifier so activations stay >= 0 and cocontraction is physiological.
# Set RECTIFY = False to recover the original (signed) ILQGbasic behaviour.
RECTIFY = True
RECT_DELTA = 0.05


def _rect(z):
    return 0.5 * (z + np.sqrt(z ** 2 + RECT_DELTA ** 2)) if RECTIFY else z


def _rect_prime(z):
    return 0.5 * (1 + z / np.sqrt(z ** 2 + RECT_DELTA ** 2)) if RECTIFY else np.ones_like(z)

# -----------------------------------------------------------------------------
# arm + muscle constants (from ILQGbasic.py)
# -----------------------------------------------------------------------------
I1, I2, m1, m2 = 0.025, 0.045, 1.4, 1.0
l1, l2, s1, s2 = 0.30, 0.33, 0.11, 0.16
tau_act = 0.06                         # muscle activation time constant [s]

a1 = I1 + I2 + m2 * l1 * l1
a2 = m2 * l1 * s2
a3 = I2
Viscous = np.array([[0.05, 0.025], [0.025, 0.05]])

# moment-arm matrix (2 joints x 6 muscles): columns are
#   0 shoulder flexor, 1 shoulder extensor, 2 elbow flexor, 3 elbow extensor,
#   4 bi-articular flexor, 5 bi-articular extensor
A_mom = np.array([[2.0, -2.0, 0.0, 0.0, 1.5, -2.0],
                  [0.0, 0.0, 2.0, -2.0, 2.0, -1.5]])
l0 = np.array([7.32, 3.26, 6.4, 4.26, 5.95, 4.04])
theta0 = np.array([[15.0, 4.88, 0.0, 0.0, 4.5, 2.12],
                   [0.0, 0.0, 80.86, 109.32, 92.96, 91.52]]) * pi / 360.0
A_pinv = np.linalg.pinv(A_mom)                 # 6x2, for the cocontraction metric
P_null = np.eye(N_MUSCLES) - A_pinv @ A_mom    # projector onto null(A_mom) (4-D)


# -----------------------------------------------------------------------------
# muscle force-length / force-velocity  (ILQGbasic.py, ported verbatim)
# -----------------------------------------------------------------------------
def muscle_flfv(th1, th2, om1, om2):
    L = 1 + A_mom[0] * (theta0[0] - th1) / l0 + A_mom[1] * (theta0[1] - th2) / l0
    V = A_mom[0] * (-om1) / l0 + A_mom[1] * (-om2) / l0
    fl = np.exp(np.abs((L ** 1.55 - 1) / 0.81))
    fv = np.where(V <= 0,
                  (-7.39 - V) / (-7.39 + 0.96 * V),
                  (0.62 - (-3.12 + 4.21 * L - 2.67 * L ** 2) * V) / (0.62 + V))
    return L, V, fl, fv


def _Minv(th2):
    Den = a3 * (a1 - a3) - a2 ** 2 * np.cos(th2) ** 2
    return np.array([[a3 / Den, (-a2 * np.cos(th2) - a3) / Den],
                     [(-a2 * np.cos(th2) - a3) / Den, (2 * a2 * np.cos(th2) + a1) / Den]])


def torque_from_activation(th1, th2, om1, om2, a):
    """Net joint torque produced by muscle activations a (6,)."""
    _, _, fl, fv = muscle_flfv(th1, th2, om1, om2)
    return A_mom @ (a * fl * fv)


# -----------------------------------------------------------------------------
# dynamics  x_dot = f(x, u)
# -----------------------------------------------------------------------------
def f(x, u, F=0):
    r, q = x[:N], x[N:]
    th1, th2, om1, om2 = q[0], q[1], q[2], q[3]
    a = q[4:10]
    _, _, fl, fv = muscle_flfv(th1, th2, om1, om2)
    tau = A_mom @ (a * fl * fv)
    C = np.array([-om2 * (2 * om1 + om2) * a2 * np.sin(th2), om1 ** 2 * a2 * np.sin(th2)])
    omdot = _Minv(th2) @ (tau - Viscous @ np.array([om1, om2]) - C + F)
    adot = (_rect(Wout @ r) - a) / tau_act
    return np.concatenate([W @ r + u, [om1, om2, omdot[0], omdot[1]], adot])


def fx(x, u):
    """Analytical Jacobian d f / d x, shape (N+10, N+10)."""
    q = x[N:]
    th1, th2, om1, om2 = q[0], q[1], q[2], q[3]
    a = q[4:10]
    om = np.array([om1, om2])

    L, V, fl, fv = muscle_flfv(th1, th2, om1, om2)
    Minv = _Minv(th2)
    C = np.array([-om2 * (2 * om1 + om2) * a2 * np.sin(th2), om1 ** 2 * a2 * np.sin(th2)])
    dCdte = np.array([-om2 * (2 * om1 + om2) * a2 * np.cos(th2), om1 ** 2 * a2 * np.cos(th2)])
    dCdos = np.array([-om2 * 2 * a2 * np.sin(th2), 2 * om1 * a2 * np.sin(th2)])
    dCdoe = np.array([(-2 * om1 - 2 * om2) * a2 * np.sin(th2), 0.0])
    dM = np.array([[-2 * a2 * np.sin(th2), -a2 * np.sin(th2)],
                   [-a2 * np.sin(th2), 0.0]])

    dldts, dldte = -A_mom[0] / l0, -A_mom[1] / l0
    dvdos, dvdoe = -A_mom[0] / l0, -A_mom[1] / l0
    dfldl = fl * np.sign((L ** 1.55 - 1) / 0.81) * (1.55 * L ** 0.55 / 0.81)
    dfvdl = np.where(V <= 0, 0.0, V * (-4.21 + 5.34 * L) / (0.62 + V))
    dfvdv = np.where(V <= 0,
                     7.39 * (1 + 0.96) / (-7.39 + 0.96 * V) ** 2,
                     -0.62 * (-3.12 + 4.21 * L - 2.67 * L ** 2 + 1) / (0.62 + V) ** 2)
    dfldts, dfldte = dfldl * dldts, dfldl * dldte
    dfvdts, dfvdte = dfvdl * dldts, dfvdl * dldte
    dfvdos, dfvdoe = dfvdv * dvdos, dfvdv * dvdoe

    d_acc_th1 = Minv @ (A_mom @ (a * (dfldts * fv + fl * dfvdts)))
    d_acc_os = Minv @ (A_mom @ (a * dfvdos * fl) - dCdos - Viscous @ np.array([1.0, 0.0]))
    d_acc_th2 = (-Minv @ (dM @ Minv @ (A_mom @ (a * fl * fv) - C - Viscous @ om))
                 + Minv @ (A_mom @ (a * (dfldte * fv + fl * dfvdte)) - dCdte))
    d_acc_oe = Minv @ (A_mom @ (a * dfvdoe * fl) - dCdoe - Viscous @ np.array([0.0, 1.0]))
    d_acc_da = Minv @ A_mom @ np.diag(fl * fv)          # (2, 6)

    A = np.zeros((N + 10, N + 10))
    A[:N, :N] = W
    b = N                                               # arm block offset
    A[b, b + 2] = 1.0                                   # th1_dot / d om1
    A[b + 1, b + 3] = 1.0                               # th2_dot / d om2
    for j in range(2):                                  # omega_dot rows (b+2, b+3)
        A[b + 2 + j, b] = d_acc_th1[j]
        A[b + 2 + j, b + 1] = d_acc_th2[j]
        A[b + 2 + j, b + 2] = d_acc_os[j]
        A[b + 2 + j, b + 3] = d_acc_oe[j]
        A[b + 2 + j, b + 4:b + 10] = d_acc_da[j]
    A[b + 4:b + 10, :N] = (_rect_prime(Wout @ x[:N])[:, None] * Wout) / tau_act
    A[b + 4:b + 10, b + 4:b + 10] = -np.eye(6) / tau_act
    return A


def fu(x, u=None):
    B = np.zeros((N + 10, N))
    B[:N, :N] = np.identity(N)
    return B


# -----------------------------------------------------------------------------
# costs  (control effort on the network input; terminal target on joint angles)
# -----------------------------------------------------------------------------
def l_cost(x, u, r1):
    return r1 * np.sum(u ** 2) / 2


def lu(x, u, r1):
    return r1 * u


def luu(x, u, r1):
    return np.diag(np.ones(len(u))) * r1


def h(x, w1, w2, xtarg):
    q = x[-10:]
    return w1 / 2 * ((q[0] - xtarg[0]) ** 2 + (q[1] - xtarg[1]) ** 2) \
        + w2 / 2 * (q[2] ** 2 + q[3] ** 2)


def hx(x, w1, w2, xtarg):
    q = x[-10:]
    g = np.zeros(len(x))
    g[-10] = w1 * (q[0] - xtarg[0])
    g[-9] = w1 * (q[1] - xtarg[1])
    g[-8] = w2 * q[2]
    g[-7] = w2 * q[3]
    return g


def hxx(x, w1, w2):
    Q = np.zeros((len(x), len(x)))
    Q[-10, -10] = w1
    Q[-9, -9] = w1
    Q[-8, -8] = w2
    Q[-7, -7] = w2
    return Q


def compute_angles_from_cartesian(x, y, ll1=30, ll2=33):
    r2 = x ** 2 + y ** 2
    sh = np.arctan2(y, x) - np.arccos((r2 + ll1 ** 2 - ll2 ** 2) / (2 * ll1 * np.sqrt(r2)))
    el = np.pi - np.arccos((ll1 ** 2 + ll2 ** 2 - r2) / (2 * ll1 * ll2))
    return sh, el


def hand_xy(x, ll1=30, ll2=33):
    q = np.asarray(x)[..., -10:]
    th1, th2 = q[..., 0], q[..., 1]
    return np.cos(th1 + th2) * ll2 + np.cos(th1) * ll1, np.sin(th1 + th2) * ll2 + np.sin(th1) * ll1


# -----------------------------------------------------------------------------
# iLQG solver (deterministic; mirrors iLQG_Combined)
# -----------------------------------------------------------------------------
def step1(x0, u, Duration):
    K = u.shape[0]
    dt = Duration / K
    xs = np.zeros((K + 1, len(x0)))
    xs[0] = x0
    for i in range(K):
        xs[i + 1] = xs[i] + dt * f(xs[i], u[i])
    return xs


def step2(x, u, Duration, w1, w2, r1, xtarg):
    K = u.shape[0]
    dt = Duration / K
    n, m = len(x[0]), len(u[0])
    A = np.zeros((K, n, n)); B = np.zeros((K, n, m))
    q = np.zeros(K + 1); qbold = np.zeros((K + 1, n))
    r = np.zeros((K, m)); Q = np.zeros((K + 1, n, n)); R = np.zeros((K, m, m))
    for i in range(K):
        A[i] = np.identity(n) + dt * fx(x[i], u[i])
        B[i] = dt * fu(x[i], u[i])
        q[i] = dt * l_cost(x[i], u[i], r1)
        r[i] = dt * lu(x[i], u[i], r1)
        R[i] = dt * luu(x[i], u[i], r1)
    q[-1] = h(x[-1], w1, w2, xtarg)
    qbold[-1] = hx(x[-1], w1, w2, xtarg)
    Q[-1] = hxx(x[-1], w1, w2)
    return A, B, q, qbold, r, Q, R


def step3(A, B, q, qbold, r, Q, R, eps, m):
    K = A.shape[0]
    n = A.shape[1]
    S = np.zeros((K + 1, n, n)); sbold = np.zeros((K + 1, n))
    lg = np.zeros((K, m)); Lg = np.zeros((K, m, n))
    S[-1] = Q[-1]; sbold[-1] = qbold[-1]
    for k in range(K - 1, -1, -1):
        Snext = S[k + 1]
        Bblk = B[k][:m, :]
        SA = Snext @ A[k]
        G = Bblk.T @ SA[:m, :]
        gbold = r[k] + Bblk.T @ sbold[k + 1][:m]
        H = R[k] + Bblk.T @ Snext[:m, :m] @ Bblk
        ASA = A[k].T @ SA
        w_eig, V_eig = np.linalg.eigh(0.5 * (H + H.T))
        w_eig = np.where(w_eig < eps, eps, w_eig)
        Hinv = (V_eig * (1.0 / w_eig)) @ V_eig.T
        S[k] = Q[k] + ASA - G.T @ (Hinv @ G)
        sbold[k] = qbold[k] + A[k].T @ sbold[k + 1] - G.T @ (Hinv @ gbold)
        lg[k] = -Hinv @ gbold
        Lg[k] = -Hinv @ G
    return lg, Lg


def total_cost(x, u, Duration, w1, w2, r1, xtarg):
    K = u.shape[0]
    dt = Duration / K
    J = h(x[-1], w1, w2, xtarg)
    for k in range(K):
        J += dt * l_cost(x[k], u[k], r1)
    return J


def forward_pass(x0, x_nom, u_nom, lg, Lg, Duration, alpha):
    K = u_nom.shape[0]
    dt = Duration / K
    xnew = np.zeros_like(x_nom); unew = np.zeros_like(u_nom)
    xnew[0] = x0
    for k in range(K):
        dx = xnew[k] - x_nom[k]
        unew[k] = u_nom[k] + alpha * lg[k] + Lg[k] @ dx
        xnew[k + 1] = xnew[k] + dt * f(xnew[k], unew[k])
    return xnew, unew


def simulate_ILQG(targets, start, Duration=0.5, w1=1e4, w2=10.0, r1=1e-5, K=100,
                  eps=1e-6, max_iter=300, tol=1e-5, print_iterations=False):
    obj1, obj2 = compute_angles_from_cartesian(targets[0], targets[1])
    st1, st2 = compute_angles_from_cartesian(start[0], start[1])
    xtarg = np.array([obj1, obj2])
    x0 = np.concatenate([np.zeros(N), [st1, st2, 0, 0], np.zeros(N_MUSCLES)])
    u = np.zeros((K, N))
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
                improved = True
                break
        if not improved:
            if print_iterations:
                print(f"converged (no improvement) at iter {it}, cost {J:.6g}")
            break
        rel = (J - Jnew) / max(abs(J), 1e-12)
        du = np.max(np.abs(unew - u))
        x, u, J = xnew, unew, Jnew
        if du < 1e-10 or rel < tol:
            if print_iterations:
                print(f"converged at iter {it}, cost {J:.6g}")
            break
    X, Y = hand_xy(x)
    return X, Y, x, u


# -----------------------------------------------------------------------------
# cocontraction: muscle drive that produces no net joint torque
# -----------------------------------------------------------------------------
def activations(states):
    """Muscle activations a (…, 6) from a state trajectory."""
    return states[..., N + 4:N + 10]


def cocontraction_index(states):
    """Per-sample cocontraction index: fraction of the activation vector that
    lies in the null space of the moment-arm matrix (i.e. co-activation of
    antagonists producing zero net torque). Shape = states[...,0] shape."""
    a = activations(states)
    null = a @ P_null.T
    return np.linalg.norm(null, axis=-1) / (np.linalg.norm(a, axis=-1) + 1e-12)


PAIR_IDX = ((0, 1), (2, 3), (4, 5))          # shoulder, elbow, bi-articular


def cocontraction_pairs(states):
    """Classic antagonist cocontraction index per pair, 2*min(ai,aj)/(ai+aj) in
    [0,1] (0 = pure agonist, 1 = equal co-activation / no net torque). Assumes
    non-negative activations (RECTIFY=True). Returns shape (..., 3)."""
    a = np.clip(activations(states), 0, None)
    out = [2 * np.minimum(a[..., i], a[..., j]) / (a[..., i] + a[..., j] + 1e-9)
           for i, j in PAIR_IDX]
    return np.stack(out, axis=-1)


if __name__ == "__main__":
    import time
    # 1) analytical Jacobian vs finite differences
    rng = np.random.default_rng(1)
    xt = np.concatenate([rng.normal(0, 0.3, N), [0.7, 1.6, 0.5, -0.4],
                         rng.normal(0, 0.2, N_MUSCLES)])
    ut = rng.normal(0, 0.1, N)
    Ja = fx(xt, ut)
    Jn = np.zeros_like(Ja)
    e = 1e-6
    f0 = f(xt, ut)
    for i in range(len(xt)):
        xp = xt.copy(); xp[i] += e
        Jn[:, i] = (f(xp, ut) - f0) / e
    print("fx vs finite-diff: max abs err = %.2e" % np.abs(Ja - Jn).max())

    # 2) single reach
    t0 = time.time()
    X, Y, x, u = simulate_ILQG(targets=[12, 40], start=[0, 40], print_iterations=True)
    err = np.hypot(X[-1] - 12, Y[-1] - 40)
    ci = cocontraction_index(x)
    print("reach: %.1fs  endpoint err %.4f cm" % (time.time() - t0, err))
    print("peak |activation| = %.3f   mean cocontraction index = %.3f"
          % (np.abs(activations(x)).max(), ci.mean()))
