import numpy as np
from math import *
from hdf import load_networks_by_spectral_radius
W, Wout = load_networks_by_spectral_radius()
I1 = 0.025
I2 = 0.045
m1 = 1.4
m2 = 1
l1 = 0.3
l2 = 0.33
s1 = 0.11
s2 = 0.16
K = 1 / 0.06
tau = 0.06


a1 = I1 + I2 + m2 * l1 * l1
a2 = m2 * l1 * s2
a3 = I2

Viscous = np.array([[0.05, 0.025], [0.025, 0.05]])

# Optional plant linearization used as the "no-nonlinearity" baseline. When
# LINEARIZE is True the arm inertia matrix is frozen at elbow angle THETA2_REF
# and the Coriolis/centripetal terms are dropped, so the limb dynamics -- and
# hence the whole plant -- become linear. Kept False for the main model.
LINEARIZE = False
THETA2_REF = np.pi/2


def compute_angles_from_cartesian(x, y, l1=30, l2=33):
    """
    Computes joint angles in radians based on cartesian coordinates.

    Parameters:
        x (float): x-coordinate of the end effector.
        y (float): y-coordinate of the end effector.
        l1 (float): Length of the first link.
        l2 (float): Length of the second link.

    Returns:
        angles (float): Computed angle in radians.
    """
    r_squared = x**2 + y**2

    shoulder_angle = np.arctan2(y, x) - np.arccos(
        (r_squared + l1**2 - l2**2) / (2 * l1 * np.sqrt(r_squared))
    )

    elbow_angle = np.pi - np.arccos((l1**2 + l2**2 - r_squared) / (2 * l1 * l2))
    return shoulder_angle, elbow_angle


def compute_forcefield(theta, omega, coefficient):
    """
    Compute the joint angles acceleration resulting from a lateral
    velocity-dependent forcefield.

    Args:
        theta : current joint angles
        omega : current joint angular velocities
        acc : current joint angular accelerations
        coefficient : Multiplier coefficient on the force field such that yddot = 13 * coeff * xdot

    """
    D = np.array([[0, coefficient], [0, 0]])
    Jacobian = np.array(
        [
            [
                -33 * np.sin(theta[0] + theta[1]) - 30 * np.sin(theta[0]),
                -33 * np.sin(theta[0] + theta[1]),
            ],
            [
                33 * np.cos(theta[0] + theta[1]) + 30 * np.cos(theta[0]),
                33 * np.cos(theta[0] + theta[1]),
            ],
        ]
    )

    return -Jacobian.T @ D @ Jacobian @ omega


def fx(x, u):
    """
    Parameters :
        - x : the state of the system
        - alpha : the body tilt

    return :
        The Jacobian Matrix of the dynamic of the system around the state x
    """
    m = len(x) - 6
    q = x[:-6]
    theta1, theta2, dtheta1, dtheta2,torque1,torque2 = x[-6:]
    torque = np.array([torque1,torque2])
    # Baseline linear plant: freeze inertia at THETA2_REF, drop Coriolis terms.
    cor = 0.0 if LINEARIZE else 1.0
    theta2_M = THETA2_REF if LINEARIZE else theta2
    C = cor * np.array(
        [
            -dtheta2 * (2 * dtheta1 + dtheta2) * a2 * np.sin(theta2),
            dtheta1**2 * a2 * np.sin(theta2),
        ]
    )

    dCdte = cor * np.array(
        [
            -dtheta2 * (2 * dtheta1 + dtheta2) * a2 * np.cos(theta2),
            dtheta1**2 * a2 * np.cos(theta2),
        ]
    )
    dCdos = cor * np.array(
        [-dtheta2 * 2 * a2 * np.sin(theta2), 2 * dtheta1 * a2 * np.sin(theta2)]
    )
    dCdoe = cor * np.array([(-2 * dtheta1 - 2 * dtheta2) * a2 * np.sin(theta2), 0])

    M = np.array(
        [
            [a1 + 2 * a2 * np.cos(theta2_M), a3 + a2 * np.cos(theta2_M)],
            [a3 + a2 * np.cos(theta2_M), a3],
        ]
    )

    Minv = np.linalg.inv(M)

    dM = cor * np.array(
        [[-2 * a2 * np.sin(theta2), -a2 * np.sin(theta2)], [-a2 * np.sin(theta2), 0]]
    )

    
    # Compute acceleration dependencies
    dtheta = np.array([dtheta1, dtheta2])

    d_accel_dtheta1 = Minv @ ( - dCdos - Viscous @ np.array([1, 0])
    )
    d_accel_theta2 = -Minv @ (
        dM @ Minv @ (torque - C - Viscous @ dtheta)
    ) + Minv @ (- dCdte)
    d_accel_dtheta2 = Minv @ (
        - dCdoe - Viscous @ np.array([0, 1])
    )
    d_accel_tau1 = Minv @ np.array([1, 0])
    d_accel_tau2 = Minv @ np.array([0, 1])

    # Construct the Jacobian matrix
    A_arm = np.zeros((6, 6))

    A_arm[0, 2] = 1
    A_arm[1, 3] = 1

    # Acceleration contributions
   
    A_arm[2, 2] = d_accel_dtheta1[0]
    A_arm[2, 1] = d_accel_theta2[0]
    A_arm[2, 3] = d_accel_dtheta2[0]
    A_arm[2, 4] = d_accel_tau1[0]
    A_arm[2, 5] = d_accel_tau2[0]

    A_arm[3, 2] = d_accel_dtheta1[1]
    A_arm[3, 1] = d_accel_theta2[1]
    A_arm[3, 3] = d_accel_dtheta2[1]
    A_arm[3, 4] = d_accel_tau1[1]
    A_arm[3, 5] = d_accel_tau2[1]

    # First-order muscle/actuation dynamics: tau_dot = (Wout @ r - tau) / tau_act
    # d(tau_dot)/d(tau) = -1/tau_act ; d(tau_dot)/d(r) = Wout/tau_act
    A_arm[4, 4] = -1 / tau
    A_arm[5, 5] = -1 / tau

    A = np.zeros((m + 6, m + 6))
    A[:m, :m] = W
    A[m:, m:] = A_arm
    A[-2:, :m] = Wout / tau
    return A


def f(x, u, F=0):
    r,q = x[:-6],x[-6:]
    # Baseline linear plant: freeze inertia at THETA2_REF, drop Coriolis terms.
    cor = 0.0 if LINEARIZE else 1.0
    q1M = THETA2_REF if LINEARIZE else q[1]
    C = cor * np.array(
        [-q[3] * (2 * q[2] + q[3]) * a2 * np.sin(q[1]), q[2] ** 2 * a2 * np.sin(q[1])]
    )

    Denominator = a3 * (a1 - a3) - a2**2 * np.cos(q1M) ** 2
    Minv = np.array(
        [
            [a3 / Denominator, (-a2 * np.cos(q1M) - a3) / Denominator],
            [
                (-a2 * np.cos(q1M) - a3) / Denominator,
                (2 * a2 * np.cos(q1M) + a1) / Denominator,
            ],
        ]
    )
    theta = Minv @ (q[4:6] - Viscous @ q[2:4] - C + F)
    torque = (Wout @ r-q[4:6]) / .06

    return np.concatenate([W@r+u, [q[2], q[3], theta[0], theta[1],torque[0], torque[1]]])





def fu(x, u=None):
    m  = len(x) - 6
    B = np.zeros((len(x), m))
    B[:m,:m] = np.identity(m)

    return B


def l(x, u, r1, xtarg=0, w1=0, w2=0):
    return r1 * np.sum(u**2) / 2


def lx(x, u, xtarg=0, w1=0, w2=0):
    return np.zeros(x.shape)


def lu(x, u, r1):
    return r1 * u


def lxx(x, w1=0, w2=0):
    return np.zeros((x.shape[0], x.shape[0]))


def luu(x, u, r1):
    return np.diag(np.ones(len(u))) * r1


def h(x, w1, w2, xtarg):
    q=x[-6:]
    return w1 / 2 * ((q[0] - xtarg[0]) ** 2 + (q[1] - xtarg[1]) ** 2) + w2 / 2 * (
        q[2] ** 2 + q[3] ** 2
    )


def hx(x, w1, w2, xtarg):
    q=x[-6:]
    return np.concatenate([np.zeros(len(x) - 6), [
        w1 * (q[0] - xtarg[0]), w1 * (q[1] - xtarg[1]), w2 * q[2], w2 * q[3],0,0
    ]])


def hxx(x, w1, w2):
    Q = np.zeros((len(x), len(x)))
    Q[-6, -6] = w1
    Q[-5, -5] = w1
    Q[-4, -4] = w2
    Q[-3, -3] = w2
    return Q


def Kalman(Omega_measure, Omega_sens, A, sigma, H):
    K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
    sigma = Omega_sens + (A - K @ H) @ sigma @ A.T
    return K, sigma


def step1(x0, u, Duration):
    K = np.shape(u)[0]
    dt = Duration / (K)
    newx = np.zeros((K + 1, len(x0)))
    newx[0] = np.copy(x0)

    for i in range(K):
        newx[i + 1] = newx[i] + dt * f(newx[i], u[i])

    return newx


def step2(x, u, Duration, w1, w2, r1, xtarg):
    K = np.shape(u)[0]
    dt = Duration / K
    n, m = len(x[0]), len(u[0])

    A, B = np.zeros((K, n, n)), np.zeros((K, n, m))
    q, qbold = np.zeros(K + 1), np.zeros((K + 1, n))
    r, Q, R = np.zeros((K, m)), np.zeros((K + 1, n, n)), np.zeros((K, m, m))

    for i in range(K):
        A[i] = np.identity(n) + dt * fx(x[i], u[i])
        B[i] = dt * fu(x[i], u[i])
        q[i] = dt * l(x[i], u[i], r1, xtarg, w1, w2)
        qbold[i] = dt * lx(x[i], u[i], xtarg, w1, w2)
        r[i] = dt * lu(x[i], u[i], r1)
        Q[i] = dt * lxx(x[i], w1, w2)
        R[i] = dt * luu(x[i], u[i], r1)

    q[-1], qbold[-1], Q[-1] = (
        h(x[-1], w1, w2, xtarg),
        hx(x[-1], w1, w2, xtarg),
        hxx(x[-1], w1, w2),
    )
    return A, B, q, qbold, r, Q, R


def step3(A, B, C, cbold, q, qbold, r, Q, R, eps):
    K = A.shape[0]
    n, m = np.shape(B[0])
    S = np.zeros((K + 1, n, n))
    s = np.zeros(K + 1)
    sbold = np.zeros((K + 1, n))
    l = np.zeros((K, m))
    L = np.zeros((K, m, n))

    S[-1] = Q[-1]
    s[-1] = q[-1]
    sbold[-1] = qbold[-1]

    # Deterministic fast path: the control matrix B has nonzero entries only in
    # its top m rows (control acts on the first m = #nodes states, B[m:] = 0).
    # So B.T @ X = Bblk.T @ X[:m], which removes several (N+6)^3 matmuls per
    # timestep. Used when there is no multiplicative noise (C is None).
    deterministic = C is None

    for k in np.arange(K - 1, -1, -1):
        Snext = S[k + 1]
        if deterministic:
            Bblk = B[k][:m, :]                # top block (bottom rows are zero)
            SA = Snext @ A[k]
            G = Bblk.T @ SA[:m, :]
            gbold = r[k] + Bblk.T @ sbold[k + 1][:m]
            H = R[k] + Bblk.T @ Snext[:m, :m] @ Bblk
            ASA = A[k].T @ SA
            temp3 = 0.0
        else:
            temp1 = temp2 = temp3 = 0
            for i in range(m):
                temp1 += C[k, i, :, :].T @ Snext @ cbold[k, i, :]
                temp2 += C[k, i, :, :].T @ Snext @ C[k, i, :, :]
                temp3 += cbold[k, i, :].T @ Snext @ cbold[k, i, :]
            gbold = r[k] + B[k].T @ sbold[k + 1] + temp1
            G = B[k].T @ Snext @ A[k]
            H = R[k] + B[k].T @ Snext @ B[k] + temp2
            ASA = A[k].T @ Snext @ A[k]

        # H is symmetric PSD; use eigh (real spectrum) and floor eigenvalues at
        # eps for a stable regularized inverse (Levenberg-style).
        w_eig, V_eig = np.linalg.eigh(0.5 * (H + H.T))
        w_eig = np.where(w_eig < eps, eps, w_eig)
        Hinv = (V_eig * (1.0 / w_eig)) @ V_eig.T

        HinvG = Hinv @ G
        Hinvg = Hinv @ gbold
        S[k] = Q[k] + ASA - G.T @ HinvG
        sbold[k] = qbold[k] + A[k].T @ sbold[k + 1] - G.T @ Hinvg
        s[k] = q[k] + s[k + 1] + 0.5 * temp3 - 0.5 * gbold.T @ Hinvg

        l[k] = -Hinvg
        L[k] = -HinvG

    return l, L


def step4(l, L, K, A, B):
    m, n = L[0].shape
    x = np.zeros(n)
    u_incr = np.zeros((K, m))

    for k in range(K):
        u_incr[k] = l[k] + L[k] @ x
        x = A[k] @ x + B[k] @ u_incr[k]

    return u_incr


def step5(
    x0,
    l,
    L,
    Duration,
    Noise,
    A,
    B,
    Num_steps,
    bestu,
    kdelay,
    motornoise_variance,
    FF,
    ff_power,
):
    dt = Duration / (Num_steps)
    Num_Var = len(x0)

    x0 = np.tile(x0, kdelay + 1)
    xref = np.zeros((Num_steps + 1, Num_Var * (kdelay + 1)))
    xref[0] = np.copy(x0)
    newx = np.zeros((Num_steps + 1, Num_Var * (kdelay + 1)))
    newx[0] = np.copy(x0)
    xhat = np.zeros((Num_steps + 1, Num_Var * (kdelay + 1)))

    H = np.zeros((Num_Var, (kdelay + 1) * Num_Var))
    H[:, (kdelay) * Num_Var :] = np.identity(Num_Var)

    sigma = np.zeros((Num_Var * (kdelay + 1), Num_Var * (kdelay + 1)))
    Omega_measure = np.diag(np.ones(Num_Var * (kdelay + 1))) * 1e-4
    F = 0
    for i in range(Num_steps):
        if i != 0:
            acc = (f(newx[i, :Num_Var], u)[:, 2:4] + F).reshape(2)
        else:
            acc = np.zeros(2)
        F = (
            compute_forcefield(newx[i, 0:2], newx[i, 2:4], ff_power)
            if FF == True
            else np.array([0, 0])
        )
        Extended_A = np.zeros(((kdelay + 1) * Num_Var, (kdelay + 1) * Num_Var))
        Extended_A[:Num_Var, :Num_Var] = A[i]
        Extended_A[Num_Var:, :-Num_Var] = np.identity((kdelay) * Num_Var)
        Extended_B = np.zeros(((kdelay + 1) * Num_Var, 6))
        Extended_B[:Num_Var] = B[i]

        deltau = l[i] + L[i] @ xhat[i, :Num_Var]
        u = bestu[i] + deltau

        Omega_sens = np.zeros((len(x0), len(x0)))
        for idx in [2, 3]:
            Omega_sens[idx, idx] = motornoise_variance
        K, sigma = Kalman(Omega_measure, Omega_sens, Extended_A, sigma, H)

        passed_newx = np.copy(newx[i, :-Num_Var])
        newx[i + 1, :Num_Var] = newx[i, :Num_Var] + dt * f(newx[i, :Num_Var], u, F)
        newx[i + 1, Num_Var:] = passed_newx

        passed_xref = np.copy(xref[i, :-Num_Var])
        xref[i + 1, :Num_Var] = xref[i, :Num_Var] + dt * f(
            xref[i, :Num_Var], bestu[i], F=0
        )
        xref[i + 1, Num_Var:] = passed_xref

        if Noise:
            newx[i + 1, 2:4] += np.random.normal(0, np.sqrt(motornoise_variance), 2)

        y = H @ (newx[i] - xref[i])
        if Noise:
            y += np.random.normal(0, 1e-2, len(y))

        xhat[i + 1] = (Extended_A @ xhat[i] + Extended_B @ deltau) + K @ (
            y - H @ xhat[i]
        )
    return newx[:, :Num_Var]


def hand_xy(x, l1=30, l2=33):
    """Cartesian hand position from the full (network+arm) state trajectory.

    Only the last six entries (the arm block) are used, so this works both on a
    single state vector and on a trajectory array of shape (..., N+6).
    """
    q = np.asarray(x)[..., -6:]
    th1, th2 = q[..., 0], q[..., 1]
    X = np.cos(th1 + th2) * l2 + np.cos(th1) * l1
    Y = np.sin(th1 + th2) * l2 + np.sin(th1) * l1
    return X, Y


def total_cost(x, u, Duration, w1, w2, r1, xtarg):
    """Total finite-horizon cost of a nominal trajectory (running + terminal)."""
    K = u.shape[0]
    dt = Duration / K
    J = h(x[-1], w1, w2, xtarg)
    for k in range(K):
        J += dt * l(x[k], u[k], r1)
    return J


def forward_pass(x0, x_nom, u_nom, l_gain, L_gain, Duration, alpha=1.0):
    """Nonlinear closed-loop rollout of the iLQG update with step size alpha."""
    K = u_nom.shape[0]
    dt = Duration / K
    xnew = np.zeros_like(x_nom)
    unew = np.zeros_like(u_nom)
    xnew[0] = x0
    for k in range(K):
        dx = xnew[k] - x_nom[k]
        unew[k] = u_nom[k] + alpha * l_gain[k] + L_gain[k] @ dx
        xnew[k + 1] = xnew[k] + dt * f(xnew[k], unew[k])
    return xnew, unew


def simulate_ILQG(
    Duration=0.5,
    w1=1e3,
    w2=1,
    r1=1e-4,
    targets=[0, 50],
    start=[0, 30],
    K=120,
    eps=1e-6,
    max_iter=200,
    tol=1e-6,
    r0=None,
    print_iterations=True,
    return_gains=False,
):
    """Solve the combined network-arm reaching problem with iLQG.

    The state is x = [r (network nodes), theta_s, theta_e, omega_s, omega_e,
    tau_s, tau_e]. The control u acts directly on the N network nodes. Planning
    is deterministic (finite horizon, starting at movement onset -- no
    preparatory epoch), so the returned nominal trajectory is the optimal
    open-loop solution.

    Parameters
        - Duration : movement duration [s]
        - w1, w2   : terminal penalty on joint-angle error to target / on
                     terminal joint velocity
        - r1       : control (network input) cost
        - targets  : target hand position [x, y] in cm
        - start    : starting hand position [x, y] in cm
        - K        : number of time steps
        - eps      : eigenvalue floor for the regularized control-Hessian inverse
        - max_iter : maximum iLQG iterations
        - r0       : initial network state (defaults to rest, zeros)

    Returns
        - X, Y : cartesian hand trajectory (cm), length K+1
        - x    : full state trajectory, shape (K+1, N+6)
        - u    : optimal network inputs, shape (K, N)
        (- l_gain, L_gain if return_gains)
    """
    N = W.shape[0]

    obj1, obj2 = compute_angles_from_cartesian(targets[0], targets[1])
    st1, st2 = compute_angles_from_cartesian(start[0], start[1])
    xtarg = np.array([obj1, obj2])

    if r0 is None:
        r0 = np.zeros(N)
    x0 = np.concatenate([r0, [st1, st2, 0, 0, 0, 0]])

    u = np.zeros((K, N))
    x = step1(x0, u, Duration)
    Jcost = total_cost(x, u, Duration, w1, w2, r1, xtarg)

    alphas = 0.5 ** np.arange(12)  # backtracking line-search step sizes
    l_gain = L_gain = None
    for iterate in range(max_iter):
        A, B, q, qbold, r, Q, R = step2(x, u, Duration, w1, w2, r1, xtarg)
        l_gain, L_gain = step3(A, B, None, None, q, qbold, r, Q, R, eps)

        improved = False
        for alpha in alphas:
            xnew, unew = forward_pass(x0, x, u, l_gain, L_gain, Duration, alpha)
            Jnew = total_cost(xnew, unew, Duration, w1, w2, r1, xtarg)
            if np.isfinite(Jnew) and Jnew < Jcost:
                improved = True
                break

        if not improved:
            if print_iterations:
                print(f"Converged (no improvement) at iter {iterate}, cost {Jcost:.6g}")
            break

        du = np.max(np.abs(unew - u))
        rel = (Jcost - Jnew) / max(abs(Jcost), 1e-12)
        x, u, Jcost = xnew, unew, Jnew
        if du < 1e-10 or rel < tol:
            if print_iterations:
                print(f"Converged at iter {iterate}, cost {Jcost:.6g}")
            break
    else:
        if print_iterations:
            print(f"Reached max_iter, cost {Jcost:.6g}")

    X, Y = hand_xy(x)
    if return_gains:
        return X, Y, x, u, l_gain, L_gain
    return X, Y, x, u
