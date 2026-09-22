import numpy as np
from math import *

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

# Random network driving the 6 muscles (Kalidindi & Crevecoeur 2025, Eqs. 7-8):
#   tau_net * dr/dt = -r + W r + eps + u ,   a = Wout r   (a = 6 muscle commands)
# State of the full system: x = [r (N_NET), theta1, theta2, dtheta1, dtheta2]
N_NET = 100
N_MUSCLES = 6
TAU_NET = 0.02
G_NET = 0.8  # fully connected random network, W ~ N(0, g^2 / n)
_rng = np.random.default_rng(0)
W_NET = _rng.normal(0, G_NET / np.sqrt(N_NET), (N_NET, N_NET))
WOUT = _rng.normal(0, 0.5, (N_MUSCLES, N_NET))
EPS_NET = _rng.normal(0, 5, N_NET)  # constant input -> heterogeneous spontaneous activity
J_NET = (W_NET - np.identity(N_NET)) / TAU_NET


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

class NonlinearArm:
    def __init__(self):
        pass

    def f_arm(self, x, u, F=0):
        C = np.array(
            [-x[3] * (2 * x[2] + x[3]) * a2 * np.sin(x[1]), x[2] ** 2 * a2 * np.sin(x[1])]
        )

        Denominator = a3 * (a1 - a3) - a2**2 * np.cos(x[1]) ** 2
        Minv = np.array(
            [
                [a3 / Denominator, (-a2 * np.cos(x[1]) - a3) / Denominator],
                [
                    (-a2 * np.cos(x[1]) - a3) / Denominator,
                    (2 * a2 * np.cos(x[1]) + a1) / Denominator,
                ],
            ]
        )
        A = np.array([[2, -2, 0, 0, 1.5, -2], [0, 0, 2, -2, 2, -1.5]])

        l0 = np.array([7.32, 3.26, 6.4, 4.26, 5.95, 4.04])
        theta0 = np.array(
            [
                [
                    2 * pi / 360 * 15,
                    2 * pi / 360 * 4.88,
                    0,
                    0,
                    2 * pi / 360 * 4.5,
                    2 * pi / 360 * 2.12,
                ],
                [
                    0,
                    0,
                    2 * pi / 360 * 80.86,
                    2 * pi / 360 * 109.32,
                    2 * pi / 360 * 92.96,
                    2 * pi / 360 * 91.52,
                ],
            ]
        )
        l = 1 + A[0] * (theta0[0] - x[0]) / l0 + A[1] * (theta0[1] - x[1]) / l0
        v = A[0] * (-x[2]) / l0 + A[1] * (-x[3]) / l0

        fl = np.exp(-(np.abs((l**1.55 - 1) / 0.81) ** 2.12))

        ff_v = np.where(
            v <= 0,
            (-7.39 - v) / (-7.39 + (-3.21 + 4.17) * v),
            (0.62 - (-3.12 + 4.21 * l - 2.67 * l**2) * v) / (0.62 + v),
        )
        theta = Minv @ (A @ (u * fl * ff_v) - Viscous @ x[2:4] - C + F)

        return np.array([[x[2], x[3], theta[0], theta[1]]])

    def fx_arm(self, x, u):
        """
        Parameters :
            - x : the state of the system
            - alpha : the body tilt

        return :
            The Jacobian Matrix of the dynamic of the system around the state x
        """

        theta1, theta2, dtheta1, dtheta2 = x[:4]
        C = np.array(
            [
                -dtheta2 * (2 * dtheta1 + dtheta2) * a2 * np.sin(theta2),
                dtheta1**2 * a2 * np.sin(theta2),
            ]
        )

        dCdte = np.array(
            [
                -dtheta2 * (2 * dtheta1 + dtheta2) * a2 * np.cos(theta2),
                dtheta1**2 * a2 * np.cos(theta2),
            ]
        )
        dCdos = np.array(
            [-dtheta2 * 2 * a2 * np.sin(theta2), 2 * dtheta1 * a2 * np.sin(theta2)]
        )
        dCdoe = np.array([(-2 * dtheta1 - 2 * dtheta2) * a2 * np.sin(theta2), 0])

        M = np.array(
            [
                [a1 + 2 * a2 * np.cos(theta2), a3 + a2 * np.cos(theta2)],
                [a3 + a2 * np.cos(theta2), a3],
            ]
        )

        Minv = np.linalg.inv(M)

        dM = np.array(
            [[-2 * a2 * np.sin(theta2), -a2 * np.sin(theta2)], [-a2 * np.sin(theta2), 0]]
        )

        A = np.array([[2, -2, 0, 0, 1.5, -2], [0, 0, 2, -2, 2, -1.5]])

        l0 = np.array([7.32, 3.26, 6.4, 4.26, 5.95, 4.04])
        theta0 = np.array(
            [
                [
                    2 * pi / 360 * 15,
                    2 * pi / 360 * 4.88,
                    0,
                    0,
                    2 * pi / 360 * 4.5,
                    2 * pi / 360 * 2.12,
                ],
                [
                    0,
                    0,
                    2 * pi / 360 * 80.86,
                    2 * pi / 360 * 109.32,
                    2 * pi / 360 * 92.96,
                    2 * pi / 360 * 91.52,
                ],
            ]
        )
        l = 1 + A[0] * (theta0[0] - x[0]) / l0 + A[1] * (theta0[1] - x[1]) / l0
        dldts = -A[0] / l0
        dldte = -A[1] / l0

        v = A[0] * (-x[2]) / l0 + A[1] * (-x[3]) / l0
        dvdos = -A[0] / l0
        dvdoe = -A[1] / l0
        temp = (l**1.55 - 1) / 0.81
        fl = np.exp(-(np.abs(temp) ** 2.12))

        dfldl = (
            -fl
            * 2.12
            * np.abs(temp) ** 1.12
            * np.sign(temp)
            * (1.55 * l**0.55 / 0.81)
        )
        fv = np.where(
            v <= 0,
            (-7.39 - v) / (-7.39 + (-3.21 + 4.17) * v),
            (0.62 - (-3.12 + 4.21 * l - 2.67 * l**2) * v) / (0.62 + v),
        )
        dfvdl = np.where(v <= 0, 0, v * (-4.21 + 5.34 * l) / (0.62 + v))

        dfvdv = np.where(
            v <= 0,
            7.39 * (1 + 0.96) / (-7.39 + 0.96 * v) ** 2,
            -0.62 * (-3.12 + 4.21 * l - 2.67 * l**2 + 1) / (0.62 + v) ** 2,
        )

        dfldts = dfldl * dldts
        dfldte = dfldl * dldte
        dfvdts = dfvdl * dldts
        dfvdte = dfvdl * dldte
        dfvdos = dfvdv * dvdos
        dfvdoe = dfvdv * dvdoe

        # Compute acceleration dependencies
        dtheta = np.array([dtheta1, dtheta2])

        d_accel_theta1 = Minv @ (A @ (u * (dfldts * fv + fl * dfvdts)))
        d_accel_dtheta1 = Minv @ (
            A @ (u * dfvdos * fl) - dCdos - Viscous @ np.array([1, 0])
        )
        d_accel_theta2 = -Minv @ (
            dM @ Minv @ (A @ (u * fl * fv) - C - Viscous @ dtheta)
        ) + Minv @ (A @ (u * (dfldte * fv + fl * dfvdte)) - dCdte)
        d_accel_dtheta2 = Minv @ (
            A @ (u * dfvdoe * fl) - dCdoe - Viscous @ np.array([0, 1])
        )

        # Construct the Jacobian matrix
        A = np.zeros((4, 4))

        A[0, 2] = 1
        A[1, 3] = 1

        # Acceleration contributions
        A[2, 0] = d_accel_theta1[0]
        A[2, 2] = d_accel_dtheta1[0]
        A[2, 1] = d_accel_theta2[0]
        A[2, 3] = d_accel_dtheta2[0]

        A[3, 0] = d_accel_theta1[1]
        A[3, 2] = d_accel_dtheta1[1]
        A[3, 1] = d_accel_theta2[1]
        A[3, 3] = d_accel_dtheta2[1]

        return A



    def fu_arm(self, x, u):
        Denominator = a3 * (a1 - a3) - a2**2 * np.cos(x[1]) ** 2
        Minv = np.array(
            [
                [a3 / Denominator, (-a2 * np.cos(x[1]) - a3) / Denominator],
                [
                    (-a2 * np.cos(x[1]) - a3) / Denominator,
                    (2 * a2 * np.cos(x[1]) + a1) / Denominator,
                ],
            ]
        )
        A = np.array([[2, -2, 0, 0, 1.5, -2], [0, 0, 2, -2, 2, -1.5]])

        l0 = np.array([7.32, 3.26, 6.4, 4.26, 5.95, 4.04])
        theta0 = np.array(
            [
                [
                    2 * pi / 360 * 15,
                    2 * pi / 360 * 4.88,
                    0,
                    0,
                    2 * pi / 360 * 4.5,
                    2 * pi / 360 * 2.12,
                ],
                [
                    0,
                    0,
                    2 * pi / 360 * 80.86,
                    2 * pi / 360 * 109.32,
                    2 * pi / 360 * 92.96,
                    2 * pi / 360 * 91.52,
                ],
            ]
        )
        l = 1 + A[0] * (theta0[0] - x[0]) / l0 + A[1] * (theta0[1] - x[1]) / l0
        v = A[0] * (-x[2]) / l0 + A[1] * (-x[3]) / l0
        # Equation (6): fl(l)
        fl = np.exp(-(np.abs((l**1.55 - 1) / 0.81) ** 2.12))
        # Equation (7): ff_v(l, v)
        fv = np.where(
            v <= 0,
            (-7.39 - v) / (-7.39 + (-3.21 + 4.17) * v),
            (0.62 - (-3.12 + 4.21 * l - 2.67 * l**2) * v) / (0.62 + v),
        )
        sol = np.zeros((4, 6))
        for i in range(6):
            du = np.zeros(6)
            du[i] = 1
            sol[2:, i] = Minv @ (A @ (du * fl * fv))
        return sol



class LinearArm:
    def __init__(self):
        pass

    def f_arm(self, x, u, F=0):
        xlin = np.array([np.pi/4,np.pi/2])
        Denominator = a3 * (a1 - a3) - a2**2 * np.cos(xlin[1]) ** 2
        Minv = np.array(
            [
                [a3 / Denominator, (-a2 * np.cos(xlin[1]) - a3) / Denominator],
                [
                    (-a2 * np.cos(xlin[1]) - a3) / Denominator,
                    (2 * a2 * np.cos(xlin[1]) + a1) / Denominator,
                ],
            ]
        )
        A = np.array([[2, -2, 0, 0, 1.5, -2], [0, 0, 2, -2, 2, -1.5]])
    
        
        theta = Minv @ (A @ (u) - Viscous @ x[2:4] + F)
    
        return np.array([[x[2], x[3], theta[0], theta[1]]])
    
    def fx_arm(self, x, u):
        """
        Parameters :
            - x : the state of the system
            - alpha : the body tilt
    
        return :
            The Jacobian Matrix of the dynamic of the system around the state x
        """
    
        xlin = np.array([np.pi/4,np.pi/2])
    
        M = np.array(
            [
                [a1 + 2 * a2 * np.cos(xlin[1]), a3 + a2 * np.cos(xlin[1])],
                [a3 + a2 * np.cos(xlin[1]), a3],
            ]
        )
    
        Minv = np.linalg.inv(M)
    
    
        A = np.array([[2, -2, 0, 0, 1.5, -2], [0, 0, 2, -2, 2, -1.5]])
    

        d_accel_dtheta1 = Minv @ (- Viscous @ np.array([1, 0])
        )
        d_accel_dtheta2 = Minv @ ( - Viscous @ np.array([0, 1])
        )
    
        # Construct the Jacobian matrix
        A = np.zeros((4, 4))
    
        A[0, 2] = 1
        A[1, 3] = 1
    
        # Acceleration contributions

        A[2, 2] = d_accel_dtheta1[0]
        A[2, 3] = d_accel_dtheta2[0]
    
        A[3, 2] = d_accel_dtheta1[1]
        A[3, 3] = d_accel_dtheta2[1]
    
        return A
    
    
    def fu_arm(self, x, u):
        xlin = np.array([np.pi/4,np.pi/2])
        M = np.array(
                    [
                        [a1 + 2 * a2 * np.cos(xlin[1]), a3 + a2 * np.cos(xlin[1])],
                        [a3 + a2 * np.cos(xlin[1]), a3],
                    ]
                )
            
        Minv = np.linalg.inv(M)
            
        A = np.array([[2, -2, 0, 0, 1.5, -2], [0, 0, 2, -2, 2, -1.5]])
    
        
        sol = np.zeros((4, 6))
        for i in range(6):
            du = np.zeros(6)
            du[i] = 1
            sol[2:, i] = Minv @ (A @ (du))
        return sol



def f(Body, x, u, F=0):
    """Network-body dynamics: the network output Wout r replaces the muscle command."""
    r, q = x[:N_NET], x[N_NET:]
    drdt = (-r + W_NET @ r + EPS_NET + u) / TAU_NET
    return np.concatenate([drdt, Body.f_arm(q, WOUT @ r, F)[0]])


def fx(Body, x, u):
    r, q = x[:N_NET], x[N_NET:]
    a = WOUT @ r
    A = np.zeros((len(x), len(x)))
    A[:N_NET, :N_NET] = J_NET
    A[N_NET:, :N_NET] = Body.fu_arm(q, a) @ WOUT
    A[N_NET:, N_NET:] = Body.fx_arm(q, a)
    return A



def fu(x, u):
    B = np.zeros((len(x), N_NET))
    B[:N_NET] = np.identity(N_NET) / TAU_NET
    return B


# The effort is measured relative to the input U_REST holding the network at its
# spontaneous state (otherwise the finite-horizon controller releases it before
# the end of the movement, producing a target-independent drift of the network).
# r2 : running cost on the muscle commands a = Wout r (actuation part of the
# limb state penalised by Q in the paper). Without it, large muscle commands
# are almost free since the network amplifies u.
def l(x, u, r1, xtarg=0, w1=0, w2=0, r2=0):
    return r1 * np.sum((u) ** 2) / 2 + r2 * np.sum((WOUT @ x[:N_NET]) ** 2) / 2


def lx(x, u, xtarg=0, w1=0, w2=0, r2=0):
    g = np.zeros(len(x))
    g[:N_NET] = r2 * WOUT.T @ (WOUT @ x[:N_NET])
    return g


def lu(x, u, r1):
    return r1 * (u)


def lxx(x, w1=0, w2=0, r2=0):
    H = np.zeros((len(x), len(x)))
    H[:N_NET, :N_NET] = r2 * WOUT.T @ WOUT
    return H


def luu(x, u, r1):
    return np.diag(np.ones(len(u))) * r1


def h(x, w1, w2, xtarg):
    x = x[-4:]
    return w1 / 2 * ((x[0] - xtarg[0]) ** 2 + (x[1] - xtarg[1]) ** 2) + w2 / 2 * (
        x[2] ** 2 + x[3] ** 2
    )


def hx(x, w1, w2, xtarg):
    q = x[-4:]
    return np.concatenate([np.zeros(len(x) - 4), [
        w1 * (q[0] - xtarg[0]), w1 * (q[1] - xtarg[1]), w2 * q[2], w2 * q[3]
    ]])


def hxx(x, w1, w2):
    return np.diag(np.concatenate([np.zeros(len(x) - 4), [w1, w1, w2, w2]]))


def total_cost(x, u, Duration, w1, w2, r1, xtarg, r2=0):
    dt = Duration / len(u)
    return sum(dt * l(x[i], u[i], r1, r2=r2) for i in range(len(u))) + h(x[-1], w1, w2, xtarg)


def Kalman(Omega_measure, Omega_sens, A, sigma, H):
    K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
    sigma = Omega_sens + (A - K @ H) @ sigma @ A.T
    return K, sigma


def step1(Body, x0, u, Duration):
    K = np.shape(u)[0]
    dt = Duration / (K)
    newx = np.zeros((K + 1, len(x0)))
    newx[0] = np.copy(x0)

    for i in range(K):
        newx[i + 1] = newx[i] + dt * f(Body, newx[i], u[i])

    return newx


def step2(Body, x, u, Duration, w1, w2, r1, xtarg, r2=0):
    K = np.shape(u)[0]
    dt = Duration / K
    n, m = len(x[0]), len(u[0])

    A, B = np.zeros((K, n, n)), np.zeros((K, n, m))
    q, qbold = np.zeros(K + 1), np.zeros((K + 1, n))
    r, Q, R = np.zeros((K, m)), np.zeros((K + 1, n, n)), np.zeros((K, m, m))

    for i in range(K):
        A[i] = np.identity(n) + dt * fx(Body, x[i], u[i])
        B[i] = dt * fu(x[i], u[i])
        q[i] = dt * l(x[i], u[i], r1, xtarg, w1, w2, r2)
        qbold[i] = dt * lx(x[i], u[i], xtarg, w1, w2, r2)
        r[i] = dt * lu(x[i], u[i], r1)
        Q[i] = dt * lxx(x[i], w1, w2, r2)
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

    for k in np.arange(K - 1, -1, -1):
        temp1, temp2, temp3 = 0, 0, 0

        for i in range(C.shape[1]):
            temp1 += C[k, i, :, :].T @ S[k + 1] @ cbold[k, i, :]
            temp2 += C[k, i, :, :].T @ S[k + 1] @ C[k, i, :, :]
            temp3 += cbold[k, i, :].T @ S[k + 1] @ cbold[k, i, :]

        gbold = r[k] + B[k].T @ sbold[k + 1] + temp1
        G = B[k].T @ S[k + 1] @ A[k]
        H = R[k] + B[k].T @ S[k + 1] @ B[k] + temp2

        # H is symmetric: eigh gives orthogonal eigenvectors (no inverse needed,
        # which is the bottleneck with 100 inputs)
        eigenvalues, eigenvectors = np.linalg.eigh((H + H.T) / 2)
        eigenvalues = np.maximum(eigenvalues, eps)
        Hinv = (eigenvectors / eigenvalues) @ eigenvectors.T

        S[k] = Q[k] + A[k].T @ S[k + 1] @ A[k] - G.T @ Hinv @ G
        sbold[k] = qbold[k] + A[k].T @ sbold[k + 1] - G.T @ Hinv @ gbold
        s[k] = q[k] + s[k + 1] + 0.5 * temp3 - 0.5 * gbold.T @ Hinv @ gbold

        l[k] = -Hinv @ gbold
        L[k] = -Hinv @ G

    return l, L


def step4(l, L, K, A, B):
    m, n = L[0].shape
    x = np.zeros(n)
    u_incr = np.zeros((K, m))

    for k in range(K):
        u_incr[k] = l[k] + L[k] @ x
        x = A[k] @ x + B[k] @ u_incr[k]

    return u_incr


def forward_pass(Body, x0, xbar, ubar, l, L, alpha, Duration):
    K = np.shape(ubar)[0]
    dt = Duration / K
    x = np.zeros(np.shape(xbar))
    u = np.zeros(np.shape(ubar))
    x[0] = np.copy(x0)
    for k in range(K):
        u[k] = ubar[k] + alpha * l[k] + L[k] @ (x[k] - xbar[k])
        x[k + 1] = x[k] + dt * f(Body, x[k], u[k])
    return x, u


def step5(
    Body, 
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
    Omega_measure = np.diag(np.ones(Num_Var)) * 1e-4
    F = 0
    for i in range(Num_steps):
        F = (
            compute_forcefield(
                newx[i, N_NET : N_NET + 2], newx[i, N_NET + 2 : N_NET + 4], ff_power
            )
            if FF == True
            else np.array([0, 0])
        )
        Extended_A = np.zeros(((kdelay + 1) * Num_Var, (kdelay + 1) * Num_Var))
        Extended_A[:Num_Var, :Num_Var] = A[i]
        Extended_A[Num_Var:, :-Num_Var] = np.identity((kdelay) * Num_Var)
        Extended_B = np.zeros(((kdelay + 1) * Num_Var, B.shape[2]))
        Extended_B[:Num_Var] = B[i]

        deltau = l[i] + L[i] @ xhat[i, :Num_Var]
        u = bestu[i] + deltau

        Omega_sens = np.zeros((len(x0), len(x0)))
        for idx in [N_NET + 2, N_NET + 3]:
            Omega_sens[idx, idx] = motornoise_variance
        K, sigma = Kalman(Omega_measure, Omega_sens, Extended_A, sigma, H)

        passed_newx = np.copy(newx[i, :-Num_Var])
        newx[i + 1, :Num_Var] = newx[i, :Num_Var] + dt * f(Body, newx[i, :Num_Var], u, F)
        newx[i + 1, Num_Var:] = passed_newx

        passed_xref = np.copy(xref[i, :-Num_Var])
        xref[i + 1, :Num_Var] = xref[i, :Num_Var] + dt * f(Body, 
            xref[i, :Num_Var], bestu[i], F=0
        )
        xref[i + 1, Num_Var:] = passed_xref

        if Noise:
            newx[i + 1, N_NET + 2 : N_NET + 4] += np.random.normal(0, np.sqrt(motornoise_variance), 2)

        y = H @ (newx[i] - xref[i])
        if Noise:
            y += np.random.normal(0, 1e-2, len(y))

        xhat[i + 1] = (Extended_A @ xhat[i] + Extended_B @ deltau) + K @ (
            y - H @ xhat[i]
        )
    return newx[:, :Num_Var]



def direction_preferee(activite):
    """activite : (n_directions, n_temps), activité du neurone de 0 à 600 ms pour chaque direction.
    directions_deg : (n_directions,), direction de chaque mouvement en degrés.
    Renvoie la direction préférée (degrés) et la profondeur de modulation."""
    theta = np.linspace(0,2*np.pi,8,endpoint=False)  # directions en radians
    y = activite.mean(axis=1)                                   # activité moyenne sur 0–600 ms
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    b0, b1, b2 = np.linalg.lstsq(X, y, rcond=None)[0]           # y = b0 + b1 cos θ + b2 sin θ
    return np.rad2deg(np.arctan2(b2, b1)) % 360, np.hypot(b1, b2)


def simulate_ILQG(
    Duration=0.5,
    w1=1e3,
    w2=1,
    r1=1e-4,
    r2=1e-3,
    targets=[0, 50],
    start=[0, 30],
    K=120,
    Noise=False,
    delay=0,
    eps=1e-9,
    motornoise_variance=1e-3,
    print_iterations=True,
    FF=False,
    ff_power=0.3,
    tol=1e-6,
    Linear = False
):
    """
    Parameters :
        - Duration : Movement Duration in sec
        - w1,w2,r1 : Weight of the costs function associated to distance penalty  to the target, end velocity, and motor costs respectively
        - r2 : Weight of the running cost on the muscle commands a = Wout r
        - targets : Position of the target in cartesian coordinates
        - k : Number of iterations
        - start : Starting position of the hand
        - plot : Boolean, True ==> plotting enabled
        - Noise : Boolean, True ==> Noise Activated in the simulation
        - Delay : Sensory Delay in sec
        - motornoise_variance : Variance of the motor noise
        - tol : convergence threshold on the command increment
        - alpha : Body Tilt in radiant

    return :
        - X,Y : Cartesian coordinates of the hand trajectory
        - u : Input command
        - x : Vector state of the trajectory [r (N_NET), theta1, theta2, dtheta1, dtheta2]
    """

    obj1, obj2 = compute_angles_from_cartesian(targets[0], targets[1])
    st1, st2 = compute_angles_from_cartesian(start[0], start[1])

    x0 = np.concatenate([[0] * N_NET, [st1, st2, 0, 0]])
    m, n = N_NET, N_NET + 4
    u = np.zeros((K, m))  # Initial command sequence
    dt = Duration / K
    kdelay = int(delay / dt)

    # Additive motor noise only on the two joint velocities (C = 0)
    cbold = np.zeros((K, 2, n))
    C = np.zeros((K, 2, n, m))
    for i in range(K):
        for j in range(2):
            cbold[i, j, N_NET + 2 + j] = sqrt(motornoise_variance)

    u_incr = np.ones(u.shape) * np.inf
    Body = LinearArm() if Linear else NonlinearArm()  # Body dynamics (nonlinear)
    x = step1(
        Body, x0, u, Duration
    )  # Forward step computing the sequence of state trajectory given a sequence of input u

    for iterate in range(300):
        X = np.cos(x[:, N_NET] + x[:, N_NET + 1]) * 33 + np.cos(x[:, N_NET]) * 30
        Y = np.sin(x[:, N_NET] + x[:, N_NET + 1]) * 33 + np.sin(x[:, N_NET]) * 30

        if (
            np.max(np.abs(u_incr)) < tol
        ):  # If the trajectory improvement is small enough, stop the iteration and perform a full simulation with feedback and potential noise
            x = step5(
                Body,
                x0,
                l,
                L,
                Duration,
                Noise,
                A,
                B,
                K,
                u - u_incr,
                kdelay,
                motornoise_variance,
                FF,
                ff_power,
            )
            X = np.cos(x[:, N_NET] + x[:, N_NET + 1]) * 33 + np.cos(x[:, N_NET]) * 30
            Y = np.sin(x[:, N_NET] + x[:, N_NET + 1]) * 33 + np.sin(x[:, N_NET]) * 30
            if print_iterations:
                print("Solution found at iteration ", iterate)
            break

        A, B, q, qbold, r, Q, R = step2(
            Body, x, u, Duration, w1, w2, r1, np.array([obj1, obj2]), r2
        )  # Compute the Linearizations of the dynamic
        l, L = step3(
            A, B, C, cbold, q, qbold, r, Q, R, eps
        )  # Compute the control gains improvement (feedforward and feedback)
        # Closed-loop forward pass on the nonlinear dynamics with a backtracking
        # line search on the true cost (replaces step4): applying the open-loop
        # increments of step4 makes the iterations diverge on the network-body system.
        J = total_cost(x, u, Duration, w1, w2, r1, np.array([obj1, obj2]), r2)
        alpha = 1.0
        u_incr = np.zeros(u.shape)
        while alpha > 1e-6:
            x_new, u_new = forward_pass(Body, x0, x, u, l, L, alpha, Duration)
            if np.isfinite(x_new).all() and total_cost(
                x_new, u_new, Duration, w1, w2, r1, np.array([obj1, obj2]), r2
            ) <= J:
                u_incr = u_new - u
                x, u = x_new, u_new  # Improves the command sequence
                break
            alpha /= 2
    return X, Y, x, u


if __name__ == "__main__":
    # Center-out reaching test of the network-body iLQG controller
    import matplotlib.pyplot as plt

    CENTER = np.array([0.0, 45.0])  # hand start position [cm]
    RADIUS = 10.0  # reach distance [cm]
    N_DIR = 8
    DURATION, K_STEPS = 0.5, 50
    W1, W2, R1, R2 = 1e3, 1, 1e-4, 0

    angles = np.linspace(0, 2 * np.pi, N_DIR, endpoint=False)
    colors = plt.cm.RdBu(np.linspace(0.05, 0.95, N_DIR))
    time_ = np.linspace(0, DURATION, K_STEPS + 1)[:-1]
    pd_results = {}  # Linear -> (PD of each node [deg], number of nodes per 22.5 deg bin)

    for Linear in [True, False]:

        results = []

        for ang in angles:
            target = CENTER + RADIUS * np.array([np.cos(ang), np.sin(ang)])
            X, Y, x, u = simulate_ILQG(
                Duration=DURATION, w1=W1, w2=W2, r1=R1, r2=R2, K=K_STEPS,
                targets=target, start=CENTER,Linear = Linear
            )
            results.append((X, Y, x, u))
            print(f"target {np.degrees(ang):5.1f} deg : end error "
                  f"{np.hypot(X[-1] - target[0], Y[-1] - target[1]):.3f} cm")

        # The last sample is dropped: u[K-1] only affects r[K], which is not penalised,
        # so the optimal controller releases the holding input at the very last step.
        rates = np.array([res[2][:-1, :N_NET] for res in results])  # (N_DIR, K, N_NET)
        muscles = rates @ WOUT.T  # network output a = Wout r

        pd_of_neurons = np.zeros((N_NET))
        closest_pd = np.zeros((16))

        all_directions = np.arange(0, 360, 22.5)    
        for k in range(N_NET):
            theta = direction_preferee(rates[:, :, k])[0] 
            pd_of_neurons[k] = theta
            d = np.abs((all_directions - theta + 180) % 360 - 180)   # distance circulaire, entre 0 et 180°
            i = np.argmin(d)
            closest_pd[i] += 1
        pd_results[Linear] = (pd_of_neurons, closest_pd)

        # PCA of the network activity across time and targets (movement epoch)
        centred = rates.reshape(-1, N_NET) - rates.reshape(-1, N_NET).mean(0)
        _, sv, Vt = np.linalg.svd(centred, full_matrices=False)
        pcs = (rates - rates.reshape(-1, N_NET).mean(0)) @ Vt[:3].T
        var_expl = sv**2 / np.sum(sv**2)

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(2, 3, 1)
        for (X, Y, _, _), c, ang in zip(results, colors, angles):
            ax.plot(X, Y, color=c)
            ax.plot(*(CENTER + RADIUS * np.array([np.cos(ang), np.sin(ang)])), "s",
                    mfc="none", color=c, ms=10)
        ax.plot(*CENTER, "ko")
        ax.set_aspect("equal")
        ax.set_title("Hand paths")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")

        ax = fig.add_subplot(2, 3, 2)
        for (X, Y, _, _), c in zip(results, colors):
            ax.plot(time_, np.hypot(np.diff(X), np.diff(Y)) / (DURATION / K_STEPS) / 100,
                    color=c)
        ax.set_title("Hand speed")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("speed [m/s]")

        ax = fig.add_subplot(2, 3, 3)
        for k in range(N_DIR):
            ax.plot(time_, muscles[k], color=colors[k], lw=0.8)
        ax.set_title("Muscle commands a = Wout r")
        ax.set_xlabel("time [s]")

        for p, node in enumerate([0, 1]):
            ax = fig.add_subplot(2, 3, 4 + p)
            for k in range(N_DIR):
                ax.plot(time_, rates[k, :, node], color=colors[k])
            ax.set_title(f"Network node {node}")
            ax.set_xlabel("time [s]")

        ax = fig.add_subplot(2, 3, 6, projection="3d")
        for k in range(N_DIR):
            ax.plot(*pcs[k].T, color=colors[k])
            ax.scatter(*pcs[k, 0], color=colors[k], marker="o")
        ax.set_title("Network PCs (%.0f%% var. in 3 PCs)" % (100 * var_expl[:3].sum()))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        fig.tight_layout()
        plt.savefig("center_out_reaching_linear.png", dpi=300) if Linear else plt.savefig("center_out_reaching_nonlinear.png", dpi=300)
        plt.show()

    # Preferred-direction distributions: linear (left) vs nonlinear (right) arm, on a
    # common radial scale. Dashed line: main axis of the bimodal distribution (mean
    # resultant of the doubled angles, as in Lillicrap & Scott 2013).
    a = np.deg2rad(all_directions)
    rmax = 1.1 * max(counts.max() for _, counts in pd_results.values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), subplot_kw={"projection": "polar"})
    for ax, (lin, title, color) in zip(
        axes, [(True, "Linear arm", "#2a78d6"), (False, "Nonlinear arm", "#eb6834")]
    ):
        pds, counts = pd_results[lin]

        ax.bar(a, counts, width=np.deg2rad(22.5) * 0.88, color=color,
               edgecolor="white", linewidth=1, zorder=2)

        ax.set_ylim(0, rmax)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
        ax.tick_params(axis="y", labelsize=8, labelcolor="0.45")
        ax.set_thetagrids(range(0, 360, 45), [f"{d}°" for d in range(0, 360, 45)],
                          fontsize=9, color="0.35")
        ax.grid(color="0.85", lw=0.7)
        ax.spines["polar"].set_color("0.75")
        ax.set_title(title, fontsize=13, fontweight="semibold", pad=44)

    fig.suptitle(f"Preferred directions of the network nodes (N = {N_NET})", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), w_pad=4)
    plt.savefig("pd_distribution_linear_vs_nonlinear.png", dpi=300)
    plt.show()

    
