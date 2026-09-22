"""Combined random-network + musculoskeletal arm, controlled by iLQG.

This is `iLQG_Combined.py` with the joint-torque actuator replaced by the
six lumped muscles of Lillicrap & Scott (2013, Neuron 77:168-179; see their
Supplemental Experimental Procedures, "Musculoskeletal model specification").

Combined state, `x = [r ; s ; a]`:

    r (N)      network firing rates, linear dynamics  rdot = W r + u
    s (4)      arm state  [theta_s, theta_e, omega_s, omega_e]
               (or the Cartesian hand state [px, py, vx, vy] for the
               2-D point-mass abstraction)
    a (n_act)  actuator state -- muscle activations (4 or 6) or joint
               torques (2), driven by the network readout through
               first-order dynamics  adot = (sigma_u(Wout r) - a) / tau_act

The muscle path follows the paper exactly:

    l  = 1 + sum_j M[j,i] (theta0[j,i] - theta_j) / L0[i]        (their Eq 2)
    f  = F_max * a * f_l(l) * f_fv(l, ldot)                      (their Eq 6/7)
    tau = M * f                                                  (their Eq 1)

The force-length and force-velocity curves are their Eq 6 and Eq 7, evaluated
directly.  The paper additionally replaced the product by a fitted 5-hidden-node
sigmoidal network to remove the derivative kink at zero velocity; that step is
deliberately not taken here, so the plant sees the equations themselves.

Unlike the paper we do not train a recurrent network by backpropagation
through time: the network is fixed and random, and the *input* to it is found
by iLQG, i.e. the same optimal-control formulation used elsewhere in WP3.

`Plant` bundles the abstraction switches used for the Figure 5 ladder
(point-mass -> geometry -> intersegmental -> mono -> biarticular -> F-L/V), so
several models can be solved in one process without global-state juggling.
"""

import numpy as np
import h5py

# -----------------------------------------------------------------------------
# Arm (kept identical to iLQG_Combined.py: a human-scale two-link planar arm)
# -----------------------------------------------------------------------------
I1, I2 = 0.025, 0.045          # segment moments of inertia [kg m^2]
m1, m2 = 1.4, 1.0              # segment masses [kg]
l1_m, l2_m = 0.3, 0.33         # segment lengths [m]
s1, s2 = 0.11, 0.16            # proximal joint -> centre of mass [m]
L1, L2 = 30.0, 33.0            # segment lengths [cm] (hand-space bookkeeping)

a1 = I1 + I2 + m2 * l1_m * l1_m
a2 = m2 * l1_m * s2
a3 = I2

VISCOUS = np.array([[0.05, 0.025], [0.025, 0.05]])

TAU_ACT = 0.06                 # actuator / activation time constant [s]
M_POINT = 1.0                  # point-mass abstraction: mass [kg]

# Centre of the workspace, chosen so the joint angles match the paper's centre
# target (shoulder 32.6 deg, elbow 84.2 deg; their Figure 2B).
CENTER_JOINTS = np.deg2rad([32.6, 84.2])

# -----------------------------------------------------------------------------
# Six lumped muscles (Lillicrap & Scott 2013, Supplemental Eq 1, 4, 5)
# columns: [shoulder flexor, shoulder extensor, elbow flexor, elbow extensor,
#           biarticular flexor, biarticular extensor]
# -----------------------------------------------------------------------------
MOMENT_ARM = np.array([[2.0, -2.0, 0.0, 0.0, 1.5, -2.0],
                       [0.0, 0.0, 2.0, -2.0, 2.0, -1.5]])          # [cm]
L0 = np.array([7.32, 3.26, 6.4, 4.26, 5.95, 4.04])                  # [cm]
THETA0 = np.deg2rad([[15.0, 4.88, 0.0, 0.0, 4.5, 2.12],
                     [0.0, 0.0, 80.86, 109.32, 92.96, 91.52]])      # [rad]

F_MAX = 800.0                  # peak isometric force of a lumped muscle [N]

# Brown et al. (1999) force-length / force-velocity parameters, as listed in
# the paper's Supplemental Information.
FL_BETA, FL_OMEGA, FL_RHO = 1.55, 0.81, 2.12
FV_VMAX, FV_CV0, FV_CV1 = -7.39, -3.21, 4.17
FV_BV, FV_AV0, FV_AV1, FV_AV2 = 0.62, -3.12, 4.21, -2.67
FV_EPS = 0.0                   # 0 = the paper's Eq 7 verbatim (kinked derivative
                               # at zero velocity); > 0 cross-fades the two
                               # branches over |ldot| < FV_EPS




def force_length(l):
    """f_l(l) = exp(-|(l^beta - 1)/omega|^rho)  (Brown et al. 1999)."""
    z = (l ** FL_BETA - 1.0) / FL_OMEGA
    return np.exp(-np.abs(z) ** FL_RHO)


def d_force_length(l):
    z = (l ** FL_BETA - 1.0) / FL_OMEGA
    az = np.abs(z)
    # d/dl exp(-|z|^rho) = -rho |z|^(rho-1) sign(z) dz/dl * f_l
    dzdl = FL_BETA * l ** (FL_BETA - 1.0) / FL_OMEGA
    return -FL_RHO * az ** (FL_RHO - 1.0) * np.sign(z) * dzdl * force_length(l)


def force_velocity(l, v):
    """The paper's Eq 7 (Brown et al. force-velocity) and its partial derivatives.

              (Vmax - ldot) / (Vmax + (cV0 + cV1 l) ldot)          ldot <= 0
    f_fv  =
              (bV - (aV0 + aV1 l + aV2 l^2) ldot) / (bV + ldot)    ldot >  0

    Each branch is evaluated only on its own side of zero, so the lengthening
    branch never approaches its pole at ldot = -b_V (which lies inside the
    velocity range our reaches visit).  The two branches meet at f = 1 but with
    different slopes, i.e. df/dldot jumps at ldot = 0 -- this is the kink the
    paper removed by fitting a sigmoidal network to the combined curve.  Setting
    FV_EPS > 0 cross-fades the branches over |ldot| < FV_EPS with a quintic
    smoothstep instead; FV_EPS = 0 leaves the paper's equations untouched.

    Returns (f, df/dl, df/dv).
    """
    eps = FV_EPS
    vs = np.minimum(v, eps)                       # branch used for v <= +eps
    c = FV_CV0 + FV_CV1 * l
    den_s = FV_VMAX + c * vs
    num_s = FV_VMAX - vs
    f_s = num_s / den_s
    active_s = v < eps if eps > 0 else v <= 0
    dfs_dv = np.where(active_s, (-den_s - num_s * c) / den_s ** 2, 0.0)
    dfs_dl = np.where(active_s, -num_s * FV_CV1 * vs / den_s ** 2, 0.0)

    vl = np.maximum(v, -eps)                      # branch used for v >= -eps
    d = FV_AV0 + FV_AV1 * l + FV_AV2 * l * l
    den_l = FV_BV + vl
    num_l = FV_BV - d * vl
    f_lng = num_l / den_l
    active_l = v > -eps
    dfl_dv = np.where(active_l, (-d * den_l - num_l) / den_l ** 2, 0.0)
    dfl_dl = np.where(active_l, -(FV_AV1 + 2.0 * FV_AV2 * l) * vl / den_l, 0.0)

    if eps > 0:
        t = np.clip((v + eps) / (2.0 * eps), 0.0, 1.0)
        sw = t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t)      # 0 -> 1 smoothstep
        dsw = 30.0 * t * t * (1.0 - t) ** 2 / (2.0 * eps)
    else:
        sw = np.where(v > 0, 1.0, 0.0)
        dsw = np.zeros_like(sw)

    f = (1.0 - sw) * f_s + sw * f_lng
    df_dv = (1.0 - sw) * dfs_dv + sw * dfl_dv + dsw * (f_lng - f_s)
    df_dl = (1.0 - sw) * dfs_dl + sw * dfl_dl
    return f, df_dl, df_dv


def flv(l, v):
    """Force-length-velocity scaling, f_l(l) * f_fv(l, ldot), and its partials.

    This is the paper's Eq 6 x Eq 7 evaluated directly.
    """
    fl = force_length(l)
    fv, dfv_dl, dfv_dv = force_velocity(l, v)
    return fl * fv, d_force_length(l) * fv + fl * dfv_dl, fl * dfv_dv


# -----------------------------------------------------------------------------
# Hand kinematics (cm)
# -----------------------------------------------------------------------------
def hand_position(theta):
    th1, th2 = theta[..., 0], theta[..., 1]
    return np.stack([L1 * np.cos(th1) + L2 * np.cos(th1 + th2),
                     L1 * np.sin(th1) + L2 * np.sin(th1 + th2)], axis=-1)


def hand_jacobian(theta):
    """dp/dtheta, shape (2, 2), in cm/rad."""
    th1, th2 = theta[0], theta[1]
    s12, c12 = np.sin(th1 + th2), np.cos(th1 + th2)
    return np.array([[-L1 * np.sin(th1) - L2 * s12, -L2 * s12],
                     [L1 * np.cos(th1) + L2 * c12, L2 * c12]])


def d_hand_jacobian(theta):
    """[dJ/dtheta1, dJ/dtheta2], each (2, 2)."""
    th1, th2 = theta[0], theta[1]
    s12, c12 = np.sin(th1 + th2), np.cos(th1 + th2)
    dJ1 = np.array([[-L1 * np.cos(th1) - L2 * c12, -L2 * c12],
                    [-L1 * np.sin(th1) - L2 * s12, -L2 * s12]])
    dJ2 = np.array([[-L2 * c12, -L2 * c12],
                    [-L2 * s12, -L2 * s12]])
    return dJ1, dJ2


def inverse_kinematics(x, y):
    """Joint angles for a hand position given in cm."""
    r2 = x * x + y * y
    th1 = np.arctan2(y, x) - np.arccos((r2 + L1 ** 2 - L2 ** 2)
                                       / (2 * L1 * np.sqrt(r2)))
    th2 = np.pi - np.arccos((L1 ** 2 + L2 ** 2 - r2) / (2 * L1 * L2))
    return th1, th2


CENTER_XY = tuple(hand_position(CENTER_JOINTS))


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------
def load_network(index=2, spectral_radius=0.8, path="gaussian_networks.hdf5"):
    """Recurrent transition matrix W (= the paper's stable J) of network `index`."""
    with h5py.File(path, "r") as f:
        return f[f"{spectral_radius}/networks/network_{index}"][()]


def random_readout(n_act, N, rng, std=None):
    """Fixed random unit->actuator weights, as in Lillicrap & Scott."""
    std = 1.0 / np.sqrt(N) if std is None else std
    return rng.normal(0.0, std, size=(n_act, N))


# -----------------------------------------------------------------------------
# Plant
# -----------------------------------------------------------------------------
class Plant:
    """Combined network + periphery, with the Figure 5 abstraction switches.

    Parameters
        W          : (N, N) recurrent transition matrix
        Wout       : (n_act, N) fixed random readout onto the actuators
        geometry   : if False the limb is a 2-D point mass in hand space
        intersegmental : if False the inertia matrix is diagonalised and the
                     Coriolis/centripetal terms are dropped
        muscles    : "none" (joint-torque actuators), "mono" (4 monoarticular
                     muscles) or "bi" (all 6 muscles)
        flv        : include the force-length/velocity scaling
    """

    def __init__(self, W, Wout, geometry=True, intersegmental=True,
                 muscles="bi", flv=True, recenter_lengths=False):
        self.W = W
        self.N = W.shape[0]
        self.geometry = geometry
        self.intersegmental = intersegmental
        self.muscles = muscles
        self.flv = flv

        if muscles == "none":
            self.n_act = 2
            self.moment = None
        elif muscles == "mono":
            self.n_act = 4
            self.moment = MOMENT_ARM[:, :4]
        elif muscles == "bi":
            self.n_act = 6
            self.moment = MOMENT_ARM
        else:
            raise ValueError(muscles)

        if Wout.shape != (self.n_act, self.N):
            raise ValueError(f"Wout must be ({self.n_act}, {self.N})")
        self.Wout = Wout

        if self.moment is not None:
            L0m = L0[:self.n_act]
            self.A_len = self.moment / L0m                    # (2, n_act)
            self.l_const = 1.0 + np.sum(self.moment * THETA0[:, :self.n_act],
                                        axis=0) / L0m         # (n_act,)
            if recenter_lengths:
                # shift the optimal angles so every muscle sits at l = 1 at the
                # centre posture (our arm segments are longer than the monkey's)
                self.l_const = 1.0 + self.A_len.T @ CENTER_JOINTS
            self.moment_m = self.moment / 100.0               # cm -> m
        self.n = self.N + 4 + self.n_act

    # -- indices -------------------------------------------------------------
    @property
    def i_s(self):
        return slice(self.N, self.N + 4)

    @property
    def i_a(self):
        return slice(self.N + 4, self.n)

    # -- muscle path ---------------------------------------------------------
    def muscle_state(self, theta, omega):
        """Normalised muscle length and velocity (units of L0 and L0/s)."""
        l = self.l_const - self.A_len.T @ theta
        v = -self.A_len.T @ omega
        return l, v

    def torque(self, s, a):
        """Joint torque and, optionally, its partials wrt theta, omega, a."""
        if self.muscles == "none":
            return a, None
        theta, omega = s[:2], s[2:]
        l, v = self.muscle_state(theta, omega)
        if self.flv == "length":            # force-length only (diagnostic)
            phi, dphi_dl = force_length(l), d_force_length(l)
            dphi_dv = np.zeros(self.n_act)
        elif self.flv == "velocity":        # force-velocity only (diagnostic)
            phi, dphi_dl, dphi_dv = force_velocity(l, v)
        elif self.flv:
            phi, dphi_dl, dphi_dv = flv(l, v)
        else:
            phi = np.ones(self.n_act)
            dphi_dl = dphi_dv = np.zeros(self.n_act)
        tension = F_MAX * a * phi
        tau = self.moment_m @ tension
        # d(tension)/d(theta_j) = F_max a dphi/dl dl/dtheta_j, dl/dtheta = -A_len
        dtau_dtheta = self.moment_m @ (
            (F_MAX * a * dphi_dl)[:, None] * (-self.A_len.T))     # (2, 2)
        dtau_domega = self.moment_m @ (
            (F_MAX * a * dphi_dv)[:, None] * (-self.A_len.T))     # (2, 2)
        dtau_da = self.moment_m * (F_MAX * phi)                   # (2, n_act)
        return tau, (dtau_dtheta, dtau_domega, dtau_da)

    # -- limb ----------------------------------------------------------------
    def inertia(self, th2):
        M = np.array([[a1 + 2 * a2 * np.cos(th2), a3 + a2 * np.cos(th2)],
                      [a3 + a2 * np.cos(th2), a3]])
        dM = np.array([[-2 * a2 * np.sin(th2), -a2 * np.sin(th2)],
                       [-a2 * np.sin(th2), 0.0]])
        if not self.intersegmental:
            M = np.diag(np.diag(M))
            dM = np.diag(np.diag(dM))
        return M, dM

    def viscous(self):
        return VISCOUS if self.intersegmental else np.diag(np.diag(VISCOUS))

    # -- dynamics ------------------------------------------------------------
    def f(self, x, u):
        r, s, a = x[:self.N], x[self.i_s], x[self.i_a]
        rdot = self.W @ r + u

        drive = self.Wout @ r
        if self.muscles != "none":
            drive = sigma_u(drive)
        adot = (drive - a) / TAU_ACT

        if not self.geometry:
            # 2-D point mass in hand space: a holds the two Cartesian forces
            sdot = np.concatenate([s[2:], 100.0 * a / M_POINT])
            return np.concatenate([rdot, sdot, adot])

        theta, omega = s[:2], s[2:]
        tau, _ = self.torque(s, a)
        M, _ = self.inertia(theta[1])
        C = np.array([-omega[1] * (2 * omega[0] + omega[1]) * a2 * np.sin(theta[1]),
                      omega[0] ** 2 * a2 * np.sin(theta[1])])
        if not self.intersegmental:
            C = np.zeros(2)
        acc = np.linalg.solve(M, tau - C - self.viscous() @ omega)
        return np.concatenate([rdot, omega, acc, adot])

    def fx(self, x, u):
        """Analytic Jacobian df/dx."""
        N, n = self.N, self.n
        r, s, a = x[:N], x[self.i_s], x[self.i_a]
        A = np.zeros((n, n))
        A[:N, :N] = self.W

        # actuator block
        drive = self.Wout @ r
        gain = dsigma_u(drive) if self.muscles != "none" else np.ones(self.n_act)
        A[self.i_a, :N] = (gain[:, None] * self.Wout) / TAU_ACT
        A[self.i_a, self.i_a] = -np.eye(self.n_act) / TAU_ACT

        i0 = N
        if not self.geometry:
            A[i0 + 0, i0 + 2] = 1.0
            A[i0 + 1, i0 + 3] = 1.0
            A[i0 + 2, self.i_a] = np.array([100.0 / M_POINT, 0.0])
            A[i0 + 3, self.i_a] = np.array([0.0, 100.0 / M_POINT])
            return A

        theta, omega = s[:2], s[2:]
        A[i0 + 0, i0 + 2] = 1.0
        A[i0 + 1, i0 + 3] = 1.0

        tau, dtau = self.torque(s, a)
        M, dM = self.inertia(theta[1])
        Minv = np.linalg.inv(M)
        B = self.viscous()
        if self.intersegmental:
            st2 = np.sin(theta[1])
            ct2 = np.cos(theta[1])
            C = np.array([-omega[1] * (2 * omega[0] + omega[1]) * a2 * st2,
                          omega[0] ** 2 * a2 * st2])
            dC_dth2 = np.array([-omega[1] * (2 * omega[0] + omega[1]) * a2 * ct2,
                                omega[0] ** 2 * a2 * ct2])
            dC_domega = np.array([[-2 * omega[1] * a2 * st2,
                                   -(2 * omega[0] + 2 * omega[1]) * a2 * st2],
                                  [2 * omega[0] * a2 * st2, 0.0]])
        else:
            C = np.zeros(2)
            dC_dth2 = np.zeros(2)
            dC_domega = np.zeros((2, 2))

        net = tau - C - B @ omega                     # M acc = net
        acc = Minv @ net

        if dtau is None:                              # torque actuators
            dtau_dtheta = np.zeros((2, 2))
            dtau_domega = np.zeros((2, 2))
            dtau_da = np.eye(2)
        else:
            dtau_dtheta, dtau_domega, dtau_da = dtau

        dacc_dth1 = Minv @ dtau_dtheta[:, 0]
        dacc_dth2 = Minv @ (dtau_dtheta[:, 1] - dC_dth2) - Minv @ (dM @ acc)
        dacc_domega = Minv @ (dtau_domega - dC_domega - B)
        dacc_da = Minv @ dtau_da

        A[i0 + 2:i0 + 4, i0 + 0] = dacc_dth1
        A[i0 + 2:i0 + 4, i0 + 1] = dacc_dth2
        A[i0 + 2:i0 + 4, i0 + 2:i0 + 4] = dacc_domega
        A[i0 + 2:i0 + 4, self.i_a] = dacc_da
        return A

    def fu(self):
        B = np.zeros((self.n, self.N))
        B[:self.N, :self.N] = np.eye(self.N)
        return B

    # -- initial state -------------------------------------------------------
    def initial_state(self, start_xy, r0=None):
        r0 = np.zeros(self.N) if r0 is None else r0
        if self.geometry:
            th1, th2 = inverse_kinematics(*start_xy)
            s0 = np.array([th1, th2, 0.0, 0.0])
        else:
            s0 = np.array([start_xy[0], start_xy[1], 0.0, 0.0])
        a0 = (np.full(self.n_act, float(sigma_u(0.0)))
              if self.muscles != "none" else np.zeros(self.n_act))
        return np.concatenate([r0, s0, a0])

    # -- hand state ----------------------------------------------------------
    def hand(self, x):
        """Cartesian hand position (cm) for a state or a trajectory of states."""
        s = np.asarray(x)[..., self.i_s]
        if not self.geometry:
            return s[..., :2]
        return hand_position(s[..., :2])

    def hand_velocity(self, x):
        s = np.asarray(x)[..., self.i_s]
        if not self.geometry:
            return s[..., 2:]
        th = np.atleast_2d(s[..., :2])
        om = np.atleast_2d(s[..., 2:])
        v = np.stack([hand_jacobian(t) @ w for t, w in zip(th, om)])
        return v.reshape(np.shape(s)[:-1] + (2,))

    # -- costs ---------------------------------------------------------------
    def terminal_residual(self, x, ptarg, w1, w2):
        """Residual R and its Jacobian wrt x, so that h = 0.5 ||R||^2."""
        s = x[self.i_s]
        n = self.n
        J = np.zeros((4, n))
        if not self.geometry:
            R = np.concatenate([np.sqrt(w1) * (s[:2] - ptarg),
                                np.sqrt(w2) * s[2:]])
            J[0:2, self.N:self.N + 2] = np.sqrt(w1) * np.eye(2)
            J[2:4, self.N + 2:self.N + 4] = np.sqrt(w2) * np.eye(2)
            return R, J
        theta, omega = s[:2], s[2:]
        Jp = hand_jacobian(theta)
        p = hand_position(theta)
        v = Jp @ omega
        dJ1, dJ2 = d_hand_jacobian(theta)
        dv_dtheta = np.stack([dJ1 @ omega, dJ2 @ omega], axis=1)
        R = np.concatenate([np.sqrt(w1) * (p - ptarg), np.sqrt(w2) * v])
        J[0:2, self.N:self.N + 2] = np.sqrt(w1) * Jp
        J[2:4, self.N:self.N + 2] = np.sqrt(w2) * dv_dtheta
        J[2:4, self.N + 2:self.N + 4] = np.sqrt(w2) * Jp
        return R, J

    def h(self, x, ptarg, w1, w2):
        R, _ = self.terminal_residual(x, ptarg, w1, w2)
        return 0.5 * R @ R

    def hx(self, x, ptarg, w1, w2):
        R, J = self.terminal_residual(x, ptarg, w1, w2)
        return J.T @ R

    def hxx(self, x, ptarg, w1, w2):
        _, J = self.terminal_residual(x, ptarg, w1, w2)
        return J.T @ J                         # Gauss-Newton (PSD)

    def _path_terms(self, x, path):
        """Lateral deviation from the start->target line, and its gradient.

        `path` is (weight, start_xy, unit_normal).  The paper adds exactly this
        term to the running cost of a reach so that handpaths come out roughly
        straight without prescribing a reference trajectory.
        """
        qp, p0, nrm = path
        s = x[self.i_s]
        if not self.geometry:
            p, Jp = s[:2], np.eye(2)
        else:
            p, Jp = hand_position(s[:2]), hand_jacobian(s[:2])
        return qp, float(nrm @ (p - p0)), Jp.T @ nrm

    def l(self, x, u, r1, qr, qa, path=None):
        r, a = x[:self.N], x[self.i_a]
        J = 0.5 * (r1 * u @ u + qr * r @ r + qa * a @ a)
        if path is not None:
            qp, e, _ = self._path_terms(x, path)
            J += 0.5 * qp * e * e
        return J

    def lx(self, x, qr, qa, path=None):
        g = np.zeros(self.n)
        g[:self.N] = qr * x[:self.N]
        g[self.i_a] = qa * x[self.i_a]
        if path is not None:
            qp, e, w = self._path_terms(x, path)
            g[self.N:self.N + 2] += qp * e * w
        return g

    def lxx(self, x, qr, qa, path=None):
        d = np.zeros(self.n)
        d[:self.N] = qr
        d[self.i_a] = qa
        Q = np.diag(d)
        if path is not None:
            qp, _, w = self._path_terms(x, path)
            Q[self.N:self.N + 2, self.N:self.N + 2] += qp * np.outer(w, w)
        return Q


# -----------------------------------------------------------------------------
# iLQG
# -----------------------------------------------------------------------------
def rollout(plant, x0, u, dt):
    K = u.shape[0]
    x = np.zeros((K + 1, len(x0)))
    x[0] = x0
    for k in range(K):
        x[k + 1] = x[k] + dt * plant.f(x[k], u[k])
    return x


def total_cost(plant, x, u, dt, ptarg, w1, w2, r1, qr, qa, path=None):
    J = plant.h(x[-1], ptarg, w1, w2)
    for k in range(u.shape[0]):
        J += dt * plant.l(x[k], u[k], r1, qr, qa, path)
    return J


def backward_pass(plant, x, u, dt, ptarg, w1, w2, r1, qr, qa, path=None,
                  eps=1e-14):
    K, m = u.shape
    n = plant.n
    Bblk = dt * np.eye(m)                       # control enters the first m rows
    R = dt * r1 * np.eye(m)

    S = plant.hxx(x[-1], ptarg, w1, w2)
    sv = plant.hx(x[-1], ptarg, w1, w2)
    lgain = np.zeros((K, m))
    Lgain = np.zeros((K, m, n))

    for k in range(K - 1, -1, -1):
        A = np.eye(n) + dt * plant.fx(x[k], u[k])
        Qxx = dt * plant.lxx(x[k], qr, qa, path)
        SA = S @ A
        G = Bblk.T @ SA[:m, :]
        g = dt * r1 * u[k] + Bblk.T @ sv[:m]
        H = R + Bblk.T @ S[:m, :m] @ Bblk

        # H is symmetric PSD (R is positive definite), so only guard against
        # round-off.  The floor must stay well below the true smallest
        # eigenvalue dt*r1, otherwise the control-effort directions are
        # over-regularised and iLQG creeps.
        w_eig, V = np.linalg.eigh(0.5 * (H + H.T))
        floor = max(eps, 1e-12 * w_eig[-1])
        w_eig = np.where(w_eig < floor, floor, w_eig)
        Hinv = (V * (1.0 / w_eig)) @ V.T

        HinvG, Hinvg = Hinv @ G, Hinv @ g
        Snew = Qxx + A.T @ SA - G.T @ HinvG
        sv = dt * plant.lx(x[k], qr, qa, path) + A.T @ sv - G.T @ Hinvg
        S = 0.5 * (Snew + Snew.T)

        lgain[k] = -Hinvg
        Lgain[k] = -HinvG
    return lgain, Lgain


def forward_pass(plant, x0, x_nom, u_nom, lgain, Lgain, dt, alpha):
    K = u_nom.shape[0]
    xn = np.zeros_like(x_nom)
    un = np.zeros_like(u_nom)
    xn[0] = x0
    for k in range(K):
        un[k] = u_nom[k] + alpha * lgain[k] + Lgain[k] @ (xn[k] - x_nom[k])
        xn[k + 1] = xn[k] + dt * plant.f(xn[k], un[k])
    return xn, un


def solve(plant, start_xy, target_xy, duration=0.5, K=100,
          w1=1e4, w2=10.0, r1=1e-5, qr=0.0, qa=0.0, qp=0.0,
          eps=1e-14, max_iter=100, tol=1e-6, r0=None, verbose=False):
    """iLQG solution of one reach.  Returns (x, u, info).

    `qp` weights the paper's straight-handpath penalty (lateral deviation from
    the start->target line, integrated over the movement); qp = 0 leaves the
    path free.
    """
    dt = duration / K
    x0 = plant.initial_state(start_xy, r0)
    ptarg = np.asarray(target_xy, dtype=float)
    p0 = np.asarray(start_xy, dtype=float)

    path = None
    if qp > 0:
        v = ptarg - p0
        nrm = np.array([-v[1], v[0]]) / np.linalg.norm(v)
        path = (qp, p0, nrm)

    u = np.zeros((K, plant.N))
    x = rollout(plant, x0, u, dt)
    J = total_cost(plant, x, u, dt, ptarg, w1, w2, r1, qr, qa, path)

    alphas = 0.5 ** np.arange(12)
    for it in range(max_iter):
        lg, Lg = backward_pass(plant, x, u, dt, ptarg, w1, w2, r1, qr, qa,
                               path, eps)
        improved = False
        for alpha in alphas:
            xn, un = forward_pass(plant, x0, x, u, lg, Lg, dt, alpha)
            Jn = total_cost(plant, xn, un, dt, ptarg, w1, w2, r1, qr, qa, path)
            if np.isfinite(Jn) and Jn < J:
                improved = True
                break
        if not improved:
            break
        rel = (J - Jn) / max(abs(J), 1e-12)
        x, u, J = xn, un, Jn
        if verbose:
            print(f"    iter {it:3d}  J = {J:.6g}  (alpha {alpha:.4f})")
        if rel < tol:
            break

    p = plant.hand(x)
    info = {"cost": J, "iters": it + 1,
            "endpoint_error": float(np.hypot(*(p[-1] - ptarg)))}
    return x, u, info


# -----------------------------------------------------------------------------
# Figure 5 abstraction ladder
# -----------------------------------------------------------------------------
LADDER = [
    ("2D point mass", dict(geometry=False, intersegmental=False,
                           muscles="none", flv=False)),
    ("+ geometry", dict(geometry=True, intersegmental=False,
                        muscles="none", flv=False)),
    ("+ intersegmental\ndynamics", dict(geometry=True, intersegmental=True,
                                        muscles="none", flv=False)),
    ("+ monoarticular\nmuscles", dict(geometry=True, intersegmental=True,
                                      muscles="mono", flv=False)),
    ("+ biarticular\nmuscles", dict(geometry=True, intersegmental=True,
                                    muscles="bi", flv=False)),
    ("+ muscle\nmechanics", dict(geometry=True, intersegmental=True,
                                 muscles="bi", flv=True)),
]


def n_actuators(cfg):
    return {"none": 2, "mono": 4, "bi": 6}[cfg["muscles"]]


# -----------------------------------------------------------------------------
# Self-test: analytic Jacobian vs finite differences
# -----------------------------------------------------------------------------
def _check_jacobian(seed=0):
    rng = np.random.default_rng(seed)
    W = load_network(2)
    N = W.shape[0]
    worst = 0.0
    for name, cfg in LADDER:
        plant = Plant(W, random_readout(n_actuators(cfg), N, rng), **cfg)
        x = plant.initial_state(CENTER_XY)
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
        err = np.abs(A - Afd).max() / max(1.0, np.abs(Afd).max())
        worst = max(worst, err)
        print(f"  {name.replace(chr(10), ' '):32s} rel Jacobian error {err:.2e}")
    return worst


if __name__ == "__main__":
    lg = np.linspace(0.5, 1.55, 60)
    vg = np.linspace(-1.0, 0.9, 60)
    L, V = np.meshgrid(lg, vg, indexing="ij")
    f = flv(L, V)[0]
    print(f"Force-length-velocity (Eq 6 x Eq 7, FV_EPS = {FV_EPS:g}) over the "
          f"range our reaches visit:")
    print(f"  f_l  range [{force_length(lg).min():.3f}, "
          f"{force_length(lg).max():.3f}]")
    print(f"  f_flv range [{f.min():.3f}, {f.max():.3f}]")

    print("\nFinite-difference check of the analytic Jacobians:")
    err = _check_jacobian()
    print(f"worst relative error: {err:.2e}")
