"""Cas simple : predire le signal EMG d'un muscle, controle musculaire direct
(pas de reseau), avec la dynamique d'activation asymetrique de Li & Todorov
(2004) -- montee tau_act=30ms dependante de l'excitation, descente
tau_deact=60ms constante -- au lieu du filtre symetrique de `ilqg_muscle.py`.

Reprend la mecanique du bras + 6 muscles de `ilqg_muscle.py` (moment arms,
force-longueur/force-vitesse) mais retire completement le reseau : le controle
u (6,) pilote directement l'excitation de chaque muscle.

Convention EMG (voir discussion) : l'EMG traite (redresse + filtre passe-bas)
correspond a l'EXCITATION e(t), pas a l'activation a(t) qui est deja en aval
(couplage excitation-contraction). Les deux sont tracees pour comparaison, et
c'est e(t) qui doit etre lue comme le "signal EMG predit".
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ilqg_muscle as MU

NM = 6
NX = 4 + NM  # th1, th2, om1, om2, a1..a6

# Li & Todorov (2004), "Iterative LQR design for nonlinear biological movement
# systems", eq. (30): a_dot = (u-a)/t(u,a), t = tau_deact + u*(tau_act-tau_deact)
# si u > a, sinon tau_deact. tau_act=30ms (montee), tau_deact=60ms (descente).
TAU_ACT = 0.030
TAU_DEACT = 0.060


def excitation(u):
    """Excitation e(t) in [0, inf) -- meme rectifieur que le reste du code
    (equivalent au signal EMG apres redressement+filtrage)."""
    return MU._rect(u)


def activation_tau(e, a):
    return np.where(e > a, TAU_DEACT + e * (TAU_ACT - TAU_DEACT), TAU_DEACT)


def f(x, u):
    th1, th2, om1, om2 = x[0], x[1], x[2], x[3]
    a = x[4:4 + NM]
    e = excitation(u)
    _, _, fl, fv = MU.muscle_flfv(th1, th2, om1, om2)
    tau = MU.A_mom @ (a * fl * fv)
    C = np.array([-om2 * (2 * om1 + om2) * MU.a2 * np.sin(th2),
                  om1 ** 2 * MU.a2 * np.sin(th2)])
    omdot = MU._Minv(th2) @ (tau - MU.Viscous @ np.array([om1, om2]) - C)
    adot = (e - a) / activation_tau(e, a)
    return np.concatenate([[om1, om2], omdot, adot])


def _jac_fd(fun, x, u, eps=1e-6):
    f0 = fun(x, u)
    Jx = np.zeros((len(f0), len(x)))
    Ju = np.zeros((len(f0), len(u)))
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        Jx[:, i] = (fun(xp, u) - f0) / eps
    for i in range(len(u)):
        up = u.copy(); up[i] += eps
        Ju[:, i] = (fun(x, up) - f0) / eps
    return Jx, Ju


def h(x, w1, w2, xtarg):
    return w1 / 2 * ((x[0] - xtarg[0]) ** 2 + (x[1] - xtarg[1]) ** 2) \
        + w2 / 2 * (x[2] ** 2 + x[3] ** 2)


def hx(x, w1, w2, xtarg):
    g = np.zeros(len(x))
    g[0] = w1 * (x[0] - xtarg[0]); g[1] = w1 * (x[1] - xtarg[1])
    g[2] = w2 * x[2]; g[3] = w2 * x[3]
    return g


def hxx(x, w1, w2):
    Q = np.zeros((len(x), len(x)))
    Q[0, 0] = w1; Q[1, 1] = w1; Q[2, 2] = w2; Q[3, 3] = w2
    return Q


def simulate(target, start, Duration=0.6, w1=1e4, w2=10.0, r1=1e-3, K=120,
             eps=1e-6, max_iter=200, tol=1e-6):
    th1t, th2t = MU.compute_angles_from_cartesian(*target)
    th1s, th2s = MU.compute_angles_from_cartesian(*start)
    xtarg = np.array([th1t, th2t])
    x0 = np.array([th1s, th2s, 0.0, 0.0] + [0.0] * NM)
    dt = Duration / K
    u = np.zeros((K, NM))

    def rollout(u):
        xs = np.zeros((K + 1, NX)); xs[0] = x0
        for k in range(K):
            xs[k + 1] = xs[k] + dt * f(xs[k], u[k])
        return xs

    def cost(xs, u):
        return h(xs[-1], w1, w2, xtarg) + dt * r1 * np.sum(u ** 2) / 2

    xs = rollout(u)
    J = cost(xs, u)
    alphas = 0.5 ** np.arange(10)
    for it in range(max_iter):
        A = np.zeros((K, NX, NX)); B = np.zeros((K, NX, NM))
        r = np.zeros((K, NM)); R = np.zeros((K, NM, NM))
        for k in range(K):
            Jx, Ju = _jac_fd(f, xs[k], u[k])
            A[k] = np.eye(NX) + dt * Jx
            B[k] = dt * Ju
            r[k] = dt * r1 * u[k]
            R[k] = dt * r1 * np.eye(NM)
        q_last = h(xs[-1], w1, w2, xtarg)
        qb_last = hx(xs[-1], w1, w2, xtarg)
        Q_last = hxx(xs[-1], w1, w2)

        S, s = Q_last.copy(), qb_last.copy()
        lg = np.zeros((K, NM)); Lg = np.zeros((K, NM, NX))
        for k in range(K - 1, -1, -1):
            SA = S @ A[k]
            G = B[k].T @ SA
            g_ = r[k] + B[k].T @ s
            Hh = R[k] + B[k].T @ S @ B[k]
            w_, V_ = np.linalg.eigh(0.5 * (Hh + Hh.T))
            w_ = np.where(w_ < eps, eps, w_)
            Hinv = (V_ * (1.0 / w_)) @ V_.T
            Snew = A[k].T @ SA - G.T @ Hinv @ G
            snew = A[k].T @ s - G.T @ Hinv @ g_
            lg[k] = -Hinv @ g_
            Lg[k] = -Hinv @ G
            S, s = Snew, snew

        improved = False
        xnew = unew = None
        for alpha in alphas:
            xnew = np.zeros_like(xs); unew = np.zeros_like(u); xnew[0] = x0
            for k in range(K):
                unew[k] = u[k] + alpha * lg[k] + Lg[k] @ (xnew[k] - xs[k])
                xnew[k + 1] = xnew[k] + dt * f(xnew[k], unew[k])
            Jnew = cost(xnew, unew)
            if np.isfinite(Jnew) and Jnew < J:
                improved = True
                break
        if not improved:
            break
        rel = (J - Jnew) / max(abs(J), 1e-12)
        xs, u, J = xnew, unew, Jnew
        if rel < tol:
            break
    return xs, u, xtarg


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xs, u, xtarg = simulate(target=[12, 40], start=[0, 40])
    K = u.shape[0]
    Duration = 0.6
    t_x = np.linspace(0, Duration, K + 1)
    t_u = np.linspace(0, Duration, K) + (Duration / K) / 2

    e = MU._rect(u)               # excitation = "EMG predit" (6, K)
    a = xs[:-1, 4:4 + NM]          # activation, alignee sur les memes instants
    th1, th2 = xs[:, 0], xs[:, 1]
    Xh = np.cos(th1 + th2) * MU.l2 * 100 + np.cos(th1) * MU.l1 * 100
    Yh = np.sin(th1 + th2) * MU.l2 * 100 + np.sin(th1) * MU.l1 * 100
    speed = np.hypot(np.diff(Xh), np.diff(Yh)) / (Duration / K) / 100.0

    err = np.hypot(Xh[-1] - 12, Yh[-1] - 40)
    print(f"endpoint error = {err:.4f} cm")

    names = ["epaule flechisseur", "epaule extenseur", "coude flechisseur",
             "coude extenseur", "bi-art. flechisseur", "bi-art. extenseur"]

    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    axes[0].plot(t_u, speed, color="k")
    axes[0].set_ylabel("vitesse main (m/s)")
    axes[0].set_title("Reach 0->[12,40] cm, controle musculaire direct")

    m = 0  # muscle affiche : epaule flechisseur
    axes[1].plot(t_u, e[:, m], label=f"excitation e(t)  [= EMG predit]", color="C0")
    axes[1].plot(t_u, a[:, m], label="activation a(t)", color="C1", linestyle="--")
    axes[1].set_ylabel(names[m])
    axes[1].legend(fontsize=8)

    m2 = 1  # antagoniste : epaule extenseur
    axes[2].plot(t_u, e[:, m2], label="excitation e(t)  [= EMG predit]", color="C0")
    axes[2].plot(t_u, a[:, m2], label="activation a(t)", color="C1", linestyle="--")
    axes[2].set_ylabel(names[m2])
    axes[2].set_xlabel("temps (s)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(_HERE, "fig_emg_prediction_simple.png")
    plt.savefig(out, dpi=150)
    print(f"figure sauvegardee: {out}")
