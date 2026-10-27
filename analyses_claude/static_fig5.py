"""Lillicrap & Scott (2013) Figure 5, centre-out rows, with THEIR static model.

The static model (main text Fig. 4A, Eq. 4; Supplemental "Static model plant
approximation") optimises one time step at movement initiation: for 16 target
initial hand velocities y* (unit circle) it finds the activity z of 1000 units

    z* = argmin  1/2 ||G F M (H o sigma_u(Wout z)) - y*||^2 + a/2 ||u||^2 + b/2 ||z||^2

with a = b = 1e-5 and Wout ~ N(0, 0.001) fixed (10 networks). Matrices:

  F  torque -> joint velocity, linearised and averaged over the first half of
     the movement from rest: the derivative of the arm dynamics with respect to
     torque is M_inertia^-1, so F = (dt/2) M_inertia(theta_start)^-1, dt = T/2.
     Viscosity and Coriolis act through velocity only and vanish here.
  G  hand Jacobian at the start posture.
  M  moment arms (2 x 6, printed values, cm numbers as in the Supplemental).
  H  F-L/V factor per muscle and target, averaged from movement onset (10 % of
     peak speed) to 100 ms after peak speed. L&S took it from their dynamic
     model; here from straight, minimum-jerk 6 cm / 0.4 s reaches (it only
     depends on the kinematics), with the Brown F-L/V equations.

Rungs (Supplemental): 1 point mass G = I, F isotropic, torque actuators
(M = I, no sigma_u); 2 + geometry, F with its off-diagonal terms zeroed;
3 + intersegmental dynamics, full F; 4 + monoarticular muscles (first four
columns of M, sigma_u); 5 + biarticular muscles; 6 + muscle mechanics (H).
Unit PMDs: planar regression of z on target direction, units with p < 0.05.

Arm parameters: macaque.py's sets (L&S printed values, L&S + Cheng & Scott
inertias, all Cheng & Scott).

    python analyses_claude/static_fig5.py [--params ls_cs] [--speed 1.0]
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import f as f_dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import macaque  # noqa: E402  (parameter sets; its patches are not used here)
import fig5_overlay as F5  # noqa: E402  (plotting helpers and the paper's values)

M_ = macaque.C.M                     # iLQG_Muscle: moment arms, L0, theta0, F-L/V
ALPHA = BETA = 1e-5
N_UNITS, N_NETS, N_T = 1000, 10, 16
T_MOVE, AMP = 0.4, 6.0               # s, cm (L&S task)
THETA_START = np.radians([32.6, 84.2])
ANG = np.linspace(0, 2 * np.pi, N_T, endpoint=False)
RUNGS = [("pm", "1  2D point mass"), ("geo", "2  + geometry"),
         ("nl2", "3  + intersegmental\ndynamics"), ("mono", "4  + monoarticular\nmuscles"),
         ("nl6x", "5  + biarticular\nmuscles"), ("nl6", "6  + muscle\nmechanics")]


def sigma_u(x):
    return np.where(x < 0.5, np.exp(2 * np.minimum(x, 0.5) - 5), 2 * x * np.exp(-4.0))


def dsigma_u(x):
    return np.where(x < 0.5, 2 * np.exp(2 * np.minimum(x, 0.5) - 5), 2 * np.exp(-4.0))


# -----------------------------------------------------------------------------
# plant matrices
# -----------------------------------------------------------------------------
def arm(params):
    I1, I2, m1, m2, l1, l2, s1, s2 = macaque._SETS[params]
    th1, th2 = THETA_START
    a1, a2, a3 = I1 + I2 + m2 * l1 ** 2, m2 * l1 * s2, I2
    Mi = np.array([[a1 + 2 * a2 * np.cos(th2), a3 + a2 * np.cos(th2)],
                   [a3 + a2 * np.cos(th2), a3]])
    G = np.array([[-l1 * np.sin(th1) - l2 * np.sin(th1 + th2), -l2 * np.sin(th1 + th2)],
                  [l1 * np.cos(th1) + l2 * np.cos(th1 + th2), l2 * np.cos(th1 + th2)]])
    dt = T_MOVE / 2
    F = (dt / 2) * np.linalg.inv(Mi)
    return F, G, (l1, l2)


def ik(x, y, l1, l2):
    r2 = x ** 2 + y ** 2
    th2 = np.pi - np.arccos((l1 ** 2 + l2 ** 2 - r2) / (2 * l1 * l2))
    th1 = np.arctan2(y, x) - np.arccos((r2 + l1 ** 2 - l2 ** 2) / (2 * l1 * np.sqrt(r2)))
    return np.stack([th1, th2], -1)


def flv_factors(lengths):
    """H (6 muscles x 16 targets): F-L/V averaged from onset to 100 ms after peak
    speed along straight minimum-jerk reaches."""
    l1, l2 = lengths
    th0 = THETA_START
    start = np.array([l1 * np.cos(th0[0]) + l2 * np.cos(th0.sum()),
                      l1 * np.sin(th0[0]) + l2 * np.sin(th0.sum())])
    t = np.linspace(0, T_MOVE, 401)
    s = 10 * (t / T_MOVE) ** 3 - 15 * (t / T_MOVE) ** 4 + 6 * (t / T_MOVE) ** 5
    A_len = M_.MOMENT_ARM / M_.L0                                      # (2, 6)
    l_const = 1 + np.sum(M_.MOMENT_ARM * M_.THETA0, 0) / M_.L0
    H = np.zeros((6, N_T))
    for k, a in enumerate(ANG):
        xy = start + (AMP / 100) * np.outer(s, [np.cos(a), np.sin(a)])
        th = ik(xy[:, 0], xy[:, 1], l1, l2)
        om = np.gradient(th, t, axis=0)
        lm = l_const - th @ A_len
        vm = -om @ A_len
        phi = M_.flv(lm, vm)[0]                                        # (time, 6)
        sp = np.linalg.norm(np.gradient(xy, t, axis=0), axis=1)
        kp = int(sp.argmax())
        k0 = int(np.argmax(sp > 0.1 * sp[kp]))
        k1 = min(kp + int(round(0.1 / (t[1] - t[0]))), len(t) - 1)
        H[:, k] = phi[k0:k1 + 1].mean(0)
    return H


def flv_factors_dynamic(params, visc="todorov", weighted=False):
    """H (6 x 16) from the closed-loop iLQG solutions of rung 6 (fig5_macaque caches,
    20 networks), as L&S did with their dynamic model: F-L/V along each solution,
    averaged from onset (10 % of peak speed) to 100 ms after peak speed, then over
    networks. weighted=True weights the average by each muscle's activation.
    ls_cs has no dynamic run; it shares its segment lengths (hence the mapping from
    hand path to muscle state) with ls_printed, whose solutions are used."""
    import net_variability as NV
    src = "ls_printed" if params == "ls_cs" else params
    tag = visc + ("" if src == "ls_printed" else "_" + src)
    z = np.load(os.path.join(HERE, "sims", f"fig5_overlay_macaque_{tag}_n20.npz"))
    _, _, _, _, l1, l2, _, _ = macaque._SETS[src]
    A_len = M_.MOMENT_ARM / M_.L0
    l_const = 1 + np.sum(M_.MOMENT_ARM * M_.THETA0, 0) / M_.L0
    Hs = []
    for n, s in NV.NETS20:
        k = NV.netkey(n, s)
        hand = z[f"6cm_040s|nl6|{k}|hand"] / 100.0                      # (16, T, 2) m
        act = z[f"6cm_040s|nl6|{k}|a"]                                  # (16, T, 6)
        t = np.linspace(0, T_MOVE, hand.shape[1])
        H = np.zeros((6, N_T))
        for i in range(N_T):
            th = ik(hand[i, :, 0], hand[i, :, 1], l1, l2)
            om = np.gradient(th, t, axis=0)
            phi = M_.flv(l_const - th @ A_len, -om @ A_len)[0]          # (T, 6)
            sp = np.linalg.norm(np.gradient(hand[i], t, axis=0), axis=1)
            kp = int(sp.argmax())
            k0 = int(np.argmax(sp > 0.1 * sp[kp]))
            k1 = min(kp + int(round(0.1 / (t[1] - t[0]))), len(t) - 1)
            w = act[i, k0:k1 + 1] if weighted else np.ones_like(phi[k0:k1 + 1])
            H[:, i] = (w * phi[k0:k1 + 1]).sum(0) / w.sum(0)
        Hs.append(H)
    return np.mean(Hs, 0)


def rung_matrices(rung, F, G, H):
    """A (2 x n_act) mapping actuator output to hand velocity, H (n_act x 16) or
    None, and whether the actuators are muscles (sigma_u)."""
    Mm = M_.MOMENT_ARM.astype(float)
    if rung == "pm":
        return np.eye(2) * np.trace(F) / 2, None, False
    if rung == "geo":
        return G @ np.diag(np.diag(F)), None, False
    if rung == "nl2":
        return G @ F, None, False
    if rung == "mono":
        return G @ F @ Mm[:, :4], None, True
    if rung == "nl6x":
        return G @ F @ Mm, None, True
    return G @ F @ Mm, H, True


# -----------------------------------------------------------------------------
# optimisation of z (all 16 targets at once; they are independent)
# -----------------------------------------------------------------------------
def solve(A, H, muscles, Wout, Y):
    n_act, n = Wout.shape

    def fg(zf):
        Z = zf.reshape(n, N_T)
        D = Wout @ Z
        U = sigma_u(D) if muscles else D
        T = U * H if H is not None else U
        E = A @ T - Y
        J = 0.5 * np.sum(E ** 2) + 0.5 * ALPHA * np.sum(U ** 2) + 0.5 * BETA * np.sum(Z ** 2)
        dU = (A.T @ E) * (H if H is not None else 1.0) + ALPHA * U
        dD = dU * dsigma_u(D) if muscles else dU
        return J, (Wout.T @ dD + BETA * Z).ravel()

    res = minimize(fg, np.zeros(n * N_T), jac=True, method="L-BFGS-B",
                   options={"maxiter": 20000, "maxfun": 40000, "ftol": 1e-15, "gtol": 1e-10})
    Z = res.x.reshape(n, N_T)
    D = Wout @ Z
    U = sigma_u(D) if muscles else D
    err = np.sqrt(np.mean(np.sum((A @ (U * H if H is not None else U) - Y) ** 2, 0)))
    return Z, U, err


def plane_pd(X):
    """Planar regression of each row of X (units x 16 targets) on direction."""
    B = np.stack([np.ones(N_T), np.cos(ANG), np.sin(ANG)], 1)
    b, *_ = np.linalg.lstsq(B, X.T, rcond=None)
    pred = (B @ b).T
    ss_res = ((X - pred) ** 2).sum(1)
    ss_tot = ((X - X.mean(1, keepdims=True)) ** 2).sum(1) + 1e-30
    r2 = 1 - ss_res / ss_tot
    Fst = (r2 / 2) / ((1 - r2 + 1e-30) / (N_T - 3))
    p = 1 - f_dist.cdf(Fst, 2, N_T - 3)
    return np.arctan2(b[2], b[1]), p


def run(params, speed=1.0, seed=0):
    F, G, lengths = arm(params)
    H = flv_factors(lengths)
    Y = speed * np.stack([np.cos(ANG), np.sin(ANG)])
    rng = np.random.default_rng(seed)
    nets = [rng.normal(0, np.sqrt(1e-3), (6, N_UNITS)) for _ in range(N_NETS)]
    out = {}
    for rung, _ in RUNGS:
        A, Hr, muscles = rung_matrices(rung, F, G, H)
        n_act = A.shape[1]
        upd, mpd, midx, errs, tuned = [], [], [], [], []
        for W in nets:
            Z, U, err = solve(A, Hr, muscles, W[:n_act], Y)
            pd, p = plane_pd(Z)
            upd.append(pd[p < 0.05])
            tuned.append(np.mean(p < 0.05))
            if muscles:
                pdm, _ = plane_pd(U)
                mpd.append(pdm)
                midx.append(np.arange(n_act))
            errs.append(err)
        out[rung] = dict(unit=np.concatenate(upd), tuned=float(np.mean(tuned)),
                         err=float(np.max(errs)),
                         musc=np.concatenate(mpd) if mpd else None,
                         midx=np.concatenate(midx) if midx else None)
    return out, H


def figure(out, params, speed, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 6.4))
    gs = fig.add_gridspec(2, len(RUNGS))
    for j, (m, lab) in enumerate(RUNGS):
        ax = fig.add_subplot(gs[0, j], projection="polar")
        pd = out[m]["unit"]
        R, th, p = F5.bimodal_test(pd)
        F5.star(ax, pd, p < 1e-3)
        if F5.PAPER_UNIT[m] is not None:
            F5.axis_line(ax, F5.PAPER_UNIT[m][0], "red", "-", 2.2, "Lillicrap & Scott" if j == 1 else None)
        if p < 1e-3:
            F5.axis_line(ax, th, "#2171b5", "--", 2.2, "modèle statique (ce code)" if j == 1 else None)
        F5.polar_clean(ax)
        pap = F5.PAPER_UNIT[m]
        ptxt = f"L&S θ={pap[0]:.0f}° r={pap[1]:.2f}" if pap else "L&S : n.s."
        ax.set_title(f"{lab}\n{ptxt}\nstatique θ={th:.0f}° r={R:.2f}" + ("" if p < 1e-3 else " (n.s.)"),
                     fontsize=8)
        if j == 1:
            fig.legend(loc="upper left", bbox_to_anchor=(0.005, 0.99), fontsize=8, frameon=False)
    fig.text(0.005, 0.72, "A  centre-out\nPMD des unités", fontsize=9, fontweight="bold")
    for j, (m, lab) in enumerate(RUNGS):
        if m not in F5.PAPER_MUSC:
            continue
        ax = fig.add_subplot(gs[1, j], projection="polar")
        pd, idx = out[m]["musc"], out[m]["midx"]
        R, th, _ = F5.bimodal_test(pd)
        F5.muscle_markers(ax, pd, idx)
        ax.set_ylim(0, max(1.1, ax.get_ylim()[1]))
        top = ax.get_ylim()[1]
        for t, col, ls in ((F5.PAPER_MUSC[m][0], "red", "-"), (th, "#2171b5", "--")):
            ax.plot([np.radians(t), np.radians(t) + np.pi], [top] * 2, color=col, ls=ls, lw=2.2, zorder=5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)
        ax.set_title(f"L&S θ={F5.PAPER_MUSC[m][0]:.0f}° r={F5.PAPER_MUSC[m][1]:.2f}\n"
                     f"statique θ={th:.0f}° r={R:.2f}", fontsize=8)
    fig.text(0.005, 0.30, "B  centre-out\nPMD des muscles\n\n● mono épaule\n◆ mono coude\n△ biarticulaires",
             fontsize=9)
    fig.suptitle(f"Fig. 5 de Lillicrap & Scott refaite avec LEUR modèle statique (un pas au départ, "
                 f"10 réseaux × 1000 unités, 16 vitesses cibles) — bras {params}, |ẏ*| = {speed:g}",
                 fontsize=11)
    fig.tight_layout(rect=(0.06, 0, 1, 0.95))
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


def table(out):
    lines = []
    for m, lab in RUNGS:
        R, th, p = F5.bimodal_test(out[m]["unit"])
        pap = F5.PAPER_UNIT[m]
        line = (f"{lab.replace(chr(10), ' '):34s} unités θ={th:6.1f} r={R:.2f}{'' if p < 1e-3 else ' (n.s.)'}"
                f"  [L&S {('%.1f %.2f' % pap) if pap else 'n.s.'}]  accordées {100 * out[m]['tuned']:.0f}%"
                f"  err {out[m]['err']:.1e}")
        if out[m]["musc"] is not None:
            Rm, thm, _ = F5.bimodal_test(out[m]["musc"])
            line += f" | muscles θ={thm:6.1f} r={Rm:.2f} [L&S {F5.PAPER_MUSC[m][0]:.1f} {F5.PAPER_MUSC[m][1]:.2f}]"
        lines.append(line)
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", default="all", help="ls_printed | ls_cs | cheng_scott | all")
    p.add_argument("--speed", type=float, default=1.0, help="|target initial hand velocity|")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sets = ["ls_printed", "ls_cs", "cheng_scott"] if a.params == "all" else [a.params]
    txt = []
    for ps in sets:
        out, H = run(ps, a.speed)
        t = table(out)
        head = f"=== bras {ps}, |y*| = {a.speed:g}, H par muscle (moyenne sur cibles) {np.round(H.mean(1), 2)}"
        print(head + "\n" + t)
        txt.append(head + "\n" + t)
        figure(out, ps, a.speed, os.path.join(HERE, f"fig5_static_{ps}.png"))
    with open(os.path.join(HERE, "fig5_static_table.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(txt) + "\n")


if __name__ == "__main__":
    main()
