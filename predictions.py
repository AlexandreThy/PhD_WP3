"""Model-derived, data-testable predictions for *traces of knowledge of arm
nonlinearity* in real M1 recordings during center-out reaching.

Rationale
---------
The controller embeds an internal model of the nonlinear two-link arm, so the
network activity must carry the signals that generate the arm's *interaction*
torques -- the configuration-dependent inertia and, above all, the
Coriolis/centripetal torques C(theta, theta_dot), which scale as velocity^2.
These give three predictions that can be checked on M1 + kinematics alone (no
linear-arm baseline needed):

  P1  A component of M1 activity encodes the Coriolis torque *beyond* any linear
      kinematic tuning (cross-validated Delta R^2 > 0).
  P2  Because Coriolis ~ v^2, that component grows with the *square* of movement
      speed; the linear kinematic component grows ~ v. Split trials by speed.
  P3  It is time-locked to peak hand speed (where Coriolis peaks), not to peak
      acceleration.

`run_sweep()` solves center-out reaches over a range of speeds and caches them;
`fig_predictions()` produces the summary figure.
"""

import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import iLQG_Combined as M
import centerout as CO

a2 = M.a2
CACHE = "predictions_sweep.npz"

# (amplitude cm, duration s) chosen to sweep peak hand speed ~0.3 - ~1.1 m/s
CONDITIONS = [(10, 0.60), (12, 0.50), (15, 0.42), (18, 0.38), (22, 0.34)]


# -----------------------------------------------------------------------------
# simulate a speed sweep
# -----------------------------------------------------------------------------
def run_sweep(conditions=CONDITIONS, n_dir=8, center=(0.0, 40.0), K=100,
              w1=1e4, w2=10.0, r1=1e-5, verbose=True):
    nc = len(conditions)
    N = M.W.shape[0]
    states = np.zeros((nc, n_dir, K + 1, N + 6))
    hand = np.zeros((nc, n_dir, K + 1, 2))
    durs = np.array([d for _, d in conditions])
    amps = np.array([a for a, _ in conditions])
    ang = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    t0 = time.time()
    for c, (amp, dur) in enumerate(conditions):
        for i, a in enumerate(ang):
            tgt = [center[0] + amp * np.cos(a), center[1] + amp * np.sin(a)]
            X, Y, x, u = M.simulate_ILQG(Duration=dur, w1=w1, w2=w2, r1=r1,
                                         targets=tgt, start=list(center), K=K,
                                         print_iterations=False)
            states[c, i] = x
            hand[c, i, :, 0], hand[c, i, :, 1] = X, Y
        if verbose:
            print(f"  cond {c} amp={amp}cm dur={dur}s done ({time.time()-t0:.0f}s)")
    np.savez_compressed(CACHE, states=states, hand=hand, durs=durs, amps=amps,
                        ang=ang, N=N, K=K)
    print("saved", CACHE)


def load():
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


# -----------------------------------------------------------------------------
# kinematics / Coriolis / peak speed
# -----------------------------------------------------------------------------
def coriolis(states, N):
    """Coriolis/centripetal joint torque C(theta,theta_dot) from the arm state."""
    q = states[..., N:]
    th2, w1, w2 = q[..., 1], q[..., 2], q[..., 3]
    C1 = -w2 * (2 * w1 + w2) * a2 * np.sin(th2)
    C2 = w1 ** 2 * a2 * np.sin(th2)
    return np.stack([C1, C2], axis=-1)


def peak_speed(hand, dur):
    dt = dur / (hand.shape[-2] - 1)
    v = np.gradient(hand, dt, axis=-2) / 100.0
    return np.linalg.norm(v, axis=-1).max()


# -----------------------------------------------------------------------------
# P1/P2: cross-validated encoding of Coriolis beyond linear kinematics
# -----------------------------------------------------------------------------
def encoding_dr2(states, N, dur):
    """Per-neuron leave-one-direction-out CV R^2 for a linear-kinematics model
    and a linear+Coriolis model. Returns (R2_lin, R2_full, dR2) over neurons.

    Linear kinematic regressors: constant, joint angles, joint velocities, joint
    accelerations (a stringent classical encoding model). Nonlinear regressors:
    the two Coriolis torque components (velocity^2 terms).
    """
    nd, T, _ = states[:, :, :N].shape
    r = states[:, :, :N]
    q = states[:, :, N:]
    dt = dur / (T - 1)
    th = q[:, :, 0:2]
    w = q[:, :, 2:4]
    acc = np.gradient(w, dt, axis=1)
    C = coriolis(states, N)

    ones = np.ones((nd, T, 1))
    Xlin = np.concatenate([ones, th, w, acc], axis=2)     # (nd,T,7)
    Xful = np.concatenate([Xlin, C], axis=2)              # (nd,T,9)

    def cv_sse(X):
        sse = np.zeros(N); sst = np.zeros(N)
        for hobj in range(nd):                            # leave-one-direction-out
            tr = [d for d in range(nd) if d != hobj]
            Xtr = X[tr].reshape(-1, X.shape[2]); Ytr = r[tr].reshape(-1, N)
            beta, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
            pred = X[hobj] @ beta
            sse += ((r[hobj] - pred) ** 2).sum(0)
            sst += ((r[hobj] - Ytr.mean(0)) ** 2).sum(0)
        return sse, sst

    sse_lin, sst = cv_sse(Xlin)
    sse_ful, _ = cv_sse(Xful)
    R2lin = 1 - sse_lin / (sst + 1e-12)
    R2ful = 1 - sse_ful / (sst + 1e-12)
    # population aggregate: total SSE reduction / total variance (variance-weighted)
    pop_dr2 = (sse_lin.sum() - sse_ful.sum()) / (sst.sum() + 1e-12)
    return R2lin, R2ful, R2ful - R2lin, pop_dr2


# -----------------------------------------------------------------------------
# figure
# -----------------------------------------------------------------------------
def coriolis_from_kinematics_r2(states, N, dur):
    """How well is C linearly reconstructed from [theta,theta_dot,theta_ddot]?
    Low R^2 => C carries information independent of linear kinematics."""
    nd, T, _ = states[:, :, :N].shape
    q = states[:, :, N:]
    dt = dur / (T - 1)
    th, w = q[:, :, 0:2], q[:, :, 2:4]
    acc = np.gradient(w, dt, axis=1)
    X = np.concatenate([np.ones((nd, T, 1)), th, w, acc], 2).reshape(-1, 7)
    Y = coriolis(states, N).reshape(-1, 2)
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    p = X @ b
    return 1 - ((Y - p) ** 2).sum() / (((Y - Y.mean(0)) ** 2).sum() + 1e-12)


def fig_predictions(fname="fig_predictions.png"):
    d = load()
    states, hand, durs, amps, N = (d["states"], d["hand"], d["durs"],
                                   d["amps"], int(d["N"]))
    nc = states.shape[0]
    T = states.shape[2]
    vpk = np.array([peak_speed(hand[c], durs[c]) for c in range(nc)])

    cor_peak = np.zeros(nc); cor_frac = np.zeros(nc)
    cor_indep = np.zeros(nc); cor_var = np.zeros(nc)
    for c in range(nc):
        C = coriolis(states[c], N)
        tau = states[c, :, :, N + 4:N + 6]
        cor_peak[c] = np.linalg.norm(C, axis=2).max()
        cor_frac[c] = np.linalg.norm(C, axis=2).max() / np.linalg.norm(tau, axis=2).max()
        cor_indep[c] = 1 - coriolis_from_kinematics_r2(states[c], N, durs[c])
        cor_var[c] = (C ** 2).sum() / (tau ** 2).sum()   # variance fraction of torque

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

    # (a) TEMPORAL DISSOCIATION (the within-task signature): Coriolis peaks at
    #     peak speed (velocity^2), inertial torque is biphasic (acceleration).
    c = nc - 1
    dt = durs[c] / (T - 1)
    tc = np.arange(T) * dt * 1000
    C = coriolis(states[c], N)
    q = states[c, :, :, N:]
    acc = np.gradient(q[:, :, 2:4], dt, axis=1)
    v = np.gradient(hand[c], dt, axis=1) / 100.0
    sp = np.linalg.norm(v, axis=2).mean(0)
    Cm = np.linalg.norm(C, axis=2).mean(0)
    Am = np.linalg.norm(acc, axis=2).mean(0)
    ax = axes[0]
    ax.plot(tc, sp / sp.max(), "k", lw=2, label="hand speed")
    ax.plot(tc, Cm / Cm.max(), "crimson", lw=2, label="|Coriolis| (∝ v²)")
    ax.plot(tc, Am / Am.max(), "steelblue", lw=2, ls="--", label="|joint accel| (inertial)")
    ax.axvline(tc[sp.argmax()], color="0.6", lw=0.8)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("normalized")
    ax.set_title("Within-task signature (P3):\nCoriolis locked to PEAK SPEED, "
                 "inertial biphasic")
    ax.legend(fontsize=8)

    # (b) magnitude: Coriolis ∝ v^2, and its fraction of torque grows with amplitude
    ax = axes[1]
    ax.plot(vpk ** 2, cor_peak, "o-", color="crimson")
    ax.set_xlabel("peak hand speed$^2$ (m/s)$^2$")
    ax.set_ylabel("peak |Coriolis| (N·m)", color="crimson")
    ax2 = ax.twinx()
    ax2.plot(amps, cor_frac * 100, "s--", color="darkorange")
    ax2.set_ylabel("|Coriolis|/|torque| (%)  vs amplitude", color="darkorange")
    ax.set_title("Magnitude (P2): ∝ v²,\ngrows with reach amplitude")

    # (c) detectability caveat: C is largely INDEPENDENT of kinematics, but tiny
    #     variance -> naive regression fails; needs dissociation / timing.
    ax = axes[2]
    x = np.arange(nc)
    ax.bar(x - 0.2, cor_indep * 100, width=0.4, color="seagreen",
           label="C independent of kinematics (%)")
    ax.bar(x + 0.2, cor_var * 100, width=0.4, color="indianred",
           label="C variance / torque variance (%)")
    ax.set_xticks(x); ax.set_xticklabels([f"{a:.0f}" for a in amps])
    ax.set_xlabel("reach amplitude (cm)")
    ax.set_ylabel("%")
    ax.set_title("Why naive regression fails:\nindependent but tiny-variance")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)
    print("peak speed (m/s):        ", np.round(vpk, 2))
    print("peak |Coriolis| (N·m):   ", np.round(cor_peak, 3))
    print("|Coriolis|/|torque| (%): ", np.round(cor_frac * 100, 1))
    print("C independent of kin (%):", np.round(cor_indep * 100, 0))
    print("C variance/torque var(%):", np.round(cor_var * 100, 2))


if __name__ == "__main__":
    if not os.path.exists(CACHE):
        run_sweep()
    fig_predictions()
