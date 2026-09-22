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


def inertia_variation(states, N, dur, th2_ref=None):
    """Interaction torque from the *configuration-dependence of inertia*:

        dM(th2) @ thddot,   dM = M(th2) - M(th2_ref)
                               = a2*(cos th2 - cos th2_ref) * [[2,1],[1,0]]

    This is the other purely-nonlinear term besides Coriolis: it vanishes for the
    frozen-inertia baseline. Empirically it is the *stronger* trace in network
    activity (see `term_dr2`), and unlike Coriolis it is acceleration-locked
    (biphasic), not peak-speed-locked.
    """
    q = states[..., N:]
    th2 = q[..., 1]
    if th2_ref is None:
        th2_ref = q[..., 0, 1] if q.ndim > 2 else q[0, 1]
        th2_ref = np.asarray(th2_ref).flat[0]
    dt = dur / (states.shape[-2] - 1)
    acc = np.gradient(q[..., 2:4], dt, axis=-2)
    dc = a2 * (np.cos(th2) - np.cos(th2_ref))
    return np.stack([dc * (2 * acc[..., 0] + acc[..., 1]), dc * acc[..., 0]], -1)


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
# The sensitive test: encode the DIRECTION-EVEN component only
# -----------------------------------------------------------------------------
def term_dr2(states, N, dur, term="both", even=True):
    """Cross-validated dR2 for adding a *nonlinear* interaction-torque regressor
    on top of a linear-kinematics encoding model [1, th, thdot, thddot].

    term : 'coriolis' | 'Mvar' | 'both'
    even : restrict to the direction-even component r_e = [r(th)+r(th+pi)]/2.

    Why `even=True` matters
    -----------------------
    Both nonlinear terms are (to leading order) EVEN under reach-direction
    reversal -- Coriolis because it scales as v^2, inertia-variation because
    dM flips sign with dth2 while thddot flips too. The odd/1st-harmonic
    component carries none of them but dominates the variance, so pooling over
    all directions buries the effect (dR2 ~ 0.004). Averaging opposite
    directions first removes that variance and raises the same effect to
    dR2 ~ 0.16 -- a ~40x gain in detectability, at no cost in assumptions.

    Cross-validation is leave-one-direction-pair-out. Run it on the linearized
    arm to get the null: that controller never had to generate either term, so
    its dR2 is the kinematic confound (C and dM are partly predictable from
    kinematics), not knowledge of the nonlinearity.
    """
    st = states
    nd, T, _ = st[:, :, :N].shape
    dt = dur / (T - 1)
    r = st[:, :, :N]
    q = st[:, :, N:]
    th, w = q[:, :, 0:2], q[:, :, 2:4]
    acc = np.gradient(w, dt, axis=1)
    Xlin = np.concatenate([np.ones((nd, T, 1)), th, w, acc], 2)
    C = coriolis(st, N)
    Mv = inertia_variation(st, N, dur)
    extra = {"coriolis": C, "Mvar": Mv, "both": np.concatenate([C, Mv], 2)}[term]
    Xful = np.concatenate([Xlin, extra], 2)

    if even:
        half = nd // 2
        ev = lambda A: (A[:half] + A[half:]) / 2.0
        r, Xlin, Xful = ev(r), ev(Xlin), ev(Xful)
    nfold = r.shape[0]

    def cv(X):
        sse = np.zeros(N)
        sst = np.zeros(N)
        for h in range(nfold):
            tr = [k for k in range(nfold) if k != h]
            b, *_ = np.linalg.lstsq(X[tr].reshape(-1, X.shape[2]),
                                    r[tr].reshape(-1, N), rcond=None)
            sse += ((r[h] - X[h] @ b) ** 2).sum(0)
            sst += ((r[h] - r[tr].reshape(-1, N).mean(0)) ** 2).sum(0)
        return sse, sst

    sse_l, sst = cv(Xlin)
    sse_f, _ = cv(Xful)
    pop = (sse_l.sum() - sse_f.sum()) / (sst.sum() + 1e-12)
    return pop, (sse_l - sse_f) / (sst + 1e-12)


# -----------------------------------------------------------------------------
# The kinematic prediction: what a LINEAR internal model would do to the real arm
# -----------------------------------------------------------------------------
def linear_internal_model_error(d_nl, d_lin):
    """Drive the REAL (nonlinear) arm with the LINEAR controller's commands.

    This is the counterfactual "brain that ignores interaction torques": the
    motor command is optimal for a frozen-inertia, no-Coriolis arm, but the limb
    it actually moves is the nonlinear one. The resulting endpoint error is the
    kinematic cost of not knowing the nonlinearity -- a plain distance in cm, no
    regression involved.

    Returns dict with per-direction error vectors `E` (n_dir, 2), the rolled-out
    hand paths, the mean error, and the even/odd decomposition of the error.

    The signature (measured): the error is ~92-99% **direction-EVEN** -- reaches
    to OPPOSITE targets are pushed in the SAME spatial direction (mean cosine
    between opposite error vectors +0.90 to +0.98). Ordinary motor errors (gain,
    bias, noise) are odd, i.e. mirror-image for opposite reaches. That asymmetry
    is the fingerprint, and it needs no statistics to see.

    Magnitudes: 1.42 cm (12% of reach) at 12 cm / 0.50 s (peak 0.45 m/s);
    4.14 cm (21% of reach) at 20 cm / 0.35 s (peak 1.17 m/s). At MATCHED peak
    speed the error grows with amplitude (8.3% of reach at 8 cm -> 12.2% at
    12 cm), because Coriolis/inertial ~ D.
    """
    import iLQG_Combined as _M
    N = int(d_nl["N"])
    dur = float(d_nl["duration"])
    K = int(d_nl["K"])
    dt = dur / K
    c = d_nl["center"]
    st1, st2 = _M.compute_angles_from_cartesian(c[0], c[1])
    x0 = np.concatenate([np.zeros(N), [st1, st2, 0, 0, 0, 0]])
    targ = d_nl["targ"]
    nd = len(targ)

    prev_lin, prev_ref = _M.LINEARIZE, _M.THETA2_REF
    _M.LINEARIZE = False            # the REAL arm
    _M.THETA2_REF = st2
    try:
        paths, E = [], []
        for i in range(nd):
            x = x0.copy()
            tr = [x0]
            for k in range(K):
                x = x + dt * _M.f(x, d_lin["controls"][i][k])
                tr.append(x)
            X, Y = _M.hand_xy(np.array(tr))
            paths.append(np.stack([X, Y], 1))
            E.append([X[-1] - targ[i, 0], Y[-1] - targ[i, 1]])
    finally:
        _M.LINEARIZE, _M.THETA2_REF = prev_lin, prev_ref

    E = np.array(E)
    paths = np.array(paths)
    h = nd // 2
    even = (E[:h] + E[h:]) / 2.0
    odd = (E[:h] - E[h:]) / 2.0
    cos = np.array([np.dot(E[i], E[i + h]) /
                    (np.linalg.norm(E[i]) * np.linalg.norm(E[i + h]) + 1e-12)
                    for i in range(h)])
    ne, no = np.linalg.norm(even), np.linalg.norm(odd)
    return {
        "E": E, "paths": paths,
        "err": np.linalg.norm(E, axis=1),
        "err_mean": float(np.linalg.norm(E, axis=1).mean()),
        "even_share": float(ne ** 2 / (ne ** 2 + no ** 2)),
        "opposite_cosine": cos,
        "mean_cosine": float(cos.mean()),
    }


# -----------------------------------------------------------------------------
# Power analysis for a real recording
# -----------------------------------------------------------------------------
def simulate_recording(states, N, n_trials, rng, baseline=20.0, mod_hz=8.0,
                       binsize=0.020, dur=None):
    """Treat model activity as a firing rate, emit Poisson spikes, trial-average.

    Per-neuron activity is z-scored and rescaled to `mod_hz` (Hz SD of
    modulation) on a `baseline` Hz floor -- a plausible M1 unit. Returns the
    trial-averaged rate estimate (n_dir, n_bins, N) in Hz, plus the binning
    factor used, so the same factor can be applied to the regressors.
    """
    r = states[:, :, :N]
    sd = r.std(axis=(0, 1), keepdims=True)
    sd[sd < 1e-12] = 1e-12
    lam = np.clip(baseline + mod_hz * (r - r.mean(axis=(0, 1), keepdims=True)) / sd,
                  0.5, None)
    f = 1
    if dur is not None:
        f = max(1, int(round(binsize / (dur / (states.shape[1] - 1)))))
        nd, T, _ = lam.shape
        T2 = (T // f) * f
        lam = lam[:, :T2, :].reshape(nd, T2 // f, f, N).mean(2)
    counts = rng.poisson(lam * binsize, size=(n_trials,) + lam.shape)
    return counts.mean(0) / binsize, f


def power_curve(d, trials=(10, 20, 50, 100), n_boot=25, seed=0, term="both"):
    """How many trials per direction to see dR2_even > 0?

    Returns {n_trials: (mean, sd)} of dR2_even over simulated sessions.

    Interpretation. Under cross-validation, regressors carrying no information
    give dR2 <= 0 -- so **the sign is the null**: no linearized twin is needed.
    A linear internal model yields dR2_even < 0 at every trial count; only a
    controller that actually generates the interaction torques makes it positive.

    Model reference (100 neurons, 24 directions, 20 ms bins, 8 Hz modulation SD,
    20 cm / 0.35 s reaches; nonlinear arm vs its linearized twin):

        trials/dir     nonlinear          linear (null)
             10        -0.015 +- 0.003    -0.027 +- 0.002
             20        -0.000 +- 0.003    -0.025 +- 0.002
             50        +0.031 +- 0.003    -0.020 +- 0.002
            100        +0.063 +- 0.004    -0.015 +- 0.001
            200        +0.097 +- 0.004    -0.009 +- 0.001

    so ~50 trials/direction is the point where dR2_even turns reliably positive.
    """
    from numpy.random import default_rng
    N = int(d["N"])
    dur = float(d["duration"])
    st = d["states"]
    nd, T = st.shape[:2]
    dt = dur / (T - 1)
    q = st[:, :, N:]
    acc = np.gradient(q[:, :, 2:4], dt, axis=1)
    Xlin = np.concatenate([np.ones((nd, T, 1)), q[:, :, 0:2], q[:, :, 2:4], acc], 2)
    extra = {"coriolis": coriolis(st, N), "Mvar": inertia_variation(st, N, dur),
             "both": np.concatenate([coriolis(st, N),
                                     inertia_variation(st, N, dur)], 2)}[term]
    Xful = np.concatenate([Xlin, extra], 2)
    half = nd // 2
    rng = default_rng(seed)
    out = {}
    for K in trials:
        vals = []
        for _ in range(n_boot):
            rate, f = simulate_recording(st, N, K, rng, dur=dur)
            def crop(X):
                T2 = (X.shape[1] // f) * f
                return X[:, :T2, :].reshape(nd, T2 // f, f, X.shape[2]).mean(2)
            Xl, Xf = crop(Xlin), crop(Xful)
            ev = lambda A: (A[:half] + A[half:]) / 2.0
            re, Xl, Xf = ev(rate), ev(Xl), ev(Xf)

            def cv(X):
                sse = sst = 0.0
                for h in range(half):
                    tr = [k for k in range(half) if k != h]
                    b, *_ = np.linalg.lstsq(X[tr].reshape(-1, X.shape[2]),
                                            re[tr].reshape(-1, N), rcond=None)
                    sse += ((re[h] - X[h] @ b) ** 2).sum()
                    sst += ((re[h] - re[tr].reshape(-1, N).mean(0)) ** 2).sum()
                return sse, sst
            sl, ss = cv(Xl); sf, _ = cv(Xf)
            vals.append((sl - sf) / (ss + 1e-12))
        out[K] = (float(np.mean(vals)), float(np.std(vals)))
    return out


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
