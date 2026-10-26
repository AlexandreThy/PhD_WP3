"""Screen for nonlinearity signatures measurable in monkey M1 data.

    lin6   linearized arm + 6 rectified muscles, no F-L/V   (the 'linear' body)
    nl6    nonlinear arm + 6 rectified muscles + F-L/V      (the full body)

Both: same random network, same iLQG, same joint-space cost, same 24 targets
(sims/cmp_*.npz from compare3.py). Every measure below uses only what a monkey
experiment gives: condition-averaged firing rates per target and time (neurons
may be recorded one at a time), target directions, and hand kinematics.

Measures are computed with all 24 targets and with the 8-target subset of a
standard centre-out task.  `noise_test` repeats the decisive ones on Poisson
spike counts with a realistic number of trials.

    python analyses_claude/signatures.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SIMS = os.path.join(HERE, "sims")


def load(regime, model, every=1):
    d = np.load(os.path.join(SIMS, f"cmp_{regime}_{model}.npz"))
    N = int(d["N"])
    sel = np.arange(0, len(d["ang"]), every)
    r = d["states"][sel, :, :N]
    hand = d["hand"][sel]
    dur = float(d["duration"])
    T = r.shape[1]
    v = np.gradient(hand, dur / (T - 1), axis=1)
    speed = np.linalg.norm(v, axis=-1).mean(0)
    return dict(r=r, ang=d["ang"][sel], speed=speed, t=d["t"], dur=dur,
                k_peak=int(np.argmax(speed)))


def rates(r, base=20.0, depth=8.0):
    """Map model activity to firing rates (Hz): each unit z-scored over
    conditions x time, then base + depth * z, rectified (as power_curve)."""
    mu = r.mean((0, 1), keepdims=True)
    sd = r.std((0, 1), keepdims=True) + 1e-12
    return np.maximum(base + depth * (r - mu) / sd, 0.0)


# ---------------------------------------------------------------- tuning / PD
def cosine_fit(y, ang):
    """y: (n_dir, N) -> PD (rad), depth, R2 of y = b0 + b1 cos + b2 sin."""
    X = np.c_[np.ones_like(ang), np.cos(ang), np.sin(ang)]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    r2 = 1 - (res ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-12)
    return np.arctan2(b[2], b[1]), np.hypot(b[1], b[2]), r2


def bimodal(pd):
    z = np.exp(2j * pd).mean()
    return np.abs(z), np.degrees(np.angle(z) / 2)


def epoch_mean(r, k0, k1):
    return r[:, k0:k1, :].mean(1)


def pd_measures(d):
    r, ang, kp = d["r"], d["ang"], d["k_peak"]
    T = r.shape[1]
    pd, depth, r2 = cosine_fit(epoch_mean(r, 1, kp + 1), ang)        # L&S epoch
    keep = r2 > 0.5
    R, axis = bimodal(pd[keep])
    # PD drift: early (onset -> peak speed) vs late (peak speed -> end)
    pd_late, _, r2l = cosine_fit(epoch_mean(r, kp, T), ang)
    both = keep & (r2l > 0.5)
    drift = np.degrees(np.abs(np.angle(np.exp(1j * (pd_late - pd)))))[both]
    # 2nd harmonic share of the tuning curve, peak-speed epoch
    y = epoch_mean(r, 1, kp + 1)
    F = np.fft.rfft(y - y.mean(0), axis=0)
    h2 = (np.abs(F[2]) / (np.abs(F[1]) + 1e-12)) if len(ang) >= 6 else np.nan
    return dict(bimodal_R=R, axis=axis, cos_r2=float(np.median(r2)),
                frac_tuned=float(keep.mean()), pd_drift=float(np.median(drift)),
                h2_over_h1=float(np.median(h2)))


# --------------------------------------------------------------- population
def population_measures(d):
    r, kp = d["r"], d["k_peak"]
    n_dir, T, N = r.shape
    cis = r.mean(0, keepdims=True)                         # condition-invariant
    dep = r - cis
    tot = ((r - r.mean((0, 1))) ** 2).sum()
    cis_frac = ((cis - r.mean((0, 1))) ** 2).sum() * n_dir / tot
    # out-of-plane: at each time, across-condition variance beyond the top-2 PCs
    oop = []
    for k in range(1, T):
        s = np.linalg.svd(dep[:, k, :], compute_uv=False) ** 2
        oop.append(s[2:].sum() / (s.sum() + 1e-12))
    oop = np.array(oop)
    # even share of the condition-dependent variance (opposite targets)
    half = n_dir // 2
    ev = (dep[:half] + dep[half:]) / 2
    even_frac = 2 * (ev ** 2).sum() / ((dep ** 2).sum() + 1e-12)
    # accel / decel asymmetry of condition-dependent modulation
    mod = np.sqrt((dep ** 2).sum((0, 2)))
    acc_dec = mod[1:kp + 1].mean() / (mod[kp:].mean() + 1e-12)
    return dict(cis_frac=float(cis_frac), oop_mean=float(oop.mean()),
                oop_peak=float(oop.max()), oop_t=float(d["t"][1:][np.argmax(oop)] * 1000),
                even_frac=float(even_frac), acc_dec=float(acc_dec))


def screen():
    rows = []
    for regime in ("12cm_030s", "12cm_050s"):
        for every, nt in ((1, 24), (3, 8)):
            for m in ("lin6", "nl6"):
                d = load(regime, m, every)
                rows.append((regime, nt, m, {**pd_measures(d), **population_measures(d)}))
    keys = list(rows[0][3])
    print(f"{'régime':10s} {'cib':>3s} {'modèle':6s} " + " ".join(f"{k:>11s}" for k in keys))
    for rg, nt, m, v in rows:
        print(f"{rg:10s} {nt:3d} {m:6s} " + " ".join(f"{v[k]:11.3f}" for k in keys))
    return rows


# --------------------------------------------------------------- noise test
def noisy_average(rate, trials, dt, rng):
    """Trial-averaged firing rate from Poisson counts (Hz)."""
    lam = rate * dt
    return rng.poisson(lam[None] * np.ones((trials,) + lam.shape)).mean(0) / dt


def noise_test(regime="12cm_030s", every=3, trials=(5, 10, 20), n_rep=200, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for m in ("lin6", "nl6"):
        d = load(regime, m, every)
        rate = rates(d["r"])
        dt = d["dur"] / (rate.shape[1] - 1)
        for tr in trials:
            vals = []
            for _ in range(n_rep):
                y = noisy_average(rate, tr, dt, rng)
                dn = dict(d, r=y)
                vals.append({**pd_measures(dn), **population_measures(dn)})
            out[m, tr] = {k: np.array([v[k] for v in vals]) for k in vals[0]}
    return out


if __name__ == "__main__":
    screen()
