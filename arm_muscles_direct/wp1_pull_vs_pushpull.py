"""WP1 center-out with the WP1 iLQG (WP1/Controllers/ILQG.py): muscles that can
push (WP1 as is) vs muscles that can only pull (Lillicrap & Scott's sigma_u).

WP1's ILQG drives the arm with six muscle commands u (no activation dynamics):
joint torque = MOMENT_ARM @ (u * f_l * f_v), u free in sign -- in the WP1
solutions u < 0 about half of the time, i.e. the muscles push. Its units match
Lillicrap & Scott's tension F_max a (F_max = 800 N, moment arms in m) for
u = 8 a. A new control v is introduced:

    push-pull : u = G v,                          G = 8 * 2 e^-4
    pull-only : u = 8 sigma_u(v + v*)             (> 0: the muscle can only pull)

G is the slope of 8 sigma_u's linear branch, which passes through 0, so both
laws coincide for large commands. v* is the smallest resting command for which
the pull-only muscles exert zero net torque at the start posture (8 sigma_u(0)
alone does not), so both models start at rest. The effort weight on v is
WR * G^2: the push-pull model is then exactly WP1's ILQG (same optimum, u = G v).

WP1/Controllers/ILQG.py is not modified: its f, fx, fu are wrapped at import.
Task and settings are WP1/Final_Code's: start (0, 40) cm, 15 cm, 0.4 s, K = 40,
WP = 20000, WV = 10, WR = 0.1, motor noise 5e-4, sensory delay 60 ms, 8 targets,
15 noisy repetitions (+ one noise-free solution per target).

    python arm_muscles_direct/wp1_pull_vs_pushpull.py [--num-sim 15] [--jobs 7] [--force]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
WP1 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "WP1")
sys.path.insert(0, WP1)
warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)

import Controllers.ILQG as W  # noqa: E402

# --- WP1/Final_Code/common.py settings -------------------------------------
WP, WV, WR = 20000, 10, 0.1
MOTOR_NOISE, DELAY = 5e-4, 0.06
START = (0.0, 40.0)
AMPLITUDE, DURATION, K, N_T = 15.0, 0.4, 40, 8

SCALE = 8.0                                   # u = SCALE * a  (F_max * moment arm)
G = SCALE * 2.0 * np.exp(2.0 * 0.5 - 5.0)     # slope of SCALE * sigma_u's linear branch
CACHE = os.path.join(HERE, "wp1_sims.npz")


def sigma_u(x):
    x = np.asarray(x, dtype=float)
    return np.where(x < 0.5, np.exp(2.0 * np.minimum(x, 0.5) - 5.0), 2.0 * x * np.exp(-4.0))


def dsigma_u(x):
    x = np.asarray(x, dtype=float)
    return np.where(x < 0.5, 2.0 * np.exp(2.0 * np.minimum(x, 0.5) - 5.0), 2.0 * np.exp(-4.0))


# --- activation law, switched per process ------------------------------------
STATE = {"pull": False, "v_rest": np.zeros(6)}


def to_u(v):
    if STATE["pull"]:
        return SCALE * sigma_u(v + STATE["v_rest"])
    return G * np.asarray(v, dtype=float)


def du_dv(v):
    if STATE["pull"]:
        return SCALE * dsigma_u(v + STATE["v_rest"])
    return np.full(6, G)


_f, _fx, _fu = W.f, W.fx, W.fu


def f(x, u, F=0, plant=W.FULL_PLANT):
    return _f(x, to_u(u), F, plant)


def fx(x, u, plant=W.FULL_PLANT):
    return _fx(x, to_u(u), plant)


def fu(x, u, plant=W.FULL_PLANT):
    return _fu(x, to_u(u), plant) * du_dv(u)[None, :]


W.f, W.fx, W.fu = f, fx, fu


def rest_command():
    """Smallest v* with zero net pull-only torque at the start posture."""
    th = W.compute_angles_from_cartesian(*START)
    x0 = np.array([th[0], th[1], 0.0, 0.0])
    _, _, _, fl, fv = W.muscle_force_scaling(x0)
    tq = lambda v: W.MOMENT_ARM @ (SCALE * sigma_u(v) * fl * fv)
    res = minimize(lambda v: v @ v, np.zeros(6), method="SLSQP",
                   constraints={"type": "eq", "fun": tq}, options={"ftol": 1e-14})
    assert res.success and np.abs(tq(res.x)).max() < 1e-9, res
    return res.x


def set_model(pull):
    STATE["pull"] = pull
    STATE["v_rest"] = rest_command() if pull else np.zeros(6)


def targets():
    ang = np.arange(N_T) * 2 * np.pi / N_T
    return ang, np.array(START) + AMPLITUDE * np.stack([np.cos(ang), np.sin(ang)], 1)


def simulate_ilqg(duration, w1, w2, r1, target, start, K, noise, delay, motor_noise,
                  eps=1e-10, max_iter=300, rtol=1e-12):
    """WP1's simulate_ILQG (same steps 1-5, same line search), with one change: it
    also stops when the relative cost decrease falls below rtol, so that the
    noisy closed-loop execution (step5) always runs. WP1's only criterion,
    max |du| < 1e-14, is never met by the pull-only model within 300 iterations,
    and simulate_ILQG then returns the noise-free open-loop rollout."""
    plant = W.FULL_PLANT
    obj1, obj2 = W.compute_angles_from_cartesian(*target)
    st1, st2 = W.compute_angles_from_cartesian(*start)
    x0 = np.array([st1, st2, 0, 0])
    plant = W.replace(plant, elbow0=st2)
    m = 6
    u = np.zeros((K, m))
    kdelay = int(delay / (duration / K))
    cbold = np.zeros((K, m, 4))
    C = np.zeros((K, m, 4, m))
    for i in range(K):
        for j in range(2):
            cbold[i, j, 2 + j] = np.sqrt(motor_noise)
    xtarg = np.array([obj1, obj2])
    for it in range(max_iter):
        x = W.step1(x0, u, duration, plant)
        A, B, q, qbold, r, Q, R = W.step2(x, u, duration, w1, w2, r1, xtarg, plant)
        lg, Lg = W.step3(A, B, C, cbold, q, qbold, r, Q, R, eps)
        du = W.step4(lg, Lg, K, A, B)
        J = W.total_cost(x, u, w1, w2, r1, xtarg)
        alpha = 1.0
        while alpha > 1e-8:
            Jn = W.total_cost(W.step1(x0, u + alpha * du, duration, plant), u + alpha * du,
                              w1, w2, r1, xtarg)
            if np.isfinite(Jn) and Jn < J:
                break
            alpha /= 2
        else:
            break
        u = u + alpha * du
        if np.max(np.abs(alpha * du)) < 1e-14 or (J - Jn) / max(abs(J), 1e-12) < rtol:
            break
    x = W.step1(x0, u, duration, plant)
    A, B, q, qbold, r, Q, R = W.step2(x, u, duration, w1, w2, r1, xtarg, plant)
    lg, Lg = W.step3(A, B, C, cbold, q, qbold, r, Q, R, eps)
    x = W.step5(x0, lg * 0, Lg, duration, noise, A, B, K, u, kdelay, motor_noise, False, 0.0, plant)
    X = np.cos(x[:, 0] + x[:, 1]) * 33 + np.cos(x[:, 0]) * 30
    Y = np.sin(x[:, 0] + x[:, 1]) * 33 + np.sin(x[:, 0]) * 30
    return X, Y, x, u, it + 1


def run_one(task):
    pull, i, noise, seed = task
    set_model(pull)
    np.random.seed(seed)
    _, tg = targets()
    X, Y, x, v, _ = simulate_ilqg(DURATION, WP, WV, WR * G ** 2, list(tg[i]), list(START), K,
                                  noise, DELAY, MOTOR_NOISE)
    u = np.array([to_u(vk) for vk in v])
    return task, np.stack([X, Y], 1), x, v, u


def simulate(num_sim, jobs):
    tasks = [(pull, i, False, 0) for pull in (False, True) for i in range(N_T)]
    tasks += [(pull, i, True, 1000 * rep + i) for pull in (False, True)
              for rep in range(num_sim) for i in range(N_T)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(run_one, tasks, chunksize=1))
    out = {}
    for key, pull in (("push", False), ("pull", True)):
        det = sorted((r for r in res if r[0][0] == pull and not r[0][2]), key=lambda r: r[0][1])
        out[f"{key}|hand"] = np.array([r[1] for r in det])
        out[f"{key}|x"] = np.array([r[2] for r in det])
        out[f"{key}|v"] = np.array([r[3] for r in det])
        out[f"{key}|u"] = np.array([r[4] for r in det])
        noisy = [r for r in res if r[0][0] == pull and r[0][2]]
        hand = np.zeros((num_sim, N_T, K + 1, 2))
        xs = np.zeros((num_sim, N_T, K + 1, 4))
        for r in noisy:
            rep = r[0][3] // 1000
            hand[rep, r[0][1]] = r[1]
            xs[rep, r[0][1]] = r[2][:, :4]
        out[f"{key}|hand_noisy"] = hand
        out[f"{key}|x_noisy"] = xs
    set_model(True)
    out["v_rest"] = STATE["v_rest"]
    np.savez_compressed(CACHE, **out)


# -----------------------------------------------------------------------------
def hand_speed(x):
    """Cartesian hand speed (m/s) from joint state(s) (..., 4)."""
    ts, te, os_, oe = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    vx = -30 * np.sin(ts) * os_ - 33 * np.sin(ts + te) * (os_ + oe)
    vy = 30 * np.cos(ts) * os_ + 33 * np.cos(ts + te) * (os_ + oe)
    return np.hypot(vx, vy) / 100.0


def entropy_rank(s):
    s = s[s > 0]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def report(z):
    ang, tg = targets()
    dt = DURATION / K
    lines = [f"WP1 ILQG, centre-out {START} cm, {AMPLITUDE:.0f} cm, {DURATION} s, {N_T} cibles, "
             f"commande de repos tirent seulement v* = {np.round(z['v_rest'], 3)}",
             f"{'':34s} {'poussent+tirent':>16s} {'tirent seulement':>17s}"]
    ms = {}
    for key in ("push", "pull"):
        x, u, hand = z[f"{key}|x"], z[f"{key}|u"], z[f"{key}|hand"]
        sp = hand_speed(x)
        rel = hand - np.array(START)
        ud = np.stack([np.cos(ang), np.sin(ang)], 1)
        lat = np.abs(rel[..., 0] * ud[:, None, 1] - rel[..., 1] * ud[:, None, 0]).max(1)
        torque = np.array([[W.MOMENT_ARM @ (uk * np.prod(W.muscle_force_scaling(xk)[3:], 0))
                            for xk, uk in zip(xi[:-1], ui)] for xi, ui in zip(x, u)])
        hn = z[f"{key}|hand_noisy"]
        end_sd = np.sqrt(((hn[:, :, -1] - hn[:, :, -1].mean(0)) ** 2).sum(-1).mean(0))
        ms[key] = dict(
            vpk=sp.max(1), tpk=sp.argmax(1) * dt * 1000, curv=100 * lat / AMPLITUDE,
            err=np.hypot(*(hand[:, -1] - tg).T), frac_neg=(u < 0).mean(),
            force=np.abs(u).sum((1, 2)) * dt, torque=torque,
            coact=np.abs(u).sum((1, 2)) / np.linalg.norm(torque, axis=2).sum(1),
            er_u=float(np.median([entropy_rank(np.linalg.svd(u[:, k] - u[:, k].mean(0), compute_uv=False))
                                  for k in range(1, K)])),
            er_tq=float(np.median([entropy_rank(np.linalg.svd(torque[:, k] - torque[:, k].mean(0),
                                                              compute_uv=False)) for k in range(1, K)])),
            er_time=np.array([entropy_rank(np.linalg.svd(u[:, :, i] - u[:, :, i].mean(), compute_uv=False))
                              for i in range(6)]),
            end_sd=end_sd)
    row = lambda name, fn: lines.append(f"{name:34s} {fn('push'):>16s} {fn('pull'):>17s}")
    row("vitesse pic (m/s)", lambda k: f"{ms[k]['vpk'].mean():.3f}")
    row("instant du pic (ms)", lambda k: f"{ms[k]['tpk'].mean():.0f}")
    row("courbure moyenne / max (%)", lambda k: f"{ms[k]['curv'].mean():.1f} / {ms[k]['curv'].max():.1f}")
    row("erreur finale sans bruit (cm)", lambda k: f"{ms[k]['err'].max():.3f}")
    row("dispersion finale avec bruit (cm)", lambda k: f"{ms[k]['end_sd'].mean():.2f}")
    row("fraction de commandes < 0", lambda k: f"{ms[k]['frac_neg']:.2f}")
    row("force totale intégrée", lambda k: f"{ms[k]['force'].mean():.3f}")
    row("force totale / couple", lambda k: f"{ms[k]['coact'].mean():.2f}")
    row("erank couple (à travers cibles)", lambda k: f"{ms[k]['er_tq']:.2f}")
    row("erank commandes musculaires", lambda k: f"{ms[k]['er_u']:.2f}")
    row("complexité temporelle par muscle", lambda k: " ".join(f"{v:.1f}" for v in ms[k]["er_time"]))
    d = np.linalg.norm(z["pull|hand"] - z["push|hand"], axis=2).max(1)
    lines.append(f"écart max des trajectoires sans bruit entre modèles : {d.mean():.2f} cm en moyenne, "
                 f"{d.max():.2f} cm au pire")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "wp1_table.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    return ms


def figure(z, ms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ang, tg = targets()
    t = np.linspace(0, DURATION, K + 1) * 1000
    tu = t[:-1]
    style = {"push": ("poussent et tirent (ILQG du WP1)", "#009E73"),
             "pull": ("tirent seulement (sigma_u)", "crimson")}
    names = ["épaule fléch.", "épaule ext.", "coude fléch.", "coude ext.", "biart. fléch.", "biart. ext."]
    cols = plt.cm.hsv(np.linspace(0, 1, N_T, endpoint=False))

    fig = plt.figure(figsize=(19, 15.5))
    top = fig.add_gridspec(1, 4, top=0.91, bottom=0.70, wspace=0.3)
    mid = fig.add_gridspec(2, 6, top=0.63, bottom=0.30, hspace=0.45, wspace=0.38)
    bot = fig.add_gridspec(1, 3, top=0.22, bottom=0.04, wspace=0.35)

    for j, key in enumerate(("push", "pull")):
        ax = fig.add_subplot(top[0, j])
        lab, col = style[key]
        hn = z[f"{key}|hand_noisy"]
        for rep in range(hn.shape[0]):
            for i in range(N_T):
                ax.plot(*hn[rep, i].T, color=col, lw=0.6, alpha=0.5)
        for i in range(N_T):
            ax.plot(*hn[:, i].mean(0).T, color="k", lw=1.8)
        ax.plot(tg[:, 0], tg[:, 1], "s", ms=13, mfc="w", mec="grey", mew=2.5, zorder=0)
        ax.plot(*START, "o", ms=13, mfc="w", mec="grey", mew=2.5, zorder=0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{lab}\n{hn.shape[0]} essais bruités par cible (noir : moyenne)", fontsize=9.5)

    for j, d in enumerate((0, 2, 4)):
        pass
    ax = fig.add_subplot(top[0, 2])
    for key, (lab, col) in style.items():
        sp = hand_speed(z[f"{key}|x_noisy"])
        for i in (0, 2, 4):
            m, s = sp[:, i].mean(0), sp[:, i].std(0)
            ax.plot(t, m, color=col, lw=2, ls=["-", "--", ":"][i // 2],
                    label=f"{lab.split(' (')[0]}, {int(np.degrees(ang[i]))}°")
            ax.fill_between(t, m - s, m + s, color=col, alpha=0.15, lw=0)
    ax.set_xlabel("temps (ms)")
    ax.set_ylabel("vitesse de la main (m/s)")
    ax.set_title("Vitesse de la main, moyenne ± SD (3 directions)", fontsize=9.5)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(top[0, 3])
    for key, (lab, col) in style.items():
        tq = ms[key]["torque"]
        for i in (0, 2, 4, 6):
            ax.plot(tu, tq[i, :, 0], color=cols[i], lw=1.6, ls="-" if key == "push" else "--")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("temps (ms)")
    ax.set_title("Couple épaule, 4 directions sans bruit\nplein : poussent+tirent ; tirets : tirent seul.",
                 fontsize=9.5)

    for r, key in enumerate(("push", "pull")):
        u = z[f"{key}|u"]
        for i in range(6):
            ax = fig.add_subplot(mid[r, i])
            for k in range(N_T):
                ax.plot(tu, u[k, :, i], color=cols[k], lw=1.4)
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_title(names[i], fontsize=9)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_ylabel(f"commande u\n({style[key][0].split(' (')[0]})", fontsize=9)
            if r == 1:
                ax.set_xlabel("temps (ms)", fontsize=8)

    for j, (q, lab) in enumerate((("vpk", "vitesse pic (m/s)"), ("coact", "force totale / couple"))):
        ax = fig.add_subplot(bot[0, j], projection="polar")
        for key, (name, col) in style.items():
            v = ms[key][q]
            ax.plot(np.append(ang, ang[0]), np.append(v, v[0]), color=col, lw=2,
                    ls="-" if key == "push" else "--", label=name)
        ax.set_ylim(0, None)
        ax.set_rlabel_position(200)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.tick_params(labelsize=7)
        ax.set_title(lab, fontsize=9.5, pad=14)
        if j == 0:
            ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(-0.45, -0.08), frameon=False)
    ax = fig.add_subplot(bot[0, 2])
    w = 0.38
    for j, (key, (name, col)) in enumerate(style.items()):
        ax.bar(np.arange(6) + (j - 0.5) * w, ms[key]["er_time"], w, color=col, label=name.split(" (")[0])
    ax.axhline(2, color="0.4", ls=":", lw=1)
    ax.set_xticks(np.arange(6), [n.replace(" ", "\n", 1) for n in names], fontsize=7.5)
    ax.set_ylabel("erank (cibles x temps)", fontsize=8.5)
    ax.set_title("Complexité temporelle de chaque muscle", fontsize=9.5)
    ax.legend(fontsize=7.5)

    fig.suptitle("ILQG du WP1 (bras + 6 muscles, sans réseau), centre-out 15 cm / 0,4 s depuis (0, 40) cm, "
                 "bruit moteur et délai 60 ms comme dans WP1\nseule change la contrainte « tirer "
                 "seulement » : u = 8·sigma_u(v + v*) au lieu de u = 8·2e⁻⁴·v", fontsize=12)
    out = os.path.join(HERE, "fig_wp1_pull_vs_pushpull.png")
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-sim", type=int, default=15)
    p.add_argument("--jobs", type=int, default=7)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.force or not os.path.exists(CACHE):
        simulate(a.num_sim, a.jobs)
    z = dict(np.load(CACHE))
    figure(z, report(z))


if __name__ == "__main__":
    main()
