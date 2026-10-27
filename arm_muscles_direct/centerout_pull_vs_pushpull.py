"""Center-out with the arm driven directly by six muscles: iLQG with and without
the 'muscles can only pull' constraint (see arm_muscles.py).

Task as WP1/Final_Code (centerout_trajectories.py): start (0, 40) cm, 15 cm,
0.4 s; here 16 targets, deterministic optimal solutions (no motor noise, no
sensory delay). Both models share the arm, the moment arms, the F-L/V, the
activation time constant, the cost and the targets; only the activation law
differs.

    python arm_muscles_direct/centerout_pull_vs_pushpull.py [--force]

Writes centerout_sims.npz, centerout_table.txt and fig_centerout_pull_vs_pushpull.png
in this folder.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import arm_muscles as AM  # noqa: E402

START = (0.0, 40.0)
AMPLITUDE, DURATION, K, N_T = 15.0, 0.4, 80, 16
ANG = np.linspace(0, 2 * np.pi, N_T, endpoint=False)
CACHE = os.path.join(HERE, "centerout_sims.npz")
MODELS = {"push": ("poussent et tirent", "0.15"), "pull": ("tirent seulement", "crimson")}
NAMES = ["épaule fléch.", "épaule ext.", "coude fléch.", "coude ext.", "biart. fléch.", "biart. ext."]


def targets():
    return np.array(START) + AMPLITUDE * np.stack([np.cos(ANG), np.sin(ANG)], 1)


def simulate():
    out = {}
    for key in MODELS:
        arm = AM.ArmMuscles(pull_only=(key == "pull"), start_xy=START)
        X, U, info = [], [], []
        for tg in targets():
            x, u, inf = AM.solve(arm, tg, duration=DURATION, K=K)
            X.append(x)
            U.append(u)
            info.append((inf["cost"], inf["iters"], inf["endpoint_error"]))
        X = np.array(X)
        out[f"{key}|x"] = X
        out[f"{key}|u"] = np.array(U)
        out[f"{key}|hand"] = arm.hand(X)
        out[f"{key}|torque"] = np.array([[arm.torque(xk) for xk in xi] for xi in X])
        out[f"{key}|tension"] = np.array([[AM.M.F_MAX * xk[arm.i_a] * phi(arm, xk) for xk in xi]
                                          for xi in X])
        out[f"{key}|info"] = np.array(info)
        out[f"{key}|u_rest"] = arm.u_rest
        print(f"{key}: iters {int(min(i[1] for i in info))}-{int(max(i[1] for i in info))}, "
              f"max endpoint err {max(i[2] for i in info):.1e} cm")
    np.savez_compressed(CACHE, **out)


def phi(arm, xk):
    """F-L/V factor of each muscle at state xk."""
    l, v = arm.p.muscle_state(xk[:2], xk[2:4])
    return AM.M.flv(l, v)[0]


# -----------------------------------------------------------------------------
# measures
# -----------------------------------------------------------------------------
def entropy_rank(s):
    s = s[s > 0]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def measures(z, key):
    dt = DURATION / K
    hand = z[f"{key}|hand"]
    v = np.gradient(hand, dt, axis=1) / 100.0
    speed = np.linalg.norm(v, axis=2)
    u_dir = np.stack([np.cos(ANG), np.sin(ANG)], 1)
    rel = hand - np.array(START)
    lateral = np.abs(rel[..., 0] * u_dir[:, None, 1] - rel[..., 1] * u_dir[:, None, 0])
    a = z[f"{key}|x"][:, :, 4:]
    T = z[f"{key}|tension"]
    tq = z[f"{key}|torque"]
    across = lambda Y: float(np.median([entropy_rank(np.linalg.svd(Y[:, k] - Y[:, k].mean(0),
                                                                   compute_uv=False))
                                        for k in range(1, Y.shape[1])]))
    return dict(
        speed=speed, vpk=speed.max(1), tpk=speed.argmax(1) * dt * 1000,
        curv=100 * lateral.max(1) / AMPLITUDE,
        # co-activation: total muscle tension per unit of joint torque, over the reach
        coact=np.abs(T).sum((1, 2)) / np.linalg.norm(tq, axis=2).sum(1),
        effort=np.sum((z[f"{key}|u"] - z[f"{key}|u_rest"]) ** 2, (1, 2)) * dt,
        er_torque=across(tq), er_act=across(a),
        er_time=np.array([entropy_rank(np.linalg.svd(a[:, :, i] - a[:, :, i].mean(),
                                                     compute_uv=False)) for i in range(6)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="re-simulate even if cached")
    a = p.parse_args()
    if a.force or not os.path.exists(CACHE):
        simulate()
    z = dict(np.load(CACHE))
    me = {k: measures(z, k) for k in MODELS}

    lines = [f"centre-out, départ {START} cm, {AMPLITUDE:.0f} cm, {DURATION} s, {N_T} cibles",
             f"{'':22s} {'poussent+tirent':>16s} {'tirent seulement':>17s}"]
    row = lambda name, f: lines.append(f"{name:22s} {f('push'):>16s} {f('pull'):>17s}")
    row("vitesse pic (m/s)", lambda k: f"{me[k]['vpk'].mean():.3f} [{me[k]['vpk'].min():.2f}-{me[k]['vpk'].max():.2f}]")
    row("instant du pic (ms)", lambda k: f"{me[k]['tpk'].mean():.0f} [{me[k]['tpk'].min():.0f}-{me[k]['tpk'].max():.0f}]")
    row("courbure (% amplitude)", lambda k: f"{me[k]['curv'].mean():.2f} [max {me[k]['curv'].max():.2f}]")
    row("tension / couple", lambda k: f"{me[k]['coact'].mean():.1f} [{me[k]['coact'].min():.1f}-{me[k]['coact'].max():.1f}]")
    row("effort (u-u*)^2", lambda k: f"{me[k]['effort'].mean():.3f}")
    row("erank couple", lambda k: f"{me[k]['er_torque']:.2f}")
    row("erank activations", lambda k: f"{me[k]['er_act']:.2f}")
    row("complexité temporelle", lambda k: " ".join(f"{v:.1f}" for v in me[k]["er_time"]))
    d = np.linalg.norm(z["pull|hand"] - z["push|hand"], axis=2).max(1)
    lines.append(f"écart max des trajectoires de main entre modèles : {d.mean():.2f} cm en moyenne, "
                 f"{d.max():.2f} cm au pire")
    dtq = np.linalg.norm(z["pull|torque"] - z["push|torque"], axis=2).max(1) / \
        np.linalg.norm(z["push|torque"], axis=2).max(1)
    lines.append(f"écart max des couples articulaires (relatif au pic) : {100 * dtq.mean():.1f} % en "
                 f"moyenne, {100 * dtq.max():.1f} % au pire")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "centerout_table.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    figure(z, me)


def figure(z, me):
    t = np.linspace(0, DURATION, K + 1) * 1000
    cols = plt.cm.hsv(np.linspace(0, 1, N_T, endpoint=False))
    show = np.arange(0, N_T, 2)
    fig = plt.figure(figsize=(19, 16))
    top = fig.add_gridspec(1, 4, top=0.90, bottom=0.71, wspace=0.35, width_ratios=[1.1, 1.3, 1, 1])
    mid = fig.add_gridspec(2, 6, top=0.64, bottom=0.30, hspace=0.45, wspace=0.38)
    bot = fig.add_gridspec(1, 4, top=0.22, bottom=0.03, wspace=0.45)

    ax = fig.add_subplot(top[0, 0])
    for key, (lab, col) in MODELS.items():
        for i in range(N_T):
            ax.plot(*z[f"{key}|hand"][i].T, color=col, lw=2 if key == "pull" else 3.2,
                    ls="-" if key == "push" else "--", label=lab if i == 0 else None)
    tg = targets()
    ax.plot(tg[:, 0], tg[:, 1], "s", ms=11, mfc="w", mec="grey", mew=2, zorder=0)
    ax.plot(*START, "o", ms=11, mfc="w", mec="grey", mew=2, zorder=0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    ax.set_title("Trajectoires de la main", fontsize=10)

    ax = fig.add_subplot(top[0, 1])
    for key, (lab, col) in MODELS.items():
        for i in range(N_T):
            ax.plot(t, me[key]["speed"][i], color=col, lw=1.2, alpha=0.8,
                    ls="-" if key == "push" else "--", label=lab if i == 0 else None)
    ax.set_xlabel("temps (ms)")
    ax.set_ylabel("vitesse de la main (m/s)")
    ax.set_title("Profils de vitesse (16 directions)", fontsize=10)
    ax.legend(fontsize=8)

    for j, (jn, ji) in enumerate((("épaule", 0), ("coude", 1))):
        ax = fig.add_subplot(top[0, 2 + j])
        for i in show[::2]:
            for key, (lab, _) in MODELS.items():
                ax.plot(t, z[f"{key}|torque"][i, :, ji], color=cols[i], lw=1.6,
                        ls="-" if key == "push" else "--")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlabel("temps (ms)")
        ax.set_title(f"Couple {jn} (N·m), 4 directions\nplein : poussent+tirent ; tirets : tirent seul.",
                     fontsize=8.5)

    for r, key in enumerate(MODELS):
        a = z[f"{key}|x"][:, :, 4:]
        for i in range(6):
            ax = fig.add_subplot(mid[r, i])
            for k in show:
                ax.plot(t, a[k, :, i], color=cols[k], lw=1.4)
            ax.axhline(0, color="0.6", lw=0.6)
            ax.axvline(t[int(me[key]["speed"].mean(0).argmax())], color="0.4", ls=":", lw=0.8)
            ax.set_title(NAMES[i], fontsize=9)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_ylabel(f"activation\n({MODELS[key][0]})", fontsize=9)
            if r == 1:
                ax.set_xlabel("temps (ms)", fontsize=8)

    for j, (q, lab) in enumerate((("vpk", "vitesse pic (m/s)"), ("curv", "courbure (% amplitude)"),
                                  ("coact", "tension totale / couple"))):
        ax = fig.add_subplot(bot[0, j], projection="polar")
        for key, (name, col) in MODELS.items():
            v = me[key][q]
            ax.plot(np.append(ANG, ANG[0]), np.append(v, v[0]), color=col, lw=2,
                    ls="-" if key == "push" else "--", label=name)
        ax.set_title(lab, fontsize=9, pad=14)
        ax.tick_params(labelsize=7)
        ax.set_rlabel_position(200)
        ax.set_ylim(0, None)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        if j == 0:
            ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(-0.35, 1.18))
    ax = fig.add_subplot(bot[0, 3])
    w = 0.38
    for j, (key, (name, col)) in enumerate(MODELS.items()):
        ax.bar(np.arange(6) + (j - 0.5) * w, me[key]["er_time"], w, color=col, label=name)
    ax.axhline(2, color="0.4", ls=":", lw=1)
    ax.set_xticks(np.arange(6), [n.replace(" ", "\n", 1) for n in NAMES], fontsize=7)
    ax.set_ylabel("erank (cibles x temps)", fontsize=8)
    ax.set_title("Complexité temporelle\nde chaque muscle", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("Bras seul piloté directement par 6 muscles (iLQG, sans réseau) : centre-out "
                 f"{AMPLITUDE:.0f} cm / {DURATION:.1f} s depuis {START} cm\n"
                 "même bras, mêmes muscles, même F-L/V, même coût ; seule change la contrainte "
                 "« tirer seulement » (sigma_u)", fontsize=12)
    out = os.path.join(HERE, "fig_centerout_pull_vs_pushpull.png")
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
