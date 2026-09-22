"""Verify the network + 6-muscle center-out model and quantify cocontraction.

Cocontraction = simultaneous activation of antagonist muscles that produces no
net joint torque. Two views:
  * global: fraction of the activation vector in null(A_mom)  (index in [0,1];
    0 = pure torque-producing drive, ~0.82 = random 6-vector);
  * per pair: the three antagonist pairs (shoulder mono, elbow mono, bi-artic.),
    decomposed into a net-torque drive and a co-activation (cocontraction) part.
"""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ilqg_muscle as M

PAIRS = [("shoulder", 0, 1), ("elbow", 2, 3), ("bi-articular", 4, 5)]


def load():
    d = np.load(os.path.join(_HERE, "centerout_muscle.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def analyse():
    d = load()
    states, hand, ang, targ = d["states"], d["hand"], d["ang"], d["targ"]
    N = int(d["N"]); dur = float(d["duration"])
    nd, T = states.shape[0], states.shape[1]
    tc = np.linspace(0, dur, T) * 1000
    cols = plt.cm.hsv(ang / (2 * np.pi))

    a = M.activations(states)                      # (nd, T, 6)  muscle activations
    ci = M.cocontraction_index(states)             # (nd, T)
    # minimum-norm activation that yields the SAME net torque -> zero-cocontraction
    # reference; the excess over it is the co-activation drive.
    torque_act = a @ (M.A_pinv @ M.A_mom).T        # projection onto row space
    coact = a - torque_act                         # null-space (cocontraction) part

    fig = plt.figure(figsize=(13.5, 9.5))

    # (a) hand paths -- verification
    ax = fig.add_subplot(2, 3, 1)
    for i in range(nd):
        ax.plot(hand[i, :, 0], hand[i, :, 1], color=cols[i], lw=2)
        ax.plot(targ[i, 0], targ[i, 1], "s", color=cols[i], ms=8, mec="k", mew=0.5)
    ax.plot(d["center"][0], d["center"][1], "o", color="k", ms=7)
    ax.set_aspect("equal"); ax.set_title("Center-out reaches (verification)")
    ax.set_xticks([]); ax.set_yticks([])
    err = np.hypot(hand[:, -1, 0] - targ[:, 0], hand[:, -1, 1] - targ[:, 1])
    ax.text(0.5, -0.06, "max endpoint error %.3f cm" % err.max(),
            transform=ax.transAxes, ha="center", fontsize=9)

    # (b) the 6 muscle activations for one exemplar target
    ax = fig.add_subplot(2, 3, 2)
    ex = 0
    names = ["sh flex", "sh ext", "el flex", "el ext", "bi flex", "bi ext"]
    pal = ["#d62728", "#ff9896", "#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"]
    for m in range(6):
        ax.plot(tc, a[ex, :, m], color=pal[m], lw=1.8, label=names[m])
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("activation")
    ax.set_title("Muscle activations (target %.0f deg)" % np.degrees(ang[ex]))
    ax.legend(fontsize=7, ncol=3)

    # (c) cocontraction index over time, all targets
    ax = fig.add_subplot(2, 3, 3)
    for i in range(nd):
        ax.plot(tc, ci[i], color=cols[i], lw=1.3, alpha=0.9)
    ax.plot(tc, ci.mean(0), color="k", lw=2.5, label="mean")
    ax.axhline(0.816, color="0.5", ls="--", lw=1, label="random 6-vector")
    ax.axhline(0, color="0.5", ls=":", lw=1, label="min-effort (no cocontr.)")
    ax.set_ylim(0, 1); ax.set_xlabel("time (ms)")
    ax.set_ylabel("cocontraction index (null-space fraction)")
    ax.set_title("Cocontraction over time"); ax.legend(fontsize=7.5)

    # (d) classic per-pair antagonist cocontraction index 2*min/(sum), over time
    ax = fig.add_subplot(2, 3, 4)
    cci = M.cocontraction_pairs(states)            # (nd, T, 3)
    pair_pal = ["#d62728", "#1f77b4", "#2ca02c"]
    for p, (name, _, _) in enumerate(PAIRS):
        mp = cci[:, :, p].mean(0)
        ax.plot(tc, mp, color=pair_pal[p], lw=2, label=name)
    ax.set_ylim(0, 1); ax.set_xlabel("time (ms)")
    ax.set_ylabel("cocontraction index  2·min/(sum)")
    ax.set_title("Antagonist cocontraction per pair"); ax.legend(fontsize=8)

    # (e) cocontraction index per reach direction
    ax = fig.add_subplot(2, 3, 5, projection="polar")
    ci_dir = ci.mean(1)
    ax.bar(ang, ci_dir, width=2 * np.pi / nd * 0.9, color=cols, edgecolor="k",
           alpha=0.85, linewidth=0.5)
    ax.set_title("Cocontraction index\nby reach direction", fontsize=10)
    ax.set_yticklabels([])

    # (f) fraction of activation sign (physical muscles are >= 0)
    ax = fig.add_subplot(2, 3, 6)
    frac_pos = (a > 0).mean(axis=(0, 1))
    ax.bar(np.arange(6), frac_pos * 100, color=pal, edgecolor="k", lw=0.5)
    ax.axhline(50, color="0.5", ls="--", lw=1)
    ax.set_xticks(np.arange(6)); ax.set_xticklabels(names, rotation=35, fontsize=8)
    ax.set_ylabel("% of time activation > 0")
    ax.set_title("Activation sign (physical muscles: always > 0)")

    fig.suptitle("Network + 6-muscle center-out: cocontraction analysis", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(_HERE, "fig_cocontraction.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print("  max endpoint error         : %.4f cm" % err.max())
    print("  mean cocontraction index    : %.3f  (random ~0.82, min-effort 0)" % ci.mean())
    print("  peak cocontraction index    : %.3f" % ci.max())
    print("  mean |activation|           : %.4f   peak %.4f"
          % (np.abs(a).mean(), np.abs(a).max()))
    print("  activation > 0 fraction     : %.0f%%  (physical muscles: 100%%)"
          % (100 * (a > 0).mean()))
    cci = M.cocontraction_pairs(states)
    for p, (name, _, _) in enumerate(PAIRS):
        print("  %-13s cocontraction index (mean/peak): %.2f / %.2f"
              % (name, cci[:, :, p].mean(), cci[:, :, p].max()))


if __name__ == "__main__":
    analyse()
