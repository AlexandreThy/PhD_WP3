"""Does the direct model's inverted-U come from penalising muscle effort?
Re-run the direct-control sweep at several muscle-effort weights r1. If a smaller
r1 turns the inverted-U into a monotonic rise, effort was the cause."""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ilqg_direct as D

SIGMAS = [0.0, 1.0, 2.0, 3.0, 4.0]
R1S = [1e-5, 1e-7, 1e-9]
M_WORLDS = 3


def sweep(r1):
    # COLD start each sigma (u0=None) so every point is its own genuine optimum,
    # avoiding the warm-start freezing that made low-r1 curves look artificially
    # flat. Tighter, more iterations for reliable independent convergence.
    cc = []
    for sig in SIGMAS:
        D.set_env(0.0, sig, M_WORLDS)
        x, u, xt = D.simulate([0, 52], [0, 40], K=80, r1=r1, max_iter=250, tol=1e-6)
        cc.append(float(D.cocontraction_pairs(x).mean()))
    return np.array(cc)


def main():
    net = list(np.load(os.path.join(_HERE, "uncertainty_sweep.npz"), allow_pickle=True)["res"])
    net = [r for r in net if not r["nan"]]
    nsig = np.array([r["sigma"] for r in net]); ncc = np.array([r["cc"] for r in net])

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.plot(nsig, ncc, "o-", color="crimson", lw=2.6, ms=9,
            label="network, r1=1e-5 (reference)")
    cols = ["#4c72b0", "#55a868", "#8172b3"]
    for r1, c in zip(R1S, cols):
        cc = sweep(r1)
        mono = np.all(np.diff(cc) >= -1e-3)
        print("direct r1=%.0e :  %s   %s" % (r1, np.round(cc, 3),
                                             "MONOTONIC" if mono else "inverted-U"))
        ax.plot(SIGMAS, cc, "s--", color=c, lw=2, ms=7,
                label="direct, r1=%.0e%s" % (r1, "  (monotonic)" if mono else "  (∩)"))
    ax.set_xlabel("environmental uncertainty  σ")
    ax.set_ylabel("mean cocontraction index")
    ax.set_title("Direct control at different muscle-effort weights r1\n"
                 "(does lowering r1 restore monotonicity?)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(_HERE, "fig_direct_r1.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
