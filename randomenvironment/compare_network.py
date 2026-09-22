"""Compare cocontraction-vs-uncertainty WITH the network (ilqg_robust) and with
DIRECT muscle control (ilqg_direct). Runs the direct sweep (fast) and overlays it
on the cached network sweep."""
import os
import sys
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ilqg_direct as D

SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
M_WORLDS = 3


def run_direct():
    res = []
    uw = None
    t0 = time.time()
    for sig in SIGMAS:
        D.set_env(0.0, sig, M_WORLDS)
        x, u, xt = D.simulate([0, 52], [0, 40], K=80, max_iter=140, tol=1e-4, u0=uw)
        uw = u
        cc = float(D.cocontraction_pairs(x).mean())
        H = D.hands(x)
        err = float(np.mean([np.hypot(H[-1, m, 0], H[-1, m, 1] - 52) for m in range(M_WORLDS)]))
        res.append((sig, cc, err))
        print("direct  sigma=%4.1f  cocontraction=%.3f  reach err=%.3f cm" % (sig, cc, err))
    print("direct sweep: %.0fs total (%d solves)" % (time.time() - t0, len(SIGMAS)))
    return res


def main():
    dres = run_direct()
    dsig = np.array([r[0] for r in dres]); dcc = np.array([r[1] for r in dres])

    # cached network sweep
    net = list(np.load(os.path.join(_HERE, "uncertainty_sweep.npz"), allow_pickle=True)["res"])
    net = [r for r in net if not r["nan"]]
    nsig = np.array([r["sigma"] for r in net]); ncc = np.array([r["cc"] for r in net])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(nsig, ncc, "o-", color="crimson", lw=2.4, ms=9,
            label="with network (dim %d)" % (100 + 6 + 4 * M_WORLDS))
    ax.plot(dsig, dcc, "s--", color="steelblue", lw=2.2, ms=7,
            label="direct muscle control (dim %d)" % (6 + 4 * M_WORLDS))
    ax.set_xlabel("environmental uncertainty  σ")
    ax.set_ylabel("mean cocontraction index")
    ax.set_title("Cocontraction vs uncertainty:\nnetwork vs direct muscle control")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(_HERE, "fig_network_vs_direct.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print("network cocontraction range: %.3f -> %.3f  (+%.0f%%)"
          % (ncc[0], ncc[-1], 100 * (ncc[-1] / ncc[0] - 1)))
    print("direct  cocontraction range: %.3f -> %.3f  (+%.0f%%)"
          % (dcc[0], dcc[-1], 100 * (dcc[-1] / dcc[0] - 1)))


if __name__ == "__main__":
    main()
