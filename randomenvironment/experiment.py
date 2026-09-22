"""Cocontraction vs. environmental uncertainty.

Solves the robust (M-worlds) reach for increasing uncertainty sigma (half-spread
of the uncertain curl-field gain, mean 0), WARM-STARTING each level from the
previous one so the solutions are continuous in sigma and free of convergence
noise. Plots cocontraction (and the endpoint spread it controls) vs uncertainty.
"""
import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

SIGMAS = [0.0, 1.0, 2.0, 3.0, 4.0]
M_WORLDS = 3
K = 80
CACHE = os.path.join(_HERE, "uncertainty_sweep.npz")


def run():
    import ilqg_robust as R
    res = []
    u_warm = None
    t0 = time.time()
    for sig in SIGMAS:
        R.set_env(mu=0.0, sigma=sig, m=M_WORLDS)
        x, u, xt = R.simulate([0, 52], [0, 40], K=K, max_iter=140, tol=1e-4, u0=u_warm)
        u_warm = u                                       # warm-start next sigma
        cc_t = R.cocontraction_pairs(x).mean(axis=-1)
        H = R.hands(x)                                   # (T, M, 2)
        ep = np.array([[H[-1, m, 0], H[-1, m, 1]] for m in range(M_WORLDS)])
        ep_std = float(np.hypot(*(ep.std(0))))           # endpoint spread across worlds
        err = float(np.mean([np.hypot(ep[m, 0], ep[m, 1] - 52) for m in range(M_WORLDS)]))
        res.append(dict(sigma=sig, cc_t=cc_t, cc=float(cc_t.mean()), hands=H,
                        ep_std=ep_std, err=err, nan=bool(not np.isfinite(x).all())))
        print("sigma=%4.1f  cocontraction=%.3f  endpoint spread=%.2f cm  "
              "avg reach err=%.3f cm%s" % (sig, res[-1]["cc"], ep_std, err,
                                           "  (NaN!)" if res[-1]["nan"] else ""))
    np.savez_compressed(CACHE, res=np.array(res, dtype=object))
    print("done %.0fs" % (time.time() - t0))
    return res


def plot(res=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if res is None:
        res = list(np.load(CACHE, allow_pickle=True)["res"])
    res = [r for r in res if not r["nan"]]
    sig = np.array([r["sigma"] for r in res])
    cc = np.array([r["cc"] for r in res])
    T = res[0]["cc_t"].shape[0]
    tc = np.linspace(0, 0.5, T) * 1000
    scols = plt.cm.viridis(np.linspace(0, 0.88, len(res)))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

    # (a) THE RESULT: cocontraction rises with uncertainty
    ax = axes[0]
    ax.plot(sig, cc, "o-", color="crimson", lw=2.4, ms=9)
    ax.set_xlabel("environmental uncertainty  σ")
    ax.set_ylabel("mean cocontraction index")
    ax.set_title("Cocontraction increases with\ndynamic uncertainty")
    ax.grid(alpha=0.3)

    # (b) it is the RESPONSE to endpoint spread the field would otherwise cause
    ax = axes[1]
    eps = np.array([r["ep_std"] for r in res])
    ax.plot(sig, eps, "s-", color="steelblue", lw=2.2, ms=8)
    ax.set_xlabel("environmental uncertainty  σ")
    ax.set_ylabel("residual endpoint spread across worlds (cm)")
    ax.set_title("Endpoint spread stays small\n(cocontraction absorbs the uncertainty)")
    ax.grid(alpha=0.3)

    # (c) cocontraction time course per uncertainty level
    ax = axes[2]
    for r, c in zip(res, scols):
        ax.plot(tc, r["cc_t"], color=c, lw=2, label="σ=%.0f" % r["sigma"])
    ax.set_xlabel("time (ms)"); ax.set_ylabel("cocontraction index")
    ax.set_title("Cocontraction time course\n(higher throughout for larger σ)")
    ax.legend(fontsize=8, ncol=2)

    fig.suptitle("Cocontraction as the optimal response to uncertain environment dynamics "
                 "(network + 6-muscle arm)", fontsize=12.5, y=1.02)
    fig.tight_layout()
    out = os.path.join(_HERE, "fig_uncertainty_cocontraction.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    plot(run())
