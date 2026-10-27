"""erank of network activity with the 6-muscle body: linear vs nonlinear.

Same question as erank.py (torque model), now with the six Lillicrap & Scott
muscles (rectified, sigma_u of muscle_plant.py). All three models share the
network, the random 6x100 readout, the joint-space cost of compare3.py and the
24 targets (12 cm from M.CENTER_XY):

    lin6c  linearized arm (inertia frozen at the start posture, no Coriolis)
           + 6 muscles with F-L/V FROZEN at the start posture
           (phi_i = f_l(l_i0) f_fv(l_i0, 0), a constant per muscle: 0.70-1.00)
    nl6c   nonlinear arm + 6 muscles, F-L/V frozen         (arm nonlinearity only)
    nl6    nonlinear arm + 6 muscles + full F-L/V          (the full body)
    lin6u  as lin6c, but muscles may push as well as pull: the activation drive
           is the linear branch of sigma_u extended to negative values,
           2 e^-4 (Wout r), with rest activation 0. The whole system is then
           linear, so its activity must be affine in the target.

Freezing F-L/V at the operating point is the muscle analogue of freezing the
inertia at THETA2_REF. In lin6c / nl6c / nl6 the muscles stay rectified (they
can only pull), so lin6c is not linear-quadratic; lin6u removes that last
nonlinearity.

    python analyses_claude/fig_erank_muscles.py [--jobs 7] [--analyse]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compare3 as C  # noqa: E402
from erank import measures, even_odd, entropy_rank  # noqa: E402

M = C.M
OUT = os.path.join(HERE, "sims")
RADIUS, N_DIR, K = 12.0, 24, 100
REGIMES = {"12cm_050s": 0.50, "12cm_040s": 0.40, "12cm_030s": 0.30}
MODELS = {"lin6u": ("bras linéarisé, F-L/V figée,\nmuscles poussent et tirent", "0.15"),
          "lin6c": ("bras linéarisé,\nF-L/V figée", "0.55"),
          "nl6c": ("bras non linéaire,\nF-L/V figée", "#3182bd"),
          "nl6": ("bras non linéaire,\nF-L/V complète", "crimson")}


class FrozenFLV:
    """flv='frozen': tension = F_max a phi0, phi0 = F-L/V at the start posture."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        l0, v0 = self.muscle_state(np.asarray(M.CENTER_JOINTS, float), np.zeros(2))
        self.phi0 = M.flv(l0, v0)[0]

    def torque(self, s, a):
        if self.flv != "frozen":
            return super().torque(s, a)
        tau = self.moment_m @ (M.F_MAX * a * self.phi0)
        z = np.zeros((2, 2))
        return tau, (z, z, self.moment_m * (M.F_MAX * self.phi0))


class PushPull:
    """Muscles may push: adot = (g Wout r - a)/tau_act with g = 2 e^-4, the slope of
    sigma_u's linear branch (which passes through 0), for all signs of the drive."""
    GAIN = 2.0 * np.exp(2.0 * 0.5 - 5.0)

    def f(self, x, u):
        out = super().f(x, u)
        d = self.Wout @ x[:self.N]
        out[self.i_a] += (self.GAIN * d - M.sigma_u(d)) / M.TAU_ACT
        return out

    def fx(self, x, u):
        A = super().fx(x, u)
        A[self.i_a, :self.N] = self.GAIN * self.Wout / M.TAU_ACT
        return A

    def initial_state(self, start_xy, r0=None):
        x = super().initial_state(start_xy, r0)
        x[self.i_a] = 0.0
        return x


class Frozen(FrozenFLV, C.JointPlant):
    pass


class FrozenLinearPushPull(PushPull, FrozenFLV, C.JointLinearPlant):
    pass


class FrozenLinear(FrozenFLV, C.JointLinearPlant):
    pass


def make(model, network=2, seed=0):
    W = M.load_network(network, path=os.path.join(C.MP.PROJ, "gaussian_networks.hdf5"))
    Wout = M.random_readout(6, W.shape[0], np.random.default_rng(seed))
    if model == "lin6u":
        return FrozenLinearPushPull(W, Wout, muscles="bi", flv="frozen")
    if model == "lin6c":
        return FrozenLinear(W, Wout, muscles="bi", flv="frozen")
    if model == "nl6c":
        return Frozen(W, Wout, muscles="bi", flv="frozen")
    return C.JointPlant(W, Wout, muscles="bi", flv=True)


def targets():
    ang = np.linspace(0, 2 * np.pi, N_DIR, endpoint=False)
    xy = np.array(M.CENTER_XY) + RADIUS * np.stack([np.cos(ang), np.sin(ang)], 1)
    return ang, xy, np.array([M.inverse_kinematics(*p) for p in xy])


def solve_one(job):
    rg, model, i = job
    plant = make(model)
    _, xy, jt = targets()
    x, u, info = M.solve(plant, M.CENTER_XY, jt[i], duration=REGIMES[rg], K=K, **C.SOLVER)
    hand = plant.hand(x)
    return job, x, hand, float(np.hypot(*(hand[-1] - xy[i]))), info["iters"], plant.N


def path(rg, m):
    return os.path.join(OUT, f"erank6_{rg}_{m}.npz")


def simulate(jobs):
    os.makedirs(OUT, exist_ok=True)
    ang, xy, jt = targets()
    todo = [(rg, m) for rg in REGIMES for m in MODELS if not os.path.exists(path(rg, m))]
    work = [(rg, m, i) for rg, m in todo for i in range(N_DIR)]
    res = {c: [None] * N_DIR for c in todo}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for job, x, hand, err, it, N in ex.map(solve_one, work, chunksize=1):
            res[job[:2]][job[2]] = (x, hand, err, it, N)
    for (rg, m), lst in res.items():
        err = np.array([r[2] for r in lst])
        np.savez_compressed(path(rg, m), states=np.stack([r[0] for r in lst]),
                            hand=np.stack([r[1] for r in lst]), ang=ang, targ=xy,
                            targ_joint=jt, N=lst[0][4], duration=REGIMES[rg], K=K,
                            endpoint_error=err, iters=np.array([r[3] for r in lst]))
        print(f"{rg} {m:5s} max endpoint err {err.max():.2e} cm, "
              f"iters {min(r[3] for r in lst)}-{max(r[3] for r in lst)}")
    print(f"simulated in {(time.time() - t0) / 60:.1f} min")


# -----------------------------------------------------------------------------
# analysis
# -----------------------------------------------------------------------------
def load(rg, m):
    d = np.load(path(rg, m))
    return d["states"][:, :, :int(d["N"])], d


def non_affine(r, tj):
    """Fraction of condition-dependent variance NOT explained by
    r(theta, t) = c(t) + G(t) theta_target  (0 for a linear-quadratic model)."""
    X = np.column_stack([np.ones(len(tj)), tj])
    sse = sst = 0.0
    for k in range(1, r.shape[1]):
        Y = r[:, k, :]
        b, *_ = np.linalg.lstsq(X, Y, rcond=None)
        sse += ((Y - X @ b) ** 2).sum()
        sst += ((Y - Y.mean(0)) ** 2).sum()
    return sse / sst


def peak_speed(d):
    v = np.gradient(d["hand"], float(d["duration"]) / K, axis=1) / 100.0
    return float(np.linalg.norm(v, axis=2).max())


def erank_t(r):
    return np.array([entropy_rank(np.linalg.svd(r[:, k] - r[:, k].mean(0), compute_uv=False))
                     for k in range(1, r.shape[1])])


def analyse():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for rg in REGIMES:
        for m in MODELS:
            r, d = load(rg, m)
            ev, od = even_odd(r)
            rows.append(dict(rg=rg, m=m, all=measures(r), even=measures(ev)[0],
                             odd=measures(od)[0], naff=non_affine(r, d["targ_joint"]),
                             vpk=peak_speed(d), err=float(d["endpoint_error"].max())))

    hdr = (f"{'régime':10s} {'modèle':6s} {'v pic':>6s} | {'erank_RV':>8s} {'erank_var':>9s} "
           f"{'PR':>5s} | {'pair':>5s} {'impair':>6s} | {'non affine':>10s} | {'err (cm)':>8s}")
    lines = [hdr, "-" * len(hdr)]
    for w in rows:
        a = w["all"]
        lines.append(f"{w['rg']:10s} {w['m']:6s} {w['vpk']:6.2f} | {a[0]:8.2f} {a[1]:9.2f} "
                     f"{a[2]:5.2f} | {w['even']:5.2f} {w['odd']:6.2f} | "
                     f"{100 * w['naff']:9.2f}% | {w['err']:8.1e}")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(HERE, "erank_muscles_table.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")

    regs = list(REGIMES)
    lab = [f"12 cm / {REGIMES[g]:.2f} s\n(pic {np.mean([w['vpk'] for w in rows if w['rg'] == g]):.2f} m/s)"
           .replace(".", ",") for g in regs]
    x = np.arange(len(regs))
    wbar = 0.2
    nm = len(MODELS)
    get = lambda g, m: [w for w in rows if w["rg"] == g and w["m"] == m][0]

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for j, (m, (name, col)) in enumerate(MODELS.items()):
        off = (j - (nm - 1) / 2) * wbar
        ax[0].bar(x + off, [get(g, m)["all"][0] for g in regs], wbar, color=col,
                  label=name.replace("\n", " "))
        ax[0].plot(x + off, [get(g, m)["all"][1] for g in regs], "k_", ms=12, mew=2,
                   label="erank_var" if j == 0 else None)
    ax[0].set_xticks(x, lab, fontsize=8)
    ax[0].set_ylabel("erank")
    ax[0].set_title("A. erank de l'activité (barres : Roy & Vetterli)")
    ax[0].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    rg_fast = "12cm_030s"
    for m, (name, col) in MODELS.items():
        r, d = load(rg_fast, m)
        t = np.linspace(0, float(d["duration"]), K + 1)[1:] * 1000
        ax[1].plot(t, erank_t(r), color=col, lw=2, ls="--" if m.startswith("lin") else "-",
                   label=name.replace("\n", " "))
    ax[1].set_xlabel("temps (ms)")
    ax[1].set_ylabel("erank à travers les 24 cibles")
    ax[1].set_title("B. erank instantané (12 cm / 0,30 s)")
    ax[1].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    for j, (m, (name, col)) in enumerate(MODELS.items()):
        off = (j - (nm - 1) / 2) * wbar
        ax[2].bar(x + off - wbar / 4, [get(g, m)["odd"] for g in regs], wbar / 2, color=col,
                  alpha=0.4)
        ax[2].bar(x + off + wbar / 4, [get(g, m)["even"] for g in regs], wbar / 2, color=col)
    ax[2].bar([0], [0], color="0.5", alpha=0.4, label="impair (clair)")
    ax[2].bar([0], [0], color="0.5", label="pair (foncé)")
    ax[2].set_xticks(x, lab, fontsize=8)
    ax[2].set_ylabel("erank (Roy & Vetterli)")
    ax[2].set_title("C. erank des composantes paire et impaire")
    ax[2].legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Corps à 6 muscles : même réseau, même readout, même coût, mêmes cibles ; "
                 "seuls le bras, la F-L/V et la rectification changent", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_erank_muscles.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=7)
    p.add_argument("--analyse", action="store_true")
    a = p.parse_args()
    if not a.analyse:
        simulate(a.jobs)
    analyse()
