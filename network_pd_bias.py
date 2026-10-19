"""Can the recurrent network, by itself, bias the preferred directions (PDs)?

Every plant here is LINEAR (2-D point mass, or the frozen-inertia arm of
`iLQG_Combined` with LINEARIZE=True), and so is the network `r_dot = J r + u`.
With a quadratic cost the iLQG optimum is then available in closed form, so we
solve it exactly instead of iterating (`solve`, validated against the iLQG cache
`data/centerout_lin24.npz` in `validate`: the exact solution has a lower cost on
all 24 targets, iLQG at tol=1e-4 sits 0.06 % above it). That makes it cheap to
sweep all 50 networks of `gaussian_networks.hdf5` and to manipulate J.

Closed form. Euler-discretised exactly as iLQG (`x+ = x + dt f`), with terminal
output y = C x_K (position error and velocity) and effort sum_k dt r1 |u_k|^2/2,

    u_k = Phi_k^T lam,   Phi_k = C A^(K-1-k) B,   lam = (a I + Qf Phi Phi^T)^-1 Qf y*

i.e. the input to every node is the readout propagated backwards through the
network (the costate). Everything is linear, so r(theta, t) = G(t) d(theta) with
d the target offset: each node's PD is the angle of its row of G.

Analytic predictor (static minimum-energy argument, no simulation): the
cheapest network state producing an output force y is

    r = P Wout^T Gamma^-1 y,     Gamma = Wout P Wout^T  (2x2 "output Gramian")

with P the network's controllability Gramian. Node i's predicted tuning vector
is row i of  P Wout^T Gamma^-1, composed for the arm with the frozen-arm map
from hand direction to torque (M J_hand^-1).

All networks of the file share ONE readout (rows 2 and 6 of readout_*), nearly
isotropic (cond(Wout Wout^T) = 1.13), so network-to-network differences come
from J alone.

Run:  python network_pd_bias.py          (~30 s, writes fig_network_pd_bias.png)
"""
import os

import h5py
import numpy as np

import centerout as CO

_HERE = os.path.dirname(os.path.abspath(__file__))
HDF = os.path.join(_HERE, "gaussian_networks.hdf5")
GAINS = ["0.4", "0.6", "0.7", "0.8", "1.1"]
TAU_NET = 0.02
TAU_ACT = 0.06

# task, as in centerout.py / analysis defaults (12 cm, 0.5 s, 100 steps)
DURATION, K = 0.5, 100
RADIUS = 12.0
CENTER = (0.0, 40.0)
N_DIR = 16
R1 = 1e-5
W1_ARM, W2_ARM = 1e4, 10.0           # joint space [rad], as centerout.py
W1_PM, W2_PM = 1e4, 10.0             # hand space [m] for the point mass
M_PM, B_PM = 1.5, 0.1                # isotropic point mass [kg], viscosity
EPOCHS = ["onset-peak", "peak-end", "mean"]


# -----------------------------------------------------------------------------
# networks
# -----------------------------------------------------------------------------
def load_networks():
    """{gain: [J_0..J_9]} and the (shared) 2x100 readout."""
    nets = {}
    with h5py.File(HDF, "r") as f:
        for g in GAINS:
            nets[g] = [f[f"{g}/networks/network_{i}"][()] for i in range(10)]
        Wout = f["0.8/readouts/readout_2"][()][[2, 6]]
    return nets, Wout


def random_rotation(J, seed):
    """Q J Q^T: same eigenvalues and non-normality, new alignment with Wout."""
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=J.shape))
    return Q @ J @ Q.T


# -----------------------------------------------------------------------------
# plants: 6x6 block on q = [pos(2), vel(2), force(2)], deviation coordinates
# -----------------------------------------------------------------------------
def arm_block(Minv, visc):
    A = np.zeros((6, 6))
    A[0, 2] = A[1, 3] = 1.0
    A[2:4, 2:4] = -Minv @ visc
    A[2:4, 4:6] = Minv
    A[4, 4] = A[5, 5] = -1.0 / TAU_ACT
    return A


def inertia(th2):
    import iLQG_Combined as M
    return np.array([[M.a1 + 2 * M.a2 * np.cos(th2), M.a3 + M.a2 * np.cos(th2)],
                     [M.a3 + M.a2 * np.cos(th2), M.a3]])


def point_mass():
    return arm_block(np.eye(2) / M_PM, B_PM * np.eye(2)), (W1_PM, W2_PM)


def linear_arm(center=CENTER):
    import iLQG_Combined as M
    th2 = M.compute_angles_from_cartesian(*center)[1]
    return arm_block(np.linalg.inv(inertia(th2)), M.Viscous), (W1_ARM, W2_ARM)


def directions(n_dir=N_DIR):
    return np.linspace(0, 2 * np.pi, n_dir, endpoint=False)


def arm_offsets(ang, center=CENTER, radius=RADIUS):
    """Joint-angle offsets [rad] of the center-out targets (exact IK)."""
    import iLQG_Combined as M
    th0 = np.array(M.compute_angles_from_cartesian(*center))
    tg = [M.compute_angles_from_cartesian(center[0] + radius * np.cos(a),
                                          center[1] + radius * np.sin(a)) for a in ang]
    return np.array(tg) - th0


def pm_offsets(ang, radius=RADIUS):
    return np.stack([np.cos(ang), np.sin(ang)], 1) * radius / 100.0


# -----------------------------------------------------------------------------
# exact finite-horizon LQ solve (same discretisation and cost as iLQG)
# -----------------------------------------------------------------------------
def solve(J, Wout, plant, offsets, duration=DURATION, K=K, r1=R1):
    """Optimal trajectories for target offsets (n, 2).

    Returns r (n, K+1, N) and q (n, K+1, 6)."""
    A_arm, (w1, w2) = plant
    N = J.shape[0]
    n = N + 6
    dt = duration / K
    F = np.zeros((n, n))
    F[:N, :N] = J
    F[N:, N:] = A_arm
    F[-2:, :N] = Wout / TAU_ACT
    Ad = np.eye(n) + dt * F
    C = np.zeros((4, n))
    C[np.arange(4), N + np.arange(4)] = 1.0
    Qf = np.diag([w1, w1, w2, w2])

    Phi = np.zeros((K, 4, N))          # Phi_k = C Ad^(K-1-k) Bd,  Bd = dt [I; 0]
    R = C.copy()
    for k in range(K - 1, -1, -1):
        Phi[k] = dt * R[:, :N]
        R = R @ Ad
    gram = np.einsum("kin,kjn->ij", Phi, Phi)
    ystar = np.concatenate([offsets, np.zeros_like(offsets)], 1)       # (n, 4)
    lam = np.linalg.solve(dt * r1 * np.eye(4) + Qf @ gram, Qf @ ystar.T)
    u = np.einsum("kin,ij->jkn", Phi, lam)                              # (n, K, N)

    x = np.zeros((len(offsets), K + 1, n))
    for k in range(K):
        x[:, k + 1] = x[:, k] @ Ad.T
        x[:, k + 1, :N] += dt * u[:, k]
    return x[:, :, :N], x[:, :, N:]


# -----------------------------------------------------------------------------
# PDs: cosine fit of the epoch-averaged rate
# -----------------------------------------------------------------------------
def hand_speed(q, kind, center=CENTER):
    """Hand speed (n_dir, K+1), up to a constant factor."""
    if kind == "pm":
        return np.linalg.norm(q[:, :, 2:4], axis=2)
    import iLQG_Combined as M
    th = q[:, :, 0:2] + np.array(M.compute_angles_from_cartesian(*center))
    X, Y = M.hand_xy(np.concatenate([th, np.zeros_like(q[:, :, :4])], 2))
    return np.hypot(np.gradient(X, axis=1), np.gradient(Y, axis=1))


def epoch_rate(r, speed, epoch):
    """Rate averaged over an epoch, per direction -> (n_dir, N).

    'onset-peak' : 10 % of peak hand speed -> peak speed. The epoch Lillicrap &
                   Scott used for real neurons; the reference here.
    'peak-end'   : peak speed -> end (deceleration).
    'mean'       : whole movement (as analysis.fig_pd). It sums a push and a
                   brake of opposite sign, which inflates small anisotropies.
    """
    if epoch == "mean":
        return r.mean(1)
    out = []
    for i in range(len(r)):
        kp = int(speed[i].argmax())
        k0 = int(np.argmax(speed[i] > 0.1 * speed[i, kp]))
        out.append(r[i, k0:kp + 1].mean(0) if epoch == "onset-peak" else r[i, kp:].mean(0))
    return np.array(out)


def pds(r, ang, speed=None, epoch="mean"):
    feat = epoch_rate(r, speed, epoch)
    X = np.stack([np.ones_like(ang), np.cos(ang), np.sin(ang)], 1)
    b, *_ = np.linalg.lstsq(X, feat, rcond=None)
    return np.arctan2(b[2], b[1])


def axial(pd):
    """Bimodal Rayleigh: (r, axis in deg mod 180, p-value)."""
    n = len(pd)
    z = np.mean(np.exp(2j * pd))
    R = n * abs(z)
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))  # Zar 27.4
    return abs(z), np.degrees(0.5 * np.angle(z)) % 180, min(p, 1.0)


def thr(N, p=0.05):
    """Bimodal r needed for significance with N PDs."""
    return np.sqrt(-np.log(p) / N)


def axis_diff(a, b):
    """Smallest difference between two axes [deg], in [0, 90]."""
    d = (a - b) % 180
    return min(d, 180 - d)


def node_error(a, b):
    """Median |PD difference| per node [deg], up to ONE global sign flip of the
    population (an epoch mean dominated by braking flips every PD together;
    that is the time course, not the tuning)."""
    return min(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (a - b + s))))))
               for s in (0.0, np.pi))


# -----------------------------------------------------------------------------
# the analytic predictor
# -----------------------------------------------------------------------------
def gramian(J, duration=DURATION, K=K):
    """Finite-horizon controllability Gramian of r+ = (I + dt J) r + dt u."""
    dt = duration / K
    Ad = np.eye(len(J)) + dt * J
    P = np.zeros_like(J)
    E = np.eye(len(J))
    for _ in range(K):
        P += dt * E @ E.T
        E = Ad @ E
    return P


def arm_bio_map(center=CENTER):
    """Hand-direction -> torque map M J_hand^-1 of the frozen arm (README)."""
    import iLQG_Combined as M
    th1, th2 = M.compute_angles_from_cartesian(*center)
    l1, l2 = 30.0, 33.0
    Jh = np.array([[-l1 * np.sin(th1) - l2 * np.sin(th1 + th2), -l2 * np.sin(th1 + th2)],
                   [l1 * np.cos(th1) + l2 * np.cos(th1 + th2), l2 * np.cos(th1 + th2)]])
    return inertia(th2) @ np.linalg.inv(Jh)


def predicted_pds(J, Wout, bio=None):
    """Static minimum-energy prediction r = P Wout^T Gamma^-1 y(theta)."""
    P = gramian(J)
    T = P @ Wout.T @ np.linalg.inv(Wout @ P @ Wout.T)        # (N, 2)
    if bio is not None:
        T = T @ bio
    return np.arctan2(T[:, 1], T[:, 0])


# -----------------------------------------------------------------------------
# experiments
# -----------------------------------------------------------------------------
def validate():
    """Exact solver vs the iLQG cache (linear arm, network #2 at g=0.8)."""
    d = CO.load("centerout_lin24.npz")
    nets, Wout = load_networks()
    ang = d["ang"]
    r, _ = solve(nets["0.8"][2], Wout, linear_arm(tuple(d["center"])),
                 arm_offsets(ang, tuple(d["center"]), float(d["radius"])),
                 duration=float(d["duration"]), K=int(d["K"]))
    ri = d["states"][:, :, :100]
    rel = np.linalg.norm(r - ri) / np.linalg.norm(ri)
    print(f"validation vs iLQG cache (linear arm, net #2): activity rel. error {rel:.1e}, "
          f"median node PD difference {node_error(pds(r, ang), pds(ri, ang)):.2f} deg")


def measure(J, Wout, ang, plants, bio):
    """PDs of one network in every plant and epoch, plus the predictions."""
    out = {}
    for kind, (plant, off) in plants.items():
        r, q = solve(J, Wout, plant, off)
        sp = hand_speed(q, kind)
        for ep in EPOCHS:
            out[kind, ep] = pds(r, ang, sp, ep)
    out["pred_pm"] = predicted_pds(J, Wout)
    out["pred_arm"] = predicted_pds(J, Wout, bio)
    return out


def run():
    nets, Wout = load_networks()
    ang = directions()
    plants = {"pm": (point_mass(), pm_offsets(ang)),
              "arm": (linear_arm(), arm_offsets(ang))}
    bio = arm_bio_map()
    rows = []
    for g in GAINS:
        for i, J in enumerate(nets[g]):
            rows.append(dict(g=g, net=i, variant="network",
                             m=measure(J, Wout, ang, plants, bio)))
            rows.append(dict(g=g, net=i, variant="QJQ^T",
                             m=measure(random_rotation(J, 100 + i), Wout, ang, plants, bio)))
    rows.append(dict(g="-", net=0, variant="no recurrence",
                     m=measure(-np.eye(Wout.shape[1]) / TAU_NET, Wout, ang, plants, bio)))
    return rows


def size_scaling(sizes=(100, 300, 1000), n_net=8, g=0.8, seed=7):
    """Synthetic networks drawn like the file's: J = (g X - I)/tau, X ~ N(0, 1/N),
    a fresh Gaussian readout scaled so that Wout Wout^T matches the file's."""
    ang = directions()
    pm, off = point_mass(), pm_offsets(ang)
    rng = np.random.default_rng(seed)
    res = {}
    for N in sizes:
        rs = {"onset-peak": [], "mean": []}
        for _ in range(n_net):
            J = (g * rng.normal(0, 1 / np.sqrt(N), (N, N)) - np.eye(N)) / TAU_NET
            Wo = rng.normal(0, 0.54, (2, N)) * np.sqrt(100 / N)
            r, q = solve(J, Wo, pm, off)
            sp = hand_speed(q, "pm")
            for ep in rs:
                rs[ep].append(axial(pds(r, ang, sp, ep))[0])
        res[N] = {ep: np.array(v) for ep, v in rs.items()}
    return res


def _sel(rows, g, v):
    return [x["m"] for x in rows if x["g"] == g and x["variant"] == v]


def report(rows, scaling):
    ep = "onset-peak"
    nr = [x for x in rows if x["variant"] == "no recurrence"][0]["m"]
    print(f"\nchance for 100 uniform PDs: bimodal r ~ {np.sqrt(np.pi) / 20:.3f}; "
          f"p<0.05 needs r > {thr(100):.3f}")
    print("\nPOINT MASS, epoch onset->peak speed (isotropic body: any bias is the network's)")
    print(f"{'g':>4} {'variant':>9} | {'r':>5} {'sig':>5} {'axis spread':>11} | "
          f"{'pred-axis err':>13} {'node PD err':>11} | {'PD drift acc->dec':>17} | {'r, mean epoch':>13}")

    def line(g, v, M):
        A = [axial(m["pm", ep]) for m in M]
        z = np.mean(np.exp(2j * np.radians([a[1] for a in A])))
        spread = np.degrees(np.sqrt(-2 * np.log(max(abs(z), 1e-12)))) / 2 if len(A) > 1 else 0.0
        perr = np.median([axis_diff(a[1], axial(m["pred_pm"])[1]) for a, m in zip(A, M)])
        nerr = np.median([node_error(m["pm", ep], m["pred_pm"]) for m in M])
        drift = np.median([node_error(m["pm", "onset-peak"], m["pm", "peak-end"]) for m in M])
        print(f"{g:>4} {v[:9]:>9} | {np.median([a[0] for a in A]):5.2f} "
              f"{100 * np.mean([a[2] < .05 for a in A]):4.0f}% {spread:9.0f} d | "
              f"{perr:11.1f} d {nerr:9.1f} d | {drift:15.1f} d | "
              f"{np.median([axial(m['pm', 'mean'])[0] for m in M]):13.2f}")

    line("-", "no recur.", [nr])
    for g in GAINS:
        for v in ["network", "QJQ^T"]:
            line(g, v, _sel(rows, g, v))

    print("\nLINEAR ARM (biomechanics + network): bimodal axis across the 10 networks")
    for e in ("onset-peak", "mean"):
        a = axial(nr["arm", e])
        print(f"  epoch {e}: no recurrence r={a[0]:.2f} axis={(a[1] + 90) % 180 - 90:.0f} deg")
        for g in GAINS:
            M = _sel(rows, g, "network")
            A = [axial(m["arm", e]) for m in M]
            ax = np.array([(a[1] + 90) % 180 - 90 for a in A])
            perr = np.median([axis_diff(a[1], axial(m["pred_arm"])[1]) for a, m in zip(A, M)])
            print(f"    g={g}: r={np.median([a[0] for a in A]):.2f}  axis median {np.median(ax):6.1f}, "
                  f"range {ax.min():6.1f} .. {ax.max():6.1f} deg | predicted-axis err {perr:4.1f} deg")

    print("\nNETWORK SIZE (synthetic, g=0.8, point mass): median bimodal r")
    for N, v in scaling.items():
        o, mn = v["onset-peak"], v["mean"]
        print(f"  N={N:5d}: onset-peak r={np.median(o):.3f} ({(o > thr(N)).sum()}/{len(o)} sig) | "
              f"mean epoch r={np.median(mn):.3f} ({(mn > thr(N)).sum()}/{len(mn)} sig) | "
              f"threshold {thr(N):.3f}")


def figure(rows, scaling, fname="fig_network_pd_bias.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = "onset-peak"
    nr = [x for x in rows if x["variant"] == "no recurrence"][0]["m"]
    fig = plt.figure(figsize=(17, 10))
    top = fig.add_gridspec(1, 5, top=0.86, bottom=0.55, wspace=0.45)
    bot = fig.add_gridspec(1, 4, top=0.42, bottom=0.07, wspace=0.42)
    nb = 18
    bins = np.linspace(-np.pi, np.pi, nb + 1)
    w = 2 * np.pi / nb

    def ranked(g):
        X = [x for x in rows if x["g"] == g and x["variant"] == "network"]
        return sorted(X, key=lambda x: -axial(x["m"]["pm", ep])[0])
    r8, r11 = ranked("0.8"), ranked("1.1")
    show = [(nr, "sans récurrence"),
            (r8[5]["m"], f"g=0.8, réseau médian (#{r8[5]['net']})"),
            (r8[1]["m"], f"g=0.8, 2e plus biaisé /10 (#{r8[1]['net']})"),
            (r8[0]["m"], f"g=0.8, le plus biaisé /10 (#{r8[0]['net']})"),
            (r11[4]["m"], f"g=1.1, réseau médian (#{r11[4]['net']})")]
    for j, (m, title) in enumerate(show):
        ax = fig.add_subplot(top[0, j], projection="polar")
        pd = m["pm", ep]
        c, _ = np.histogram(pd, bins)
        ax.bar(bins[:-1] + w / 2, c, width=w, color="#4C72B0", edgecolor="k", lw=0.5)
        rmax = c.max() * 1.1
        a = axial(pd)
        ap = np.radians(axial(m["pred_pm"])[1])
        for o in (np.radians(a[1]), np.radians(a[1]) + np.pi):
            ax.plot([o, o], [0, rmax], color="crimson", lw=2.2)
        for o in (ap, ap + np.pi):
            ax.plot([o, o], [0, rmax], color="k", lw=1.3, ls="--")
        ax.set_yticklabels([])
        ax.set_xticks(np.radians([0, 90, 180, 270]), ["0°", "90°", "180°", "270°"], fontsize=8)
        ax.set_title(f"{title}\nr = {a[0]:.2f}"
                     f"{'  (p<0.05)' if a[2] < 0.05 else '  (n.s.)'}", fontsize=9)

    jit = lambda k, n: k + np.random.default_rng(k).uniform(-.13, .13, n)
    ax = fig.add_subplot(bot[0, 0])
    for k, g in enumerate(GAINS):
        rr = [axial(m["pm", ep])[0] for m in _sel(rows, g, "network")]
        ax.plot(jit(k, len(rr)), rr, "o", color="#4C72B0", ms=6, alpha=0.85)
    ax.plot([-0.6], [axial(nr["pm", ep])[0]], "s", color="0.4", ms=7, label="sans récurrence")
    ax.axhline(thr(100), color="crimson", ls=":", label="seuil p = 0.05")
    ax.set_xticks(range(len(GAINS)), [f"g={g}" for g in GAINS])
    ax.set_xlim(-0.9, len(GAINS) - 0.6)
    ax.set_ylabel("biais des PD  (r bimodal)")
    ax.set_title("1. Corps isotrope : le réseau crée un biais,\n"
                 "qui croît avec le gain g", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    ax = fig.add_subplot(bot[0, 1])
    Ns = list(scaling)
    for key, col, lab in [("onset-peak", "#4C72B0", "début → pic de vitesse"),
                          ("mean", "0.6", "tout le mouvement")]:
        ax.plot(Ns, [np.median(scaling[N][key]) for N in Ns], "o-", color=col, label=lab)
    ax.plot(Ns, [thr(N) for N in Ns], ":", color="crimson", label="seuil p = 0.05")
    ax.set_xscale("log")
    ax.set_xticks(Ns, [str(N) for N in Ns])
    ax.minorticks_off()
    ax.set_xlabel("taille du réseau N  (g = 0.8)")
    ax.set_ylabel("biais médian (r)")
    ax.set_title("2. Effet de petit réseau : il disparaît\nà N = 1000 (fenêtre de L&S)",
                 fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(bot[0, 2])
    for k, g in enumerate(GAINS):
        d = [node_error(m["pm", "onset-peak"], m["pm", "peak-end"]) for m in _sel(rows, g, "network")]
        ax.plot(jit(k, len(d)), d, "o", color="#55A868", ms=6, alpha=0.85)
    ax.plot([-0.6], [node_error(nr["pm", "onset-peak"], nr["pm", "peak-end"])], "s",
            color="0.4", ms=7, label="sans récurrence")
    ax.set_xticks(range(len(GAINS)), [f"g={g}" for g in GAINS])
    ax.set_xlim(-0.9, len(GAINS) - 0.6)
    ax.set_ylabel("|ΔPD| médian par nœud [deg]\n(accélération → freinage)")
    ax.set_title("3. Le réseau fait tourner les PD\npendant le geste", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    ax = fig.add_subplot(bot[0, 3])
    for k, g in enumerate(GAINS):
        axs = [(axial(m["arm", ep])[1] + 90) % 180 - 90 for m in _sel(rows, g, "network")]
        ax.plot(jit(k, len(axs)), axs, "o", color="#DD8452", ms=6, alpha=0.85)
    nax = (axial(nr["arm", ep])[1] + 90) % 180 - 90
    ax.axhline(nax, color="0.4", ls=":", label=f"sans récurrence ({nax:.0f}°)")
    ax.set_xticks(range(len(GAINS)), [f"g={g}" for g in GAINS])
    ax.set_xlim(-0.6, len(GAINS) - 0.6)
    ax.set_ylabel("axe bimodal des PD [deg]")
    ax.set_title("4. Bras : la biomécanique fixe l'axe,\nle réseau le décale", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Le réseau 100×100 peut-il biaiser les PD ?   Haut : corps isotrope (masse "
                 "ponctuelle), fenêtre début → pic de vitesse\nrouge : axe simulé ; tirets : axe "
                 "prédit sans simulation (P·Woutᵀ·Γ⁻¹)", fontsize=12)
    fig.savefig(os.path.join(_HERE, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


if __name__ == "__main__":
    validate()
    rows = run()
    scaling = size_scaling()
    report(rows, scaling)
    figure(rows, scaling)
