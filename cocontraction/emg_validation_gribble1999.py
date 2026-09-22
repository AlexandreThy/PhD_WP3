"""Validation quantitative contre de vraies donnees EMG : mouvement mono-
articulaire du coude (flexion de 40 deg, epaule maintenue fixe), reproduisant
l'*experiment 1* de Gribble & Ostry (1999, J Neurophysiol 82:2310-2326,
"Compensation for Interaction Torques During Single- and Multijoint Limb
Movement").

Pourquoi cette tache : c'est EXACTEMENT le groupement de 6 muscles utilise par
`ilqg_muscle.py` (issu de Li & Todorov / Gribble et al.) -- pectoralis
(flechisseur epaule), deltoid posterieur (extenseur epaule), biceps court +
brachioradialis (flechisseurs coude, regroupes ici en un seul "flechisseur
coude"), triceps lateral (extenseur coude), biceps long (flechisseur
bi-articulaire), triceps long (extenseur bi-articulaire).

Le resultat cle du papier (Fig. 3A, table ci-dessous) : lors d'une flexion du
coude PURE (l'epaule ne bouge pas), les muscles de l'epaule montrent quand
meme une activite phasique anticipatoire, pour compenser le couple
d'interaction cree par le mouvement du coude. Precisement, l'onset de
l'activite EMG (temps par rapport au debut du mouvement, en ms, moyenne sur
tous sujets/essais, n>1600 par muscle) :

    pectoralis (epaule flech.)         -78 ms
    deltoid (epaule ext.)              +31 ms
    biceps long (bi-art. flech.)       -41 ms
    biceps court (coude flech.)        -38 ms
    brachioradialis (coude flech.)     -27 ms
    triceps lateral (coude ext.)       +59 ms
    triceps long (bi-art. ext.)        +51 ms

Pattern qualitatif attendu, et teste ici : TOUS les muscles "flechisseurs"
(y compris a l'epaule, qui ne bouge pourtant pas) demarrent AVANT le
mouvement (onset negatif), tous les "extenseurs" demarrent APRES (onset
positif) -- une organisation temporelle identique au joint mobile et au
joint stationnaire.

Methode d'onset (approximative, a lire comme telle) : le papier utilise un
seuil "3 SD au-dessus du bruit de fond" sur de l'EMG bruite -- non applicable
tel quel a une simulation deterministe sans bruit. On utilise ici un seuil
simple sur l'excitation simulee (10% de l'exursion min-max de son propre
burst). La comparaison de SIGNE et d'ORDRE DE GRANDEUR relatif est valide;
l'accord numerique exact en ms ne doit pas etre surinterprete.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import emg_prediction_simple as EP  # f, _jac_fd, excitation, NM, NX, MU
import ilqg_muscle as MU

NM, NX = EP.NM, EP.NX

# -----------------------------------------------------------------------------
# cout : cout courant qui maintient th1 (epaule) fixe + cout terminal sur th2
# -----------------------------------------------------------------------------
def l_state(x, th1_0, w_hold):
    return w_hold / 2 * (x[0] - th1_0) ** 2


def l_state_x(x, th1_0, w_hold):
    g = np.zeros(len(x)); g[0] = w_hold * (x[0] - th1_0)
    return g


def l_state_xx(x, w_hold):
    Q = np.zeros((len(x), len(x))); Q[0, 0] = w_hold
    return Q


def h(x, th2_targ, th1_0, w1, w1end, w2):
    return w1 / 2 * (x[1] - th2_targ) ** 2 + w1end / 2 * (x[0] - th1_0) ** 2 \
        + w2 / 2 * (x[2] ** 2 + x[3] ** 2)


def hx(x, th2_targ, th1_0, w1, w1end, w2):
    g = np.zeros(len(x))
    g[0] = w1end * (x[0] - th1_0)
    g[1] = w1 * (x[1] - th2_targ)
    g[2] = w2 * x[2]; g[3] = w2 * x[3]
    return g


def hxx(w1, w1end, w2):
    Q = np.zeros((NX, NX))
    Q[0, 0] = w1end; Q[1, 1] = w1; Q[2, 2] = w2; Q[3, 3] = w2
    return Q


def simulate_singlejoint(th1_0=0.3, th2_0=1.2, th2_amp=40 * np.pi / 180,
                          Duration=0.5, w_hold=3e3, w1=1e4, w1end=1e3, w2=10.0,
                          r1=1e-3, K=150, eps=1e-6, max_iter=250, tol=1e-6):
    th2_targ = th2_0 + th2_amp
    x0 = np.array([th1_0, th2_0, 0.0, 0.0] + [0.0] * NM)
    dt = Duration / K
    u = np.zeros((K, NM))

    def rollout(u):
        xs = np.zeros((K + 1, NX)); xs[0] = x0
        for k in range(K):
            xs[k + 1] = xs[k] + dt * EP.f(xs[k], u[k])
        return xs

    def cost(xs, u):
        J = h(xs[-1], th2_targ, th1_0, w1, w1end, w2)
        for k in range(K):
            J += dt * (l_state(xs[k], th1_0, w_hold) + r1 * np.sum(u[k] ** 2) / 2)
        return J

    xs = rollout(u)
    J = cost(xs, u)
    alphas = 0.5 ** np.arange(10)
    for it in range(max_iter):
        A = np.zeros((K, NX, NX)); B = np.zeros((K, NX, NM))
        q = np.zeros(K + 1); qb = np.zeros((K + 1, NX))
        r = np.zeros((K, NM)); Q = np.zeros((K + 1, NX, NX)); R = np.zeros((K, NM, NM))
        for k in range(K):
            Jx, Ju = EP._jac_fd(EP.f, xs[k], u[k])
            A[k] = np.eye(NX) + dt * Jx
            B[k] = dt * Ju
            qb[k] = dt * l_state_x(xs[k], th1_0, w_hold)
            Q[k] = dt * l_state_xx(xs[k], w_hold)
            r[k] = dt * r1 * u[k]
            R[k] = dt * r1 * np.eye(NM)
        qb[-1] = hx(xs[-1], th2_targ, th1_0, w1, w1end, w2)
        Q[-1] = hxx(w1, w1end, w2)

        S, s = Q[-1].copy(), qb[-1].copy()
        lg = np.zeros((K, NM)); Lg = np.zeros((K, NM, NX))
        for k in range(K - 1, -1, -1):
            SA = S @ A[k]
            G = B[k].T @ SA
            g_ = r[k] + B[k].T @ s
            Hh = R[k] + B[k].T @ S @ B[k]
            w_, V_ = np.linalg.eigh(0.5 * (Hh + Hh.T))
            w_ = np.where(w_ < eps, eps, w_)
            Hinv = (V_ * (1.0 / w_)) @ V_.T
            Snew = Q[k] + A[k].T @ SA - G.T @ Hinv @ G
            snew = qb[k] + A[k].T @ s - G.T @ Hinv @ g_
            lg[k] = -Hinv @ g_
            Lg[k] = -Hinv @ G
            S, s = Snew, snew

        improved = False
        xnew = unew = None
        for alpha in alphas:
            xnew = np.zeros_like(xs); unew = np.zeros_like(u); xnew[0] = x0
            for k in range(K):
                unew[k] = u[k] + alpha * lg[k] + Lg[k] @ (xnew[k] - xs[k])
                xnew[k + 1] = xnew[k] + dt * EP.f(xnew[k], unew[k])
            Jnew = cost(xnew, unew)
            if np.isfinite(Jnew) and Jnew < J:
                improved = True
                break
        if not improved:
            break
        rel = (J - Jnew) / max(abs(J), 1e-12)
        xs, u, J = xnew, unew, Jnew
        if rel < tol:
            break
    return xs, u, th2_targ


def onset_time(t, sig, frac=0.10):
    baseline, peak = sig.min(), sig.max()
    thr = baseline + frac * (peak - baseline)
    idx = int(np.argmax(sig >= thr))
    return t[idx]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xs, u, th2_targ = simulate_singlejoint()
    K = u.shape[0]
    Duration = 0.5
    dt = Duration / K
    t_u = np.arange(K) * dt + dt / 2
    t_x = np.arange(K + 1) * dt

    e = EP.excitation(u)                    # (K, 6) -- signal "EMG predit"
    om2 = xs[:-1, 3]
    th1_err_deg = np.degrees(xs[:, 0] - xs[0, 0])
    th2_deg = np.degrees(xs[:, 1])

    print(f"deviation max epaule (censee etre 0) : {np.abs(th1_err_deg).max():.3f} deg")
    print(f"coude : {np.degrees(xs[0,1]):.1f} -> {np.degrees(xs[-1,1]):.1f} deg "
          f"(cible {np.degrees(th2_targ):.1f})")

    t_mvt = onset_time(t_u, om2, frac=0.05)   # "movement onset" (5% du pic de omega2)

    names = ["epaule flechisseur", "epaule extenseur", "coude flechisseur",
             "coude extenseur", "bi-art. flechisseur", "bi-art. extenseur"]
    paper_ms = [-78, 31, -32.5, 59, -41, 51]
    paper_detail = ["pectoralis -78", "deltoid +31",
                    "biceps court -38 / brachioradialis -27 (moy -32.5)",
                    "triceps lateral +59", "biceps long -41", "triceps long +51"]

    print(f"\n{'muscle':<22}{'modele (ms)':>14}{'papier (ms)':>14}   meme signe ?  detail papier")
    model_ms = []
    for m in range(NM):
        t0 = onset_time(t_u, e[:, m], frac=0.10)
        lat = (t0 - t_mvt) * 1000
        model_ms.append(lat)
        same = "oui" if np.sign(lat) == np.sign(paper_ms[m]) else "NON"
        print(f"{names[m]:<22}{lat:>14.0f}{paper_ms[m]:>14.0f}   {same:<12}  {paper_detail[m]}")

    fig, axes = plt.subplots(4, 2, figsize=(9, 9), sharex=True)
    axes[0, 0].plot(t_x, th1_err_deg, color="k"); axes[0, 0].set_ylabel("d(epaule) [deg]")
    axes[0, 1].plot(t_x, th2_deg, color="k"); axes[0, 1].set_ylabel("coude [deg]")
    for m in range(NM):
        r, c = 1 + m // 2, m % 2
        axes[r, c].plot(t_u, e[:, m], color="C0")
        axes[r, c].axvline(t_mvt, color="gray", linestyle=":", linewidth=1)
        lat = model_ms[m]
        axes[r, c].set_ylabel(f"{names[m]}\n({lat:+.0f} ms)")
    axes[-1, 0].set_xlabel("temps (s)"); axes[-1, 1].set_xlabel("temps (s)")
    fig.suptitle("Flexion coude 40 deg, epaule maintenue -- vs Gribble & Ostry 1999 (Exp. 1)")
    plt.tight_layout()
    out = os.path.join(_HERE, "fig_emg_validation_gribble1999.png")
    plt.savefig(out, dpi=150)
    print(f"\nfigure sauvegardee: {out}")
