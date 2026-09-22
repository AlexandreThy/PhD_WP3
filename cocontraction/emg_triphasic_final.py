"""Figure finale : la variable a(t) (activation musculaire = drive neural filtre
passe-bas = analogue de l'enveloppe EMG lissee) reproduit qualitativement le
patron (bi/tri)phasique AG1-ANT(-AG2) d'une flexion coude mono-articulaire
rapide, compare a la forme idealisee de reference (Hallett 1975, Gottlieb 1989).

On lance 200 et 250 ms (mouvements balistiques). Les 2-3 derniers echantillons
sont un artefact de bord (le cout terminal sur la vitesse force une commande de
freinage impulsive au tout dernier pas de discretisation) et sont retires pour
l'affichage et la detection de bouffees.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import emg_prediction_simple as EP
import emg_validation_gribble1999 as V

EL_FLEX, EL_EXT = 2, 3
TRIM = 3   # echantillons de bord retires


def run(Duration, amp_deg=55.0, r1=4e-4, w2=60.0, K=None):
    if K is None:
        K = int(round(Duration / 0.25 * 150))
    xs, u, th2t = V.simulate_singlejoint(
        th1_0=0.35, th2_0=1.1, th2_amp=amp_deg * np.pi / 180.0,
        Duration=Duration, w_hold=5e3, w1=2e4, w1end=2e3, w2=w2,
        r1=r1, K=K, max_iter=300)
    dt = Duration / K
    sl = slice(0, K - TRIM)
    t = (np.arange(K) * dt + dt / 2)[sl]
    e = EP.excitation(u)[sl]
    a = xs[:-1, 4:4 + EP.NM][sl]
    om2 = xs[:-1, 3][sl]
    return dict(t=t, e=e, a=a, om2=om2, Duration=Duration,
                th1dev=np.degrees(np.abs(xs[:, 0] - xs[0, 0])).max())


def ideal_triphasic(t, T):
    """Forme idealisee de reference (3 bosses gaussiennes), pour comparaison
    visuelle uniquement -- pas un ajustement."""
    def g(mu, sd, amp):
        return amp * np.exp(-0.5 * ((t - mu) / sd) ** 2)
    ag = g(0.12 * T, 0.06 * T, 1.0) + g(0.9 * T, 0.07 * T, 0.4)   # AG1 + AG2
    ant = g(0.55 * T, 0.09 * T, 0.7)                              # ANT
    return ag, ant


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    runs = [run(0.20), run(0.25)]

    fig, axes = plt.subplots(3, len(runs), figsize=(5 * len(runs), 8.5), sharex="col")
    for j, R in enumerate(runs):
        t, a, e, om2, T = R['t'], R['a'], R['e'], R['om2'], R['Duration']
        tpk = t[np.argmax(np.abs(om2))]
        ag_i, ant_i = ideal_triphasic(t, T)

        axes[0, j].plot(t, np.degrees(om2), "k")
        axes[0, j].axvline(tpk, color="gray", ls=":", lw=1)
        axes[0, j].set_title(f"flexion coude 55 deg en {T*1000:.0f} ms "
                             f"(epaule fixe, dev {R['th1dev']:.2f} deg)")
        if j == 0:
            axes[0, j].set_ylabel("vitesse coude\n[deg/s]")

        # agoniste
        af = a[:, EL_FLEX]
        axes[1, j].plot(t, af / af.max(), "C0", lw=2, label="a(t) modele [EMG lisse]")
        axes[1, j].plot(t, ag_i, "C0", lw=1, ls="--", alpha=0.5,
                        label="forme idealisee (ref.)")
        axes[1, j].axvline(tpk, color="gray", ls=":", lw=1)
        axes[1, j].annotate("AG1", (t[np.argmax(af)], 1.0), color="C0",
                            fontsize=11, fontweight="bold", ha="center", va="bottom")
        if j == 0:
            axes[1, j].set_ylabel("AGONISTE\n(flechisseur coude)\nnormalise")
            axes[1, j].legend(fontsize=7, loc="upper right")

        # antagoniste
        ae = a[:, EL_EXT]
        axes[2, j].plot(t, ae / ae.max(), "C3", lw=2, label="a(t) modele [EMG lisse]")
        axes[2, j].plot(t, ant_i / ant_i.max(), "C3", lw=1, ls="--", alpha=0.5,
                        label="forme idealisee (ref.)")
        axes[2, j].axvline(tpk, color="gray", ls=":", lw=1)
        axes[2, j].annotate("ANT", (t[np.argmax(ae)], 1.0), color="C3",
                            fontsize=11, fontweight="bold", ha="center", va="bottom")
        axes[2, j].set_xlabel("temps (s)")
        if j == 0:
            axes[2, j].set_ylabel("ANTAGONISTE\n(extenseur coude)\nnormalise")

    fig.suptitle("a(t) reproduit l'enveloppe EMG (bi/tri)phasique — pointille = pic de vitesse",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(_HERE, "fig_emg_triphasic_final.png")
    plt.savefig(out, dpi=150)

    # ---- diagnostic timing (fractions de la duree) ----
    print(f"{'duree':>7} {'AG1 pic':>10} {'ANT pic':>10} {'pic vitesse':>12}  (en % de la duree)")
    for R in runs:
        t, a, om2, T = R['t'], R['a'], R['om2'], R['Duration']
        tag = t[np.argmax(a[:, EL_FLEX])] / T
        tant = t[np.argmax(a[:, EL_EXT])] / T
        tv = t[np.argmax(np.abs(om2))] / T
        print(f"{T*1000:>6.0f}m {tag*100:>9.0f}% {tant*100:>9.0f}% {tv*100:>11.0f}%")
    print(f"\nfigure : {out}")
