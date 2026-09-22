"""Reproduction QUALITATIVE de l'enveloppe EMG lissee d'un mouvement simple :
le PATRON TRIPHASIQUE (AG1 - ANT - AG2) d'un mouvement mono-articulaire rapide
du coude (Wachholder & Altenburger 1926 ; Hallett et al. 1975 ; Gottlieb,
Corcos & Agarwal 1989).

Patron attendu, robuste et universel pour un mouvement balistique :
  * AG1  : 1ere bouffee de l'AGONISTE (flechisseur coude), avant/au debut du
           mouvement -- accelere le membre.
  * ANT  : bouffee de l'ANTAGONISTE (extenseur coude), en phase de deceleration
           (~milieu-fin) -- freine le membre.
  * AG2  : 2eme bouffee de l'agoniste, en fin de mouvement -- stabilise/arrete.

Quelle variable lit-on comme "enveloppe EMG lissee" ? L'EMG de surface traite =
redresse + filtre passe-bas. Dans un modele muscle a dynamique d'activation :
    excitation e(t)  --[filtre passe-bas 1er ordre, tau_act/tau_deact]-->  a(t)
L'enveloppe EMG lissee correspond donc a a(t) (drive neural deja filtre). e(t)
est le drive neural brut. On trace les deux ; a(t) est l'analogue "EMG lisse".

Mono-articulaire : bras 2-liens avec EPAULE VERROUILLEE par un cout (comme
Gribble & Ostry Exp. 1). On lit les muscles du coude : #2 flechisseur (agoniste),
#3 extenseur (antagoniste).
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import emg_prediction_simple as EP
import emg_validation_gribble1999 as V
import ilqg_muscle as MU

EL_FLEX, EL_EXT = 2, 3   # indices muscle : coude flechisseur / extenseur


def bursts(sig, t, k_move_end):
    """Detecte des extrema locaux (bouffees) dans un signal lisse et positif."""
    peaks = []
    for i in range(1, len(sig) - 1):
        if sig[i] > sig[i - 1] and sig[i] >= sig[i + 1] and sig[i] > 0.15 * sig.max():
            peaks.append((t[i], sig[i]))
    return peaks


def run_one(Duration, amp_deg=50.0, r1=4e-4, w2=40.0, K=None):
    if K is None:
        K = max(120, int(Duration / 0.5 * 150))
    xs, u, th2t = V.simulate_singlejoint(
        th1_0=0.35, th2_0=1.1, th2_amp=amp_deg * np.pi / 180.0,
        Duration=Duration, w_hold=5e3, w1=2e4, w1end=2e3, w2=w2,
        r1=r1, K=K, max_iter=300)
    dt = Duration / K
    t_u = np.arange(K) * dt + dt / 2
    e = EP.excitation(u)                       # drive neural brut (K,6)
    a = xs[:-1, 4:4 + EP.NM]                    # activation = enveloppe EMG lissee
    om2 = xs[:-1, 3]
    th1dev = np.degrees(np.abs(xs[:, 0] - xs[0, 0])).max()
    return dict(t=t_u, e=e, a=a, om2=om2, xs=xs, Duration=Duration,
                K=K, th1dev=th1dev, th2t=th2t)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    durations = [0.25, 0.35, 0.5]
    runs = [run_one(D) for D in durations]

    # ---- diagnostic texte : compte des bouffees AG / ANT ----
    print(f"{'duree':>6}  {'dev.epaule':>10}  {'AG bursts':>28}  {'ANT bursts':>20}")
    for R in runs:
        ag = bursts(R['a'][:, EL_FLEX], R['t'], R['K'])
        ant = bursts(R['a'][:, EL_EXT], R['t'], R['K'])
        ag_s = ", ".join(f"{tt*1000:.0f}ms" for tt, _ in ag)
        ant_s = ", ".join(f"{tt*1000:.0f}ms" for tt, _ in ant)
        print(f"{R['Duration']:>6.2f}  {R['th1dev']:>9.3f}d  {ag_s:>28}  {ant_s:>20}")

    # ---- figure ----
    fig, axes = plt.subplots(3, len(runs), figsize=(4 * len(runs), 8), sharex="col")
    for j, R in enumerate(runs):
        t, e, a, om2 = R['t'], R['e'], R['a'], R['om2']
        tpk = t[np.argmax(np.abs(om2))]
        # vitesse angulaire coude
        axes[0, j].plot(t, np.degrees(om2), color="k")
        axes[0, j].set_title(f"duree {R['Duration']*1000:.0f} ms "
                             f"(dev.epaule {R['th1dev']:.2f}d)")
        axes[0, j].set_ylabel("vit. coude [deg/s]" if j == 0 else "")
        # agoniste = flechisseur coude
        axes[1, j].plot(t, a[:, EL_FLEX], color="C0", label="a(t) [EMG lisse]")
        axes[1, j].plot(t, e[:, EL_FLEX], color="C0", alpha=0.35, linestyle="--",
                        label="e(t) [drive brut]")
        axes[1, j].axvline(tpk, color="gray", ls=":", lw=1)
        axes[1, j].set_ylabel("AGONISTE\n(flech. coude)" if j == 0 else "")
        # antagoniste = extenseur coude
        axes[2, j].plot(t, a[:, EL_EXT], color="C3", label="a(t) [EMG lisse]")
        axes[2, j].plot(t, e[:, EL_EXT], color="C3", alpha=0.35, linestyle="--",
                        label="e(t) [drive brut]")
        axes[2, j].axvline(tpk, color="gray", ls=":", lw=1)
        axes[2, j].set_ylabel("ANTAGONISTE\n(ext. coude)" if j == 0 else "")
        axes[2, j].set_xlabel("temps (s)")
        if j == 0:
            axes[1, j].legend(fontsize=7, loc="upper right")
    fig.suptitle("Patron triphasique EMG (AG1-ANT-AG2) — flexion coude mono-articulaire\n"
                 "pointille gris = pic de vitesse", fontsize=11)
    plt.tight_layout()
    out = os.path.join(_HERE, "fig_emg_triphasic.png")
    plt.savefig(out, dpi=150)
    print(f"\nfigure : {out}")
