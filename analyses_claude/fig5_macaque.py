"""Fig. 5 overlay (instantaneous and mean F-L/V) with the macaque arm and the task
of Lillicrap & Scott: 6 cm reaches in 0.4 s, 16 targets (see macaque.py).

Caches: sims/fig5_overlay_macaque_<visc>[_n20].npz, sims/fig5_meanflv_macaque_<visc>[_n20].npz.

    python analyses_claude/fig5_macaque.py                 # 6 (network, readout) pairs
    python analyses_claude/fig5_macaque.py --nets 20       # all 10 networks x 2 readouts

Besides the figure, prints for every rung the unit-PMD axis of each network on
its own and a 95 % bootstrap interval of the pooled axis (resampling networks),
written to fig5_macaque_<visc>[_n20]_networks.txt. Viscosity: MACAQUE_VISCOSITY; arm parameters: MACAQUE_PARAMS (see macaque.py).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import macaque  # noqa: E402,F401  -- must come first: patches the arm and task
import compare3 as C  # noqa: E402
import fig5_overlay as F  # noqa: E402
import fig5_meanflv as FM  # noqa: E402
import net_variability as NV  # noqa: E402

RG = macaque.REGIME


def axial_mean(ax_deg):
    """Mean of axes (deg, mod 180) and the resultant length of the doubled angles."""
    z = np.mean(np.exp(2j * np.radians(ax_deg)))
    return np.degrees(np.angle(z)) / 2 % 180, abs(z)


def network_table(z, nets, n_boot=2000, seed=0):
    """Per rung: per-network axes, pooled axis and its bootstrap CI over networks."""
    rng = np.random.default_rng(seed)
    lines, out = [], {}
    for m, lab in F.RUNGS:
        per = []
        for n, s in nets:
            pd, _ = F.pmds(z, RG, m, "r", nets=[(n, s)])
            R, th, p = F.bimodal_test(pd, n_boot=500)
            per.append((th, R, p))
        R_all, th_all, _ = F.bimodal_test(F.pmds(z, RG, m, "r")[0])
        boot = []
        for _ in range(n_boot):
            pick = [nets[i] for i in rng.integers(0, len(nets), len(nets))]
            boot.append(F.bimodal_test(F.pmds(z, RG, m, "r", nets=pick)[0], n_boot=1)[1])
        d = (np.array(boot) - th_all + 90) % 180 - 90           # deviations in (-90, 90]
        lo, hi = th_all + np.percentile(d, 2.5), th_all + np.percentile(d, 97.5)
        ths = np.array([x[0] for x in per])
        mth, conc = axial_mean(ths)
        old6 = F.bimodal_test(F.pmds(z, RG, m, "r", nets=[p for p in nets if p in NV.NETS])[0])
        out[m] = dict(per=per, pooled=(th_all, R_all), ci=(lo % 180, hi % 180), old6=old6)
        lines.append(f"{lab.replace(chr(10), ' '):46s} pooled θ={th_all:6.1f} r={R_all:.2f} "
                     f"CI95 [{lo % 180:5.1f}, {hi % 180:5.1f}] | original 6 nets θ={old6[1]:6.1f} | "
                     f"per-network θ: mean {mth:6.1f}, concentration {conc:.2f}, "
                     f"sig {sum(x[2] < 1e-3 for x in per)}/{len(per)}")
        lines.append("    " + " ".join(f"{x[0]:5.0f}" for x in per))
    return "\n".join(lines), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", type=int, choices=(6, 20), default=6)
    p.add_argument("--jobs", type=int, default=7)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    nets = NV.NETS20 if a.nets == 20 else NV.NETS
    F.NETS = FM.NETS = nets
    C.REGIMES = {RG: C.DURATIONS[RG]}
    tag = (macaque.VISCOSITY + ("" if macaque.PARAMS == "ls_printed" else "_" + macaque.PARAMS)
           + ("_n20" if a.nets == 20 else ""))
    F.CACHE = os.path.join(HERE, "sims", f"fig5_overlay_macaque_{tag}.npz")
    FM.CACHE = os.path.join(HERE, "sims", f"fig5_meanflv_macaque_{tag}.npz")
    F.TITLE_EXTRA = f" — bras de macaque ({macaque.PARAMS}), viscosité {macaque.VISCOSITY}"
    print(f"centre {np.round(C.M.CENTER_XY, 2)} cm, a1={C.M.a1:.4f} a2={C.M.a2:.4f} "
          f"a3={C.M.a3:.4f}, viscosity {C.M.VISCOUS.tolist()}, {len(nets)} networks")
    os.makedirs(os.path.join(HERE, "sims"), exist_ok=True)
    if not os.path.exists(F.CACHE):
        F.simulate(a.jobs)
    z0 = dict(np.load(F.CACHE))
    if not os.path.exists(FM.CACHE):
        FM.simulate(z0, a.jobs)
    z = {**z0, **dict(np.load(FM.CACHE))}

    errs = {m: max(float(z[f"{RG}|{m}|{NV.netkey(n, s)}|err"]) for n, s in nets)
            for m in [r[0] for r in F.RUNGS] + ["nl6m"]}
    print("max endpoint err (cm):", {k: f"{v:.1e}" for k, v in errs.items()})
    H = np.stack([z[f"{RG}|nl6m|{NV.netkey(n, s)}|H"] for n, s in nets])
    print(f"mean F-L/V factors H: range {H.min():.2f}-{H.max():.2f}, "
          f"per muscle {np.round(H.mean((0, 1)), 2)}")

    F.RUNGS = F.RUNGS[:5] + [("nl6", "6  + muscle mechanics\n(FLV instantanée)"),
                             ("nl6m", "6'  + muscle mechanics\n(FLV moyenne, comme L&S)")]
    F.PAPER_UNIT["nl6m"] = F.PAPER_UNIT["nl6"]
    F.PAPER_MUSC["nl6m"] = F.PAPER_MUSC["nl6"]
    F.figure(z, RG, suffix=f"_macaque_{tag}")
    for m, lab in F.RUNGS:
        pu, _ = F.pmds(z, RG, m, "r")
        Ru, tu, pv = F.bimodal_test(pu)
        line = f"{lab.replace(chr(10), ' '):48s} units θ={tu:6.1f} r={Ru:.2f}" + ("" if pv < 1e-3 else " (n.s.)")
        if m in F.PAPER_MUSC:
            pa, _ = F.pmds(z, RG, m, "a")
            Ra, ta, _ = F.bimodal_test(pa)
            line += f" | muscles θ={ta:6.1f} r={Ra:.2f}"
        print(line)

    table, _ = network_table(z, nets)
    print(table)
    with open(os.path.join(HERE, f"fig5_macaque_{tag}_networks.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
