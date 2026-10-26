"""Fig. 5 overlay (instantaneous and mean F-L/V) with the macaque arm and the task
of Lillicrap & Scott: 6 cm reaches in 0.4 s, 16 targets (see macaque.py).

Caches: sims/fig5_overlay_macaque.npz, sims/fig5_meanflv_macaque.npz.

    python analyses_claude/fig5_macaque.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import macaque  # noqa: E402,F401  -- must come first: patches the arm and task
import compare3 as C  # noqa: E402
import fig5_overlay as F  # noqa: E402
import fig5_meanflv as FM  # noqa: E402
from net_variability import NETS  # noqa: E402

RG = macaque.REGIME


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    C.REGIMES = {RG: C.DURATIONS[RG]}
    tag = macaque.VISCOSITY
    F.CACHE = os.path.join(HERE, "sims", f"fig5_overlay_macaque_{tag}.npz")
    FM.CACHE = os.path.join(HERE, "sims", f"fig5_meanflv_macaque_{tag}.npz")
    F.TITLE_EXTRA = f" — bras de macaque, viscosité {tag}"
    print(f"centre {np.round(C.M.CENTER_XY, 2)} cm, a1={C.M.a1:.4f} a2={C.M.a2:.4f} a3={C.M.a3:.4f}")
    if not os.path.exists(F.CACHE):
        F.simulate()
    z0 = dict(np.load(F.CACHE))
    if not os.path.exists(FM.CACHE):
        FM.simulate(z0)
    z = {**z0, **dict(np.load(FM.CACHE))}

    errs = {m: max(float(z[f"{RG}|{m}|{n}|err"]) for n, _ in NETS)
            for m in [r[0] for r in F.RUNGS] + ["nl6m"]}
    print("max endpoint err (cm):", {k: f"{v:.1e}" for k, v in errs.items()})
    H = np.stack([z[f"{RG}|nl6m|{n}|H"] for n, _ in NETS])
    print(f"mean F-L/V factors H: range {H.min():.2f}-{H.max():.2f}, per muscle {np.round(H.mean((0, 1)), 2)}")

    F.RUNGS = F.RUNGS[:5] + [("nl6", "6  + muscle mechanics\n(FLV instantanée)"),
                             ("nl6m", "6'  + muscle mechanics\n(FLV moyenne, comme L&S)")]
    F.PAPER_UNIT["nl6m"] = F.PAPER_UNIT["nl6"]
    F.PAPER_MUSC["nl6m"] = F.PAPER_MUSC["nl6"]
    F.figure(z, RG, suffix=f"_macaque_{tag}")
    for m, lab in F.RUNGS:
        pu, _ = F.pmds(z, RG, m, "r")
        Ru, tu, p = F.bimodal_test(pu)
        line = f"{lab.replace(chr(10), ' '):48s} units θ={tu:6.1f} r={Ru:.2f}" + ("" if p < 1e-3 else " (n.s.)")
        if m in F.PAPER_MUSC:
            pa, _ = F.pmds(z, RG, m, "a")
            Ra, ta, _ = F.bimodal_test(pa)
            line += f" | muscles θ={ta:6.1f} r={Ra:.2f}"
        print(line)


if __name__ == "__main__":
    main()
