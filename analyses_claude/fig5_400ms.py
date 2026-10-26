"""Fig. 5 overlay (instantaneous and mean F-L/V) at 12 cm / 0.40 s, the duration of
Lillicrap & Scott's dynamic-model reaches (their Fig. 2D).

Reuses fig5_overlay.py and fig5_meanflv.py with their own caches:
sims/fig5_overlay_040s.npz and sims/fig5_meanflv_040s.npz.

    python analyses_claude/fig5_400ms.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compare3 as C  # noqa: E402
import fig5_overlay as F  # noqa: E402
import fig5_meanflv as FM  # noqa: E402
from net_variability import NETS  # noqa: E402

RG = "12cm_040s"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    C.REGIMES = {RG: 0.4}
    F.CACHE = os.path.join(HERE, "sims", "fig5_overlay_040s.npz")
    FM.CACHE = os.path.join(HERE, "sims", "fig5_meanflv_040s.npz")
    if not os.path.exists(F.CACHE):
        F.simulate()
    z0 = dict(np.load(F.CACHE))
    if not os.path.exists(FM.CACHE):
        FM.simulate(z0)
    z = {**z0, **dict(np.load(FM.CACHE))}

    errs = {m: max(float(z[f"{RG}|{m}|{n}|err"]) for n, _ in NETS)
            for m in [r[0] for r in F.RUNGS] + ["nl6m"]}
    print("max endpoint err (cm):", {k: f"{v:.1e}" for k, v in errs.items()})

    F.RUNGS = F.RUNGS[:5] + [("nl6", "6  + muscle mechanics\n(FLV instantanée)"),
                             ("nl6m", "6'  + muscle mechanics\n(FLV moyenne, comme L&S)")]
    F.PAPER_UNIT["nl6m"] = F.PAPER_UNIT["nl6"]
    F.PAPER_MUSC["nl6m"] = F.PAPER_MUSC["nl6"]
    F.figure(z, RG, suffix="_meanflv")
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
