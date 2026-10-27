"""Switch iLQG_Muscle (and compare3) to the macaque arm and task of Lillicrap & Scott.

Importing this module patches the module-level constants that iLQG_Muscle's
functions read at call time; iLQG_Muscle.py itself is not modified. Import it
at the top of a driver script so that worker processes (which re-import the
driver on Windows) are patched too.

Arm parameters: monkey arm of Lillicrap & Scott (2013) as implemented in the
public code of Kalidindi et al. (2021, eLife), github.com/Hteja/CorticalDynamics,
'reach task/muscularArmClass.py' ("fixed Monkey arm parameters ... refer to
Lillicrap et al 2013"). Muscle moment arms, L0, theta0 and Brown F-L/V
parameters there are identical to iLQG_Muscle's. F_MAX is not given in that code
(activations are in arbitrary units) and is left unchanged.

Task: 6 cm reaches (their Fig. 1A / methods), 0.4 s (their Fig. 2D).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compare3 as C  # noqa: E402

M = C.M

# Arm parameter set, environment variable MACAQUE_PARAMS:
#   ls_printed  (default) values printed in the L&S Supplemental. Their inertias
#               I1 = 0.025, I2 = 0.045 kg m^2 are the HUMAN values of Li & Todorov
#               (2004), 15-40x the macaque inertias of the source they cite
#               (Cheng & Scott 2000, Table 6). With them the L&S static model gives
#               168 deg at rung 3 instead of the published 131.4.
#   ls_cs       L&S masses/lengths, inertias about the proximal joint from Cheng &
#               Scott Table 6: upper arm 1.63e-3; forearm + hand about the elbow
#               1.07e-3 + 3.93e-5 + 0.057 (0.154 + 0.39 * 0.103)^2 = 3.26e-3.
#               Reproduces the published rungs 2-3 (109.6 / 132.3 deg, static model).
#   cheng_scott every segment parameter from Cheng & Scott means (Tables 1, 6; n = 6,
#               7.9 kg), segment 2 = forearm + hand, hand position = mid-hand.
PARAMS = os.environ.get("MACAQUE_PARAMS", "ls_printed")
_SETS = {  # I1, I2, m1, m2, l1, l2, s1, s2
    "ls_printed": (0.025, 0.045, 0.2108, 0.1938, 0.145, 0.284, 0.0749, 0.0757),
    "ls_cs": (1.63e-3, 3.26e-3, 0.2108, 0.1938, 0.145, 0.284, 0.0749, 0.0757),
    "cheng_scott": (1.63e-3, 3.26e-3, 0.294, 0.251, 0.144, 0.2055, 0.072, 0.0965),
}
(M.I1, M.I2, M.m1, M.m2, M.l1_m, M.l2_m, M.s1, M.s2) = _SETS[PARAMS]

# cheng_scott also replaces the single F_MAX = 800 N by a force per lumped muscle:
# summed PCSA of its muscles (Cheng & Scott Table 3, cm^2) x 22.5 N/cm^2. Groups as
# in the L&S Supplemental, except that the triceps long head (biarticular) is
# counted only in the biarticular extensors and the monoarticular triceps medial
# joins the elbow extensors (L&S list TLo in both groups and omit TMe):
#   shoulder flex PM + DA, shoulder ext DP + DM, elbow flex B + Br + ECRL,
#   elbow ext TLa + TMe, biarticular flex BL + BS, biarticular ext De + TLo.
F_MAX_CS = 22.5 * np.array([7.6 + 1.9, 2.0 + 5.2, 3.7 + 1.7 + 1.5, 7.7 + 4.4,
                            4.6 + 2.6, 1.4 + 10.4])                   # N: 214 162 155 272 162 266
if PARAMS == "cheng_scott":
    M.F_MAX = F_MAX_CS.copy()
    _torque = M.Plant.torque

    def _torque_per_muscle(self, s, a):
        """The monoarticular-only rung uses the first four muscles."""
        if self.muscles != "mono":
            return _torque(self, s, a)
        M.F_MAX = F_MAX_CS[:4]
        try:
            return _torque(self, s, a)
        finally:
            M.F_MAX = F_MAX_CS.copy()

    M.Plant.torque = _torque_per_muscle
M.L1, M.L2 = 100 * M.l1_m, 100 * M.l2_m   # cm, hand-space bookkeeping
M.a1 = M.I1 + M.I2 + M.m2 * M.l1_m ** 2
M.a2 = M.m2 * M.l1_m * M.s2
M.a3 = M.I2
# Viscosity: the Supplemental gives only the symbols b_ij and cites Todorov (2004)
# for the arm equations; that model uses [[0.05, 0.025], [0.025, 0.05]] (also
# iLQG_Muscle's default). Kalidindi et al.'s code uses [[0.5, 0.1], [0.1, 0.5]].
# Choose with the environment variable MACAQUE_VISCOSITY = todorov | kalidindi | none.
VISCOSITY = os.environ.get("MACAQUE_VISCOSITY", "todorov")
M.VISCOUS = {"todorov": np.array([[0.05, 0.025], [0.025, 0.05]]),
             "kalidindi": np.array([[0.5, 0.1], [0.1, 0.5]]),
             "none": np.zeros((2, 2))}[VISCOSITY]       # diagnostic only
M.CENTER_XY = tuple(M.hand_position(M.CENTER_JOINTS))   # same joint angles: 32.6, 84.2 deg

C.RADIUS = 6.0
REGIME = "6cm_040s"
C.DURATIONS[REGIME] = 0.4
