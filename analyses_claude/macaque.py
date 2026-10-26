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

M.I1, M.I2 = 0.025, 0.045             # kg m^2
M.m1, M.m2 = 0.2108, 0.1938           # kg
M.l1_m, M.l2_m = 0.145, 0.284         # m
M.s1, M.s2 = 0.0749, 0.0757           # m, joint -> centre of mass
M.L1, M.L2 = 14.5, 28.4               # cm, hand-space bookkeeping
M.a1 = M.I1 + M.I2 + M.m2 * M.l1_m ** 2
M.a2 = M.m2 * M.l1_m * M.s2
M.a3 = M.I2
# Viscosity: the Supplemental gives only the symbols b_ij and cites Todorov (2004)
# for the arm equations; that model uses [[0.05, 0.025], [0.025, 0.05]] (also
# iLQG_Muscle's default). Kalidindi et al.'s code uses [[0.5, 0.1], [0.1, 0.5]].
# Choose with the environment variable MACAQUE_VISCOSITY = todorov | kalidindi.
VISCOSITY = os.environ.get("MACAQUE_VISCOSITY", "todorov")
M.VISCOUS = {"todorov": np.array([[0.05, 0.025], [0.025, 0.05]]),
             "kalidindi": np.array([[0.5, 0.1], [0.1, 0.5]])}[VISCOSITY]
M.CENTER_XY = tuple(M.hand_position(M.CENTER_JOINTS))   # same joint angles: 32.6, 84.2 deg

C.RADIUS = 6.0
REGIME = "6cm_040s"
C.DURATIONS[REGIME] = 0.4
