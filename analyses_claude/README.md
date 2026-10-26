# analyses_claude

Analyses of the random-network + iLQG model: signatures of arm and muscle
nonlinearities in network activity, and a closed-loop redo of Lillicrap & Scott
(2013) Figure 5. Simulation outputs go to `sims/` (not versioned); every
script regenerates what it needs. Run from `WP3/PhD_WP3`, e.g.
`python analyses_claude/fig5_macaque.py`.

## Shared pieces
| file | role |
|---|---|
| `muscle_plant.py` | `sigma_u` (missing from `iLQG_Muscle.py`, restored from the L&S Supplemental, breakpoint 1/2), linearized-arm plant, warm-start solver |
| `compare3.py` | model zoo with a joint-space cost (exactly LQ for `lin2`): `lin2`, `nl2`, `nl6`, `lin6`, `nl6x`, `lin6L`, `lin6V`, `lin6f`, `pm`, `geo`, `mono` |
| `macaque.py` | switches to the L&S macaque arm (Supplemental) and task (6 cm, 0.4 s); viscosity via `MACAQUE_VISCOSITY=todorov|kalidindi` |
| `erank.py` | erank (Roy & Vetterli), even/odd split over opposite targets |

## Analyses
| script | output | question |
|---|---|---|
| `fig_null_model.py` | `fig_null_model.png` | torque model: condition rank and dR2_even, linearized vs nonlinear arm |
| `erank.py` | `fig_erank.png`, `erank_table.txt` | erank of torque-model activity across regimes |
| `muscle_centerout.py`, `muscle_analysis.py` | `fig_muscle_flv.png`, `muscle_table.txt` | F-L/V variants, hand-space cost |
| `fig_erank_3models.py` | `fig_erank_3models.png` | erank: linear / nonlinear / nonlinear + F-L/V |
| `compare3.py`, `compare3_analysis.py` | `fig_compare3.png` | lin2 vs nl2 vs nl6 with a common joint-space cost |
| `signatures.py`, `net_variability.py`, `fig_signatures.py` | `fig_signatures.png` | signatures measurable in monkey data (PD axis, PD reversal), noise and network robustness |
| `ladder.py` | `fig_ladder.png` | Fig. 5 ladder, axis and r |
| `fig5_overlay.py`, `fig5_meanflv.py`, `fig5_400ms.py` | `fig5_overlay_12cm_*.png` | Fig. 5 centre-out rows, instantaneous vs mean F-L/V |
| `fig5_macaque.py` | `fig5_overlay_6cm_040s_macaque_*.png` | Fig. 5 with the macaque arm and the L&S task |

## Caveats
- Every muscle result **except** `fig5_overlay_6cm_040s_macaque_*` was produced
  with an earlier, wrong reconstruction of `sigma_u` (quadratic ramp, baseline
  co-contraction 1/8). Re-run those scripts before relying on them.
- The L&S Supplemental gives no numerical viscosity. It sets the rung-3 axis:
  ~107 deg with B = [[0.5, 0.1], [0.1, 0.5]], ~164 deg with [[0.05, 0.025],
  [0.025, 0.05]], paper 131 deg; a closed-form estimate with (kappa M + B) J^-1
  predicts both and points to b11 ~ 0.22 with b12 = b11 / 5 (not simulated).
- One network per analysis unless stated (`net_variability.py` uses six); no
  sensory delay, motor noise or preparatory epoch.
