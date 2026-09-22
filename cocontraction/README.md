# Modèle réseau + 6 muscles (cocontraction)

Remplace l'actionneur à 2 couples de `iLQG_Combined.py` par le modèle à **6 muscles**
de `ILQGbasic.py` (3 paires antagonistes : épaule mono, coude mono, bi-articulaire),
piloté par un réseau aléatoire sous contrôle optimal (iLQG). Le dossier principal
n'est pas modifié.

## Chaîne du modèle (`ilqg_muscle.py`)

```
réseau  ->  6 activations musculaires  ->  forces  ->  couples articulaires  ->  bras
r_dot = W r + u            a_dot = (g(Wout r) - a)/tau_act
F_m = a · fl(l) · fv(l,v)  (par muscle)      tau = A_mom @ F_m
```

État `x = [ r(100), θ1, θ2, ω1, ω2, a1..a6 ]` (dim 110). Le contrôle `u` agit sur les
100 nœuds du réseau.

- **W** : réseau #2 du HDF5 `gaussian_networks.hdf5` (rayon spectral 0.8), inchangé.
- **Wout** : généré 6×N, entrées ~ **N(0, 0.5)** (papier Kalidindi & Crevecoeur 2026,
  Eq. 8, étendu à 6 muscles), de sorte que `Wout @ r` soit de dimension 6 = entrée du
  modèle musculaire. `make_Wout(N)`, graine 0.
- **Mécanique musculaire** (bras de levier `A_mom` 2×6, `l0`, `theta0`, `fl`, `fv`) :
  portée telle quelle depuis `ILQGbasic.py`.
- **RECTIFY** (défaut `True`) : `ILQGbasic` laisse les activations **signées** (non
  physiologiques). Avec `RECTIFY`, l'excitation `Wout r` passe par un rectifieur lisse
  `g(z) = ½(z + √(z²+δ²))` pour garder `a ≥ 0` — indispensable pour une cocontraction
  physiologique. `RECTIFY = False` redonne le comportement signé d'origine.

La jacobienne analytique `fx` est vérifiée par différences finies (erreur max 2·10⁻⁵).

## Utilisation

```bash
python centerout_muscle.py        # résout 8 cibles, cache centerout_muscle.npz
python cocontraction_analysis.py  # figure fig_cocontraction.png + métriques
```

## Résultats

- **La tâche center-out fonctionne** : 8 cibles atteintes, erreur terminale 0.000 cm.
- Activations **99–100 % positives** (muscles physiologiques) avec `RECTIFY`.
- **Cocontraction** (indice antagoniste classique `2·min(a_i,a_j)/(a_i+a_j)`,
  moyenne / pic sur le mouvement) :

  | paire         | moyenne | pic  |
  |---------------|---------|------|
  | épaule        | 0.40    | 1.00 |
  | coude         | 0.48    | 1.00 |
  | bi-articulaire| 0.44    | 1.00 |

  Profil temporel typique : cocontraction **élevée en début/fin de mouvement**
  (stabilisation posturale, couple net ≈ 0) et **plus faible pendant la phase
  balistique** (l'agoniste domine). Niveau ~uniforme selon la direction.

### À noter

Le coût pénalise l'entrée réseau `u` (`r1·|u|²`), **pas** l'effort musculaire. La
cocontraction n'est donc pas régularisée : son niveau (~0.4) reflète la redondance du
système musculaire pilotée par un readout aléatoire. Pour la **moduler** (comme le fait
le SNC selon la stabilité/précision requise), il faudrait ajouter un coût sur l'activation
musculaire ou un bruit dépendant du signal (Harris & Wolpert) — extension naturelle.
