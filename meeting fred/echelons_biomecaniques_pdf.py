import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = r"D:\thyriona\Desktop\Codes\My_research\WP3\meeting fred\echelons_biomecaniques.pdf"
H_IN = 11.69

COMMON = [
    ("Réseau", "N = 100 unités. W = (gX − I)/τ_r, avec X_ij ~ N(0, 1/N) tirés indépendamment, g = 0.8 "
               "et τ_r = 20 ms. W est donc stable (valeurs "
               "propres de partie réelle < 0). 10 réseaux tirés ainsi. L'entrée u (une par unité) est "
               "optimisée par iLQG. Lecture fixe aléatoire Wout ~ N(0, 1/N) vers les actionneurs.",
     [r"$\dot r = W\,r + u$"]),
    ("Bras", "Macaque, Cheng & Scott (2000) : masses 0.294 / 0.251 kg, longueurs 14.4 / 20.6 cm. "
             "Départ : épaule 32.6°, coude 84.2° (L&S, Fig. 2B).", []),
    ("Tâche et coût", "Centre-out, 16 cibles, 6 cm en 0.4 s. Coût terminal sur la position articulaire "
                      "(1e4) et la vitesse (10), plus un effort 1e-5·‖u‖².", []),
    ("Réseaux", "20 couples (réseau, lecture) : 10 réseaux × 2 tirages de Wout. Mêmes couples à tous "
                "les échelons.", []),
    ("PD", "Activité moyenne du début du mouvement au pic de vitesse.", []),
    ("Axe", "Chaque unité i a un PD φ_i (angle de 0 à 360°). On double les angles, 2φ_i, pour qu'une "
            "direction et son opposée tombent au même endroit, on fait la moyenne vectorielle de ces "
            "angles doublés sur toutes les unités des 20 couples, puis on divise l'angle obtenu par 2. "
            "L'axe θ est donc une orientation, définie de 0 à 180°.",
     [r"$\theta = \frac{1}{2}\,\arg\,\left(\frac{1}{n}\,\Sigma_i\; e^{\,\mathrm{i}\,2\varphi_i}\right)$"]),
    ("Muscles", "Échelons 4 à 6 : activation rectifiée, les muscles ne font que tirer.",
     [r"$\sigma_u(x) = e^{2x-5}\ \ (x<1/2), \quad \sigma_u(x) = 2x\,e^{-4}\ \ (x\geq 1/2)$"]),
    ("Viscosité", "Seule différence entre les deux figures. Todorov : B = [[0.05, 0.025], [0.025, 0.05]]. "
                  "Kalidindi : B = [[0.5, 0.1], [0.1, 0.5]] (N·m·s/rad).", []),
]

ACT = r"\tau_a\,\dot a = W_{out}\,r - a"
ACT_M = r"\tau_a\,\dot a = \sigma_u(W_{out}\,r) - a"
ARM = r"M(\theta)\,\ddot\theta + C(\theta,\dot\theta) + B\,\dot\theta = R\,T"
SEP = r"\qquad "
RUNGS = [
    ("1", "Masse ponctuelle", "Masse de 1 kg dans l'espace de la main, poussée par 2 forces "
          "cartésiennes (x, y). Pas de bras, pas de viscosité, cible en coordonnées main.",
     ["$" + ACT + SEP + r"m\,\ddot p = a$"]),
    ("2", "+ géométrie", "Bras à 2 articulations, mû par 2 couples articulaires. Inertie diagonale "
          "(pas de couplage épaule-coude), sans Coriolis ni centripète, viscosité diagonale.",
     ["$" + ACT + SEP + r"\mathrm{diag}(M(\theta))\,\ddot\theta + \mathrm{diag}(B)\,\dot\theta = a$"]),
    ("3", "+ dynamique\nintersegmentaire", "Inertie complète (couplage épaule-coude), forces de "
          "Coriolis et centripètes, matrice de viscosité complète. Toujours 2 couples.",
     ["$" + ACT + SEP + r"M(\theta)\,\ddot\theta + C(\theta,\dot\theta) + B\,\dot\theta = a$"]),
    ("4", "+ muscles\nmonoarticulaires", "4 muscles : fléchisseur et extenseur de l'épaule, puis du coude. "
          "Bras de levier constants de 2 cm. Pas de F-L/V.",
     ["$" + ACT_M + SEP + r"T = F_{max} \odot a$", "$" + ARM + r", \quad R \in \mathbb{R}^{2\times 4}$"]),
    ("5", "+ muscles\nbiarticulaires", "On ajoute un fléchisseur et un extenseur biarticulaires, "
          "soit 6 muscles.",
     ["$" + ACT_M + SEP + r"T = F_{max} \odot a$", "$" + ARM + r", \quad R \in \mathbb{R}^{2\times 6}$"]),
    ("6", "+ mécanique\nmusculaire", "Force-longueur et force-vitesse de Brown "
          "et al. (1999), recalculées à chaque instant à partir des angles.",
     ["$" + ACT_M + SEP + r"T = F_{max} \odot a \odot f_l(l) \odot f_v(l, \dot l)$",
      "$" + ARM + SEP + r"l = l_c - A^\top\theta, \quad \dot l = -A^\top\dot\theta$"]),
]


def para(fig, x, y, text, width, size=8.5, **kw):
    lines = textwrap.wrap(text, width)
    fig.text(x, y, "\n".join(lines), fontsize=size, va="top", linespacing=1.35, **kw)
    return y - len(lines) * size * 1.35 / 72 / H_IN - 0.004


def equations(fig, x, y, eqs, size=9):
    for e in eqs:
        y -= 0.003
        fig.text(x, y - (0.003 if r"rac" in e else 0), e, fontsize=size, va="top", color="#08306b")
        y -= size * 1.55 / 72 / H_IN + (0.01 if r"rac" in e else 0)
    return y


with PdfPages(OUT) as pdf:
    fig = plt.figure(figsize=(8.27, H_IN))
    fig.text(0.07, 0.965, "Fig. 5 de Lillicrap & Scott (2013) : construction des échelons",
             fontsize=14, fontweight="bold", va="top")
    y = 0.935
    fig.text(0.07, y, "Commun à tous les échelons", fontsize=11, fontweight="bold", va="top", color="#2171b5")
    y -= 0.022
    for k, v, eqs in COMMON:
        fig.text(0.07, y, k, fontsize=8.5, fontweight="bold", va="top")
        y = equations(fig, 0.23, para(fig, 0.23, y, v, 96), eqs) - (0.006 if eqs else 0.002)
    y -= 0.01
    fig.text(0.07, y, "Échelons", fontsize=11, fontweight="bold", va="top", color="#2171b5")
    y -= 0.022
    for n, name, desc, eqs in RUNGS:
        fig.text(0.07, y, n, fontsize=10, fontweight="bold", va="top")
        fig.text(0.10, y, name, fontsize=8.5, fontweight="bold", va="top", linespacing=1.35)
        y2 = equations(fig, 0.33, para(fig, 0.33, y, desc, 82), eqs)
        y = min(y2, y - (name.count("\n") + 1) * 0.0155 - 0.006) - 0.006
        fig.add_artist(plt.Line2D([0.07, 0.93], [y + 0.004, y + 0.004], color="0.85", lw=0.6))
    pdf.savefig(fig)
print("wrote", OUT)
