import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = r"D:\thyriona\Desktop\Codes\My_research\WP3\meeting fred\modeles_LS_statique_dynamique.pdf"
H_IN = 11.69
EQ = "#08306b"

DYNAMIC = [
    r"$z(t) = \sigma_z\!\left(W_{filt}\ \sigma_v(W_{in}\,[x(t);\,c(t);\,y^*(t)] + b_v) + b_z\right)$",
    r"$u(t) = \sigma_u(W_{out}\,z(t)), \qquad \tau(t) = M\,[\,u(t) \odot f_{flv}(l, \dot l)\,]$",
    r"$\ddot\theta = \mathcal{M}(\theta)^{-1}\,(\tau - C(\theta,\dot\theta) - B\,\dot\theta), \qquad y = g(x)$",
    r"$J = \sum_i \frac{1}{2}\sum_{t} \Vert g(x(t)) - y^*_i(t)\Vert^2 + \alpha\Vert u(t)\Vert^2 + \beta\Vert z(t)\Vert^2"
    r" \;+\; \frac{\gamma}{2}\Vert w\Vert^2$",
]

STATIC = [
    r"$\dot y = G\,F\,M\,(H \odot \sigma_u(W_{out}\,z))$",
    r"$z^* = \arg\min_z\ \frac{1}{2}\Vert \dot y - \dot y^*\Vert^2 + \frac{\alpha}{2}\Vert u\Vert^2 + \frac{\beta}{2}\Vert z\Vert^2$",
]

TERMS = [
    ("σ_u, W_out, M", "identiques au modèle dynamique.", []),
    ("H", "F-L/V du modèle dynamique, moyennée du début du mouvement à 100 ms après le pic de vitesse : "
          "un facteur par muscle et par cible.", []),
    ("F", "couple → vitesse articulaire : linéarisation de la dynamique du bras f, moyennée sur la "
          "première moitié du mouvement (Δt = T/2). Si l'on néglige les termes de vitesse (Coriolis C, "
          "viscosité B) et le changement de posture, elle se réduit à l'inertie au départ :",
     [r"$F = \frac{1}{\Delta t}\int_0^{\Delta t} \frac{\partial \dot\theta(t)}{\partial \tau}\,dt"
      r" \;\approx\; \frac{\Delta t}{2}\,\mathcal{M}(\theta_0)^{-1}$"]),
    ("G", "vitesse articulaire → vitesse de la main : jacobien de la position de la main g(θ), moyenné "
          "sur la même fenêtre ; pour 6 cm il reste proche du jacobien au départ.",
     [r"$g(\theta) = \left(l_1\cos\theta_1 + l_2\cos(\theta_1+\theta_2),\ \ l_1\sin\theta_1 + l_2\sin(\theta_1+\theta_2)\right)$",
      r"$G = \frac{1}{\Delta t}\int_0^{\Delta t} \frac{\partial g}{\partial\theta}(\theta(t))\,dt"
      r" \;\approx\; \frac{\partial g}{\partial\theta}(\theta_0)$"]),
]


def eq_block(fig, y, eqs, size=10.5):
    for e in eqs:
        tall = r"\frac" in e or r"\left" in e
        fig.text(0.09, y, e, fontsize=size, va="top", color=EQ)
        y -= 0.034 if tall else 0.026
    return y


with PdfPages(OUT) as pdf:
    fig = plt.figure(figsize=(8.27, H_IN))
    fig.text(0.07, 0.965, "Lillicrap & Scott (2013) : modèles dynamique et statique",
             fontsize=14, fontweight="bold", va="top")
    fig.text(0.07, 0.925, "1. Modèle dynamique", fontsize=11, fontweight="bold", va="top", color="#2171b5")
    y = eq_block(fig, 0.895, DYNAMIC)
    fig.text(0.07, y - 0.015, "2. Modèle statique (un seul pas, au début du mouvement)",
             fontsize=11, fontweight="bold", va="top", color="#2171b5")
    y = eq_block(fig, y - 0.045, STATIC)
    y -= 0.008
    for k, v, eqs in TERMS:
        fig.text(0.09, y, k, fontsize=9, fontweight="bold", va="top")
        lines = textwrap.wrap(v, 82)
        fig.text(0.26, y, "\n".join(lines), fontsize=9, va="top", linespacing=1.35)
        y -= 0.010 + 0.0145 * len(lines)
        for e in eqs:
            tall = r"\frac" in e or r"\left" in e
            fig.text(0.26, y - (0.002 if tall else 0), e, fontsize=9.5, va="top", color=EQ)
            y -= 0.032 if tall else 0.024
    pdf.savefig(fig)
print("wrote", OUT)
