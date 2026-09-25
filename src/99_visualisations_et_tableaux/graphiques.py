# -*- coding: utf-8 -*-
"""Graphiques matplotlib réutilisables sur les tables de `data/donnees_valides/bkt/` - la version
"hors Excel" des graphiques déjà intégrés au classeur (voir 'export_excel.py'), pour une utilisation
directe dans un notebook, un rapport, ou toute autre restitution.

Chaque fonction prend un DataFrame déjà chargé (pas de lecture de fichier ici - voir 'pipeline.py' ou
les notebooks pour charger les tables) et retourne (fig, ax) : à afficher (`plt.show()`), sauvegarder
(`fig.savefig(...)`) ou insérer ailleurs, au choix de l'appelant."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

COULEUR_OBSERVE = "#7f8c8d"
COULEUR_EXTRAPOLE = "#2a78d6"
COULEUR_MINI = "#eb6834"
COULEUR_IC = "#2a78d6"


def bkt_national(indicateurs: pd.DataFrame, *, titre: str = "BKT national, méthode retenue") -> tuple[plt.Figure, plt.Axes]:
    """BKT observé et extrapolé national, par année - le graphique de synthèse du pipeline."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(indicateurs["annee"], indicateurs["bkt_obs_Mdkm"], marker="o", label="observé (capteurs seulement)", color=COULEUR_OBSERVE)
    ax.plot(indicateurs["annee"], indicateurs["bkt_ext_Mdkm"], marker="o", label="extrapolé (BKT national)", color=COULEUR_EXTRAPOLE)
    ax.set_ylabel("BKT (Md km)")
    ax.set_title(titre)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def bkt_avec_intervalle_confiance(indicateurs: pd.DataFrame, ic: pd.DataFrame, *, colonne_point: str = "bkt_ext_Mdkm", titre: str = "BKT extrapolé national, avec IC 95%") -> tuple[plt.Figure, plt.Axes]:
    """BKT extrapolé (point) avec sa bande d'intervalle de confiance bootstrap à 95% - RAPPEL :
    l'incertitude de NOTRE méthode d'extrapolation, pas celle du "vrai" BKT (voir le README)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(indicateurs["annee"], indicateurs[colonne_point], marker="o", color=COULEUR_EXTRAPOLE, label="BKT_ext (point)")
    ax.fill_between(ic["annee"], ic["ic_bas_95"], ic["ic_haut_95"], alpha=0.2, color=COULEUR_IC, label="IC 95% (bootstrap)")
    ax.set_ylabel("BKT (Md km)")
    ax.set_title(titre)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def bkt_standard_vs_mini(indicateurs: pd.DataFrame, *, titre: str = "BKT standard vs panel équilibré") -> tuple[plt.Figure, plt.Axes]:
    """Compare le BKT extrapolé standard (échantillon de capteurs qui grossit chaque année) au BKT
    "mini" (panel équilibré, échantillon FIXE) - un test de robustesse de la croissance mesurée."""
    fig, ax = plt.subplots(figsize=(7, 4))
    n_standard, n_panel = indicateurs["n_capteurs_actifs"].iloc[-1], indicateurs["n_capteurs_panel"].iloc[0]
    ax.plot(indicateurs["annee"], indicateurs["bkt_ext_Mdkm"], marker="o", label=f"standard ({n_standard} capteurs en {indicateurs['annee'].iloc[-1]})", color=COULEUR_EXTRAPOLE)
    ax.plot(indicateurs["annee"], indicateurs["bkt_mini_ext_Mdkm"], marker="o", label=f"panel équilibré ({n_panel} capteurs, constant)", color=COULEUR_MINI)
    ax.set_ylabel("BKT extrapolé (Md km)")
    ax.set_title(titre)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def indice_base_100(indicateurs: pd.DataFrame, colonnes: dict[str, str], *, titre: str, y_min: float = 90) -> tuple[plt.Figure, plt.Axes]:
    """Plusieurs indicateurs sur un même graphique, ramenés à un indice base 100 = première année -
    comparables entre eux malgré des unités très différentes (population, km, passages/an...).
    'colonnes' : {nom_colonne: libellé affiché}."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for colonne, libelle in colonnes.items():
        indice = indicateurs[colonne] / indicateurs[colonne].iloc[0] * 100
        ax.plot(indicateurs["annee"], indice, marker="o", label=libelle)
    ax.axhline(100, color="#c9c7c1", linewidth=1, linestyle="--")
    ax.set_ylabel("indice (première année = 100)")
    ax.set_ylim(bottom=y_min)
    ax.set_title(titre)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def taux_par_cluster(detail_cluster: pd.DataFrame, annee: int, *, titre: str | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Taux robuste (bike-km / km de réseau non observé) de chaque cluster, pour une année - montre
    quels clusters roulent le plus par km de réseau disponible (voir '06_calcul_du_bkt_extrapole')."""
    ligne = detail_cluster[detail_cluster["annee"] == annee].sort_values("taux_fréquentation")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(ligne["cluster_nom"], ligne["taux_fréquentation"], color=COULEUR_EXTRAPOLE)
    ax.set_xlabel("taux robuste (bike-km / km de réseau)")
    ax.set_title(titre or f"Taux robuste par cluster, {annee}")
    fig.tight_layout()
    return fig, ax
