"""Dessine, pour chaque cluster, la courbe moyenne (± un écart-type entre capteurs) d'une échelle de
temps donnée - la même image sert à la fois de figure "de contrôle" (PNG sur disque) et de vignette
intégrée dans le panneau de la carte interactive ('carte.py'), pour ne pas dupliquer le calcul.

explication du paramètre PROFIL :
- "journalier"    courbe des 24 heures de la journée (colonnes 'd00..d23')
- "hebdomadaire"  courbe des 168 heures de la semaine (colonnes 'h000..h167')
- "annuel"        courbe des 52 semaines de l'année (colonnes 'y00..y51')
- "tous"          les trois courbes ci-dessus, une par échelle

explication du paramètre SAVE_AS (où écrire/renvoyer le résultat) :
- False   rien n'est écrit sur disque - la fonction renvoie les figures matplotlib elles-mêmes
          (pratique pour un simple coup d'oeil dans un notebook, 'plt.show()'-style).
- "png"   chaque courbe est écrite en PNG dans 'dossier_sortie' ; la fonction renvoie les chemins.
- True    RIEN n'est écrit sur disque - chaque courbe est encodée en base64, prête à être intégrée
          directement dans le HTML de la carte interactive ('carte.py') ; la fonction renvoie ces
          chaînes base64. C'est le mode utilisé PAR 'carte_capteurs()'.
- "both"  les deux à la fois : les PNG sont écrits sur disque ET les chaînes base64 sont renvoyées.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "00_transformation_des_donnees"
        / "d_debit"
    ),
)
from profils_usage import COLONNES_ANNEE, COLONNES_JOUR, COLONNES_SEMAINE  # noqa: E402

MOIS_FR = [
    "Jan",
    "Fév",
    "Mar",
    "Avr",
    "Mai",
    "Juin",
    "Juil",
    "Août",
    "Sep",
    "Oct",
    "Nov",
    "Déc",
]

# une échelle = les colonnes du profil, et comment légender son axe des abscisses
ECHELLES = {
    "journalier": {
        "colonnes": COLONNES_JOUR,
        "titre": "Journalier",
        "xlabel": "Heure",
        "xticks": np.arange(0, 24, 3),
        "xticklabels": None,
    },
    "hebdomadaire": {
        "colonnes": COLONNES_SEMAINE,
        "titre": "Hebdomadaire",
        "xlabel": "Heure",
        "xticks": np.arange(0, 168, 24),
        "xticklabels": ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"],
    },
    "annuel": {
        "colonnes": COLONNES_ANNEE,
        "titre": "Annuel",
        "xlabel": "Semaine",
        "xticks": np.arange(0, 52, 4),
        "xticklabels": [
            MOIS_FR[(semaine * 7 // 30) % 12] for semaine in range(0, 52, 4)
        ],
    },
}


def _echelles_demandees(profil: str) -> list[str]:
    if profil == "tous":
        return list(ECHELLES)
    if profil not in ECHELLES:
        raise ValueError(f"profil={profil!r} attendu parmi {['tous', *ECHELLES]}")
    return [profil]


def _norme_unitaire(X: np.ndarray) -> np.ndarray:
    """Même convention que 'representation.py' : on compare des FORMES de courbe, pas des volumes."""
    normes = np.linalg.norm(X, axis=1, keepdims=True)
    normes[normes == 0] = 1.0
    return X / normes


def _dessiner_courbe(
    matrice: np.ndarray, couleur: str, titre: str, echelle: dict, y_max: float
):
    """Une figure matplotlib : la moyenne du cluster en trait plein, ± un écart-type entre capteurs en
    zone ombrée. Renvoie la 'Figure' (pas encore encodée/sauvegardée - voir '_exporter_figure')."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.3, 2.7), dpi=110)
    moyenne, ecart = np.nanmean(matrice, axis=0), np.nanstd(matrice, axis=0)
    x = np.arange(matrice.shape[1])
    ax.fill_between(
        x,
        np.maximum(moyenne - ecart, 0),
        moyenne + ecart,
        color=couleur,
        alpha=0.2,
        linewidth=0,
    )
    ax.plot(x, moyenne, color=couleur, linewidth=1.8)
    ax.set_ylim(0, y_max)
    ax.set_xticks(echelle["xticks"])
    if echelle["xticklabels"] is not None:
        ax.set_xticklabels(echelle["xticklabels"], fontsize=7)
    ax.set_xlabel(echelle["xlabel"], fontsize=8)
    ax.set_ylabel("Proportion", fontsize=8)
    ax.set_title(titre, fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _exporter_figure(
    fig, save_as: bool | str, chemin_png: Path | None
) -> str | Path | object:
    """Applique le mode 'save_as' à UNE figure déjà dessinée : chemin PNG écrit, chaîne base64, ou la
    figure elle-même selon le mode (voir le docstring du module)."""
    import matplotlib.pyplot as plt

    if save_as is False:
        return fig  # rien à écrire : on rend la main à l'appelant (ex. affichage direct en notebook)

    resultat = None
    if save_as in ("png", "both"):
        chemin_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(chemin_png)
        resultat = chemin_png
    if save_as in (True, "both"):
        tampon = io.BytesIO()
        fig.savefig(tampon, format="png")
        b64 = base64.b64encode(tampon.getvalue()).decode("ascii")
        resultat = b64 if save_as is True else {"png": resultat, "base64": b64}
    plt.close(fig)
    return resultat


def construire_images_profils(
    profils: pd.DataFrame,
    *,
    colonne_cluster: str = "cluster",
    profil: str = "tous",
    save_as: bool | str = False,
    couleurs: dict | None = None,
    noms_clusters: dict | None = None,
    dossier_sortie: Path | str | None = None,
) -> dict:
    """Une image par (échelle demandée, cluster présent dans 'profils').

    'profils' : une ligne par capteur, avec les colonnes de profil ('d00..d23' etc., voir
    'representation.py') ET une colonne 'colonne_cluster' (le cluster de chaque capteur - un
    entier ou un nom, peu importe : sert seulement de clé de regroupement et de clé dans 'couleurs').

    'couleurs' : '{valeur_de_cluster: couleur}' - une couleur par défaut (palette matplotlib) est
    utilisée pour les clusters absents de ce dictionnaire.

    'noms_clusters' : '{valeur_de_cluster: nom affiché}', optionnel - sinon la valeur brute du
    cluster sert de nom dans le titre de chaque image.

    Renvoie '{echelle: {valeur_de_cluster: résultat}}' où 'résultat' dépend de 'save_as' (figure
    matplotlib, chemin PNG, chaîne base64, ou les deux - voir le docstring du module).
    """
    if save_as in ("png", "both") and dossier_sortie is None:
        raise ValueError("save_as='png'/'both' nécessite dossier_sortie")
    dossier_sortie = Path(dossier_sortie) if dossier_sortie is not None else None

    valeurs_cluster = sorted(
        {
            v.item() if hasattr(v, "item") else v
            for v in profils[colonne_cluster].dropna().unique()
        },
        key=str,
    )
    palette_defaut = [
        "#2a78d6",
        "#eb6834",
        "#eda100",
        "#1baf7a",
        "#8e44ad",
        "#c0392b",
        "#16a085",
        "#7f8c8d",
    ]
    couleurs = dict(couleurs) if couleurs else {}
    for i, v in enumerate(valeurs_cluster):
        couleurs.setdefault(v, palette_defaut[i % len(palette_defaut)])
    noms_clusters = noms_clusters or {}

    images: dict[str, dict] = {}
    for echelle_nom in _echelles_demandees(profil):
        echelle = ECHELLES[echelle_nom]
        matrices = {}
        for v in valeurs_cluster:
            sous_table = profils.loc[profils[colonne_cluster] == v, echelle["colonnes"]]
            if sous_table.empty:
                continue
            matrices[v] = _norme_unitaire(sous_table.to_numpy(dtype=float))
        if not matrices:
            continue
        # même échelle verticale pour tous les clusters d'une échelle temporelle donnée - comparaison directe possible
        y_max = (
            max(
                (np.nanmean(m, axis=0) + np.nanstd(m, axis=0)).max()
                for m in matrices.values()
            )
            * 1.05
        )

        images[echelle_nom] = {}
        for v, matrice in matrices.items():
            titre = f"{noms_clusters.get(v, v)} — {echelle['titre']}"
            fig = _dessiner_courbe(matrice, couleurs[v], titre, echelle, y_max)
            chemin_png = (
                dossier_sortie / f"profil_{echelle_nom}_cluster_{v}.png"
                if dossier_sortie
                else None
            )
            images[echelle_nom][v] = _exporter_figure(fig, save_as, chemin_png)

    return images
