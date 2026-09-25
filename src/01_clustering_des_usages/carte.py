"""Carte interactive (Leaflet/HTML, via folium) des capteurs, colorés par cluster d'usage : légende
cliquable pour isoler un ou plusieurs clusters, panneau à onglets avec le profil moyen (un onglet par
échelle demandée, voir 'profils.py') de chaque cluster, plusieurs fonds de carte (bouton en haut à
droite). Générique sur le nombre de clusters : leurs couleurs et leurs noms viennent des données et
des paramètres, rien n'est câblé en dur pour K=4.

Toujours exportée en HTML : c'est une carte interactive par nature (profils, filtre de cluster, fond de
carte) - une image statique n'aurait pas grand sens ici.

Les gros blocs CSS/JS/HTML de la carte sont dans des fichiers séparés (dossier 'gabarits/'), pas dans
ce fichier Python - voir '_css()'/'_js()'/'_legende_html()' plus bas, qui se contentent de les lire
et d'y substituer les valeurs de CETTE carte (couleurs, images de profil...).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
from profils import ECHELLES, construire_images_profils

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "00_transformation_des_donnees"
        / "d_zones"
    ),
)
from zones import DEPARTEMENTS_METROPOLE  # noqa: E402

GABARITS = Path(__file__).resolve().parent / "gabarits"
COLONNES_REQUISES = ("id_site", "nom_site", "lat", "long", "cluster")


def _circle_svg(couleur: str, r: int = 6) -> str:
    return f'<svg width="{2 * r}" height="{2 * r}"><circle cx="{r}" cy="{r}" r="{r - 1}" fill="{couleur}" stroke="#333" stroke-width="0.8"/></svg>'


def _secteurs_svg(couleurs: list[str], r: int = 6) -> str:
    """Plusieurs capteurs au même point (même commune, plusieurs sites) : un disque à secteurs, un
    secteur par capteur, au lieu d'un marqueur qui en cache un autre."""
    if len(couleurs) == 1:
        return _circle_svg(couleurs[0], r)
    n = len(couleurs)
    parts = []
    for i, c in enumerate(couleurs):
        a0, a1 = (
            2 * math.pi * (i / n) - math.pi / 2,
            2 * math.pi * ((i + 1) / n) - math.pi / 2,
        )
        x1, y1 = r + r * math.cos(a0), r + r * math.sin(a0)
        x2, y2 = r + r * math.cos(a1), r + r * math.sin(a1)
        large = 1 if (a1 - a0) > math.pi else 0
        parts.append(
            f'<path d="M {r} {r} L {x1} {y1} A {r} {r} 0 {large} 1 {x2} {y2} Z" fill="{c}" stroke="#333" stroke-width="0.5"/>'
        )
    return f'<svg width="{2 * r}" height="{2 * r}">{"".join(parts)}</svg>'


def _palette_completee(valeurs_cluster: list, couleurs: dict | None) -> dict:
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
    return couleurs


def _legende_html(titre_modele: str) -> str:
    return (
        (GABARITS / "legende.html")
        .read_text(encoding="utf-8")
        .replace("__TITRE_MODELE__", titre_modele)
    )


def _css() -> str:
    return f"<style>\n{(GABARITS / 'carte_capteurs.css').read_text(encoding='utf-8')}\n</style>"


def _js(
    tabs: list[dict],
    images: dict,
    couleurs: dict,
    noms: dict,
    counts: dict,
    valeurs_cluster: list,
    marqueurs: list,
    bounds: list,
    nom_carte: str,
) -> str:
    gabarit = (GABARITS / "carte_capteurs.js").read_text(encoding="utf-8")
    return (
        gabarit.replace("__IMAGES__", json.dumps(images))
        .replace("__COLORS__", json.dumps({str(k): v for k, v in couleurs.items()}))
        .replace(
            "__NAMES__",
            json.dumps({str(k): v for k, v in noms.items()}, ensure_ascii=False),
        )
        .replace("__COUNTS__", json.dumps({str(k): v for k, v in counts.items()}))
        .replace("__CLUSTER_IDS__", json.dumps([str(v) for v in valeurs_cluster]))
        .replace("__MARKERS__", json.dumps(marqueurs))
        .replace("__BOUNDS__", json.dumps(bounds))
        .replace("__MAPNAME__", nom_carte)
        .replace("__TABS__", json.dumps(tabs, ensure_ascii=False))
    )


def carte_capteurs(
    donnees: pd.DataFrame,
    *,
    couleurs: dict | None = None,
    noms_clusters: dict | None = None,
    profil: str = "tous",
    titre: str = "Clustering d'usage",
    sortie: Path | str | None = None,
) -> Path:
    """Construit la carte des capteurs et l'écrit sur disque (HTML).

    'donnees' : UNE LIGNE PAR CAPTEUR, avec les colonnes 'id_site, nom_site, lat, long, cluster'
    (le cluster de CE capteur, un entier ou un nom - sert de clé dans 'couleurs'/'noms_clusters') et
    les colonnes de profil nécessaires à 'profil' ('d00..d23' / 'h000..h167' / 'y00..y51' - voir
    'profils.py'/'representation.py'). C'est typiquement la table renvoyée par
    'appliquer_modele.assigner_clusters' FUSIONNÉE avec les coordonnées et les profils des capteurs
    (voir le notebook de ce dossier pour un exemple complet).

    'couleurs' : '{valeur_de_cluster: couleur}' - une couleur par défaut est utilisée pour les
    clusters absents de ce dictionnaire (voir '_palette_completee').

    'noms_clusters' : '{valeur_de_cluster: nom affiché}', optionnel - sinon la valeur brute du
    cluster (ex. '0', '1'...) sert de nom dans la légende.

    'profil' : quelle(s) échelle(s) de profil afficher dans le panneau (voir 'profils.py' -
    "journalier", "hebdomadaire", "annuel" ou "tous").

    Si 'donnees' a une colonne 'code_dept', les capteurs hors de la France métropolitaine (Corse
    incluse - même périmètre que le reste du projet, voir 'DEPARTEMENTS_METROPOLE') sont retirés AVANT
    de tracer la carte : quelques capteurs ultramarins isolés (La Réunion...) forcent sinon un cadrage
    mondial qui écrase la France métropolitaine à l'ouverture. Sans cette colonne, aucun filtre n'est
    appliqué - passez-la si vos données peuvent contenir des capteurs hors métropole.
    """
    manquantes = [c for c in COLONNES_REQUISES if c not in donnees.columns]
    if manquantes:
        raise ValueError(f"colonnes manquantes dans 'donnees' : {manquantes}")

    donnees = donnees.dropna(subset=["lat", "long", "cluster"]).reset_index(drop=True)
    if "code_dept" in donnees.columns:
        code_dept = (
            donnees["code_dept"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(2)
        )  # "974.0" -> "974", "2.0" -> "02"
        avant = len(donnees)
        donnees = donnees[code_dept.isin(DEPARTEMENTS_METROPOLE)].reset_index(drop=True)
        if len(donnees) < avant:
            print(
                f"[carte] {avant - len(donnees)} capteur(s) hors France métropolitaine retiré(s) (colonne 'code_dept')"
            )
    if donnees.empty:
        raise ValueError("aucun capteur avec cluster + coordonnées à afficher")
    # types Python natifs (pas numpy.int64) : sinon la sérialisation JSON des dictionnaires couleurs/noms/
    # images côté _js() échoue sur des clés numpy - voir la même remarque dans profils.py
    donnees["cluster"] = donnees["cluster"].map(
        lambda v: v.item() if hasattr(v, "item") else v
    )

    valeurs_cluster = sorted(donnees["cluster"].unique(), key=str)
    couleurs = _palette_completee(valeurs_cluster, couleurs)
    noms_clusters = noms_clusters or {}
    counts = {v: int(n) for v, n in donnees["cluster"].value_counts().items()}

    # étape 1 : les images de profil (une par échelle demandée x cluster), en base64 pour le panneau -
    # c'est 'profils.py' qui sait dessiner une courbe, cette fonction ne fait que les assembler dans la carte.
    images = construire_images_profils(
        donnees,
        profil=profil,
        save_as=True,
        couleurs=couleurs,
        noms_clusters=noms_clusters,
    )
    tabs = [{"cle": echelle, "titre": ECHELLES[echelle]["titre"]} for echelle in images]

    import folium

    # NB : ni tuiles CartoDB ("cartodbpositron", exige une clé API depuis 2025) ni tuiles OpenStreetMap
    # officielles (tile.openstreetmap.org bloque le hotlinking/l'usage programmatique - 403 "tile usage
    # policy") - remplacées par des couches Esri, gratuites, sans clé, pas de blocage connu (même choix
    # qu'aux autres cartes de ce dépôt, voir '99_visualisations_et_tableaux/rendu.py').
    m = folium.Map(
        location=[46.6, 2.5],
        zoom_start=6,
        tiles=None,
        prefer_canvas=True,
        zoom_control=False,
    )
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        name="Rues",
        attr="Esri",
        show=False,
    ).add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        name="Fond gris",
        attr="Esri",
        show=True,
    ).add_to(m)
    grp = folium.FeatureGroup(name="Capteurs", show=True).add_to(m)

    # étape 2 : un marqueur par POSITION (pas par capteur - plusieurs capteurs peuvent partager un
    # point) ; un disque simple pour un capteur seul, un disque à secteurs sinon (voir _secteurs_svg).
    donnees = donnees.copy()
    donnees["coord_key"] = (
        donnees["lat"].round(6).astype(str) + "," + donnees["long"].round(6).astype(str)
    )
    marqueurs = []
    for _, g in donnees.groupby("coord_key", sort=False):
        g = g.reset_index(drop=True)
        lat, lon = float(g.loc[0, "lat"]), float(g.loc[0, "long"])
        if len(g) == 1:
            c = g.loc[0, "cluster"]
            svg, size = _circle_svg(couleurs[c]), (12, 12)
            popup = f"<b>{g.loc[0, 'nom_site']}</b><br>{noms_clusters.get(c, c)}"
        else:
            cols = [couleurs[c] for c in g["cluster"]]
            svg = _secteurs_svg(cols) if len(g) <= 8 else _circle_svg(cols[0], 7)
            size = (14, 14)
            popup = "<br>".join(
                f"<b>{r.nom_site}</b> - {noms_clusters.get(r.cluster, r.cluster)}"
                for r in g.itertuples()
            )
        icon = folium.DivIcon(
            html=f'<div style="font-size:0;line-height:0">{svg}</div>',
            icon_size=size,
            icon_anchor=(size[0] // 2, size[1] // 2),
        )
        mk = folium.Marker(
            [lat, lon], icon=icon, popup=folium.Popup(popup, max_width=340)
        ).add_to(grp)
        marqueurs.append(
            {
                "marker_name": mk.get_name(),
                "clusters": sorted({str(c) for c in g["cluster"]}),
            }
        )

    folium.LayerControl(collapsed=False).add_to(m)
    bounds = [
        [float(donnees["lat"].min()), float(donnees["long"].min())],
        [float(donnees["lat"].max()), float(donnees["long"].max())],
    ]

    # étape 3 : légende + CSS + JS (tous lus depuis gabarits/, voir le docstring du module) injectés
    # tels quels dans le HTML de la carte folium.
    root = m.get_root()
    root.html.add_child(folium.Element(_legende_html(titre)))
    root.html.add_child(folium.Element(_css()))
    root.html.add_child(
        folium.Element(
            _js(
                tabs,
                images,
                couleurs,
                noms_clusters,
                counts,
                valeurs_cluster,
                marqueurs,
                bounds,
                m.get_name(),
            )
        )
    )

    out = (
        Path(sortie)
        if sortie
        else Path(__file__).resolve().parents[2]
        / "outputs"
        / "cartes"
        / "carte_capteurs.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(
        f"[carte] écrit : {out}  ({out.stat().st_size / 1024:.0f} Ko, {len(donnees):,} capteurs, {len(marqueurs):,} marqueurs, {len(valeurs_cluster)} clusters)"
    )
    return out
