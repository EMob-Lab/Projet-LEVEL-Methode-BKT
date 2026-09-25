# -*- coding: utf-8 -*-
"""Rendu partagé des cartes de communes (voir 'cartes.py') : un GeoDataFrame déjà catégorisé (une
colonne 'categorie_col' + une couleur par valeur, dans 'couleurs') devient soit une image PNG
(matplotlib, 'dpi' réglable), soit une carte HTML interactive (folium, plusieurs fonds de carte,
survol = infobulle). Les deux acceptent 'bordures' : dessiner ou non le contour de chaque commune, et
de quelle couleur - le même réglage se comporte pareil dans les deux rendus, pour que PNG et HTML
restent visuellement cohérents."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import geopandas as gpd

Bordures = Literal["aucune", "neutres", "colorees"]
"""'aucune' : pas de contour du tout (remplissage seul). 'neutres' (par défaut) : un liseré gris
discret, identique pour toutes les catégories. 'colorees' : le contour reprend la couleur de la
catégorie (plus visible, utile quand deux catégories voisines ont des couleurs proches)."""


def _bordure_png(bordures: Bordures, couleur_categorie: str, largeur: float) -> dict:
    if bordures == "aucune":
        return {"edgecolor": "none", "linewidth": 0}
    if bordures == "colorees":
        return {"edgecolor": couleur_categorie, "linewidth": largeur}
    return {"edgecolor": "#555555", "linewidth": largeur}  # "neutres"


def carte_png(
    gdf: gpd.GeoDataFrame,
    *,
    categorie_col: str,
    couleurs: dict[str, str],
    titre: str,
    sortie: Path,
    dpi: int = 200,
    bordures: Bordures = "neutres",
    largeur_bordure: float = 0.15,
    figsize: tuple[float, float] = (11, 11),
) -> Path:
    """Image statique. 'couleurs' fixe aussi l'ORDRE de la légende (un dict Python garde son ordre
    d'insertion) - les catégories absentes du GeoDataFrame sont juste sautées, pas une erreur."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)
    for cat, coul in couleurs.items():
        sous_ensemble = gdf[gdf[categorie_col] == cat]
        if sous_ensemble.empty:
            continue
        sous_ensemble.plot(ax=ax, color=coul, **_bordure_png(bordures, coul, largeur_bordure))
    poignees = [
        Patch(facecolor=coul, edgecolor=("none" if bordures == "aucune" else "#555555"), label=f"{cat} ({(gdf[categorie_col] == cat).sum():,})")
        for cat, coul in couleurs.items()
        if (gdf[categorie_col] == cat).any()
    ]
    ax.legend(handles=poignees, loc="lower left", fontsize=9, framealpha=0.95, title_fontsize=10)
    ax.set_axis_off()
    ax.set_title(titre, fontsize=14, fontweight="bold")
    fig.tight_layout()
    sortie.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sortie, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return sortie


def carte_html(
    gdf: gpd.GeoDataFrame,
    *,
    categorie_col: str,
    couleurs: dict[str, str],
    titre: str,
    sortie: Path,
    popup_cols: list[str] | None = None,
    bordures: Bordures = "neutres",
    largeur_bordure: float = 1.0,
    simplification_m: float = 200.0,
) -> Path:
    """Carte interactive : plusieurs fonds de carte (bouton en haut à droite), survol = infobulle
    ('categorie_col' + 'popup_cols'), légende fixe en haut à droite (compte par catégorie).

    'simplification_m' : tolérance de simplification des contours (mètres) AVANT export - sans elle,
    les 34 428 communes à pleine précision IGN produisent un HTML énorme (injouable dans un
    navigateur). 200 m est invisible à l'échelle nationale ; réduire (voire 0, pour désactiver) si la
    carte ne couvre qu'une petite zone où le détail du contour redevient visible."""
    import folium

    gdf_wgs = gdf.to_crs(4326) if gdf.crs is not None and gdf.crs.to_epsg() != 4326 else gdf
    if simplification_m > 0:
        gdf_wgs = gdf_wgs.copy()
        gdf_wgs["geometry"] = gdf_wgs.to_crs(2154).geometry.simplify(simplification_m).to_crs(4326)
    centre = gdf_wgs.geometry.union_all().centroid
    m = folium.Map(location=[centre.y, centre.x], zoom_start=6, tiles=None)
    # ni tuiles CartoDB (exige une clé API depuis 2025) ni tuiles OpenStreetMap officielles (bloquent le
    # hotlinking/l'usage programmatique) - couches Esri à la place, gratuites, sans clé.
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        name="Fond clair", attr="Esri", show=True,
    ).add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        name="Rues", attr="Esri", show=False,
    ).add_to(m)

    def fonction_style(feature):
        cat = feature["properties"][categorie_col]
        coul = couleurs.get(cat, "#999999")
        if bordures == "aucune":
            return {"fillColor": coul, "color": coul, "weight": 0, "fillOpacity": 0.85}
        if bordures == "colorees":
            return {"fillColor": coul, "color": coul, "weight": largeur_bordure, "fillOpacity": 0.85}
        return {"fillColor": coul, "color": "#555555", "weight": largeur_bordure * 0.5, "fillOpacity": 0.85}

    champs = [categorie_col] + [c for c in (popup_cols or []) if c in gdf_wgs.columns]
    folium.GeoJson(
        gdf_wgs[[*champs, "geometry"]].to_json(),
        style_function=fonction_style,
        tooltip=folium.GeoJsonTooltip(fields=champs),
        name=titre,
    ).add_to(m)

    legende = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="width:12px;height:12px;background:{coul};display:inline-block;margin-right:6px;border:1px solid #555;flex-shrink:0;"></span>'
        f"{cat} ({(gdf[categorie_col] == cat).sum():,})</div>"
        for cat, coul in couleurs.items()
        if (gdf[categorie_col] == cat).any()
    )
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:10px;right:10px;z-index:9999;background:white;'
        f'border:2px solid #888;border-radius:6px;padding:10px 12px;font-family:sans-serif;font-size:12px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.2);">'
        f'<div style="font-weight:bold;margin-bottom:6px;">{titre}</div>{legende}</div>'
    ))
    folium.LayerControl(collapsed=False).add_to(m)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(sortie))
    return sortie
