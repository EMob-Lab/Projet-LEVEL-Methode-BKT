"""Distance de chaque commune au GRAND CENTRE URBAIN (GCU) le plus proche.

L'INSEE classe les communes selon une grille de densité à 7 niveaux ; le niveau le plus élevé est
"Grands centres urbains" (GCU). Les pratiques cyclables dépendent beaucoup de la distance à une
grande ville, donc on mesure cette distance jusqu'à la MAIRIE de chaque "groupe" de communes GCU :

    1. les communes GCU "de base" sont celles du niveau de densité le plus élevé ;
    2. un "centre urbain intermédiaire" (niveau juste en dessous) qui n'est NI collé à un GCU NI à
       moins de 15 km d'un GCU est "promu" GCU à part entière (une préfecture régionale isolée, loin
       de toute métropole, doit quand même compter comme un centre urbain) ;
    3. les communes GCU qui se touchent forment un GROUPE (les composantes connexes d'un graphe de
       contact - une grande ville et ses communes GCU voisines ne forment qu'un seul groupe) ;
    4. chaque groupe est représenté par la mairie de sa commune de base la plus dense (le centroïde
       si aucune mairie OSM n'est connue pour cette commune) ;
    5. la variable finale est la distance à vol d'oiseau entre le centroïde de CHAQUE commune et la
       mairie du groupe GCU le plus proche.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.ops import unary_union

RAYON_TERRE_KM = 6371.0
SEUIL_PROMOTION_KM = 15
GCU, INTERMEDIAIRE = "Grands centres urbains", "Centres urbains intermédiaires"


def _vecteurs_unitaires(lat, lon) -> np.ndarray:
    """Transforme des coordonnées latitude/longitude en vecteurs 3D (pour mesurer une distance à vol
    d'oiseau correcte sur une sphère, pas une distance euclidienne approximative)."""
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    return np.column_stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)]
    )


def _corde_vers_km(corde) -> np.ndarray:
    return RAYON_TERRE_KM * 2.0 * np.arcsin(np.clip(corde / 2.0, -1.0, 1.0))


def _centroides(communes: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    centroide = communes.geometry.centroid
    return centroide.y.to_numpy(), centroide.x.to_numpy()


def groupes_gcu(
    communes: gpd.GeoDataFrame, seuil_promotion_km: float = SEUIL_PROMOTION_KM
) -> gpd.GeoDataFrame:
    """Communes GCU (de base + centres intermédiaires promus), avec un 'group_id' par groupe connexe."""
    base = communes[communes["LIBDENS7"] == GCU].copy()
    intermediaires = communes[communes["LIBDENS7"] == INTERMEDIAIRE].copy()
    lat_base, lon_base = _centroides(base)
    lat_inter, lon_inter = _centroides(intermediaires)
    corde, _ = cKDTree(_vecteurs_unitaires(lat_base, lon_base)).query(
        _vecteurs_unitaires(lat_inter, lon_inter), k=1
    )
    intermediaires["dist_gcu_le_plus_proche_km"] = _corde_vers_km(corde)
    union_base = unary_union(base.geometry.values)
    intermediaires["touche_un_gcu"] = intermediaires.geometry.apply(
        lambda g: g.intersects(union_base)
    )
    promues = intermediaires[
        (~intermediaires["touche_un_gcu"])
        & (intermediaires["dist_gcu_le_plus_proche_km"] > seuil_promotion_km)
    ]
    gcu = gpd.GeoDataFrame(
        pd.concat([base, promues], ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    print(f"[urbain] GCU : {len(base)} de base + {len(promues)} promue(s)")

    projete = gcu.to_crs("EPSG:2154")
    graphe = nx.Graph()
    graphe.add_nodes_from(range(len(gcu)))
    for i in range(len(gcu)):
        for j in projete.sindex.query(projete.geometry.iloc[i], predicate="intersects"):
            if j > i and projete.geometry.iloc[i].intersects(projete.geometry.iloc[j]):
                graphe.add_edge(i, j)
    gcu["group_id"] = -1
    for groupe, membres in enumerate(nx.connected_components(graphe)):
        gcu.loc[list(membres), "group_id"] = groupe
    print(f"[urbain] {gcu['group_id'].nunique()} groupes GCU")
    return gcu


def mairie_de_chaque_groupe(
    gcu: gpd.GeoDataFrame, mairies: pd.DataFrame
) -> pd.DataFrame:
    """Un point représentatif par groupe GCU : la mairie de sa commune de base la plus dense (le
    centroïde de cette commune si sa mairie n'est pas connue)."""
    correspondance = (
        mairies.dropna(subset=["code_commune"])
        .drop_duplicates(subset=["code_commune"])
        .set_index("code_commune")[["lat", "lon"]]
    )
    lignes = []
    for groupe in sorted(gcu["group_id"].unique()):
        membres = gcu[gcu["group_id"] == groupe].copy()
        membres["_densite"] = (
            membres["population_commune"] / membres["aire_km2_commune"]
        )
        coeur = membres[membres["LIBDENS7"] == GCU]
        ensemble = coeur if len(coeur) else membres
        commune = ensemble.loc[ensemble["_densite"].idxmax()]
        code = commune["code_commune"]
        if code in correspondance.index:
            lat, lon = correspondance.loc[code, "lat"], correspondance.loc[code, "lon"]
        else:
            lat, lon = commune.geometry.centroid.y, commune.geometry.centroid.x
        lignes.append(
            {
                "group_id": groupe,
                "mairie_code": code,
                "mairie_lat": lat,
                "mairie_lon": lon,
            }
        )
    return pd.DataFrame(lignes)


def distance_au_gcu(
    communes: gpd.GeoDataFrame,
    mairies: pd.DataFrame,
    seuil_promotion_km: float = SEUIL_PROMOTION_KM,
) -> pd.Series:
    """Distance à vol d'oiseau (km) du centroïde de chaque commune à la mairie du groupe GCU le plus proche."""
    mairies_des_groupes = mairie_de_chaque_groupe(
        groupes_gcu(communes, seuil_promotion_km), mairies
    )
    arbre = cKDTree(
        _vecteurs_unitaires(
            mairies_des_groupes["mairie_lat"].to_numpy(),
            mairies_des_groupes["mairie_lon"].to_numpy(),
        )
    )
    lat, lon = _centroides(communes)
    corde, _ = arbre.query(_vecteurs_unitaires(lat, lon), k=1)
    return pd.Series(
        _corde_vers_km(corde),
        index=communes["code_commune"].to_numpy(),
        name=f"dist_gcu_group_km_T{seuil_promotion_km}",
    )


def preparer_distance_urbaine(
    communes_avec_densite: gpd.GeoDataFrame, dossier_geo: Path, dossier_sortie: Path
) -> Path:
    """'communes_avec_densite' doit avoir le code, la population, la surface, le niveau de densité
    ('LIBDENS7') et la géométrie de chaque commune - voir 'table_communes.py' pour comment cette table
    est construite (jointure zones + socio-éco)."""
    mairies = pd.read_parquet(dossier_geo / "town_halls.parquet")
    distance = distance_au_gcu(communes_avec_densite, mairies)
    table = distance.rename_axis("code_commune").reset_index()
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemin = dossier_sortie / "urban_distance.parquet"
    table.to_parquet(chemin, index=False)
    print(f"[urbain] distance au GCU pour {len(table):,} communes -> {chemin}")
    return chemin
