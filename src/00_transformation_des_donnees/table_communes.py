"""Assemble TOUTES les variables communales préparées par les autres fichiers de ce dossier en une
seule grande table, une ligne par commune - c'est ce que les étapes suivantes du projet (classification
des communes, calcul du BKT) utiliseront comme entrée.

Sont jointes ici, toutes indexées par 'code_commune' :

    zones          nom, département, population, surface                     (d_zones/zones.py)
    densité        niveaux de densité INSEE DENS/LIBDENS/DENS7/LIBDENS7       (d_socio_eco/socio_economique.py)
    relief         altitude du centroïde + statistiques zonales du MNT        (d_geographie/altitude.py)
    eau            distance à la côte / à une rivière / à un canal            (d_geographie/eau.py)
    urbain         distance à la mairie du grand centre urbain le plus proche (d_geographie/urbain.py)
    socio-éco.     revenu médian, aménagements cyclables, accueil vélo        (d_socio_eco/socio_economique.py)

... plus quelques variables DÉRIVÉES que les modèles suivants utilisent directement (densité de
population, versions "log" de certaines distances, indicateur "commune côtière", indicateur "commune
montagneuse").
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "d_geographie"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "d_socio_eco"))
from altitude import preparer_altitude  # noqa: E402
from eau import preparer_distances_eau  # noqa: E402
from geographie_osm import preparer_geographie_osm  # noqa: E402
from socio_economique import preparer_socio_economique  # noqa: E402
from urbain import preparer_distance_urbaine  # noqa: E402

SEUIL_COTIER_KM = 20  # une commune est "côtière" en deçà de cette distance à la mer
SEUIL_MONTAGNEUX_M = (
    200  # une commune est "montagneuse" si son étendue d'altitude dépasse cette valeur
)


def ajouter_variables_derivees(table: pd.DataFrame) -> pd.DataFrame:
    """Densité de population, distances en échelle logarithmique, indicateurs côtier et montagneux
    (une entrée manquante donne simplement une sortie manquante, pas d'erreur)."""
    sortie = table.copy()
    sortie["pop_density"] = sortie["population_commune"] / sortie["aire_km2_commune"]
    sortie["log_pop"] = np.log1p(sortie["population_commune"])
    sortie["log_pop_density"] = np.log1p(sortie["pop_density"])
    for type_eau in ("coast", "river", "canal"):
        sortie[f"log_dist_{type_eau}"] = np.log1p(sortie[f"dist_{type_eau}_km"])
    sortie["near_coast_20km"] = (sortie["dist_coast_km"] < SEUIL_COTIER_KM).astype(int)
    sortie["is_mountainous"] = (sortie["alt_range_m"] > SEUIL_MONTAGNEUX_M).astype(int)
    return sortie


def preparer_geographie(
    pbf: Path,
    dossier_zones: Path,
    dossier_geo: Path,
    mosaique_srtm_existante: Path | None = None,
) -> None:
    """Toutes les couches géographiques qui demandent un calcul un peu lourd (cours d'eau, mairies,
    altitude) - chacune est sautée si son fichier de sortie existe déjà."""
    preparer_geographie_osm(pbf, dossier_zones, dossier_geo)
    if not (dossier_geo / "water_distance.parquet").exists():
        preparer_distances_eau(dossier_zones, dossier_geo, dossier_geo)
    if not (dossier_geo / "altitude.parquet").exists():
        preparer_altitude(dossier_zones, dossier_geo, mosaique_srtm_existante)


def construire_table_communes(
    dossier_brutes: Path,
    dossier_zones: Path,
    dossier_geo: Path,
    dossier_socio_eco: Path,
) -> pd.DataFrame:
    """Assemble la table finale à partir des fichiers déjà préparés par les autres fonctions de ce dossier."""
    zones = gpd.read_parquet(dossier_zones / "communes.parquet")

    socio = preparer_socio_economique(dossier_brutes, dossier_zones, dossier_socio_eco)
    social = pd.read_parquet(socio)
    avec_densite = zones.merge(
        social[["code_commune", "LIBDENS7"]], on="code_commune", how="left"
    )

    distance_urbaine = pd.read_parquet(
        preparer_distance_urbaine(avec_densite, dossier_geo, dossier_geo)
    )

    table = (
        pd.DataFrame(zones.drop(columns="geometry"))
        .merge(social, on="code_commune", how="left")
        .merge(distance_urbaine, on="code_commune", how="left")
    )
    table = table.merge(
        pd.read_parquet(dossier_geo / "water_distance.parquet"),
        on="code_commune",
        how="left",
    )
    relief = pd.read_parquet(dossier_geo / "altitude.parquet").rename(
        columns={"alt_centroid_m": "alt_commune_m"}
    )
    table = table.merge(relief, on="code_commune", how="left")
    return ajouter_variables_derivees(table)


def preparer_table_communes(
    dossier_brutes: Path,
    dossier_zones: Path,
    dossier_geo: Path,
    dossier_socio_eco: Path,
    dossier_sortie: Path,
) -> Path:
    """Construit la table finale et l'écrit en parquet."""
    table = construire_table_communes(
        dossier_brutes, dossier_zones, dossier_geo, dossier_socio_eco
    )
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemin = dossier_sortie / "commune_features.parquet"
    table.to_parquet(chemin, index=False)
    print(
        f"[table communes] {len(table):,} communes x {table.shape[1]} colonnes -> {chemin}"
    )
    return chemin
