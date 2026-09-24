"""Distance de chaque commune à la mer, au cours d'eau le plus proche et au canal le plus proche.

La distance est mesurée du CENTROÏDE de la commune à la ligne OSM la plus proche de chaque type
(cours d'eau/canal/littoral, préparés par 'geographie_osm.py'), en projection Lambert-93 (des mètres,
converties en km). Les cours d'eau et canaux comptent pour le vélo de loisir (chemins de halage,
coulées vertes) ; le littoral pour le tourisme balnéaire.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

COLONNE_PAR_TYPE = {
    "coastline": "dist_coast_km",
    "river": "dist_river_km",
    "canal": "dist_canal_km",
}


def distances_a_l_eau(
    communes: gpd.GeoDataFrame, cours_deau: gpd.GeoDataFrame
) -> pd.DataFrame:
    """'code_commune, dist_coast_km, dist_river_km, dist_canal_km'."""
    centroides = gpd.GeoSeries(communes.geometry.centroid, crs="EPSG:4326").to_crs(2154)
    lignes = cours_deau.to_crs(2154)
    sortie = pd.DataFrame({"code_commune": communes["code_commune"].to_numpy()})
    for type_eau, colonne in COLONNE_PAR_TYPE.items():
        arbre = shapely.STRtree(lignes.geometry[lignes["type"] == type_eau].to_numpy())
        _, distance = arbre.query_nearest(
            centroides.to_numpy(), return_distance=True, all_matches=False
        )
        sortie[colonne] = (distance / 1000.0).round(3)
    return sortie


def preparer_distances_eau(
    dossier_communes: Path, dossier_geo: Path, dossier_sortie: Path
) -> Path:
    """Calcule les distances à l'eau et les écrit en parquet.

    'dossier_communes' doit contenir 'communes.parquet', 'dossier_geo' doit contenir
    'waterways_coastline.gpkg' (les deux préparés en amont)."""
    communes = gpd.read_parquet(
        dossier_communes / "communes.parquet", columns=["code_commune", "geometry"]
    )
    table = distances_a_l_eau(
        communes, gpd.read_file(dossier_geo / "waterways_coastline.gpkg")
    )
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemin = dossier_sortie / "water_distance.parquet"
    table.to_parquet(chemin, index=False)
    print(f"[eau] distances calculées pour {len(table):,} communes -> {chemin}")
    return chemin
