"""Deux variables géographiques tirées d'un extrait OpenStreetMap (OSM) : les cours d'eau/le littoral,
et les mairies.

* COURS D'EAU ET LITTORAL - les tronçons ("ways" OSM) tagués 'waterway=river', 'waterway=canal' ou
  'natural=coastline', à l'intérieur de la boîte englobante de la France. Serviront à calculer la
  distance de chaque commune à l'eau (fichier 'eau.py').
* MAIRIES - les points ("nœuds" OSM) tagués 'amenity=townhall', rattachés à leur commune (le
  polygone qui les contient, ou la commune la plus proche si aucun polygone ne les contient
  exactement). Utilisées comme le "vrai" centre d'une grande ville pour la distance urbaine
  (fichier 'urbain.py').

Les deux utilisent la même technique de lecture (voir 'outils.positions_des_noeuds') : plutôt que de
charger la position de TOUS les nœuds du fichier OSM (un demi-milliard, bien trop pour la mémoire),
on ne demande que les positions des nœuds qui nous intéressent déjà.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outils import apparier_noeuds, positions_des_noeuds  # noqa: E402

BOITE_COURS_EAU = (
    -5.5,
    41.0,
    10.0,
    52.0,
)  # longitude min, latitude min, longitude max, latitude max (métropole + Corse)
BOITE_MAIRIES = (41.3, -5.1, 51.1, 8.2)  # sud, ouest, nord, est
TAGS_COURS_EAU = (
    ("waterway", "river"),
    ("waterway", "canal"),
    ("natural", "coastline"),
)


def _type_de_troncon(tags) -> str:
    voie_eau = tags.get("waterway")
    return voie_eau if voie_eau in ("river", "canal") else "coastline"


def extraire_cours_deau(pbf: Path) -> gpd.GeoDataFrame:
    """Rivières, canaux et littoral, sous forme de lignes (WGS84) avec leur type, leur nom et leur id OSM."""
    import osmium

    troncons = osmium.FileProcessor(str(pbf), osmium.osm.WAY).with_filter(
        osmium.filter.TagFilter(*TAGS_COURS_EAU)
    )
    enregistrements, references = [], []
    for troncon in troncons:
        refs_noeuds = [n.ref for n in troncon.nodes]
        enregistrements.append(
            (
                _type_de_troncon(troncon.tags),
                troncon.tags.get("name", ""),
                troncon.id,
                len(refs_noeuds),
            )
        )
        references.extend(refs_noeuds)
    print(
        f"[géo OSM] {len(enregistrements):,} tronçons cours d'eau/littoral, {len(references):,} références de nœuds"
    )

    references = np.asarray(references, dtype=np.int64)
    trouves, lat, lon = positions_des_noeuds(pbf, references)
    position, ok = apparier_noeuds(trouves, references)
    troncon_de_ref = np.repeat(
        np.arange(len(enregistrements)), [e[3] for e in enregistrements]
    )
    gardes = troncon_de_ref[ok]
    comptes = np.bincount(gardes, minlength=len(enregistrements))
    lons, lats = lon[position[ok]], lat[position[ok]]
    lon_min, lat_min, lon_max, lat_max = BOITE_COURS_EAU
    dedans = (
        (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
    )
    touche_la_boite = np.bincount(gardes[dedans], minlength=len(enregistrements)) > 0
    utilisable = (
        comptes >= 2
    ) & touche_la_boite  # une ligne a besoin d'au moins 2 points
    coords = np.column_stack([lons, lats])
    points_selectionnes = utilisable[gardes]
    lignes = shapely.linestrings(
        coords[points_selectionnes],
        indices=np.repeat(np.arange(int(utilisable.sum())), comptes[utilisable]),
    )
    table = (
        pd.DataFrame(enregistrements, columns=["type", "name", "osm_id", "n"])
        .loc[utilisable, ["type", "name", "osm_id"]]
        .reset_index(drop=True)
    )
    print(
        f"[géo OSM] {len(table):,} lignes gardées ({table['type'].value_counts().to_dict()})"
    )
    return gpd.GeoDataFrame(table, geometry=lignes, crs="EPSG:4326")


def extraire_mairies(pbf: Path) -> pd.DataFrame:
    """Position et nom de chaque mairie (nœud 'amenity=townhall') à l'intérieur de la boîte de la France."""
    import osmium

    sud, ouest, nord, est = BOITE_MAIRIES
    noeuds = osmium.FileProcessor(str(pbf), osmium.osm.NODE).with_filter(
        osmium.filter.TagFilter(("amenity", "townhall"))
    )
    lignes = [
        (n.id, n.location.lat, n.location.lon, n.tags.get("name", ""))
        for n in noeuds
        if n.location.valid()
    ]
    table = pd.DataFrame(lignes, columns=["osm_id", "lat", "lon", "osm_name"])
    return table[
        (table["lat"].between(sud, nord)) & (table["lon"].between(ouest, est))
    ].reset_index(drop=True)


def rattacher_communes(
    mairies: pd.DataFrame, communes: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Commune de chaque mairie : le polygone qui la contient, sinon la commune dont le centroïde est
    le plus proche (utile pour les mairies mal placées dans OSM, tombées juste à côté du bon polygone)."""
    from scipy.spatial import cKDTree

    points = gpd.GeoDataFrame(
        mairies,
        geometry=gpd.points_from_xy(mairies["lon"], mairies["lat"]),
        crs="EPSG:4326",
    )
    reference = communes[["code_commune", "nom_commune", "geometry"]]
    jointure = gpd.sjoin(points, reference, how="left", predicate="within").drop(
        columns="index_right"
    )
    jointure = jointure[~jointure.index.duplicated()]
    manquantes = jointure["code_commune"].isna()
    if manquantes.any():
        centroides = reference.to_crs(2154).geometry.centroid
        arbre = cKDTree(
            np.column_stack([centroides.x.to_numpy(), centroides.y.to_numpy()])
        )
        non_rattachees = jointure[manquantes].to_crs(2154)
        _, plus_proche = arbre.query(
            np.column_stack(
                [
                    non_rattachees.geometry.x.to_numpy(),
                    non_rattachees.geometry.y.to_numpy(),
                ]
            ),
            k=1,
        )
        jointure.loc[manquantes, "code_commune"] = reference["code_commune"].to_numpy()[
            plus_proche
        ]
        jointure.loc[manquantes, "nom_commune"] = reference["nom_commune"].to_numpy()[
            plus_proche
        ]
        print(
            f"[géo OSM] {int(manquantes.sum())} mairie(s) hors de tout polygone communal : rattachée(s) à la commune la plus proche"
        )
    return pd.DataFrame(jointure.drop(columns="geometry"))


def preparer_geographie_osm(
    pbf: Path, dossier_communes: Path, dossier_sortie: Path
) -> tuple[Path, Path]:
    """Extrait cours d'eau (.gpkg) et mairies (parquet) d'un extrait OSM, et les écrit sur disque
    (saute le travail si le fichier existe déjà)."""
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemin_eau = dossier_sortie / "waterways_coastline.gpkg"
    if not chemin_eau.exists():
        extraire_cours_deau(pbf).to_file(chemin_eau, driver="GPKG")
        print(f"[géo OSM] cours d'eau -> {chemin_eau}")
    chemin_mairies = dossier_sortie / "town_halls.parquet"
    if not chemin_mairies.exists():
        communes = gpd.read_parquet(
            dossier_communes / "communes.parquet",
            columns=["code_commune", "nom_commune", "geometry"],
        )
        mairies = rattacher_communes(extraire_mairies(pbf), communes)
        mairies.to_parquet(chemin_mairies, index=False)
        print(f"[géo OSM] {len(mairies):,} mairies -> {chemin_mairies}")
    return chemin_eau, chemin_mairies
