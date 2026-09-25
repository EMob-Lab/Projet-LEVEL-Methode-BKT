"""Extrait une PETITE ZONE (un rectangle lat/lon) d'un extrait OpenStreetMap France (.osm.pbf) - de
quoi tester 'filtre_regles.py'/'propagation_spatiale.py' sur un vrai quartier sans traiter la France
entière (voir '02_FOB/' pour l'extraction complète, optimisée pour ça).

Deux passes, comme le fait '02_FOB/' à plus grande échelle (voir sa docstring) : d'abord tous les
NŒUDS dans la zone (leurs coordonnées), puis tous les TRONÇONS 'highway=*' dont au moins un nœud est
dans la zone (leurs tags + la géométrie, reconstruite à partir des nœuds de la passe 1 - un tronçon à
cheval sur le bord de la zone aura une géométrie partielle, c'est attendu pour une simple démonstration)."""

from __future__ import annotations

from array import array
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

# les mêmes clés que le filtre à règles (filtre_regles.py) a besoin, plus le nom (pour propagation_spatiale.py)
TAGS_GARDES = (
    "highway",
    "bicycle",
    "cycleway",
    "access",
    "maxspeed",
    "surface",
    "name",
)


def _noeuds_dans_la_zone(
    pbf: str | Path, bbox: tuple[float, float, float, float]
) -> dict[int, tuple[float, float]]:
    """'bbox' = (lon_min, lat_min, lon_max, lat_max). Une passe sur TOUS les nœuds du fichier - coûteux
    sur un extrait France entière (voir le docstring du module), mais la seule façon d'extraire une
    zone sans index spatial préexistant."""
    import osmium

    lon_min, lat_min, lon_max, lat_max = bbox
    noeuds: dict[int, tuple[float, float]] = {}
    for noeud in osmium.FileProcessor(str(pbf), osmium.osm.NODE):
        position = noeud.location
        if (
            position.valid()
            and lon_min <= position.lon <= lon_max
            and lat_min <= position.lat <= lat_max
        ):
            noeuds[noeud.id] = (position.lat, position.lon)
    return noeuds


def extraire_zone(
    pbf: str | Path, bbox: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Tronçons 'highway=*' de la zone : une ligne par tronçon, colonnes 'id_osm' + TAGS_GARDES +
    'geometry' (LineString WGS84, éventuellement partielle - voir le docstring du module)."""
    import osmium

    print(f"[extraction_zone] passe 1/2 : nœuds dans {bbox}...")
    noeuds = _noeuds_dans_la_zone(pbf, bbox)
    print(f"[extraction_zone] {len(noeuds):,} nœuds trouvés dans la zone")

    print("[extraction_zone] passe 2/2 : tronçons touchant la zone...")
    lignes = []
    for voie in osmium.FileProcessor(str(pbf), osmium.osm.WAY):
        if "highway" not in voie.tags:
            continue
        points = [noeuds[n.ref] for n in voie.nodes if n.ref in noeuds]
        if len(points) < 2:
            continue
        ligne = {
            "id_osm": voie.id,
            "geometry": LineString([(lon, lat) for lat, lon in points]),
        }
        for cle in TAGS_GARDES:
            ligne[cle] = voie.tags.get(cle)
        lignes.append(ligne)
    print(f"[extraction_zone] {len(lignes):,} tronçons extraits")

    table = gpd.GeoDataFrame(pd.DataFrame(lignes), geometry="geometry", crs="EPSG:4326")
    return table


def bbox_autour_de(
    centre_lat: float, centre_lon: float, rayon_km: float
) -> tuple[float, float, float, float]:
    """Rectangle lat/lon approximatif de 'rayon_km' autour d'un point (approximation suffisante pour une
    petite zone de test - pas de projection précise nécessaire)."""
    delta_lat = rayon_km / 111.0  # ~111 km par degré de latitude, partout
    delta_lon = rayon_km / (
        111.0 * np.cos(np.radians(centre_lat))
    )  # un degré de longitude est plus court loin de l'équateur
    return (
        centre_lon - delta_lon,
        centre_lat - delta_lat,
        centre_lon + delta_lon,
        centre_lat + delta_lat,
    )
