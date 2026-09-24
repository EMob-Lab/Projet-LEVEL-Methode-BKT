"""Zones administratives françaises : communes, départements, régions - avec leur population INSEE et
leur surface calculée. Source : fichiers geojson de l'IGN (déjà simplifiés) + tables de population
légale INSEE.

Pour chaque niveau (commune / département / région), le traitement est le même :
    1. on garde le code, le nom, le code du niveau parent et le contour (géométrie) ;
    2. on rattache la population INSEE ("PTOT" dans les fichiers sources) ;
    3. on calcule la surface en km² (projection Lambert-93, la projection officielle française) ;
    4. on retire les polygones qui EN CONTIENNENT un autre : Paris, Lyon et Marseille existent à la
       fois comme une commune entière et comme leurs arrondissements - on garde les arrondissements,
       pas la ville entière qui les contient tous ;
    5. on ne garde que la France métropolitaine, Corse exclue (ses codes ne sont pas numériques -
       "2A"/"2B" - donc simples à filtrer) ;
    6. on vérifie que le résultat est cohérent (pas de surface négative, pas de code en double...).

La table des communes (34 428 lignes attendues) définit le périmètre géographique de tout le projet :
tout le reste (capteurs, réseau cyclable, BKT) est rattaché à une de ces communes.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

LAMBERT93 = "EPSG:2154"

# Départements de métropole hors Corse : "01" à "95", en texte avec un zéro devant si besoin.
DEPARTEMENTS_METROPOLE = [str(i).zfill(2) for i in range(1, 96)]
# Régions de métropole (la Corse, code région "94", est exclue).
REGIONS_METROPOLE = [
    "11",
    "24",
    "27",
    "28",
    "32",
    "44",
    "52",
    "53",
    "75",
    "76",
    "84",
    "93",
]


def _surface_km2(zones: gpd.GeoDataFrame) -> pd.Series:
    """Surface de chaque polygone en km², calculée en projection Lambert-93 (la seule où un calcul de
    surface en mètres a un sens pour la France - la géométrie brute est en latitude/longitude)."""
    return zones.to_crs(LAMBERT93).geometry.area / 1e6


def retirer_les_conteneurs(
    zones: gpd.GeoDataFrame, nom_niveau: str
) -> gpd.GeoDataFrame:
    """Retire les polygones qui en contiennent un autre de la même table (garde le plus PETIT des deux,
    donc dans le cas Paris/Lyon/Marseille : garde les arrondissements, retire la ville entière)."""
    projete = zones.to_crs(LAMBERT93)
    surface = projete.geometry.area
    paires = gpd.sjoin(
        projete[["geometry"]], projete[["geometry"]], how="inner", predicate="contains"
    ).reset_index(names="i")
    paires = paires.rename(columns={"index_right": "j"})
    paires = paires[
        paires["i"] != paires["j"]
    ]  # un polygone se "contient" toujours lui-même, on l'ignore
    conteneurs = {
        int(i) if surface.iloc[int(i)] >= surface.iloc[int(j)] else int(j)
        for i, j in zip(paires["i"], paires["j"])
    }
    if conteneurs:
        print(
            f"  [{nom_niveau}] {len(conteneurs)} polygone(s) qui en contenaient un autre, retiré(s)"
        )
    return zones.drop(index=sorted(conteneurs)).copy()


def _lire_population(chemin: Path, colonne_code: str) -> pd.DataFrame:
    """Lit un fichier de population légale INSEE (CSV point-virgule) et ne garde que le code et la population totale."""
    table = pd.read_csv(chemin, sep=";", dtype={colonne_code: str})
    return table[[colonne_code, "PTOT"]]


def _verifier(
    zones: gpd.GeoDataFrame, colonne_surface: str, colonne_population: str | None
) -> None:
    """Petites vérifications de bon sens avant d'écrire le résultat sur disque."""
    assert (zones[colonne_surface] > 0).all(), (
        f"{colonne_surface} doit toujours être positive"
    )
    if colonne_population:
        assert (zones[colonne_population].dropna() >= 0).all(), (
            f"{colonne_population} ne peut pas être négative"
        )
    assert zones.iloc[:, 0].is_unique, "il y a des codes en double"


def construire_communes(
    communes_geojson: Path, population_csv: Path
) -> gpd.GeoDataFrame:
    """Construit la table des communes : code, nom, département, population, surface, contour."""
    communes = gpd.read_file(communes_geojson)[
        ["code", "nom", "departement", "geometry"]
    ]
    communes = communes.rename(
        columns={
            "code": "code_commune",
            "nom": "nom_commune",
            "departement": "code_departement",
        }
    )
    population = _lire_population(population_csv, "COM").rename(
        columns={"COM": "code_commune", "PTOT": "population_commune"}
    )
    communes = communes.merge(population, on="code_commune", how="left")
    communes["aire_km2_commune"] = _surface_km2(communes)
    communes = retirer_les_conteneurs(communes, "communes")
    communes = communes[
        communes["code_departement"].isin(DEPARTEMENTS_METROPOLE)
    ].copy()
    communes = communes[
        [
            "code_commune",
            "nom_commune",
            "code_departement",
            "population_commune",
            "aire_km2_commune",
            "geometry",
        ]
    ]
    _verifier(communes, "aire_km2_commune", "population_commune")
    return communes


def construire_departements(
    departements_geojson: Path, population_csv: Path
) -> gpd.GeoDataFrame:
    """Construit la table des départements : code, région, nom, population, surface, contour."""
    zones = gpd.read_file(departements_geojson)[["code", "nom", "region", "geometry"]]
    zones = zones.rename(
        columns={
            "code": "code_departement",
            "nom": "nom_departement",
            "region": "code_region",
        }
    )
    population = _lire_population(population_csv, "DEP").rename(
        columns={"DEP": "code_departement", "PTOT": "population_departement"}
    )
    zones = zones.merge(population, on="code_departement", how="left")
    zones["aire_km2_departement"] = _surface_km2(zones)
    zones = retirer_les_conteneurs(zones, "départements")
    zones = zones[zones["code_departement"].isin(DEPARTEMENTS_METROPOLE)].copy()
    zones = zones[
        [
            "code_departement",
            "code_region",
            "nom_departement",
            "population_departement",
            "aire_km2_departement",
            "geometry",
        ]
    ]
    _verifier(zones, "aire_km2_departement", "population_departement")
    return zones


def construire_regions(regions_geojson: Path, population_csv: Path) -> gpd.GeoDataFrame:
    """Construit la table des régions : code, nom, population, surface, contour."""
    zones = gpd.read_file(regions_geojson)[["code", "nom", "geometry"]].rename(
        columns={"code": "code_region", "nom": "nom_region"}
    )
    population = _lire_population(population_csv, "REG").rename(
        columns={"REG": "code_region", "PTOT": "population_region"}
    )
    population["code_region"] = population["code_region"].astype(str).str.zfill(2)
    zones = zones.merge(population, on="code_region", how="left")
    zones["aire_km2_region"] = _surface_km2(zones)
    zones = retirer_les_conteneurs(zones, "régions")
    zones = zones[zones["code_region"].isin(REGIONS_METROPOLE)].copy()
    zones = zones[
        [
            "code_region",
            "nom_region",
            "population_region",
            "aire_km2_region",
            "geometry",
        ]
    ]
    _verifier(zones, "aire_km2_region", "population_region")
    return zones


def preparer_zones(dossier_brutes: Path, dossier_sortie: Path) -> dict[str, Path]:
    """Construit les trois tables (communes, départements, régions) et les écrit en parquet.

    'dossier_brutes' doit contenir 'zones/communes-5m.geojson', 'zones/departements-5m.geojson',
    'zones/regions-5m.geojson' et 'zones/population_{communes,departements,regions}.csv'."""
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    zones_brutes = dossier_brutes / "zones"
    resultats = {
        "communes": (
            construire_communes(
                zones_brutes / "communes-5m.geojson",
                zones_brutes / "population_communes.csv",
            ),
            dossier_sortie / "communes.parquet",
        ),
        "départements": (
            construire_departements(
                zones_brutes / "departements-5m.geojson",
                zones_brutes / "population_departements.csv",
            ),
            dossier_sortie / "departements.parquet",
        ),
        "régions": (
            construire_regions(
                zones_brutes / "regions-5m.geojson",
                zones_brutes / "population_regions.csv",
            ),
            dossier_sortie / "regions.parquet",
        ),
    }
    chemins: dict[str, Path] = {}
    for nom, (table, chemin) in resultats.items():
        table.to_parquet(chemin, index=False)
        print(f"[zones] {len(table):,} {nom} -> {chemin}")
        chemins[nom] = chemin
    return chemins
