"""La méthode GÉOVÉLO SEUL : la longueur du réseau cyclable d'une commune, prise TELLE QUELLE dans les
exports GéoVélo annuels qui sont des données issues d'OpenStreetMap tiré de leur méthode propriétaire (contrairement à FOB, voir
'02_FOB/').

GéoVélo découpe déjà ses segments aux frontières communales et enregistre, pour chaque segment, la
commune de son côté droit ('code_com_d') et de son côté gauche ('code_com_g'), plus le sens de
circulation de chaque côté. La longueur EFFECTIVE d'une commune est la somme (en Lambert-93, la
projection officielle française) de ses segments, un côté BIDIRECTIONNEL comptant deux fois (convention
du projet, variable 'alpha' - un aménagement où l'on peut rouler dans les deux sens "vaut" deux fois plus
de linéaire utile qu'un aménagement à sens unique).

Limite importante : GéoVélo ne référence QUE le réseau que ses contributeurs ont utilisé - un
aménagement réel mais absent de GéoVélo est absent d'ici (c'est justement ce que FOB, à partir
d'OpenStreetMap, cherche à corriger - voir la comparaison des trois méthodes, README du dossier
parent).

Aucun export GéoVélo brut n'est fourni avec ce dépôt
Les jeux de données GéoVélo tiré d'OSM sont disponible sur data.gouv.fr mais pas ici : ce module
est donc écrit prêt à l'emploi, mais son résultat réel pour ce dépôt est celui déjà vendorisé dans
'data/donnees_brutes/reference/longueurs_reseau_reference.csv' (voir 'reference.py', même dossier)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkb as shapely_wkb

ORTHOGRAPHES_BIDIRECTIONNEL = {
    "bidirectionnel",
    "bidirectional",
    "bi-directionnel",
    "2 sens",
    "double sens",
}
COLONNES = ["code_com_d", "code_com_g", "geometry", "sens_d", "sens_g"]
LAMBERT93 = 2154


def lire_export(chemin: Path) -> gpd.GeoDataFrame:
    """Un export GéoVélo annuel (parquet avec géométrie WKB, ou geojson) en tronçons WGS84."""
    if chemin.suffix == ".parquet":
        brut = pd.read_parquet(chemin, columns=COLONNES)
        brut["geometry"] = brut["geometry"].apply(
            lambda b: shapely_wkb.loads(b) if b is not None else None
        )
        return gpd.GeoDataFrame(
            brut.dropna(subset=["geometry"]), geometry="geometry", crs="EPSG:4326"
        )
    complet = gpd.read_file(chemin)
    table = complet[[c for c in COLONNES if c in complet.columns]].dropna(
        subset=["geometry"]
    )
    table = table[table.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    table = gpd.GeoDataFrame(table, geometry="geometry", crs=complet.crs or "EPSG:4326")
    return (
        table.set_crs("EPSG:4326", allow_override=True)
        if complet.crs is None
        else table.to_crs("EPSG:4326")
    )


def _est_bidirectionnel(sens: pd.Series) -> np.ndarray:
    return (
        sens.astype(str)
        .str.strip()
        .str.lower()
        .isin(ORTHOGRAPHES_BIDIRECTIONNEL)
        .to_numpy()
    )


def longueurs_par_commune(
    troncons: gpd.GeoDataFrame, alpha: float = 2.0
) -> pd.DataFrame:
    """Longueur effective par commune : 'code_commune, len_d, len_g' (km).

    'alpha' : multiplicateur d'un côté bidirectionnel (convention du projet : 2 - voir le docstring du
    module)."""
    km = (troncons.to_crs(LAMBERT93).geometry.length / 1000.0).round(4)
    len_d = (
        (km * np.where(_est_bidirectionnel(troncons["sens_d"]), alpha, 1.0))
        .groupby(troncons["code_com_d"].to_numpy())
        .sum()
        .rename("len_d")
    )
    len_g = (
        (km * np.where(_est_bidirectionnel(troncons["sens_g"]), alpha, 1.0))
        .groupby(troncons["code_com_g"].to_numpy())
        .sum()
        .rename("len_g")
    )
    sortie = pd.DataFrame({"len_d": len_d, "len_g": len_g}).fillna(0.0)
    sortie.index = sortie.index.astype(str).str.strip()
    sortie.index.name = "code_commune"
    return sortie.reset_index()


def preparer_longueurs_geovelo(
    dossier_exports: Path, annees: tuple[int, ...], alpha: float = 2.0
) -> pd.DataFrame:
    """Table complète ('annee, code_commune, len_d, len_g') à partir des exports GéoVélo annuels de
    'dossier_exports' (un fichier 'geovelo_{annee}.parquet' ou '.geojson' par année - voir 'lire_export')."""
    morceaux = []
    for annee in annees:
        chemin = next(
            (
                p
                for p in dossier_exports.glob(f"geovelo_{annee}.*")
                if p.suffix in (".parquet", ".geojson")
            ),
            None,
        )
        if chemin is None:
            raise FileNotFoundError(
                f"aucun export GéoVélo pour {annee} dans {dossier_exports} (attendu : geovelo_{annee}.parquet ou .geojson)"
            )
        longueurs = longueurs_par_commune(lire_export(chemin), alpha)
        print(
            f"[geovelo] {annee} ({chemin.name}) : {longueurs['len_d'].sum() + longueurs['len_g'].sum():.0f} km effectifs"
        )
        morceaux.append(longueurs.assign(annee=annee))
    return pd.concat(morceaux, ignore_index=True)[
        ["annee", "code_commune", "len_d", "len_g"]
    ]
