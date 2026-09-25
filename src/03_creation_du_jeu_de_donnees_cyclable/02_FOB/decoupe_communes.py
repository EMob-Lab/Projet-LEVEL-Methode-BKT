# -*- coding: utf-8 -*-
"""Découpe les tronçons FOB par les polygones communaux, pour qu'un tronçon traversant plusieurs
communes soit réparti entre elles plutôt qu'attribué en entier à une seule.

Pourquoi : GéoVélo découpe déjà ses segments aux frontières communales (voir '01_geovelo_seul/'), mais
un tronçon OSM peut traverser plusieurs communes. Lui attribuer sa longueur entière à une seule commune
(celle de son point médian) répartirait mal le linéaire ; le découper donne à chaque commune exactement
sa part.

Pour chaque intersection (tronçon, commune), on garde un MORCEAU :

    length_m = part x longueur OSM du tronçon cette année-là   (part = fraction de la géométrie à l'intérieur de la commune)

pour que la longueur annuelle reste juste même quand la géométrie vient d'un snapshot différent de
l'année (tronçon découpé ou modifié plus tard sur OSM) - seule la RÉPARTITION entre communes vient de
la géométrie, la longueur totale vient toujours des tags de l'année.

Les polygones communaux ne couvrent que la France métropolitaine hors Corse (voir
'00_transformation_des_donnees/d_zones/zones.py') : tout ce qui est en dehors (Corse, outre-mer,
étranger, mer) ne reçoit aucun morceau."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

LAMBERT93 = 2154
MORCEAU_MIN_M = 0.01  # retire les intersections numériquement vides
SOURCE_DECOUPE, SOURCE_POINT_MEDIAN = "decoupe", "point_median"


# ═══════════════════════════════════════════════════════════════════════════════════════════════ polygones
@dataclass
class PolygonesCommunes:
    codes: np.ndarray  # codes INSEE (str)
    polygones: np.ndarray  # polygones shapely en Lambert-93, préparés pour des prédicats rapides
    wgs84: gpd.GeoDataFrame  # les mêmes polygones en WGS84 (pour les recherches par point médian)

    @classmethod
    def depuis_wgs84(cls, communes: gpd.GeoDataFrame) -> "PolygonesCommunes":
        """Construit à partir d'une table WGS84 avec 'code_commune' et 'geometry' (voir
        '00_transformation_des_donnees/d_zones/zones.py')."""
        wgs84 = communes[["code_commune", "geometry"]].copy()
        wgs84["code_commune"] = wgs84["code_commune"].astype(str).str.zfill(5)
        polygones = wgs84.to_crs(LAMBERT93).geometry.to_numpy()
        shapely.prepare(polygones)
        return cls(wgs84["code_commune"].to_numpy(), polygones, wgs84)

    @classmethod
    def charger(cls, chemin: str | Path) -> "PolygonesCommunes":
        return cls.depuis_wgs84(gpd.read_parquet(chemin, columns=["code_commune", "geometry"]))


# ═══════════════════════════════════════════════════════════════════════════════════════════════ résultat
@dataclass
class ResultatDecoupe:
    morceaux: pd.DataFrame  # id_osm, code_commune, length_m, part, source
    totaux_par_troncon: pd.DataFrame  # une ligne par tronçon FOB (voir 'decouper_troncons')

    def sauvegarder(self, dossier_sortie: Path) -> None:
        dossier_sortie.mkdir(parents=True, exist_ok=True)
        self.morceaux.to_parquet(dossier_sortie / "pieces.parquet", compression="zstd")
        self.totaux_par_troncon.to_parquet(dossier_sortie / "way_totals.parquet", compression="zstd")

    @classmethod
    def charger(cls, dossier_sortie: Path) -> "ResultatDecoupe":
        return cls(pd.read_parquet(dossier_sortie / "pieces.parquet"), pd.read_parquet(dossier_sortie / "way_totals.parquet"))

    def statistiques(self, annee: int, n_troncons_fob: int) -> dict:
        """Résumé utilisé par la feuille de contrôle (voir le notebook '02_FOB.ipynb')."""
        t = self.totaux_par_troncon
        return dict(
            annee=annee,
            n_troncons=n_troncons_fob,
            n_troncons_geometrie=int((t["source"] == SOURCE_DECOUPE).sum()),
            n_troncons_repli_point_median=int((t["source"] != SOURCE_DECOUPE).sum()),
            n_troncons_hors_perimetre=int((t["length_in_scope_m"] == 0).sum()),
            n_troncons_multi_communes=int((t["n_communes"] >= 2).sum()),
            km_osm=float(t["length_osm_m"].sum() / 1000),
            km_perimetre=float(t["length_in_scope_m"].sum() / 1000),
            km_hors_perimetre=float(t["outside_m"].sum() / 1000),
        )


# ═══════════════════════════════════════════════════════════════════════════════════════════════ découpe géométrique
def _decouper_geometries(ids_troncons: np.ndarray, wkb: np.ndarray, communes: PolygonesCommunes) -> tuple[pd.DataFrame, np.ndarray]:
    """Intersecte chaque ligne WGS84 avec les communes qu'elle touche.

    Renvoie ('morceaux', 'longueur_geom_m') où 'morceaux' a 'id_osm, code_commune, length_geom_m, part'
    et 'longueur_geom_m' est la longueur Lambert-93 de chaque ligne d'entrée (alignée avec 'ids_troncons')."""
    lignes_wgs = shapely.from_wkb(wkb)
    xy = shapely.get_coordinates(lignes_wgs)
    x, y = Transformer.from_crs(4326, LAMBERT93, always_xy=True).transform(xy[:, 0], xy[:, 1])
    n_points = shapely.get_num_coordinates(lignes_wgs)
    lignes = shapely.linestrings(np.column_stack([x, y]), indices=np.repeat(np.arange(len(lignes_wgs)), n_points))
    longueur_troncon = shapely.length(lignes)

    idx_ligne, idx_poly = shapely.STRtree(communes.polygones).query(lignes, predicate="intersects")
    dedans = shapely.covers(communes.polygones[idx_poly], lignes[idx_ligne])
    morceau = np.empty(len(idx_ligne))
    morceau[dedans] = longueur_troncon[idx_ligne[dedans]]  # entièrement dans une commune : rien à découper
    traverse = ~dedans
    morceau[traverse] = shapely.length(shapely.intersection(lignes[idx_ligne[traverse]], communes.polygones[idx_poly[traverse]]))
    print(
        f"[decoupe] {len(idx_ligne):,} paires (tronçon, commune) pour {len(lignes):,} tronçons ; "
        f"{100 * dedans.mean() if len(dedans) else 0.0:.1f}% entièrement dans une commune, {traverse.sum():,} à découper"
    )

    morceaux = pd.DataFrame({"id_osm": ids_troncons[idx_ligne], "code_commune": communes.codes[idx_poly], "length_geom_m": morceau, "way_geom_m": longueur_troncon[idx_ligne]})
    morceaux = morceaux[morceaux["length_geom_m"] > MORCEAU_MIN_M].copy()
    morceaux["part"] = morceaux["length_geom_m"] / morceaux["way_geom_m"].clip(lower=1e-9)
    return morceaux.drop(columns="way_geom_m"), longueur_troncon


def _morceaux_point_median(troncons: pd.DataFrame, communes: PolygonesCommunes) -> pd.DataFrame:
    """Repli : un morceau par tronçon, dans la commune contenant son point médian (aucun morceau si
    hors de tout polygone)."""
    points = gpd.GeoDataFrame({"i": np.arange(len(troncons))}, geometry=gpd.points_from_xy(troncons["lon"], troncons["lat"]), crs="EPSG:4326")
    jointes = gpd.sjoin(points, communes.wgs84, how="left", predicate="within")
    jointes = jointes[~jointes.index.duplicated()].sort_values("i")
    code = jointes["code_commune"].to_numpy()
    dedans = pd.notna(code)
    return pd.DataFrame(
        {
            "id_osm": troncons["id_osm"].to_numpy()[dedans],
            "code_commune": code[dedans],
            "length_m": troncons["length_m"].fillna(0.0).to_numpy()[dedans],
            "part": 1.0,
            "source": SOURCE_POINT_MEDIAN,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════════════════════ API publique
def decouper_troncons(troncons: pd.DataFrame, geometries: pd.DataFrame, communes: PolygonesCommunes) -> ResultatDecoupe:
    """Découpe les tronçons FOB d'un snapshot par les polygones communaux.

    'troncons' : tronçons FOB du snapshot avec 'id_osm, lat, lon, length_m' (longueur OSM de l'année).
    'geometries' : 'id_osm' et 'wkb' pour les tronçons dont la géométrie est connue (voir
    'osm_extraction.TronconsLocalises.vers_wkb').
    'communes' : polygones définissant le périmètre de découpe."""
    longueur_annee = troncons.set_index("id_osm")["length_m"].fillna(0.0)
    geometries = geometries[geometries["id_osm"].isin(longueur_annee.index)]  # ignore les tronçons sortis du réseau FOB
    morceaux, longueur_geom = _decouper_geometries(geometries["id_osm"].to_numpy(), geometries["wkb"].to_numpy(), communes)
    morceaux["length_m"] = morceaux["part"].to_numpy() * morceaux["id_osm"].map(longueur_annee).to_numpy()
    morceaux["source"] = SOURCE_DECOUPE
    morceaux = morceaux.drop(columns="length_geom_m")

    ids_decoupes = geometries["id_osm"].to_numpy()
    repli = troncons[~troncons["id_osm"].isin(ids_decoupes) & troncons["lat"].notna()]
    if len(repli):
        morceaux = pd.concat([morceaux, _morceaux_point_median(repli, communes)], ignore_index=True)

    couverts = np.concatenate([ids_decoupes, repli["id_osm"].to_numpy()])
    totaux = pd.DataFrame(
        {
            "id_osm": couverts,
            "length_osm_m": longueur_annee.reindex(couverts).to_numpy(),
            "length_geom_m": np.concatenate([longueur_geom, np.full(len(repli), np.nan)]),
            "source": np.concatenate([np.full(len(ids_decoupes), SOURCE_DECOUPE), np.full(len(repli), SOURCE_POINT_MEDIAN)]),
        }
    )
    agg = morceaux.groupby("id_osm").agg(length_in_scope_m=("length_m", "sum"), n_communes=("code_commune", "nunique"))
    totaux = totaux.merge(agg, on="id_osm", how="left").fillna({"length_in_scope_m": 0.0, "n_communes": 0})
    totaux["n_communes"] = totaux["n_communes"].astype(int)
    totaux["outside_m"] = (totaux["length_osm_m"] - totaux["length_in_scope_m"]).clip(lower=0)
    print(
        f"[decoupe] {len(morceaux):,} morceaux ; {(totaux['n_communes'] >= 2).sum():,} tronçons sur >= 2 communes ; "
        f"{(totaux['length_in_scope_m'] == 0).sum():,} tronçons entièrement hors périmètre ; {len(repli):,} en repli point médian ; "
        f"{totaux['length_osm_m'].sum() / 1000:.0f} km OSM, {totaux['length_in_scope_m'].sum() / 1000:.0f} km dans le périmètre"
    )
    return ResultatDecoupe(morceaux, totaux)


def ecrire_geometries(dossier_sortie: Path, ids: np.ndarray, wkb: np.ndarray, source: str) -> pd.DataFrame:
    """Stocke les géométries à côté de la découpe pour qu'elle puisse être refaite sans repasser par le
    PBF ; les renvoie en table."""
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"id_osm": ids, "wkb": wkb})
    table.to_parquet(dossier_sortie / "geometries.parquet", compression="zstd")
    (dossier_sortie / "SOURCE.txt").write_text(f"{source}\n", encoding="utf-8")
    return table


def lire_geometries(dossier_sortie: Path) -> pd.DataFrame:
    return pd.read_parquet(dossier_sortie / "geometries.parquet")
