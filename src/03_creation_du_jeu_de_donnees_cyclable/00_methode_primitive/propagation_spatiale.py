"""Rattrape une partie des tronçons 'incertain' de 'filtre_regles.py' : un même itinéraire cyclable est
souvent découpé par OSM en plusieurs tronçons aux tags inégaux (un panneau vélo photographié sur un
tronçon, pas sur le suivant). Si un tronçon 'incertain' porte le MÊME NOM DE RUE qu'un tronçon
'cyclable' tout proche, on le reclasse 'cyclable' à son tour.

Méthode assez prudente : aucune propagation par simple contact géométrique (deux rues différentes
qui se croisent ne devraient pas se "contaminer") - il faut le même nom de rue, et des types de voie
compatibles (voir 'ROUTES_COMPATIBLES')."""

from __future__ import annotations

import re

import geopandas as gpd
import numpy as np
from filtre_regles import CYCLABLE, HIGHWAY_VOIE_RAPIDE, INCERTAIN
from shapely.geometry import Point

LAMBERT93 = 2154
ROUTES_LOCALES = {"residential", "service", "unclassified", "living_street"}
ROUTES_COLLECTRICES = {"tertiary", "tertiary_link", "secondary", "secondary_link"}


def _normaliser_nom(nom) -> str:
    """Nettoie un nom de rue pour comparaison (casse, espaces, ponctuation) - GARDE le préfixe (rue,
    avenue...) : 'Rue Victor Hugo' et 'Avenue Victor Hugo' doivent rester DIFFÉRENTS."""
    if not nom or (isinstance(nom, float) and np.isnan(nom)):
        return ""
    nom = str(nom).strip().lower()
    nom = re.sub(r"[,.]+", "", nom)
    return re.sub(r"\s+", " ", nom).strip()


def _routes_compatibles(route_1: str, route_2: str) -> bool:
    """Deux types de voie OSM peuvent-ils raisonnablement faire partie du même itinéraire ?"""
    if route_1 == route_2:
        return True
    if route_1 in ROUTES_LOCALES and route_2 in ROUTES_LOCALES:
        return True
    if "cycleway" in (route_1, route_2):
        return True
    if route_1 in ROUTES_COLLECTRICES and route_2 in ROUTES_COLLECTRICES:
        return True
    return (route_1 in ROUTES_LOCALES and route_2 in ROUTES_COLLECTRICES) or (
        route_2 in ROUTES_LOCALES and route_1 in ROUTES_COLLECTRICES
    )


def propager(
    troncons: gpd.GeoDataFrame,
    colonne_statut: str = "statut",
    *,
    rayon_m: float = 5.0,
    iterations_max: int = 2,
) -> gpd.GeoDataFrame:
    """Reclasse en 'cyclable' les tronçons 'incertain' qui touchent (à 'rayon_m' près) un tronçon déjà
    'cyclable' de MÊME NOM DE RUE - ou dont les DEUX extrémités touchent chacune un tronçon cyclable de
    même nom (un "pont" entre deux tronçons cyclables du même itinéraire).

    Répète jusqu'à 'iterations_max' fois (une reclassification peut en permettre une autre juste à
    côté) ou jusqu'à ce que plus rien ne change. Renvoie une COPIE de 'troncons' (colonne 'colonne_statut'
    mise à jour)."""
    troncons = troncons.copy()
    if troncons.crs is None:
        troncons = troncons.set_crs("EPSG:4326")
    if troncons.crs.is_geographic:
        troncons = troncons.to_crs(LAMBERT93)
    index_spatial = troncons.sindex

    for iteration in range(1, iterations_max + 1):
        masque_incertain = troncons[colonne_statut] == INCERTAIN
        avant = int((troncons[colonne_statut] == CYCLABLE).sum())
        for idx in troncons.index[masque_incertain]:
            if _a_reclassifier(troncons, idx, index_spatial, colonne_statut, rayon_m):
                troncons.loc[idx, colonne_statut] = CYCLABLE
        nouveaux = int((troncons[colonne_statut] == CYCLABLE).sum()) - avant
        print(
            f"[propagation] itération {iteration} : {nouveaux} tronçon(s) incertain(s) -> cyclable"
        )
        if nouveaux == 0:
            break
    return troncons


def _a_reclassifier(
    troncons: gpd.GeoDataFrame, idx, index_spatial, colonne_statut: str, rayon_m: float
) -> bool:
    segment = troncons.loc[idx]
    nom_segment = _normaliser_nom(segment.get("name"))
    if not nom_segment:
        return False
    route_segment = str(segment.get("highway", "")).lower()
    if route_segment in HIGHWAY_VOIE_RAPIDE:
        return False

    zone = segment.geometry.buffer(rayon_m)
    candidats = [
        i for i in index_spatial.intersection(zone.bounds) if troncons.index[i] != idx
    ]
    if not candidats:
        return False
    voisins = troncons.iloc[candidats]
    voisins_cyclables = voisins[voisins[colonne_statut] == CYCLABLE]
    if voisins_cyclables.empty:
        return False

    # condition 1 : un voisin cyclable de même nom (et route compatible) à proximité
    for _, voisin in voisins_cyclables.iterrows():
        if _normaliser_nom(voisin.get("name")) != nom_segment:
            continue
        if _routes_compatibles(route_segment, str(voisin.get("highway", "")).lower()):
            return True

    # condition 2 : "pont" - les deux extrémités du segment touchent chacune un cyclable de même nom
    coords = list(segment.geometry.coords)
    if len(coords) < 2:
        return False
    debut, fin = Point(coords[0]), Point(coords[-1])
    touche_debut = touche_fin = False
    for _, voisin in voisins_cyclables.iterrows():
        if _normaliser_nom(voisin.get("name")) != nom_segment:
            continue
        if not _routes_compatibles(
            route_segment, str(voisin.get("highway", "")).lower()
        ):
            continue
        if voisin.geometry.distance(debut) < 0.3:
            touche_debut = True
        if voisin.geometry.distance(fin) < 0.3:
            touche_fin = True
    return touche_debut and touche_fin
