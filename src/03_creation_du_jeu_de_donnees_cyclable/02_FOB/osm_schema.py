# -*- coding: utf-8 -*-
"""Ce qu'on garde d'OpenStreetMap pour FOB : les clés de tags, et deux indicateurs peu coûteux dérivés
du texte libre (type de nom, type de référence)."""

from __future__ import annotations

import re

# Clés de tags brutes gardées par tronçon ("way"), choisies par prévalence parmi les tronçons GéoVélo
# (voir 'etiquettes.py') - ~80 clés, contre les 6 du filtre à règles de '00_methode_primitive/'.
CLES: tuple[str, ...] = (
    "highway", "surface", "oneway", "bicycle", "foot", "maxspeed", "lit", "segregated",
    "oneway:bicycle", "cycleway:right", "cycleway", "smoothness", "motor_vehicle", "cycleway:left",
    "sidewalk", "cycleway:both", "source:maxspeed", "traffic_sign", "zone:maxspeed", "footway",
    "cycleway:right:lane", "cycleway:both:lane", "cycleway:left:lane", "cycleway:lane",
    "cycleway:left:oneway", "cycleway:right:oneway", "cycleway:surface", "footway:surface",
    "lane_markings", "bridge", "horse", "access", "tracktype", "sidewalk:both", "sidewalk:left",
    "sidewalk:right", "crossing", "railway", "abandoned:railway", "junction", "service",
    "maxspeed:type", "busway", "busway:right", "busway:left", "psv", "bus", "lcn", "lanes", "width",
    "cycleway:width", "motorcar", "traffic_calming", "zone:traffic", "ramp:bicycle", "tunnel",
    "surface:colour", "shoulder:right", "bicycle_road", "cyclestreet", "cycleway:right:segregated",
    "cycleway:left:segregated", "cycleway:both:segregated", "sidewalk:bicycle", "bicycle:backward",
    "bicycle:forward", "designation", "cycleway:right:width", "cycleway:left:width",
    "cycleway:both:width", "layer", "area", "construction",
)
INDEX_CLES = {cle: i for i, cle in enumerate(CLES)}

# rang du réseau d'itinéraires cyclables auquel appartient un tronçon (relations OSM route=bicycle)
RANG_RESEAU = {"lcn": 1, "rcn": 2, "ncn": 3, "icn": 4}

_MOTIFS_NOM = (
    (1, re.compile(r"voie[ -]verte|greenway")),
    (2, re.compile(r"v[eé]lo\s*route|eurovelo|v[eé]loroute|tour de france")),
    (3, re.compile(r"v[eé]lo|cyclab|cycliste|cycle")),
    (4, re.compile(r"canal|halage|berge|digue|quai|rivi[eè]re|fleuve")),
    (5, re.compile(r"chemin|sentier|all[eé]e|passage|sente|piste|traverse")),
)
_REF_CYCLABLE = re.compile(r"^(eurovelo|ev|v)\s*\d+", re.I)


def type_de_nom(nom: str | None) -> int:
    """0 = pas de nom, 6 = nom sans mot-clé reconnu, 1..5 = classe de mot-clé (voie verte, véloroute,
    vélo, canal, chemin) - un indicateur peu coûteux calculé une fois à l'extraction, pas à chaque
    prédiction."""
    if not nom:
        return 0
    minuscules = nom.lower()
    for code, motif in _MOTIFS_NOM:
        if motif.search(minuscules):
            return code
    return 6


def type_de_reference(ref: str | None) -> int:
    """0 = pas de référence, 1 = référence d'itinéraire cyclable (ex. 'V50', 'EV6'), 2 = toute autre
    référence (numéro de route...)."""
    if not ref:
        return 0
    return 1 if _REF_CYCLABLE.match(ref.strip()) else 2
