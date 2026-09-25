"""La MÉTHODE PRIMITIVE : un filtre à RÈGLES ÉCRITES À LA MAIN (pas de modèle appris) qui classe
chaque tronçon OpenStreetMap en 'cyclable' / 'non_cyclable' / 'incertain' à partir de ses seuls tags
bruts ('highway', 'bicycle', 'cycleway', 'surface', 'maxspeed', 'access'). Testée sur une petite zone
(2 km autour de Lyon, puis Grenoble) AVANT le passage à l'échelle de la France entière.

**Abandonnée au profit de FOB** (voir '02_FOB/') : un filtre à règles fixes ne capture pas la variété
des façons de tagger une même réalité sur le terrain (d'où la catégorie fourre-tout "incertain", ~20 à
30% des tronçons dans les tests Lyon/Grenoble) - FOB apprend ces règles à partir de vrais exemples
GéoVélo plutôt que de les écrire à la main, et gère beaucoup plus de tags (~80 clés contre 6 ici).

Gardée ici pour comparaison (voir le README de ce dossier et le notebook comparatif) - PAS utilisée par
les étapes suivantes du projet.

explication du CLASSEMENT :

Trois catégories, par ordre de priorité (une règle NON_CYCLABLE bat toujours une règle CYCLABLE) :

- 'non_cyclable' : interdiction explicite ('bicycle=no', 'access=no'), voie rapide sans aménagement
  protecteur, vitesse > 50 km/h sans aménagement, mauvais revêtement sans aménagement, ou aménagement
  "léger" (bande simplement peinte) à plus de 70 km/h (dangereux).
- 'cyclable' : signal POSITIF clair - voie cyclable dédiée ('highway=cycleway'), aménagement protecteur
  ('cycleway=track'/'separate') à vitesse raisonnable, aménagement léger à vitesse basse, rue
  piétonne/zone de rencontre avec vélo autorisé, ou 'bicycle=designated'.
- 'incertain' : ni signal positif ni négatif net - par exemple une rue résidentielle à 30 km/h SANS
  aucun tag vélo. Catégorie délibérément large : le filtre est CONSERVATEUR (mieux vaut "je ne sais
  pas" qu'un faux positif)."""

from __future__ import annotations

import re

import pandas as pd

NON_CYCLABLE = "non_cyclable"
CYCLABLE = "cyclable"
INCERTAIN = "incertain"

HIGHWAY_VOIE_RAPIDE = {"motorway", "motorway_link", "trunk", "trunk_link"}
CYCLEWAY_PROTECTEUR = {
    "track",
    "separate",
    "opposite_track",
}  # infrastructure séparée de la route
CYCLEWAY_LEGER = {
    "lane",
    "shared_lane",
    "advisory",
    "exclusive",
    "link",
    "opposite_lane",
}  # marquage au sol seulement
MAUVAIS_REVETEMENTS = {
    "unpaved",
    "gravel",
    "dirt",
    "ground",
    "grass",
    "sand",
    "cobblestone",
    "sett",
}

_NOMBRE = re.compile(r"\d+")


def _vitesse_numerique(maxspeed: str | None) -> int | None:
    """Extrait un nombre de 'maxspeed' (ex. '50', '30 mph') - None si absent/illisible."""
    if not maxspeed or pd.isna(maxspeed):
        return None
    trouve = _NOMBRE.search(str(maxspeed))
    return int(trouve.group()) if trouve else None


def classifier_troncon(tags: dict) -> str:
    """Classe UN tronçon OSM à partir de ses tags bruts (voir le docstring du module pour les règles).

    'tags' : un dictionnaire (ou une ligne de DataFrame convertie via '.to_dict()') avec au moins les
    clés 'highway', 'bicycle', 'cycleway', 'access', 'maxspeed', 'surface' - une clé absente est traitée
    comme une chaîne vide."""
    highway = str(tags.get("highway") or "").strip().lower()
    cycleway = str(tags.get("cycleway") or "").strip().lower()
    bicycle = str(tags.get("bicycle") or "").strip().lower()
    access = str(tags.get("access") or "").strip().lower()
    surface = str(tags.get("surface") or "").strip().lower()
    vitesse = _vitesse_numerique(tags.get("maxspeed"))

    # ── NON_CYCLABLE : un seul signal négatif suffit ──────────────────────────────────
    if bicycle == "no":
        return NON_CYCLABLE
    if access == "no" and bicycle not in ("yes", "designated", "permissive"):
        return NON_CYCLABLE
    if (
        highway in HIGHWAY_VOIE_RAPIDE
        and cycleway not in CYCLEWAY_PROTECTEUR
        and bicycle not in ("yes", "designated")
    ):
        return NON_CYCLABLE
    if (
        vitesse is not None
        and vitesse > 50
        and cycleway not in (CYCLEWAY_PROTECTEUR | {"lane"})
        and bicycle not in ("yes", "designated")
    ):
        return NON_CYCLABLE
    if (
        surface in MAUVAIS_REVETEMENTS
        and cycleway not in CYCLEWAY_PROTECTEUR
        and bicycle not in ("yes", "designated")
    ):
        return NON_CYCLABLE
    if cycleway in CYCLEWAY_LEGER and vitesse is not None and vitesse > 70:
        return NON_CYCLABLE

    # ── CYCLABLE : seulement un signal POSITIF clair ──────────────────────────────────
    if highway == "cycleway":
        return CYCLABLE
    if highway == "pedestrian" and bicycle in ("yes", "designated"):
        return CYCLABLE
    if highway == "living_street":
        return CYCLABLE
    if highway == "path" and bicycle in ("designated", "yes"):
        return CYCLABLE
    if cycleway in CYCLEWAY_PROTECTEUR and (vitesse is None or vitesse <= 50):
        return CYCLABLE
    if cycleway in CYCLEWAY_LEGER:
        if vitesse is not None and vitesse <= 30:
            return CYCLABLE
        if (
            vitesse is not None
            and vitesse <= 50
            and surface in {"asphalt", "concrete", "paved"}
        ):
            return CYCLABLE
        if vitesse is None and highway in ("residential", "service", "unclassified"):
            return CYCLABLE
    if bicycle == "designated":
        return CYCLABLE

    # ── ni positif ni négatif : on ne sait pas (volontairement large, voir le docstring) ──
    return INCERTAIN


def classifier_troncons(troncons: pd.DataFrame) -> pd.Series:
    """Applique 'classifier_troncon' à chaque ligne d'une table de tronçons - une colonne par tag
    utilisé ('highway', 'bicycle', 'cycleway', 'access', 'maxspeed', 'surface')."""
    return troncons.apply(lambda ligne: classifier_troncon(ligne.to_dict()), axis=1)
