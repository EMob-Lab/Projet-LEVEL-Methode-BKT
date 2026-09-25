# -*- coding: utf-8 -*-
"""Des tags OSM bruts (voir 'osm_extraction.py') à la MATRICE DE VARIABLES que les modèles LightGBM
reçoivent réellement (voir 'entrainement.py'/'application_modele.py').

Le VOCABULAIRE de chaque clé est FIGÉ dans une "spec" calculée une fois sur le snapshot d'entraînement
('construire_spec'), pour qu'une catégorie signifie la même chose d'une année à l'autre : une valeur
absente de la spec devient '(autre)', et un tag absent reste une vraie valeur manquante (ce que
LightGBM gère nativement, sans imputation).

En plus des tags bruts, quelques colonnes DÉRIVÉES : l'aménagement cyclable "effectif" de chaque côté
(repli de 'cycleway:right' vers 'cycleway:both' puis 'cycleway' simple si absent), des lectures
numériques de vitesse/largeur/nombre de voies, et un comptage des clés liées au vélo présentes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from osm_schema import CLES

AUTRE = "(autre)"
CLES_NUMERIQUES = {"maxspeed": "maxspeed_num", "lanes": "lanes_num", "width": "width_num", "cycleway:width": "cw_width_num"}
COLONNES_PETITS_ENTIERS = ("name_kind", "ref_kind", "rel_rank")
_CATEGORIES_ENTIERES_FIGEES = {"name_kind": list(range(0, 7)), "ref_kind": [0, 1, 2], "rel_rank": [-1, 0, 1, 2, 3, 4]}
_PREFIXES_CLES_VELO = ("cycleway", "bicycle", "oneway:bicycle", "cyclestreet")

_NOMBRE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)")
_VITESSES_NOMMEES = {"fr:urban": 50.0, "urban": 50.0, "fr:rural": 80.0, "rural": 80.0, "fr:motorway": 130.0, "walk": 7.0, "fr:walk": 7.0}
_MPH_VERS_KMH = 1.609


def nombre_depuis_tag(valeur: str | None) -> float:
    """Valeur numérique d'un tag comme '50', '30 mph', '2,5 m' ou 'fr:urban' (NaN si aucune)."""
    if valeur is None:
        return np.nan
    minuscules = valeur.lower()
    if minuscules in _VITESSES_NOMMEES:
        return _VITESSES_NOMMEES[minuscules]
    trouve = _NOMBRE.match(valeur)
    if not trouve:
        return np.nan
    nombre = float(trouve.group(1).replace(",", "."))
    return nombre * _MPH_VERS_KMH if "mph" in minuscules else nombre


# ── vocabulaire figé (la "spec") ────────────────────────────────────────────────────────────────────
def construire_spec(comptages_valeurs: pd.DataFrame, effectif_min: int = 20, max_valeurs: int = 250) -> dict[str, list[str]]:
    """Vocabulaire figé par clé : les 'max_valeurs' valeurs les plus fréquentes vues au moins
    'effectif_min' fois (voir 'osm_extraction.ecrire_table_tags' pour 'comptages_valeurs')."""
    spec = {}
    for cle in CLES:
        lignes = comptages_valeurs[comptages_valeurs["key"] == cle].sort_values("n_ways", ascending=False)
        lignes = lignes[(lignes["n_ways"] >= effectif_min) & (lignes["value"] != AUTRE)].head(max_valeurs)
        spec[cle] = lignes["value"].tolist()
    return spec


def sauvegarder_spec(spec: dict, chemin: str | Path) -> None:
    Path(chemin).write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")


def charger_spec(chemin: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(chemin).read_text(encoding="utf-8"))


# ── matrice de variables ────────────────────────────────────────────────────────────────────────────
def _recoder(colonne: pd.Series, valeurs: list[str]) -> pd.Categorical:
    """Réexprime une colonne catégorielle brute dans les catégories figées de la spec (+ '(autre)')."""
    categories = valeurs + [AUTRE]
    position = {v: i for i, v in enumerate(categories)}
    correspondance = np.array([position.get(v, len(categories) - 1) for v in colonne.cat.categories] + [-1], dtype=np.int16)
    return pd.Categorical.from_codes(correspondance[colonne.cat.codes.to_numpy()], categories=categories)


def _premier_present(colonnes: list[pd.Series], categories: list[str]) -> pd.Categorical:
    """Premier non-nul, ligne par ligne, parmi 'colonnes' (dans l'ordre de priorité donné)."""
    resultat = colonnes[0].astype(object)
    for autre in colonnes[1:]:
        resultat = resultat.where(resultat.notna(), autre.astype(object))
    return pd.Categorical(resultat, categories=categories)


def _categories_de(spec: dict, cles: list[str]) -> list[str]:
    return sorted({v for k in cles for v in spec[k]} | {AUTRE})


def _numerique_de(X: pd.DataFrame, cle: str) -> pd.Series:
    categories = list(X[cle].cat.categories)
    table = np.array([nombre_depuis_tag(v) if v != AUTRE else np.nan for v in categories] + [np.nan])
    return pd.Series(table[X[cle].cat.codes.to_numpy()], index=X.index)


_REPLIS_PAR_COTE = {
    "cw_right_eff": ["cycleway:right", "cycleway:both", "cycleway"],
    "cw_left_eff": ["cycleway:left", "cycleway:both", "cycleway"],
    "cw_lane_right_eff": ["cycleway:right:lane", "cycleway:both:lane", "cycleway:lane"],
    "cw_lane_left_eff": ["cycleway:left:lane", "cycleway:both:lane", "cycleway:lane"],
}


def vers_matrice(brut: pd.DataFrame, spec: dict[str, list[str]]) -> pd.DataFrame:
    """Matrice de variables (catégorielles + numériques) pour une tranche de la table de tags brute.

    Les noms de colonnes utilisent '__' au lieu de ':' (LightGBM rejette les deux-points) - voir
    'nom_lisible' pour les restaurer à l'affichage."""
    X = pd.DataFrame(index=brut.index)
    for cle in CLES:
        X[cle] = _recoder(brut[cle], spec[cle])
    for cle, nom in CLES_NUMERIQUES.items():
        X[nom] = _numerique_de(X, cle).to_numpy()
    for nom, cles in _REPLIS_PAR_COTE.items():
        X[nom] = _premier_present([X[k] for k in cles], _categories_de(spec, cles))
    colonnes_largeur = [X["cw_width_num"]] + [_numerique_de(X, k) for k in ("cycleway:right:width", "cycleway:left:width", "cycleway:both:width")]
    X["cw_width_eff"] = pd.concat(colonnes_largeur, axis=1).bfill(axis=1).iloc[:, 0]
    cles_velo = [k for k in CLES if k.startswith(_PREFIXES_CLES_VELO)]
    X["n_cycle_keys"] = brut[cles_velo].notna().sum(axis=1).astype(np.int8)
    for nom in COLONNES_PETITS_ENTIERS:
        X[nom] = pd.Categorical(brut[nom].to_numpy(), categories=_CATEGORIES_ENTIERES_FIGEES[nom])
    X["rel_count"] = brut["rel_count"].astype(np.int16)
    X["n_nodes"] = brut["n_nodes"].astype(np.int32)
    X.columns = [c.replace(":", "__") for c in X.columns]
    return X


def nom_lisible(nom: str) -> str:
    return nom.replace("__", ":")
