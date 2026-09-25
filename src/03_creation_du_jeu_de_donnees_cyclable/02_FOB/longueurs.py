# -*- coding: utf-8 -*-
"""Longueur EFFECTIVE du réseau FOB par commune - le pont entre le réseau OSM prédit et le calcul du
BKT (étapes suivantes du projet, pas encore construites dans ce dépôt).

La longueur effective d'un tronçon compte un côté BIDIRECTIONNEL deux fois (convention du projet,
voir 'etiquettes.multiplicateur_direction'), répartie en 'len_d'/'len_g'. Renvoie TOUJOURS deux tables :

    FOB     tout le réseau que le modèle classe comme cyclable, y compris les tronçons "juste"
            cyclables sans aménagement particulier (rue résidentielle calme, par exemple) - voir
            'application_modele.py'
    FOB_AM  le même réseau, restreint aux tronçons dont l'aménagement prédit est un vrai aménagement
            DÉDIÉ (exclut AUTRE/AUCUN/NON_RENSEIGNE, voir 'etiquettes.est_amenagement')

NI L'UNE NI L'AUTRE n'est "la meilleure" dans l'absolu : FOB couvre tout ce qui est effectivement
roulable à vélo (pertinent pour une exposition/un linéaire réellement parcouru), FOB_AM ne garde que ce
qui a une infrastructure dédiée (pertinent pour analyser l'infrastructure elle-même, voir le notebook,
section "analyse des aménagements", qui ne porte que sur FOB_AM). C'est aux étapes suivantes du projet
de choisir selon leur usage - c'est pour ça que la table de référence vendorisée (voir 'reference.py' du
dossier parent) garde les deux colonnes plutôt que de trancher ici."""

from __future__ import annotations

import numpy as np
import pandas as pd

import etiquettes as L
from decoupe_communes import ResultatDecoupe


def _multiplicateurs_par_cote(troncons: pd.DataFrame, alpha: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    return L.multiplicateur_direction(troncons["pred_sens_d"], alpha), L.multiplicateur_direction(troncons["pred_sens_g"], alpha)


def preparer_troncons_fob(troncons_annee: pd.DataFrame, alpha: float = 2.0) -> pd.DataFrame:
    """Tronçons FOB d'une année (déjà filtrés sur 'incl'), avec les multiplicateurs de côté et
    l'indicateur "aménagement dédié" ajoutés."""
    troncons = troncons_annee[troncons_annee["incl"]].reset_index(drop=True)
    troncons["md"], troncons["mg"] = _multiplicateurs_par_cote(troncons, alpha)
    troncons["amenagement"] = L.est_amenagement(L.type_principal(troncons["pred_ame_d"], troncons["pred_ame_g"]))
    return troncons


def _agreger(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(FOB entier, FOB aménagements seuls) - longueur effective par commune à partir de lignes avec
    'code_commune, len_d, len_g, amenagement'."""
    colonnes = ["len_d", "len_g"]
    return table.groupby("code_commune")[colonnes].sum().reset_index(), table[table["amenagement"]].groupby("code_commune")[colonnes].sum().reset_index()


# ═══════════════════════════════════════════════════════════════════════════════════════════════ FOB (découpe communale)
def fob_depuis_decoupe(decoupe: ResultatDecoupe, troncons_fob: pd.DataFrame, n_troncons_annee: int, annee: int, alpha: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Longueurs FOB par commune à partir de la découpe communale (voir 'decoupe_communes.py' : un
    tronçon à cheval sur plusieurs communes est réparti entre elles au prorata de la longueur qui tombe
    dans chacune, pas attribué en bloc à une seule).

    Renvoie '(fob, fob_amenagements_seuls, statistiques)'."""
    table = decoupe.morceaux.merge(troncons_fob[["id_osm", "md", "mg", "amenagement"]], on="id_osm", how="inner")
    table["len_d"] = table["length_m"] / 1000 * table["md"]
    table["len_g"] = table["length_m"] / 1000 * table["mg"]
    fob, amenagements = _agreger(table)
    stats = decoupe.statistiques(annee, n_troncons_annee)
    stats["km_effectif_perimetre"] = float(fob["len_d"].sum() + fob["len_g"].sum())
    print(f"[longueurs] FOB {annee} (découpe) : {len(troncons_fob):,} tronçons -> {len(fob):,} communes, {stats['km_effectif_perimetre']:.0f} km effectifs, {stats['km_hors_perimetre']:.0f} km hors périmètre")
    return fob, amenagements, stats
