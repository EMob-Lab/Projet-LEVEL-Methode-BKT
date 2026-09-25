# -*- coding: utf-8 -*-
"""Les longueurs de réseau cyclable DE RÉFÉRENCE, vendorisées dans
'data/donnees_brutes/reference/longueurs_reseau_reference.csv' : le résultat déjà calculé, pour
CHACUNE des méthodes comparées dans ce dossier (sauf la méthode primitive, jamais passée à l'échelle
de la France - voir 'README.md'), par commune et par année, 2019-2025.

Utilisée ici pour DEUX choses :

1. Vérifier '01_geovelo_seul/longueur_geovelo.py' et '02_FOB/' sans avoir besoin des sources brutes
   (exports GéoVélo, extraits OpenStreetMap annuels) - volumineuses, tierces, pas fournies avec ce
   dépôt (voir '00_transformation_des_donnees/README.md' pour la même remarque).
2. Le comparatif des trois méthodes (voir le notebook de ce dossier).

Trois scénarios dans le fichier :

    GV       GéoVélo seul (voir '01_geovelo_seul/') - le réseau tel que numérisé par GéoVélo, point.
    FOB      OpenStreetMap + modèle appris (voir '02_FOB/') - TOUT ce que le modèle classe comme
             appartenant au réseau cyclable "Familles d'Objets Bicyclette", aménagements dédiés ou non
             (une rue calme en zone 30 sans marquage peut compter, si le modèle la juge cyclable).
    FOB_AM   le même réseau FOB, restreint aux tronçons dont l'AMÉNAGEMENT prédit est un vrai
             aménagement dédié (pas 'AUTRE'/'AUCUN'/'NON_RENSEIGNE') - voir 'labels.NON_FACILITY_TYPES'
             dans '02_FOB/'. C'est la colonne à regarder pour une analyse des AMÉNAGEMENTS
             (voir le notebook '02_FOB/02_FOB.ipynb', section aménagements).

Chaque scénario a deux colonnes, 'len_d'/'len_g' (longueur effective côté droit/gauche - un côté
BIDIRECTIONNEL compte double, convention du projet, voir '01_geovelo_seul/longueur_geovelo.py')."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCENARIOS = ("GV", "FOB", "FOB_AM")


def charger_reference(fichier: Path | str) -> pd.DataFrame:
    """La table de référence telle quelle : 'annee, code_commune, nom_commune, code_departement,
    len_d_GV, len_g_GV, len_d_FOB, len_g_FOB, len_d_FOB_AM, len_g_FOB_AM'."""
    return pd.read_csv(fichier, dtype={"code_commune": str, "code_departement": str})


def longueur_effective(reference: pd.DataFrame, scenario: str) -> pd.Series:
    """Longueur effective totale (len_d + len_g) d'un scénario, par ligne (commune, année)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario={scenario!r} attendu parmi {SCENARIOS}")
    return reference[f"len_d_{scenario}"].fillna(0.0) + reference[f"len_g_{scenario}"].fillna(0.0)


def totaux_nationaux(reference: pd.DataFrame) -> pd.DataFrame:
    """Linéaire national (km) par année et par scénario - le premier chiffre à regarder pour comparer
    les méthodes (voir le notebook comparatif)."""
    table = reference.copy()
    for scenario in SCENARIOS:
        table[scenario] = longueur_effective(table, scenario)
    return table.groupby("annee")[list(SCENARIOS)].sum().round(0)
