# -*- coding: utf-8 -*-
"""Le cluster K=4 prédit de chaque commune DE RÉFÉRENCE : le run historique du projet, vendorisé dans
'data/donnees_brutes/reference/commune_clusters_k4_reference.csv' - PAS un modèle qu'on réentraîne,
juste une table de prédictions déjà connues et éprouvées pour les 34 428 communes.

**Utilisée PAR DÉFAUT** (voir 'pipeline.py', paramètre 'SOURCE_MODELE = "reference"') : une forêt
aléatoire n'est pas parfaitement déterministe d'un entraînement à l'autre (les quasi-égalités entre
arbres peuvent basculer) - même en réentraînant sur exactement les mêmes données avec la même méthode
('entrainement.py'), environ 2 % des communes changeraient de cluster prédit d'une exécution à l'autre.
Utiliser cette table de référence par défaut évite cette instabilité : les résultats du pipeline restent
les mêmes d'une exécution à l'autre, et cohérents avec l'historique du projet. Changez 'SOURCE_MODELE'
(dans 'pipeline.py' ou le notebook) pour entraîner un nouveau modèle à la place."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_COLONNES_PROBA = ["proba_cluster_0", "proba_cluster_1", "proba_cluster_2", "proba_cluster_3"]


def charger_reference(fichier: Path | str) -> pd.DataFrame:
    """'code_commune, nom_commune, cluster_predit, proba_cluster_0..3, extrapolee' de chaque commune,
    depuis la table de référence (voir la docstring du module) - déjà au même format que
    'prediction.predire_communes' produit pour un modèle fraîchement entraîné."""
    table = pd.read_csv(fichier, dtype={"code_commune": str})
    table["code_commune"] = table["code_commune"].astype(str).str.zfill(5)
    table = table.rename(columns={"predicted_cluster": "cluster_predit", "extrapolated": "extrapolee"})
    return table[["code_commune", "nom_commune", "cluster_predit", *_COLONNES_PROBA, "extrapolee"]]


def predictions_reference(communes: pd.DataFrame, fichier: Path | str, *, noms_clusters: dict[int, str] | None = None) -> pd.DataFrame:
    """Version "référence" de 'prediction.predire_communes' : pas de forêt aléatoire, juste une
    jointure sur les prédictions déjà connues - même forme de sortie, pour rester interchangeable avec
    un modèle fraîchement entraîné."""
    reference = charger_reference(fichier)
    sortie = communes[["code_commune"]].merge(reference, on="code_commune", how="left")
    manquantes = sortie["cluster_predit"].isna().sum()
    if manquantes:
        print(f"[reference] {manquantes:,} communes sans prédiction de référence (absentes de la table) - garde-fou honnête, ne devrait pas arriver")
    sortie["cluster_predit"] = sortie["cluster_predit"].astype(int)
    sortie["extrapolee"] = sortie["extrapolee"].astype(int)
    if noms_clusters:
        sortie["nom_cluster_predit"] = sortie["cluster_predit"].map(noms_clusters)
    return sortie
