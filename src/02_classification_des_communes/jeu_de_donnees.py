# -*- coding: utf-8 -*-
"""Construit le jeu d'entraînement de la classification : quel TYPE de commune correspond à quel
cluster d'usage (voir '01_clustering_des_usages') ?

Les capteurs nous disent le cluster d'usage des communes qui en hébergent un (via l'assignation de
cluster de l'étape 01). Apprendre d'eux quel type de commune correspond à chaque cluster (population,
niveau de densité, distance à la mer/aux rivières/à une grande ville, relief) permet de PRÉDIRE le
cluster de toute autre commune, même sans capteur (voir 'prediction.py').

Les lignes d'entraînement sont les CAPTEURS (une ligne par capteur, portant les variables de sa
commune) - une commune à plusieurs capteurs pèserait alors plus lourd dans l'apprentissage, donc chaque
ligne est pondérée '1 / nombre de capteurs de la commune' et les plis de validation croisée sont
groupés par commune (voir 'entrainement.py' - aucune commune ne doit apparaître à la fois dans
l'entraînement et la validation d'un même pli)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

NIVEAU_DENSITE = "LIBDENS7"  # grille de densité INSEE à 7 niveaux (voir 00_transformation_des_donnees/d_socio_eco)
DISTANCE_URBAINE = "dist_gcu_group_km_T15"  # préparée par l'étape 00 (d_geographie/urbain.py)
VARIABLES_NUMERIQUES = [
    "log_pop",
    "log_pop_density",
    "dist_coast_km",
    "dist_river_km",
    "dist_canal_km",
    "log_dist_coast",
    "log_dist_river",
    "log_dist_canal",
    "near_coast_20km",
    "alt_commune_m",
    "alt_range_m",
    "alt_std_m",
    "alt_mean_m",
    "is_mountainous",
]
# la distance urbaine ajoutée aux variables purement géographiques : améliore légèrement la prédiction
# (testé par validation croisée groupée, voir 'entrainement.py').
VARIABLES = [DISTANCE_URBAINE] + VARIABLES_NUMERIQUES


@dataclass
class JeuEntrainement:
    X: pd.DataFrame
    y: pd.Series
    poids: pd.Series
    groupes: pd.Series  # commune de chaque ligne, pour la validation croisée groupée


def matrice_conception(communes: pd.DataFrame, colonnes: list[str] | None = None) -> pd.DataFrame:
    """Variables numériques plus niveaux de densité en one-hot. Avec 'colonnes', aligne sur un modèle
    déjà entraîné (colonnes absentes mises à 0 - une commune dont le niveau de densité n'a pas été vu à
    l'entraînement ne doit pas faire planter la prédiction)."""
    disponibles = [c for c in VARIABLES if c in communes.columns]
    X = communes[disponibles].copy().join(pd.get_dummies(communes[NIVEAU_DENSITE], prefix="DENS7"))
    if colonnes is None:
        return X
    for colonne in colonnes:
        if colonne not in X.columns:
            X[colonne] = 0.0
    return X[colonnes].fillna(0.0)


def jeu_entrainement(communes: pd.DataFrame, capteurs: pd.DataFrame, labels: pd.Series) -> JeuEntrainement:
    """'communes' : table des communes ('code_commune' + variables, voir 'table_communes.py' de
    l'étape 00). 'capteurs' : 'id_site, id_commune_str' (voir 'sensor_years.parquet'). 'labels' :
    cluster par 'id_site' (voir 'cluster_assignments.parquet' de l'étape 01)."""
    lignes = capteurs.drop_duplicates("id_site")[["id_site", "id_commune_str"]].copy()
    lignes["cluster"] = lignes["id_site"].map(labels)
    lignes = lignes.dropna(subset=["cluster", "id_commune_str"])
    lignes["poids"] = 1.0 / lignes.groupby("id_commune_str")["id_site"].transform("size")
    jointure = lignes.merge(communes, left_on="id_commune_str", right_on="code_commune", how="inner").reset_index(drop=True)
    X = matrice_conception(jointure)
    valide = X.notna().all(axis=1)
    return JeuEntrainement(X[valide], jointure.loc[valide, "cluster"].astype(int), jointure.loc[valide, "poids"], jointure.loc[valide, "id_commune_str"])


def effectifs_par_cluster(jeu: JeuEntrainement) -> pd.DataFrame:
    """Lignes et communes distinctes par cluster - les clusters représentés par trop peu de communes ne
    peuvent pas être appris de façon fiable (voir 'clusters_rares')."""
    return pd.DataFrame({"lignes": jeu.y.value_counts().sort_index(), "communes": jeu.groupes.groupby(jeu.y).nunique()}).fillna(0).astype(int)


def clusters_rares(jeu: JeuEntrainement, min_communes: int = 10) -> list[int]:
    """Clusters représentés par moins de 'min_communes' communes distinctes - pas assez pour une
    validation croisée groupée à 5 plis fiable (voir 'entrainement.py')."""
    effectifs = effectifs_par_cluster(jeu)
    return effectifs.index[effectifs["communes"] < min_communes].tolist()
