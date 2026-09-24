# -*- coding: utf-8 -*-
"""Prédit le cluster d'usage de CHAQUE commune de France métropolitaine (34 428, pas seulement les
~1000 qui hébergent un capteur), et enchaîne jeu de données + entraînement + prédiction en une seule
fonction ('preparer_classification_communes').

Le clustering de l'étape 01 ne connaît que les communes qui hébergent (ou ont hébergé) un capteur.
Pour extrapoler le BKT à la France entière (étapes suivantes du projet), il faut prédire la pratique de
TOUTE commune, même sans capteur - c'est le rôle de la forêt aléatoire de 'entrainement.py', entraînée
sur la géographie des communes instrumentées (population, densité, distances, altitude) et LEUR cluster
réel (celui de leur(s) capteur(s), voir l'étape 01), puis appliquée à toutes les communes."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrainement import ModeleClassification, ajuster_foret_aleatoire  # noqa: E402
from jeu_de_donnees import NIVEAU_DENSITE, jeu_entrainement, matrice_conception  # noqa: E402


def predire_communes(modele: ModeleClassification, communes: pd.DataFrame, observees: set[str], noms_clusters: dict[int, str] | None = None) -> pd.DataFrame:
    """Cluster et probabilités de classe de chaque commune. 'extrapolee' = 1 si la commune n'héberge
    aucun capteur utilisé à l'entraînement (son cluster est une pure prédiction, pas une observation
    directe) - un simple indicateur, pas un filtre : toutes les communes sont prédites de la même façon."""
    X = matrice_conception(communes, modele.colonnes)
    sortie = communes[["code_commune", "nom_commune", NIVEAU_DENSITE]].copy()
    sortie["cluster_predit"] = modele.estimateur.predict(X)
    if noms_clusters:
        sortie["nom_cluster_predit"] = sortie["cluster_predit"].map(noms_clusters)
    probabilites = modele.estimateur.predict_proba(X)
    for position, cluster in enumerate(modele.estimateur.classes_):
        sortie[f"proba_cluster_{cluster}"] = probabilites[:, position].round(4)
    sortie["extrapolee"] = (~sortie["code_commune"].isin(observees)).astype(int)
    return sortie


def preparer_classification_communes(
    communes: pd.DataFrame,
    capteurs: pd.DataFrame,
    assignation_clusters: pd.DataFrame,
    *,
    noms_clusters: dict[int, str] | None = None,
    recherche: bool = False,
) -> tuple[pd.DataFrame, ModeleClassification]:
    """Entraîne sur le cluster de NOS propres capteurs ('assignation_clusters', voir
    '01_clustering_des_usages/pipeline.py' - PAS un modèle vendorisé) et prédit le cluster de TOUTE
    commune. Renvoie '(predictions, modele)' - sauvegardez le modèle vous-même si besoin (voir
    'ModeleClassification.sauvegarder')."""
    labels = assignation_clusters.set_index("id_site")["cluster"]
    jeu = jeu_entrainement(communes, capteurs, labels)
    print(f"[classification] jeu d'entraînement : {len(jeu.X)} capteurs dans {jeu.groupes.nunique()} communes, {jeu.X.shape[1]} variables")
    modele = ajuster_foret_aleatoire(jeu, recherche=recherche)
    observees = set(jeu.groupes.unique())
    predictions = predire_communes(modele, communes, observees, noms_clusters)
    return predictions, modele
