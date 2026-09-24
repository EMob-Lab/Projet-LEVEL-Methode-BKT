# -*- coding: utf-8 -*-
"""Étape 02 — classification des communes : entraîne une forêt aléatoire sur les communes qui
hébergent un capteur (et son cluster d'usage, voir '01_clustering_des_usages'), puis prédit le cluster
de TOUTE commune française, même sans capteur.

    python pipeline.py

Nécessite les sorties de '00_transformation_des_donnees' (table finale des communes, capteurs-années)
ET de '01_clustering_des_usages' (assignation de cluster) déjà préparées - lancez leurs 'pipeline.py'
d'abord si besoin.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - tout ce qu'il y a à changer pour adapter ce script se trouve ici.
# ═══════════════════════════════════════════════════════════════════════════════════════════════

# Racine du dépôt VV6 (ce fichier est dans src/02_classification_des_communes/, donc la racine est 2
# niveaux au-dessus) - déduite automatiquement, pas besoin d'y toucher sauf cas particulier.
RACINE = Path(__file__).resolve().parents[2]

DONNEES_VALIDES = RACINE / "data" / "donnees_valides"
FICHIER_COMMUNES = DONNEES_VALIDES / "commune_features.parquet"  # étape 00
FICHIER_CAPTEURS_ANNEES = DONNEES_VALIDES / "capteurs" / "sensor_years.parquet"  # étape 00
FICHIER_ASSIGNATION_CLUSTERS = DONNEES_VALIDES / "clustering" / "cluster_assignments.parquet"  # étape 01

DOSSIER_SORTIE = DONNEES_VALIDES / "classification"
DOSSIER_MODELES = RACINE / "models" / "classification_des_communes"

# Noms lisibles des clusters, facultatif ({0: "...", 1: "...", ...}) - laissé à None par défaut : le
# nombre de clusters (K) dépend du modèle choisi à l'étape 01, pas fixé ici (voir son README).
NOMS_CLUSTERS: dict[int, str] | None = None

# True pour relancer le réglage des hyperparamètres (RandomizedSearchCV, quelques minutes) plutôt que
# d'utiliser entrainement.PARAMETRES_PAR_DEFAUT.
RECHERCHE_HYPERPARAMETRES = False

FORCER_LE_RECALCUL = False  # True pour ignorer les fichiers déjà présents et tout recalculer

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# à partir d'ici, plus aucune configuration - seulement l'enchaînement des étapes
# ═══════════════════════════════════════════════════════════════════════════════════════════════

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

import pandas as pd  # noqa: E402

from prediction import preparer_classification_communes  # noqa: E402


def main() -> None:
    fichier_predictions = DOSSIER_SORTIE / "commune_clusters.parquet"
    fichier_modele = DOSSIER_MODELES / "foret_aleatoire.joblib"
    if fichier_predictions.exists() and fichier_modele.exists() and not FORCER_LE_RECALCUL:
        print(f"== classification des communes : déjà préparé ({fichier_predictions.name})")
        return

    for fichier in (FICHIER_COMMUNES, FICHIER_CAPTEURS_ANNEES, FICHIER_ASSIGNATION_CLUSTERS):
        if not fichier.exists():
            raise FileNotFoundError(f"{fichier} manquant - lancez d'abord le pipeline.py de l'étape qui le produit (voir le README de ce dossier).")

    print("== classification des communes")
    communes = pd.read_parquet(FICHIER_COMMUNES)
    capteurs = pd.read_parquet(FICHIER_CAPTEURS_ANNEES)
    assignation = pd.read_parquet(FICHIER_ASSIGNATION_CLUSTERS)

    predictions, modele = preparer_classification_communes(
        communes, capteurs, assignation, noms_clusters=NOMS_CLUSTERS, recherche=RECHERCHE_HYPERPARAMETRES
    )

    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(fichier_predictions, index=False)
    print(f"écrit : {fichier_predictions}")

    DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
    modele.sauvegarder(fichier_modele)
    print(f"écrit : {fichier_modele}")


if __name__ == "__main__":
    main()
