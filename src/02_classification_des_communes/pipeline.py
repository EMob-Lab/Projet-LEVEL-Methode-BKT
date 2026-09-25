"""Étape 02 — classification des communes : entraîne une forêt aléatoire sur les communes qui
hébergent un capteur (et son cluster d'usage, voir '01_clustering_des_usages'), puis prédit le cluster
de TOUTE commune française, même sans capteur.

    python pipeline.py

**Priorité à un modèle déjà entraîné** : si 'models/classification_des_communes/foret_aleatoire.joblib'
existe déjà, ce script l'APPLIQUE tel quel plutôt que d'en entraîner un nouveau - un modèle choisi à la
main (par exemple avec 'RECHERCHE_HYPERPARAMETRES = True' un jour, ou des 'NOMS_CLUSTERS' précis) ne
doit jamais être silencieusement écrasé par un modèle aux paramètres par défaut. Mettez
'FORCER_LE_RECALCUL = True' pour réentraîner malgré tout.

Nécessite les sorties de '00_transformation_des_donnees' (table finale des communes, capteurs-années)
ET de '01_clustering_des_usages' (assignation de cluster) déjà préparées - lancez leur 'pipeline.py'
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
FICHIER_REFERENCE = RACINE / "data" / "donnees_brutes" / "reference" / "commune_clusters_k4_reference.csv"

DOSSIER_SORTIE = DONNEES_VALIDES / "classification"
DOSSIER_MODELES = RACINE / "models" / "classification_des_communes"

# "reference" (par défaut) : prédictions déjà connues, vendorisées (voir 'reference.py') - aucun
# entraînement nécessaire, résultats stables d'une exécution à l'autre. "modele" : entraîne (ou
# réutilise, voir FORCER_LE_RECALCUL) une forêt aléatoire à la place - voir 'entrainement.py'. Même
# principe que 'SOURCE_MODELE' à l'étape 01.
SOURCE_MODELE = "reference"

# Noms lisibles des clusters, facultatif ({0: "...", 1: "...", ...}) - laissé à None par défaut : le
# nombre de clusters (K) dépend du modèle choisi à l'étape 01, pas fixé ici (voir son README). Sans
# effet si un modèle déjà entraîné est réutilisé (voir ci-dessus) : ses propres noms, s'il en a, priment.
NOMS_CLUSTERS: dict[int, str] | None = None

# True pour relancer le réglage des hyperparamètres (RandomizedSearchCV, quelques minutes) plutôt que
# d'utiliser entrainement.PARAMETRES_PAR_DEFAUT - seulement si un NOUVEAU modèle est entraîné (voir
# la priorité au modèle déjà entraîné, ci-dessus).
RECHERCHE_HYPERPARAMETRES = False

FORCER_LE_RECALCUL = False  # True pour ignorer le modèle et les fichiers déjà présents et tout recalculer

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# à partir d'ici, plus aucune configuration - seulement l'enchaînement des étapes
# ═══════════════════════════════════════════════════════════════════════════════════════════════

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

import pandas as pd  # noqa: E402

from entrainement import ModeleClassification  # noqa: E402
from prediction import appliquer_modele_classification, preparer_classification_communes  # noqa: E402
from reference import predictions_reference  # noqa: E402


def main() -> None:
    fichier_predictions = DOSSIER_SORTIE / "commune_clusters.parquet"
    if fichier_predictions.exists() and not FORCER_LE_RECALCUL:
        print(f"== classification des communes : déjà préparé ({fichier_predictions.name})")
        return

    if not FICHIER_COMMUNES.exists():
        raise FileNotFoundError(f"{FICHIER_COMMUNES} manquant - lancez d'abord le pipeline.py de l'étape qui le produit (voir le README de ce dossier).")
    communes = pd.read_parquet(FICHIER_COMMUNES)

    if SOURCE_MODELE == "reference":
        if not FICHIER_REFERENCE.exists():
            raise FileNotFoundError(f"{FICHIER_REFERENCE} manquant.")
        print("== classification des communes : prédictions de référence (voir reference.py)")
        predictions = predictions_reference(communes, FICHIER_REFERENCE, noms_clusters=NOMS_CLUSTERS)
    elif SOURCE_MODELE == "modele":
        for fichier in (FICHIER_CAPTEURS_ANNEES, FICHIER_ASSIGNATION_CLUSTERS):
            if not fichier.exists():
                raise FileNotFoundError(f"{fichier} manquant - lancez d'abord le pipeline.py de l'étape qui le produit (voir le README de ce dossier).")
        capteurs = pd.read_parquet(FICHIER_CAPTEURS_ANNEES)
        assignation = pd.read_parquet(FICHIER_ASSIGNATION_CLUSTERS)
        fichier_modele = DOSSIER_MODELES / "foret_aleatoire.joblib"

        if fichier_modele.exists() and not FORCER_LE_RECALCUL:
            print(f"== classification des communes : modèle déjà entraîné réutilisé ({fichier_modele.name})")
            modele = ModeleClassification.charger(fichier_modele)
            predictions = appliquer_modele_classification(modele, communes, capteurs, assignation, noms_clusters=NOMS_CLUSTERS)
        else:
            print("== classification des communes : entraînement d'un nouveau modèle")
            predictions, modele = preparer_classification_communes(
                communes, capteurs, assignation, noms_clusters=NOMS_CLUSTERS, recherche=RECHERCHE_HYPERPARAMETRES
            )
            DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
            modele.sauvegarder(fichier_modele)
            print(f"écrit : {fichier_modele}")
    else:
        raise ValueError(f"SOURCE_MODELE doit être 'reference' ou 'modele', reçu {SOURCE_MODELE!r}")

    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(fichier_predictions, index=False)
    print(f"écrit : {fichier_predictions}")


if __name__ == "__main__":
    main()
