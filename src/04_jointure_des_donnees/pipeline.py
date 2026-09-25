# -*- coding: utf-8 -*-
"""Étape 04 — jointure des données : assemble les sorties des étapes 00 à 03 (capteurs, communes,
clusters d'usage, classification, réseau cyclable) en TROIS TABLES - le pont vers les étapes 05/06/99,
qui ne liront plus QUE 'data/donnees_valides/bkt/' (jamais les dossiers des étapes 00-03 directement,
voir le README de ce dossier).

    python pipeline.py

Nécessite les sorties de '00_transformation_des_donnees' (communes, capteurs-années, population),
'01_clustering_des_usages' (assignation de cluster par capteur) et '02_classification_des_communes'
(cluster prédit par commune) déjà préparées - lancez leurs 'pipeline.py' d'abord si besoin. Le réseau
cyclable vient par défaut de la table de référence vendorisée (voir 'longueurs.py',
SOURCE_LONGUEURS)."""

from __future__ import annotations

import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - tout ce qu'il y a à changer pour adapter ce script se trouve ici.
# ═══════════════════════════════════════════════════════════════════════════════════════════════

RACINE = Path(__file__).resolve().parents[2]
DONNEES_VALIDES = RACINE / "data" / "donnees_valides"

FICHIER_COMMUNES_ZONES = DONNEES_VALIDES / "zones" / "communes.parquet"  # étape 00
FICHIER_SENSOR_YEARS = DONNEES_VALIDES / "capteurs" / "sensor_years.parquet"  # étape 00
DOSSIER_POPULATION = DONNEES_VALIDES / "zones"  # étape 00 (population.parquet)
FICHIER_CLUSTER_ASSIGNMENTS = DONNEES_VALIDES / "clustering" / "cluster_assignments.parquet"  # étape 01
FICHIER_COMMUNE_CLUSTERS = DONNEES_VALIDES / "classification" / "commune_clusters.parquet"  # étape 02
DOSSIER_RESEAU_FOB = DONNEES_VALIDES / "reseau_fob"  # étape 03 (02_FOB)
FICHIER_LONGUEURS_REFERENCE = RACINE / "data" / "donnees_brutes" / "reference" / "longueurs_reseau_reference.csv"

DOSSIER_SORTIE = DONNEES_VALIDES / "bkt"

ANNEES: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024, 2025)

# "reference" (par défaut) : longueurs vendorisées, les 7 années, aucun calcul frais nécessaire.
# "calcul" : utilise la vraie sortie de 02_FOB/pipeline.py pour les années déjà calculées (2024 dans ce
# dépôt), retombe sur la référence pour les autres - voir 'longueurs.py'.
SOURCE_LONGUEURS = "reference"

FORCER_LE_RECALCUL = False  # True pour ignorer les fichiers déjà présents et tout réassembler

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# à partir d'ici, plus aucune configuration - seulement l'enchaînement des étapes
# ═══════════════════════════════════════════════════════════════════════════════════════════════

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
sys.path.insert(0, str(RACINE / "src" / "00_transformation_des_donnees" / "d_zones"))

import pandas as pd  # noqa: E402

from assemblage import assembler, table_population_utilisee  # noqa: E402
from entrees import capteurs_actifs, communes_instrumentees, resoudre_cluster_communes, table_communes  # noqa: E402
from longueurs import SCENARIOS, charger_longueurs  # noqa: E402
from population import charger_population  # noqa: E402


def main() -> None:
    fichier_communes = DOSSIER_SORTIE / "communes.parquet"
    if fichier_communes.exists() and not FORCER_LE_RECALCUL:
        print(f"== jointure des données : déjà assemblé ({fichier_communes.name})")
        return

    for fichier in (FICHIER_COMMUNES_ZONES, FICHIER_SENSOR_YEARS, FICHIER_CLUSTER_ASSIGNMENTS, FICHIER_COMMUNE_CLUSTERS, FICHIER_LONGUEURS_REFERENCE):
        if not fichier.exists():
            raise FileNotFoundError(f"{fichier} manquant - lancez d'abord le pipeline.py de l'étape qui le produit (voir le README de ce dossier).")

    print("== jointure des données : assemblage des sorties des étapes 00 à 03")
    capteurs = capteurs_actifs(FICHIER_SENSOR_YEARS, FICHIER_CLUSTER_ASSIGNMENTS)
    communes = table_communes(FICHIER_COMMUNES_ZONES, FICHIER_COMMUNE_CLUSTERS)
    cluster_par_annee = resoudre_cluster_communes(capteurs, communes, ANNEES)
    instrumentees = communes_instrumentees(FICHIER_SENSOR_YEARS)
    longueurs = charger_longueurs(FICHIER_LONGUEURS_REFERENCE, DOSSIER_RESEAU_FOB, ANNEES, source=SOURCE_LONGUEURS)

    populations, population_utilisee = table_population_utilisee(ANNEES, lambda millesime: charger_population(DOSSIER_POPULATION, millesime))

    communes_assemblees = assembler(communes, longueurs, instrumentees, populations, cluster_par_annee, ANNEES, SCENARIOS)

    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    communes_assemblees.to_parquet(DOSSIER_SORTIE / "communes.parquet", index=False)
    capteurs.to_parquet(DOSSIER_SORTIE / "sensor_detail.parquet", index=False)
    population_utilisee.to_parquet(DOSSIER_SORTIE / "population_used.parquet", index=False)
    print(f"écrit : {DOSSIER_SORTIE} (communes.parquet, sensor_detail.parquet, population_used.parquet)")
    print(f"{len(communes_assemblees):,} lignes (année, commune) ; {len(capteurs):,} lignes (année, capteur)")


if __name__ == "__main__":
    main()
    print("\nétape 04 (jointure des données) : ok - les étapes suivantes ne lisent plus que data/donnees_valides/bkt/.")
