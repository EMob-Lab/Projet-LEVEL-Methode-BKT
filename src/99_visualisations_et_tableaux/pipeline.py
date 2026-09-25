# -*- coding: utf-8 -*-
"""Étape 99 — indicateurs, incertitude statistique, export Excel, graphiques et cartes : tout ce qui
RESTITUE le BKT calculé aux étapes 04-06, rien qui le recalcule différemment.

    python pipeline.py

**Les indicateurs et intervalles de confiance de ce dossier décrivent NOTRE méthode d'estimation du
BKT (04-06), pas le "vrai" BKT national - personne ne l'observe, il n'y a pas de vérité terrain à
laquelle comparer notre estimation (voir le README).**
"""

from __future__ import annotations

import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - tout ce qu'il y a à changer pour adapter ce script se trouve ici.
# ═══════════════════════════════════════════════════════════════════════════════════════════════

RACINE = Path(__file__).resolve().parents[2]
DOSSIER_BKT = RACINE / "data" / "donnees_valides" / "bkt"

FICHIER_COMMUNES_BKT_OBS = DOSSIER_BKT / "communes_bkt_obs.parquet"  # étape 05
FICHIER_SENSOR_DETAIL = DOSSIER_BKT / "sensor_detail.parquet"  # étape 04
FICHIER_POPULATION_USED = DOSSIER_BKT / "population_used.parquet"  # étape 04

ANNEES: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024, 2025)
FICHIER_COMMUNES_ZONES = RACINE / "data" / "donnees_valides" / "zones" / "communes.parquet"  # étape 00
GRAINE_BOOTSTRAP = 42  # fixe : deux exécutions donnent exactement le même intervalle
CALCULER_INTERVALLES_CONFIANCE = True  # False pour ne calculer que les indicateurs (plus rapide)
CONSTRUIRE_CLASSEUR_EXCEL = True
CONSTRUIRE_CARTES = True
ANNEE_CARTES = 2025
FORCER_LE_RECALCUL = False

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# à partir d'ici, plus aucune configuration - seulement l'enchaînement des étapes
# ═══════════════════════════════════════════════════════════════════════════════════════════════

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from indicateurs import capteurs_par_cluster_mini, capteurs_par_cluster_standard, communes_mini, debits_par_cluster, tous_les_indicateurs  # noqa: E402


def _entrees() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for fichier in (FICHIER_COMMUNES_BKT_OBS, FICHIER_SENSOR_DETAIL, FICHIER_POPULATION_USED):
        if not fichier.exists():
            raise FileNotFoundError(f"{fichier} manquant - lancez d'abord les pipeline.py des étapes 04 et 05.")
    return pd.read_parquet(FICHIER_COMMUNES_BKT_OBS), pd.read_parquet(FICHIER_SENSOR_DETAIL), pd.read_parquet(FICHIER_POPULATION_USED)


def main() -> None:
    fichier_indicateurs = DOSSIER_BKT / "indicateurs.csv"
    if fichier_indicateurs.exists() and not FORCER_LE_RECALCUL:
        print(f"== indicateurs : déjà calculés ({fichier_indicateurs.name})")
    else:
        communes, sensor_detail, population_used = _entrees()
        print("== indicateurs de base (population, capteurs, réseau, débits, BKT standard/mini)")
        tous_les_indicateurs(communes, sensor_detail, population_used, ANNEES).to_csv(fichier_indicateurs, index=False)
        debits_par_cluster(sensor_detail).to_csv(DOSSIER_BKT / "debits_par_cluster.csv", index=False)
        capteurs_par_cluster_standard(sensor_detail).to_csv(DOSSIER_BKT / "capteurs_par_cluster_standard.csv", index=False)
        capteurs_par_cluster_mini(sensor_detail, ANNEES).to_csv(DOSSIER_BKT / "capteurs_par_cluster_mini.csv", index=False)
        print(f"écrit : {fichier_indicateurs} (+ debits_par_cluster, capteurs_par_cluster_standard/mini)")

    if CALCULER_INTERVALLES_CONFIANCE:
        fichier_ic = DOSSIER_BKT / "ic_standard.csv"
        if fichier_ic.exists() and not FORCER_LE_RECALCUL:
            print(f"\n== intervalles de confiance : déjà calculés ({fichier_ic.name})")
        else:
            from intervalle_confiance import ic_mini, ic_standard, test_croissance, test_croissance_par_cluster

            communes, sensor_detail, _ = _entrees()
            mini = communes_mini(communes, sensor_detail, ANNEES)
            rng = np.random.default_rng(GRAINE_BOOTSTRAP)

            print("\n== intervalles de confiance bootstrap (incertitude de NOTRE méthode, voir le README)")
            ic_standard(communes, ANNEES, rng).to_csv(fichier_ic, index=False)
            ic_mini(mini, ANNEES, rng).to_csv(DOSSIER_BKT / "ic_mini.csv", index=False)
            test_croissance(communes, ANNEES, rng).to_csv(DOSSIER_BKT / "test_croissance.csv", index=False)
            test_croissance_par_cluster(communes, ANNEES, rng).to_csv(DOSSIER_BKT / "test_croissance_par_cluster.csv", index=False)
            print(f"écrit : {fichier_ic} (+ ic_mini, test_croissance, test_croissance_par_cluster)")

    if CONSTRUIRE_CLASSEUR_EXCEL:
        fichier_xlsx = DOSSIER_BKT / "bkt_final.xlsx"
        if fichier_xlsx.exists() and not FORCER_LE_RECALCUL:
            print(f"\n== classeur Excel : déjà construit ({fichier_xlsx.name})")
        else:
            from export_excel import construire_classeur
            from indicateurs import CLUSTER_NAMES

            print("\n== construction du classeur Excel final")
            construire_classeur(str(fichier_xlsx), DOSSIER_BKT, CLUSTER_NAMES)
            print(f"écrit : {fichier_xlsx}")

    if CONSTRUIRE_CARTES:
        dossier_cartes = DOSSIER_BKT / "cartes"
        fichier_couverture = dossier_cartes / f"carte_couverture_communes_{ANNEE_CARTES}.html"
        if fichier_couverture.exists() and not FORCER_LE_RECALCUL:
            print(f"\n== cartes : déjà construites ({dossier_cartes})")
        else:
            from cartes import carte_clusters_communes, carte_couverture_communes

            if not FICHIER_COMMUNES_ZONES.exists():
                raise FileNotFoundError(f"{FICHIER_COMMUNES_ZONES} manquant - lancez d'abord 00_transformation_des_donnees/pipeline.py.")
            fichier_communes_assemblees = DOSSIER_BKT / "communes.parquet"  # étape 04 (brute, pas l'enrichie bkt_obs de l'étape 05)
            if not fichier_communes_assemblees.exists():
                raise FileNotFoundError(f"{fichier_communes_assemblees} manquant - lancez d'abord 04_jointure_des_donnees/pipeline.py.")
            print(f"\n== cartes de communes ({ANNEE_CARTES})")
            dossier_cartes.mkdir(parents=True, exist_ok=True)
            carte_couverture_communes(FICHIER_COMMUNES_ZONES, fichier_communes_assemblees, annee=ANNEE_CARTES, html=True, sortie=fichier_couverture)
            carte_clusters_communes(FICHIER_COMMUNES_ZONES, fichier_communes_assemblees, annee=ANNEE_CARTES, html=True, sortie=dossier_cartes / f"carte_clusters_communes_{ANNEE_CARTES}.html")


if __name__ == "__main__":
    main()
    print("\nétape 99 (indicateurs + intervalles de confiance + export Excel) : ok.")
