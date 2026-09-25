"""Étape 06 — calcul du BKT extrapolé : à partir du BKT observé (étape 05), estime le BKT NATIONAL en
extrapolant aux communes sans capteur - voir 'bkt_extrapole.py' pour la formule (« rognage Z »).

C'est ICI que le chiffre final du BKT est produit (colonne 'bkt_ext_Mdkm' de 'bkt_final_national.csv').

    python pipeline.py
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

ANNEES: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024, 2025)
FORCER_LE_RECALCUL = False

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# à partir d'ici, plus aucune configuration - seulement l'enchaînement des étapes
# ═══════════════════════════════════════════════════════════════════════════════════════════════

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

import pandas as pd  # noqa: E402
from bkt_extrapole import table_annee_cluster, table_nationale  # noqa: E402


def main() -> None:
    fichier_national = DOSSIER_BKT / "bkt_final_national.csv"
    if fichier_national.exists() and not FORCER_LE_RECALCUL:
        print(f"== calcul du BKT extrapolé : déjà calculé ({fichier_national.name})")
        return

    if not FICHIER_COMMUNES_BKT_OBS.exists():
        raise FileNotFoundError(
            f"{FICHIER_COMMUNES_BKT_OBS} manquant - lancez d'abord 05_calcul_du_bkt_observe/pipeline.py."
        )
    communes = pd.read_parquet(FICHIER_COMMUNES_BKT_OBS)

    print("== calcul du BKT extrapolé")
    detail = table_annee_cluster(communes, ANNEES)
    national = table_nationale(detail)
    detail.to_csv(DOSSIER_BKT / "bkt_final_detail_par_cluster.csv", index=False)
    national.to_csv(fichier_national, index=False)
    print(
        f"écrit : {DOSSIER_BKT / 'bkt_final_detail_par_cluster.csv'}\nécrit : {fichier_national}"
    )
    print("\nBKT national (Md km) :")
    print(national.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
    print(
        "\nétape 06 (calcul du BKT extrapolé) : ok - le chiffre final du BKT est dans bkt_final_national.csv."
    )
