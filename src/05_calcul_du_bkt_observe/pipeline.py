"""Étape 05 — calcul du BKT observé : à partir des trois tables assemblées par l'étape 04
(data/donnees_valides/bkt/), calcule le BKT observé de chaque (année, commune, cluster) - voir
'bkt_observe.py' pour la formule.

PREMIÈRE ÉTAPE QUI NE LIT PLUS que 'data/donnees_valides/bkt/' (jamais les dossiers des étapes 00-03
directement).

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

FICHIER_COMMUNES = DOSSIER_BKT / "communes.parquet"  # étape 04
FICHIER_SENSOR_DETAIL = DOSSIER_BKT / "sensor_detail.parquet"  # étape 04

FORCER_LE_RECALCUL = False

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# à partir d'ici, plus aucune configuration - seulement l'enchaînement des étapes
# ═══════════════════════════════════════════════════════════════════════════════════════════════

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

import pandas as pd  # noqa: E402
from bkt_observe import bkt_observe_commune  # noqa: E402


def main() -> None:
    fichier_sortie = DOSSIER_BKT / "communes_bkt_obs.parquet"
    if fichier_sortie.exists() and not FORCER_LE_RECALCUL:
        print(f"== calcul du BKT observé : déjà calculé ({fichier_sortie.name})")
        return

    for fichier in (FICHIER_COMMUNES, FICHIER_SENSOR_DETAIL):
        if not fichier.exists():
            raise FileNotFoundError(
                f"{fichier} manquant - lancez d'abord 04_jointure_des_donnees/pipeline.py."
            )
    communes = pd.read_parquet(FICHIER_COMMUNES)
    sensor_detail = pd.read_parquet(FICHIER_SENSOR_DETAIL)

    print("== calcul du BKT observé")
    resultat = bkt_observe_commune(communes, sensor_detail)
    resultat.to_parquet(fichier_sortie, index=False)
    national = resultat.groupby("annee")["bkt_obs"].sum() / 1e9
    print(f"écrit : {fichier_sortie}")
    print("\nBKT observé national, par année (Md km) :")
    print(national.round(3).to_string())


if __name__ == "__main__":
    main()
    print("\nétape 05 (calcul du BKT observé) : ok.")
