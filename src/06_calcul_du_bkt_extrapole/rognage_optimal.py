# -*- coding: utf-8 -*-
"""Choix du rognage utilisé par 'bkt_extrapole.taux_rogne' (constante 'LOOCV_OPTIMAL_TRIM') :
validation croisée LEAVE-ONE-OUT sur une grille de rognages candidats, par cluster.

Pour chaque (année, commune, cluster) actif d'un cluster, on cache son taux et on le prédit à partir
de la moyenne rognée de tous les AUTRES - même formule que 'bkt_extrapole.taux_rogne', appliquée une
fois par ligne cachée - puis on mesure l'erreur médiane absolue. Le rognage retenu pour ce cluster est
celui qui minimise cette erreur sur la grille.

    python rognage_optimal.py

Ce script VÉRIFIE / RE-DÉRIVE le choix déjà câblé dans 'bkt_extrapole.LOOCV_OPTIMAL_TRIM' - il ne le
met pas à jour automatiquement (pas de risque qu'un run accidentel change silencieusement un nombre
utilisé par tout le reste du pipeline). Si le résultat diffère de la constante, mettez-la à jour à la
main dans 'bkt_extrapole.py' et relancez les étapes 06 et 99."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

GRILLE_ROGNAGE = np.round(np.arange(0.00, 0.49, 0.02), 2)  # jusqu'à 0.48 : au-delà, trop peu de points gardés

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
from bkt_extrapole import CLUSTER_NAMES, LOOCV_OPTIMAL_TRIM  # noqa: E402


def erreur_loocv(taux: np.ndarray, rognage: float, n_min: int = 5) -> float:
    """Erreur médiane absolue (leave-one-out) de prédire le taux de CHAQUE ligne à partir de la moyenne
    rognée de toutes les AUTRES lignes du même cluster - même formule que 'bkt_extrapole.taux_rogne',
    appliquée n fois en cachant une ligne différente à chaque fois."""
    n = len(taux)
    erreurs = np.empty(n)
    for i in range(n):
        autres = np.delete(taux, i)
        if len(autres) >= n_min:
            bas, haut = np.quantile(autres, [rognage, 1 - rognage])
        else:
            bas, haut = autres.min(), autres.max()
        gardes = autres[(autres >= bas) & (autres <= haut)]
        prediction = gardes.mean() if len(gardes) else autres.mean()
        erreurs[i] = abs(prediction - taux[i])
    return float(np.median(erreurs))


def rognage_optimal_par_cluster(
    communes_bkt_obs: pd.DataFrame, cluster_names: dict[int, str], grille: np.ndarray = GRILLE_ROGNAGE
) -> pd.DataFrame:
    """Une ligne par cluster : erreur LOOCV pour chaque rognage de la grille, et le rognage optimal.
    Calculé sur toutes les années poolées, sur les mêmes (commune, cluster) actifs que
    'bkt_extrapole.cadre_cluster' (Z complet : instrumentée, linéaire > 0, BKT observé > 0)."""
    actives = communes_bkt_obs[
        (communes_bkt_obs["instrumentee"] == 1)
        & (communes_bkt_obs["lineaire_km"] > 0)
        & (communes_bkt_obs["bkt_obs"] > 0)
    ].copy()
    actives["taux"] = actives["bkt_obs"] / actives["lineaire_km"]

    lignes = []
    for cl, nom in cluster_names.items():
        taux = actives.loc[actives["cluster_K4"] == cl, "taux"].to_numpy()
        erreurs = [erreur_loocv(taux, r) for r in grille]
        meilleur = int(np.argmin(erreurs))
        ligne = {"cluster": cl, "cluster_nom": nom, "n": len(taux)}
        ligne.update({f"erreur_{r:.0%}": round(e, 0) for r, e in zip(grille, erreurs, strict=True)})
        ligne["rognage_optimal"] = grille[meilleur]
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def main() -> None:
    racine = Path(__file__).resolve().parents[2]
    fichier = racine / "data" / "donnees_valides" / "bkt" / "communes_bkt_obs.parquet"
    if not fichier.exists():
        raise FileNotFoundError(f"{fichier} manquant - lancez d'abord 05_calcul_du_bkt_observe/pipeline.py.")
    communes_bkt_obs = pd.read_parquet(fichier)

    resultat = rognage_optimal_par_cluster(communes_bkt_obs, CLUSTER_NAMES)
    print("Rognage optimal par cluster (validation croisée leave-one-out, années poolées) :\n")
    print(resultat[["cluster_nom", "n", "rognage_optimal"]].to_string(index=False))

    print("\nComparaison avec bkt_extrapole.LOOCV_OPTIMAL_TRIM :")
    for cl, nom in CLUSTER_NAMES.items():
        actuel = LOOCV_OPTIMAL_TRIM[cl]
        recalcule = float(resultat.loc[resultat["cluster"] == cl, "rognage_optimal"].iloc[0])
        marqueur = "" if abs(actuel - recalcule) < 1e-9 else "  <-- DIFFÉRENT"
        print(f"  {nom:22s} : actuel {actuel:.0%}, recalculé {recalcule:.0%}{marqueur}")

    sortie = racine / "data" / "donnees_valides" / "bkt" / "rognage_optimal_loocv.csv"
    resultat.to_csv(sortie, index=False)
    print(f"\nécrit : {sortie}")


if __name__ == "__main__":
    main()
