# -*- coding: utf-8 -*-
"""Indicateurs de base : population, capteurs, débits, réseau, et BKT « mini ».

En plus des chiffres bruts (population, nombre de capteurs actifs, linéaire de réseau), calcule un BKT
« mini » : recalculé avec EXACTEMENT la même méthode qu'à l'étape 06 (avec n, Z complet, rognage
LOOCV), mais en ne gardant QUE les capteurs présents TOUTES les années de la série SANS INTERRUPTION
(un « panel équilibré »). Ça exclut tout capteur installé ou retiré en cours de route.

Pourquoi c'est utile : la méthode standard voit son échantillon de capteurs grossir chaque année -
une partie de la croissance mesurée du BKT pourrait donc venir de "on regarde plus d'endroits", pas de
"on roule plus à vélo". Le BKT mini répond à ça : en gardant un groupe FIXE de capteurs d'un bout à
l'autre de la période, toute variation qu'on y observe vient forcément du comportement, pas de
l'échantillon. Comparer BKT standard et BKT mini est donc un test de robustesse de la croissance mesurée.

**Ces indicateurs décrivent NOTRE méthode d'estimation du BKT, pas le BKT réel national (que personne
ne peut observer directement) - voir le README de ce dossier.**"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLUSTER_NAMES: dict[int, str] = {0: "Péri-urbain loisirs", 1: "Domicile travail", 2: "Tourisme aménagé", 3: "Tourisme rural"}
N_CLUSTERS = len(CLUSTER_NAMES)
# Même rognage qu'à l'étape 06 (voir 06_calcul_du_bkt_extrapole/bkt_extrapole.py pour la justification) -
# dupliqué ici (petite constante) plutôt qu'importé, pour que ce module reste lisible seul.
LOOCV_OPTIMAL_TRIM: dict[int, float] = {0: 0.34, 1: 0.46, 2: 0.48, 3: 0.40}


def taux_rogne(taux: np.ndarray, rognage: float, n_min: int = 5) -> float:
    """Moyenne rognée - voir 06_calcul_du_bkt_extrapole/bkt_extrapole.py pour le détail (même formule)."""
    if len(taux) < n_min:
        return float(np.median(taux)) if len(taux) else 0.0
    bas, haut = np.quantile(taux, [rognage, 1 - rognage])
    gardes = taux[(taux >= bas) & (taux <= haut)]
    return float(gardes.mean()) if len(gardes) else 0.0


def panel_equilibre(sensor_detail: pd.DataFrame, annees: tuple[int, ...]) -> pd.Index:
    """Les identifiants de capteurs présents TOUTES les années de 'annees' sans interruption, parmi
    tous les capteurs actifs jamais vus (généralement une minorité - la plupart sont installés en cours
    de route et n'ont donc pas de série complète)."""
    annees_par_site = sensor_detail.groupby("id_site")["annee"].nunique()
    return annees_par_site[annees_par_site == len(annees)].index


def communes_mini(communes: pd.DataFrame, sensor_detail: pd.DataFrame, annees: tuple[int, ...]) -> pd.DataFrame:
    """Reconstruit la table (année, commune, cluster, réseau, bkt_obs_mini) EXACTEMENT comme l'étape 05
    construit 'communes_bkt_obs.parquet' (une ligne par (année, commune, cluster)), mais en ne
    moyennant la QTA que sur les capteurs du panel équilibré - le linéaire déjà réparti par cluster
    ('lineaire_km') est repris tel quel, seule la moyenne de débit change. Colonne 'bkt_obs_mini' en
    miroir de 'bkt_obs'."""
    panel = panel_equilibre(sensor_detail, annees)
    s_mini = sensor_detail[sensor_detail["id_site"].isin(panel)].dropna(subset=["cluster_K4"]).copy()
    s_mini["cluster_K4"] = s_mini["cluster_K4"].astype(int)

    base = communes[["annee", "code_commune", "cluster_K4", "lineaire_km"]]
    agg = (
        s_mini.groupby(["annee", "id_commune_str", "cluster_K4"])["qta"]
        .mean()
        .rename("qta_moyen_mini")
        .reset_index()
        .rename(columns={"id_commune_str": "code_commune"})
    )
    fusion = base.merge(agg, on=["annee", "code_commune", "cluster_K4"], how="left")
    fusion["qta_moyen_mini"] = fusion["qta_moyen_mini"].fillna(0.0)
    fusion["bkt_obs_mini"] = fusion["lineaire_km"] * fusion["qta_moyen_mini"]
    return fusion


def bkt_ext_cluster(cadre: pd.DataFrame, colonne_bkt: str, annee: int) -> float:
    """La même formule qu'à l'étape 06 (Z complet + rognage LOOCV par cluster), générique sur le nom de
    la colonne BKT_obs - réutilisée pour le calcul standard ET le calcul mini, sans dupliquer la
    logique. 'cadre' a une ligne par (année, commune, cluster) - toujours 'lineaire_km' (déjà réparti),
    jamais 'total_length_km' (compterait plusieurs fois le réseau d'une commune à plusieurs clusters)."""
    total = 0.0
    for cl in range(N_CLUSTERS):
        c = cadre[(cadre["annee"] == annee) & (cadre["cluster_K4"] == cl)]
        l_total = c["lineaire_km"].sum()
        actif = c[(c["lineaire_km"] > 0) & (c[colonne_bkt] > 0)]
        taux_individuels = (actif[colonne_bkt] / actif["lineaire_km"]).to_numpy()
        taux = taux_rogne(taux_individuels, LOOCV_OPTIMAL_TRIM[cl])
        l_obs, bkt_obs_cl = actif["lineaire_km"].sum(), actif[colonne_bkt].sum()
        total += bkt_obs_cl + taux * max(l_total - l_obs, 0.0)
    return total


def tous_les_indicateurs(communes: pd.DataFrame, sensor_detail: pd.DataFrame, population_used: pd.DataFrame, annees: tuple[int, ...]) -> pd.DataFrame:
    """La table "tout au même niveau" : une ligne par année, une colonne par indicateur (population,
    capteurs standard/panel équilibré, réseau, débits, BKT standard et mini, observé et extrapolé)."""
    panel = panel_equilibre(sensor_detail, annees)
    mini = communes_mini(communes, sensor_detail, annees)

    lignes = []
    for annee in annees:
        c, cm = communes[communes["annee"] == annee], mini[mini["annee"] == annee]
        s = sensor_detail[sensor_detail["annee"] == annee]
        actives = c[(c["instrumentee"] == 1) & (c["lineaire_km"] > 0) & (c["bkt_obs"] > 0)]
        lignes.append(
            dict(
                annee=annee,
                population=population_used.loc[population_used["annee"] == annee, "population_totale"].iloc[0],
                n_capteurs_actifs=s["id_site"].nunique(),
                n_capteurs_panel=len(panel),
                lineaire_fob_km=c["lineaire_km"].sum(),  # 'lineaire_km' partitionne exactement le réseau - jamais 'total_length_km', répété sur chaque cluster d'une commune multi-cluster
                somme_debits=s["annual_flow"].sum(),
                bkt_obs_Mdkm=actives["bkt_obs"].sum() / 1e9,
                bkt_ext_Mdkm=bkt_ext_cluster(c, "bkt_obs", annee) / 1e9,
                bkt_mini_obs_Mdkm=cm["bkt_obs_mini"].sum() / 1e9,
                bkt_mini_ext_Mdkm=bkt_ext_cluster(cm, "bkt_obs_mini", annee) / 1e9,
            )
        )
    return pd.DataFrame(lignes)


def _pivot_par_cluster(table: pd.DataFrame, colonne_valeur: str, agg: str) -> pd.DataFrame:
    """Pivote (année x cluster) -> une colonne par cluster, en garantissant les N_CLUSTERS colonnes
    même si un cluster n'a aucune ligne pour une année donnée."""
    piv = table.groupby(["annee", "cluster_K4"])[colonne_valeur].agg(agg).unstack("cluster_K4")
    piv = piv.reindex(columns=range(N_CLUSTERS), fill_value=0)
    return piv.rename(columns=CLUSTER_NAMES)[[CLUSTER_NAMES[i] for i in range(N_CLUSTERS)]].reset_index()


def debits_par_cluster(sensor_detail: pd.DataFrame) -> pd.DataFrame:
    """Somme des débits bruts (passages/an, 'annual_flow'), par cluster et par année - sans aucune
    correction (ni extrapolation aux heures manquantes comme 'qta', ni linéaire) : juste ce que les
    compteurs ont physiquement enregistré."""
    return _pivot_par_cluster(sensor_detail, "annual_flow", "sum")


def capteurs_par_cluster_standard(sensor_detail: pd.DataFrame) -> pd.DataFrame:
    return _pivot_par_cluster(sensor_detail, "id_site", "nunique")


def capteurs_par_cluster_mini(sensor_detail: pd.DataFrame, annees: tuple[int, ...]) -> pd.DataFrame:
    """Par construction, le panel équilibré est LE MÊME chaque année (même liste de capteurs) - ce
    tableau est donc constant ligne par ligne ; calculé année par année quand même pour rester
    directement comparable au tableau standard, colonne par colonne."""
    s_mini = sensor_detail[sensor_detail["id_site"].isin(panel_equilibre(sensor_detail, annees))]
    return _pivot_par_cluster(s_mini, "id_site", "nunique")
