# -*- coding: utf-8 -*-
"""Assemble la table dont tout le reste du pipeline (étapes 05, 06, 99) a besoin, à partir des sorties
déjà produites par les étapes amont (00 à 03) - aucun calcul de source brute ici, seulement des
jointures et des agrégations.

    communes.parquet         une ligne par (année, commune) : réseau (3 scénarios), population, cluster
                              d'usage retenu (voir 'entrees.resoudre_cluster_communes'), statut
                              "instrumentée"
    population_used.parquet  quel millésime de population a été utilisé pour chaque année, et pourquoi
                              (voir 'entrees.millesime_population')

('sensor_detail.parquet', le débit et le QTA de chaque capteur actif, est la sortie directe de
'entrees.capteurs_actifs' - pas de traitement supplémentaire ici, voir 'pipeline.py'.)"""

from __future__ import annotations

import pandas as pd

from entrees import millesime_population


def base_par_commune(communes: pd.DataFrame, longueurs: pd.DataFrame, annee: int, scenario: str, population: pd.Series) -> pd.DataFrame:
    """Une ligne par commune, avec la longueur de réseau de 'scenario' et la population de 'annee'."""
    ln = longueurs[(longueurs["annee"] == annee) & (longueurs["scenario"] == scenario)][["code_commune", "len_d", "len_g"]]
    base = communes.merge(ln, on="code_commune", how="left")
    base[["len_d", "len_g"]] = base[["len_d", "len_g"]].fillna(0.0)
    base["total_length_km"] = base["len_d"] + base["len_g"]
    base["population"] = base["code_commune"].map(population).fillna(0.0)
    return base


def table_population_utilisee(annees: tuple[int, ...], charger_un_millesime) -> tuple[dict[int, pd.Series], pd.DataFrame]:
    """(populations par millésime chargé, table de contrôle 'quel millésime pour quelle année et
    pourquoi') - 'charger_un_millesime(millesime) -> pd.Series' vient de '00_transformation_des_donnees'."""
    populations: dict[int, pd.Series] = {}
    lignes = []
    for annee in annees:
        millesime, regle = millesime_population(annee)
        if millesime not in populations:
            populations[millesime] = charger_un_millesime(millesime)
        lignes.append(dict(annee=annee, millesime_utilise=millesime, regle=regle, population_totale=float(populations[millesime].sum()), n_communes=len(populations[millesime])))
    return populations, pd.DataFrame(lignes)


def assembler(communes: pd.DataFrame, longueurs: pd.DataFrame, instrumentees: set[str], populations: dict[int, pd.Series], cluster_par_annee: pd.DataFrame, annees: tuple[int, ...], scenarios: tuple[str, ...]) -> pd.DataFrame:
    """Boucle sur les années et les scénarios réseau -> communes.parquet (une ligne par année x commune
    ayant du réseau dans au moins un scénario). 'cluster_par_annee' (voir
    'entrees.resoudre_cluster_communes') donne le cluster K4 déjà résolu pour chaque (année, commune) -
    celui de ses capteurs actifs quand elle en a, sinon le cluster prédit à l'étape 02."""
    lignes_communes = []
    for annee in annees:
        millesime, _ = millesime_population(annee)
        population = populations[millesime]
        bases = {scenario: base_par_commune(communes, longueurs, annee, scenario, population) for scenario in scenarios}

        par_commune = communes[["code_commune"]].merge(
            cluster_par_annee[cluster_par_annee["annee"] == annee][["code_commune", "cluster_K4"]],
            on="code_commune",
            how="left",
        )
        par_commune["population"] = par_commune["code_commune"].map(population).fillna(0.0)
        for scenario in scenarios:
            par_commune[f"len_d_{scenario}"] = bases[scenario]["len_d"].to_numpy()
            par_commune[f"len_g_{scenario}"] = bases[scenario]["len_g"].to_numpy()
        colonnes_longueur = [f"len_{cote}_{scenario}" for cote in ("d", "g") for scenario in scenarios]
        a_du_reseau = par_commune[colonnes_longueur].sum(axis=1) > 0
        par_commune = par_commune[a_du_reseau].copy()
        par_commune.insert(0, "annee", annee)
        par_commune["instrumentee"] = par_commune["code_commune"].isin(instrumentees).astype(int)
        lignes_communes.append(par_commune.merge(communes[["code_commune", "nom_commune", "code_departement"]], on="code_commune", how="left"))

    return pd.concat(lignes_communes, ignore_index=True)
