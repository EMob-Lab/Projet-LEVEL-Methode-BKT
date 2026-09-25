# -*- coding: utf-8 -*-
"""Assemble les entrées de la jointure depuis les étapes amont : capteurs (étape 00 + 01), communes
(étape 00 + 02), population (étape 00), réseau cyclable (étape 03 - voir 'longueurs.py').

Cette étape ne lit jamais une source brute et n'ajuste aucun modèle ; elle ne fait que COMBINER les
sorties déjà produites - une ligne par (capteur, année) et par (commune, année), rien de plus."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

POPULATION_VINTAGES: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023)
"""Millésimes de population qui existent réellement dans data/donnees_valides/zones/population.parquet
(étape 00) - 2024 et 2025 réutilisent le dernier disponible (délai de publication INSEE)."""

# Paris, Lyon et Marseille existent à la fois comme une seule commune (dans certaines sources, dont les
# codes commune parfois utilisés par les capteurs) et comme des arrondissements (dans le découpage
# communal utilisé partout ailleurs dans ce dépôt, voir '00_transformation_des_donnees'). ville ->
# (codes des arrondissements, codes sous lesquels la ville entière peut apparaître).
ARRONDISSEMENTS_PLM: dict[str, tuple[set[str], set[str]]] = {
    "paris": ({f"751{i:02d}" for i in range(1, 21)}, {"75056", "75100", "00751"}),
    "lyon": ({f"6938{i}" for i in range(1, 10)}, {"69123", "69380"}),
    "marseille": ({f"132{i:02d}" for i in range(1, 17)}, {"13055", "13200", "01305"}),
}


def etendre_arrondissements_plm(codes: set[str]) -> set[str]:
    """Ajoute tous les arrondissements d'une ville dès que la ville, ou l'un de ses arrondissements, est
    présent - pour qu'un capteur signalé sous le code de la ville entière (ou un seul arrondissement)
    marque bien TOUS ses arrondissements comme instrumentés (voir 'communes_instrumentees')."""
    etendus = set(codes)
    for arrondissements, parents in ARRONDISSEMENTS_PLM.values():
        if etendus & (arrondissements | parents):
            etendus.update(arrondissements)
    return etendus


def millesime_population(annee: int) -> tuple[int, str]:
    """(millésime à utiliser, règle) pour une année du BKT : celui de l'année elle-même si publié,
    sinon le plus proche PUBLIÉ APRÈS, sinon le plus proche avant (aucun de plus récent)."""
    if annee in POPULATION_VINTAGES:
        return annee, "année"
    apres = [v for v in POPULATION_VINTAGES if v > annee]
    if apres:
        return min(apres), "plus proche après"
    return max(v for v in POPULATION_VINTAGES if v < annee), "plus proche avant (aucun millésime plus récent n'existe)"


def capteurs_actifs(fichier_sensor_years: Path, fichier_cluster_assignments: Path) -> pd.DataFrame:
    """Une ligne par (capteur, année) ACTIF (voir '00_transformation_des_donnees', colonne 'active' :
    filtre funFEM, plus de 5 % des heures non nulles et débit total > 100), avec son cluster d'usage
    K4 (voir '01_clustering_des_usages', 'reference.py' - clusters de référence, pas régénérés ici).

    Les capteurs actifs sans cluster K4 (id_site absent de la référence) sont gardés avec
    'cluster_K4' = NaN - c'est à l'appelant de décider s'il les exclut (voir 'assemblage.py')."""
    sensor_years = pd.read_parquet(fichier_sensor_years)
    actifs = sensor_years[sensor_years["active"]].copy()
    clusters = pd.read_parquet(fichier_cluster_assignments, columns=["id_site", "cluster"]).set_index("id_site")["cluster"]
    actifs["cluster_K4"] = actifs["id_site"].map(clusters)
    print(f"[entrees] {len(actifs):,} capteurs-années actifs ; {actifs['cluster_K4'].isna().sum():,} sans cluster K4 (id_site absent de la référence 01)")
    return actifs.drop(columns=["activity_share", "total_flow", "valid", "active"]).reset_index(drop=True)


def table_communes(fichier_communes_zones: Path, fichier_commune_clusters: Path) -> pd.DataFrame:
    """Une ligne par commune (métropole hors Corse, voir étape 00) : nom, département et cluster K4
    prédit (étape 02 - prédictions déjà calculées, pas régénérées ici). Ce cluster ne sert que de
    repli pour les communes sans capteur actif - voir 'resoudre_cluster_communes' pour le cluster
    effectivement retenu, année par année."""
    base = pd.read_parquet(fichier_communes_zones, columns=["code_commune", "nom_commune", "code_departement"])
    base["code_commune"] = base["code_commune"].astype(str).str.zfill(5)
    predites = pd.read_parquet(fichier_commune_clusters, columns=["code_commune", "cluster_predit"])
    return base.merge(predites.rename(columns={"cluster_predit": "cluster_K4"}), on="code_commune", how="left")


def resoudre_cluster_communes(capteurs: pd.DataFrame, communes: pd.DataFrame, annees: tuple[int, ...]) -> pd.DataFrame:
    """Cluster K4 retenu pour chaque (année, commune) : celui de ses capteurs actifs quand elle en a -
    le cluster au débit total ('qta') le plus fort parmi eux (un seul capteur actif = son propre
    cluster, cas particulier de la même règle) ; sinon le cluster prédit à l'étape 02 (voir
    'table_communes'), seule source possible pour une commune sans capteur."""
    votants = capteurs.dropna(subset=["cluster_K4"])
    debit_par_cluster = votants.groupby(["annee", "id_commune_str", "cluster_K4"], as_index=False)["qta"].sum()
    vote = (
        debit_par_cluster.sort_values("qta", ascending=False)
        .drop_duplicates(["annee", "id_commune_str"], keep="first")
        .rename(columns={"id_commune_str": "code_commune", "cluster_K4": "cluster_vote"})[
            ["annee", "code_commune", "cluster_vote"]
        ]
    )

    base = pd.MultiIndex.from_product([annees, communes["code_commune"]], names=["annee", "code_commune"]).to_frame(index=False)
    base = base.merge(
        communes[["code_commune", "cluster_K4"]].rename(columns={"cluster_K4": "cluster_etape02"}),
        on="code_commune",
        how="left",
    )
    resultat = base.merge(vote, on=["annee", "code_commune"], how="left")
    resultat["cluster_K4"] = resultat["cluster_vote"].fillna(resultat["cluster_etape02"])
    return resultat[["annee", "code_commune", "cluster_K4"]]


def communes_instrumentees(fichier_sensor_years: Path) -> set[str]:
    """Communes hébergeant un capteur ACTIF une année quelconque, arrondissements de
    Paris/Lyon/Marseille étendus (voir 'etendre_arrondissements_plm') : un signal réel dans le 5ème
    arrondissement de Paris ne doit pas laisser un autre arrondissement à tort hors de l'univers
    instrumenté à cause d'une différence de convention de code commune entre les sources.

    PAS de réintégration des capteurs "orphelins" (valides mais pas actifs) ici, volontairement : la
    méthode retenue (voir '06_calcul_du_bkt_extrapole') ne teste JAMAIS 'instrumentee' seule - toujours
    en ET avec 'bkt_obs > 0', qui n'est calculé qu'à partir des capteurs ACTIFS (étape 05). Un
    orphelin ne contribue donc jamais à 'bkt_obs' : l'ajouter à l'univers instrumenté ne change
    JAMAIS quelle commune compte comme "observée" - vérifié : sur les communes que la réintégration des
    orphelins ajoutait, aucune n'a jamais 'bkt_obs > 0', aucune année. Ce n'était utile qu'à la
    méthode historique (retirée de ce dépôt, voir son ancien calcul par capteur)."""
    sensor_years = pd.read_parquet(fichier_sensor_years, columns=["id_commune_str", "active"])
    actifs = set(sensor_years.loc[sensor_years["active"], "id_commune_str"].dropna())
    return etendre_arrondissements_plm(actifs)
