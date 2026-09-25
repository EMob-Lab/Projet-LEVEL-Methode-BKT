"""BKT observé : combien de kilomètres à vélo les capteurs laissent directement mesurer, commune par
commune ET cluster par cluster - avant extrapolation aux communes sans capteur (étape 06).

Un capteur a son propre cluster d'usage (étape 01, funFEM sur son profil horaire/hebdomadaire/
saisonnier) - une commune peut donc héberger des capteurs de plusieurs clusters à la fois. Le linéaire
de la commune est réparti entre ces clusters au prorata du débit total de chacun :

    débit_total(commune, cluster) = somme des QTA des capteurs DE CE CLUSTER dans la commune
    part(commune, cluster)        = débit_total(commune, cluster) / débit_total(commune)
    linéaire(commune, cluster)    = total_length_km(commune) x part(commune, cluster)

    BKT_obs(commune, cluster) = moyenne( QTA(capteur) ),  capteurs DE CE CLUSTER dans la commune
                                 x linéaire(commune, cluster)

Chaque capteur d'un cluster donné compte pour UN dans sa moyenne, quel que soit son débit propre -
plutôt qu'une répartition du linéaire du cluster au prorata du débit de chaque capteur, qui pèserait
mécaniquement deux fois le capteur le plus fréquenté (une fois via la part de réseau qu'on lui
attribue, une fois via son propre débit déjà élevé). Une commune à un seul cluster de capteurs (la
grande majorité des communes instrumentées) reçoit 100% de son linéaire sur ce cluster - résultat
identique à un calcul commune par commune. Une commune sans capteur actif cette année-là n'a rien à
répartir : tout son linéaire reste sur le cluster déjà résolu à l'étape 04 (classification de l'étape
02, seule source possible en l'absence de capteur).

'QTA' ("Quantité de Trafic Annuel", voir '00_transformation_des_donnees/d_debit/capteurs_annees.py')
est déjà l'estimation annuelle correcte du nombre de passages d'un capteur - aucune correction n'est
appliquée ici, le calcul est fait une fois pour toutes à la source."""

from __future__ import annotations

import pandas as pd

SCENARIO = "FOB"  # scénario réseau retenu partout dans ce dépôt (voir 03_creation_du_jeu_de_donnees_cyclable)

COLONNES_METADONNEES = ["annee", "code_commune", "nom_commune", "code_departement", "instrumentee"]


def bkt_observe_commune(
    communes: pd.DataFrame, sensor_detail: pd.DataFrame, *, scenario: str = SCENARIO
) -> pd.DataFrame:
    """Une ligne par (année, commune, cluster) ayant du linéaire attribué - plusieurs lignes pour une
    commune dont les capteurs actifs appartiennent à plusieurs clusters, une seule sinon. Table PIVOT
    que les étapes suivantes (06, 99) réutilisent."""
    # ---- communes : le réseau du bon scénario (les 2 côtés de la voirie, droit + gauche)
    base = communes.rename(columns={f"len_d_{scenario}": "len_d", f"len_g_{scenario}": "len_g"}).copy()
    base["total_length_km"] = base["len_d"] + base["len_g"]

    # ---- débit moyen et nombre de capteurs, par (année, commune, cluster DU CAPTEUR - étape 01)
    votants = sensor_detail.dropna(subset=["cluster_K4"]).copy()
    votants["cluster_K4"] = votants["cluster_K4"].astype(int)
    par_cluster = (
        votants.groupby(["annee", "id_commune_str", "cluster_K4"])
        .agg(qta_moyen=("qta", "mean"), n_capteurs_actifs=("id_site", "nunique"))
        .reset_index()
        .rename(columns={"id_commune_str": "code_commune"})
    )

    # ---- part du linéaire de la commune attribuée à chaque cluster, au prorata du débit TOTAL du cluster
    par_cluster["debit_total_cluster"] = par_cluster["qta_moyen"] * par_cluster["n_capteurs_actifs"]
    debit_total_commune = par_cluster.groupby(["annee", "code_commune"])["debit_total_cluster"].transform("sum")
    par_cluster["part_lineaire"] = par_cluster["debit_total_cluster"] / debit_total_commune
    par_cluster = par_cluster.drop(columns="debit_total_cluster")

    par_cluster = par_cluster.merge(
        base[COLONNES_METADONNEES + ["total_length_km"]], on=["annee", "code_commune"], how="left"
    )
    par_cluster["lineaire_km"] = par_cluster["total_length_km"] * par_cluster["part_lineaire"]
    par_cluster["bkt_obs"] = par_cluster["qta_moyen"] * par_cluster["lineaire_km"]

    # ---- communes (année, commune) SANS capteur actif cette année-là (donc absentes de 'par_cluster') :
    # une seule ligne, tout le linéaire sur le cluster déjà résolu à l'étape 04 (repli étape 02)
    split_eligible = par_cluster[["annee", "code_commune"]].drop_duplicates()
    split_eligible["_eligible"] = True
    hors_split = base.merge(split_eligible, on=["annee", "code_commune"], how="left")
    hors_split = hors_split[hors_split["_eligible"].isna()].copy()
    hors_split["part_lineaire"] = 1.0
    hors_split["lineaire_km"] = hors_split["total_length_km"]
    hors_split["qta_moyen"] = 0.0
    hors_split["n_capteurs_actifs"] = 0
    hors_split["bkt_obs"] = 0.0
    hors_split = hors_split[COLONNES_METADONNEES + ["cluster_K4", "total_length_km", "part_lineaire", "lineaire_km", "n_capteurs_actifs", "qta_moyen", "bkt_obs"]]

    colonnes = COLONNES_METADONNEES + ["cluster_K4", "total_length_km", "part_lineaire", "lineaire_km", "n_capteurs_actifs", "qta_moyen", "bkt_obs"]
    return pd.concat([par_cluster[colonnes], hors_split[colonnes]], ignore_index=True)
