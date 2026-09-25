"""BKT extrapolé : estime le kilométrage cycliste des communes SANS capteur, cluster d'usage par
cluster d'usage, à partir du taux (bike-km / km de réseau) observé dans les communes qui EN ont un.

    BKT_ext(cluster) = BKT_obs(cluster) + taux_fréquentation(cluster) x L_non_obs(cluster)

où L_non_obs = L_total - L_obs (le réseau qui n'a PAS de capteur, celui qu'on doit deviner).

Deux idées composent cette formule :

1. « Z complet » (instrumentée recalculée chaque année). Marquer une commune comme "instrumentée" une
   fois pour toutes dans son histoire est trompeur : si son unique capteur tombe en panne une année
   donnée, la commune resterait comptée comme "observée" avec BKT_obs = 0 cette année-là. Ces zéros
   fantômes contamineraient le taux moyen et du coup le BKT extrapolé.
   Ici, un (commune, cluster) ne compte comme observé CETTE année-là QUE
   si il a effectivement produit du signal ('bkt_obs > 0') - son linéaire bascule sinon vers
   L_non_obs, comme n'importe quelle commune jamais observée.

2. Taux de fréquentation (rogné). Au lieu de faire un ratio de sommes ('somme(BKT_obs) / somme(réseau)' - une
   commune à 600 km de réseau pèserait alors 600x plus qu'une commune à 1 km), on calcule le taux
   (km parcourus / km de réseau) de CHAQUE (commune, cluster) séparément, on retire les valeurs extrêmes
   de chaque queue de distribution (le "rognage", un % différent par cluster - voir
   LOOCV_OPTIMAL_TRIM), puis on moyenne ce qui reste - une commune atypique ne peut plus, à elle seule,
   faire basculer le taux moyen de tout un cluster.

Le linéaire d'une commune est réparti entre les clusters de ses propres capteurs au prorata de leur
débit (voir '05_calcul_du_bkt_observe/bkt_observe.py') - une commune peut donc apparaître dans
plusieurs clusters à la fois, chacun avec sa part de linéaire et son BKT observé."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLUSTER_NAMES: dict[int, str] = {
    0: "Péri-urbain loisirs",
    1: "Domicile travail",
    2: "Tourisme aménagé",
    3: "Tourisme rural",
}
N_CLUSTERS = len(CLUSTER_NAMES)

# Rognage (part de chaque queue de distribution retirée avant de moyenner), un % par cluster - choisi
# par validation croisée leave-one-out sur les données poolées 2019-2025 (retirer une commune, prédire
# son taux à partir des autres, mesurer l'erreur - répété pour chaque % candidat, on garde celui qui
# minimise l'erreur médiane) - voir 'rognage_optimal.py' pour le script qui calcule ce choix.
LOOCV_OPTIMAL_TRIM: dict[int, float] = {0: 0.34, 1: 0.46, 2: 0.48, 3: 0.40}


def taux_rogne(taux: np.ndarray, rognage: float, n_min: int = 5) -> float:
    """Moyenne rognée : retire les 'rognage' % les plus bas ET les plus hauts, moyenne le reste. En
    dessous de 'n_min' observations, le rognage n'a plus de sens statistique (trop peu de points) - on
    retombe sur la médiane, déjà fréquentation par nature."""
    if len(taux) < n_min:
        return float(np.median(taux)) if len(taux) else 0.0
    bas, haut = np.quantile(taux, [rognage, 1 - rognage])
    gardes = taux[(taux >= bas) & (taux <= haut)]
    return float(gardes.mean()) if len(gardes) else 0.0


def cadre_cluster(
    communes: pd.DataFrame, cluster: int, annee: int
) -> tuple[pd.DataFrame, float, float, float]:
    """Isole un (cluster, année) : les (commune, cluster) ACTIFS cette année-là (Z complet, voir le
    docstring du module), leur taux individuel, et les 3 agrégats dont la formule a besoin. 'communes'
    a une ligne par (année, commune, cluster) - voir '05_calcul_du_bkt_observe/bkt_observe.py' - donc
    tous les agrégats de linéaire utilisent 'lineaire_km' (déjà réparti), jamais 'total_length_km' (le
    linéaire total de la commune, qui compterait plusieurs fois le réseau d'une commune à plusieurs
    clusters). Retourne (lignes_actives_avec_taux, L_obs, L_total, BKT_obs)."""
    c = communes[(communes["annee"] == annee) & (communes["cluster_K4"] == cluster)]
    l_total = c["lineaire_km"].sum()  # TOUT le réseau attribué au cluster, observé et non observé
    actives = c[
        (c["instrumentee"] == 1)
        & (c["lineaire_km"] > 0)
        & (c["bkt_obs"] > 0)
    ].copy()  # Z complet
    actives["taux"] = actives["bkt_obs"] / actives["lineaire_km"]
    l_obs, bkt_obs = actives["lineaire_km"].sum(), actives["bkt_obs"].sum()
    return actives, l_obs, l_total, bkt_obs


def estimer(communes: pd.DataFrame, annee: int, cluster: int) -> dict:
    """Applique la formule complète à un (cluster, année) et retourne le détail (utile pour
    l'inspection, pas seulement le résultat final)."""
    actives, l_obs, l_total, bkt_obs = cadre_cluster(communes, cluster, annee)
    taux = taux_rogne(actives["taux"].to_numpy(), LOOCV_OPTIMAL_TRIM[cluster])
    l_non_obs = max(
        l_total - l_obs, 0.0
    )  # garde-fou si jamais L_obs > L_total (ne devrait pas arriver)
    bkt_ext = bkt_obs + taux * l_non_obs
    return dict(
        n_communes_actives=len(actives),
        l_obs=l_obs,
        l_total=l_total,
        taux_fréquentation=taux,
        bkt_obs=bkt_obs,
        bkt_ext=bkt_ext,
    )


def table_annee_cluster(
    communes: pd.DataFrame, annees: tuple[int, ...]
) -> pd.DataFrame:
    """Le détail complet : une ligne par (année, cluster)."""
    lignes = [
        dict(
            annee=a,
            cluster=cl,
            cluster_nom=CLUSTER_NAMES[cl],
            **estimer(communes, a, cl),
        )
        for a in annees
        for cl in range(N_CLUSTERS)
    ]
    return pd.DataFrame(lignes)


def table_nationale(detail: pd.DataFrame) -> pd.DataFrame:
    """Somme les clusters : le BKT national, observé et extrapolé, par année."""
    return (
        detail.groupby("annee")
        .agg(bkt_obs_Mdkm=("bkt_obs", "sum"), bkt_ext_Mdkm=("bkt_ext", "sum"))
        .div(1e9)
        .reset_index()
    )
