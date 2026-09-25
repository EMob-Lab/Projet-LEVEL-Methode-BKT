# -*- coding: utf-8 -*-
"""Incertitude STATISTIQUE de notre estimation du BKT : intervalle de confiance à 95 % (bootstrap) et
test de significativité de la croissance.

**Attention à ce que ça mesure : l'incertitude d'ÉCHANTILLONNAGE de notre méthode d'extrapolation, PAS
une incertitude sur le "vrai" BKT national.** Personne n'observe le vrai BKT national - il n'existe pas
de vérité terrain à laquelle comparer notre estimation. Ce que ce module quantifie, c'est : "si on avait
eu un échantillon différent de communes actives (mais avec le même nombre), à quel point notre taux
rogné et le BKT_ext qui en découle auraient-ils pu varier ?" - une incertitude PROPRE à notre méthode
(06_calcul_du_bkt_extrapole), pas une mesure de l'écart à une réalité qu'on ne peut pas observer. Voir
le README de ce dossier.

Pourquoi seul le TAUX est rééchantillonné, pas BKT_obs : BKT_obs est une somme de mesures réelles (les
capteurs ont vraiment compté ce trafic-là) - ce n'est pas un échantillon d'une population plus large,
donc pas de raison statistique de le faire varier. Le taux robuste, lui, EST une estimation à partir
d'un échantillon (les communes actives d'un cluster) censée représenter tout le cluster - c'est la seule
partie de la formule qui a une incertitude d'échantillonnage à quantifier.

Le bootstrap, en une phrase : on tire au hasard AVEC REMISE autant de communes qu'il y en a dans
l'échantillon réel (certaines communes sont tirées plusieurs fois, d'autres pas du tout), on recalcule
le taux robuste sur ce tirage, on refait ça N_BOOT fois - la dispersion des résultats approxime
l'incertitude qu'on aurait si on avait pu observer un échantillon différent de communes.

Test de croissance : bootstrap APPARIÉ - le MÊME tirage aléatoire de communes sert à calculer le taux
des DEUX années comparées (pas deux tirages indépendants), ce qui isole l'effet de l'ANNÉE plutôt que
l'effet du hasard d'échantillonnage. H0 = « pas de croissance » (ratio <= 1) ; la p-value est la
proportion des tirages où cette hypothèse nulle n'est PAS rejetée."""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 3000
BOOTSTRAP_SEED = 42  # graine fixe : deux exécutions donnent exactement le même intervalle

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


def _cadre_cluster_standard(communes: pd.DataFrame, cluster: int, annee: int):
    """Isole un (cluster, année) sur la table 'communes_bkt_obs' standard - même logique que l'étape 06
    ('cadre_cluster' de bkt_extrapole.py, dupliquée ici pour que ce module reste lisible seul). 'communes'
    a une ligne par (année, commune, cluster) - toujours 'lineaire_km' (déjà réparti par cluster), jamais
    'total_length_km' (compterait plusieurs fois le réseau d'une commune à plusieurs clusters)."""
    c = communes[(communes["annee"] == annee) & (communes["cluster_K4"] == cluster)]
    l_total = c["lineaire_km"].sum()
    actif = c[(c["instrumentee"] == 1) & (c["lineaire_km"] > 0) & (c["bkt_obs"] > 0)].copy()
    actif["taux"] = actif["bkt_obs"] / actif["lineaire_km"]
    l_obs, bkt_obs = actif["lineaire_km"].sum(), actif["bkt_obs"].sum()
    return actif, l_obs, l_total, bkt_obs


def cadre_cluster_mini(mini: pd.DataFrame, cluster: int, annee: int):
    """Équivalent de '_cadre_cluster_standard', mais sur la table 'communes_mini' (panel équilibré, voir
    'indicateurs.py') au lieu de la table standard - même structure de retour, pour pouvoir réutiliser
    'bootstrap_national' avec l'une ou l'autre indifféremment."""
    c = mini[(mini["annee"] == annee) & (mini["cluster_K4"] == cluster)]
    l_total = c["lineaire_km"].sum()
    actif = c[(c["lineaire_km"] > 0) & (c["bkt_obs_mini"] > 0)].copy()
    actif["taux"] = actif["bkt_obs_mini"] / actif["lineaire_km"]
    l_obs, bkt_obs = actif["lineaire_km"].sum(), actif["bkt_obs_mini"].sum()
    return actif, l_obs, l_total, bkt_obs


def bootstrap_national(cadre_fn, source: pd.DataFrame, annee: int, rng: np.random.Generator, n_boot: int = N_BOOT) -> np.ndarray:
    """N_BOOT tirages du BKT_ext national d'UNE année : pour chaque cluster, rééchantillonne les
    communes actives, recalcule le taux rogné, applique-le au réseau non observé réel (jamais
    rééchantillonné), somme les clusters. 'cadre_fn' est soit '_cadre_cluster_standard' soit
    'cadre_cluster_mini'."""
    par_cluster = []
    for cl in range(N_CLUSTERS):
        actif, l_obs, l_total, bkt_obs = cadre_fn(source, cl, annee)
        taux, rognage, n = actif["taux"].to_numpy(), LOOCV_OPTIMAL_TRIM[cl], len(actif)
        tirages = np.empty(n_boot)
        for b in range(n_boot):
            echantillon = taux[rng.integers(0, n, n)] if n else taux
            tirages[b] = bkt_obs + taux_rogne(echantillon, rognage) * max(l_total - l_obs, 0.0)
        par_cluster.append(tirages)
    return np.sum(par_cluster, axis=0)


def _ic_toutes_annees(cadre_fn, source: pd.DataFrame, annees: tuple[int, ...], rng: np.random.Generator) -> pd.DataFrame:
    lignes = []
    for annee in annees:
        tirages = bootstrap_national(cadre_fn, source, annee, rng) / 1e9
        lignes.append(
            dict(
                annee=annee,
                mediane_bootstrap=np.median(tirages),
                ic_bas_95=np.percentile(tirages, 2.5),
                ic_haut_95=np.percentile(tirages, 97.5),
                largeur_relative_pct=100 * (np.percentile(tirages, 97.5) - np.percentile(tirages, 2.5)) / np.median(tirages),
            )
        )
    return pd.DataFrame(lignes)


def ic_standard(communes: pd.DataFrame, annees: tuple[int, ...], rng: np.random.Generator) -> pd.DataFrame:
    return _ic_toutes_annees(_cadre_cluster_standard, communes, annees, rng)


def ic_mini(mini: pd.DataFrame, annees: tuple[int, ...], rng: np.random.Generator) -> pd.DataFrame:
    return _ic_toutes_annees(cadre_cluster_mini, mini, annees, rng)


def bootstrap_croissance(communes: pd.DataFrame, annee_ref: int, annee_cmp: int, rng: np.random.Generator, n_boot: int = N_BOOT) -> np.ndarray:
    """N_BOOT tirages du ratio BKT_ext(annee_cmp) / BKT_ext(annee_ref), appariés : le même index de
    tirage sert aux deux années, uniquement sur les communes actives LES DEUX années (les seules
    comparables d'une année à l'autre)."""
    tirages_ref, tirages_cmp = np.zeros(n_boot), np.zeros(n_boot)
    for cl in range(N_CLUSTERS):
        c_ref, l_obs_ref, l_total_ref, bkt_obs_ref = _cadre_cluster_standard(communes, cl, annee_ref)
        c_cmp, l_obs_cmp, l_total_cmp, bkt_obs_cmp = _cadre_cluster_standard(communes, cl, annee_cmp)
        commun = c_ref.set_index("code_commune").index.intersection(c_cmp.set_index("code_commune").index)
        t_ref = c_ref.set_index("code_commune").loc[commun, "taux"].to_numpy()
        t_cmp = c_cmp.set_index("code_commune").loc[commun, "taux"].to_numpy()
        n, rognage = len(commun), LOOCV_OPTIMAL_TRIM[cl]
        for b in range(n_boot):
            idx = rng.integers(0, n, n) if n else np.array([], dtype=int)  # LE MÊME idx pour ref et cmp : c'est ça, "apparié"
            tirages_ref[b] += bkt_obs_ref + taux_rogne(t_ref[idx], rognage) * max(l_total_ref - l_obs_ref, 0.0)
            tirages_cmp[b] += bkt_obs_cmp + taux_rogne(t_cmp[idx], rognage) * max(l_total_cmp - l_obs_cmp, 0.0)
    return tirages_cmp / tirages_ref


def bootstrap_croissance_par_cluster(communes: pd.DataFrame, annee_ref: int, annee_cmp: int, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict[int, np.ndarray]:
    """Comme 'bootstrap_croissance', mais SANS sommer les clusters : retourne le ratio par cluster
    séparément - pour voir si l'incertitude de la croissance est la même dans tous les clusters, ou si
    un cluster précis tire (ou freine) la croissance nationale plus que les autres."""
    ratios: dict[int, np.ndarray] = {}
    for cl in range(N_CLUSTERS):
        c_ref, l_obs_ref, l_total_ref, bkt_obs_ref = _cadre_cluster_standard(communes, cl, annee_ref)
        c_cmp, l_obs_cmp, l_total_cmp, bkt_obs_cmp = _cadre_cluster_standard(communes, cl, annee_cmp)
        commun = c_ref.set_index("code_commune").index.intersection(c_cmp.set_index("code_commune").index)
        t_ref = c_ref.set_index("code_commune").loc[commun, "taux"].to_numpy()
        t_cmp = c_cmp.set_index("code_commune").loc[commun, "taux"].to_numpy()
        n, rognage = len(commun), LOOCV_OPTIMAL_TRIM[cl]
        tirages_ref, tirages_cmp = np.zeros(n_boot), np.zeros(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n) if n else np.array([], dtype=int)
            tirages_ref[b] = bkt_obs_ref + taux_rogne(t_ref[idx], rognage) * max(l_total_ref - l_obs_ref, 0.0)
            tirages_cmp[b] = bkt_obs_cmp + taux_rogne(t_cmp[idx], rognage) * max(l_total_cmp - l_obs_cmp, 0.0)
        ratios[cl] = tirages_cmp / tirages_ref
    return ratios


def test_croissance_par_cluster(communes: pd.DataFrame, annees: tuple[int, ...], rng: np.random.Generator) -> pd.DataFrame:
    """Une ligne par (transition, cluster) : ratio de croissance du BKT_ext médian et son IC 95 %,
    cluster par cluster - le pendant par-cluster de 'test_croissance' (national)."""
    transitions = [(y0, y1, f"{y0} -> {y1}") for y0, y1 in zip(annees[:-1], annees[1:], strict=True)]
    transitions.append((annees[0], annees[-1], f"{annees[0]} -> {annees[-1]} (cumulé)"))
    lignes = []
    for y0, y1, label in transitions:
        for cl, ratios in bootstrap_croissance_par_cluster(communes, y0, y1, rng).items():
            lignes.append(dict(transition=label, cluster=cl, cluster_nom=CLUSTER_NAMES[cl], ratio_median=np.median(ratios), ic_bas_95=np.percentile(ratios, 2.5), ic_haut_95=np.percentile(ratios, 97.5)))
    return pd.DataFrame(lignes)


def test_croissance(communes: pd.DataFrame, annees: tuple[int, ...], rng: np.random.Generator) -> pd.DataFrame:
    """Une ligne par transition année sur année, plus une ligne pour la croissance cumulée sur toute la série."""
    lignes = []
    for y0, y1 in zip(annees[:-1], annees[1:], strict=True):
        ratios = bootstrap_croissance(communes, y0, y1, rng)
        p = float(np.mean(ratios <= 1.0))  # proportion de tirages compatibles avec H0 : pas de croissance
        lignes.append(dict(transition=f"{y0} -> {y1}", ratio_median=np.median(ratios), ic_bas_95=np.percentile(ratios, 2.5), ic_haut_95=np.percentile(ratios, 97.5), p_value_pas_de_croissance=p, significatif_5pct="oui" if p < 0.05 else "non"))
    ratios_cum = bootstrap_croissance(communes, annees[0], annees[-1], rng)
    p_cum = float(np.mean(ratios_cum <= 1.0))
    lignes.append(
        dict(
            transition=f"{annees[0]} -> {annees[-1]} (cumulé)",
            ratio_median=np.median(ratios_cum),
            ic_bas_95=np.percentile(ratios_cum, 2.5),
            ic_haut_95=np.percentile(ratios_cum, 97.5),
            p_value_pas_de_croissance=p_cum,
            significatif_5pct="oui" if p_cum < 0.05 else "non",
        )
    )
    return pd.DataFrame(lignes)
