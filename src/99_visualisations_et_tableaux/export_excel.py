# -*- coding: utf-8 -*-
"""Export du classeur Excel final, à partir des sorties des étapes 04-06 et de 'indicateurs.py'/
'intervalle_confiance.py' (même dossier) - DERNIER MODULE DU PIPELINE : il ne fait que relire des
tables déjà calculées et les mettre en forme (voir 'xlsx.py', l'utilitaire xlsxwriter du même dossier).

6 feuilles :
    Résumé              BKT observé/extrapolé national, % extrapolé, avec IC 95% (résumé)
    Indicateurs de base toutes les statistiques réunies (dont débit/capteur), valeurs brutes + indice
                        base 100 (échelle à partir de 90, pas 0 - rien ne repasse sous 100), 3 graphiques
    Détail par cluster  chaque métrique décomposée par cluster (+ colonne Total national où elle a un
                        sens) : débit, débit/capteur, capteurs, réseau, % de réseau et de BKT extrapolés,
                        taux robuste, BKT observé/extrapolé, facteurs de croissance, IC de la croissance
    Intervalle de confiance   détail du bootstrap, méthode standard
    IC BKT mini          même bootstrap, sur le panel équilibré
    Test de croissance   bootstrap apparié, la croissance est-elle statistiquement significative ?

**Rappel (voir le README) : les IC de ce classeur décrivent l'incertitude de NOTRE méthode, pas celle
du "vrai" BKT national.**"""

from __future__ import annotations

import pandas as pd

from xlsx import BON, FOND_ENTETE, PALETTE, Classeur, plage, serie

LIBELLES_INDICATEURS = {
    "population": "Population",
    "n_capteurs_actifs": "N capteurs actifs (standard)",
    "n_capteurs_panel": "N capteurs (panel équilibré)",
    "lineaire_fob_km": "Linéaire FOB (km)",
    "somme_debits": "Somme des débits (passages/an)",
    "debit_par_capteur": "Débit / capteur (passages/an)",
    "bkt_obs_Mdkm": "BKT observé (Md km)",
    "bkt_ext_Mdkm": "BKT extrapolé (Md km)",
    "bkt_mini_obs_Mdkm": "BKT mini observé (Md km)",
    "bkt_mini_ext_Mdkm": "BKT mini extrapolé (Md km)",
}
COULEURS_INDICATEURS = ["#17324d", "#eb6834", "#8a8a8a", "#eda100", "#4a3aa7", "#c9963c", "#2a78d6", "#d03b3b", "#7ab8e6", "#e87ba4"]
COULEUR_PAR_INDICATEUR = dict(zip(LIBELLES_INDICATEURS, COULEURS_INDICATEURS, strict=True))
CLES_BKT = ["bkt_obs_Mdkm", "bkt_ext_Mdkm", "bkt_mini_obs_Mdkm", "bkt_mini_ext_Mdkm"]
CLES_AUTRES = [k for k in LIBELLES_INDICATEURS if k not in CLES_BKT]
ECART_GRAPHIQUES = 24  # lignes entre deux graphiques empilés (hauteur approx. d'un graphique h=420px)


def _indicateurs_derives(indicateurs: pd.DataFrame) -> pd.DataFrame:
    """Deux métriques dérivées, calculées ici (pas besoin de recalculer les indicateurs) : le débit
    moyen par capteur actif, et la part du BKT national qui vient de l'extrapolation plutôt que de la
    mesure directe."""
    indicateurs = indicateurs.copy()
    indicateurs["debit_par_capteur"] = indicateurs["somme_debits"] / indicateurs["n_capteurs_actifs"]
    indicateurs["pct_bkt_extrapole"] = (indicateurs["bkt_ext_Mdkm"] - indicateurs["bkt_obs_Mdkm"]) / indicateurs["bkt_ext_Mdkm"] * 100
    return indicateurs


def _pivot_cluster(detail_cl: pd.DataFrame, colonnes_cluster: list[str], colonne_valeur: str, total: pd.Series | str | None = "sum") -> pd.DataFrame:
    """Pivote la table longue (une ligne par année x cluster) en large (une colonne par cluster).
    'total="sum"' ajoute une colonne Total (correct pour une métrique ADDITIVE : débit, réseau, BKT) ;
    une 'pd.Series' indexée par année fournit un total déjà recalculé correctement (taux/pourcentages,
    jamais une simple somme des clusters) ; 'None' n'ajoute pas de colonne Total."""
    piv = detail_cl.pivot(index="annee", columns="cluster_nom", values=colonne_valeur).reindex(columns=colonnes_cluster, fill_value=0)
    if isinstance(total, str) and total == "sum":
        piv["Total"] = piv[colonnes_cluster].sum(axis=1)
    elif total is not None:
        piv["Total"] = total.reindex(piv.index)
    return piv.reset_index()


def _avec_total(large: pd.DataFrame, colonnes_cluster: list[str]) -> pd.DataFrame:
    large = large.copy()
    large["Total"] = large[colonnes_cluster].sum(axis=1)
    return large


def _ratio_par_cluster(numerateur: pd.DataFrame, denominateur: pd.DataFrame, colonnes_cluster: list[str]) -> pd.DataFrame:
    """Division élément par élément de deux tables 'annee x cluster [x Total]' de même forme (ex. débit
    / n capteurs) - recalcule un Total correct (Total numérateur / Total dénominateur), PAS la somme
    des ratios de cluster. 0/0 (cluster sans capteur une année donnée) donne 0, pas une erreur."""
    cols = [c for c in colonnes_cluster + ["Total"] if c in numerateur.columns and c in denominateur.columns]
    num, den = numerateur.set_index("annee")[cols], denominateur.set_index("annee")[cols]
    return (num / den.replace(0, pd.NA)).fillna(0.0).reset_index()


def _facteur_croissance(large: pd.DataFrame) -> pd.DataFrame:
    """Facteur de croissance annuel (valeur(N) / valeur(N-1)) de chaque colonne (cluster + Total)."""
    sortie = large[["annee"]].copy()
    for c in large.columns:
        if c != "annee":
            sortie[c] = large[c] / large[c].shift(1)
    return sortie


def _croissance_par_cluster_et_national(test_par_cluster: pd.DataFrame, test_national: pd.DataFrame, colonnes_cluster: list[str]) -> pd.DataFrame:
    """Une ligne par (transition, cluster), plus une ligne « Total (national) » par transition (le
    bootstrap national de 'test_croissance', pas la somme des IC de cluster - une somme d'IC n'aurait
    aucun sens statistique)."""
    lignes = []
    for t in test_national["transition"]:
        for nom_cluster in colonnes_cluster:
            ligne = test_par_cluster[(test_par_cluster["transition"] == t) & (test_par_cluster["cluster_nom"] == nom_cluster)].iloc[0]
            lignes.append({"Transition": t, "Cluster": nom_cluster, "Ratio médian": ligne["ratio_median"], "IC bas (2.5%)": ligne["ic_bas_95"], "IC haut (97.5%)": ligne["ic_haut_95"]})
        national = test_national[test_national["transition"] == t].iloc[0]
        lignes.append({"Transition": t, "Cluster": "Total (national)", "Ratio médian": national["ratio_median"], "IC bas (2.5%)": national["ic_bas_95"], "IC haut (97.5%)": national["ic_haut_95"]})
    return pd.DataFrame(lignes)


def construire_classeur(chemin_xlsx: str, dossier_bkt, cluster_names: dict[int, str]) -> None:
    """Lit toutes les tables déjà écrites dans 'dossier_bkt' (étapes 04-06, indicateurs.py,
    intervalle_confiance.py) et construit le classeur final."""
    colonnes_cluster = [cluster_names[i] for i in range(len(cluster_names))]

    indicateurs = _indicateurs_derives(pd.read_csv(dossier_bkt / "indicateurs.csv"))
    ic = pd.read_csv(dossier_bkt / "ic_standard.csv")
    ic_mini = pd.read_csv(dossier_bkt / "ic_mini.csv")
    test = pd.read_csv(dossier_bkt / "test_croissance.csv")
    test_cl = pd.read_csv(dossier_bkt / "test_croissance_par_cluster.csv")
    debits_cl = pd.read_csv(dossier_bkt / "debits_par_cluster.csv")
    capteurs_std_cl = pd.read_csv(dossier_bkt / "capteurs_par_cluster_standard.csv")
    capteurs_mini_cl = pd.read_csv(dossier_bkt / "capteurs_par_cluster_mini.csv")
    detail_cl = pd.read_csv(dossier_bkt / "bkt_final_detail_par_cluster.csv")

    cls = Classeur(chemin_xlsx)
    _feuille_resume(cls, indicateurs, ic)
    _feuille_indicateurs(cls, indicateurs)
    _feuille_detail_cluster(cls, debits_cl, capteurs_std_cl, capteurs_mini_cl, detail_cl, test_cl, test, colonnes_cluster)
    _feuille_ic(cls, indicateurs, ic, "Intervalle de confiance", PALETTE[0], "standard", "bkt_obs_Mdkm", "bkt_ext_Mdkm")
    _feuille_ic(cls, indicateurs, ic_mini, "IC BKT mini", PALETTE[6], "mini", "bkt_mini_obs_Mdkm", "bkt_mini_ext_Mdkm")
    _feuille_test(cls, test)
    cls.fermer()


# ============================================================================= 1. Résumé
def _feuille_resume(cls: Classeur, indicateurs: pd.DataFrame, ic: pd.DataFrame) -> None:
    ws = cls.feuille("Résumé", onglet=FOND_ENTETE)
    r = cls.titre(
        ws, 0, "BKT final : avec n, Z complet, rognage LOOCV par cluster",
        "Quelle confiance accorder au résultat ? BKT observé/extrapolé national avec IC 95% (incertitude "
        "de NOTRE méthode, voir le README), et test de significativité de la croissance. 3000 tirages "
        "bootstrap par année/comparaison.", largeur=8,
    )
    disp = ic.rename(columns={"annee": "Année", "mediane_bootstrap": "Médiane bootstrap", "ic_bas_95": "IC bas (2.5%)", "ic_haut_95": "IC haut (97.5%)", "largeur_relative_pct": "Largeur relative (%)"})
    disp.insert(1, "BKT observé", indicateurs.set_index("annee")["bkt_obs_Mdkm"].reindex(disp["Année"]).to_numpy())
    disp.insert(2, "BKT_ext (point)", indicateurs.set_index("annee")["bkt_ext_Mdkm"].reindex(disp["Année"]).to_numpy())
    disp.insert(3, "% extrapolé", indicateurs.set_index("annee")["pct_bkt_extrapole"].reindex(disp["Année"]).to_numpy())
    disp = disp[["Année", "BKT observé", "BKT_ext (point)", "% extrapolé", "Médiane bootstrap", "IC bas (2.5%)", "IC haut (97.5%)", "Largeur relative (%)"]]
    r0 = r
    r, _ = cls.tableau(
        ws, r, 0, disp,
        formats={c: "0.000" for c in disp.columns if c not in ("Année", "Largeur relative (%)", "% extrapolé")} | {"Largeur relative (%)": "0.0", "% extrapolé": "0.0"},
        largeurs={c: 16 for c in disp.columns}, premiere_colonne_grasse=True,
    )
    r = cls.note(ws, r, "« % extrapolé » = part du BKT_ext qui vient du taux robuste appliqué au réseau NON observé (BKT_ext - BKT observé) / BKT_ext - le reste est mesure directe.", largeur=7)
    r += 1
    ch = cls.graphique("line", "BKT observé et extrapolé national, avec IC 95% (Md km)", titre_y="Md km", titre_x="Année", legende="bottom")
    cats = plage("Résumé", r0 + 1, 0, r0 + len(disp), 0)
    for j, couleur, tiret in ((1, "#2a78d6", None), (2, "#17324d", None), (5, "#c9c7c1", "dash"), (6, "#c9c7c1", "dash")):
        extra = {"line": {"color": couleur, "width": 1.5 if tiret else 2.5, "dash_type": tiret}} if tiret else {}
        ch.add_series(serie(plage("Résumé", r0, j), cats, plage("Résumé", r0 + 1, j, r0 + len(disp), j), couleur, type_="line", **extra))
    ws.insert_chart(r + 1, 0, ch)
    ws.set_column(0, 0, 9)


# ============================================================================= 2. Indicateurs de base
def _feuille_indicateurs(cls: Classeur, indicateurs: pd.DataFrame) -> None:
    ws = cls.feuille("Indicateurs de base", onglet=PALETTE[1])
    r = cls.titre(ws, 0, "Tous les indicateurs — évolution", largeur=8)
    cols = list(LIBELLES_INDICATEURS)

    r = cls.sous_titre(ws, r, "Valeurs brutes", 0, len(cols))
    disp = indicateurs.rename(columns={"annee": "Année", **LIBELLES_INDICATEURS})[["Année"] + [LIBELLES_INDICATEURS[c] for c in cols]]
    fmt_entier = {"Population", "N capteurs actifs (standard)", "N capteurs (panel équilibré)", "Linéaire FOB (km)", "Somme des débits (passages/an)", "Débit / capteur (passages/an)"}
    formats = {c: ("#,##0" if c in fmt_entier else "0.000") for c in disp.columns if c != "Année"}
    r, _ = cls.tableau(ws, r, 0, disp, formats=formats, largeurs={c: 15 for c in disp.columns}, premiere_colonne_grasse=True)
    r += 1

    r = cls.sous_titre(ws, r, "Indice base 100 = première année", 0, len(cols))
    idx = indicateurs.copy()
    for c in cols:
        idx[c] = idx[c] / idx[c].iloc[0] * 100
    idx_disp = idx.rename(columns={"annee": "Année", **LIBELLES_INDICATEURS})[["Année"] + [LIBELLES_INDICATEURS[c] for c in cols]]
    r0 = r
    r, _ = cls.tableau(ws, r, 0, idx_disp, formats={c: "0.0" for c in idx_disp.columns if c != "Année"}, largeurs={c: 15 for c in idx_disp.columns}, premiere_colonne_grasse=True)
    r += 1
    index_colonne = {cle: i + 1 for i, cle in enumerate(cols)}

    def _graphique_indicateurs(titre: str, cles: list[str]) -> None:
        nonlocal r
        ch = cls.graphique("line", titre, titre_y="indice (première année = 100)", titre_x="Année", legende="bottom", largeur=900, hauteur=420, y_min=90)
        cats = plage("Indicateurs de base", r0 + 1, 0, r0 + len(idx_disp), 0)
        for cle in cles:
            j, couleur = index_colonne[cle], COULEUR_PAR_INDICATEUR[cle]
            ch.add_series(serie(plage("Indicateurs de base", r0, j), cats, plage("Indicateurs de base", r0 + 1, j, r0 + len(idx_disp), j), couleur, type_="line"))
        ws.insert_chart(r, 0, ch)
        r += ECART_GRAPHIQUES

    _graphique_indicateurs("Tous les indicateurs (indice base 100)", cols)
    _graphique_indicateurs("BKT — standard et mini, observé et extrapolé (indice base 100)", CLES_BKT)
    _graphique_indicateurs("Population, capteurs, réseau, débit (indice base 100)", CLES_AUTRES)
    ws.set_column(0, 0, 9)


# ============================================================================= 3. Détail par cluster
def _feuille_detail_cluster(
    cls: Classeur, debits_cl: pd.DataFrame, capteurs_std_cl: pd.DataFrame, capteurs_mini_cl: pd.DataFrame,
    detail_cl: pd.DataFrame, test_cl: pd.DataFrame, test_national: pd.DataFrame, colonnes_cluster: list[str],
) -> None:
    """Décompose CHAQUE métrique du pipeline par cluster d'usage (K=4) et par année."""
    ws = cls.feuille("Détail par cluster", onglet=PALETTE[4])
    r = cls.titre(
        ws, 0, "Décomposition par cluster et par métrique",
        "Méthode standard (capteurs actifs cette année-là) vs panel équilibré (capteurs présents toute "
        "la série, constant). « % extrapolé » = part qui n'est PAS mesurée directement. « Total » = "
        "national, recalculé correctement (pas une somme de taux/pourcentages) là où ça a un sens.", largeur=8,
    )

    detail_cl = detail_cl.copy()
    detail_cl["pct_reseau_extrapole"] = (1 - detail_cl["l_obs"] / detail_cl["l_total"]) * 100
    detail_cl["pct_bkt_extrapole"] = (detail_cl["bkt_ext"] - detail_cl["bkt_obs"]) / detail_cl["bkt_ext"] * 100
    detail_cl["bkt_obs_Mdkm"] = detail_cl["bkt_obs"] / 1e9
    detail_cl["bkt_ext_Mdkm"] = detail_cl["bkt_ext"] / 1e9
    national_an = detail_cl.groupby("annee").agg(l_obs=("l_obs", "sum"), l_total=("l_total", "sum"), bkt_obs=("bkt_obs", "sum"), bkt_ext=("bkt_ext", "sum"))
    pct_reseau_total = (1 - national_an["l_obs"] / national_an["l_total"]) * 100
    pct_bkt_total = (national_an["bkt_ext"] - national_an["bkt_obs"]) / national_an["bkt_ext"] * 100

    debits_cl, capteurs_std_cl, capteurs_mini_cl = _avec_total(debits_cl, colonnes_cluster), _avec_total(capteurs_std_cl, colonnes_cluster), _avec_total(capteurs_mini_cl, colonnes_cluster)
    debit_par_capteur_cl = _ratio_par_cluster(debits_cl, capteurs_std_cl, colonnes_cluster)
    bkt_obs_cl, bkt_ext_cl = _pivot_cluster(detail_cl, colonnes_cluster, "bkt_obs_Mdkm"), _pivot_cluster(detail_cl, colonnes_cluster, "bkt_ext_Mdkm")

    blocs = [
        ("Somme des débits par cluster (passages/an)", debits_cl, "#,##0"),
        ("N capteurs par cluster — méthode standard", capteurs_std_cl, "#,##0"),
        ("N capteurs par cluster — panel équilibré (mini)", capteurs_mini_cl, "#,##0"),
        ("Débit / capteur par cluster (passages/an)", debit_par_capteur_cl, "#,##0"),
        ("Réseau total par cluster (km)", _pivot_cluster(detail_cl, colonnes_cluster, "l_total"), "#,##0"),
        ("Réseau OBSERVÉ par cluster (km)", _pivot_cluster(detail_cl, colonnes_cluster, "l_obs"), "#,##0"),
        ("% du réseau extrapolé par cluster", _pivot_cluster(detail_cl, colonnes_cluster, "pct_reseau_extrapole", total=pct_reseau_total), "0.0"),
        ("Taux robuste par cluster (bike-km / km de réseau non observé)", _pivot_cluster(detail_cl, colonnes_cluster, "taux_fréquentation", total=None), "#,##0"),
        ("BKT observé par cluster (Md km)", bkt_obs_cl, "0.000"),
        ("BKT extrapolé par cluster (Md km)", bkt_ext_cl, "0.000"),
        ("% du BKT extrapolé par cluster", _pivot_cluster(detail_cl, colonnes_cluster, "pct_bkt_extrapole", total=pct_bkt_total), "0.0"),
        ("Facteur de croissance du BKT observé par cluster (année N / année N-1)", _facteur_croissance(bkt_obs_cl), '0.00"×"'),
        ("Facteur de croissance du BKT extrapolé par cluster (année N / année N-1)", _facteur_croissance(bkt_ext_cl), '0.00"×"'),
    ]
    for libelle, df, format_nombre in blocs:
        r = cls.sous_titre(ws, r, libelle, 0, len(df.columns) - 1)
        disp = df.rename(columns={"annee": "Année"})
        r, _ = cls.tableau(ws, r, 0, disp, formats={c: format_nombre for c in disp.columns if c != "Année"}, largeurs={c: 20 for c in disp.columns}, premiere_colonne_grasse=True)
        r += 1

    r = cls.sous_titre(ws, r, "Croissance du BKT extrapolé par cluster, avec IC 95% (bootstrap apparié, communes actives les 2 années)", 0, 4)
    r = cls.note(
        ws, r, "« Ratio médian » = médiane de 3000 tirages bootstrap, PAS le même calcul que le facteur de croissance "
        "ci-dessus (un ratio simple des deux points) - les deux peuvent légèrement différer, c'est normal (le rognage "
        "n'est pas une moyenne linéaire). Le facteur de croissance ci-dessus est le chiffre officiel ; ce bloc-ci "
        "donne son incertitude.", largeur=7,
    )
    croissance = _croissance_par_cluster_et_national(test_cl, test_national, colonnes_cluster)
    cls.tableau(
        ws, r, 0, croissance,
        formats={"Ratio médian": '0.00"×"', "IC bas (2.5%)": '0.00"×"', "IC haut (97.5%)": '0.00"×"'},
        largeurs={"Transition": 22, "Cluster": 20, "Ratio médian": 14, "IC bas (2.5%)": 14, "IC haut (97.5%)": 14},
        premiere_colonne_grasse=True,
    )
    ws.set_column(0, 0, 9)


# ============================================================================= 4-5. Intervalle de confiance
def _feuille_ic(cls: Classeur, indicateurs: pd.DataFrame, ic: pd.DataFrame, nom_feuille: str, couleur_onglet: str, variante: str, col_obs: str, col_ext: str) -> None:
    ws = cls.feuille(nom_feuille, onglet=couleur_onglet)
    r = cls.titre(
        ws, 0, f"Bootstrap à 3000 tirages — {variante}",
        "Rééchantillonnage avec remise des communes actives de chaque cluster, recalcul du taux robuste "
        "(rognage LOOCV), application au réseau non observé réel. BKT_obs gardé fixe.", largeur=9,
    )
    disp = ic.rename(columns={"annee": "Année", "mediane_bootstrap": "Médiane bootstrap", "ic_bas_95": "IC bas (2.5%)", "ic_haut_95": "IC haut (97.5%)", "largeur_relative_pct": "Largeur relative (%)"})
    disp.insert(1, "BKT observé", indicateurs.set_index("annee")[col_obs].reindex(disp["Année"]).to_numpy())
    disp.insert(2, "BKT extrapolé (point)", indicateurs.set_index("annee")[col_ext].reindex(disp["Année"]).to_numpy())
    r0 = r
    r, _ = cls.tableau(ws, r, 0, disp, formats={c: "0.000" for c in disp.columns if c not in ("Année", "Largeur relative (%)")} | {"Largeur relative (%)": "0.0"}, largeurs={c: 18 for c in disp.columns}, premiere_colonne_grasse=True)
    ch = cls.graphique("line", f"BKT — {variante}, avec IC 95% (Md km)", titre_y="Md km", titre_x="Année", legende="bottom")
    cats = plage(nom_feuille, r0 + 1, 0, r0 + len(disp), 0)
    for j, couleur, tiret in ((1, "#2a78d6", None), (2, "#d03b3b", None), (4, "#c9c7c1", "dash"), (5, "#c9c7c1", "dash")):
        extra = {"line": {"color": couleur, "width": 1.5 if tiret else 2.5, "dash_type": tiret}} if tiret else {}
        ch.add_series(serie(plage(nom_feuille, r0, j), cats, plage(nom_feuille, r0 + 1, j, r0 + len(disp), j), couleur, type_="line", **extra))
    ws.insert_chart(r + 1, 0, ch)
    ws.set_column(0, 0, 9)


# ============================================================================= 6. Test de croissance
def _feuille_test(cls: Classeur, test: pd.DataFrame) -> None:
    ws = cls.feuille("Test de croissance", onglet=PALETTE[2])
    r = cls.titre(
        ws, 0, "Bootstrap apparié : la croissance est-elle statistiquement significative ?",
        "H0 : « pas de croissance » (ratio <= 1). p-value = part des tirages où le ratio tombe à 1 ou en "
        "dessous - comparaison appariée, communes actives les deux années.", largeur=9,
    )
    disp = test.rename(columns={
        "transition": "Transition", "ratio_median": "Ratio médian", "ic_bas_95": "IC bas (2.5%)", "ic_haut_95": "IC haut (97.5%)",
        "p_value_pas_de_croissance": "p-value (H0 : pas de croissance)", "significatif_5pct": "Significatif à 5% ?",
    })
    cls.tableau(
        ws, r, 0, disp,
        formats={"Ratio médian": '0.00"×"', "IC bas (2.5%)": '0.00"×"', "IC haut (97.5%)": '0.00"×"', "p-value (H0 : pas de croissance)": "0.0000"},
        largeurs={"Transition": 22, "Ratio médian": 15, "IC bas (2.5%)": 15, "IC haut (97.5%)": 15, "p-value (H0 : pas de croissance)": 24, "Significatif à 5% ?": 16},
        premiere_colonne_grasse=True,
    )
    ws.set_column(0, 0, 9)
