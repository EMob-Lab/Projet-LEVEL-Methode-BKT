# -*- coding: utf-8 -*-
"""Entraîne les modèles FOB en deux étages : *inclusion* (ce tronçon OSM est-il dans GéoVélo ?) puis
*attributs* (qu'est-ce que GéoVélo en dit ? - 10 cibles, voir 'etiquettes.CIBLES').

NON EXÉCUTABLE dans ce dépôt : il faut un fichier de labels GéoVélo bruts comme vérité terrain (une
table 'id_osm -> valeur de chacune des 10 cibles', voir 'fichier_labels' de 'entrainer' ci-dessous) pour
appairer chaque tronçon OSM à son codage GéoVélo - une source tierce volumineuse, non fournie avec ce
dépôt (même remarque que pour les autres sources brutes, voir '00_transformation_des_donnees/README.md'
et 'artefacts.py'). Ce module est un PORT FIDÈLE du code qui a réellement produit les modèles
vendorisés dans 'models/fob/' (racine du dépôt - voir son 'meta.json' pour les vraies métriques et la
vraie durée d'entraînement, ~33 min) : il documente la méthode, il ne cherche pas à la reproduire ici.

Étage 1 - inclusion : un LightGBM binaire sur TOUS les tronçons 'highway=*' du snapshot d'entraînement.
Les négatifs (hors GéoVélo, l'immense majorité) sont sous-échantillonnés (25 %, voir
NEGATIVE_SAMPLE_FRACTION) et repondérés en conséquence ; le seuil de décision maximise le F1 sur le pli
de validation, pas un seuil fixe à 0.5 (voir 'entrainer_inclusion').

Étage 2 - attributs : un LightGBM multiclasse par attribut GéoVélo (10 cibles), entraîné seulement sur
les tronçons présents dans GéoVélo. Les deux modèles de type d'aménagement ('ame_d'/'ame_g') sont
réglés sur une petite grille d'hyperparamètres ; les huit autres cibles réutilisent directement les
paramètres gagnants de 'ame_d' (même vocabulaire de tags en entrée, pas la peine de re-régler 10 fois -
voir 'entrainer_attributs').

Découpage entraînement/validation/test : déterministe, par hachage de l'id OSM du tronçon (voir
'pli_de') - pas un tirage aléatoire, pour qu'un même tronçon tombe toujours dans le même pli d'un
snapshot à l'autre (un tronçon vu à l'entraînement sur le snapshot 2026 ne doit jamais se retrouver
dans le test d'un snapshot 2021)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics as skm

import caracteristiques as C
import etiquettes as L
from osm_extraction import lire_table_tags

N_PLIS = 5
PLI_TEST = 0  # pli 0 = test (jamais vu pendant le réglage) ; pli 1 = validation (arrêt anticipé) ; plis 2-4 = entraînement
PLI_VALIDATION = 1
_MULTIPLICATEUR_HACHAGE = np.uint64(2_654_435_761)  # hachage multiplicatif de Knuth

NEGATIVE_SAMPLE_FRACTION = 0.25
PARAMETRES_BASE = dict(
    learning_rate=0.06,
    num_leaves=63,
    min_data_in_leaf=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_cat_threshold=64,
    cat_smooth=10.0,
    cat_l2=10.0,
    max_bin=255,
    verbose=-1,
    seed=42,
    num_threads=min(10, os.cpu_count() or 1),
)
GRILLE_HYPERPARAMETRES = (
    dict(num_leaves=31, min_data_in_leaf=20, feature_fraction=0.8),
    dict(num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8),
    dict(num_leaves=127, min_data_in_leaf=10, feature_fraction=0.7),
    dict(num_leaves=255, min_data_in_leaf=5, feature_fraction=0.7),
    dict(num_leaves=63, min_data_in_leaf=50, feature_fraction=0.6, lambda_l2=5.0),
)
CIBLES_AMENAGEMENT = ("ame_d", "ame_g")  # les seules réglées sur la grille - les autres réutilisent leurs paramètres


def pli_de(ids_osm: np.ndarray) -> np.ndarray:
    """Numéro de pli (0..N_PLIS-1) de chaque id de tronçon OSM - déterministe, voir le docstring du module."""
    hache = (ids_osm.astype(np.uint64) * _MULTIPLICATEUR_HACHAGE) >> np.uint64(7)
    return (hache % np.uint64(N_PLIS)).astype(np.int8)


@dataclass
class Ajustement:
    modele: lgb.Booster
    parametres: dict
    score: float
    iteration: int


def _rechercher(grille, X_tr, y_tr, X_va, y_va, *, objectif, metrique=None, n_classes=None, poids_tr=None, poids_va=None, plus_grand_meilleur=False) -> Ajustement:
    """Ajuste chaque jeu d'hyperparamètres de 'grille' avec arrêt anticipé sur la validation, garde le meilleur."""
    meilleur: Ajustement | None = None
    for parametres in grille:
        complets = {**PARAMETRES_BASE, **parametres, "objective": objectif}
        if metrique:
            complets["metric"] = metrique
        if n_classes:
            complets["num_class"] = n_classes
        jeu_entrainement = lgb.Dataset(X_tr, y_tr, weight=poids_tr, free_raw_data=False)
        jeu_validation = lgb.Dataset(X_va, y_va, weight=poids_va, reference=jeu_entrainement, free_raw_data=False)
        modele = lgb.train(
            complets, jeu_entrainement, num_boost_round=2000, valid_sets=[jeu_validation], valid_names=["val"], callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        nom_metrique = next(iter(modele.best_score["val"]))
        score = float(modele.best_score["val"][nom_metrique])
        mieux = meilleur is None or (score > meilleur.score if plus_grand_meilleur else score < meilleur.score)
        if mieux:
            meilleur = Ajustement(modele, parametres, score, modele.best_iteration)
    assert meilleur is not None
    return meilleur


# ═══════════════════════════════════════════════════════════════════════════════════════ étage 1
def _comptes_binaires(y: np.ndarray, predit: np.ndarray) -> dict:
    """Vrais/faux positifs et négatifs - mêmes noms de champs que le 'meta.json' vendorisé (tp/fp/fn/tn),
    pour pouvoir comparer directement la sortie de ce module aux vraies métriques."""
    tp, fp = int((predit & (y == 1)).sum()), int((predit & (y == 0)).sum())
    fn, tn = int((~predit & (y == 1)).sum()), int((~predit & (y == 0)).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1))


def entrainer_inclusion(brut: pd.DataFrame, spec: dict, dans_geovelo: np.ndarray, pli: np.ndarray, grille, dossier_modeles: Path) -> dict:
    """Ajuste le modèle d'inclusion, le sauvegarde ('inclusion.txt'), retourne ses métriques de test (pli 0)."""
    y = dans_geovelo.astype(np.int8)
    garder = (y == 1) | (np.random.RandomState(0).rand(len(y)) < NEGATIVE_SAMPLE_FRACTION)
    entrainement, validation, test = (pli >= 2) & garder, (pli == PLI_VALIDATION) & garder, pli == PLI_TEST
    poids = lambda m: np.where(y[m] == 1, 1.0, 1.0 / NEGATIVE_SAMPLE_FRACTION)
    X_tr = C.vers_matrice(brut[entrainement].reset_index(drop=True), spec)
    X_va = C.vers_matrice(brut[validation].reset_index(drop=True), spec)
    ajustement = _rechercher(
        grille,
        X_tr,
        y[entrainement],
        X_va,
        y[validation],
        objectif="binary",
        metrique="average_precision",
        poids_tr=poids(entrainement),
        poids_va=poids(validation),
        plus_grand_meilleur=True,
    )
    ajustement.modele.save_model(str(dossier_modeles / "inclusion.txt"), num_iteration=ajustement.iteration)

    p_validation = ajustement.modele.predict(X_va, num_iteration=ajustement.iteration)
    precision, rappel, seuils = skm.precision_recall_curve(y[validation], p_validation, sample_weight=poids(validation))
    f1 = 2 * precision * rappel / (precision + rappel + 1e-12)
    seuil = float(seuils[f1[:-1].argmax()])  # seuil qui maximise le F1 sur la validation - pas 0.5 fixe

    X_test = C.vers_matrice(brut[test].reset_index(drop=True), spec)
    p_test = ajustement.modele.predict(X_test, num_iteration=ajustement.iteration)
    return dict(
        ap=float(skm.average_precision_score(y[test], p_test)),
        auc=float(skm.roc_auc_score(y[test], p_test)),
        threshold=seuil,
        params=ajustement.parametres,
        best_iteration=int(ajustement.iteration),
        neg_sample_frac=NEGATIVE_SAMPLE_FRACTION,
        n_holdout=int(test.sum()),
        n_pos_holdout=int(y[test].sum()),
        **_comptes_binaires(y[test], p_test >= seuil),
    )


# ═══════════════════════════════════════════════════════════════════════════════════════ étage 2
def _base_par_lookup(X: pd.DataFrame, etiquettes: pd.Series, entrainement: np.ndarray, test: np.ndarray, colonnes: list[str], defaut: str) -> float:
    """Accuracy de la ligne de base "étiquette la plus fréquente pour cette combinaison de tags" - la
    barre que le modèle doit dépasser pour justifier d'apprendre autre chose qu'un simple dictionnaire."""

    def cle(lignes):
        parties = [X.loc[lignes, c].astype(object).fillna("(absent)").astype(str) for c in colonnes]
        return parties[0].str.cat(parties[1:], sep="|") if len(parties) > 1 else parties[0]

    table = pd.DataFrame({"k": cle(entrainement).to_numpy(), "v": etiquettes.to_numpy()[entrainement]}).groupby("k")["v"].agg(lambda s: s.value_counts().idxmax())
    devine = cle(test).map(table).fillna(defaut).to_numpy()
    return float((devine == etiquettes.to_numpy()[test]).mean())


def entrainer_attributs(
    brut: pd.DataFrame, spec: dict, verite: pd.DataFrame, ids: np.ndarray, dans_geovelo: np.ndarray, pli: np.ndarray, grille, dossier_modeles: Path
) -> tuple[dict, list[str]]:
    """Ajuste un modèle multiclasse par attribut GéoVélo (voir 'etiquettes.CIBLES') ; retourne (métriques
    de test par cible, noms des colonnes de la matrice de variables)."""
    lignes = np.flatnonzero(dans_geovelo)
    verite_alignee = verite.set_index("id_osm").loc[ids[lignes]].reset_index()
    X = C.vers_matrice(brut.iloc[lignes].reset_index(drop=True), spec)
    pli_lignes = pli[lignes]
    entrainement, validation, test = pli_lignes >= 2, pli_lignes == PLI_VALIDATION, pli_lignes == PLI_TEST

    resultats: dict[str, dict] = {}
    parametres_partages = None
    for cible in L.CIBLES:
        classes = sorted(verite_alignee[cible].unique())
        code = verite_alignee[cible].map({c: i for i, c in enumerate(classes)}).to_numpy()
        regler = cible in CIBLES_AMENAGEMENT or parametres_partages is None
        ajustement = _rechercher(
            grille if regler else [parametres_partages], X[entrainement], code[entrainement], X[validation], code[validation], objectif="multiclass", n_classes=len(classes)
        )
        if cible == "ame_d":
            parametres_partages = ajustement.parametres
        ajustement.modele.save_model(str(dossier_modeles / f"{cible}.txt"), num_iteration=ajustement.iteration)

        proba = ajustement.modele.predict(X[test], num_iteration=ajustement.iteration)
        predit = np.array(classes)[proba.argmax(1)]
        reel = verite_alignee[cible].to_numpy()[test]
        rapport = skm.classification_report(reel, predit, labels=classes, output_dict=True, zero_division=0)
        majoritaire = pd.Series(verite_alignee[cible].to_numpy()[entrainement]).mode()[0]
        cote = cible[-1]
        colonnes_base = ["highway", {"d": "cw_right_eff", "g": "cw_left_eff"}[cote]] if cible in CIBLES_AMENAGEMENT else ["highway"]
        resultats[cible] = dict(
            accuracy=float(skm.accuracy_score(reel, predit)),
            balanced_accuracy=float(skm.balanced_accuracy_score(reel, predit)),
            macro_f1=float(rapport["macro avg"]["f1-score"]),
            weighted_f1=float(rapport["weighted avg"]["f1-score"]),
            cohen_kappa=float(skm.cohen_kappa_score(reel, predit)),
            baseline_majoritaire=float((reel == majoritaire).mean()),
            baseline_lookup=_base_par_lookup(X, verite_alignee[cible], np.flatnonzero(entrainement), np.flatnonzero(test), colonnes_base, majoritaire),
            n_holdout=int(test.sum()),
            n_train=int(entrainement.sum()),
            classes=classes,
            params=ajustement.parametres,
            best_iteration=int(ajustement.iteration),
            val_logloss=ajustement.score,
            majority_class=majoritaire,
        )
    return resultats, list(X.columns)


# ═══════════════════════════════════════════════════════════════════════════════════════ point d'entrée
def entrainer(dossier_tags_snapshot: Path, fichier_labels: Path, dossier_modeles_sortie: Path, *, rapide: bool = False) -> Path:
    """Entraîne tous les modèles FOB (inclusion + 10 attributs) sur la sortie de la passe A d'un
    snapshot OSM ('dossier_tags_snapshot', voir 'osm_extraction.ecrire_table_tags') et les labels
    GéoVélo ('fichier_labels' : une table 'id_osm -> valeur de chacune des 10 cibles') ; écrit les
    modèles LightGBM et 'meta.json' dans 'dossier_modeles_sortie' (même mise en forme que
    'models/fob/meta.json', voir 'artefacts.resume_qualite' pour l'afficher).

    NON EXÉCUTABLE ici : 'fichier_labels' n'existe pas dans ce dépôt (voir le docstring du module)."""
    debut = time.time()
    (dossier_modeles_sortie / "models").mkdir(parents=True, exist_ok=True)
    brut = lire_table_tags(dossier_tags_snapshot)
    spec = C.construire_spec(pd.read_parquet(dossier_tags_snapshot / "value_counts.parquet"))
    C.sauvegarder_spec(spec, dossier_modeles_sortie / "feature_spec.json")
    labels = pd.read_parquet(fichier_labels)

    ids = brut["id_osm"].to_numpy()
    dans_geovelo = np.isin(ids, labels["id_osm"].to_numpy())
    pli = pli_de(ids)
    grille = GRILLE_HYPERPARAMETRES[:2] if rapide else GRILLE_HYPERPARAMETRES

    meta = dict(
        tag=str(dossier_tags_snapshot),
        n_ways=int(len(ids)),
        n_pos=int(dans_geovelo.sum()),
        n_labels_gv=int(len(labels)),
        holdout_fold=PLI_TEST,
        n_folds=N_PLIS,
    )
    meta["inclusion"] = entrainer_inclusion(brut, spec, dans_geovelo, pli, grille, dossier_modeles_sortie / "models")
    meta["targets"], meta["features"] = entrainer_attributs(brut, spec, labels, ids, dans_geovelo, pli, grille, dossier_modeles_sortie / "models")
    meta["train_seconds"] = round(time.time() - debut)
    (dossier_modeles_sortie / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return dossier_modeles_sortie
