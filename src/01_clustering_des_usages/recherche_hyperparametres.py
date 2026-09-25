"""Recherche d'hyperparamètres (Optuna) pour le clustering d'usage funFEM : fait varier le nombre de
clusters K, le modèle de covariance funFEM, et les hyperparamètres de la représentation multi-échelle
('representation.py' : tailles de base B-splines et poids de chaque échelle temporelle), et sauvegarde
les meilleures configurations trouvées.

**Outil interactif, PAS un maillon du pipeline** : rien dans '00_transformation_des_donnees' ni un
futur 'pipeline.py' de ce dossier n'appelle 'recherche_optuna' automatiquement - une recherche
d'hyperparamètres est un choix ponctuel (voir le run historique K=4 vendorisé, 'reference.py')
qu'on ne referait qu'à la main, pour explorer ou mettre à jour la référence. Le notebook de ce dossier
montre comment l'appeler.

explication du fonctionnement D'OPTUNA :

Optuna explore un ESPACE DE RECHERCHE (ici : K, modèle, tailles de base, poids) en proposant des
combinaisons ("essais"/'trial') à une fonction 'objectif' qu'on lui fournit, qui doit renvoyer UN
nombre à maximiser (ici : le critère funFEM choisi - BIC/AIC/ICL). Deux façons de proposer ces
combinaisons (paramètre 'echantillonneur') :
- "tpe" (par défaut) : un échantillonneur BAYÉSIEN (Tree-structured Parzen Estimator) - après
  quelques essais, Optuna apprend quelles zones de l'espace donnent de bons résultats et concentre les
  essais suivants dessus. Adapté ici car l'espace complet (K x modèle x 3 tailles de base x 4 poids)
  contient des dizaines de milliers de combinaisons : le tester intégralement serait beaucoup trop long
  (c'est exactement pour cette raison que l'étude historique, voir
  'V2/FunFEM_clustering/grid_search_in_depth.py', utilisait déjà un TIRAGE plutôt qu'une grille
  complète). 'n_essais' contrôle combien de combinaisons sont réellement testées.
- "grille" : un 'GridSampler' EXHAUSTIF - teste toutes les combinaisons de la grille, dans un
  ordre aléatoire mais sans jamais en répéter une. À réserver à une grille volontairement petite (sinon
  'n_essais' sera atteint bien avant d'avoir fait le tour de l'espace, et le résultat sera partiel).

Chaque essai est enregistré dans une "étude" ('optuna.Study'), qui peut être sauvegardée dans un
fichier SQLite ('stockage') pour être reprise plus tard sans recommencer à zéro ('load_if_exists').

explication de SAUVEGARDER_TOP :

Une fois la recherche terminée, les essais sont classés par critère décroissant, et les 'sauvegarder_top'
meilleurs (valeurs typiques : 1, 5, 10, 15, 20, 25 - un 'int' quelconque convient) sont REFIT (funFEM
est déterministe à graine fixée, mais Optuna ne garde pas le modèle entraîné de chaque essai pour
économiser la mémoire - on le recalcule donc une seconde fois, seulement pour les meilleurs) puis
sauvegardés chacun dans son propre dossier 'dossier_modeles/<rang>_<identifiant>/' avec :

    config.json               hyperparamètres, K, modèle, valeur du critère, graine
    modele.joblib              l'objet 'FunFEM.funFEM' entraîné (voir 'joblib.load' dans
                               'appliquer_modele.py' pour le recharger)
    assignation_clusters.csv   'id_site, cluster' pour tous les capteurs utilisés dans cet essai
    carte.html                 la carte interactive de ce cluster (voir 'carte.py') - seulement SI
                               'donnees_capteurs' est fourni (coordonnées des capteurs)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from appliquer_modele import sauvegarder_modele  # noqa: E402
from FunFEM import funFEM  # noqa: E402
from representation import representation_multiechelle  # noqa: E402

MAX_TENTATIVES_AJUSTEMENT = (
    5  # nombre de graines essayées si funFEM échoue (cluster vide, etc.)
)


def _ajuster_avec_relances(
    X: np.ndarray, W: np.ndarray, k: int, modele: str, critere: str, graine: int
) -> funFEM | None:
    """Ajuste funFEM(K fixé, un seul modèle) en réessayant avec une graine différente si l'ajustement
    échoue (surface de vraisemblance accidentée - un tirage initial malchanceux vide un cluster)."""
    for tentative in range(MAX_TENTATIVES_AJUSTEMENT):
        try:
            m = funFEM(
                cluster_interval=(k, k),
                model=[modele],
                crit=critere,
                init="kmeans",
                maxit=100,
                eps=1e-6,
                random_state=graine + tentative,
            )
            m.fit(X, W)
            return m
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    return None


def _construire_objectif(
    profils: pd.DataFrame, espace: dict, critere: str, graine_base: int
) -> tuple:
    """Construit la fonction 'objectif' qu'Optuna appelle à chaque essai, et un cache
    'essai -> résultat' (le modèle entraîné n'est PAS gardé, seulement ses métriques - voir le
    docstring du module) pour retrouver plus tard les hyperparamètres du meilleur essai."""

    def objectif(trial: "optuna.Trial") -> float:
        hp = {
            "k": trial.suggest_categorical("k", espace["k"]),
            "modele": trial.suggest_categorical("modele", espace["modele"]),
            "n_base_jour": trial.suggest_categorical(
                "n_base_jour", espace["n_base_jour"]
            ),
            "n_base_semaine": trial.suggest_categorical(
                "n_base_semaine", espace["n_base_semaine"]
            ),
            "n_base_annee": trial.suggest_categorical(
                "n_base_annee", espace["n_base_annee"]
            ),
            "poids_jour": trial.suggest_categorical("poids_jour", espace["poids_jour"]),
            "poids_semaine": trial.suggest_categorical(
                "poids_semaine", espace["poids_semaine"]
            ),
            "poids_annee": trial.suggest_categorical(
                "poids_annee", espace["poids_annee"]
            ),
            "poids_spectral": trial.suggest_categorical(
                "poids_spectral", espace["poids_spectral"]
            ),
        }
        X, W = representation_multiechelle(
            profils,
            n_base_jour=hp["n_base_jour"],
            n_base_semaine=hp["n_base_semaine"],
            n_base_annee=hp["n_base_annee"],
            poids_jour=hp["poids_jour"],
            poids_semaine=hp["poids_semaine"],
            poids_annee=hp["poids_annee"],
            poids_spectral=hp["poids_spectral"],
        )
        # graine dérivée du numéro d'essai : reproductible, différente d'un essai à l'autre
        m = _ajuster_avec_relances(
            X, W, hp["k"], hp["modele"], critere, graine_base + trial.number
        )
        if m is None:
            raise optuna.exceptions.TrialPruned(
                "funFEM n'a convergé sur aucune graine pour ces hyperparamètres"
            )
        trial.set_user_attr("n_clusters_obtenus", int(m.K))
        return float(getattr(m, critere))

    return objectif


def recherche_optuna(
    profils: pd.DataFrame,
    *,
    k_min: int = 2,
    k_max: int = 8,
    modeles: tuple[str, ...] = ("AkBk",),
    critere: str = "bic",
    grille_n_base_jour: tuple[int, ...] = (8, 10, 12, 14),
    grille_n_base_semaine: tuple[int, ...] = (12, 15, 18),
    grille_n_base_annee: tuple[int, ...] = (4, 6, 8, 10),
    grille_poids_jour: tuple[float, ...] = (0.75, 1.0, 1.25),
    grille_poids_semaine: tuple[float, ...] = (1.5, 2.0, 2.5),
    grille_poids_annee: tuple[float, ...] = (1.0, 1.5, 2.0),
    grille_poids_spectral: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0),
    echantillonneur: str = "tpe",
    n_essais: int = 80,
    graine: int = 42,
    nom_etude: str = "clustering_debits",
    stockage: str | None = None,
    sauvegarder_top: int = 1,
    dossier_modeles: Path | str | None = None,
    donnees_capteurs: pd.DataFrame | None = None,
    couleurs: dict | None = None,
) -> pd.DataFrame:
    """Explore l'espace des hyperparamètres funFEM et sauvegarde les 'sauvegarder_top' meilleures
    configurations (voir le docstring du module pour le détail des deux paramètres).

    'profils' : une ligne par capteur, colonnes de profil 'd00..d23' / 'h000..h167' / 'y00..y51'
    / 's0..s4' (voir '00_transformation_des_donnees/d_debit/profils_usage.py') et 'id_site'.

    'donnees_capteurs' : 'id_site, lat, long, nom_site' (voir
    '00_transformation_des_donnees/d_debit/capteurs_annees.py') - optionnel, seulement nécessaire
    pour que les meilleures configurations sauvegardées incluent une carte (sinon cette étape est
    silencieusement sautée : config/modèle/assignation sont sauvegardés quand même).

    Renvoie la table de tous les essais (triée par critère décroissant), avec pour chacun ses
    hyperparamètres, le critère obtenu et le K réellement atteint.
    """
    if critere not in ("bic", "aic", "icl"):
        raise ValueError(f"critere={critere!r} attendu parmi 'bic', 'aic', 'icl'")
    if sauvegarder_top < 1:
        raise ValueError("sauvegarder_top doit être >= 1")

    espace = {
        "k": list(range(k_min, k_max + 1)),
        "modele": list(modeles),
        "n_base_jour": list(grille_n_base_jour),
        "n_base_semaine": list(grille_n_base_semaine),
        "n_base_annee": list(grille_n_base_annee),
        "poids_jour": list(grille_poids_jour),
        "poids_semaine": list(grille_poids_semaine),
        "poids_annee": list(grille_poids_annee),
        "poids_spectral": list(grille_poids_spectral),
    }
    taille_grille = 1
    for valeurs in espace.values():
        taille_grille *= len(valeurs)
    print(
        f"[recherche] espace de recherche : {taille_grille:,} combinaisons ; {n_essais} essai(s) réellement exécuté(s) ({echantillonneur})"
    )

    if echantillonneur == "tpe":
        sampler = optuna.samplers.TPESampler(seed=graine)
    elif echantillonneur == "grille":
        sampler = optuna.samplers.GridSampler(espace, seed=graine)
    else:
        raise ValueError(
            f"echantillonneur={echantillonneur!r} attendu parmi 'tpe', 'grille'"
        )

    etude = optuna.create_study(
        study_name=nom_etude,
        storage=stockage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=stockage is not None,
    )
    objectif = _construire_objectif(profils, espace, critere, graine)
    etude.optimize(
        objectif, n_trials=n_essais, show_progress_bar=True, catch=(Exception,)
    )

    essais_termines = [
        t for t in etude.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not essais_termines:
        raise RuntimeError(
            "aucun essai n'a abouti - élargir la grille ou vérifier 'profils'"
        )

    table_essais = (
        pd.DataFrame(
            [
                {
                    "essai": t.number,
                    critere: t.value,
                    "k_obtenu": t.user_attrs.get("n_clusters_obtenus"),
                    **t.params,
                }
                for t in essais_termines
            ]
        )
        .sort_values(critere, ascending=False)
        .reset_index(drop=True)
    )

    _sauvegarder_meilleurs_essais(
        table_essais.head(sauvegarder_top),
        profils,
        critere,
        graine,
        dossier_modeles=Path(dossier_modeles)
        if dossier_modeles
        else Path(__file__).resolve().parents[2] / "models" / "clustering_des_debits",
        donnees_capteurs=donnees_capteurs,
        couleurs=couleurs,
    )
    return table_essais


def _sauvegarder_meilleurs_essais(
    top: pd.DataFrame,
    profils: pd.DataFrame,
    critere: str,
    graine: int,
    *,
    dossier_modeles: Path,
    donnees_capteurs: pd.DataFrame | None,
    couleurs: dict | None,
) -> None:
    """Refit et sauvegarde chacun des essais de 'top' (voir 'appliquer_modele.sauvegarder_modele' pour
    le détail des 4 fichiers écrits par essai)."""
    for rang, ligne in enumerate(top.itertuples(), start=1):
        hp = {
            c: getattr(ligne, c)
            for c in (
                "n_base_jour",
                "n_base_semaine",
                "n_base_annee",
                "poids_jour",
                "poids_semaine",
                "poids_annee",
                "poids_spectral",
            )
        }
        X, W = representation_multiechelle(profils, **hp)
        modele = _ajuster_avec_relances(
            X, W, int(ligne.k_obtenu), ligne.modele, critere, graine + int(ligne.essai)
        )
        if (
            modele is None
        ):  # ne devrait pas arriver (cet essai avait déjà réussi) mais on n'écrase rien de partiel
            print(
                f"[recherche] rang {rang} (essai {ligne.essai}) : le refit a échoué, ignoré"
            )
            continue

        identifiant = f"{rang:02d}_essai{int(ligne.essai):04d}_K{int(ligne.k_obtenu)}_{ligne.modele}"
        dossier = dossier_modeles / identifiant
        config = {
            "rang": rang,
            "essai": int(ligne.essai),
            critere: float(getattr(ligne, critere)),
            "k": int(ligne.k_obtenu),
            "modele": ligne.modele,
            "graine": graine + int(ligne.essai),
            "representation": "multiechelle",
            **hp,
        }
        sauvegarder_modele(
            dossier,
            modele,
            config,
            profils,
            donnees_capteurs=donnees_capteurs,
            couleurs=couleurs,
        )
        print(
            f"[recherche] rang {rang}/{len(top)} sauvegardé : {dossier} ({critere}={getattr(ligne, critere):.2f}, K={int(ligne.k_obtenu)})"
        )
