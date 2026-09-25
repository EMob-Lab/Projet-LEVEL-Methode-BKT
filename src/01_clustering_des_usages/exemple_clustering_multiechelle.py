"""Exemple : ajuster funFEM directement sur PLUSIEURS profils à la fois (la représentation
MULTI-ÉCHELLE de 'representation.py' - jour + semaine + année + spectral), plutôt que sur le seul
profil hebdomadaire ('modele_classique.py') ou via une recherche d'hyperparamètres complète
('recherche_hyperparametres.py').

Fichier de démonstration : il montre juste comment passer de plusieurs profils à un clustering funFEM.
Pour l'utiliser pour de vrai dans le pipeline, passez par 'recherche_hyperparametres.py' (qui règle
aussi les poids de chaque échelle) ou copiez le motif ci-dessous à la main.

Pourquoi PLUSIEURS profils plutôt qu'un seul :

Un capteur peut avoir le même rythme hebdomadaire (168 points : quelle heure de quel jour) que deux
pratiques d'usage bien différentes - un usage touristique saisonnier n'a pas la même forme sur le
profil annuel (52 points) qu'un usage régulier toute l'année, même s'ils se ressemblent semaine par
semaine. En donnant à funFEM les quatre échelles à la fois (jour/semaine/année/spectral, voir
'representation.representation_multiechelle'), le clustering peut distinguer ces cas - au prix d'un
hyperparamètre de plus par échelle (une taille de base B-splines et un poids, voir la docstring de
'representation.py') à choisir ou régler."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from FunFEM import funFEM  # noqa: E402
from representation import (  # noqa: E402
    hyperparametres_par_defaut,
    representation_multiechelle,
)

MAX_TENTATIVES = 20  # un seul modèle à ajuster ici (pas une recherche sur des dizaines d'essais), on peut se le permettre


def ajuster_modele_multiechelle(
    profils: pd.DataFrame,
    *,
    k: int = 4,
    modele: str = "AkBk",
    critere: str = "bic",
    hyperparametres: dict[str, float] | None = None,
    graine: int = 42,
    tentatives: int = MAX_TENTATIVES,
) -> tuple[funFEM, int]:
    """Ajuste funFEM(K='k' fixé) sur les QUATRE échelles à la fois (jour/semaine/année/spectral), en
    essayant plusieurs graines et en gardant le meilleur ajustement CONVERGÉ selon 'critere' (funFEM
    peut échouer sur une graine malchanceuse - surface de vraisemblance accidentée, voir la docstring
    de 'FunFEM.py').

    'hyperparametres' : les tailles de base et poids de chaque échelle (voir
    'representation.representation_multiechelle') - par défaut, ceux de
    'representation.hyperparametres_par_defaut()' (le point de départ raisonnable de l'étude
    historique), à ajuster à la main si le résultat ne convainc pas (voir la section 4 du notebook de
    ce dossier pour comment lire les profils moyens obtenus, cluster par cluster).

    Renvoie '(modele_funFEM, graine_retenue)'."""
    hp = hyperparametres or hyperparametres_par_defaut()
    X, W = representation_multiechelle(profils, **hp)

    meilleur, meilleure_graine, meilleur_score = None, None, -np.inf
    for tentative in range(tentatives):
        graine_essai = graine + tentative
        try:
            m = funFEM(
                cluster_interval=(k, k),
                model=[modele],
                crit=critere,
                init="kmeans",
                maxit=100,
                eps=1e-6,
                random_state=graine_essai,
            )
            m.fit(X, W)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        score = float(getattr(m, critere))
        if score > meilleur_score:
            meilleur, meilleure_graine, meilleur_score = m, graine_essai, score
    if meilleur is None:
        raise RuntimeError(
            f"funFEM n'a convergé sur aucune des {tentatives} graines essayées"
        )
    print(
        f"[exemple_clustering_multiechelle] K={k} {modele} sur les 4 échelles : {critere}={meilleur_score:.2f} (graine {meilleure_graine} retenue sur {tentatives} essayées)"
    )
    return meilleur, meilleure_graine


if __name__ == "__main__":
    # Démonstration directement exécutable : mêmes profils que le reste de l'étape 01
    # ('00_transformation_des_donnees/usage_profiles.parquet'), le même K=4 que partout ailleurs dans
    # ce dépôt, aucune sauvegarde - juste la taille de chaque cluster obtenu.
    RACINE = Path(__file__).resolve().parents[2]
    FICHIER_PROFILS = (
        RACINE / "data" / "donnees_valides" / "capteurs" / "usage_profiles.parquet"
    )
    ANNEE_REFERENCE = 2024

    if not FICHIER_PROFILS.exists():
        raise FileNotFoundError(
            f"{FICHIER_PROFILS} manquant - lancez d'abord 00_transformation_des_donnees/pipeline.py."
        )

    profils_toutes_annees = pd.read_parquet(FICHIER_PROFILS)
    profils = profils_toutes_annees[
        profils_toutes_annees["annee"] == ANNEE_REFERENCE
    ].reset_index(drop=True)
    print(f"{len(profils):,} capteurs, année {ANNEE_REFERENCE}")

    modele_funfem, graine_retenue = ajuster_modele_multiechelle(profils, k=4)
    tailles = pd.Series(modele_funfem.cls).value_counts().sort_index()
    print("\ntaille de chaque cluster :")
    print(tailles.to_string())
