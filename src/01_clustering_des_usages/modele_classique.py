"""Un modèle de clustering "classique" : funFEM K=4 ajusté sur le SEUL profil hebdomadaire (voir
'representation.representation_hebdomadaire'), SANS recherche d'hyperparamètres - l'alternative simple
à 'recherche_hyperparametres.py' quand on veut juste un K=4 dans l'esprit de la taxonomie historique du
projet (4 pratiques d'usage), sans
reproduire le run exact : ce dépôt n'utilise plus la donnée vendorisée de l'étude d'origine (voir le
README du projet) - ce module entraîne un modèle comparable, mais sur VOS propres capteurs.

explication : pourquoi une seule échelle temporelle suffit-elle ici ?

Le rythme hebdomadaire (168 points : quelle heure de quel jour de la semaine) porte à lui seul
l'essentiel de ce qui distingue des pratiques d'usage différentes - un pic du matin ET du soir en
semaine (domicile-travail), un usage étalé le week-end plutôt qu'en semaine (loisirs). La
représentation multi-échelle de 'representation.py' (jour + semaine + année + spectral, chacun avec un
poids à régler) capture plus de nuances mais demande une recherche d'hyperparamètres (voir
'recherche_hyperparametres.py') pour bien régler ces poids ; ce module fait le choix inverse : une
seule échelle, aucun réglage, un résultat "raisonnable" tout de suite.

**Limite assumée** : sans le profil annuel, ce modèle ne peut pas distinguer un usage touristique
SAISONNIER d'un usage régulier au même rythme hebdomadaire - à garder en tête en lisant les clusters
obtenus (regardez leur profil ANNUEL, section 6 du notebook, une fois le clustering fait, pour
vérifier). La carte affiche quand même TOUS les profils (jour/semaine/année - voir 'carte.py') : seul
le CLUSTERING ignore les échelles jour/année/spectral, pas leur AFFICHAGE."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from FunFEM import funFEM  # noqa: E402
from representation import (  # noqa: E402
    N_BASE_SEMAINE_DEFAUT,
    representation_hebdomadaire,
)

MAX_TENTATIVES = 20  # plus que recherche_hyperparametres.py : un seul modèle à ajuster ici, on peut se le permettre


def ajuster_modele_classique(
    profils: pd.DataFrame,
    *,
    k: int = 4,
    modele: str = "AkBk",
    critere: str = "bic",
    n_base_semaine: int = N_BASE_SEMAINE_DEFAUT,
    graine: int = 42,
    tentatives: int = MAX_TENTATIVES,
) -> tuple[funFEM, int]:
    """Ajuste funFEM(K='k' fixé) sur la représentation hebdomadaire seule, en essayant plusieurs
    graines et en gardant le meilleur ajustement CONVERGÉ selon 'critere' (funFEM peut échouer sur une
    graine malchanceuse - surface de vraisemblance accidentée, voir la docstring de 'FunFEM.py').

    Renvoie '(modele_funFEM, graine_retenue)' - passez les deux à 'sauvegarder_modele'
    ('appliquer_modele.py') pour vendoriser ce modèle comme n'importe quel résultat de
    'recherche_hyperparametres.recherche_optuna'.
    """
    X, W = representation_hebdomadaire(profils, n_base_semaine=n_base_semaine)
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
        f"[modele_classique] K={k} {modele} sur le profil hebdomadaire seul : {critere}={meilleur_score:.2f} (graine {meilleure_graine} retenue sur {tentatives} essayées)"
    )
    return meilleur, meilleure_graine


def config_modele_classique(
    modele_funfem: funFEM,
    graine: int,
    *,
    k: int,
    modele: str,
    critere: str,
    n_base_semaine: int,
) -> dict:
    """Construit le 'config.json' de ce modèle - même forme que celui de
    'recherche_hyperparametres.py' (clé 'representation' comprise, voir 'appliquer_modele.py'), pour
    que 'appliquer_modele.assigner_clusters' le recharge sans distinction."""
    return {
        "rang": 1,
        "essai": 0,
        critere: float(getattr(modele_funfem, critere)),
        "k": k,
        "modele": modele,
        "graine": graine,
        "representation": "hebdomadaire",
        "n_base_semaine": n_base_semaine,
        "poids_semaine": 1.0,
    }
