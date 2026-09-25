"""Transforme un tableau de profils de capteurs (colonnes 'd00..d23', 'h000..h167', 'y00..y51',
's0..s4' produites par '00_transformation_des_donnees/d_debit/profils_usage.py') en la représentation
numérique '(X, W)' attendue par funFEM (voir 'FunFEM.py').

explication de la méthode :

funFEM classe des données FONCTIONNELLES (des courbes), pas des vecteurs bruts. On a ici QUATRE courbes
par capteur, à des échelles de temps différentes :

    jour       24 points   (débit moyen de chaque heure de la journée)
    semaine    168 points  (débit moyen de chaque heure de la semaine)
    année      52 points   (débit moyen de chaque semaine de l'année - la saisonnalité)
    spectral   5 points    (pas une courbe : 5 nombres qui résument le rythme du signal)

Chaque courbe est d'abord ramenée à une FORME comparable (norme unitaire : deux capteurs à fort et
faible trafic mais au même rythme doivent être proches), puis centrée-réduite colonne par colonne, puis
lissée sur une base de B-splines (moins de coefficients que de points bruts = moins de bruit, plus de
généralisation). Le bloc spectral n'est pas une courbe : juste centré-réduit, sans base.

On concatène les coefficients des 4 blocs en un seul vecteur par capteur ('X'), et on donne à funFEM
une métrique 'W' (produit scalaire) qui n'est PAS l'identité : 'W' est diagonale par blocs, un bloc
par échelle de temps, chaque bloc étant le produit scalaire des splines de son échelle MULTIPLIÉ par un
poids. Plus le poids d'une échelle est grand, plus elle pèse dans la distance que funFEM utilise pour
regrouper les capteurs - c'est le principal levier de réglage de cette représentation (voir
'recherche_hyperparametres.py' qui fait varier ces poids et ces tailles de base automatiquement).

explication des PARAMÈTRES :
- n_base_jour / n_base_semaine / n_base_annee : nombre de fonctions B-splines de chaque bloc temporel
  (plus grand = la courbe lissée colle plus fidèlement aux 24/168/52 points bruts, au risque de
  réintroduire du bruit individuel si c'est trop grand).
- poids_jour / poids_semaine / poids_annee / poids_spectral : importance de chaque bloc dans la
  distance utilisée par funFEM (voir plus haut).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import skfda
from skfda.representation.basis import BSplineBasis

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "00_transformation_des_donnees"
        / "d_debit"
    ),
)
from profils_usage import (  # noqa: E402
    COLONNES_ANNEE,
    COLONNES_JOUR,
    COLONNES_SEMAINE,
    COLONNES_SPECTRALES,
)

EPS = 1e-12

# valeurs par défaut = celles de l'étude K=4 historique
N_BASE_JOUR_DEFAUT = 12
N_BASE_SEMAINE_DEFAUT = 18
N_BASE_ANNEE_DEFAUT = 6
POIDS_JOUR_DEFAUT = 1.25
POIDS_SEMAINE_DEFAUT = 2.5
POIDS_ANNEE_DEFAUT = 1.5
POIDS_SPECTRAL_DEFAUT = 0.75


@dataclass(frozen=True)
class Bloc:
    colonnes: list[str]
    n_base: (
        int | None
    )  # None = utilisé tel quel (bloc spectral, pas de lissage par base)
    poids: float


def _norme_unitaire(X: np.ndarray) -> np.ndarray:
    """Ramène chaque ligne (le profil d'UN capteur) à une norme de 1 - on compare des FORMES d'usage,
    pas des volumes de trafic."""
    normes = np.linalg.norm(X, axis=1, keepdims=True)
    normes[normes == 0] = 1.0
    return X / normes


def _centre_reduit(X: np.ndarray) -> np.ndarray:
    """Centre-réduit colonne par colonne (chaque point temporel devient comparable aux autres)."""
    return (X - X.mean(axis=0, keepdims=True)) / np.maximum(
        X.std(axis=0, keepdims=True), EPS
    )


def representation_multiechelle(
    profils: pd.DataFrame,
    *,
    n_base_jour: int = N_BASE_JOUR_DEFAUT,
    n_base_semaine: int = N_BASE_SEMAINE_DEFAUT,
    n_base_annee: int = N_BASE_ANNEE_DEFAUT,
    poids_jour: float = POIDS_JOUR_DEFAUT,
    poids_semaine: float = POIDS_SEMAINE_DEFAUT,
    poids_annee: float = POIDS_ANNEE_DEFAUT,
    poids_spectral: float = POIDS_SPECTRAL_DEFAUT,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit '(X, W)' à partir des colonnes de profil de 'profils' (une ligne par capteur).

    'X' : les coefficients (un capteur par ligne) que funFEM reçoit comme données.
    'W' : la métrique (produit scalaire) diagonale par blocs que funFEM reçoit en plus de 'X' -
    c'est elle qui fait qu'une distance entre deux capteurs n'est pas juste euclidienne sur les
    coefficients, mais pondérée bloc par bloc comme expliqué dans le docstring du module.
    """
    blocs = {
        "jour": Bloc(COLONNES_JOUR, n_base_jour, poids_jour),
        "semaine": Bloc(COLONNES_SEMAINE, n_base_semaine, poids_semaine),
        "annee": Bloc(COLONNES_ANNEE, n_base_annee, poids_annee),
        "spectral": Bloc(COLONNES_SPECTRALES, None, poids_spectral),
    }

    coefficients, metriques = [], []
    for bloc in blocs.values():
        X = profils[bloc.colonnes].to_numpy(dtype=float)

        if bloc.n_base is None:
            # bloc spectral : pas de courbe à lisser, juste centré-réduit, métrique = identité pondérée
            coefficients.append(_centre_reduit(X))
            metriques.append(bloc.poids * np.eye(X.shape[1]))
            continue

        # étape 1 : forme (norme 1) puis centrage-réduction colonne par colonne
        X_forme = _centre_reduit(_norme_unitaire(X))
        # étape 2 : lissage sur une base de B-splines (une grille [0, 1] abstraite - seule la FORME compte)
        base = BSplineBasis(domain_range=(0, 1), n_basis=bloc.n_base)
        lisse = skfda.FDataGrid(
            data_matrix=X_forme, grid_points=np.linspace(0, 1, X.shape[1])
        ).to_basis(base)
        # étape 3 : les coefficients sont divisés par sqrt(nb de fonctions de base) pour que des bases de
        # tailles différentes restent comparables en échelle (sinon plus de base = plus de "poids" mécanique)
        coefficients.append(lisse.coefficients / np.sqrt(lisse.coefficients.shape[1]))
        metriques.append(bloc.poids * base.inner_product_matrix())

    X = np.concatenate(coefficients, axis=1)
    # standardisation finale, colonne par colonne, sur la concaténation des 4 blocs
    etalement = X.std(axis=0)
    etalement[etalement == 0] = 1.0
    X = (X - X.mean(axis=0)) / etalement
    W = scipy.linalg.block_diag(*metriques)
    return X, W


def representation_hebdomadaire(
    profils: pd.DataFrame,
    *,
    n_base_semaine: int = N_BASE_SEMAINE_DEFAUT,
    poids_semaine: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Version "classique", UNE SEULE échelle temporelle : construit '(X, W)' à partir du seul profil
    hebdomadaire ('h000'..'h167'), sans jour/année/spectral - l'alternative simple à
    'representation_multiechelle' pour un clustering direct (voir 'modele_classique.py'), sans
    recherche d'hyperparamètres : rien à régler, un seul bloc, un seul poids qui n'a plus vraiment de
    sens vu qu'il n'y a rien d'autre à pondérer relativement (gardé pour la forme, laissez-le à 1.0).

    Même traitement que le bloc "semaine" de 'representation_multiechelle' (voir son docstring pour le
    détail : norme unitaire, centrage-réduction, lissage B-splines) - seulement, ici, SANS
    concaténation avec d'autres blocs."""
    X_brut = profils[COLONNES_SEMAINE].to_numpy(dtype=float)
    X_forme = _centre_reduit(_norme_unitaire(X_brut))
    base = BSplineBasis(domain_range=(0, 1), n_basis=n_base_semaine)
    lisse = skfda.FDataGrid(
        data_matrix=X_forme, grid_points=np.linspace(0, 1, X_brut.shape[1])
    ).to_basis(base)
    coefficients = lisse.coefficients / np.sqrt(lisse.coefficients.shape[1])
    etalement = coefficients.std(axis=0)
    etalement[etalement == 0] = 1.0
    X = (coefficients - coefficients.mean(axis=0)) / etalement
    W = poids_semaine * base.inner_product_matrix()
    return X, W


def hyperparametres_par_defaut() -> dict[str, float]:
    """Le jeu d'hyperparamètres par défaut (celui de l'étude K=4 historique), pratique pour initialiser
    une recherche d'hyperparamètres (voir 'recherche_hyperparametres.py') autour d'un point connu."""
    return {
        "n_base_jour": N_BASE_JOUR_DEFAUT,
        "n_base_semaine": N_BASE_SEMAINE_DEFAUT,
        "n_base_annee": N_BASE_ANNEE_DEFAUT,
        "poids_jour": POIDS_JOUR_DEFAUT,
        "poids_semaine": POIDS_SEMAINE_DEFAUT,
        "poids_annee": POIDS_ANNEE_DEFAUT,
        "poids_spectral": POIDS_SPECTRAL_DEFAUT,
    }
