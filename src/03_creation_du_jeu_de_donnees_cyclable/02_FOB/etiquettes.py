# -*- coding: utf-8 -*-
"""Le vocabulaire GéoVélo et les étiquettes d'entraînement au niveau des tronçons ("ways") OSM - la
VÉRITÉ à partir de laquelle les modèles FOB apprennent (voir 'entrainement.py').

GéoVélo décrit chaque segment cyclable avec des attributs pairés droite/gauche ('ame_d'/'ame_g'...).
Ce module porte le VOCABULAIRE (cibles prédites, hiérarchie de protection, familles affichées dans les
analyses) - utilisé aussi bien pour ENTRAÎNER (pas exécutable dans ce dépôt, voir 'entrainement.py')
que pour INTERPRÉTER les prédictions d'un modèle déjà entraîné (voir 'longueurs.py',
'application_modele.py')."""

from __future__ import annotations

import numpy as np
import pandas as pd

CIBLES: tuple[str, ...] = ("ame_d", "ame_g", "revet_d", "revet_g", "sens_d", "sens_g", "regime_d", "regime_g", "local_d", "local_g")
"""Les 10 attributs prédits par les modèles d'attribut (voir 'application_modele.py') : ame = type
d'aménagement, revet = revêtement, sens = direction de circulation, regime = régime de circulation,
local = localisation ; suffixe _d/_g = côté droit/gauche du tronçon."""

NON_RENSEIGNE = "NON_RENSEIGNE"
BIDIRECTIONNEL = "BIDIRECTIONNEL"

# hiérarchie de protection, la MEILLEURE d'abord : le "type principal" d'un tronçon est le meilleur de
# ses deux côtés (voir 'type_principal').
HIERARCHIE: tuple[str, ...] = (
    "VOIE VERTE",
    "PISTE CYCLABLE",
    "AMENAGEMENT MIXTE PIETON VELO HORS VOIE VERTE",
    "GOULOTTE",
    "BANDE CYCLABLE",
    "COULOIR BUS+VELO",
    "ACCOTEMENT REVETU HORS CVCB",
    "CHAUSSEE A VOIE CENTRALE BANALISEE",
    "DOUBLE SENS CYCLABLE PISTE",
    "DOUBLE SENS CYCLABLE BANDE",
    "DOUBLE SENS CYCLABLE NON MATERIALISE",
    "VELO RUE",
    "AUTRE",
    "AUCUN",
    NON_RENSEIGNE,
)
_RANG = {nom: i for i, nom in enumerate(HIERARCHIE)}

# types qui ne sont PAS un aménagement dédié - exclus du scénario "aménagements seuls" (FOB_AM, voir
# 'reference.py' du dossier parent et 'longueurs.py').
TYPES_NON_AMENAGES = frozenset({"AUTRE", "AUCUN", NON_RENSEIGNE})

# regroupement en familles lisibles, pour une analyse par type d'aménagement (voir le notebook,
# section "analyse des aménagements") plutôt que par les 15 catégories brutes de HIERARCHIE.
#
# ATTENTION : seules les clés listées ici SONT des aménagements dédiés (même ensemble que HIERARCHIE
# moins TYPES_NON_AMENAGES, voir 'est_amenagement') - AUTRE/AUCUN/NON_RENSEIGNE n'ont PAS d'entrée dans
# cette table (ce ne sont PAS des familles d'aménagement, voir 'longueurs.py' et le notebook) : les y
# ajouter recréerait le bug corrigé ici, où une fausse famille "Autre / non spécifié" mélangeait de
# vrais aménagements (GOULOTTE) avec des tronçons SANS aménagement dédié (AUCUN) - trompeur, un
# utilisateur pouvait lire "15 % du réseau en famille Autre" en pensant à un type d'infrastructure alors
# qu'il s'agissait en grande partie de rues sans aménagement particulier.
FAMILLE: dict[str, str] = {
    "PISTE CYCLABLE": "Piste cyclable",
    "VOIE VERTE": "Voie verte",
    "AMENAGEMENT MIXTE PIETON VELO HORS VOIE VERTE": "Aménagement mixte piéton-vélo",
    "BANDE CYCLABLE": "Bande cyclable",
    "COULOIR BUS+VELO": "Autre aménagement de chaussée",
    "ACCOTEMENT REVETU HORS CVCB": "Autre aménagement de chaussée",
    "CHAUSSEE A VOIE CENTRALE BANALISEE": "Autre aménagement de chaussée",
    "GOULOTTE": "Autre aménagement de chaussée",
    "DOUBLE SENS CYCLABLE PISTE": "Double sens cyclable / vélo-rue",
    "DOUBLE SENS CYCLABLE BANDE": "Double sens cyclable / vélo-rue",
    "DOUBLE SENS CYCLABLE NON MATERIALISE": "Double sens cyclable / vélo-rue",
    "VELO RUE": "Double sens cyclable / vélo-rue",
}


def type_principal(droite: pd.Series, gauche: pd.Series) -> pd.Series:
    """Meilleur type d'aménagement d'un tronçon d'après ses côtés droit et gauche (voir HIERARCHIE)."""
    rang_d = droite.map(_RANG).fillna(len(HIERARCHIE)).to_numpy()
    rang_g = gauche.map(_RANG).fillna(len(HIERARCHIE)).to_numpy()
    noms = np.array(list(HIERARCHIE) + [NON_RENSEIGNE])
    return pd.Series(noms[np.minimum(rang_d, rang_g).astype(int)], index=droite.index)


def est_amenagement(types_principaux: pd.Series) -> np.ndarray:
    """Masque booléen : le tronçon porte un aménagement DÉDIÉ (ni AUTRE, ni AUCUN, ni non documenté) -
    c'est ce masque qui distingue FOB (tout) de FOB_AM (aménagements seuls), voir 'longueurs.py'."""
    return ~types_principaux.isin(TYPES_NON_AMENAGES).to_numpy()


def multiplicateur_direction(direction: pd.Series, alpha: float = 2.0) -> np.ndarray:
    """Multiplicateur de longueur effective d'un côté : un côté BIDIRECTIONNEL compte 'alpha' fois
    (convention du projet : 2 - on peut y circuler dans les deux sens, donc il "vaut" le double de
    linéaire utile qu'un aménagement à sens unique)."""
    return np.where(direction.astype(str) == BIDIRECTIONNEL, alpha, 1.0)
