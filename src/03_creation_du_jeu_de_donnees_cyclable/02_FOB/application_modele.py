# -*- coding: utf-8 -*-
"""Applique les modèles LightGBM DÉJÀ ENTRAÎNÉS (vendorisés, voir 'artefacts.py' - ce dépôt n'a pas les
exports GéoVélo bruts nécessaires pour les réentraîner, voir 'entrainement.py') à la table de tags
brute d'un snapshot OSM (voir 'osm_extraction.py').

Chaque tronçon 'highway=*' reçoit une probabilité d'INCLUSION dans le réseau cyclable. Les tronçons
au-dessus de 'PROBA_MIN_CANDIDAT' (2%) sont gardés comme CANDIDATS et reçoivent les 10 attributs
GéoVélo prédits (voir 'etiquettes.CIBLES') avec leur confiance ; ceux au-dessus du seuil de décision du
modèle (voir 'Modeles.seuil') sont marqués 'incl' - ce sont eux qui forment le réseau FOB."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import caracteristiques as C
import etiquettes as L

PROBA_MIN_CANDIDAT = 0.02
LIGNES_PAR_LOT = 1_500_000  # lignes converties en matrice de variables à la fois (borne le pic mémoire)


@dataclass
class Modeles:
    meta: dict
    spec: dict
    inclusion: lgb.Booster
    attributs: dict[str, lgb.Booster]

    @property
    def seuil(self) -> float:
        return self.meta["inclusion"]["threshold"]


def charger_modeles(dossier_modeles: str | Path) -> Modeles:
    """Recharge les modèles vendorisés (voir 'artefacts.installer_modeles') : le modèle d'inclusion,
    les 10 modèles d'attribut, et leur méta-données (métriques de validation, vocabulaire figé)."""
    dossier_modeles = Path(dossier_modeles)
    meta = json.loads((dossier_modeles / "meta.json").read_text(encoding="utf-8"))
    return Modeles(
        meta=meta,
        spec=C.charger_spec(dossier_modeles / "feature_spec.json"),
        inclusion=lgb.Booster(model_file=str(dossier_modeles / "models" / "inclusion.txt")),
        attributs={cible: lgb.Booster(model_file=str(dossier_modeles / "models" / f"{cible}.txt")) for cible in L.CIBLES},
    )


def predire_candidats(brut: pd.DataFrame, modeles: Modeles) -> pd.DataFrame:
    """Probabilité d'inclusion pour TOUS les tronçons, attributs pour les CANDIDATS seulement (bien
    moins nombreux - voir PROBA_MIN_CANDIDAT).

    Renvoie une ligne par candidat : 'row' (position dans 'brut'), 'id_osm', 'p_incl', 'highway',
    'pred_<cible>' et 'conf_<cible>' pour chacune des 10 cibles (voir 'etiquettes.CIBLES')."""
    n = len(brut)
    proba = np.empty(n, dtype=np.float32)
    for debut in range(0, n, LIGNES_PAR_LOT):
        lot = brut.iloc[debut : debut + LIGNES_PAR_LOT].reset_index(drop=True)
        proba[debut : debut + LIGNES_PAR_LOT] = modeles.inclusion.predict(C.vers_matrice(lot, modeles.spec))
        print(f"  [application_modele] inclusion {min(debut + LIGNES_PAR_LOT, n):,} / {n:,}")
    lignes = np.flatnonzero(proba >= PROBA_MIN_CANDIDAT)
    print(
        f"[application_modele] {len(lignes):,} candidats (p >= {PROBA_MIN_CANDIDAT:.2f}) ; "
        f"{(proba >= modeles.seuil).sum():,} au-dessus du seuil d'inclusion {modeles.seuil:.3f}"
    )

    X = C.vers_matrice(brut.iloc[lignes].reset_index(drop=True), modeles.spec)
    sortie = pd.DataFrame({"row": lignes, "id_osm": brut["id_osm"].to_numpy()[lignes], "p_incl": proba[lignes], "highway": brut["highway"].astype(object).to_numpy()[lignes]})
    for cible in L.CIBLES:
        classes = np.array(modeles.meta["targets"][cible]["classes"])
        scores = modeles.attributs[cible].predict(X)
        sortie[f"pred_{cible}"] = classes[scores.argmax(1)]
        sortie[f"conf_{cible}"] = scores.max(1).astype(np.float32)
    return sortie
