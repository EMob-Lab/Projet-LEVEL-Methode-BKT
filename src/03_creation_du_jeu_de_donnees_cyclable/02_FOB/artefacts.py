# -*- coding: utf-8 -*-
"""Les modèles FOB DÉJÀ ENTRAÎNÉS, vendorisés dans 'models/fob/' (racine du dépôt) : le résultat d'un
entraînement qui a besoin des exports GéoVélo bruts (voir 'entrainement.py') - pas fournis avec ce
dépôt (source tierce volumineuse, même remarque que pour les autres sources brutes, voir
'00_transformation_des_donnees/README.md').

Ce module ne fait qu'une chose : vérifier que ces modèles sont bien là et vendre leurs métriques de
validation (déjà calculées à l'entraînement, dans 'meta.json') - PAS les réentraîner. Voir
'application_modele.charger_modeles' pour les charger réellement."""

from __future__ import annotations

import json
from pathlib import Path

FICHIERS_ATTENDUS = ("meta.json", "feature_spec.json")


def verifier_modeles(dossier_modeles: Path) -> dict:
    """Vérifie que 'dossier_modeles' contient bien des modèles FOB vendorisés (meta.json,
    feature_spec.json, et un fichier '.txt' LightGBM par cible - voir 'etiquettes.CIBLES' + le modèle
    d'inclusion) et renvoie leurs métriques de validation."""
    manquants = [f for f in FICHIERS_ATTENDUS if not (dossier_modeles / f).exists()]
    if manquants or not (dossier_modeles / "models").is_dir():
        raise FileNotFoundError(
            f"modèles FOB introuvables sous {dossier_modeles} ({manquants or 'dossier models/ absent'}) - "
            "ils doivent être vendorisés (voir le README de ce dossier), pas réentraînés à la volée."
        )
    meta = json.loads((dossier_modeles / "meta.json").read_text(encoding="utf-8"))
    return meta


def resume_qualite(meta: dict) -> str:
    """Un résumé lisible des métriques de validation du modèle d'inclusion et des modèles d'attribut -
    pour vérifier d'un coup d'œil qu'on n'utilise pas un modèle dégradé (voir le notebook, section 0)."""
    incl = meta["inclusion"]
    lignes = [f"inclusion : précision={incl['precision']:.3f}, rappel={incl['recall']:.3f}, F1={incl['f1']:.3f} (seuil={incl['threshold']:.3f})"]
    for cible, valeurs in meta["targets"].items():
        lignes.append(f"{cible:10s} : accuracy={valeurs['accuracy']:.3f}, accuracy équilibrée={valeurs['balanced_accuracy']:.3f}")
    return "\n".join(lignes)
