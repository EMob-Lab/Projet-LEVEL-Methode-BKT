"""Charge/sauvegarde/applique un modèle funFEM déjà entraîné (dossier
'models/clustering_des_debits/<identifiant>/' - écrit par 'recherche_hyperparametres.py' OU par
'modele_classique.py', même format des deux côtés, voir 'sauvegarder_modele') à de NOUVEAUX profils de
capteurs, sans refaire tourner l'algorithme EM - juste la projection sur les axes discriminants déjà
appris (la méthode 'predict' de 'FunFEM.funFEM', voir sa docstring pour le détail mathématique).

C'est la fonction que le reste du projet (carte de référence, forêt aléatoire des communes...) doit
utiliser pour retrouver le cluster d'un capteur : 'recherche_optuna'/'modele_classique' servent à
CHOISIR/mettre à jour un modèle, ce module sert à s'en SERVIR - les deux usages sont volontairement
séparés.

explication : pourquoi la représentation compte-t-elle aussi ?

funFEM ne classe pas les profils bruts, mais leur REPRÉSENTATION - '(X, W)', voir 'representation.py'
(soit multi-échelle : jour+semaine+année+spectral, soit hebdomadaire seule, voir 'REPRESENTATIONS'
plus bas) - la même transformation (mêmes tailles de base, mêmes poids) que celle utilisée pour
ENTRAÎNER le modèle doit être reproduite pour de nouveaux capteurs, sinon la projection n'a pas de
sens. C'est pour ça que 'config.json' garde ces hyperparamètres ET quelle représentation les
interpréter avec (clé 'representation') : ce module relit les deux pour reconstruire la même
transformation avant d'appeler 'predict'.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from representation import representation_hebdomadaire, representation_multiechelle

# une représentation = (la fonction qui construit (X, W), les clés de config.json à lui passer en kwargs)
REPRESENTATIONS = {
    "multiechelle": (
        representation_multiechelle,
        (
            "n_base_jour",
            "n_base_semaine",
            "n_base_annee",
            "poids_jour",
            "poids_semaine",
            "poids_annee",
            "poids_spectral",
        ),
    ),
    "hebdomadaire": (representation_hebdomadaire, ("n_base_semaine", "poids_semaine")),
}


def charger_modele(dossier_modele: Path | str) -> tuple[object, dict]:
    """Recharge le modèle funFEM entraîné ('modele.joblib') et sa configuration
    ('config.json' : hyperparamètres de représentation, K, critère...) depuis un dossier écrit par
    'sauvegarder_modele'."""
    dossier_modele = Path(dossier_modele)
    modele = joblib.load(dossier_modele / "modele.joblib")
    config = json.loads((dossier_modele / "config.json").read_text(encoding="utf-8"))
    return modele, config


def sauvegarder_modele(
    dossier: Path,
    modele,
    config: dict,
    profils: pd.DataFrame,
    *,
    donnees_capteurs: pd.DataFrame | None = None,
    couleurs: dict | None = None,
    noms_clusters: dict | None = None,
) -> None:
    """Écrit UN modèle entraîné dans 'dossier' - le même format quelle que soit son origine
    ('recherche_hyperparametres.py' ou 'modele_classique.py') :

        config.json               'config' - DOIT contenir 'representation' (voir REPRESENTATIONS)
                                   et les hyperparamètres qu'elle attend, en plus de 'k'/'modele'.
        modele.joblib              l'objet 'FunFEM.funFEM' entraîné
        assignation_clusters.csv   'id_site, cluster' pour tous les capteurs de 'profils'
        carte.html                 la carte interactive de ce modèle - seulement SI 'donnees_capteurs'
                                   est fourni (coordonnées des capteurs, voir 'carte.carte_capteurs')
    """
    if "representation" not in config:
        raise ValueError(
            "config doit contenir la clé 'representation' (voir REPRESENTATIONS)"
        )
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(modele, dossier / "modele.joblib")

    assignation = pd.DataFrame(
        {"id_site": profils["id_site"].to_numpy(), "cluster": modele.cls}
    )
    assignation.to_csv(
        dossier / "assignation_clusters.csv", index=False, encoding="utf-8"
    )

    if donnees_capteurs is not None:
        from carte import carte_capteurs

        fusion = profils.merge(assignation, on="id_site").merge(
            donnees_capteurs, on="id_site", how="inner"
        )
        carte_capteurs(
            fusion,
            couleurs=couleurs,
            noms_clusters=noms_clusters,
            sortie=dossier / "carte.html",
        )


def assigner_clusters(
    profils: pd.DataFrame, dossier_modele: Path | str
) -> pd.DataFrame:
    """Attribue à chaque capteur de 'profils' le cluster du modèle sauvegardé dans 'dossier_modele'.

    'profils' : une ligne par capteur, avec 'id_site' et les colonnes de profil ('d00..d23' etc.,
    voir 'representation.py' / '00_transformation_des_donnees/d_debit/profils_usage.py') - PAS
    forcément les mêmes capteurs que ceux utilisés à l'entraînement (c'est tout l'intérêt de
    'predict' : classer des capteurs jamais vus, ou reclasser une année différente).

    Renvoie 'id_site, cluster' (un entier, dans la numérotation propre à CE modèle - voir
    'FunFEM.funFEM.predict').
    """
    dossier_modele = Path(dossier_modele)
    modele, config = charger_modele(dossier_modele)
    construire_representation, cles = REPRESENTATIONS[config["representation"]]
    hp = {cle: config[cle] for cle in cles}

    # étape 1 : reconstruit EXACTEMENT la même représentation que celle utilisée à l'entraînement (même
    # fonction, mêmes hyperparamètres, relus depuis config.json) - seul W change de taille avec le
    # nombre de capteurs, la transformation (tailles de base, poids) doit rester identique.
    X, _ = construire_representation(profils, **hp)

    # étape 2 : projection sur les axes discriminants déjà appris (pas de ré-estimation, voir
    # FunFEM.funFEM.predict/predict_proba) - rapide, contrairement à un ajustement complet.
    clusters = modele.predict(X)

    print(
        f"[assignation] modèle {dossier_modele.name} (K={config['k']}, {config['modele']}, représentation={config['representation']}) appliqué à {len(profils):,} capteurs"
    )
    return pd.DataFrame({"id_site": profils["id_site"].to_numpy(), "cluster": clusters})
