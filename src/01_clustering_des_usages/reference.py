"""Le clustering d'usage K=4 DE RÉFÉRENCE : le run historique du projet (funFEM 'AkBk', K=4,
run "006"), vendorisé dans 'data/donnees_brutes/reference/station_clusters_k4_run006.csv' - PAS un
modèle qu'on réajuste, juste une table de labels déjà connus et éprouvés pour ~1593 capteurs.

**Utilisé PAR DÉFAUT** (voir 'pipeline.py', paramètre 'SOURCE_MODELE = "reference"') : ni recherche
Optuna, ni ajustement funFEM, juste une jointure - le plus simple et le plus rapide des trois chemins
possibles de ce dossier. Changez 'SOURCE_MODELE' (dans 'pipeline.py' ou le notebook) pour ajuster un
nouveau modèle à la place ('modele_classique.py' ou 'recherche_hyperparametres.py').

Bonus de ce choix : contrairement à un modèle fraîchement ajusté (dont on ne sait pas a priori ce que
représente chaque numéro de cluster), CE run est celui dont le projet a déjà établi la signification de
chaque cluster - voir CLUSTER_NAMES/CLUSTER_COULEURS plus bas.

**Limite assumée** : cette table ne couvre que les ~1593 capteurs actifs à l'année du run (2024) - un
capteur entré dans le réseau depuis, ou une année différente, n'y figure pas ('assigner_clusters_reference'
l'écarte silencieusement, voir sa docstring - pas une erreur, juste moins de capteurs affichés)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# la signification de chaque cluster POUR CE RUN précisément (voir VV5/src/bkt/commun/config.py -
# CLUSTER_NAMES/cartes/couleurs.py, mêmes identifiants numériques, mêmes noms/couleurs).
CLUSTER_NAMES: dict[int, str] = {
    0: "Péri-urbain loisirs",
    1: "Domicile travail",
    2: "Tourisme aménagé",
    3: "Tourisme rural",
}
CLUSTER_COULEURS: dict[int, str] = {
    0: "#2a78d6",
    1: "#eb6834",
    2: "#eda100",
    3: "#1baf7a",
}


def charger_reference(fichier: Path | str) -> pd.Series:
    """'id_site -> cluster' du run historique (voir la docstring du module)."""
    table = pd.read_csv(fichier)
    return pd.Series(
        table["cluster"].astype(int).to_numpy(),
        index=table["Identifiant_Site"].astype(int).to_numpy(),
        name="cluster",
    )


def assigner_clusters_reference(
    profils: pd.DataFrame, fichier: Path | str
) -> pd.DataFrame:
    """Version "reference" de 'appliquer_modele.assigner_clusters' : pas de projection funFEM, juste
    une jointure sur les labels déjà connus. Les capteurs de 'profils' absents du run (voir la
    limite dans la docstring du module) sont écartés, pas gardés avec un cluster manquant.

    Renvoie 'id_site, cluster' - même forme que 'appliquer_modele.assigner_clusters'."""
    labels = charger_reference(fichier)
    fusion = profils[["id_site"]].copy()
    fusion["cluster"] = fusion["id_site"].map(labels)
    trouves = fusion.dropna(subset=["cluster"]).copy()
    trouves["cluster"] = trouves["cluster"].astype(int)
    print(
        f"[reference] {len(trouves):,}/{len(profils):,} capteurs avec un cluster de référence (run historique 2024)"
    )
    return trouves.reset_index(drop=True)
