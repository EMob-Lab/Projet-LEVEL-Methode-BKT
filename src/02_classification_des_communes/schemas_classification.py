# -*- coding: utf-8 -*-
"""Schémas Pydantic des données produites par ce dossier - même principe que
'00_transformation_des_donnees/schemas.py' (documentation vivante, pas du code utilisé par le
pipeline) et 'schemas_clusters.py' de l'étape 01 (dont le nom de fichier suit la même logique : un nom
distinct par étape, pour ne jamais avoir deux 'schemas.py' importés en même temps dans le même script)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LigneClassificationCommune(BaseModel):
    """UNE ligne de 'commune_clusters.parquet' (sortie de 'pipeline.py') : le cluster d'usage PRÉDIT
    d'UNE commune - qu'elle héberge un capteur ou non. Les colonnes 'proba_cluster_*' sont dynamiques
    (une par cluster du modèle, K variable selon celui choisi à l'étape 01) - ce modèle autorise donc
    les clés supplémentaires ('extra="allow"') plutôt que de les lister une par une."""

    model_config = ConfigDict(extra="allow")

    code_commune: str = Field(description="Code INSEE de la commune (5 caractères).")
    nom_commune: str = Field(description="Nom de la commune.")
    cluster_predit: int = Field(ge=0, description="Cluster d'usage prédit par la forêt aléatoire (numérotation propre au modèle - voir son 'config.json' à l'étape 01).")
    extrapolee: int = Field(ge=0, le=1, description="1 si aucun capteur de cette commune n'a servi à l'entraînement (le cluster est une pure prédiction) ; 0 si la commune a fourni au moins un capteur étiqueté (une observation directe).")
