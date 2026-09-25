"""Schémas Pydantic des tables produites par ce dossier - même principe que
'00_transformation_des_donnees/schemas.py'. Nom de fichier distinct pour la même raison qu'aux étapes
précédentes - éviter un conflit d'import si plusieurs schémas sont utilisés ensemble."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LigneCommuneClusterBktObserve(BaseModel):
    """UNE ligne de 'communes_bkt_obs.parquet' (sortie de 'pipeline.py') : UN (année, commune, cluster)
    ayant du linéaire attribué - plusieurs lignes pour une commune dont les capteurs actifs
    appartiennent à plusieurs clusters, une seule sinon (voir 'bkt_observe.py')."""

    annee: int = Field(description="Année du BKT.")
    code_commune: str = Field(description="Code INSEE de la commune.")
    nom_commune: str = Field(description="Nom de la commune.")
    code_departement: str = Field(description="Code INSEE du département.")
    instrumentee: int = Field(
        description="1 si la commune héberge au moins un capteur actif une année quelconque, sinon 0."
    )
    cluster_K4: float = Field(description="Cluster de cette ligne (celui des capteurs qui la composent, ou le repli étape 02 si la commune n'a pas de capteur actif cette année-là).")
    total_length_km: float = Field(ge=0, description="Linéaire TOTAL de la commune (référence, non réparti).")
    part_lineaire: float = Field(
        ge=0, le=1, description="Part de 'total_length_km' attribuée à ce cluster (débit total du cluster / débit total de la commune ; 1.0 si un seul cluster présent)."
    )
    lineaire_km: float = Field(ge=0, description="total_length_km x part_lineaire - le linéaire réellement attribué à ce cluster.")
    n_capteurs_actifs: int = Field(ge=0, description="Nombre de capteurs actifs de ce cluster dans la commune cette année-là.")
    qta_moyen: float = Field(
        ge=0,
        description="Moyenne simple de la QTA (Quantité de Trafic Annuel) des capteurs actifs DE CE CLUSTER dans la commune (0 si aucun).",
    )
    bkt_obs: float = Field(ge=0, description="BKT observé de ce (commune, cluster) : qta_moyen x lineaire_km.")
