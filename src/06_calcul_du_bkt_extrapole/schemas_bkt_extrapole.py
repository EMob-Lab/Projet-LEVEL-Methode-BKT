"""Schémas Pydantic des tables produites par ce dossier - même principe que
'00_transformation_des_donnees/schemas.py'."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LigneDetailClusterAnnee(BaseModel):
    """UNE ligne de 'bkt_final_detail_par_cluster.csv' : un (année, cluster) - voir 'cadre_cluster' et
    'estimer' dans 'bkt_extrapole.py'."""

    annee: int = Field(description="Année du BKT.")
    cluster: int = Field(ge=0, le=3, description="Identifiant du cluster d'usage K=4.")
    cluster_nom: str = Field(
        description="Nom lisible du cluster (voir bkt_extrapole.CLUSTER_NAMES)."
    )
    n_communes_actives: int = Field(
        ge=0,
        description="Nombre de (commune, cluster) actifs cette année-là dans ce cluster (Z complet - voir le docstring du module).",
    )
    l_obs: float = Field(ge=0, description="Linéaire attribué à ce cluster, sur les (commune, cluster) actifs seulement (km).")
    l_total: float = Field(ge=0, description="Linéaire TOTAL attribué à ce cluster, actif ou non (km).")
    taux_fréquentation: float = Field(
        ge=0, description="Taux rogné (bike-km / km de réseau) de ce cluster - voir LOOCV_OPTIMAL_TRIM."
    )
    bkt_obs: float = Field(
        ge=0,
        description="BKT observé du cluster (somme des (commune, cluster) actifs cette année-là).",
    )
    bkt_ext: float = Field(
        ge=0,
        description="BKT extrapolé du cluster (observé + taux_fréquentation x linéaire non observé).",
    )


class LigneNationale(BaseModel):
    """UNE ligne de 'bkt_final_national.csv' : le BKT national d'UNE année, en milliards de km (somme
    des 4 clusters)."""

    annee: int = Field(description="Année du BKT.")
    bkt_obs_Mdkm: float = Field(
        ge=0, description="BKT observé national (milliards de km)."
    )
    bkt_ext_Mdkm: float = Field(
        ge=0,
        description="BKT extrapolé national (milliards de km) - le chiffre final du BKT.",
    )
