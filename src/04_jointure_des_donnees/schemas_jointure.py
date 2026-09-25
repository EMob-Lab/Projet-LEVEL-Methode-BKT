"""Schémas Pydantic des trois tables assemblées par ce dossier - même principe que
'00_transformation_des_donnees/schemas.py' (documentation vivante, pas du code utilisé par le
pipeline). Nom de fichier distinct ('schemas_jointure.py') pour la même raison qu'aux étapes 01/02/03 -
éviter un conflit d'import si plusieurs schémas sont utilisés ensemble dans le même script."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LigneCommuneAssemblee(BaseModel):
    """UNE ligne de 'communes.parquet' (sortie de 'pipeline.py') : UNE commune, POUR UNE année - gardée
    seulement si elle a du réseau dans au moins un scénario (voir 'assemblage.assembler')."""

    annee: int = Field(description="Année du BKT (2019-2025).")
    code_commune: str = Field(description="Code INSEE de la commune (5 caractères).")
    cluster_K4: float | None = Field(
        default=None,
        description=(
            "Cluster d'usage K=4 retenu pour cette (année, commune) : celui de ses capteurs actifs "
            "quand elle en a (débit total le plus fort, voir 'entrees.resoudre_cluster_communes'), "
            "sinon le cluster prédit à l'étape 02 - peut être manquant."
        ),
    )
    population: float = Field(
        ge=0,
        description="Population de la commune au millésime retenu pour cette année (voir 'entrees.millesime_population').",
    )
    len_d_GV: float = Field(
        ge=0, description="Longueur effective (km) côté droit, scénario GéoVélo seul."
    )
    len_g_GV: float = Field(
        ge=0, description="Longueur effective (km) côté gauche, scénario GéoVélo seul."
    )
    len_d_FOB: float = Field(
        ge=0, description="Longueur effective (km) côté droit, scénario FOB (retenu)."
    )
    len_g_FOB: float = Field(
        ge=0, description="Longueur effective (km) côté gauche, scénario FOB (retenu)."
    )
    len_d_FOB_AM: float = Field(
        ge=0,
        description="Longueur effective (km) côté droit, scénario FOB_AM (aménagements dédiés seulement).",
    )
    len_g_FOB_AM: float = Field(
        ge=0, description="Longueur effective (km) côté gauche, scénario FOB_AM."
    )
    instrumentee: int = Field(
        description="1 si la commune héberge au moins un capteur actif une année quelconque, sinon 0."
    )
    nom_commune: str = Field(description="Nom de la commune.")
    code_departement: str = Field(description="Code INSEE du département.")


class LigneCapteurAssemble(BaseModel):
    """UNE ligne de 'sensor_detail.parquet' (sortie de 'pipeline.py') : UN capteur ACTIF, POUR UNE
    année - sortie directe de 'entrees.capteurs_actifs' (étape 00 + cluster de l'étape 01), aucun
    calcul supplémentaire ici. Voir '05_calcul_du_bkt_observe' pour comment 'qta' devient le BKT observé."""

    id_site: int = Field(description="Identifiant du capteur.")
    nom_site: str = Field(description="Nom du capteur.")
    id_commune_str: str = Field(description="Code INSEE de la commune du capteur.")
    annee: int = Field(description="Année du BKT.")
    annual_flow: float = Field(ge=0, description="Somme des comptages horaires valides de l'année.")
    qta: float = Field(ge=0, description="Quantité de Trafic Annuel - estimation du nombre total de passages sur l'année (voir '00_transformation_des_donnees/d_debit/capteurs_annees.py').")
    cluster_K4: float | None = Field(default=None, description="Cluster d'usage K=4 du capteur (étape 01, référence) - peut être manquant.")


class LignePopulationUtilisee(BaseModel):
    """UNE ligne de 'population_used.parquet' : quel millésime de population a été utilisé pour une
    année du BKT, et pourquoi (voir 'entrees.millesime_population') - table de contrôle, pas une entrée
    du calcul lui-même."""

    annee: int = Field(description="Année du BKT.")
    millesime_utilise: int = Field(
        description="Millésime de population INSEE réellement utilisé pour cette année."
    )
    regle: str = Field(
        description="Pourquoi ce millésime : 'année', 'plus proche après', ou 'plus proche avant'."
    )
    population_totale: float = Field(
        ge=0,
        description="Population totale de ce millésime, toutes communes confondues (contrôle).",
    )
    n_communes: int = Field(
        ge=0,
        description="Nombre de communes ayant une population à ce millésime (contrôle).",
    )
