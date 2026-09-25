# -*- coding: utf-8 -*-
"""Schémas Pydantic des tables produites par ce dossier - même principe que
'00_transformation_des_donnees/schemas.py' (documentation vivante, pas du code utilisé par le
pipeline) : chaque champ porte une description. Nom de fichier distinct ('schemas_fob.py', pas
'schemas.py') pour la même raison qu'aux étapes 01/02 - éviter un conflit d'import si plusieurs schémas
sont utilisés ensemble dans le même script."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LigneTronconFOB(BaseModel):
    """UNE ligne de 'ways.parquet' (sortie de 'snapshot.py', un fichier par année) : UN tronçon OSM
    retenu dans le réseau FOB, avec ses 10 attributs prédits (voir 'etiquettes.CIBLES') et sa
    localisation. Les colonnes 'pred_*'/'conf_*' dynamiques (une paire par cible) ne sont pas listées
    une par une ici (clés supplémentaires autorisées, 'extra=\"allow\"')."""

    model_config = ConfigDict(extra="allow")

    id_osm: int = Field(description="Identifiant du tronçon (\"way\") OpenStreetMap.")
    p_incl: float = Field(ge=0, le=1, description="Probabilité prédite d'appartenir au réseau cyclable FOB (voir 'application_modele.py').")
    incl: bool = Field(description="True : ce tronçon appartient au réseau FOB retenu (p_incl au-dessus du seuil de décision du modèle).")
    highway: str = Field(description="Type de voie OSM ('cycleway', 'residential', 'path'...).")
    length_m: float | None = Field(default=None, ge=0, description="Longueur OSM du tronçon cette année-là (mètres) - None si sa géométrie n'a pas pu être localisée.")
    lat: float | None = Field(default=None, description="Latitude du point médian localisé du tronçon.")
    lon: float | None = Field(default=None, description="Longitude du point médian localisé du tronçon.")
    pred_ame_d: str = Field(description="Type d'aménagement prédit, côté droit (voir 'etiquettes.HIERARCHIE').")
    pred_ame_g: str = Field(description="Type d'aménagement prédit, côté gauche.")
    pred_sens_d: str = Field(description="Sens de circulation prédit, côté droit ('BIDIRECTIONNEL' ou non - voir 'etiquettes.multiplicateur_direction').")
    pred_sens_g: str = Field(description="Sens de circulation prédit, côté gauche.")


class LigneLongueurCommune(BaseModel):
    """UNE ligne de 'longueurs_fob.parquet' ou 'longueurs_fob_amenagements.parquet' (sortie de
    'longueurs.py') : la longueur EFFECTIVE (un côté bidirectionnel compte double) du réseau FOB d'UNE
    commune, pour UNE année."""

    code_commune: str = Field(description="Code INSEE de la commune (5 caractères).")
    len_d: float = Field(ge=0, description="Longueur effective côté droit (km).")
    len_g: float = Field(ge=0, description="Longueur effective côté gauche (km).")


class LigneMorceauDecoupe(BaseModel):
    """UNE ligne de 'decoupe/pieces.parquet' (sortie de 'decoupe_communes.decouper_troncons') : la
    part d'UN tronçon FOB qui tombe dans UNE commune (un tronçon traversant N communes donne N lignes)."""

    id_osm: int = Field(description="Identifiant du tronçon OpenStreetMap découpé.")
    code_commune: str = Field(description="Code INSEE de la commune qui reçoit ce morceau.")
    length_m: float = Field(ge=0, description="Longueur (mètres) de CE morceau (= part x longueur OSM du tronçon cette année-là).")
    part: float = Field(ge=0, le=1, description="Fraction de la géométrie du tronçon tombant dans cette commune.")
    source: str = Field(description="'decoupe' (intersection géométrique réelle) ou 'point_median' (repli : tronçon sans géométrie localisée).")
