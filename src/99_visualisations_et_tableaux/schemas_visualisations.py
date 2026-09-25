# -*- coding: utf-8 -*-
"""Schémas Pydantic des tables produites par ce dossier - même principe que
'00_transformation_des_donnees/schemas.py'."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LigneIndicateurs(BaseModel):
    """UNE ligne de 'indicateurs.csv' : une année, tous les indicateurs de base au même niveau (voir
    'indicateurs.py'). Rappel : décrit NOTRE méthode d'estimation, pas le "vrai" BKT (voir le README)."""

    annee: int = Field(description="Année.")
    population: float = Field(ge=0, description="Population nationale utilisée cette année (étape 04).")
    n_capteurs_actifs: int = Field(ge=0, description="Nombre de capteurs actifs cette année (échantillon standard, grandit avec le temps).")
    n_capteurs_panel: int = Field(ge=0, description="Nombre de capteurs du panel équilibré (constant, présents toute la série).")
    lineaire_fob_km: float = Field(ge=0, description="Linéaire de réseau FOB total cette année-là (observé + non observé).")
    somme_debits: float = Field(ge=0, description="Somme des débits bruts (passages/an) de tous les capteurs actifs.")
    bkt_obs_Mdkm: float = Field(ge=0, description="BKT observé national, méthode standard (Md km).")
    bkt_ext_Mdkm: float = Field(ge=0, description="BKT extrapolé national, méthode standard (Md km) - le chiffre officiel.")
    bkt_mini_obs_Mdkm: float = Field(ge=0, description="BKT observé, panel équilibré seulement (Md km).")
    bkt_mini_ext_Mdkm: float = Field(ge=0, description="BKT extrapolé, panel équilibré seulement (Md km) - test de robustesse de la croissance.")


class LigneIntervalleConfiance(BaseModel):
    """UNE ligne de 'ic_standard.csv' ou 'ic_mini.csv' : intervalle de confiance bootstrap à 95 % du
    BKT extrapolé national d'UNE année - incertitude d'ÉCHANTILLONNAGE de notre méthode, pas une
    incertitude sur le vrai BKT (voir le README)."""

    annee: int = Field(description="Année.")
    mediane_bootstrap: float = Field(ge=0, description="Médiane des tirages bootstrap du BKT_ext national (Md km).")
    ic_bas_95: float = Field(ge=0, description="Borne basse de l'intervalle de confiance à 95 % (Md km).")
    ic_haut_95: float = Field(ge=0, description="Borne haute de l'intervalle de confiance à 95 % (Md km).")
    largeur_relative_pct: float = Field(ge=0, description="Largeur de l'IC en % de la médiane - plus c'est petit, plus l'estimation est précise.")


class LigneTestCroissance(BaseModel):
    """UNE ligne de 'test_croissance.csv' : test bootstrap apparié de la croissance du BKT_ext entre
    deux années - H0 = pas de croissance (ratio <= 1)."""

    transition: str = Field(description="Les deux années comparées, ex. '2019 -> 2020' ou '2019 -> 2025 (cumulé)'.")
    ratio_median: float = Field(gt=0, description="Ratio médian BKT_ext(fin) / BKT_ext(début) sur les tirages bootstrap appariés.")
    ic_bas_95: float = Field(gt=0, description="Borne basse à 95 % du ratio.")
    ic_haut_95: float = Field(gt=0, description="Borne haute à 95 % du ratio.")
    p_value_pas_de_croissance: float = Field(ge=0, le=1, description="Proportion des tirages compatibles avec H0 (pas de croissance).")
    significatif_5pct: str = Field(description="'oui' si p_value < 0.05, sinon 'non'.")
