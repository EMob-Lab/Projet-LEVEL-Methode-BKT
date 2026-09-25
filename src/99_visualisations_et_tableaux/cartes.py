# -*- coding: utf-8 -*-
"""Cartes de communes du pipeline BKT (voir aussi '01_clustering_des_usages/carte.py' pour la carte
INTERACTIVE des capteurs par cluster - celle-ci ne la refait pas, elle la complète avec deux cartes au
niveau COMMUNE, construites à partir de la table assemblée de l'étape 04) :

    couverture_communes    chaque commune classée en Instrumentée / Linéaire seulement / Sans rien
    clusters_communes      cluster d'usage K4 de chaque commune AYANT du linéaire cyclable

Les deux acceptent 'html=True' (carte interactive, voir 'rendu.carte_html') ou 'html=False' (PNG
statique, voir 'rendu.carte_png') - même rendu partagé, pour rester visuellement cohérentes."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

import rendu

SCENARIO = "FOB"
CATEGORIES_COUVERTURE = {"Instrumentée": "#2a78d6", "Linéaire seulement": "#1baf7a", "Sans rien": "#e2e2e2"}

# Mêmes identifiants, noms et couleurs de cluster que partout ailleurs dans ce dépôt (voir
# '01_clustering_des_usages/reference.py') - dupliqués ici (petites constantes) plutôt qu'importés
# d'un autre dossier, pour que ce module reste lisible seul.
CLUSTER_NAMES: dict[int, str] = {0: "Péri-urbain loisirs", 1: "Domicile travail", 2: "Tourisme aménagé", 3: "Tourisme rural"}
CLUSTER_COULEURS: dict[int, str] = {0: "#2a78d6", 1: "#eb6834", 2: "#eda100", 3: "#1baf7a"}


def communes_categorisees(fichier_communes_zones: Path, fichier_communes_assemblees: Path, annee: int) -> gpd.GeoDataFrame:
    """Le fond de carte des communes (étape 00) enrichi de la catégorie de couverture de 'annee' (étape
    04) - une commune absente de la table assemblée (jamais de linéaire d'aucun scénario) est
    catégorisée "Sans rien" par défaut, pas une erreur."""
    geo = gpd.read_parquet(fichier_communes_zones)[["code_commune", "nom_commune", "geometry"]]
    attrs = pd.read_parquet(fichier_communes_assemblees)
    attrs = attrs[attrs["annee"] == annee].copy()
    attrs["total_length_km"] = attrs[f"len_d_{SCENARIO}"] + attrs[f"len_g_{SCENARIO}"]
    fusion = geo.merge(attrs[["code_commune", "instrumentee", "total_length_km"]], on="code_commune", how="left")
    fusion["total_length_km"] = fusion["total_length_km"].fillna(0.0)
    fusion["instrumentee"] = fusion["instrumentee"].fillna(0).astype(int)
    fusion["categorie"] = "Sans rien"
    fusion.loc[fusion["total_length_km"] > 0, "categorie"] = "Linéaire seulement"
    fusion.loc[fusion["instrumentee"] == 1, "categorie"] = "Instrumentée"
    return fusion


def carte_couverture_communes(
    fichier_communes_zones: Path,
    fichier_communes_assemblees: Path,
    *,
    annee: int,
    html: bool = False,
    sortie: Path | str | None = None,
    dpi: int = 200,
    bordures: rendu.Bordures = "neutres",
    simplification_m: float = 200.0,
) -> Path:
    """'html=True' : carte interactive (survol = nom + catégorie, fond de carte au choix). Sinon : PNG
    statique à 'dpi'. 'simplification_m' (HTML seulement) : tolérance de simplification des contours en
    mètres, voir 'rendu.carte_html'."""
    gdf = communes_categorisees(fichier_communes_zones, fichier_communes_assemblees, annee)
    sortie = Path(sortie) if sortie else Path(f"carte_couverture_communes_{annee}.{'html' if html else 'png'}")
    titre = f"Communes : instrumentées, avec linéaire, ou sans rien ({annee})"
    fn = rendu.carte_html if html else rendu.carte_png
    kwargs = {"popup_cols": ["nom_commune"], "simplification_m": simplification_m} if html else {"dpi": dpi}
    chemin = fn(gdf, categorie_col="categorie", couleurs=CATEGORIES_COUVERTURE, titre=titre, sortie=sortie, bordures=bordures, **kwargs)
    print(f"écrit : {chemin}  ({len(gdf):,} communes : {gdf['categorie'].value_counts().to_dict()})")
    return chemin


def communes_avec_cluster(fichier_communes_zones: Path, fichier_communes_assemblees: Path, annee: int) -> gpd.GeoDataFrame:
    """Les communes avec du linéaire cyclable (scénario retenu) cette année-là, fond de carte de
    l'étape 00 + cluster K4 déjà résolu par l'étape 04 (celui de ses capteurs actifs quand elle en a,
    sinon le cluster prédit à l'étape 02 - voir '04_jointure_des_donnees/entrees.resoudre_cluster_communes')."""
    geo = gpd.read_parquet(fichier_communes_zones)[["code_commune", "nom_commune", "geometry"]]
    attrs = pd.read_parquet(fichier_communes_assemblees)
    attrs = attrs[attrs["annee"] == annee].copy()
    attrs["total_length_km"] = attrs[f"len_d_{SCENARIO}"] + attrs[f"len_g_{SCENARIO}"]
    avec_lineaire = attrs.loc[attrs["total_length_km"] > 0, ["code_commune", "total_length_km", "cluster_K4"]]

    fusion = geo.merge(avec_lineaire, on="code_commune", how="inner")
    fusion["cluster_nom"] = fusion["cluster_K4"].map(CLUSTER_NAMES)
    manquants = fusion["cluster_nom"].isna().sum()
    if manquants:
        fusion["cluster_nom"] = fusion["cluster_nom"].fillna("Cluster inconnu")  # garde-fou honnête (commune sans cluster prédit) plutôt qu'un crash
    return fusion


def carte_clusters_communes(
    fichier_communes_zones: Path,
    fichier_communes_assemblees: Path,
    *,
    annee: int,
    html: bool = False,
    sortie: Path | str | None = None,
    dpi: int = 200,
    bordures: rendu.Bordures = "neutres",
    simplification_m: float = 200.0,
) -> Path:
    """'html=True' : carte interactive (survol = nom + cluster + linéaire, fond de carte au choix).
    Sinon : PNG statique à 'dpi'. Couleurs de cluster : les mêmes que partout ailleurs dans ce dépôt
    (voir '01_clustering_des_usages/reference.py', CLUSTER_COULEURS)."""
    gdf = communes_avec_cluster(fichier_communes_zones, fichier_communes_assemblees, annee)
    couleurs = {CLUSTER_NAMES[i]: CLUSTER_COULEURS[i] for i in sorted(CLUSTER_NAMES)}
    sortie = Path(sortie) if sortie else Path(f"carte_clusters_communes_{annee}.{'html' if html else 'png'}")
    titre = f"Cluster des communes avec linéaire cyclable ({annee})"
    fn = rendu.carte_html if html else rendu.carte_png
    kwargs = {"popup_cols": ["nom_commune", "total_length_km"], "simplification_m": simplification_m} if html else {"dpi": dpi}
    chemin = fn(gdf, categorie_col="cluster_nom", couleurs=couleurs, titre=titre, sortie=sortie, bordures=bordures, **kwargs)
    print(f"écrit : {chemin}  ({len(gdf):,} communes avec linéaire : {gdf['cluster_nom'].value_counts().to_dict()})")
    return chemin
