"""Variables socio-économiques et touristiques de chaque commune.

    grille de densité   niveaux de densité INSEE : DENS/LIBDENS (3 niveaux), DENS7/LIBDENS7 (7
                        niveaux). Les arrondissements de Paris/Lyon/Marseille n'existent pas dans
                        cette grille officielle : on leur donne le niveau le plus élevé ("Grands
                        centres urbains"), puisqu'ils EN font clairement partie.
    niveau de vie       revenu médian par commune (INSEE, euros par unité de consommation)
    aménagements vélo   nombre d'équipements cyclables par type et par année, longueur de voirie,
                        taux de cyclabilité
    accueil vélo        nombre de sites labellisés "accueil vélo" (hébergement/restauration/visite)
                        par type et par année

Tout est joint (jointure à gauche) sur le code commune à 5 caractères : une commune absente d'une
source garde des valeurs manquantes, SAUF pour les comptages d'équipements (une commune absente du
fichier des aménagements cyclables n'en a simplement aucun -> 0, pas une valeur manquante).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arrondissements import ARRONDISSEMENTS

COLONNES_DENSITE = ["CODGEO", "DENS", "LIBDENS", "DENS7", "LIBDENS7"]
NIVEAU_LE_PLUS_DENSE = "Grands centres urbains"


def grille_de_densite(chemin_xlsx: Path, communes: pd.DataFrame) -> pd.DataFrame:
    """Niveau de densité INSEE de chaque commune ; les arrondissements PLM (absents de la grille
    officielle) reçoivent le niveau le plus dense car ils font partie des grands centres urbains."""
    grille = pd.read_excel(
        chemin_xlsx, sheet_name="Maille communale", header=4, usecols=COLONNES_DENSITE
    )
    grille["code_commune"] = grille["CODGEO"].astype(str).str.strip()
    fusion = communes[["code_commune"]].merge(
        grille.drop(columns="CODGEO"), on="code_commune", how="left"
    )
    fusion["LIBDENS"] = fusion["LIBDENS"].astype(object)
    est_arrondissement = fusion["code_commune"].isin(ARRONDISSEMENTS)
    fusion.loc[est_arrondissement, "LIBDENS7"] = NIVEAU_LE_PLUS_DENSE
    fusion.loc[est_arrondissement, "LIBDENS"] = "1"
    return fusion


def niveau_de_vie_median(chemin_csv: Path) -> pd.DataFrame:
    """Dernier niveau de vie médian connu de chaque commune."""
    table = pd.read_csv(chemin_csv, dtype={"code_com": str})
    table = table.sort_values("annee").drop_duplicates("code_com", keep="last")
    return table.rename(
        columns={
            "code_com": "code_commune",
            "valeur": "median_income",
            "annee": "median_income_year",
        }
    )[["code_commune", "median_income", "median_income_year"]]


def amenagements_cyclables(chemin_csv: Path) -> pd.DataFrame:
    """Aménagements cyclables par commune (comptages par type, 2021-2024, taux de cyclabilité, longueur de voirie)."""
    table = pd.read_csv(
        chemin_csv, sep=";", encoding="utf-8-sig", dtype={"INSEE_COM": str}
    )
    table["code_commune"] = table["INSEE_COM"].str.zfill(5)
    table = table.drop(columns=["INSEE_COM", "POP", "INSEE_EPCI"], errors="ignore")
    return table.rename(
        columns={c: "cyc_" + c.lower() for c in table.columns if c != "code_commune"}
    )


def accueil_velo(chemin_csv: Path) -> pd.DataFrame:
    """Sites labellisés "accueil vélo" par commune et par type (2022-2024)."""
    table = pd.read_csv(chemin_csv, dtype={"INSEE_COM": str})
    table["code_commune"] = table["INSEE_COM"].str.zfill(5)
    table = table.drop(columns="INSEE_COM")
    return table.rename(
        columns={c: "avel_" + c.lower() for c in table.columns if c != "code_commune"}
    )


def remplir_les_comptages_manquants(
    table: pd.DataFrame, prefixes: tuple[str, ...]
) -> pd.DataFrame:
    """Une commune absente d'une source de COMPTAGE d'équipements n'en a simplement aucun : les
    comptages manquants deviennent 0 (mais pas les taux/longueurs, qui restent manquants - on ne
    sait pas les déduire d'une absence)."""
    colonnes_comptage = [
        c
        for c in table.columns
        if c.startswith(prefixes) and "taux" not in c and "voirie" not in c
    ]
    table[colonnes_comptage] = table[colonnes_comptage].fillna(0)
    return table


def construire_table_socio_economique(
    dossier_brutes: Path, communes: pd.DataFrame
) -> pd.DataFrame:
    """Assemble toutes les sources socio-économiques en une seule table, une ligne par commune."""
    dossier = dossier_brutes / "socioeconomic"
    table = grille_de_densite(dossier / "density_grid.xlsx", communes)
    for source in (
        niveau_de_vie_median(dossier / "median_income.csv"),
        amenagements_cyclables(dossier / "cycle_infrastructure.csv"),
        accueil_velo(dossier / "bike_welcome.csv"),
    ):
        table = table.merge(source, on="code_commune", how="left")
    return remplir_les_comptages_manquants(table, ("avel_nb_",))


def preparer_socio_economique(
    dossier_brutes: Path, dossier_communes: Path, dossier_sortie: Path
) -> Path:
    """Construit la table socio-économique et l'écrit en parquet.

    'dossier_communes' doit contenir 'communes.parquet' (préparé par 'd_zones/zones.py')."""
    communes = pd.read_parquet(
        dossier_communes / "communes.parquet", columns=["code_commune"]
    )
    table = construire_table_socio_economique(dossier_brutes, communes)
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemin = dossier_sortie / "communes.parquet"
    table.to_parquet(chemin, index=False)
    print(
        f"[socio-éco] {len(table):,} communes x {table.shape[1]} colonnes -> {chemin}"
    )
    return chemin
