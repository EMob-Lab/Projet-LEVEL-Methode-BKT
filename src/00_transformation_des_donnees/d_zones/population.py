"""Population municipale INSEE, un tableau par millésime (année de recensement publiée), rassemblés
en une seule table longue.

Chaque millésime est un classeur Excel INSEE (feuille "Communes", les données commencent après 7
lignes d'en-tête). Le résultat a une ligne par (millésime, commune), avec la population totale.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def lire_un_millesime(chemin_xlsx: Path, millesime: int) -> pd.DataFrame:
    """Population de chaque commune pour UN millésime (le code commune = code département + numéro de
    commune sur 3 chiffres, collés)."""
    df = pd.read_excel(chemin_xlsx, sheet_name="Communes", header=7)
    departement = df["Code département"].astype(str).str.strip().str.zfill(2)
    commune = df["Code commune"].astype(str).str.strip().str.zfill(3)
    return pd.DataFrame(
        {
            "millesime": millesime,
            "code_commune": (departement + commune).to_numpy(),
            "population": df["Population totale"].astype(float).to_numpy(),
        }
    )


def preparer_population(
    dossier_brutes: Path, dossier_sortie: Path, millesimes: tuple[int, ...]
) -> Path:
    """Empile tous les millésimes demandés en une seule table et l'écrit en parquet.

    Attend un fichier 'population/population_{millesime}.xlsx' par millésime dans 'dossier_brutes'."""
    table = pd.concat(
        [
            lire_un_millesime(dossier_brutes / "population" / f"population_{m}.xlsx", m)
            for m in millesimes
        ],
        ignore_index=True,
    )
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemin = dossier_sortie / "population.parquet"
    table.to_parquet(chemin, index=False)
    print(
        f"[population] {len(millesimes)} millésimes, {len(table):,} lignes -> {chemin}"
    )
    return chemin


def charger_population(dossier_sortie: Path, millesime: int) -> pd.Series:
    """Relit la table déjà préparée et renvoie la population d'UN millésime, indexée par code commune."""
    table = pd.read_parquet(dossier_sortie / "population.parquet")
    lignes = table[table["millesime"] == millesime]
    return pd.Series(
        lignes["population"].to_numpy(),
        index=lignes["code_commune"].to_numpy(),
        name="population",
    )
