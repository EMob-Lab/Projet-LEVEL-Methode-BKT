"""Débits horaires des capteurs vélo : convertit les exports Excel bruts annuels (un fichier par
année, ~5000 colonnes) en un format compact (msgpack), un fichier par année.

Chaque classeur brut a 3 lignes d'en-tête (identifiant du site, identifiant du "débit", nom du débit)
puis une ligne par heure de l'année. Chaque colonne après la première est un DÉBIT : un site peut en
avoir plusieurs (un sens de circulation = un débit, une voie = un débit...). Les valeurs sont
stockées en entier compact (voir 'outils.encoder_debit' / 'decoder_debit').
"""

from __future__ import annotations

import sys
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outils import encoder_debit

NB_LIGNES_ENTETE = 3


def lire_excel_brut(chemin_xlsx: Path) -> pd.DataFrame:
    """Lit le classeur brut d'une année. Utilise le moteur "calamine" (beaucoup plus rapide qu'openpyxl
    sur un fichier aussi large - environ 30 secondes contre plusieurs minutes)."""
    return pd.read_excel(chemin_xlsx, engine="calamine")


def encoder_une_annee(brut: pd.DataFrame, annee: int) -> dict:
    """Transforme la feuille brute d'une année en un dictionnaire prêt à être sauvegardé (msgpack)."""
    heures = brut.iloc[NB_LIGNES_ENTETE:, 0]
    valeurs = brut.iloc[NB_LIGNES_ENTETE:, 1:].to_numpy(dtype=np.float64)
    id_sites, id_debits, noms_debits = (
        brut.iloc[0, 1:],
        brut.iloc[1, 1:],
        brut.iloc[2, 1:],
    )
    debits = {}
    for colonne in range(valeurs.shape[1]):
        id_debit = str(id_debits.iloc[colonne])
        debits[id_debit] = {
            "flow_id": int(id_debit),
            "site_id": int(str(id_sites.iloc[colonne])),
            "flow_name": str(noms_debits.iloc[colonne]),
            "year": annee,
            "values": encoder_debit(valeurs[:, colonne]).tobytes(),
        }
    return {
        "year": annee,
        "time_start": pd.to_datetime(heures.iloc[0]).isoformat(),
        "flows": debits,
    }


def preparer_debits_bruts(
    dossier_brutes: Path, dossier_sortie: Path, annees: tuple[int, ...]
) -> dict[int, Path]:
    """Convertit le classeur brut de chaque année demandée (saute les années déjà converties - relancer
    ce script ne refait donc pas tout le travail à chaque fois).

    Attend un fichier 'counters/flows_{annee}.xlsx' par année dans 'dossier_brutes'."""
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemins: dict[int, Path] = {}
    for annee in annees:
        cible = dossier_sortie / f"flow_{annee}.msgpack"
        chemins[annee] = cible
        if cible.exists():
            print(f"[débits] {annee} déjà préparé")
            continue
        payload = encoder_une_annee(
            lire_excel_brut(dossier_brutes / "counters" / f"flows_{annee}.xlsx"), annee
        )
        with open(cible, "wb") as f:
            msgpack.pack(payload, f)
        print(f"[débits] {annee} : {len(payload['flows'])} débits -> {cible.name}")
    return chemins
