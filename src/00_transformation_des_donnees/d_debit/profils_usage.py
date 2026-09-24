"""Résume le débit horaire de chaque (capteur, année) en "profils" à plusieurs échelles de temps -
c'est la matière première du clustering d'usage (étape suivante du projet, pas faite ici).

Pour chaque (capteur, année), le débit horaire (une heure manquante compte pour zéro, comme dans
l'étude d'origine) est résumé en :

    d00..d23    débit moyen de chaque HEURE DE LA JOURNÉE (24 valeurs)
    h000..h167  débit moyen de chaque HEURE DE LA SEMAINE, lundi 00h à dimanche 23h (168 valeurs)
    y00..y51    débit moyen de chaque SEMAINE DE L'ANNÉE (52 valeurs, la saisonnalité)
    s0..s4      5 variables SPECTRALES : part d'énergie du signal aux rythmes demi-journalier,
                journalier, hebdomadaire et annuel, plus l'entropie spectrale (une mesure de
                "à quel point le signal est prévisible/répétitif")

Ce fichier ne fait AUCUN lissage ni normalisation "de forme" (ça, c'est le travail du clustering) -
juste la transformation propre des séries horaires en tableau.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outils import decoder_debit

HEURES_PAR_JOUR = 24
HEURES_PAR_SEMAINE = 7 * 24
SEMAINES_PAR_AN = 52
PERIODES_SPECTRALES_H = {
    "demi_journaliere": 12,
    "journaliere": 24,
    "hebdomadaire": 24 * 7,
    "annuelle": 24 * 365,
}

COLONNES_JOUR = [f"d{h:02d}" for h in range(HEURES_PAR_JOUR)]
COLONNES_SEMAINE = [f"h{h:03d}" for h in range(HEURES_PAR_SEMAINE)]
COLONNES_ANNEE = [f"y{s:02d}" for s in range(SEMAINES_PAR_AN)]
COLONNES_SPECTRALES = [f"s{i}" for i in range(5)]
COLONNES_PROFIL = (
    COLONNES_JOUR + COLONNES_SEMAINE + COLONNES_ANNEE + COLONNES_SPECTRALES
)


def position_dans_la_semaine(index: pd.DatetimeIndex) -> np.ndarray:
    """Heure-de-semaine de chaque horodatage (0 = lundi 00h, 167 = dimanche 23h)."""
    return index.dayofweek.to_numpy() * 24 + index.hour.to_numpy()


def _moyenne_groupee(
    valeurs: np.ndarray, position: np.ndarray, taille: int
) -> np.ndarray:
    """Moyenne de 'valeurs' groupée par 'position' (ex. par heure de la journée) ; les positions
    au-delà de 'taille' (une 53e semaine, par exemple) sont ignorées."""
    sommes = np.bincount(position[: len(valeurs)], weights=valeurs, minlength=taille)[
        :taille
    ]
    comptes = np.bincount(position, minlength=taille)[:taille]
    return np.divide(sommes, comptes, out=np.zeros(taille), where=comptes > 0)


def variables_spectrales(valeurs: np.ndarray) -> np.ndarray:
    """Transforme une série horaire en 5 nombres qui résument sa forme dans le domaine fréquentiel."""
    centre = valeurs - valeurs.mean()
    puissance = np.abs(rfft(centre)) ** 2
    frequences = rfftfreq(len(centre), d=1.0)
    total = puissance.sum() + 1e-12
    parts = [
        puissance[np.argmin(np.abs(frequences - 1 / PERIODES_SPECTRALES_H[nom]))]
        / total
        for nom in ("demi_journaliere", "journaliere", "hebdomadaire", "annuelle")
    ]
    p = puissance / (puissance.sum() + 1e-12)
    entropie = -(p * np.log(p + 1e-12)).sum()
    return np.array([*parts, entropie])


def _series_de_l_annee(payload: dict, annee: int) -> dict[int, np.ndarray]:
    series = {}
    for enregistrement in payload["capteurs"].values():
        brut = enregistrement["aggregated_flow_values"].get(str(annee))
        if isinstance(brut, bytes):
            series[int(enregistrement["id_site"])] = decoder_debit(brut)
    return series


def construire_profils(payload: dict, annees: tuple[int, ...]) -> pd.DataFrame:
    """Calcule les profils de chaque (capteur, année) présent dans le fichier des capteurs."""
    lignes, ids, annee_de = [], [], []
    for annee in annees:
        series = _series_de_l_annee(payload, annee)
        if not series:
            continue
        n_heures = max(len(v) for v in series.values())
        debut = pd.to_datetime(payload["time_starts"][str(annee)])
        index = pd.date_range(debut, periods=n_heures, freq="h")
        heure_du_jour, heure_de_la_semaine = (
            index.hour.to_numpy(),
            position_dans_la_semaine(index),
        )
        semaine_de_l_annee = ((index - index.min()).days // 7).astype(int).to_numpy()
        for site in sorted(series):
            valeurs = np.zeros(n_heures)
            valeurs[: len(series[site])] = np.where(
                np.isnan(series[site]), 0.0, series[site]
            )
            lignes.append(
                np.concatenate(
                    [
                        _moyenne_groupee(valeurs, heure_du_jour, HEURES_PAR_JOUR),
                        _moyenne_groupee(
                            valeurs, heure_de_la_semaine, HEURES_PAR_SEMAINE
                        ),
                        _moyenne_groupee(valeurs, semaine_de_l_annee, SEMAINES_PAR_AN),
                        variables_spectrales(valeurs.astype(np.float32)),
                    ]
                )
            )
            ids.append(site)
            annee_de.append(annee)
    table = pd.DataFrame(np.vstack(lignes), columns=COLONNES_PROFIL)
    table.insert(0, "annee", annee_de)
    table.insert(0, "id_site", ids)
    return table


def preparer_profils_usage(
    dossier_capteurs: Path, dossier_sortie: Path, annees: tuple[int, ...]
) -> Path:
    """Charge 'sensors.msgpack' (déjà préparé par 'capteurs.py') et écrit la table des profils en parquet."""
    import msgpack

    with open(dossier_capteurs / "sensors.msgpack", "rb") as f:
        payload = msgpack.unpack(f, raw=False)
    table = construire_profils(payload, annees)
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemin = dossier_sortie / "usage_profiles.parquet"
    table.to_parquet(chemin, index=False)
    print(
        f"[profils] {len(table):,} profils ({len(COLONNES_PROFIL)} colonnes chacun) -> {chemin}"
    )
    return chemin
