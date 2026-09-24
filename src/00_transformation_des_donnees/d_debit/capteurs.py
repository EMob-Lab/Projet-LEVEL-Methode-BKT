"""Fusionne les métadonnées des capteurs (nom, position, commune...) avec leur débit horaire de
chaque année, un enregistrement par site.

* Les métadonnées viennent de deux classeurs statiques (la liste des sites enrichie + les anciennes
  coordonnées d'une liste plus ancienne) - une "photo" du réseau d'aujourd'hui, réutilisée pour toutes
  les années passées.
* Le débit d'un site pour une année est la SOMME de tous ses débits (un site peut avoir plusieurs
  voies/sens de comptage = plusieurs flux), heure par heure.
* Les sites qui n'ont jamais eu le moindre débit sont retirés.
* Les sites HORS FRANCE MÉTROPOLITAINE (Corse incluse) sont retirés ICI, à la source - même périmètre
  que le reste du projet (voir 'd_zones/zones.py', 'DEPARTEMENTS_METROPOLE'). Ce filtre doit rester le
  PLUS EN AMONT possible : 'sensors.msgpack' est le point de départ de TOUT ce qui touche aux capteurs
  (clustering d'usage, classification des communes, calcul du BKT...) - le refaire plus tard (dans une
  carte, par exemple) serait trop tard : un capteur ultramarin aurait déjà pollué chaque étape en
  amont de cette carte.

Sortie: sensors.msgpack : {"time_starts": {annee: date_iso}, "capteurs": {id_site: enregistrement}}.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outils import decoder_debit, encoder_debit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "d_zones"))
from zones import DEPARTEMENTS_METROPOLE  # noqa: E402


def code_dept_normalise(valeur: Any) -> str:
    """'974.0' -> '974' (Réunion, écarté), '2.0' -> '02' (métropole, gardé), '2A' -> '2A' (Corse,
    écarté) - les codes département arrivent parfois en float depuis Excel, ou déjà en texte."""
    texte = str(valeur).strip()
    if texte.endswith(".0"):
        texte = texte[:-2]
    return texte.zfill(2)


# petites conversions tolérantes aux valeurs sales
def _vers_entier(valeur: Any, defaut: int = 0) -> int:
    try:
        return int(float(str(valeur).replace(",", ".")))
    except (ValueError, TypeError):
        return defaut


def _vers_reel(valeur: Any, defaut: float = 0.0) -> float:
    try:
        return float(str(valeur).replace(",", "."))
    except (ValueError, TypeError):
        return defaut


def _decouper(valeur: Any, separateur: str = ",") -> list[str]:
    if pd.isna(valeur) or valeur == "":
        return []
    return [
        morceau.strip() for morceau in str(valeur).split(separateur) if morceau.strip()
    ]


# lecture des sources
def charger_metadonnees(chemin_sites: Path, chemin_anciens_sites: Path) -> pd.DataFrame:
    """Table des sites enrichie, complétée (jointure à gauche) par les anciennes coordonnées connues."""
    principale = pd.read_excel(chemin_sites)
    anciens = pd.read_excel(chemin_anciens_sites)[["Identifiant", "Lat.", "Long"]]
    anciens = anciens.rename(
        columns={"Identifiant": "site_id", "Lat.": "former_lat", "Long": "former_long"}
    )
    principale["site_id"] = principale["site_id"].astype(int)
    anciens["site_id"] = anciens["site_id"].astype(int)
    return principale.merge(anciens, on="site_id", how="left")


def charger_debits_par_site(
    dossier_debits: Path, annees: tuple[int, ...]
) -> tuple[dict[int, dict[str, list[dict]]], dict[str, str]]:
    """Regroupe les débits (déjà préparés par 'debits_bruts.py') par site : '{site: {annee: [débits]}}',
    plus le premier horodatage de chaque année."""
    par_site: dict[int, dict[str, list[dict]]] = {}
    debuts_annee: dict[str, str] = {}
    for annee in annees:
        with open(dossier_debits / f"flow_{annee}.msgpack", "rb") as f:
            payload = msgpack.unpack(f, raw=False)
        debuts_annee[str(payload["year"])] = payload["time_start"]
        for debit in payload["flows"].values():
            par_site.setdefault(int(debit["site_id"]), {}).setdefault(
                str(payload["year"]), []
            ).append(debit)
    print(f"[capteurs] débits chargés pour {len(par_site)} sites distincts")
    return par_site, debuts_annee


def additionner_debits_du_site(debits: list[dict]) -> np.ndarray:
    """Somme heure par heure de tous les débits d'un même site (les séries de longueurs différentes
    sont tronquées à la plus courte)."""
    if not debits:
        return np.array([], dtype=np.float32)
    decodes = [decoder_debit(d["values"]) for d in debits]
    longueur = min(len(d) for d in decodes)
    total = np.zeros(longueur, dtype=np.float32)
    for serie in decodes:
        total += serie[:longueur]
    return total


# construction de l'enregistrement
def construire_enregistrement(
    ligne: pd.Series,
    debit_agrege: dict[str, np.ndarray],
    ids_debits: dict[str, list[int]],
) -> dict[str, Any]:
    """Assemble toutes les informations d'UN capteur
    (métadonnées + débit codé de chaque année) dans un
    seul dictionnaire, prêt à être sauvegardé."""
    return {
        "_id": str(ligne.get("_id", "")),
        "id_site": int(ligne["site_id"]),
        "nom_site": str(ligne.get("site_name", "")),
        "id_commune": _vers_entier(ligne.get("comm_code")),
        "flow_ids": ids_debits,
        "code_dept": str(ligne.get("dept_code", "")),
        "code_region": str(ligne.get("region22_code", "")),
        "first_data_date": str(ligne.get("Première donnée", None))
        if ligne.get("Première donnée", None) is not None
        else "",
        "last_data_date": str(ligne.get("Dernière donnée", None))
        if ligne.get("Dernière donnée", None) is not None
        else "",
        "usagers": _decouper(ligne.get("Pratiques"), "/"),
        "long": _vers_reel(ligne.get("xlong")),
        "lat": _vers_reel(ligne.get("ylat")),
        "pratique_typique": str(ligne.get("pratique_type_algo", "")),
        "status_voie": str(ligne.get("Statut de la voie_declaratif", "")),
        "revetement_voie": str(ligne.get("Revêtement_declaratif", "")),
        "osm_way_id": _vers_entier(ligne.get("way_id")),
        "validation_status": str(ligne.get("status", "")),
        "way_geometry": str(ligne.get("geometry", None)),
        "amenagement": str(ligne.get("amenagement", "")),
        "aggregated_flow_values": {
            annee: encoder_debit(valeurs).tobytes()
            for annee, valeurs in debit_agrege.items()
        },
        "former_long": _vers_reel(ligne.get("former_long")),
        "former_lat": _vers_reel(ligne.get("former_lat")),
    }


def preparer_capteurs(
    dossier_brutes: Path,
    dossier_debits: Path,
    dossier_sortie: Path,
    annees: tuple[int, ...],
) -> Path:
    """Fusionne métadonnées et débits annuels dans un seul fichier 'sensors.msgpack'.

    Attend 'counters/counter_stations.xlsx' et 'counter_stations_former.xlsx' dans 'dossier_brutes',
    et les fichiers 'flow_{annee}.msgpack' (déjà produits par 'debits_bruts.preparer_debits_bruts')
    dans 'dossier_debits'."""
    metadonnees = charger_metadonnees(
        dossier_brutes / "counters" / "counter_stations.xlsx",
        dossier_brutes / "counters" / "counter_stations_former.xlsx",
    )
    debits_par_site, debuts_annee = charger_debits_par_site(dossier_debits, annees)

    capteurs: dict[str, dict] = {}
    hors_metropole = 0
    for _, ligne in metadonnees.iterrows():
        if code_dept_normalise(ligne.get("dept_code")) not in DEPARTEMENTS_METROPOLE:
            hors_metropole += 1
            continue
        site = int(ligne["site_id"])
        agrege: dict[str, np.ndarray] = {}
        ids_debits: dict[str, list[int]] = {}
        for annee, debits in debits_par_site.get(site, {}).items():
            total = additionner_debits_du_site(debits)
            if len(total) > 0:
                agrege[annee] = total
            ids_debits[annee] = sorted(d["flow_id"] for d in debits)
        if agrege:  # on ne garde que les sites qui ont eu du débit au moins une fois
            capteurs[str(site)] = construire_enregistrement(ligne, agrege, ids_debits)

    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemin = dossier_sortie / "sensors.msgpack"
    with open(chemin, "wb") as f:
        msgpack.pack({"time_starts": debuts_annee, "capteurs": capteurs}, f)
    print(f"[capteurs] {len(capteurs)} capteurs avec des données de débit -> {chemin} ({hors_metropole} hors France métropolitaine écartés)")
    return chemin


def charger_capteurs(dossier_sortie: Path) -> dict:
    """Relit le fichier 'sensors.msgpack' déjà préparé."""
    with open(dossier_sortie / "sensors.msgpack", "rb") as f:
        return msgpack.unpack(f, raw=False)
