# -*- coding: utf-8 -*-
"""Table "capteur x année" : combien chaque capteur a mesuré chaque année, et si cette mesure est
utilisable pour le BKT calculé par CE dépôt (méthode retenue - voir le README du projet).

Pour chaque (capteur, année), calculés à partir du débit horaire :

    annual_flow    somme des comptages horaires valides sur l'année
    n_valid_h      nombre d'heures avec un comptage valide (les autres sont "manquantes")
    qta            "Quantité de Trafic Annuel" - le débit HORAIRE moyen (annual_flow / n_valid_h),
                   extrapolé aux 8760 heures d'une année complète : une estimation du nombre TOTAL de
                   passages sur l'année si le capteur avait mesuré en continu. Le calcul du BKT (étape
                   05) multiplie directement un linéaire (km) par 'qta' (passages/an) pour obtenir des
                   bike-km/an, voir sa docstring. Calculée ICI, directement juste - aucune correction
                   supplémentaire n'est nécessaire plus loin dans le pipeline.
    valid          le capteur a mesuré un peu de débit (qta > 0)
    active         le capteur a mesuré SUFFISAMMENT : plus de 5% des heures de l'année sont non
                   nulles ET le débit total dépasse 100 passages. Un capteur "valide" mais pas
                   "actif" est un "orphelin" (exclu du BKT de cette année-là) - SEULS les capteurs
                   actifs de l'année entrent dans le calcul du BKT observé (voir la méthode retenue,
                   pas la méthode historique qui réintégrait certains orphelins : ce dépôt ne
                   construit que ce dont la méthode retenue a besoin).
"""

from __future__ import annotations

import sys
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outils import decoder_debit  # noqa: E402

SEUIL_PART_ACTIVE = 0.05
SEUIL_DEBIT_TOTAL = 100.0


def charger_capteurs(dossier_capteurs: Path) -> dict:
    with open(dossier_capteurs / "sensors.msgpack", "rb") as f:
        return msgpack.unpack(f, raw=False)


def construire_capteurs_annees(payload: dict, annees: tuple[int, ...]) -> pd.DataFrame:
    """Une ligne par (capteur, année) présent dans le fichier des capteurs."""
    lignes = []
    for annee in annees:
        series = {}
        for enregistrement in payload["capteurs"].values():
            brut = enregistrement["aggregated_flow_values"].get(str(annee))
            if isinstance(brut, bytes):
                series[int(enregistrement["id_site"])] = (enregistrement, decoder_debit(brut))
        if not series:
            continue
        n_heures = max(len(valeurs) for _, valeurs in series.values())  # une série plus courte compte zéro sur la fin
        for site, (enregistrement, valeurs) in series.items():
            valide = ~np.isnan(valeurs)
            n_valides = int(valide.sum())
            annuel = float(np.nansum(valeurs)) if n_valides else 0.0
            remplie = np.where(valide, valeurs, 0.0)
            commune = enregistrement.get("id_commune")
            lignes.append({
                "id_site": site,
                "nom_site": enregistrement["nom_site"],
                "id_commune_str": str(int(commune)).zfill(5) if pd.notna(commune) else None,
                "lat": enregistrement.get("lat"),
                "long": enregistrement.get("long"),
                "code_dept": enregistrement.get("code_dept"),
                "annee": annee,
                "annual_flow": annuel,
                "n_valid_h": n_valides,
                "qta": annuel / n_valides * 8760.0 if n_valides else 0.0,
                "activity_share": float((remplie > 0).sum() / n_heures),
                "total_flow": float(remplie.sum()),
            })
    table = pd.DataFrame(lignes)
    table["valid"] = table["qta"] > 0
    table["active"] = (table["activity_share"] > SEUIL_PART_ACTIVE) & (table["total_flow"] > SEUIL_DEBIT_TOTAL)
    return table


def preparer_capteurs_annees(dossier_capteurs: Path, dossier_sortie: Path, annees: tuple[int, ...]) -> Path:
    """Construit la table capteur x année et l'écrit en parquet.

    Nécessite 'sensors.msgpack' (préparé par 'capteurs.py') dans 'dossier_capteurs'."""
    payload = charger_capteurs(dossier_capteurs)
    table = construire_capteurs_annees(payload, annees)
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemin_table = dossier_sortie / "sensor_years.parquet"
    table.to_parquet(chemin_table, index=False)
    print(f"[capteurs-années] {len(table):,} lignes (capteur, année), {int(table['active'].sum()):,} actives")
    return chemin_table
