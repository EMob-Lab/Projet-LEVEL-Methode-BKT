# -*- coding: utf-8 -*-
"""Télécharge un extrait OpenStreetMap France (.osm.pbf) depuis Geofabrik, si besoin d'un snapshot
propre à une année précise (voir la docstring de 'pipeline.py' : par défaut, ce dépôt réutilise le
snapshot déjà téléchargé par l'étape 00, pas besoin d'en re-télécharger un par année).

Réutilise 'telecharger()' de '00_transformation_des_donnees/outils.py' (téléchargement reprenable, déjà
utilisé pour la mosaïque SRTM) plutôt que de dupliquer cette logique."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "00_transformation_des_donnees"))
from outils import telecharger  # noqa: E402

# Geofabrik garde un historique de snapshots datés, format 'https://download.geofabrik.de/europe/france-YYMMDD.osm.pbf'
URL_GEOFABRIK = "https://download.geofabrik.de/europe/france-{date}.osm.pbf"


def obtenir_pbf(date_snapshot: str, dossier: Path) -> Path:
    """Chemin du PBF Geofabrik daté 'date_snapshot' (format 'AAMMJJ', ex. '250101') - le télécharge
    d'abord s'il n'est pas déjà sur disque (voir 'telecharger', reprenable)."""
    cible = dossier / f"france-{date_snapshot}.osm.pbf"
    if cible.exists():
        return cible
    return telecharger(URL_GEOFABRIK.format(date=date_snapshot), cible)
