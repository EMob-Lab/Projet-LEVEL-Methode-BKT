# -*- coding: utf-8 -*-
"""Étape 03 (méthode FOB) — construit le jeu de données cyclable FOB : d'un extrait OpenStreetMap à la
longueur de réseau (FOB et FOB_AM) par commune, pour les années demandées.

    python pipeline.py

**L'étape la plus lourde du projet si elle recalcule vraiment tout** : la passe A (tags, voir
'osm_extraction.py') lit l'intégralité de l'extrait OSM France (plusieurs Go), et la passe B
(géométrie) localise les nœuds des tronçons retenus. Idempotent comme le reste du projet : une année
déjà traitée n'est pas refaite (voir 'FORCER_LE_RECALCUL').

Nécessite :
- 'models/fob/' à la racine du dépôt : les modèles LightGBM déjà entraînés (voir 'artefacts.py' - ce
  dépôt ne peut PAS les réentraîner, il manque les exports GéoVélo bruts).
- 'data/donnees_valides/zones/communes.parquet' (étape 00) : les polygones de découpe.
- un extrait OpenStreetMap France (.osm.pbf) - celui déjà téléchargé par l'étape 00
  ('data/donnees_brutes/geography/osm_reference.osm.pbf') convient (voir la docstring de
  '00_transformation_des_donnees/d_geographie/geographie_osm.py' : "un extrait quelconque, pas besoin
  qu'il soit très à jour" - même remarque ici, un snapshot n'a pas besoin d'être EXACTEMENT celui de
  l'année demandée, voir 'ANNEES').
"""

from __future__ import annotations

import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - tout ce qu'il y a à changer pour adapter ce script se trouve ici.
# ═══════════════════════════════════════════════════════════════════════════════════════════════

# Racine du dépôt VV6 (ce fichier est dans src/03_creation_du_jeu_de_donnees_cyclable/02_FOB/, donc la
# racine est 3 niveaux au-dessus) - déduite automatiquement, pas besoin d'y toucher sauf cas particulier.
RACINE = Path(__file__).resolve().parents[3]

DOSSIER_MODELES_FOB = RACINE / "models" / "fob"
FICHIER_COMMUNES = RACINE / "data" / "donnees_valides" / "zones" / "communes.parquet"  # étape 00
FICHIER_OSM = RACINE / "data" / "donnees_brutes" / "geography" / "osm_reference.osm.pbf"  # étape 00
DOSSIER_SORTIE = RACINE / "data" / "donnees_valides" / "reseau_fob"

# Les années à traiter avec CE snapshot OSM (voir la docstring du module : un seul extrait sert pour
# plusieurs années "logiques" - il représente l'état du réseau à SA date de collecte, pas à celle de
# chaque année demandée. Un vrai traitement multi-année télécharge un snapshot par année, voir
# 'osm_telechargement.py').
ANNEES = (2024,)

ALPHA_BIDIRECTIONNEL = 2.0  # un côté bidirectionnel compte pour ce facteur (convention du projet)

FORCER_LE_RECALCUL = False  # True pour ignorer les fichiers déjà présents et tout recalculer

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# à partir d'ici, plus aucune configuration - seulement l'enchaînement des étapes
# ═══════════════════════════════════════════════════════════════════════════════════════════════

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

from application_modele import charger_modeles  # noqa: E402
from artefacts import verifier_modeles  # noqa: E402
from decoupe_communes import PolygonesCommunes  # noqa: E402
from snapshot import traiter_snapshot  # noqa: E402


def main() -> None:
    verifier_modeles(DOSSIER_MODELES_FOB)
    modeles = charger_modeles(DOSSIER_MODELES_FOB)
    communes = PolygonesCommunes.charger(FICHIER_COMMUNES)
    print(f"== réseau cyclable FOB : {len(communes.codes):,} communes, {len(ANNEES)} année(s) à traiter")

    for annee in ANNEES:
        dossier_annee = DOSSIER_SORTIE / str(annee)
        print(f"\n-- année {annee} --")
        meta = traiter_snapshot(FICHIER_OSM, annee, dossier_annee, modeles=modeles, communes=communes, alpha=ALPHA_BIDIRECTIONNEL, forcer=FORCER_LE_RECALCUL)
        print(f"   {meta['n_incl']:,} tronçons FOB, {meta['km_effectif_perimetre']:.0f} km effectifs")

    print(f"\nTerminé : réseau FOB écrit dans {DOSSIER_SORTIE}")


if __name__ == "__main__":
    main()
