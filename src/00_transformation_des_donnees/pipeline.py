"""Étape 00 — transformation des données : lance TOUTES les transformations de ce dossier, des
sources brutes jusqu'à la table finale des communes, en une seule commande.

    python pipeline.py

Toute la CONFIGURATION (quels dossiers, quelles années...) est réunie juste en dessous - c'est la
SEULE chose à modifier pour adapter ce script à un autre poste de travail ou à un sous-ensemble
d'années. Le reste du fichier ne fait qu'enchaîner les fonctions des autres fichiers du dossier, dans
l'ordre où elles dépendent les unes des autres :

    zones ──────────────► communes / départements / régions (+ population INSEE, surface)
    population ─────────► population INSEE de chaque millésime
    débits bruts ─► capteurs ─┬─► capteurs-années  (débit annuel, QTA, indicateur d'activité)
                              └─► profils d'usage (profil horaire/hebdomadaire/annuel de chaque capteur)
    géographie OSM ─────► cours d'eau, littoral, mairies
    altitude ───────────► relief de chaque commune (modèle SRTM)
    eau ────────────────► distance de chaque commune à la mer/une rivière/un canal
    urbain ─────────────► distance de chaque commune au grand centre urbain le plus proche
    socio-économie ─────► grille de densité, revenu médian, aménagements cyclables, accueil vélo
    table communes ─────► TOUTES les variables ci-dessus jointes en une seule table (+ dérivées)

Chaque étape est sautée si son fichier de sortie existe déjà (pratique pour relancer le script après
une interruption sans tout refaire) - sauf si 'FORCER_LE_RECALCUL = True' ci-dessous.

NB : ce dossier ne prépare PAS le réseau cyclable (GéoVélo/OSM) - c'est fait dans l'étape suivante du
projet ('03_creation_du_jeu_de_donnees_cyclable/'), volontairement séparée de la transformation des
données de base.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - tout ce qu'il y a à changer pour adapter ce script se trouve ici.
# ═══════════════════════════════════════════════════════════════════════════════════════════════

# Racine du dépôt VV6 (ce fichier est dans src/00_transformation_des_donnees/, donc la racine est 2
# niveaux au-dessus) - déduite automatiquement, pas besoin d'y toucher sauf cas particulier.
RACINE = Path(__file__).resolve().parents[2]

DOSSIER_DONNEES_BRUTES = RACINE / "data" / "donnees_brutes"
DOSSIER_DONNEES_VALIDES = RACINE / "data" / "donnees_valides"

# Sous-dossiers de sortie (dans DOSSIER_DONNEES_VALIDES), un par domaine - reflète l'organisation en
# 'd_zones/', 'd_debit/', 'd_geographie/', 'd_socio_eco/' de ce dossier.
DOSSIER_ZONES = DOSSIER_DONNEES_VALIDES / "zones"
DOSSIER_DEBIT_BRUT = (
    DOSSIER_DONNEES_VALIDES / "debit_brut"
)  # msgpack intermédiaire, un fichier par année
DOSSIER_CAPTEURS = DOSSIER_DONNEES_VALIDES / "capteurs"
DOSSIER_GEOGRAPHIE = DOSSIER_DONNEES_VALIDES / "geographie"
DOSSIER_SOCIO_ECO = DOSSIER_DONNEES_VALIDES / "socio_economique"

ANNEES = (2019, 2020, 2021, 2022, 2023, 2024, 2025)
MILLESIMES_POPULATION = (
    2019,
    2020,
    2021,
    2022,
    2023,
)  # 2024 et 2025 réutilisent le dernier millésime publié

# Extrait OSM (.osm.pbf) utilisé pour les cours d'eau/littoral et les mairies - un extrait France
# récent quelconque convient, pas besoin qu'il soit très à jour pour ces deux usages.
FICHIER_OSM = DOSSIER_DONNEES_BRUTES / "geography" / "osm_reference.osm.pbf"

# Une mosaïque SRTM (relief) déjà prête, si vous en avez déjà une sous la main - laisse à None pour
# la (re)télécharger et la construire automatiquement (15 tuiles, ~300 Mo, une seule fois).
MOSAIQUE_SRTM_EXISTANTE: Path | None = None

FORCER_LE_RECALCUL = (
    False  # True pour ignorer les fichiers déjà présents et tout recalculer
)

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# à partir d'ici, plus aucune configuration - seulement l'enchaînement des étapes
# ═══════════════════════════════════════════════════════════════════════════════════════════════

ICI = Path(__file__).resolve().parent
for sous_dossier in ("d_zones", "d_debit", "d_geographie", "d_socio_eco"):
    sys.path.insert(0, str(ICI / sous_dossier))
sys.path.insert(0, str(ICI))

from capteurs import preparer_capteurs  # noqa: E402
from capteurs_annees import preparer_capteurs_annees  # noqa: E402
from debits_bruts import preparer_debits_bruts  # noqa: E402
from population import preparer_population  # noqa: E402
from profils_usage import preparer_profils_usage  # noqa: E402
from table_communes import preparer_geographie, preparer_table_communes  # noqa: E402
from zones import preparer_zones  # noqa: E402


def _etape(nom: str, fichier_sortie: Path, calcul) -> None:
    """Affiche le nom de l'étape et l'exécute, sauf si son fichier de sortie existe déjà (et qu'on
    n'a pas demandé 'FORCER_LE_RECALCUL')."""
    if fichier_sortie.exists() and not FORCER_LE_RECALCUL:
        print(f"== {nom} : déjà préparé ({fichier_sortie.name})")
        return
    print(f"== {nom}")
    calcul()


def main() -> None:
    print(f"Données brutes  : {DOSSIER_DONNEES_BRUTES}")
    print(f"Données valides : {DOSSIER_DONNEES_VALIDES}")
    print()

    _etape(
        "zones (communes / départements / régions)",
        DOSSIER_ZONES / "communes.parquet",
        lambda: preparer_zones(DOSSIER_DONNEES_BRUTES, DOSSIER_ZONES),
    )
    _etape(
        "population INSEE",
        DOSSIER_ZONES / "population.parquet",
        lambda: preparer_population(
            DOSSIER_DONNEES_BRUTES, DOSSIER_ZONES, MILLESIMES_POPULATION
        ),
    )

    preparer_debits_bruts(
        DOSSIER_DONNEES_BRUTES, DOSSIER_DEBIT_BRUT, ANNEES
    )  # le saut par année est géré à l'intérieur
    _etape(
        "capteurs (métadonnées + débit)",
        DOSSIER_CAPTEURS / "sensors.msgpack",
        lambda: preparer_capteurs(
            DOSSIER_DONNEES_BRUTES, DOSSIER_DEBIT_BRUT, DOSSIER_CAPTEURS, ANNEES
        ),
    )
    _etape(
        "profils d'usage",
        DOSSIER_CAPTEURS / "usage_profiles.parquet",
        lambda: preparer_profils_usage(DOSSIER_CAPTEURS, DOSSIER_CAPTEURS, ANNEES),
    )
    _etape(
        "capteurs-années",
        DOSSIER_CAPTEURS / "sensor_years.parquet",
        lambda: preparer_capteurs_annees(DOSSIER_CAPTEURS, DOSSIER_CAPTEURS, ANNEES),
    )

    preparer_geographie(
        FICHIER_OSM, DOSSIER_ZONES, DOSSIER_GEOGRAPHIE, MOSAIQUE_SRTM_EXISTANTE
    )
    _etape(
        "table finale des communes",
        DOSSIER_DONNEES_VALIDES / "commune_features.parquet",
        lambda: preparer_table_communes(
            DOSSIER_DONNEES_BRUTES,
            DOSSIER_ZONES,
            DOSSIER_GEOGRAPHIE,
            DOSSIER_SOCIO_ECO,
            DOSSIER_DONNEES_VALIDES,
        ),
    )

    print(
        "\nTerminé : la table finale des communes est prête pour les étapes suivantes du projet."
    )


if __name__ == "__main__":
    main()
