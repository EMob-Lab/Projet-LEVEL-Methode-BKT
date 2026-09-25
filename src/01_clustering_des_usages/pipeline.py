"""Étape 01 — clustering d'usage : attribue un cluster à chaque capteur de l'année de référence, et
écrit l'assignation.

    python pipeline.py

**Une seule chose à changer pour tout ce dossier : 'SOURCE_MODELE' ci-dessous.**

- "reference" (par défaut) : le clustering K=4 DE RÉFÉRENCE du projet, vendorisé (voir 'reference.py')
  - pas d'ajustement, juste une jointure. Le plus simple, le plus rapide, et le seul dont on connaisse
  déjà la signification de chaque cluster (voir 'reference.CLUSTER_NAMES').
- "modele" : un modèle fraîchement ajusté et sauvegardé sous 'models/clustering_des_debits/' (voir
  'modele_classique.py' ou 'recherche_hyperparametres.py' pour en produire un - PAS appelé
  automatiquement ici, voir leur docstring). Utilise celui classé "01_...".

Changer CE SEUL paramètre !! bascule tout le comportement du script - rien d'autre à toucher.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - tout ce qu'il y a à changer pour adapter ce script se trouve ici.
# ═══════════════════════════════════════════════════════════════════════════════════════════════

# Racine du dépôt VV6 (ce fichier est dans src/01_clustering_des_usages/, donc la racine est 2
# niveaux au-dessus) - déduite automatiquement, pas besoin d'y toucher sauf cas particulier.
RACINE = Path(__file__).resolve().parents[2]

DOSSIER_CAPTEURS_ETAPE00 = RACINE / "data" / "donnees_valides" / "capteurs"
DOSSIER_MODELES = RACINE / "models" / "clustering_des_debits"
DOSSIER_SORTIE = RACINE / "data" / "donnees_valides" / "clustering"
FICHIER_REFERENCE = (
    RACINE / "data" / "donnees_brutes" / "reference" / "station_clusters_k4_run006.csv"
)

# "reference" (par défaut, voir la docstring du module) ou "modele" (le dernier modèle ajusté).
SOURCE_MODELE = "reference"

# L'année du run de référence - funFEM classe des capteurs, pas des (capteur, année) : il faut choisir
# UNE année de profils à classer.
ANNEE_REFERENCE = 2024

FORCER_LE_RECALCUL = (
    False  # True pour ignorer le fichier déjà présent et tout recalculer
)

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# à partir d'ici, plus aucune configuration - seulement l'enchaînement des étapes
# ═══════════════════════════════════════════════════════════════════════════════════════════════

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

import pandas as pd  # noqa: E402
from appliquer_modele import assigner_clusters  # noqa: E402
from reference import assigner_clusters_reference  # noqa: E402


def _dernier_modele_retenu(dossier_modeles: Path) -> Path:
    """Le modèle classé '01_...' (premier rang) du dernier ajustement sauvegardé - voir
    'appliquer_modele.sauvegarder_modele' pour la convention de nommage."""
    candidats = sorted(dossier_modeles.glob("01_*"))
    if not candidats:
        raise FileNotFoundError(
            f"aucun modèle sous {dossier_modeles} - ajustez-en un d'abord (modele_classique.py ou "
            "recherche_hyperparametres.py, voir le notebook de ce dossier, section 3) avant de relancer ce script, "
            "ou repassez SOURCE_MODELE à 'reference' ci-dessus."
        )
    return candidats[-1]


def main() -> None:
    fichier_sortie = DOSSIER_SORTIE / "cluster_assignments.parquet"
    if fichier_sortie.exists() and not FORCER_LE_RECALCUL:
        print(f"== clustering d'usage : déjà préparé ({fichier_sortie.name})")
        return

    profils = pd.read_parquet(DOSSIER_CAPTEURS_ETAPE00 / "usage_profiles.parquet")
    profils = profils[profils["annee"] == ANNEE_REFERENCE].reset_index(drop=True)

    if SOURCE_MODELE == "reference":
        print(
            "== clustering d'usage : source = reference (run historique K=4, voir reference.py)"
        )
        assignation = assigner_clusters_reference(profils, FICHIER_REFERENCE)
        nom_source = "reference_run006"
    elif SOURCE_MODELE == "modele":
        modele_dir = _dernier_modele_retenu(DOSSIER_MODELES)
        print(f"== clustering d'usage : source = modele ({modele_dir.name})")
        assignation = assigner_clusters(profils, modele_dir)
        nom_source = modele_dir.name
    else:
        raise ValueError(
            f"SOURCE_MODELE={SOURCE_MODELE!r} attendu parmi 'reference', 'modele'"
        )

    assignation.insert(1, "annee_reference", ANNEE_REFERENCE)
    assignation.insert(2, "modele", nom_source)

    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    assignation.to_parquet(fichier_sortie, index=False)
    print(f"écrit : {fichier_sortie}")


if __name__ == "__main__":
    main()
