# -*- coding: utf-8 -*-
"""Relance tout le pipeline BKT, étape par étape et dans l'ordre - une commande unique plutôt que
d'appeler chaque `pipeline.py` à la main, un dossier après l'autre.

    uv run python full_pipeline.py

Chaque étape est lancée EXACTEMENT comme si on faisait `cd <dossier> && python pipeline.py` - un
sous-processus séparé par étape (jamais un import direct : plusieurs étapes ont des fichiers de même
nom, comme `entrees.py` ou `reference.py`, qui se marcheraient dessus dans un seul processus Python).
Une étape déjà calculée est sautée par son propre `pipeline.py` (voir `FORCER_LE_RECALCUL` dans chacun)
- relancer ce script après une interruption ne refait donc que ce qui manque.

Nécessite un interpréteur Python avec toutes les dépendances du projet - voir `pyproject.toml`
(`uv sync` installe tout) et DEPENDANCES_REQUISES ci-dessous, vérifiées avant de lancer quoi que ce
soit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - tout ce qu'il y a à changer pour adapter ce script se trouve ici.
# ═══════════════════════════════════════════════════════════════════════════════════════════════

RACINE = Path(__file__).resolve().parent
DOSSIER_SRC = RACINE / "src"

# L'interpréteur à utiliser pour CHAQUE étape - par défaut celui qui lance ce script (`uv run python
# full_pipeline.py` utilise le `.venv` de VV6, voir `pyproject.toml`/`uv sync`). Remplacez cette ligne
# par un autre chemin si besoin d'un environnement différent.
PYTHON = sys.executable

DEPENDANCES_REQUISES: tuple[str, ...] = (
    "pandas", "numpy", "scipy", "geopandas", "shapely", "pyproj", "rasterio", "rasterstats", "osmium",
    "sklearn", "lightgbm", "joblib", "skfda", "networkx", "matplotlib", "folium", "xlsxwriter",
    "openpyxl", "python_calamine", "msgpack", "pydantic", "optuna",
)

ETAPES: tuple[str, ...] = (
    "00_transformation_des_donnees",
    "01_clustering_des_usages",
    "02_classification_des_communes",
    "03_creation_du_jeu_de_donnees_cyclable/02_FOB",
    "04_jointure_des_donnees",
    "05_calcul_du_bkt_observe",
    "06_calcul_du_bkt_extrapole",
    "99_visualisations_et_tableaux",
)

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# à partir d'ici, plus aucune configuration - seulement l'enchaînement des étapes
# ═══════════════════════════════════════════════════════════════════════════════════════════════


def _verifier_dependances() -> None:
    """Vérifie AVANT de lancer quoi que ce soit que PYTHON a tout ce dont le pipeline a besoin - plutôt
    que de découvrir un 'ModuleNotFoundError' après quelques étapes déjà (re)calculées."""
    script = "import importlib.util\nmanquants = [m for m in %r if importlib.util.find_spec(m) is None]\nprint(','.join(manquants))" % (DEPENDANCES_REQUISES,)
    resultat = subprocess.run([PYTHON, "-c", script], capture_output=True, text=True)
    manquants = [m for m in resultat.stdout.strip().split(",") if m]
    if manquants:
        raise RuntimeError(
            f"l'interpréteur {PYTHON!r} n'a pas : {', '.join(manquants)}. Installez-les, ou changez "
            "PYTHON en haut de ce fichier pour pointer vers un environnement qui les a déjà."
        )


def main() -> None:
    _verifier_dependances()
    for etape in ETAPES:
        dossier = DOSSIER_SRC / etape
        fichier = dossier / "pipeline.py"
        if not fichier.exists():
            raise FileNotFoundError(f"{fichier} introuvable - vérifiez ETAPES ci-dessus.")

        print(f"\n{'=' * 88}\n{etape}\n{'=' * 88}", flush=True)
        resultat = subprocess.run([PYTHON, "pipeline.py"], cwd=dossier)
        if resultat.returncode != 0:
            raise RuntimeError(
                f"{etape}/pipeline.py a échoué (code {resultat.returncode}) - pipeline arrêté là. "
                "Corrigez l'erreur ci-dessus puis relancez : les étapes déjà calculées seront sautées."
            )


if __name__ == "__main__":
    main()
    print("\npipeline complet : ok - le chiffre final du BKT est dans data/donnees_valides/bkt/bkt_final_national.csv.")
