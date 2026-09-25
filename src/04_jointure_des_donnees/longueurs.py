"""Longueur du réseau cyclable par commune, année et scénario - le pont entre l'étape 03
(réseau cyclable) et la jointure.

Longueurs en kilomètres EFFECTIFS, réparties par côté ('len_d'/'len_g' - un côté BIDIRECTIONNEL compte
double, convention du projet). Trois scénarios (voir '03_creation_du_jeu_de_donnees_cyclable/README.md') :

    GV       export GéoVélo annuel tel quel (référence, incomplet mais déclaratif)
    FOB      tout tronçon prédit cyclable par le modèle (étape 03/02_FOB) - la méthode retenue
    FOB_AM   FOB restreint aux aménagements DÉDIÉS (sans AUTRE/AUCUN/NON_RENSEIGNÉ)

SOURCE_LONGUEURS (voir 'pipeline.py', même principe que 'SOURCE_MODELE' aux étapes 01/02) :

    "reference"  (par défaut) la table de référence vendorisée (voir '03_.../reference.py'), qui
                 couvre les 7 années 2019-2025 - AUCUN calcul FOB frais n'est nécessaire.
    "calcul"     la vraie sortie de 02_FOB/pipeline.py (data/donnees_valides/reseau_fob/<année>/) POUR
                 LES SEULES ANNÉES déjà calculées (2024 par défaut dans ce dépôt, voir son README) ;
                 les autres années retombent automatiquement sur la référence, avec un avertissement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCENARIOS: tuple[str, ...] = ("GV", "FOB", "FOB_AM")


def _long_depuis_large(reference: pd.DataFrame) -> pd.DataFrame:
    """CSV de référence (une colonne 'len_d_<scenario>'/'len_g_<scenario>' par scénario) -> table
    longue 'annee, scenario, code_commune, len_d, len_g'."""
    morceaux = []
    for scenario in SCENARIOS:
        m = reference[
            ["annee", "code_commune", f"len_d_{scenario}", f"len_g_{scenario}"]
        ].copy()
        m.columns = ["annee", "code_commune", "len_d", "len_g"]
        m["scenario"] = scenario
        morceaux.append(m)
    longue = pd.concat(morceaux, ignore_index=True)
    longue["code_commune"] = longue["code_commune"].astype(str).str.zfill(5)
    return longue


def charger_longueurs_reference(fichier_reference: Path) -> pd.DataFrame:
    """Longueurs des 3 scénarios, 7 années (2019-2025), depuis la table de référence vendorisée."""
    reference = pd.read_csv(fichier_reference)
    longue = _long_depuis_large(reference)
    print(
        f"[longueurs] référence : {longue['annee'].nunique()} années, {len(longue):,} lignes (scenario, commune)"
    )
    return longue


def charger_longueurs(
    fichier_reference: Path,
    dossier_reseau_fob: Path,
    annees: tuple[int, ...],
    *,
    source: str = "reference",
) -> pd.DataFrame:
    """Longueurs des 3 scénarios pour 'annees' - voir le docstring du module pour 'source'.

    'source="calcul"' ne recalcule RIEN lui-même : il lit la sortie déjà écrite par
    '02_FOB/pipeline.py' (longueurs_fob.parquet / longueurs_fob_amenagements.parquet) pour chaque année
    où elle existe, et retombe sur la référence pour les autres (GV vient toujours de la référence :
    ce dépôt n'a pas les exports GéoVélo annuels bruts, voir '01_geovelo_seul/README.md')."""
    reference = charger_longueurs_reference(fichier_reference)
    if source == "reference":
        return reference[reference["annee"].isin(annees)].reset_index(drop=True)
    if source != "calcul":
        raise ValueError(f"source doit être 'reference' ou 'calcul', reçu {source!r}")

    morceaux = []
    for annee in annees:
        dossier_annee = dossier_reseau_fob / str(annee)
        fichier_fob, fichier_fob_am = (
            dossier_annee / "longueurs_fob.parquet",
            dossier_annee / "longueurs_fob_amenagements.parquet",
        )
        if not (fichier_fob.exists() and fichier_fob_am.exists()):
            print(
                f"[longueurs] {annee} : pas de calcul FOB frais (voir 02_FOB/README.md) -> référence vendorisée utilisée pour cette année"
            )
            morceaux.append(reference[reference["annee"] == annee])
            continue
        morceaux.append(
            reference[(reference["annee"] == annee) & (reference["scenario"] == "GV")]
        )
        for scenario, fichier in (("FOB", fichier_fob), ("FOB_AM", fichier_fob_am)):
            table = pd.read_parquet(fichier)
            table["code_commune"] = table["code_commune"].astype(str).str.zfill(5)
            table.insert(0, "annee", annee)
            table["scenario"] = scenario
            morceaux.append(
                table[["annee", "code_commune", "len_d", "len_g", "scenario"]]
            )
        print(f"[longueurs] {annee} : calcul FOB frais utilisé (02_FOB/pipeline.py)")
    return pd.concat(morceaux, ignore_index=True)
