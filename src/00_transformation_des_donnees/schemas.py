"""Schémas Pydantic des tables produites par ce dossier - pas du code exécuté par le pipeline (aucune
fonction de 'pipeline.py' n'importe ce fichier), mais une "visualisation" de la forme de chaque
donnée : chaque champ porte une description, et 'valider_echantillon()'
permet de vérifier concrètement qu'une table déjà écrite sur disque respecte bien ce schéma
utile après une modification d'un des fichiers de ce dossier, ou pour comprendre rapidement ce
que contient un fichier parquet/msgpack sans relire tout le code qui l'a produit.

Focus mis sur les données de CAPTEURS (le cœur du sujet : débit, activité, profils d'usage), qui ont
trois représentations différentes à trois étapes du pipeline - c'est justement pour clarifier CETTE
différence que ces schémas existent :

    EnregistrementCapteur   UN capteur, tel que stocké dans sensors.msgpack (capteurs.py) - ses
                             métadonnées + son débit codé de chaque année, format de stockage compact.
    LigneCapteurAnnee       UNE ligne de sensor_years.parquet (capteurs_annees.py) - un (capteur,
                             année) : débit annuel, QTA, indicateurs d'activité.
    LigneProfilUsage        UNE ligne de usage_profiles.parquet (profils_usage.py) - un (capteur,
                             année) résumé en 4 profils temporels (jour/semaine/année/spectral), la
                             matière première du clustering d'usage (étape suivante du projet).

Les autres tables de ce dossier (géographie, socio-économie, table finale des communes) sont beaucoup
plus larges (74 à 100 colonnes) - les modéliser colonne par colonne ici serait plus dur à lire que le
code qui les produit ; seules leurs clés géographiques (communes, population) sont schématisées
ci-dessous, à titre d'exemple, en plus des données de capteurs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# ── sensors.msgpack (capteurs.py) ──────────────────────────────────────────────────────────────────


class EnregistrementCapteur(BaseModel):
    """UN capteur dans 'sensors.msgpack' (clé = 'id_site' sous 'payload["capteurs"]') : ses métadonnées
    statiques (position, commune, pratique déclarée...) et son débit codé de chaque année. Une "photo"
    du réseau actuelle réutilisée pour toutes les années passées (voir 'capteurs.py')."""

    model_config = ConfigDict(populate_by_name=True)

    identifiant_mongo: str = Field(
        alias="_id",
        description="Identifiant interne de la source (classeur des sites) - opaque, non réutilisé ailleurs dans le pipeline.",
    )
    id_site: int = Field(
        description="Identifiant numérique du capteur - la clé utilisée PARTOUT ailleurs dans le pipeline pour désigner ce capteur."
    )
    nom_site: str = Field(
        description="Nom lisible du site de comptage (ex. 'Pont de la Concorde')."
    )
    id_commune: int = Field(
        description="Code INSEE de la commune où se trouve le capteur, en entier (donc sans le zéro de tête éventuel - à re-zfill(5) au besoin)."
    )
    flow_ids: dict[str, list[int]] = Field(
        description="Par année (clé string), les identifiants des 'débits' (voies/sens de comptage) sommés pour obtenir le débit total de ce capteur cette année-là."
    )
    code_dept: str = Field(
        description="Code département (2 ou 3 caractères, ex. '75', '974')."
    )
    code_region: str = Field(description="Code région administrative.")
    first_data_date: str = Field(
        description="Date de la première mesure connue de ce capteur (chaîne, pas forcément un format uniforme selon la source)."
    )
    last_data_date: str = Field(
        description="Date de la dernière mesure connue de ce capteur."
    )
    usagers: list[str] = Field(
        description="Pratiques déclarées pour ce site (ex. ['Vélo', 'Piéton']) - déclaratif, pas mesuré."
    )
    long: float = Field(
        description="Longitude WGS84 du capteur (0.0 si inconnue - PAS un NaN, à filtrer explicitement si besoin)."
    )
    lat: float = Field(description="Latitude WGS84 du capteur (0.0 si inconnue).")
    pratique_typique: str = Field(
        description="Pratique dominante déterminée par un algorithme externe à ce dépôt (donnée fournie telle quelle)."
    )
    status_voie: str = Field(
        description="Statut déclaratif de la voie (ex. 'piste cyclable', 'voie partagée')."
    )
    revetement_voie: str = Field(description="Revêtement déclaratif de la voie.")
    osm_way_id: int = Field(
        description="Identifiant du tronçon OpenStreetMap associé (0 si non apparié)."
    )
    validation_status: str = Field(
        description="Statut de validation de la fiche capteur dans la source."
    )
    way_geometry: str = Field(
        description="Représentation textuelle de la géométrie du tronçon associé (peut être la chaîne 'None')."
    )
    amenagement: str = Field(
        description="Type d'aménagement cyclable déclaré au niveau du capteur."
    )
    aggregated_flow_values: dict[str, bytes] = Field(
        description="Par année (clé string), le débit horaire agrégé du site, encodé (voir 'outils.encoder_debit'/'decoder_debit' - entiers 16 bits, PAS des float bruts)."
    )
    former_long: float = Field(
        description="Longitude d'une liste de sites plus ancienne, utilisée en repli si 'long' est manquante en amont."
    )
    former_lat: float = Field(description="Latitude de repli correspondante.")


class TableCapteurs(BaseModel):
    """L'enveloppe complète de 'sensors.msgpack' : un horodatage de départ par année (pour recaler les
    séries horaires) et le dictionnaire de tous les capteurs."""

    time_starts: dict[str, str] = Field(
        description="Par année (clé string), l'horodatage ISO du premier point de la série horaire de cette année - nécessaire pour reconstruire un DatetimeIndex à partir des séries codées."
    )
    capteurs: dict[str, EnregistrementCapteur] = Field(
        description="Par 'id_site' (clé string, PAS int, contrainte du format msgpack), l'enregistrement complet du capteur."
    )


# ── sensor_years.parquet (capteurs_annees.py) ──────────────────────────────────────────────────────


class LigneCapteurAnnee(BaseModel):
    """UNE ligne de 'sensor_years.parquet' : un (capteur, année) - combien il a mesuré, et si cette
    mesure est utilisable pour le BKT (méthode retenue par ce dépôt - voir 'capteurs_annees.py')."""

    id_site: int = Field(
        description="Identifiant du capteur (même clé que dans sensors.msgpack)."
    )
    nom_site: str = Field(description="Nom du site (copié depuis sensors.msgpack).")
    id_commune_str: str | None = Field(
        description="Code INSEE de la commune, en chaîne zfill(5) - None si la commune du capteur est inconnue."
    )
    lat: float | None = Field(
        description="Latitude du capteur (peut être None si absente en amont)."
    )
    long: float | None = Field(description="Longitude du capteur.")
    code_dept: str | None = Field(description="Code département du capteur.")
    annee: int = Field(description="Année de cette mesure.")
    annual_flow: float = Field(
        ge=0,
        description="Somme des comptages horaires VALIDES de l'année (les heures manquantes ne comptent pas, ne sont pas mises à zéro).",
    )
    n_valid_h: int = Field(
        ge=0,
        description="Nombre d'heures avec un comptage valide cette année (les autres sont 'manquantes', pas 'nulles').",
    )
    qta: float = Field(
        ge=0,
        description="'Quantité de Trafic Annuel' = annual_flow / n_valid_h x 8760 - estimation du nombre total de passages sur l'année (voir la docstring de 'capteurs_annees.py'). Déjà la grandeur utilisée telle quelle par le calcul du BKT (étape 05), aucune correction supplémentaire.",
    )
    activity_share: float = Field(
        ge=0,
        le=1,
        description="Part des heures de l'année (sur le total possible, pas seulement les heures valides) où le débit est non nul.",
    )
    active: bool = Field(
        description="True si le capteur a mesuré SUFFISAMMENT cette année (activity_share > 5% ET total_flow > 100) - SEULS les capteurs actifs entrent dans le calcul du BKT observé."
    )
    valid: bool = Field(
        description="True si le capteur a mesuré un peu de débit (qta > 0) - condition plus faible que 'active'."
    )
    total_flow: float = Field(
        ge=0,
        description="Somme des comptages de l'année, heures manquantes comptées comme zéro (contrairement à 'annual_flow') - utilisée par le filtre d'activité.",
    )


# ── usage_profiles.parquet (profils_usage.py) ──────────────────────────────────────────────────────


class LigneProfilUsage(BaseModel):
    """UNE ligne de 'usage_profiles.parquet' : un (capteur, année) résumé en 4 profils temporels - la
    matière première du clustering d'usage (funFEM, étape suivante du projet). Le fichier parquet lui-
    même stocke ces profils en 249 colonnes PLATES ('d00'..'d23', 'h000'..'h167', 'y00'..'y51',
    's0'..'s4') ; ce schéma les regroupe par échelle temporelle pour rester lisible - voir
    'depuis_ligne_plate()' pour convertir une ligne du parquet (Series pandas) en cet objet."""

    id_site: int = Field(description="Identifiant du capteur.")
    annee: int = Field(description="Année de ce profil.")
    profil_journalier: list[float] = Field(
        min_length=24,
        max_length=24,
        description="Débit moyen de chaque HEURE DE LA JOURNÉE (colonnes 'd00'..'d23' du parquet) - la forme d'une journée type.",
    )
    profil_hebdomadaire: list[float] = Field(
        min_length=168,
        max_length=168,
        description="Débit moyen de chaque HEURE DE LA SEMAINE, lundi 00h à dimanche 23h (colonnes 'h000'..'h167') - la forme d'une semaine type.",
    )
    profil_annuel: list[float] = Field(
        min_length=52,
        max_length=52,
        description="Débit moyen de chaque SEMAINE DE L'ANNÉE (colonnes 'y00'..'y51') - la saisonnalité.",
    )
    profil_spectral: list[float] = Field(
        min_length=5,
        max_length=5,
        description="5 variables spectrales (colonnes 's0'..'s4') : part d'énergie du signal aux rythmes demi-journalier/journalier/hebdomadaire/annuel, puis l'entropie spectrale (à quel point le signal est prévisible/répétitif).",
    )

    @classmethod
    def depuis_ligne_plate(cls, ligne: pd.Series) -> "LigneProfilUsage":
        """Construit une 'LigneProfilUsage' à partir d'une ligne du parquet réel (colonnes plates
        'd00'..'s4'), en regroupant les colonnes par échelle temporelle."""
        return cls(
            id_site=int(ligne["id_site"]),
            annee=int(ligne["annee"]),
            profil_journalier=[float(ligne[f"d{h:02d}"]) for h in range(24)],
            profil_hebdomadaire=[float(ligne[f"h{h:03d}"]) for h in range(168)],
            profil_annuel=[float(ligne[f"y{s:02d}"]) for s in range(52)],
            profil_spectral=[float(ligne[f"s{i}"]) for i in range(5)],
        )


# ── zones/communes.parquet et zones/population.parquet (zones.py, population.py) ──────────────────


class LigneCommune(BaseModel):
    """UNE ligne de 'zones/communes.parquet' (sans la colonne 'geometry', un polygone Shapely - pas un
    type Pydantic simple ; voir 'zones.py' pour la géométrie elle-même)."""

    code_commune: str = Field(description="Code INSEE de la commune (5 caractères).")
    nom_commune: str = Field(
        description="Nom de la commune (ou de l'arrondissement pour Paris/Lyon/Marseille)."
    )
    code_departement: str = Field(description="Code département (2 ou 3 caractères).")
    population_commune: float | None = Field(
        description="Population légale la plus récente connue à la construction de cette table (peut être None)."
    )
    aire_km2_commune: float = Field(
        gt=0,
        description="Surface de la commune en km², calculée en projection Lambert-93 (la projection officielle française, pas WGS84).",
    )


class LignePopulation(BaseModel):
    """UNE ligne de 'zones/population.parquet' : la population d'UNE commune à UN millésime INSEE."""

    millesime: int = Field(
        description="Année du millésime INSEE (2019 à 2023 - 2024/2025 réutilisent le dernier millésime publié, voir 'pipeline.py')."
    )
    code_commune: str = Field(description="Code INSEE de la commune.")
    population: float = Field(
        ge=0, description="Population légale de la commune à ce millésime."
    )


# ── validation d'un échantillon déjà écrit sur disque ──────────────────────────────────────────────


def valider_echantillon(
    table: pd.DataFrame, modele: type[BaseModel], n: int = 20, depuis_ligne=None
) -> None:
    """Valide les 'n' premières lignes de 'table' contre 'modele' et affiche un résumé - lève la
    première erreur rencontrée avec le numéro de ligne fautif, pour un diagnostic rapide.

    'depuis_ligne' : fonction 'Series -> modele' à utiliser à la place de 'modele(**ligne.to_dict())'
    quand la table ne correspond pas 1-pour-1 aux champs du modèle (voir
    'LigneProfilUsage.depuis_ligne_plate' pour un exemple - colonnes plates regroupées en listes).

    Usage :
        >>> import pandas as pd
        >>> from schemas import LigneCapteurAnnee, valider_echantillon
        >>> table = pd.read_parquet("sensor_years.parquet")
        >>> valider_echantillon(table, LigneCapteurAnnee)
        [schemas] 20/20 lignes valides pour LigneCapteurAnnee
    """
    echantillon = table.head(n)
    convertisseur = depuis_ligne or (
        lambda ligne: modele(
            **{
                cle: (
                    None if isinstance(valeur, float) and np.isnan(valeur) else valeur
                )
                for cle, valeur in ligne.to_dict().items()
            }
        )
    )
    for position, (_, ligne) in enumerate(echantillon.iterrows()):
        try:
            convertisseur(ligne)
        except ValidationError as erreur:
            raise ValueError(
                f"[schemas] ligne {position} invalide pour {modele.__name__} :\n{erreur}"
            ) from erreur
    print(
        f"[schemas] {len(echantillon)}/{len(echantillon)} lignes valides pour {modele.__name__}"
    )


def valider_capteurs(payload: dict[str, Any]) -> TableCapteurs:
    """Valide l'intégralité de 'sensors.msgpack' déjà chargé (voir 'capteurs.charger_capteurs') contre
    'TableCapteurs' - plus coûteux qu'un échantillon (valide TOUS les capteurs), mais rapide en
    pratique (quelques milliers d'enregistrements, pas des millions)."""
    return TableCapteurs.model_validate(payload)
