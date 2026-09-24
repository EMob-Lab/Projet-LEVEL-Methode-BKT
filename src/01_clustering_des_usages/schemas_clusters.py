"""Schémas Pydantic des données de CLUSTER produites par ce dossier (voir aussi
'00_transformation_des_donnees/schemas.py' pour les données de capteurs en amont, dont celles-ci
dérivent). Documentation vivante, pas du code utilisé par le pipeline - voir 'valider_echantillon'
dans le module de l'étape 00 pour l'outil de validation (réutilisé ici, pas redéfini)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

CRITERES_POSSIBLES = ("bic", "aic", "icl")
REPRESENTATIONS_POSSIBLES = (
    "multiechelle",
    "hebdomadaire",
)  # voir appliquer_modele.REPRESENTATIONS


class ConfigModele(BaseModel):
    """'config.json' d'UN modèle sauvegardé sous 'models/clustering_des_debits/<identifiant>/' (voir
    'appliquer_modele.sauvegarder_modele') - les hyperparamètres de représentation utilisés à
    l'entraînement, à relire pour appliquer ce modèle à de nouveaux capteurs.

    Deux origines possibles (champ 'representation', voir 'appliquer_modele.REPRESENTATIONS') qui
    N'UTILISENT PAS les mêmes hyperparamètres :

    - "multiechelle" (voir 'recherche_hyperparametres.py') : les 7 champs 'n_base_*'/'poids_*' sont
      TOUS renseignés.
    - "hebdomadaire" (voir 'modele_classique.py') : SEULS 'n_base_semaine' et 'poids_semaine' sont
      renseignés - les autres 'n_base_*'/'poids_*' valent 'None' et n'ont pas de sens pour ce modèle.

    C'est pour ça que ces 7 champs sont optionnels ici plutôt qu'obligatoires : un modèle "classique"
    valide n'en a que 2 sur 7.

    Le CRITÈRE utilisé (une valeur parmi 'bic'/'aic'/'icl', voir CRITERES_POSSIBLES) est écrit sous sa
    propre clé (ex. 'bic': -9172430.6) plutôt qu'un champ fixe 'critere_valeur' - ce modèle autorise
    donc les clés supplémentaires ('extra="allow"') pour accepter n'importe laquelle des trois sans
    dupliquer le schéma trois fois."""

    model_config = ConfigDict(extra="allow")

    rang: int = Field(
        ge=1,
        description="Rang de ce modèle (1 = le retenu par pipeline.py - voir sa docstring).",
    )
    essai: int = Field(
        ge=0,
        description="Numéro de l'essai Optuna d'origine (0 pour un modèle 'hebdomadaire', qui n'est pas issu d'une recherche).",
    )
    k: int = Field(ge=2, description="Nombre de clusters de ce modèle.")
    modele: str = Field(
        description="Nom du modèle de covariance funFEM utilisé (ex. 'AkBk' - voir FunFEM.py)."
    )
    graine: int = Field(
        description="Graine aléatoire exacte utilisée pour l'ajustement funFEM final (reproductible)."
    )
    representation: str = Field(
        description=f"Quelle fonction de representation.py reconstruit (X, W) pour ce modèle - une valeur parmi {REPRESENTATIONS_POSSIBLES} (voir appliquer_modele.REPRESENTATIONS)."
    )
    n_base_jour: int | None = Field(
        default=None,
        gt=0,
        description="Nombre de fonctions B-splines du bloc journalier - None si representation='hebdomadaire' (pas de bloc journalier).",
    )
    n_base_semaine: int = Field(
        gt=0,
        description="Nombre de fonctions B-splines du bloc hebdomadaire - toujours renseigné, les deux representations en ont un.",
    )
    n_base_annee: int | None = Field(
        default=None,
        gt=0,
        description="Nombre de fonctions B-splines du bloc annuel - None si representation='hebdomadaire'.",
    )
    poids_jour: float | None = Field(
        default=None,
        ge=0,
        description="Poids du bloc journalier dans la métrique W - None si representation='hebdomadaire'.",
    )
    poids_semaine: float = Field(
        ge=0, description="Poids du bloc hebdomadaire - toujours renseigné."
    )
    poids_annee: float | None = Field(
        default=None,
        ge=0,
        description="Poids du bloc annuel - None si representation='hebdomadaire'.",
    )
    poids_spectral: float | None = Field(
        default=None,
        ge=0,
        description="Poids du bloc spectral - None si representation='hebdomadaire'.",
    )


class LigneAssignationCluster(BaseModel):
    """UNE ligne de 'assignation_clusters.csv' (écrit à côté de chaque 'config.json') : le cluster
    attribué à un capteur par CE modèle, dans SA numérotation propre (deux modèles différents peuvent
    numéroter le même groupe de capteurs différemment - voir 'appliquer_modele.py')."""

    id_site: int = Field(
        description="Identifiant du capteur (même clé que dans sensor_years.parquet / usage_profiles.parquet)."
    )
    cluster: int = Field(
        ge=0, description="Cluster attribué à ce capteur par ce modèle (0-indexé)."
    )
