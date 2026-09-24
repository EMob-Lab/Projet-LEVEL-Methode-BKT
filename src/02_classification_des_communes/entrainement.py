# -*- coding: utf-8 -*-
"""Le classifieur de communes : une forêt aléatoire à classes équilibrées, réglée sous validation
croisée groupée.

explication de la méthode :

- 'RandomForestClassifier(class_weight="balanced", random_state=GRAINE)' sur les communes des
  capteurs, lignes pondérées '1 / capteurs dans la commune' (voir 'jeu_de_donnees.py') ;
- hyperparamètres choisis par 'RandomizedSearchCV' (N_TIRAGES tirages de ESPACE_RECHERCHE) sous
  'GroupKFold(N_PLIS)' groupé par commune, noté par l'accuracy équilibrée - donc aucune commune
  n'apparaît à la fois dans l'entraînement et la validation d'un même pli ;
- PARAMETRES_PAR_DEFAUT est un jeu d'hyperparamètres raisonnable, utilisé directement par défaut
  (déterministe et rapide) ; 'recherche=True' relance le réglage (RandomizedSearchCV, prend quelques
  minutes).

Les prédictions HORS-PLI ('cross_val_predict') donnent une accuracy honnête (chaque commune est
prédite par un modèle qui ne l'a jamais vue à l'entraînement) ; les clusters de moins de 10 communes ne
peuvent pas être appris de façon fiable sous validation croisée groupée à 5 plis, donc un second score
les exclut (voir 'jeu_de_donnees.clusters_rares')."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jeu_de_donnees import JeuEntrainement, clusters_rares  # noqa: E402

GRAINE = 42
N_PLIS = 5
N_TIRAGES = 40
ESPACE_RECHERCHE = {
    "n_estimators": [300, 500, 800, 1000],
    "max_depth": [5, 7, 10, 12, 15, None],
    "min_samples_leaf": [5, 8, 12, 18, 25],
    "min_samples_split": [10, 15, 20, 30],
    "max_features": ["sqrt", "log2", 0.5],
}
PARAMETRES_PAR_DEFAUT = {"n_estimators": 800, "min_samples_split": 30, "min_samples_leaf": 8, "max_features": 0.5, "max_depth": None}


@dataclass
class ModeleClassification:
    estimateur: RandomForestClassifier
    colonnes: list[str]
    parametres: dict
    recherche_effectuee: bool
    accuracy_equilibree_recherche: float | None  # meilleur score de la recherche (None si PARAMETRES_PAR_DEFAUT a été utilisé)
    accuracy_equilibree_hors_pli: float
    accuracy_hors_pli: float
    accuracy_equilibree_hors_pli_bien_representes: float
    clusters_rares: list[int]

    def sauvegarder(self, chemin) -> None:
        joblib.dump(self, chemin)

    @classmethod
    def charger(cls, chemin) -> "ModeleClassification":
        return joblib.load(chemin)


def _foret(parametres: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**parametres, class_weight="balanced", random_state=GRAINE, n_jobs=-1)


def ajuster_foret_aleatoire(jeu: JeuEntrainement, *, recherche: bool = False) -> ModeleClassification:
    """Ajuste le classifieur sur 'jeu'. 'recherche=True' relance le réglage des hyperparamètres ; par
    défaut, utilise PARAMETRES_PAR_DEFAUT."""
    X, y, poids, groupes = jeu.X, jeu.y, jeu.poids, jeu.groupes
    plis = GroupKFold(n_splits=N_PLIS)
    parametres, score_recherche = dict(PARAMETRES_PAR_DEFAUT), None
    if recherche:
        regleur = RandomizedSearchCV(_foret({}), ESPACE_RECHERCHE, n_iter=N_TIRAGES, cv=plis, scoring="balanced_accuracy", n_jobs=-1, random_state=GRAINE)
        regleur.fit(X, y, sample_weight=poids, groups=groupes)
        parametres, score_recherche = dict(regleur.best_params_), float(regleur.best_score_)
        print(f"[classification] recherche : meilleure accuracy équilibrée en VC {score_recherche:.4f} avec {parametres}")

    foret = _foret(parametres)
    hors_pli = cross_val_predict(foret, X, y, cv=plis, groups=groupes, n_jobs=-1, params={"sample_weight": poids})
    rares = clusters_rares(jeu)
    bien_representes = ~y.isin(rares)
    foret.fit(X, y, sample_weight=poids)
    modele = ModeleClassification(
        estimateur=foret,
        colonnes=X.columns.tolist(),
        parametres=parametres,
        recherche_effectuee=recherche,
        accuracy_equilibree_recherche=score_recherche,
        accuracy_equilibree_hors_pli=float(balanced_accuracy_score(y, hors_pli)),
        accuracy_hors_pli=float((hors_pli == y.to_numpy()).mean()),
        accuracy_equilibree_hors_pli_bien_representes=(
            float(balanced_accuracy_score(y[bien_representes], hors_pli[bien_representes.to_numpy()])) if bien_representes.any() else float("nan")
        ),
        clusters_rares=rares,
    )
    print(f"[classification] forêt aléatoire sur {len(X)} capteurs : accuracy équilibrée hors-pli {modele.accuracy_equilibree_hors_pli:.4f}, accuracy {modele.accuracy_hors_pli:.4f}")
    return modele
