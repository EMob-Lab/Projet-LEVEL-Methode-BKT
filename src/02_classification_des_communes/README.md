# 02_classification_des_communes

Prédit le cluster d'usage de TOUTE commune française (34 428, France métropolitaine), y compris celles
qui n'hébergent aucun capteur. Le clustering de `01_clustering_des_usages` ne connaît que les ~1000
communes qui en hébergent un ; cette étape apprend, à partir d'elles, quel TYPE de commune (population,
densité, distances, relief) correspond à quel cluster, par une forêt aléatoire - et applique ce qu'elle
a appris à toutes les autres.

## Comment le lancer

- **`pipeline.py`** (`python pipeline.py`) : entraîne la forêt aléatoire et prédit le cluster de
  toutes les communes en une seule commande. Nécessite que `00_transformation_des_donnees` et
  `01_clustering_des_usages` aient déjà été lancés (voir leur `pipeline.py` respectif).
- **`classification_des_communes.ipynb`** : montre pas à pas comment le jeu d'entraînement est
  construit, comment le modèle est évalué, et ce que donnent ses prédictions.

## Le dossier, fichier par fichier

```
jeu_de_donnees.py         quel type de commune correspond à quel cluster - construit les données
                           d'entraînement (une ligne par capteur, pondérée par commune)
entrainement.py            la forêt aléatoire elle-même (hyperparamètres, validation croisée groupée)
prediction.py               entraîne ET applique le modèle à toutes les communes en une fonction
schemas_classification.py   à quoi ressemble une ligne de commune_clusters.parquet
pipeline.py                  la commande "tout faire" (config en haut, enchaînement en bas)
```

## Où vont les données

```
data/donnees_valides/classification/commune_clusters.parquet   une ligne par commune (34 428) : le
    cluster prédit, ses probabilités, et si la commune a fourni une observation directe ou non
    (colonne "extrapolee").

models/classification_des_communes/foret_aleatoire.joblib   le modèle entraîné (ModeleClassification,
    voir entrainement.py) - à recharger avec ModeleClassification.charger() plutôt qu'à réentraîner.
```

## Ce que cette étape n'essaie PAS de faire

Aucune donnée vendorisée ni aucune référence historique ici : le modèle est entraîné entièrement sur
VOS propres capteurs et VOTRE propre clustering d'usage (`01_clustering_des_usages`), pas sur un
classifieur ou des labels déjà publiés. La qualité de cette étape dépend directement de la qualité du
clustering choisi à l'étape 01 - un clustering issu d'une petite recherche exploratoire donnera une
classification plus bruitée qu'un clustering issu d'une recherche Optuna approfondie.
