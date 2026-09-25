# 02_classification_des_communes

Prédit le cluster d'usage K=4 de CHAQUE commune de France métropolitaine (34 428, pas seulement les
~1000 qui hébergent un capteur) : une forêt aléatoire entraînée sur la géographie des communes
instrumentées (population, densité, distances, altitude) et le cluster réel de leur(s) capteur(s)
(étape 01), appliquée à toutes les communes. Par défaut, reprend des prédictions déjà connues,
vendorisées - aucun entraînement à refaire.

## Comment lancer le pipeline

```
python pipeline.py
```

Un seul paramètre à changer si besoin (`SOURCE_MODELE` dans `pipeline.py`) : `"reference"` (par
défaut, prédictions déjà connues) ou `"modele"` (entraîne, ou réutilise, une forêt aléatoire - voir
`entrainement.py`). Le notebook `classification_des_communes.ipynb` montre la démarche en détail.

## Où trouver quoi

```
reference.py               prédictions de référence vendorisées, utilisées par défaut
jeu_de_donnees.py          assemble le jeu d'entraînement (communes instrumentées x leur cluster réel)
entrainement.py            entraîne la forêt aléatoire (RandomizedSearchCV en option)
prediction.py              applique le modèle à toutes les communes
schemas_classification.py  à quoi ressemble la table produite
pipeline.py                la commande "tout faire"
```

Sortie :

```
data/donnees_valides/classification/commune_clusters.parquet   lue par l'étape 04
```
