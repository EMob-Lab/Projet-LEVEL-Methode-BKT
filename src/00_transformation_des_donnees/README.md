# 00_transformation_des_donnees

Transforme les sources brutes (`data/donnees_brutes/`) en tables propres et prêtes à l'emploi
(`data/donnees_valides/`) : géographie des communes, population, capteurs et leur débit, relief,
distance à l'eau, distance urbaine, socio-économie. C'est la toute première étape du projet - rien
ici ne dépend d'une autre étape.

**Ne prépare que ce dont la méthode de BKT retenue par ce dépôt a besoin**.

## Comment le lancer

Deux façons de faire exactement la même chose :

- **`pipeline.py`** : une seule commande (`python pipeline.py`) qui enchaîne toutes les étapes. Toute
  la configuration (quels dossiers, quelles années) est en haut du fichier - c'est la seule chose à
  changer pour l'adapter à un autre poste de travail.
- **`00_transformation_des_donnees.ipynb`** : le même enchaînement, mais pas à pas - une section qui
  explique, une cellule qui appelle la fonction correspondante et montre son résultat. Utile pour
  comprendre ou déboguer une étape en particulier.

Une étape déjà faite (son fichier de sortie existe déjà) est sautée, pas refaite - relancer
`pipeline.py` après une interruption ne repart pas de zéro.

## Le dossier, fichier par fichier

```
pipeline.py       la commande "tout faire" (config en haut, enchaînement en bas)
schemas.py        à quoi ressemble chaque table produite (voir plus bas)
outils.py         petites fonctions partagées par tout le dossier (encodage du débit, téléchargement...)
arrondissements.py  Paris / Lyon / Marseille comptent comme plusieurs communes (leurs arrondissements)

d_zones/         communes, départements, régions, population INSEE
d_debit/         capteurs : débit brut, métadonnées, profils d'usage, indicateur d'activité
d_geographie/    relief (SRTM), cours d'eau/littoral, mairies, distance au centre urbain
d_socio_eco/     grille de densité, revenu médian, aménagements cyclables, accueil vélo

table_communes.py  joint TOUT ce qui précède en une seule table, une ligne par commune
```

Chaque fichier ne fait qu'UNE chose, et l'explique en tête de fichier (docstring) - ce README ne
répète pas ce détail, il donne juste la carte du dossier.

## Où vont les données

```
data/donnees_brutes/   les sources telles quelles, jamais modifiées par ce dossier
        │
        ▼  (pipeline.py, ou le notebook, pas à pas)
        │
data/donnees_valides/  les tables propres, prêtes pour la suite du projet
```

`data/donnees_brutes/` n'est pas fourni avec le dépôt (sources tierces volumineuses ou soumises à
condition d'usage) - à remplir soi-même selon la disposition attendue par chaque fonction (voir sa
docstring, ou lancez le notebook : la première erreur dit toujours quel fichier manque).

## `schemas.py` - comprendre une table sans relire le code qui l'a produite

Chaque table importante (capteurs, profils d'usage, communes...) a un schéma
[Pydantic](https://docs.pydantic.dev/) dans `schemas.py` : un champ = une description de ce qu'il
contient. C'est de la documentation, pas du code utilisé par le pipeline mais utile pour
vérifier qu'un fichier déjà écrit est correct :

```python
import pandas as pd
from schemas import LigneCapteurAnnee, valider_echantillon

table = pd.read_parquet("chemin/vers/sensor_years.parquet")
valider_echantillon(table, LigneCapteurAnnee)
```
