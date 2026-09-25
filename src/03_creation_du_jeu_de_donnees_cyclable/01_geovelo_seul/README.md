# 01_geovelo_seul

La longueur du réseau cyclable d'une commune prise TELLE QUELLE dans les exports GéoVélo annuels -
sans passer par OpenStreetMap ni par un modèle (contrairement à FOB, voir `../02_FOB/`).

**Gardée pour comparaison** (voir `../README.md`) - N'est PAS utilisée par les étapes suivantes du
projet : FOB (`../02_FOB/`) est la méthode retenue.

## Pourquoi elle ne suffit pas

GéoVélo ne référence que le réseau que ses contributeurs ont numérisé. C'est fiable (donnée
directement déclarative, pas de prédiction), mais incomplet : un aménagement réel mais absent de
GéoVélo est absent d'ici. Voir le comparatif (`../README.md`) - l'écart avec FOB (qui, lui, part
d'OpenStreetMap) grandit d'année en année.

## Le dossier, fichier par fichier

```
longueur_geovelo.py   lit un export GéoVélo annuel, calcule la longueur effective par commune
```

## Comment le lancer

**Aucun export GéoVélo brut n'est fourni avec ce dépôt** (source tierce, même remarque que pour les
autres sources brutes - voir `00_transformation_des_donnees/README.md`) : `longueur_geovelo.py` est
donc écrit prêt à l'emploi (voir `01_geovelo_seul.ipynb`), mais son résultat réel pour ce dépôt est
celui déjà vendorisé dans `data/donnees_brutes/reference/longueurs_reseau_reference.csv` (voir
`../reference.py`, colonnes `len_d_GV`/`len_g_GV`).
