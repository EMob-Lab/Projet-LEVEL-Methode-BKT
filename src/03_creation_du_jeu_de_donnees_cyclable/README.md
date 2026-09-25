# 03_creation_du_jeu_de_donnees_cyclable

Prépare le réseau cyclable (linéaire km par commune et par année), séparément de la transformation des
données de base faite à l'étape 00. Trois méthodes comparées, dans trois sous-dossiers :

```
00_methode_primitive/   filtre à règles écrites à la main sur 6 tags OSM - jamais passé à l'échelle
01_geovelo_seul/         le réseau tel que numérisé par GéoVélo, point
02_FOB/                  MÉTHODE RETENUE - modèle LightGBM appris sur OpenStreetMap (voir son README)
```

`02_FOB/` est celle que l'étape 04 (et la suite du projet) utilise. Les deux autres sont conservées
pour comparaison (voir `comparatif_des_trois_methodes.ipynb` et `reference.py`, qui vendorise le
résultat déjà calculé des trois méthodes).

## Comment lancer le pipeline

```
cd 02_FOB
python pipeline.py
```

Voir le README de `02_FOB/` pour le détail (modèles vendorisés, pas réentraînables sans les sources
GéoVélo brutes).

## Où trouver quoi

```
reference.py    longueurs de référence vendorisées, les 3 méthodes, 2019-2025
comparatif_des_trois_methodes.ipynb   compare GV / FOB / FOB_AM
00_methode_primitive/, 01_geovelo_seul/, 02_FOB/   un README et un pipeline.py chacun
```

Sortie retenue :

```
data/donnees_valides/reseau_fob/<annee>/longueurs_fob*.parquet   lue par l'étape 04
```
