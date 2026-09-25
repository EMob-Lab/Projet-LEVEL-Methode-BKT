# VV6 — Pipeline BKT

**⚠️ Toujours en développement.**

Estime le BKT (Bike-Kilomètres Travelled) national : des données brutes (capteurs, communes, réseau
cyclable) au chiffre final, cluster par cluster et année par année.
Méthode retenue simplement décrite :
Nous disposons de débit de capteurs, de linéaire cyclable.
Nous utilisons les débits pour trouver les différents clusters ie pratiques cyclable dominantes de ces capteurs.
Nous utilisons les délimitations des communes pour définir une "zone d'influence" pour les capteurs. Le mots zone et commune sont interchangeables pour tout le projet.
Pour toutes les communes, avec du linéaire cyclable et des capteurs, nous attribuons le linéaire de la commune proportionnellement au débit de chaque cluster présent dans celle-ci.
Nous calculons le bkt observé c'est à dire le nombre de kilomètres parcourues observé, ainsi : linéaire du cluster dans la commune * (somme des débits des capteurs du cluster dans la commune/ nombre de capteurs du cluster dans la commune).
Nous "extrapolons" ensuite, utilisant des données socio-économiques et géographiques, pour trouver la pratique dominante des communes non instrumentées. Pour ce faire nous décidons que dans les communes instrumentées donc qui possède un ou plusieurs capteurs donc un ou plusieurs clusters, la pratique dominante de cette commune est le cluster au plus gros débit.
Maintenant, nous disposons du bkt observé, et des communes non observés c'est-à-dire du linéaire cyclable réparti par cluster sans débit.
Pour remédier au débit absent, nous calculons un "taux de fréquentation par km" pour chaque cluster en calculant pour chaque zone le nombre de kilomètres parcourus / le nombre de kilomètres disponible. On en fait ensuite une moyenne générale par cluster en utilisant une moyenne spéciale (chercher LOOCV dans les notebooks) qui élimine l'influence de valeurs extrêmes.
Enfin, après avoir du "linéaire cyclable non observé", et un "taux de fréquentation par km" pour chaque cluster, nous multiplions l'un par l'autre pour obtenir la part "extrapolée" du BKT.
Nous additionnons enfin la part observée (bkt observé) et la part extrapolée et nous obtenons le BKT national (aussi appelé BKT ext ou extrapolé dans les notebooks).

## Installation

```
uv sync
```

Pour tout installer et exécuter les notebooks et autres code annexe au pipeline du BKT :

```
uv sync --extra notebooks
```

## Lancer le pipeline

Le pipeline complet :

```
uv run python full_pipeline.py
```

Étape par étape (utile pour déboguer, ou relancer juste une étape) :

```
cd src/<dossier_étape>
uv run python pipeline.py
```

**Une étape déjà calculée est sautée automatiquement** - relancer après une interruption ne refait donc
que ce qui manque (`FORCER_LE_RECALCUL = True` dans le `pipeline.py` de l'étape pour forcer).

## Où trouver quoi

```
src/
    00_transformation_des_donnees/       sources brutes -> tables validées (zones, capteurs, QTA...)
    01_clustering_des_usages/            cluster d'usage K4 de chaque capteur
    02_classification_des_communes/      cluster K4 prédit de chaque commune sans capteur mais avec des aménagements cyclables
    03_creation_du_jeu_de_donnees_cyclable/   jeu de données du réseau cyclable (méthode retenue : FOB)
    04_jointure_des_donnees/             assemble toutes les données en 3 tables
    05_calcul_du_bkt_observe/            BKT observé
    06_calcul_du_bkt_extrapole/          BKT extrapolé - le chiffre final
    99_visualisations_et_tableaux/       indicateurs, intervalles de confiance, Excel, cartes

data/donnees_valides/bkt/bkt_final_national.csv   le chiffre final du BKT, par année
data/donnees_valides/bkt/bkt_final.xlsx           le classeur Excel complet
data/donnees_valides/bkt/cartes/                  cartes de communes (couverture, cluster)
```

Chaque dossier de `src/` a son propre README (dossier, comment le lancer, où vont ses données) et son
propre notebook pédagogique.
