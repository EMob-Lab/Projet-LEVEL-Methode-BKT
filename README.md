# VV6 — Pipeline BKT

**⚠️ Toujours en développement.**

Estime le BKT (Bike-Kilomètres Travelled) national : des données brutes (capteurs, communes, réseau
cyclable) au chiffre final, cluster par cluster et année par année. Méthode retenue : linéaire d'une
commune réparti entre ses clusters au prorata du débit de ses capteurs, "Z complet", rognage choisi
par validation croisée leave-one-out.

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
