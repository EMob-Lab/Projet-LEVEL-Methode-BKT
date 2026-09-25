**| [Aperçu](#aperçu) | [Installation](#installation) | [Lancer le pipeline](#lancer-le-pipeline) | [Où trouver quoi](#où-trouver-quoi) | [Licence](#licence) | [Contact](#contact) |**

# VV6 — Pipeline BKT

![Status](https://img.shields.io/badge/status-en%20développement-orange)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-LGPL--3.0-blue)

**⚠️ Toujours en développement.**

Estime le BKT (Bike-Kilomètres Travelled) national, dans le cadre du projet LEVEL : des données
brutes (capteurs, communes, réseau cyclable) au chiffre final, cluster par cluster et année par année.

## Aperçu

Méthode retenue, simplement décrite :

Nous disposons de débits de capteurs et de linéaire cyclable. Nous utilisons les débits pour trouver
les différents clusters, c'est-à-dire les pratiques cyclables dominantes de ces capteurs. Nous
utilisons les délimitations des communes pour définir une "zone d'influence" pour les capteurs - les
mots "zone" et "commune" sont interchangeables pour tout le projet.

Pour toutes les communes qui ont du linéaire cyclable ET des capteurs, nous attribuons le linéaire de
la commune proportionnellement au débit de chaque cluster présent dans celle-ci. Nous calculons
ensuite le BKT observé, c'est-à-dire le nombre de kilomètres parcourus observé, ainsi :

```
BKT observé = linéaire du cluster dans la commune × (somme des débits des capteurs du cluster dans la commune / nombre de capteurs du cluster dans la commune)
```

Nous « extrapolons » ensuite, en utilisant des données socio-économiques et géographiques, pour
trouver la pratique dominante des communes non instrumentées. Pour ce faire, dans les communes
instrumentées (qui possèdent un ou plusieurs capteurs, donc un ou plusieurs clusters), nous décidons
que la pratique dominante de cette commune est le cluster au plus gros débit.

Nous disposons maintenant du BKT observé, et des communes non observées, c'est-à-dire du linéaire
cyclable réparti par cluster mais sans débit associé. Pour remédier à ce débit absent, nous calculons
un « taux de fréquentation par km » pour chaque cluster : pour chaque zone, le nombre de kilomètres
parcourus / le nombre de kilomètres disponibles. Nous en faisons ensuite une moyenne générale par
cluster, avec une moyenne spéciale (rognage choisi par validation croisée LOOCV, voir
[`06_calcul_du_bkt_extrapole/rognage_optimal.py`](src/06_calcul_du_bkt_extrapole/rognage_optimal.py))
qui élimine l'influence des valeurs extrêmes.

Enfin, avec le linéaire cyclable non observé et le taux de fréquentation par km de chaque cluster, nous
multiplions l'un par l'autre pour obtenir la part « extrapolée » du BKT. Nous additionnons la part
observée (BKT observé) et la part extrapolée pour obtenir le BKT national (aussi appelé BKT_ext ou
extrapolé dans les notebooks).

## Installation

Ce dépôt utilise [uv](https://docs.astral.sh/uv/) pour la gestion des dépendances (voir
`pyproject.toml`) :

```bash
uv sync
```

Pour tout installer et exécuter les notebooks et autres code annexe au pipeline du BKT :

```bash
uv sync --extra notebooks
```

## Lancer le pipeline

Le pipeline complet :

```bash
uv run python full_pipeline.py
```

Étape par étape (utile pour déboguer, ou relancer juste une étape) :

```bash
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
data/donnees_brutes/README.md                     où obtenir chaque source brute (pas fournie avec ce dépôt)
outputs/cartes/carte_capteurs.html                carte interactive des capteurs par cluster (notebook étape 01)
```

Chaque dossier de `src/` a son propre README (dossier, comment le lancer, où vont ses données) et son
propre notebook pédagogique.

## Licence

Ce projet est distribué sous licence [GNU LGPL-3.0](LICENSE).

## Contact

Projet mené dans le cadre du stage d'Olivier Borot - pour toute question, ouvrir une
[issue](../../issues) sur ce dépôt.
