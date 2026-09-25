# 05_calcul_du_bkt_observe

À partir des trois tables assemblées par l'étape 04, calcule le BKT OBSERVÉ de chaque (année, commune
instrumentée, cluster) : le kilométrage parcouru à vélo que les capteurs laissent directement mesurer
- pas encore une estimation nationale (voir l'étape 06 pour l'extrapolation aux communes sans
capteur). Première étape qui ne lit plus que `data/donnees_valides/bkt/`.

## Comment lancer le pipeline

```
python pipeline.py
```

Nécessite `communes.parquet` et `sensor_detail.parquet` (étape 04 - lancez son `pipeline.py`
d'abord si besoin). Le notebook `05_calcul_du_bkt_observe.ipynb` montre la formule en détail, avec un
exemple réel de commune à plusieurs clusters.

## Où trouver quoi

```
bkt_observe.py            la formule (répartition du linéaire par cluster au prorata du débit)
schemas_bkt_observe.py     à quoi ressemble la table produite
pipeline.py                 la commande "tout faire"
```

Sortie :

```
data/donnees_valides/bkt/communes_bkt_obs.parquet   lue par l'étape 06
```
