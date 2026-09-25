# 04_jointure_des_donnees

Assemble les sorties des étapes 00 à 03 (capteurs, communes, clusters d'usage, classification, réseau
cyclable) en trois tables - le pont vers les étapes suivantes (05, 06, 99), qui ne liront plus QUE
`data/donnees_valides/bkt/`, jamais les dossiers des étapes 00-03 directement. Aucun calcul de source
brute ici, seulement des jointures et des agrégations.

## Comment lancer le pipeline

```
python pipeline.py
```

Nécessite les sorties déjà préparées des étapes 00, 01, 02 (lancez leur `pipeline.py` d'abord si
besoin). Le notebook `04_jointure_des_donnees.ipynb` montre la jointure pas à pas.

## Où trouver quoi

```
entrees.py            charge capteurs actifs, table des communes, communes instrumentées (PLM),
                       le cluster K4 retenu par (année, commune) - voir 'resoudre_cluster_communes'
longueurs.py           réseau cyclable : référence vendorisée ou calcul FOB frais
assemblage.py           la jointure elle-même
schemas_jointure.py      à quoi ressemblent les tables produites
pipeline.py               la commande "tout faire"
```

Sorties :

```
data/donnees_valides/bkt/
    communes.parquet         une ligne par (année, commune) : réseau, population, cluster K4, statut instrumentée
    sensor_detail.parquet    une ligne par (année, capteur) actif : débit, QTA, cluster d'usage
    population_used.parquet  quel millésime de population utilisé pour chaque année
```
