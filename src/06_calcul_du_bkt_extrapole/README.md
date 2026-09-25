# 06_calcul_du_bkt_extrapole

C'est ici que le chiffre FINAL du BKT est produit : à partir du BKT observé (étape 05, qui ne couvre
que les communes instrumentées), extrapole aux communes SANS capteur, cluster par cluster - voir
`bkt_extrapole.py` pour la formule (« rognage Z »).

## Comment lancer le pipeline

```
python pipeline.py
```

Nécessite `communes_bkt_obs.parquet` (étape 05 - lancez son `pipeline.py` d'abord si besoin). Le
notebook `06_calcul_du_bkt_extrapole.ipynb` montre la formule et le résultat final, avec ses contrôles
de cohérence.

## Où trouver quoi

```
bkt_extrapole.py            la formule (Z complet + taux rogné par cluster)
rognage_optimal.py           vérifie/recalcule LOOCV_OPTIMAL_TRIM (validation croisée leave-one-out)
schemas_bkt_extrapole.py     à quoi ressemblent les tables produites
pipeline.py                   la commande "tout faire"
```

Sorties :

```
data/donnees_valides/bkt/
    bkt_final_detail_par_cluster.csv   une ligne par (année, cluster)
    bkt_final_national.csv              le chiffre final du BKT, par année (colonne bkt_ext_Mdkm)
```
