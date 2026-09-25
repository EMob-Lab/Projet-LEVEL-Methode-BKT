# 00_transformation_des_donnees

Transforme les sources brutes (`data/donnees_brutes/`) en tables prêtes à l'emploi
(`data/donnees_valides/`) : zones administratives, population, capteurs et leur débit (dont QTA -
Quantité de Trafic Annuel, calculée une fois pour toutes ici), géographie, socio-économie, et la table
finale des communes qui réunit tout ça. Le réseau cyclable (GéoVélo/OSM) est préparé à part, à l'étape
03.

## Comment lancer le pipeline

```
python pipeline.py
```

Nécessite les sources brutes sous `data/donnees_brutes/` (voir leur documentation pour où les
obtenir). Une étape déjà calculée est sautée (`FORCER_LE_RECALCUL = True` dans `pipeline.py` pour
forcer). Le notebook `00_transformation_des_donnees.ipynb` montre chaque fonction séparément.

## Où trouver quoi

```
d_zones/zones.py                communes / départements / régions
d_zones/population.py           population INSEE, tous les millésimes
d_debit/debits_bruts.py         classeurs Excel bruts -> format compact
d_debit/capteurs.py             métadonnées des capteurs + débit horaire
d_debit/profils_usage.py        débit horaire -> profils (entrée du clustering, étape 01)
d_debit/capteurs_annees.py       débit annuel, QTA, indicateur d'activité par (capteur, année)
d_geographie/*.py                cours d'eau, littoral, altitude, distance urbaine
d_socio_eco/socio_economique.py grille de densité, revenu médian, aménagements cyclables
table_communes.py                joint tout ce qui précède en une seule table
schemas.py                       à quoi ressemble chaque table produite
pipeline.py                      la commande "tout faire"
```

Sorties :

```
data/donnees_valides/
    zones/{communes,departements,regions,population}.parquet
    capteurs/{sensors.msgpack,usage_profiles.parquet,sensor_years.parquet}
    debit_brut/flow_<annee>.msgpack
    geographie/*.parquet
    socio_economique/communes.parquet
    commune_features.parquet   table finale, lue par les étapes suivantes
```
