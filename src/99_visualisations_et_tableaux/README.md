# 99_visualisations_et_tableaux

Tout ce qui RESTITUE le BKT calculé aux étapes 04-06 : indicateurs de base, incertitude statistique,
export Excel, graphiques, cartes. Rien ici ne recalcule le BKT différemment - ce dossier ne fait que
lire `data/donnees_valides/bkt/` et le mettre en forme.

## ⚠️ Ce que les intervalles de confiance mesurent (et ce qu'ils NE mesurent PAS)

**Les indicateurs et intervalles de confiance de ce dossier décrivent l'incertitude de NOTRE méthode
d'estimation du BKT (étapes 04-06), pas une incertitude sur le "vrai" BKT national.** Personne
n'observe le vrai BKT national - il n'existe aucune vérité terrain à laquelle comparer notre
estimation. Ce que `intervalle_confiance.py` quantifie : "si on avait eu un échantillon différent de
communes actives (mais de même taille), à quel point notre taux rogné - et le BKT_ext qui en découle -
auraient-ils pu varier ?" - une incertitude D'ÉCHANTILLONNAGE propre à la méthode des étapes 05/06
(voir leurs README), pas une mesure d'écart à une réalité inobservable. Un IC étroit signifie "notre
méthode est stable sur cet échantillon de communes", PAS "le vrai BKT national est probablement dans
cette fourchette".

## Le dossier, fichier par fichier

```
indicateurs.py             population, capteurs (standard/panel équilibré), réseau, débits, BKT mini
intervalle_confiance.py     bootstrap 3000 tirages : IC 95% du BKT_ext, test de significativité de la
                            croissance - voir l'avertissement ci-dessus
xlsx.py                     utilitaire xlsxwriter réutilisable (classe Classeur, fonctions serie/plage)
export_excel.py             construit le classeur final bkt_final.xlsx (6 feuilles) à partir de xlsx.py
graphiques.py                fonctions matplotlib réutilisables (hors Excel) : mêmes vues, indépendantes
rendu.py                     rendu PNG/HTML partagé des cartes de communes (matplotlib / folium)
cartes.py                    carte de couverture des communes + carte du cluster prédit par commune
schemas_visualisations.py    à quoi ressemblent les tables produites
pipeline.py                  la commande "tout faire" (config en haut, enchaînement en bas)
```

Note : la carte INTERACTIVE des capteurs par cluster (profils journalier/hebdomadaire/saisonnier) est
dans `01_clustering_des_usages/carte.py`, pas ici - ce dossier ajoute les deux cartes au niveau
COMMUNE (couverture, cluster prédit) qui n'existaient pas encore.

## Vérifications

Voir les notebooks : schémas validés (`schemas_visualisations.py`), test de croissance significatif à
5 % sur toutes les transitions 2019-2025, largeurs d'intervalle de confiance qui se resserrent avec le
nombre de capteurs actifs (comme attendu d'un bootstrap - plus l'échantillon est grand, plus
l'estimation est précise). Le classeur Excel et les cartes sont construits à partir des mêmes tables
déjà vérifiées aux étapes 04-06.

## Comment le lancer

- **`pipeline.py`** (`python pipeline.py`) : nécessite `data/donnees_valides/bkt/communes.parquet` et
  `communes_bkt_obs.parquet` (étapes 04/05 - lancez leurs `pipeline.py` d'abord si besoin). Calcule dans
  l'ordre : indicateurs -> intervalles de confiance (quelques minutes, bootstrap) -> classeur Excel ->
  cartes de couverture/clusters (HTML, interactives).
- **`99a_indicateurs_et_intervalles_de_confiance.ipynb`**, **`99b_export_excel.ipynb`**,
  **`99c_graphiques.ipynb`**, **`99d_cartes.ipynb`** : montrent chaque module en détail.

## Où vont les données

```
data/donnees_valides/bkt/
    indicateurs.csv, debits_par_cluster.csv, capteurs_par_cluster_standard.csv, capteurs_par_cluster_mini.csv
    ic_standard.csv, ic_mini.csv, test_croissance.csv, test_croissance_par_cluster.csv
    bkt_final.xlsx
    cartes/
        carte_couverture_communes_<annee>.html
        carte_clusters_communes_<annee>.html
```
