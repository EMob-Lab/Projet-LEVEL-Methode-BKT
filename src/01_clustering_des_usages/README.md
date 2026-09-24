# 01_clustering_des_usages

Classe chaque capteur selon sa pratique (le rythme de son débit : domicile-travail, loisirs,
tourisme...), par la méthode funFEM (voir `FunFEM.py`) appliquée aux profils de débit produits par
`00_transformation_des_donnees/`. C'est cette classification que les étapes suivantes (classification
des communes, calcul du BKT) utilisent ensuite.

## Un seul paramètre pilote ce dossier : `SOURCE_MODELE`

Réglable en haut de `pipeline.py` (et section 0 du notebook) :

- **`"reference"` (par défaut)** : le clustering K=4 DE RÉFÉRENCE du projet, déjà connu et vendorisé
  (voir `reference.py`) - pas d'ajustement funFEM, juste une jointure. Le plus simple, le plus rapide,
  et le seul dont les clusters ont déjà un nom (`reference.CLUSTER_NAMES`) et des couleurs
  (`reference.CLUSTER_COULEURS`).
- **`"modele"`** : un modèle fraîchement ajusté sur vos capteurs. Réutilise le dernier modèle
  sauvegardé sous `models/clustering_des_debits/` s'il y en a un, sinon en ajuste un nouveau - le
  modèle "classique" par défaut (`modele_classique.py` : K=4 sur le seul profil hebdomadaire, aucun
  réglage) ; `recherche_hyperparametres.py` (recherche Optuna, K variable) reste disponible pour
  montrer la manière de faire pour explorer plus large si le modèle de référence ne convient pas.

Changer ce paramètre bascule tout le comportement de `pipeline.py` ET du notebook - rien d'autre à
toucher. Aucun des deux chemins n'est un choix qu'un pipeline automatique fait tout seul en dehors de
ce paramètre : c'est toujours une décision explicite.

## Comment le lancer

- **`pipeline.py`** (`python pipeline.py`) : attribue un cluster à chaque capteur de l'année de
  référence selon `SOURCE_MODELE`, écrit l'assignation "officielle".
- **`01_clustering_des_usages.ipynb`** : montre pas à pas ce que fait `pipeline.py`, et affiche les
  résultats (profils par cluster, carte interactive) au passage.

## Le dossier, fichier par fichier

```
reference.py                  le clustering K=4 DE RÉFÉRENCE (vendorisé) - le choix par défaut
modele_classique.py           CHOISIR un modèle simple : K=4 direct sur le profil hebdomadaire seul
recherche_hyperparametres.py  CHOISIR un modèle par recherche Optuna (K variable) - outil ponctuel
appliquer_modele.py           charge/sauvegarde/UTILISE un modèle fraîchement ajusté (les deux origines
                               ci-dessus écrivent et lisent le même format, voir REPRESENTATIONS)
FunFEM.py                     l'algorithme funFEM lui-même (classification de données fonctionnelles)
representation.py             transforme des profils bruts en (X, W) - deux versions : multi-échelle
                               (jour+semaine+année+spectral) et hebdomadaire seule (le modèle classique)
profils.py                    dessine la courbe moyenne d'un cluster à une échelle de temps donnée
carte.py                      la carte interactive des capteurs colorés par cluster
schemas_clusters.py           à quoi ressemble un modèle sauvegardé (config.json, assignation) - voir
                               aussi 00_transformation_des_donnees/schemas.py pour les données de capteurs
pipeline.py                   la commande "tout faire" - lit SOURCE_MODELE et fait ce qu'il faut
gabarits/                     CSS/JS/HTML de la carte interactive (fichiers à part, pas de gros bloc
                               de texte au milieu du code Python de carte.py)
```

## Où vont les données

```
data/donnees_brutes/reference/station_clusters_k4_run006.csv   le clustering de référence vendorisé
    (Identifiant_Site, Nom_Site, cluster) - lu par reference.py, jamais modifié.

models/clustering_des_debits/<identifiant>/   un modèle FRAÎCHEMENT AJUSTÉ (SOURCE_MODELE="modele"
                                               seulement - rien n'est écrit ici en mode "reference") :
    config.json                hyperparamètres, K, modèle, valeur du critère, representation utilisée
    modele.joblib               le modèle funFEM entraîné
    assignation_clusters.csv    id_site, cluster (sur les capteurs utilisés pour ajuster ce modèle)
    carte.html                  la carte de CE modèle

data/donnees_valides/clustering/cluster_assignments.parquet   sortie de pipeline.py : l'assignation
    "officielle", quelle que soit SOURCE_MODELE - c'est CELLE-CI que les étapes suivantes du projet
    doivent lire, jamais un fichier de reference.py ou un dossier de modèle individuel directement.
```

## `schemas_clusters.py`

Comme `00_transformation_des_donnees/schemas.py`, mais pour les données de CLUSTER (`ConfigModele`,
`LigneAssignationCluster`) - fichier séparé, même nom aurait provoqué un conflit d'import si les deux
schémas sont utilisés ensemble dans le même script.
