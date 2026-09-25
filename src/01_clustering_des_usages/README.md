# 01_clustering_des_usages

Attribue à chaque capteur un cluster d'usage K=4 (funFEM, sur son profil de débit) : quatre pratiques
cyclables distinctes (voir `reference.CLUSTER_NAMES`). Par défaut, reprend le clustering de référence
déjà vendorisé - aucun ajustement à refaire.

## Comment lancer le pipeline

```
python pipeline.py
```

Un seul paramètre à changer si besoin (`SOURCE_MODELE` dans `pipeline.py`) : `"reference"` (par
défaut, une jointure sur des labels déjà connus) ou `"modele"` (un modèle fraîchement ajusté avec
`modele_classique.py` ou `recherche_hyperparametres.py`, non lancés automatiquement). Le notebook
`01_clustering_des_usages.ipynb` montre la démarche en détail.

## Où trouver quoi

```
reference.py                  clustering de référence vendorisé, utilisé par défaut
FunFEM.py                     l'algorithme funFEM lui-même
representation.py             profils de capteurs -> représentation numérique attendue par funFEM
modele_classique.py           ajuste un K=4 sur le seul profil hebdomadaire, sans recherche d'hyperparamètres
recherche_hyperparametres.py  recherche Optuna (K, covariance, poids des échelles temporelles)
exemple_clustering_multiechelle.py  exemple : ajuster funFEM sur plusieurs profils à la fois
appliquer_modele.py           charge/applique un modèle déjà entraîné à de nouveaux profils
profils.py                    courbe moyenne par cluster (figure de contrôle + vignette de carte.py)
carte.py                      carte interactive des capteurs par cluster
schemas_clusters.py           à quoi ressemble la table produite
pipeline.py                   la commande "tout faire"
```

Sortie :

```
data/donnees_valides/clustering/cluster_assignments.parquet   lue par les étapes 02 et 04
```
