# 02_FOB

**La méthode RETENUE** - c'est CE dossier que les étapes suivantes du projet (04 et après, pas encore
construites dans ce dépôt) doivent utiliser pour le réseau cyclable.

FOB ("Familles d'Objets Bicyclette") : pour chaque tronçon `highway=*` d'OpenStreetMap, un modèle
LightGBM (entraîné à reconnaître le codage GéoVélo à partir des seuls tags OSM) prédit s'il appartient
au réseau cyclable, puis, pour les tronçons retenus, 10 attributs (type d'aménagement, revêtement,
sens de circulation, régime, localisation - voir `etiquettes.py`). Le réseau retenu est ensuite
découpé commune par commune pour obtenir un linéaire (km) par commune et par année. Deux passes bien
séparées (voir `osm_extraction.py`) : une passe A LÉGÈRE sur TOUS les tronçons (tags -> matrice ->
probabilité d'inclusion), puis une passe B COÛTEUSE (géométrie) réservée aux seuls tronçons que la
passe A a retenus - deux ordres de grandeur plus petits, ce qui rend FOB praticable sur un poste de
travail malgré les ~12 millions de tronçons `highway=*` de la France entière.

## Pourquoi FOB plutôt que les deux autres méthodes

- vs `../00_methode_primitive/` : un modèle APPRIS sur de vrais exemples GéoVélo généralise mieux
  qu'un filtre à règles écrites à la main sur seulement 6 tags, évalué à la main sur une petite zone
  (voir le README de ce dossier).
- vs `../01_geovelo_seul/` : FOB part d'OpenStreetMap, une base bien plus exhaustive et à jour que ce
  que GéoVélo a numérisé - voir le comparatif (`../README.md`), l'écart FOB/GV grandit chaque année.

## Ce dépôt ne peut PAS réentraîner les modèles

L'entraînement (voir `entrainement.py` - la MÉTHODE est portée ici, un port fidèle du code qui a
réellement produit les modèles) a besoin des exports GéoVélo bruts comme vérité terrain - une source
tierce volumineuse, pas fournie avec ce dépôt (même remarque que pour les autres sources brutes, voir
`00_transformation_des_donnees/README.md`). `entrainement.entrainer(...)` n'est donc PAS exécutable
ici : il documente comment les modèles ont été obtenus, il ne les recalcule pas.

Ce dossier utilise donc des modèles DÉJÀ ENTRAÎNÉS, vendorisés dans `models/fob/` (racine du dépôt) -
voir `artefacts.py`. Leurs métriques de validation (précision/rappel/F1 du modèle d'inclusion, accuracy
de chaque modèle d'attribut vs. deux lignes de base) et la méthode d'entraînement (hyperparamètres,
découpage entraînement/validation/test) sont présentées comme un vrai tableau de bord en section 0 du
notebook, avec un bilan en section 6.

## Le dossier, fichier par fichier

```
osm_schema.py           les ~80 tags OSM gardés (voir 'features.parquet', produit par osm_extraction.py)
osm_extraction.py       extraction en 2 passes d'un .osm.pbf (passe A tags légère sur tout ; passe B
                         géométrie coûteuse, réservée aux tronçons retenus - voir le README ci-dessus)
etiquettes.py            le vocabulaire GéoVélo (hiérarchie de protection, familles d'aménagement) -
                         utilisé pour ENTRAÎNER (pas exécutable ici) et pour INTERPRÉTER les prédictions
caracteristiques.py      tags OSM bruts -> matrice de variables (vocabulaire figé à l'entraînement)
entrainement.py           MÉTHODE d'entraînement (2 étages, LightGBM) - port fidèle, non exécutable ici
application_modele.py    charge les modèles vendorisés et prédit inclusion + 10 attributs
artefacts.py             vérifie que les modèles vendorisés sont bien là, affiche leurs métriques
decoupe_communes.py      répartit un tronçon entre les communes qu'il traverse
longueurs.py             agrège la découpe en longueur effective (km) par commune - FOB et FOB_AM
osm_telechargement.py    télécharge un snapshot Geofabrik daté (si besoin d'une année précise)
snapshot.py               enchaîne tout ce qui précède pour UN extrait OSM, UNE année
schemas_fob.py            à quoi ressemblent les tables produites par ce dossier
pipeline.py                la commande "tout faire" (config en haut, enchaînement en bas)
```

## Comment le lancer

- **`pipeline.py`** (`python pipeline.py`) : traite les années de `ANNEES` (2024 par défaut) avec
  l'extrait OSM déjà téléchargé par l'étape 00. Nécessite `models/fob/` (vendorisé) et
  `data/donnees_valides/zones/communes.parquet` (étape 00).
- **`02_FOB.ipynb`** : le tableau de bord de la méthode (section 0 - hyperparamètres, découpage
  entraînement/validation/test, qualité du modèle d'inclusion et des 10 modèles d'attribut), puis
  chaque étape pas à pas avec les vrais résultats, l'analyse des aménagements DÉDIÉS uniquement (quels
  TYPES d'infrastructure composent FOB_AM, voir `etiquettes.FAMILLE` - la part de FOB sans aménagement
  dédié est reportée à part, jamais comptée comme une "famille"), et un bilan (section 6).

## Où vont les données

```
data/donnees_valides/reseau_fob/<annee>/
    tags/                          sortie de la passe A (features.parquet, ids.npy, node_refs.bin...)
    ways.parquet                    tronçons FOB retenus + leurs 10 attributs prédits
    geometrie/geometries.parquet    géométrie (WKB) des tronçons localisés
    decoupe/                        pieces.parquet, way_totals.parquet (voir decoupe_communes.py)
    longueurs_fob.parquet           longueur par commune, réseau FOB entier
    longueurs_fob_amenagements.parquet   longueur par commune, aménagements dédiés seulement (FOB_AM)
    meta.json                       résumé de contrôle (nombre de tronçons, km, etc.)
```
