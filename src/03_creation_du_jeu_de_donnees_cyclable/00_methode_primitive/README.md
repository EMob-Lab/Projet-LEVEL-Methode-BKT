# 00_methode_primitive

La toute première tentative du projet pour distinguer un tronçon OpenStreetMap cyclable d'un tronçon
qui ne l'est pas - trois étapes, sur une petite zone de test (2 km autour de Lyon, puis Grenoble) :

1. **Annotation manuelle** (voir `annotation_manuelle/`) : une webapp statique, autonome, sans serveur
   (`map_selectable_segments.html`, un simple fichier `.html` ouvert dans un navigateur) affiche les
   ~2 000 tronçons OSM de la zone sur une carte Leaflet et permet de les parcourir un par un (bouton
   "Démarrer l'itération", raccourcis clavier) pour marquer à la main s'ils sont cyclables. Le résultat
   est exporté en CSV côté client (bouton "Sauvegarder la liste") : `selected_segment_by_hand.csv`,
   433 tronçons annotés - la vérité terrain de cette méthode.
2. **Filtre à règles écrites à la main** (`filtre_regles.py`) : classe un tronçon en
   cyclable / non_cyclable / incertain à partir de 6 tags seulement (`highway`, `bicycle`, `cycleway`,
   `access`, `maxspeed`, `surface`) - pas de modèle appris, des règles fixes écrites par nous.
3. **Propagation spatiale** (`propagation_spatiale.py`) : rattrape une partie des "incertain" par
   proximité géométrique/nom de rue avec un tronçon déjà classé cyclable.

Le filtre est ensuite évalué en le faisant tourner sur les mêmes tronçons que l'annotation manuelle et
en comparant ses prédictions à `selected_segment_by_hand.csv` (précision/rappel par classe) - c'est
cette évaluation qui a montré les limites de l'approche (voir ci-dessous).

**Abandonnée au profit de FOB** (voir `../02_FOB/`) - gardée ici pour montrer l'évolution des méthodes
du projet, voir `../README.md`. N'est PAS utilisée par les étapes suivantes du projet.

## Pourquoi elle a été abandonnée

Un filtre à règles fixes ne capture que ce que ses auteurs ont pensé à écrire, avec seulement 6 tags en
entrée : évalué contre la vérité terrain de `annotation_manuelle/selected_segment_by_hand.csv`, une
bonne partie des tronçons de la zone de test tombent dans la catégorie fourre-tout "incertain" - ni
assez de signal positif, ni assez de signal négatif pour trancher (voir l'évaluation d'origine,
`V2/BDD/by_hand_selection_analysis.ipynb`, non reproduite ici - cette méthode n'est gardée que pour
comparaison, voir `../README.md`). FOB (`../02_FOB/`) apprend ces règles à partir de vrais exemples
GéoVélo plutôt que de les écrire à la main, à l'échelle de la France entière, et exploite ~80 tags au
lieu de 6.

## Le dossier, fichier par fichier

```
annotation_manuelle/
    map_selectable_segments.html   la webapp d'annotation (statique, sans serveur - ouvrir dans un
                                    navigateur) telle qu'utilisée à l'origine, vendorisée telle quelle
    selected_segment_by_hand.csv   son export : 433 tronçons annotés à la main - la vérité terrain
filtre_regles.py         classe un tronçon en cyclable / non_cyclable / incertain (6 tags, règles fixes)
propagation_spatiale.py  rattrape une partie des "incertain" : même nom de rue qu'un tronçon cyclable
                          proche -> reclassé cyclable (prudent : jamais par simple contact géométrique)
extraction_zone.py       extrait une petite zone (rectangle lat/lon) d'un extrait OSM France - pour
                          tester ce dossier sur un vrai quartier sans traiter la France entière
```

## Comment le lancer

`00_methode_primitive.ipynb` montre `filtre_regles.py`/`propagation_spatiale.py` sur une vraie
extraction (zone de 2 km autour de Lyon, centre `45.78107, 4.89758` - les mêmes coordonnées que
l'annotation manuelle d'origine), à partir de l'extrait OSM déjà téléchargé par l'étape 00
(`00_transformation_des_donnees`). La webapp `annotation_manuelle/map_selectable_segments.html` se
lance en l'ouvrant simplement dans un navigateur (aucune dépendance, aucun serveur) - elle n'est pas
appelée par le notebook, elle est vendorisée comme trace de la méthode d'origine.

Pas de `pipeline.py` dans ce dossier : cette méthode n'a jamais tourné à l'échelle de la France (voir
"pourquoi elle a été abandonnée" ci-dessus) - il n'y a donc rien à "lancer en un coup" au sens des
autres dossiers du projet, seulement une démonstration sur une petite zone.
