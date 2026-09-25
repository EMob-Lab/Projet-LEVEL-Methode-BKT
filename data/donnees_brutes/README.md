# data/donnees_brutes/

Sources tierces, non fournies avec ce dépôt (volumineuses ou soumises à condition d'usage - voir
`00_transformation_des_donnees/README.md`). Ce fichier dit où trouver chacune. Seul `reference/` est
vendorisé (petit, pas une source tierce) et committé.

| Dossier | Fichier(s) attendu(s) | Où l'obtenir |
|---|---|---|
| `zones/` | `communes-5m.geojson`, `departements-5m.geojson`, `regions-5m.geojson`, `population_{communes,departements,regions}.csv` | [Contours administratifs](https://www.data.gouv.fr/datasets/contours-administratifs/) (Etalab, à partir de l'IGN ADMIN EXPRESS COG) |
| `population/` | `population_{2019..2025}.xlsx` | [Populations légales](https://www.insee.fr/fr/statistiques?debut=0&theme=10) (Insee) - une page par millésime, chercher "Populations légales `<année>`" |
| `counters/` | `counter_stations.xlsx`, `counter_stations_former.xlsx`, `flows_{2019..2025}.xlsx` | [Plateforme nationale des fréquentations](https://www.velo-territoires.org/observatoires/plateforme-nationale-de-frequentation/) (Vélo & Territoires / Eco-Compteur) - export sur demande, pas un simple bouton de téléchargement |
| `geography/` | `osm_reference.osm.pbf` | [Geofabrik](https://download.geofabrik.de/europe/france.html) - extrait France, `.osm.pbf` (voir aussi `03_creation_du_jeu_de_donnees_cyclable/02_FOB/osm_telechargement.py`, qui le télécharge automatiquement) |
| `socioeconomic/` | `density_grid.xlsx` | [La grille de densité](https://www.insee.fr/fr/information/6439600) (Insee, grille communale à 7 niveaux) |
| `socioeconomic/` | `median_income.csv` | [Revenu des français à la commune](https://www.data.gouv.fr/datasets/revenu-des-francais-a-la-commune/) (data.gouv.fr, source Insee-Filosofi) |
| `socioeconomic/` | `cycle_infrastructure.csv` | [Aménagements cyclables - France métropolitaine](https://transport.data.gouv.fr/datasets/amenagements-cyclables-france-metropolitaine) (Base Nationale des Aménagements Cyclables, transport.data.gouv.fr) |
| `socioeconomic/` | `bike_welcome.csv` | [Offices de tourisme labellisés Accueil Vélo](https://www.data.gouv.fr/datasets/offices-de-tourisme-et-bureaux-dinformation-touristique-labellises-pour-laccueil-velo/) (data.gouv.fr, France Vélo Tourisme) |
| *(réseau cyclable, étape 03)* | exports GéoVélo bruts (`01_geovelo_seul/`), labels GéoVélo (`02_FOB/entrainement.py`) | [Organisation Geovelo](https://www.data.gouv.fr/organizations/geovelo/datasets/) (data.gouv.fr) |
| *(relief, `altitude.py`)* | tuiles SRTM (téléchargées automatiquement) | [SRTM 90m - CGIAR-CSI](https://srtm.csi.cgiar.org/) |

Les deux dernières lignes n'ont pas besoin d'un dépôt manuel dans `donnees_brutes/` : le code les
télécharge lui-même (voir `osm_telechargement.py` et `altitude.py`).

Aucun lien ci-dessus n'est garanti stable indéfiniment (pages Insee et data.gouv.fr renommées d'un
millésime à l'autre) - en cas de lien mort, chercher le nom du jeu de données tel quel.
