"""Petits utilitaires partagés par plusieurs fichiers de ce dossier (codage du débit, téléchargement,
lecture ciblée d'un fichier OSM). Rien ici n'est une "configuration" - ce sont des fonctions pures,
sans chemin ni paramètre propre au projet : la configuration (quels fichiers, quelles années...) vit
uniquement dans 'pipeline.py'.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from array import array
from pathlib import Path

import numpy as np

# ============================================================================= débit horaire : stockage compact
#
# Un débit horaire est stocké en entier 16 bits pour économiser la place (des millions d'heures x
# capteurs à garder) : chaque valeur réelle est multipliée par 100 et arrondie, 65535 signifie "heure
# manquante" (une vraie mesure ne peut jamais valoir exactement 655.35 passages).

FLOW_SCALE = 100
FLOW_MISSING_SENTINEL = 65535


def encoder_debit(valeurs) -> np.ndarray:
    """Transforme des comptages horaires (NaN = heure manquante) en entiers compacts (uint16)."""
    tableau = np.asarray(valeurs, dtype=np.float32)
    code = np.round(tableau * FLOW_SCALE).astype(np.uint16)
    code[np.isnan(tableau)] = FLOW_MISSING_SENTINEL
    return code


def decoder_debit(code: bytes | np.ndarray) -> np.ndarray:
    """Transforme des entiers compacts (ou leurs octets bruts) en comptages horaires réels (NaN = heure manquante)."""
    if isinstance(code, bytes):
        code = np.frombuffer(code, dtype=np.uint16)
    valeurs = code.astype(np.float32)
    manquant = code == FLOW_MISSING_SENTINEL
    valeurs /= FLOW_SCALE
    valeurs[manquant] = np.nan
    return valeurs


# ============================================================================= téléchargement reprenable
#
# Certains serveurs (Geofabrik pour OSM, CGIAR pour le relief) n'autorisent qu'une connexion à la
# fois et coupent parfois en cours de route : 'telecharger' reprend un fichier interrompu là où il
# s'est arrêté plutôt que de tout recommencer.

TAILLE_BLOC = 1 << 20
MAX_TENTATIVES = 6


def _taille_distante(url: str) -> int | None:
    requete = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(requete, timeout=60) as reponse:
        taille = reponse.headers.get("Content-Length")
    return int(taille) if taille else None


def telecharger(url: str, cible: Path) -> Path:
    """Récupère 'url' dans le fichier 'cible', en reprenant un téléchargement partiel si besoin."""
    cible.parent.mkdir(parents=True, exist_ok=True)
    partiel = cible.with_suffix(cible.suffix + ".part")
    total = _taille_distante(url)
    for tentative in range(1, MAX_TENTATIVES + 1):
        fait = partiel.stat().st_size if partiel.exists() else 0
        if total is not None and fait == total:
            break
        try:
            requete = urllib.request.Request(
                url, headers={"Range": f"bytes={fait}-"} if fait else {}
            )
            with (
                urllib.request.urlopen(requete, timeout=120) as reponse,
                open(partiel, "ab" if fait else "wb") as sortie,
            ):
                while morceau := reponse.read(TAILLE_BLOC):
                    sortie.write(morceau)
                    fait += len(morceau)
            if total is None or partiel.stat().st_size == total:
                break
        except (urllib.error.URLError, ConnectionError, TimeoutError) as erreur:
            print(
                f"  [telecharger] tentative {tentative}/{MAX_TENTATIVES} échouée ({erreur}), nouvel essai..."
            )
            time.sleep(10)
    else:
        raise RuntimeError(f"impossible de télécharger {url}")
    partiel.replace(cible)
    print(f"  [telecharger] {cible.name} ({cible.stat().st_size / 1e6:.0f} Mo)")
    return cible


# ============================================================================= lecture ciblée d'un fichier OSM (.osm.pbf)
#
# Un fichier .osm.pbf France contient environ un demi-milliard de nœuds : construire un index complet
# de leurs positions en mémoire est trop lourd. Comme on connaît déjà les identifiants qui nous
# intéressent (les nœuds d'une rivière, d'une mairie...), on demande directement à 'osmium' de ne lire
# QUE ces identifiants-là - beaucoup plus léger.


def positions_des_noeuds(
    pbf: str | Path, ids_noeuds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordonnées des nœuds demandés qui existent dans le fichier -> (ids, latitudes, longitudes), triés par id."""
    import osmium

    voulus = np.unique(ids_noeuds.astype(np.int64))
    trouve_id, trouve_lat, trouve_lon = array("q"), array("d"), array("d")
    lecteur = osmium.FileProcessor(str(pbf), osmium.osm.NODE).with_filter(
        osmium.filter.IdFilter(voulus.tolist())
    )
    for noeud in lecteur:
        position = noeud.location
        if position.valid():
            trouve_id.append(noeud.id)
            trouve_lat.append(position.lat)
            trouve_lon.append(position.lon)
    ids = np.frombuffer(trouve_id, dtype=np.int64)
    ordre = np.argsort(ids)
    return (
        ids[ordre],
        np.frombuffer(trouve_lat, dtype=np.float64)[ordre],
        np.frombuffer(trouve_lon, dtype=np.float64)[ordre],
    )


def apparier_noeuds(
    ids_trouves: np.ndarray, ids_demandes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pour chaque id demandé : sa position dans 'ids_trouves', et s'il a vraiment été trouvé (booléen)."""
    if len(ids_trouves) == 0:
        return np.zeros(len(ids_demandes), dtype=np.int64), np.zeros(
            len(ids_demandes), dtype=bool
        )
    position = np.minimum(
        np.searchsorted(ids_trouves, ids_demandes), len(ids_trouves) - 1
    )
    return position, ids_trouves[position] == ids_demandes
