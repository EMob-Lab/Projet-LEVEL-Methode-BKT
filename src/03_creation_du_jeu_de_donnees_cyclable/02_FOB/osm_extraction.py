# -*- coding: utf-8 -*-
"""Extraction en DEUX PASSES depuis un extrait OpenStreetMap France (.osm.pbf), pensée pour tourner sur
un poste de travail - AUCUN index de tous les nœuds de France en mémoire (contrairement à
'00_methode_primitive/extraction_zone.py', qui peut se le permettre sur une petite zone).

**Passe A - tags** ('extraire_tags') : pour chaque tronçon 'highway=*', garde ~80 tags (voir
'osm_schema.CLES') sous forme de codes entiers compacts (un vocabulaire par clé, construit à la volée),
le meilleur réseau 'route=bicycle' auquel il appartient, des indicateurs de nom/référence, et les
références de ses nœuds (déversées sur disque, 'node_refs.bin' - il y en a des centaines de millions,
bien trop pour la RAM).

**Passe B - géométrie** ('localiser_troncons') : pour les tronçons CANDIDATS seulement (ceux que le
modèle d'inclusion juge probables, voir 'application_modele.py' - un sous-ensemble bien plus petit),
relit les nœuds dont on a besoin (voir '00_transformation_des_donnees/outils.py',
'positions_des_noeuds'/'apparier_noeuds' - les mêmes fonctions que l'étape 00 utilise déjà pour les
cours d'eau/mairies) et en déduit longueur, point représentatif, et géométrie complète (pour la découpe
communale, voir 'decoupe_communes.py')."""

from __future__ import annotations

import json
import sys
import time
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osmium
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from osm_schema import CLES, INDEX_CLES, RANG_RESEAU, type_de_nom, type_de_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "00_transformation_des_donnees"))
from outils import apparier_noeuds, positions_des_noeuds  # noqa: E402

CAP_CODES = 32_000  # nombre max de valeurs distinctes gardées par clé (codes int16) - au-delà -> "(autre)"
VIDAGE_REFS = 4_000_000  # références de nœuds tamponnées avant écriture sur disque
RAYON_TERRE_M = 6_371_008.8
FICHIER_REFS_NOEUDS = "node_refs.bin"


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# passe A - tags
# ═══════════════════════════════════════════════════════════════════════════════════════════════
class _CollecteurTags(osmium.SimpleHandler):
    """Lit les tronçons 'highway' et les relations d'itinéraires cyclables en UN SEUL passage du
    fichier (osmium appelle 'way'/'relation' pour chaque élément du bon type - noms imposés par l'API
    d'osmium, pas renommés en français ici)."""

    def __init__(self, chemin_refs: Path):
        super().__init__()
        n_cles = len(CLES)
        self.codes = array("h")  # n_cles codes par tronçon, aplatis
        self.ids = array("q")
        self.type_nom = array("b")
        self.type_ref = array("b")
        self.n_noeuds = array("i")
        self.vocabulaire: list[dict[str, int]] = [{} for _ in range(n_cles)]
        self._ligne_vide = [-1] * n_cles
        self.compteur_cles: Counter[str] = Counter()  # usage de CHAQUE clé, gardée ou non
        self.rang_relation: dict[int, int] = {}  # id du tronçon -> meilleur rang de réseau
        self.compte_relation: Counter[int] = Counter()
        self.n_troncons = 0
        self.n_relations = 0
        self._fichier_refs = open(chemin_refs, "wb")
        self._tampon_refs = array("q")
        self._t0 = time.time()

    def way(self, troncon):
        tags = troncon.tags
        if "highway" not in tags:
            return
        self.n_troncons += 1
        self.ids.append(troncon.id)
        ligne = self._ligne_vide[:]
        nom = ref = None
        for tag in tags:
            self.compteur_cles[tag.k] += 1
            j = INDEX_CLES.get(tag.k)
            if j is not None:
                vocab = self.vocabulaire[j]
                code = vocab.get(tag.v)
                if code is None:
                    code = len(vocab) if len(vocab) < CAP_CODES else CAP_CODES
                    if code < CAP_CODES:
                        vocab[tag.v] = code
                ligne[j] = code
            elif tag.k == "name":
                nom = tag.v
            elif tag.k == "ref":
                ref = tag.v
        self.codes.extend(ligne)
        self.type_nom.append(type_de_nom(nom))
        self.type_ref.append(type_de_reference(ref))
        refs = [n.ref for n in troncon.nodes]
        self.n_noeuds.append(len(refs))
        self._tampon_refs.extend(refs)
        if len(self._tampon_refs) > VIDAGE_REFS:
            self._tampon_refs.tofile(self._fichier_refs)
            self._tampon_refs = array("q")
        if self.n_troncons % 2_000_000 == 0:
            print(f"  {self.n_troncons:,} tronçons 'highway' analysés ({time.time() - self._t0:.0f}s)")

    def relation(self, rel):
        tags = rel.tags
        if tags.get("type") != "route" or tags.get("route") != "bicycle":
            return
        self.n_relations += 1
        rang = RANG_RESEAU.get(tags.get("network", ""), 0)
        for membre in rel.members:
            if membre.type == "w":
                if rang > self.rang_relation.get(membre.ref, -1):
                    self.rang_relation[membre.ref] = rang
                self.compte_relation[membre.ref] += 1

    def fermer(self):
        if len(self._tampon_refs):
            self._tampon_refs.tofile(self._fichier_refs)
        self._fichier_refs.close()


@dataclass
class TableTags:
    """Résultat de la passe A. Les tableaux sont alignés : la ligne i décrit le tronçon 'ids[i]'."""

    ids: np.ndarray
    codes: np.ndarray  # (n_troncons, n_cles) int16, -1 = tag absent
    vocabulaire: list[list[str]]  # par clé : la valeur de chaque code
    n_noeuds: np.ndarray
    type_nom: np.ndarray
    type_ref: np.ndarray
    rang_relation: np.ndarray  # -1 = dans aucune relation cyclable
    compte_relation: np.ndarray
    compteur_cles: Counter
    n_relations: int
    chemin_refs: Path

    @property
    def n_troncons(self) -> int:
        return len(self.ids)


def extraire_tags(pbf: str | Path, dossier_travail: str | Path) -> TableTags:
    """Passe A. Déverse les références de nœuds de chaque tronçon dans 'dossier_travail/node_refs.bin'."""
    dossier_travail = Path(dossier_travail)
    dossier_travail.mkdir(parents=True, exist_ok=True)
    chemin_refs = dossier_travail / FICHIER_REFS_NOEUDS
    collecteur = _CollecteurTags(chemin_refs)
    t0 = time.time()
    collecteur.apply_file(str(pbf))
    collecteur.fermer()
    n = collecteur.n_troncons
    print(f"[extraction] passe A : {n:,} tronçons 'highway', {collecteur.n_relations:,} relations cyclables, {len(collecteur.rang_relation):,} tronçons dans une relation ({time.time() - t0:.0f}s)")
    ids = np.frombuffer(collecteur.ids, dtype=np.int64)
    return TableTags(
        ids=ids,
        codes=np.frombuffer(collecteur.codes, dtype=np.int16).reshape(n, len(CLES)),
        vocabulaire=[sorted(v, key=v.get) for v in collecteur.vocabulaire],
        n_noeuds=np.frombuffer(collecteur.n_noeuds, dtype=np.int32),
        type_nom=np.frombuffer(collecteur.type_nom, dtype=np.int8),
        type_ref=np.frombuffer(collecteur.type_ref, dtype=np.int8),
        rang_relation=np.array([collecteur.rang_relation.get(int(i), -1) for i in ids], dtype=np.int8),
        compte_relation=np.array([collecteur.compte_relation.get(int(i), 0) for i in ids], dtype=np.int16),
        compteur_cles=collecteur.compteur_cles,
        n_relations=collecteur.n_relations,
        chemin_refs=chemin_refs,
    )


def ecrire_table_tags(table: TableTags, dossier_sortie: str | Path) -> None:
    """Persiste la passe A : 'features.parquet' (tags encodés en dictionnaire), comptages de
    vocabulaire, ids, méta - tout ce dont 'caracteristiques.construire_spec'/'vers_matrice' ont besoin."""
    dossier_sortie = Path(dossier_sortie)
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    tableaux, noms = [pa.array(table.ids)], ["id_osm"]
    comptages_valeurs: list[tuple[str, str, int]] = []
    for j, cle in enumerate(CLES):
        col, vocab = table.codes[:, j], table.vocabulaire[j]
        present = col >= 0
        comptes = np.bincount(col[present].astype(np.int64), minlength=len(vocab) + 1)
        for code in np.argsort(-comptes)[:500]:
            if comptes[code] > 0:
                comptages_valeurs.append((cle, vocab[code] if code < len(vocab) else "(autre)", int(comptes[code])))
        index = pa.array(np.where(present, col, 0).astype(np.int16), mask=~present)
        tableaux.append(pa.DictionaryArray.from_arrays(index, pa.array(vocab + ["(autre)"], type=pa.string())))
        noms.append(cle)
    for nom in ("type_nom", "type_ref", "n_noeuds", "rang_relation", "compte_relation"):
        tableaux.append(pa.array(getattr(table, nom)))
        noms.append({"type_nom": "name_kind", "type_ref": "ref_kind", "n_noeuds": "n_nodes", "rang_relation": "rel_rank", "compte_relation": "rel_count"}[nom])
    pq.write_table(pa.Table.from_arrays(tableaux, names=noms), dossier_sortie / "features.parquet", compression="zstd")
    pd.DataFrame(comptages_valeurs, columns=["key", "value", "n_ways"]).to_parquet(dossier_sortie / "value_counts.parquet")
    pd.DataFrame(table.compteur_cles.most_common(), columns=["key", "n_ways"]).to_parquet(dossier_sortie / "key_counts.parquet")
    np.save(dossier_sortie / "ids.npy", table.ids)
    np.save(dossier_sortie / "n_nodes.npy", table.n_noeuds)
    (dossier_sortie / "meta.json").write_text(json.dumps({"n_ways": table.n_troncons, "n_rel_bike": table.n_relations, "keys": list(CLES)}))


def lire_table_tags(dossier_tags: str | Path, colonnes: list[str] | None = None) -> pd.DataFrame:
    """Recharge la table de variables brute (encodée en dictionnaire) écrite par 'ecrire_table_tags'."""
    return pq.read_table(Path(dossier_tags) / "features.parquet", columns=colonnes).to_pandas()


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# passe B - géométrie (des tronçons CANDIDATS seulement - voir application_modele.py)
# ═══════════════════════════════════════════════════════════════════════════════════════════════
def distance_haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distance à vol d'oiseau en mètres, élément par élément."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * RAYON_TERRE_M * np.arcsin(np.sqrt(a))


@dataclass
class TronconsLocalises:
    """Coordonnées des nœuds localisés d'un ensemble de tronçons, stockées à plat (pas une liste de
    listes - trop lent en Python pur à cette échelle).

    'lignes[i]' est la position du i-ème tronçon dans les tableaux de la passe A ; ses points sont
    'lon/lat[offsets[i]:offsets[i+1]]' dans l'ordre du tronçon. Un nœud absent du fichier est ignoré."""

    lignes: np.ndarray
    offsets: np.ndarray
    lon: np.ndarray
    lat: np.ndarray

    @property
    def comptes(self) -> np.ndarray:
        return np.diff(self.offsets)

    def longueurs_m(self) -> np.ndarray:
        """Longueur haversine de chaque tronçon ; NaN si moins de deux nœuds ont pu être localisés."""
        comptes = self.comptes
        troncon_du_point = np.repeat(np.arange(len(comptes)), comptes)
        pas = distance_haversine_m(self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:])
        pas = np.where(troncon_du_point[1:] == troncon_du_point[:-1], pas, 0.0)
        longueurs = np.bincount(troncon_du_point[:-1], weights=pas, minlength=len(comptes))
        return np.where(comptes >= 2, longueurs, np.nan)

    def points_medians(self) -> tuple[np.ndarray, np.ndarray]:
        """(lat, lon) du nœud localisé du milieu de chaque tronçon (NaN si < 2 nœuds)."""
        comptes = self.comptes
        idx = np.minimum(self.offsets[:-1] + comptes // 2, max(len(self.lat) - 1, 0))
        ok = comptes >= 2
        return np.where(ok, self.lat[idx], np.nan), np.where(ok, self.lon[idx], np.nan)

    def vers_wkb(self, selection: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """LineString WGS84 (WKB) des tronçons sélectionnés (par défaut : tous ceux avec >= 2 points) ->
        (positions, wkb) - pour la découpe communale (voir 'decoupe_communes.py')."""
        comptes = self.comptes
        garder = comptes >= 2 if selection is None else selection & (comptes >= 2)
        pos = np.flatnonzero(garder)
        masque_points = np.repeat(garder, comptes)
        coords = np.column_stack([self.lon[masque_points], self.lat[masque_points]])
        lignes = shapely.linestrings(coords, indices=np.repeat(np.arange(len(pos)), comptes[pos]))
        return pos, shapely.to_wkb(lignes)


def localiser_troncons(pbf: str | Path, chemin_refs: str | Path, ids_troncons: np.ndarray, n_noeuds: np.ndarray, cible: np.ndarray) -> TronconsLocalises:
    """Passe B : résout les coordonnées des nœuds des tronçons 'cible' (masque booléen sur les lignes
    de la passe A) - réutilise 'positions_des_noeuds'/'apparier_noeuds' de
    '00_transformation_des_donnees/outils.py' (même principe que pour les cours d'eau/mairies : pas
    d'index complet des nœuds en mémoire, seulement ceux qu'on sait déjà vouloir)."""
    offsets = np.zeros(len(ids_troncons) + 1, dtype=np.int64)
    np.cumsum(n_noeuds, out=offsets[1:])
    refs = np.memmap(chemin_refs, dtype=np.int64, mode="r")
    lignes = np.flatnonzero(cible)
    morceaux = [np.asarray(refs[offsets[i] : offsets[i + 1]]) for i in lignes]
    refs_a_plat = np.concatenate(morceaux) if morceaux else np.empty(0, dtype=np.int64)
    troncon_de_ref = np.repeat(np.arange(len(lignes)), [len(m) for m in morceaux])
    voulus = np.unique(refs_a_plat)
    print(f"[extraction] passe B : {len(lignes):,} tronçons, {len(voulus):,} nœuds distincts à localiser")

    t0 = time.time()
    trouve_id, trouve_lat, trouve_lon = positions_des_noeuds(pbf, voulus)
    print(f"  {len(trouve_id):,} / {len(voulus):,} nœuds trouvés ({time.time() - t0:.0f}s)")
    position, ok = apparier_noeuds(trouve_id, refs_a_plat)
    comptes = np.bincount(troncon_de_ref[ok], minlength=len(lignes))
    offsets_sortie = np.concatenate([[0], np.cumsum(comptes)])
    return TronconsLocalises(lignes=lignes, offsets=offsets_sortie, lon=trouve_lon[position[ok]], lat=trouve_lat[position[ok]])
