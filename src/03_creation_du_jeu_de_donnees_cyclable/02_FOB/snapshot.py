# -*- coding: utf-8 -*-
"""Orchestration bout en bout : d'UN extrait OSM (.osm.pbf) à la longueur FOB par commune, pour UNE
année - enchaîne tous les autres fichiers de ce dossier dans l'ordre où ils dépendent les uns des
autres (voir le schéma du README) :

    osm_extraction.extraire_tags        -> tags de tous les tronçons 'highway' (passe A)
    application_modele.predire_candidats -> probabilité d'inclusion + attributs des candidats
    osm_extraction.localiser_troncons    -> géométrie des candidats INCLUS (passe B, ciblée)
    decoupe_communes.decouper_troncons   -> répartition par commune
    longueurs.fob_depuis_decoupe         -> longueur effective (km) par commune, FOB et FOB_AM

Idempotent comme le reste du projet : si le résultat d'une étape existe déjà sur disque, elle est
sautée (voir 'FORCER_LE_RECALCUL' dans 'pipeline.py')."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import osm_extraction as E
from application_modele import Modeles, predire_candidats
from decoupe_communes import PolygonesCommunes, decouper_troncons, ecrire_geometries
from longueurs import fob_depuis_decoupe, preparer_troncons_fob


def traiter_snapshot(pbf: Path, annee: int, dossier_annee: Path, *, modeles: Modeles, communes: PolygonesCommunes, alpha: float = 2.0, forcer: bool = False) -> dict:
    """Traite un snapshot OSM pour UNE année : écrit 'ways.parquet' (tronçons FOB avec leurs
    attributs prédits) et les sorties de la découpe communale dans 'dossier_annee'. Renvoie le résumé
    de contrôle (voir 'decoupe_communes.ResultatDecoupe.statistiques')."""
    dossier_annee.mkdir(parents=True, exist_ok=True)
    chemin_ways = dossier_annee / "ways.parquet"
    chemin_meta = dossier_annee / "meta.json"
    if chemin_ways.exists() and chemin_meta.exists() and not forcer:
        print(f"[snapshot] {annee} : déjà traité ({chemin_ways.name})")
        return json.loads(chemin_meta.read_text(encoding="utf-8"))

    # passe A : tags de tous les tronçons - écrit 'node_refs.bin' (passe B), 'ids.npy'/'n_nodes.npy' et
    # 'features.parquet' (la table lue juste après) dans 'dossier_tags' ; sautée si déjà faite.
    dossier_tags = dossier_annee / "tags"
    if not (dossier_tags / "features.parquet").exists() or forcer:
        E.ecrire_table_tags(E.extraire_tags(pbf, dossier_tags), dossier_tags)
    brut = E.lire_table_tags(dossier_tags)

    candidats = predire_candidats(brut, modeles)
    inclus = candidats[candidats["p_incl"] >= modeles.seuil].reset_index(drop=True)

    # passe B : géométrie SEULEMENT pour les tronçons inclus (bien moins nombreux que tous les candidats)
    ids_tags = np.load(dossier_tags / "ids.npy")
    n_noeuds = np.load(dossier_tags / "n_nodes.npy")
    cible = np.isin(ids_tags, inclus["id_osm"].to_numpy())
    localises = E.localiser_troncons(pbf, dossier_tags / E.FICHIER_REFS_NOEUDS, ids_tags, n_noeuds, cible)

    lat_medians, lon_medians = localises.points_medians()
    longueurs_m = localises.longueurs_m()
    ways = inclus.copy()
    ways["incl"] = True
    ways["length_m"] = pd.Series(longueurs_m, index=np.flatnonzero(cible)).reindex(ways["row"]).to_numpy()
    ways["lat"] = pd.Series(lat_medians, index=np.flatnonzero(cible)).reindex(ways["row"]).to_numpy()
    ways["lon"] = pd.Series(lon_medians, index=np.flatnonzero(cible)).reindex(ways["row"]).to_numpy()
    ways.to_parquet(chemin_ways, index=False)

    positions, wkb = localises.vers_wkb()
    geometries = ecrire_geometries(dossier_annee / "geometrie", ids_tags[np.flatnonzero(cible)[positions]], wkb, source="osm_extraction")

    troncons_fob = preparer_troncons_fob(ways, alpha)
    decoupe = decouper_troncons(troncons_fob, geometries, communes)
    decoupe.sauvegarder(dossier_annee / "decoupe")
    fob, fob_amenagements, stats = fob_depuis_decoupe(decoupe, troncons_fob, len(candidats), annee, alpha)
    fob.to_parquet(dossier_annee / "longueurs_fob.parquet", index=False)
    fob_amenagements.to_parquet(dossier_annee / "longueurs_fob_amenagements.parquet", index=False)

    meta = {"n_candidates": len(candidats), "n_incl": len(inclus), "km_incl": float(ways["length_m"].fillna(0).sum() / 1000), **stats}
    chemin_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
