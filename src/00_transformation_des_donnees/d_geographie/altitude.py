"""Relief de chaque commune, à partir du modèle numérique de terrain SRTM (résolution 90 m).

Pour chaque commune, on garde :
    - l'altitude de son CENTROÏDE (son "point central") ;
    - des STATISTIQUES ZONALES sur toute la surface de la commune : minimum, maximum, moyenne,
      médiane, écart-type, nombre de cellules du modèle touchées, et l'ÉTENDUE (max - min, le
      "dénivelé" de la commune).

Une commune est dite "montagneuse" si son étendue d'altitude dépasse 200 m (voir 'table_communes.py').

La mosaïque (l'image du relief de toute la France) est construite une seule fois, à partir de 15
tuiles téléchargées (CGIAR, ~300 Mo au total). Si une mosaïque existe déjà quelque part, on peut la
réutiliser directement pour éviter de la retélécharger.
"""

from __future__ import annotations

import io
import shutil
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterstats import zonal_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outils import telecharger  # noqa: E402

BOITE_FRANCE = (
    -5.5,
    41.0,
    10.0,
    52.0,
)  # (longitude min, latitude min, longitude max, latitude max)
URL_SRTM = (
    "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/{tuile}.zip"
)
TUILES_SRTM = [
    f"srtm_{lon:02d}_{lat:02d}" for lon in range(35, 40) for lat in range(2, 5)
]
VALEUR_MANQUANTE_MNT = -32768
STATISTIQUES_ZONALES = ["min", "max", "mean", "median", "std", "range", "count"]


def construire_mosaique(cible: Path) -> Path:
    """Télécharge les tuiles SRTM qui couvrent la France et les assemble en une seule image."""
    dossier_tuiles = cible.parent / "srtm_tiles"
    dossier_tuiles.mkdir(parents=True, exist_ok=True)
    chemins = []
    for tuile in TUILES_SRTM:
        tif = dossier_tuiles / f"{tuile}.tif"
        if not tif.exists():
            archive = telecharger(
                URL_SRTM.format(tuile=tuile), dossier_tuiles / f"{tuile}.zip"
            )
            with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as z:
                nom = next(n for n in z.namelist() if n.endswith(".tif"))
                z.extract(nom, dossier_tuiles)
                (dossier_tuiles / nom).rename(tif)
            archive.unlink()
        chemins.append(tif)
    sources = [rasterio.open(p) for p in chemins]
    mosaique, transformation = merge(sources, bounds=BOITE_FRANCE, res=0.0008333333333)
    meta = sources[0].meta.copy()
    meta.update(
        driver="GTiff",
        height=mosaique.shape[1],
        width=mosaique.shape[2],
        transform=transformation,
        compress="lzw",
    )
    with rasterio.open(cible, "w", **meta) as dst:
        dst.write(mosaique)
    for s in sources:
        s.close()
    shutil.rmtree(dossier_tuiles, ignore_errors=True)
    return cible


def obtenir_mosaique(dossier_sortie: Path, mosaique_existante: Path | None) -> Path:
    """Renvoie le chemin d'une mosaïque déjà prête (fournie ou déjà construite), sinon la construit."""
    cible = dossier_sortie / "france_srtm.tif"
    if cible.exists():
        return cible
    if mosaique_existante is not None and mosaique_existante.exists():
        return mosaique_existante
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    return construire_mosaique(cible)


def altitude_des_communes(communes: gpd.GeoDataFrame, mnt: Path) -> pd.DataFrame:
    """Coordonnées et altitude du centroïde, plus les statistiques zonales, pour chaque commune."""
    centroides = communes.geometry.centroid
    sortie = pd.DataFrame(
        {
            "code_commune": communes["code_commune"].to_numpy(),
            "cent_lat": centroides.y.to_numpy(),
            "cent_lon": centroides.x.to_numpy(),
        }
    )
    with rasterio.open(mnt) as src:
        echantillonne = np.array(
            [
                v[0]
                for v in src.sample(
                    zip(sortie["cent_lon"], sortie["cent_lat"], strict=True)
                )
            ],
            dtype=float,
        )
        sortie["alt_centroid_m"] = np.where(
            echantillonne != src.nodata, echantillonne.round(1), np.nan
        )
    stats = zonal_stats(
        communes.geometry,
        str(mnt),
        stats=STATISTIQUES_ZONALES,
        nodata=VALEUR_MANQUANTE_MNT,
        all_touched=True,
    )
    for cle in STATISTIQUES_ZONALES:
        sortie[f"alt_{cle}_m"] = [
            float(s[cle]) if s and s.get(cle) is not None else np.nan for s in stats
        ]
    sortie["alt_range_m"] = sortie["alt_max_m"] - sortie["alt_min_m"]
    return sortie


def preparer_altitude(
    dossier_communes: Path, dossier_sortie: Path, mosaique_existante: Path | None = None
) -> Path:
    """Calcule l'altitude de chaque commune et l'écrit en parquet.

    'dossier_communes' doit contenir 'communes.parquet' (préparé par 'd_zones/zones.py')."""
    mnt = obtenir_mosaique(dossier_sortie, mosaique_existante)
    communes = gpd.read_parquet(
        dossier_communes / "communes.parquet", columns=["code_commune", "geometry"]
    )
    table = altitude_des_communes(communes, mnt)
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemin = dossier_sortie / "altitude.parquet"
    table.to_parquet(chemin, index=False)
    print(f"[altitude] {len(table):,} communes (MNT {mnt.name}) -> {chemin}")
    return chemin
