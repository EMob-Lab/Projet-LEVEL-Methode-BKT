"""Paris, Lyon et Marseille existent dans les sources sous DEUX formes différentes : soit comme une
seule grande commune, soit comme leurs arrondissements (16 pour Paris, 9 pour Lyon, 16 pour
Marseille). La table des communes officielle utilise les arrondissements ; mais les capteurs vélo et
certains fichiers INSEE utilisent parfois la ville entière. Ce petit fichier sert à harmoniser les
deux : dès qu'une ville OU un de ses arrondissements apparaît quelque part, on considère que TOUS ses
arrondissements sont concernés.
"""

from __future__ import annotations

# Pour chaque ville : (les codes de ses arrondissements, les codes sous lesquels la ville ENTIÈRE peut apparaître)
PLM: dict[str, tuple[set[str], set[str]]] = {
    "paris": ({f"751{i:02d}" for i in range(1, 21)}, {"75056", "75100", "00751"}),
    "lyon": ({f"6938{i}" for i in range(1, 10)}, {"69123", "69380"}),
    "marseille": ({f"132{i:02d}" for i in range(1, 17)}, {"13055", "13200", "01305"}),
}

# Tous les codes d'arrondissement des 3 villes réunis, pratique pour un test rapide "est-ce un arrondissement ?"
ARRONDISSEMENTS: set[str] = set().union(
    *(arrondissements for arrondissements, _ in PLM.values())
)


def developper_plm(codes) -> set[str]:
    """Complète un ensemble de codes commune : dès qu'une ville PLM (ou un de ses arrondissements) est
    présente, ajoute TOUS les arrondissements de cette ville à l'ensemble."""
    complet = set(codes)
    for arrondissements, villes_entieres in PLM.values():
        if complet & (arrondissements | villes_entieres):
            complet.update(arrondissements)
    return complet
