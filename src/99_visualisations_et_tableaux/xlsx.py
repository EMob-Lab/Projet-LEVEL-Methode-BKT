# -*- coding: utf-8 -*-
"""Utilitaires xlsxwriter : formats cohérents (Arial), tables, KPI, graphiques - la brique réutilisée
par 'export_excel.py' pour construire le classeur final. Code générique (pas spécifique au BKT)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name, xl_rowcol_to_cell

POLICE = "Arial"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#8a8a8a"]
ENCRE, ENCRE2, ATTENUE, GRILLE = "#0b0b0b", "#52514e", "#898781", "#e1e0da"
FOND_ENTETE, TEXTE_ENTETE, BANDE = "#17324d", "#ffffff", "#f4f3ef"
BON, ATTENTION, MAUVAIS = "#1baf7a", "#eda100", "#d03b3b"


class Classeur:
    """Enveloppe fine autour de 'xlsxwriter.Workbook' : formats mis en cache (une police, une taille
    par défaut), et des méthodes de haut niveau (titre, tableau, KPI, graphique) plutôt que d'écrire
    cellule par cellule à chaque feuille."""

    def __init__(self, chemin: str):
        self.wb = xlsxwriter.Workbook(chemin, {"nan_inf_to_errors": False, "strings_to_numbers": False})
        self._formats: dict = {}
        self.wb.set_properties({"author": "VV6", "comments": "Généré par le pipeline BKT"})

    def format(self, **kw):
        cle = tuple(sorted(kw.items()))
        if cle not in self._formats:
            base = {"font_name": POLICE, "font_size": 10, "valign": "vcenter"}
            base.update(kw)
            self._formats[cle] = self.wb.add_format(base)
        return self._formats[cle]

    def feuille(self, nom: str, onglet=None, zoom=90, quadrillage=False):
        ws = self.wb.add_worksheet(nom)
        ws.hide_gridlines(0 if quadrillage else 2)
        ws.set_zoom(zoom)
        if onglet:
            ws.set_tab_color(onglet)
        return ws

    def fermer(self):
        self.wb.close()

    # ---------- textes ----------
    def titre(self, ws, ligne, texte, sous_texte=None, largeur=12):
        ws.set_row(ligne, 30)
        ws.merge_range(ligne, 0, ligne, largeur, texte, self.format(bold=True, font_size=18, font_color=FOND_ENTETE))
        if sous_texte:
            ws.merge_range(ligne + 1, 0, ligne + 1, largeur, sous_texte, self.format(font_size=10, font_color=ENCRE2, italic=True))
        return ligne + (3 if sous_texte else 2)

    def sous_titre(self, ws, ligne, texte, col=0, largeur=12):
        ws.merge_range(ligne, col, ligne, col + largeur, texte, self.format(bold=True, font_size=12, font_color="#ffffff", bg_color=FOND_ENTETE, indent=1))
        return ligne + 1

    def note(self, ws, ligne, texte, col=0, largeur=12, hauteur=None, italique=True):
        f = self.format(font_color=ENCRE2, italic=italique, text_wrap=True, valign="top", font_size=9)
        ws.merge_range(ligne, col, ligne, col + largeur, texte, f)
        n = max(1, math.ceil(len(texte) / (12 * (largeur + 1) * 1.15)))
        ws.set_row(ligne, hauteur or 13 * n + 2)
        return ligne + 1

    def paragraphe(self, ws, ligne, texte, col=0, largeur=12):
        f = self.format(text_wrap=True, valign="top")
        ws.merge_range(ligne, col, ligne, col + largeur, texte, f)
        n = max(1, math.ceil(len(texte) / (11 * (largeur + 1) * 1.1)))
        ws.set_row(ligne, 14 * n + 3)
        return ligne + 1

    def kpi(self, ws, ligne, col, libelle, valeur, format_nombre="#,##0", couleur=FOND_ENTETE, largeur=2, sous_texte=None, valeur_en_cache=None):
        """Tuile KPI : libellé (petit) au-dessus, valeur (grande) en dessous."""
        ws.merge_range(ligne, col, ligne, col + largeur - 1, libelle, self.format(font_size=9, font_color=ENCRE2, bg_color=BANDE, align="center", top=1, top_color=couleur, text_wrap=True))
        ws.set_row(ligne, 26)
        ws.set_row(ligne + 1, 30)
        f = self.format(bold=True, font_size=18, font_color=couleur, bg_color=BANDE, align="center", num_format=format_nombre)
        if isinstance(valeur, str) and valeur.startswith("="):
            ws.merge_range(ligne + 1, col, ligne + 1, col + largeur - 1, "", f)
            ws.write_formula(ligne + 1, col, valeur, f, valeur_en_cache if valeur_en_cache is not None else 0)
        else:
            ws.merge_range(ligne + 1, col, ligne + 1, col + largeur - 1, valeur, f)
        if sous_texte:
            ws.merge_range(ligne + 2, col, ligne + 2, col + largeur - 1, sous_texte, self.format(font_size=8, font_color=ATTENUE, bg_color=BANDE, align="center", text_wrap=True))
        return ligne + (3 if sous_texte else 2)

    # ---------- tableaux ----------
    def tableau(
        self,
        ws,
        ligne,
        col,
        df: pd.DataFrame,
        formats: dict | None = None,
        largeurs: dict | None = None,
        format_entete=None,
        bande=True,
        autofiltre=False,
        max_lignes=None,
        entete_retour_ligne=True,
        premiere_colonne_grasse=False,
        hauteur_entete=None,
    ):
        """Écrit un DataFrame. 'formats' : {colonne: format_nombre}. Retourne (ligne_fin_exclusive, col_fin_exclusive)."""
        formats = formats or {}
        hf = format_entete or self.format(bold=True, font_color=TEXTE_ENTETE, bg_color=FOND_ENTETE, align="center", text_wrap=entete_retour_ligne, border=1, border_color=FOND_ENTETE)
        if max_lignes:
            df = df.head(max_lignes)
        ws.set_row(ligne, hauteur_entete or (30 if entete_retour_ligne else 18))
        for j, c in enumerate(df.columns):
            ws.write(ligne, col + j, str(c), hf)
        for i, rec in enumerate(df.itertuples(index=False), start=1):
            bg = BANDE if (bande and i % 2 == 0) else None
            for j, v in enumerate(rec):
                c = df.columns[j]
                kw = {"bottom": 1, "bottom_color": GRILLE}
                if bg:
                    kw["bg_color"] = bg
                nf = formats.get(c)
                if nf:
                    kw["num_format"] = nf
                if premiere_colonne_grasse and j == 0:
                    kw["bold"] = True
                f = self.format(**kw)
                self._ecrire(ws, ligne + i, col + j, v, f)
        if autofiltre:
            ws.autofilter(ligne, col, ligne + len(df), col + len(df.columns) - 1)
        if largeurs:
            for c, w in largeurs.items():
                j = list(df.columns).index(c)
                ws.set_column(col + j, col + j, w)
        return ligne + 1 + len(df), col + len(df.columns)

    @staticmethod
    def _ecrire(ws, r, c, v, f):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or v is pd.NA or v is pd.NaT:
            ws.write_blank(r, c, None, f)
        elif isinstance(v, (np.integer,)):
            ws.write_number(r, c, int(v), f)
        elif isinstance(v, (np.floating, float)):
            ws.write_number(r, c, float(v), f)
        elif isinstance(v, (bool, np.bool_)):
            ws.write_string(r, c, "oui" if v else "non", f)
        elif isinstance(v, (int,)):
            ws.write_number(r, c, v, f)
        else:
            ws.write_string(r, c, str(v), f)

    # ---------- graphiques ----------
    def graphique(self, type_, titre, sous_type=None, largeur=620, hauteur=340, legende="bottom", titre_y=None, titre_x=None, format_y=None, format_x=None, y_min=None, y_max=None, log=False):
        opts = {"type": type_}
        if sous_type:
            opts["subtype"] = sous_type
        ch = self.wb.add_chart(opts)
        ch.set_title({"name": titre, "name_font": {"name": POLICE, "size": 12, "bold": True, "color": FOND_ENTETE}, "overlay": False})
        ch.set_size({"width": largeur, "height": hauteur})
        ch.set_chartarea({"border": {"none": True}, "fill": {"color": "#ffffff"}})
        ch.set_plotarea({"fill": {"none": True}})
        if legende:
            ch.set_legend({"position": legende, "font": {"name": POLICE, "size": 9, "color": ENCRE2}})
        else:
            ch.set_legend({"none": True})
        ax = {"name_font": {"name": POLICE, "size": 9, "color": ENCRE2}, "num_font": {"name": POLICE, "size": 9, "color": ENCRE2}, "line": {"color": GRILLE}, "major_gridlines": {"visible": False}}
        yax = dict(ax, major_gridlines={"visible": True, "line": {"color": GRILLE, "width": 0.75}})
        if titre_y:
            yax["name"] = titre_y
        if titre_x:
            ax["name"] = titre_x
        if format_y:
            yax["num_format"] = format_y
        if format_x:
            ax["num_format"] = format_x
        if y_min is not None:
            yax["min"] = y_min
        if y_max is not None:
            yax["max"] = y_max
        if log:
            yax["log_base"] = 10
        if type_ == "bar":  # barres horizontales : axes permutés
            ch.set_x_axis(yax)
            ch.set_y_axis(dict(ax, reverse=True))
        else:
            ch.set_x_axis(ax)
            ch.set_y_axis(yax)
        return ch


def serie(reference_nom, categories, valeurs, couleur, type_="col", marqueur=True, etiquettes=False, epaisseur=2.25, **extra):
    s = {"name": reference_nom, "categories": categories, "values": valeurs}
    if type_ in ("col", "bar"):
        s["fill"] = {"color": couleur}
        s["border"] = {"color": "#ffffff", "width": 0.75}
        s["gap"] = 60
    else:
        s["line"] = {"color": couleur, "width": epaisseur}
        s["marker"] = {"type": "circle", "size": 6, "border": {"color": "#ffffff"}, "fill": {"color": couleur}} if marqueur else {"type": "none"}
    if etiquettes:
        s["data_labels"] = {"value": True, "font": {"name": POLICE, "size": 8, "color": ENCRE2}}
    s.update(extra)
    return s


def plage(feuille: str, r1, c1, r2=None, c2=None):
    """Référence de plage [feuille, r1, c1, r2, c2] pour xlsxwriter."""
    return [feuille, r1, c1, r2 if r2 is not None else r1, c2 if c2 is not None else c1]


def nom_colonne(c):
    return xl_col_to_name(c)


def cellule(r, c, absolu=False):
    return xl_rowcol_to_cell(r, c, row_abs=absolu, col_abs=absolu)
