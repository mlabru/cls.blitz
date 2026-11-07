# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
slm_mapbase
base comum dos mapas do slmRender.py e do slmMapper.py: figura + eixo cartopy
com oceano/terra/costa/fronteiras/estados e grade, caixa de HUD, legenda e
gravação do PNG.  Tudo num lugar só para os dois mapas não divergirem a cada
ajuste de estilo.

uso típico:
    l_figure, l_axes, l_projection = make_basemap(figsize, dpi)
    l_axes.set_extent(...)            # cada chamador escolhe o recorte
    ...                               # desenha os dados
    draw_hud(l_axes, "texto")
    style_legend(l_axes, handles, "lower right")
    finalize_and_save(l_figure, "saida.png", dpi)

2026.out  mlabru  [fix] finalize_and_save grava o PNG de forma atômica (slm_ioutil),
                  pela própria figura (f_figure.savefig)
2026.out  mlabru  refatoração para legibilidade: make_basemap dividido em
                  _add_geographic_features e _add_gridlines; comentário do
                  lru_cache corrigido (um processo por frame, não um processo
                  de vida longa); parâmetros lax/lfig renomeados
2026.ago  mlabru  initial version (Linux/Python) -- extraído de slmRender.py /
                  slmMapper.py
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import functools
import typing

# matplotLib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text

# cartoPy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# locals
import work.slm_ioutil as ioutil
import work.slm_mapdefs as mdf

# < constants >────────────────────────────────────────────────────────────────────────

# resolução das feições Natural Earth ("10m", "50m" ou "110m")
CS_NATURAL_EARTH_SCALE = "50m"

# ─────────────────────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _cached_geometries(fs_category: str, fs_name: str, fs_scale: str) -> tuple:
    """
    lê as geometrias de uma feição Natural Earth uma única vez por processo.

    [fix] ax.add_feature() com uma NaturalEarthFeature relê e reparseia o
    shapefile do disco a cada chamada.  Aqui as geometrias são lidas uma vez
    (memoizadas por categoria/nome/escala) e reembaladas por _cached_feature.

    [doc] o cache vive só dentro de um processo python.  O cron chama o
    slmRender.py num processo novo por frame (slm_common.sh, render_frame),
    então o ganho é entre as várias feições/chamadas de um mesmo processo,
    não entre frames.

    :param fs_category: "physical" ou "cultural"
    :param fs_name: nome da feição (ex.: "coastline")
    :param fs_scale: resolução (ex.: "50m")
    :returns: tupla de geometrias shapely
    """
    # o cartopy baixa o shapefile na primeira vez e depois usa o cache em disco
    ls_shapefile_path = shpreader.natural_earth(resolution=fs_scale,
                                                category=fs_category, name=fs_name)

    return tuple(shpreader.Reader(ls_shapefile_path).geometries())

# ─────────────────────────────────────────────────────────────────────────────────────
def _cached_feature(fs_category: str, fs_name: str, fs_scale: str,
                    **fdct_style) -> cfeature.ShapelyFeature:
    """
    feição cartopy com as geometrias do cache: mesmo visual de
    cfeature.NaturalEarthFeature(categoria, nome, escala), sem reler o shapefile.

    :param fs_category: "physical" ou "cultural"
    :param fs_name: nome da feição
    :param fs_scale: resolução
    :param fdct_style: estilo repassado à feição (cores, zorder...)
    :returns: ShapelyFeature pronta para ax.add_feature
    """
    ltup_geometries = _cached_geometries(fs_category, fs_name, fs_scale)

    return cfeature.ShapelyFeature(ltup_geometries, ccrs.PlateCarree(), **fdct_style)

# ─────────────────────────────────────────────────────────────────────────────────────
def make_basemap(ft_figsize: tuple, fi_dpi: int) -> tuple:
    """
    cria a figura e o eixo cartopy com as feições geográficas e a grade.

    NÃO define o recorte (extent): cada chamador aplica o seu com
    set_extent() (fixo para o Brasil no slmRender, da região pedida no slmMapper).

    :param ft_figsize: (largura, altura) da figura, em polegadas
    :param fi_dpi: resolução da figura
    :returns: (figura, eixo, projeção PlateCarree)
    """
    l_projection = ccrs.PlateCarree()

    l_figure = plt.figure(figsize=ft_figsize, dpi=fi_dpi)
    l_figure.patch.set_facecolor(mdf.CS_BACKGROUND)

    l_axes = l_figure.add_subplot(1, 1, 1, projection=l_projection)

    # cor de fundo enquanto as feições não cobrem o eixo
    l_axes.set_facecolor(mdf.CS_OCEAN_COLOR)

    _add_geographic_features(l_axes)
    _add_gridlines(l_axes, l_projection)

    return l_figure, l_axes, l_projection

# ─────────────────────────────────────────────────────────────────────────────────────
def _add_geographic_features(f_axes: Axes) -> None:
    """
    desenha oceano, terra, costa, fronteiras nacionais e limites estaduais.
    a ordem e os zorders definem o que fica por cima (não reordenar).

    :param f_axes: eixo cartopy
    """
    ls_scale = CS_NATURAL_EARTH_SCALE

    f_axes.add_feature(_cached_feature("physical", "ocean", ls_scale),
                       facecolor=mdf.CS_OCEAN_COLOR, edgecolor="none", zorder=0)
    f_axes.add_feature(_cached_feature("physical", "land", ls_scale),
                       facecolor=mdf.CS_LAND_COLOR, edgecolor="none", zorder=1)
    f_axes.add_feature(_cached_feature("physical", "coastline", ls_scale),
                       edgecolor=mdf.CS_COASTLINE_COLOR, facecolor="none",
                       linewidth=0.8, zorder=2)
    f_axes.add_feature(_cached_feature("cultural", "admin_0_boundary_lines_land", ls_scale),
                       edgecolor=mdf.CS_BORDER_COLOR, facecolor="none",
                       linewidth=0.7, linestyle="-", zorder=2)
    f_axes.add_feature(_cached_feature("cultural", "admin_1_states_provinces_lines", ls_scale),
                       edgecolor=mdf.CS_STATE_COLOR, facecolor="none",
                       linewidth=0.4, zorder=2)

# ─────────────────────────────────────────────────────────────────────────────────────
def _add_gridlines(f_axes: Axes, f_projection: ccrs.Projection) -> None:
    """
    desenha a grade lat/lng tracejada, com rótulos só embaixo e à esquerda.

    :param f_axes: eixo cartopy
    :param f_projection: projeção dos dados
    """
    l_gridliner = f_axes.gridlines(crs=f_projection,
                                   draw_labels=True,
                                   color=mdf.CS_GRID_COLOR,
                                   linestyle="--",
                                   linewidth=0.5,
                                   alpha=0.6,
                                   zorder=3)

    l_gridliner.top_labels = False
    l_gridliner.right_labels = False

    l_gridliner.xformatter = LONGITUDE_FORMATTER
    l_gridliner.yformatter = LATITUDE_FORMATTER

    l_gridliner.xlabel_style = {"color": mdf.CS_TEXT_COLOR, "fontsize": 9}
    l_gridliner.ylabel_style = {"color": mdf.CS_TEXT_COLOR, "fontsize": 9}

# ─────────────────────────────────────────────────────────────────────────────────────
def draw_hud(f_axes: Axes, fs_text: str, fi_fontsize: int = 12) -> Text:
    """
    desenha a caixa de HUD no canto superior esquerdo (fundo escuro
    semitransparente, borda branca, acima de todas as camadas).

    :param f_axes: eixo cartopy
    :param fs_text: texto do HUD (linhas separadas por \\n)
    :param fi_fontsize: tamanho da fonte
    :returns: o objeto de texto criado
    """
    return f_axes.text(0.015, 0.98, fs_text,
                       transform=f_axes.transAxes,
                       va="top",
                       fontsize=fi_fontsize,
                       color=mdf.CS_TEXT_COLOR,
                       bbox=dict(facecolor=mdf.CS_HUD_BG, alpha=0.85, edgecolor="white"),
                       zorder=10)

# ─────────────────────────────────────────────────────────────────────────────────────
def style_legend(f_axes: Axes, flst_handles: list, fs_location: str,
                 fi_fontsize: int = 11) -> Legend:
    """
    cria a legenda no estilo do projeto (fundo do HUD, texto branco, por cima
    das demais camadas).

    :param f_axes: eixo cartopy
    :param flst_handles: itens da legenda (ex.: Line2D)
    :param fs_location: posição (ex.: "lower right")
    :param fi_fontsize: tamanho da fonte
    :returns: a legenda criada
    """
    l_legend = f_axes.legend(handles=flst_handles,
                             loc=fs_location,
                             facecolor=mdf.CS_HUD_BG,
                             edgecolor="white",
                             fontsize=fi_fontsize)

    l_legend.set_zorder(10)

    for l_text in l_legend.get_texts():
        l_text.set_color(mdf.CS_TEXT_COLOR)

    return l_legend

# ─────────────────────────────────────────────────────────────────────────────────────
def finalize_and_save(f_figure: Figure, fs_output_path: str, fi_dpi: int) -> None:
    """
    aplica as margens fixas (mdf.CDCT_LAYOUT), grava o PNG e fecha a figura
    para liberar memória.

    [fix] a gravação é atômica (slm_ioutil): o PNG publicado pode estar sendo
    lido (ffmpeg, servidor web) enquanto o próximo é gravado.  Grava-se pela
    própria figura (f_figure.savefig), não pela "figura corrente" do pyplot.
    O formato é informado explicitamente porque o temporário termina em .tmp.

    :param f_figure: figura a gravar
    :param fs_output_path: caminho do PNG
    :param fi_dpi: resolução de saída
    """
    f_figure.subplots_adjust(**mdf.CDCT_LAYOUT)

    def write_png(f_file: typing.BinaryIO) -> None:
        f_figure.savefig(f_file, format="png", dpi=fi_dpi,
                         facecolor=f_figure.get_facecolor())

    try:
        ioutil.atomic_write(fs_output_path, write_png, fb_binary=True)

    finally:
        plt.close(f_figure)

# < the end >──────────────────────────────────────────────────────────────────────────
