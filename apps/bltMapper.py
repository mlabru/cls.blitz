# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
bltMapper
mapa simples de incidência de raios: cada raio do CSV vira um ponto, colorido
pelo horário (do mais antigo, roxo, ao mais recente, amarelo).  Não agrupa nem
rastreia nada -- é independente do bltTracker/bltRender.

fluxo (main):
  1) lê o CSV (le_input), ignorando linhas inválidas;
  2) mantém só os raios dentro da região pedida (filtra_regiao);
  3) desenha e grava o PNG (plot_mapa_raios).

    python3 apps/bltMapper.py -r BR -s ./data/output -f mapper.png slm_lightning.csv

códigos de retorno: 0 ok; 2 argumento inválido (argparse); 3 nenhum raio na
região (nada desenhado); 1 exceção não capturada do Python.

2026.out  mlabru  [fix] nenhum raio na região: aviso no log e rc 3, não mais
                  RuntimeError com traceback
2026.out  mlabru  refatoração para legibilidade: plot_mapa_raios dividido
                  (_set_region_extent, _draw_scatter, _style_time_colorbar,
                  _draw_hud, _draw_time_legend), le_input com _parse_row;
                  validação de região repetida removida (o argparse já
                  restringe com choices); main(flst_argv) testável
2026.may  mlabru  [fix#1] colormap "hot" trocado por "plasma" (hot começa em
                  preto, invisível sobre fundo escuro); colorbar no tema
                  escuro com horários reais; recorte derivado de CDCT_REGIONS
                  (e não do bounding box dos pontos); paleta unificada com o
                  bltRender; feições 50 m; fig.add_subplot; subplots_adjust no
                  lugar de bbox_inches="tight"; HUD com estatísticas; asserts
                  removidos; fromtimestamp tz-aware (UTC); diretório de saída
                  criado automaticamente; marcador adaptativo à densidade;
                  imports ccrs/cfeature
2025.nov  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import argparse
import csv
import datetime
import logging
import pathlib
import sys

# matplotLib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# cartoPy
import cartopy.crs as ccrs

# locals
import work.blt_mapbase as mpb
import work.blt_mapdefs as mdf

# < constants >────────────────────────────────────────────────────────────────────────

# códigos de retorno
CI_RC_OK = 0
# nenhum raio dentro da região: nada a desenhar, o PNG anterior fica como está
CI_RC_NO_LIGHTNING = 3

# tamanho do marcador (px²): 3000 / número de raios, limitado a [4, 60] --
# conjuntos densos usam pontos menores para não virar uma mancha
CI_MARKER_SIZE_BUDGET = 3000
CI_MARKER_SIZE_MIN = 4
CI_MARKER_SIZE_MAX = 60

# posições (fração do intervalo de tempo) dos rótulos da barra de cores
CT_COLORBAR_TICK_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

# < logging >──────────────────────────────────────────────────────────────────────────

M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────────────
def arg_parse(flst_argv: list | None = None) -> argparse.Namespace:
    """
    lê os argumentos de linha de comando.

    :param flst_argv: argumentos (None = sys.argv)
    :returns: Namespace com regiao, dpi, fig_w, fig_h, fname, saida e input
    """
    l_parser = argparse.ArgumentParser(description="Incidências de raios (STSC).")

    l_parser.add_argument("-r", "--regiao",
                          help=f"Região {list(mdf.CDCT_REGIONS.keys())}",
                          type=str, default="BR",
                          choices=list(mdf.CDCT_REGIONS.keys()))
    l_parser.add_argument("-d", "--dpi",
                          help="DPI da imagem de saída",
                          type=int, default=mdf.CI_DPI)
    l_parser.add_argument("-l", "--fig-w",
                          help="Largura da figura em polegadas",
                          type=float, default=mdf.CI_FIG_W)
    l_parser.add_argument("-a", "--fig-h",
                          help="Altura da figura em polegadas",
                          type=float, default=mdf.CI_FIG_H)
    l_parser.add_argument("-f", "--fname",
                          help="Nome do arquivo PNG de saída",
                          type=str, default="mapper.png")
    l_parser.add_argument("-s", "--saida",
                          help="Diretório de saída",
                          type=str, default="./")
    l_parser.add_argument("input",
                          help="CSV de entrada (colunas: timestamp, lat, lng)")

    return l_parser.parse_args(flst_argv)

# ─────────────────────────────────────────────────────────────────────────────────────
def filtra_regiao(flst_ts: list, flst_lats: list, flst_lngs: list,
                  fs_regiao: str) -> tuple[list, list, list]:
    """
    mantém só os raios dentro do retângulo da região (limites inclusive).

    :param flst_ts: horários
    :param flst_lats: latitudes
    :param flst_lngs: longitudes
    :param fs_regiao: chave da região (ver mdf.CDCT_REGIONS)
    :returns: (horários, latitudes, longitudes) filtrados, na ordem original
    """
    lf_lat_max, lf_lat_min, lf_lng_max, lf_lng_min = mdf.CDCT_REGIONS[fs_regiao]

    llst_ts, llst_lats, llst_lngs = [], [], []

    for l_time, lf_lat, lf_lng in zip(flst_ts, flst_lats, flst_lngs):
        lb_inside = lf_lat_min <= lf_lat <= lf_lat_max and lf_lng_min <= lf_lng <= lf_lng_max

        if lb_inside:
            llst_ts.append(l_time)
            llst_lats.append(lf_lat)
            llst_lngs.append(lf_lng)

    M_LOG.info("%d/%d eventos na região %s", len(llst_ts), len(flst_ts), fs_regiao)

    return llst_ts, llst_lats, llst_lngs

# ─────────────────────────────────────────────────────────────────────────────────────
def le_input(fpth_filein: pathlib.Path) -> tuple[list, list, list]:
    """
    lê o CSV de raios (colunas obrigatórias: timestamp, lat, lng).
    linhas com valor ausente ou inválido são ignoradas, com warning.

    :param fpth_filein: caminho do CSV
    :returns: (horários tz-aware UTC, latitudes, longitudes)
    """
    llst_ts, llst_lats, llst_lngs = [], [], []
    li_skipped = 0

    with open(fpth_filein, newline="", encoding="utf-8") as l_file:
        # start=2: a linha 1 do arquivo é o cabeçalho
        for li_line_number, ldct_row in enumerate(csv.DictReader(l_file), start=2):
            try:
                ldt_time, lf_lat, lf_lng = _parse_row(ldct_row)

            except (KeyError, ValueError) as l_error:
                M_LOG.warning("linha %d ignorada — %s", li_line_number, l_error)
                li_skipped += 1
                continue

            llst_ts.append(ldt_time)
            llst_lats.append(lf_lat)
            llst_lngs.append(lf_lng)

    M_LOG.info("%d eventos lidos, %d ignorados (%s)", len(llst_ts), li_skipped, fpth_filein.name)

    return llst_ts, llst_lats, llst_lngs

# ─────────────────────────────────────────────────────────────────────────────────────
def _parse_row(fdct_row: dict) -> tuple:
    """
    converte uma linha do CSV.

    [fix] fromtimestamp tz-aware (UTC) -- evita depender do fuso da máquina.

    :param fdct_row: linha do csv.DictReader
    :returns: (horário UTC, lat, lng)
    :raises KeyError: coluna ausente
    :raises ValueError: valor não numérico
    """
    ldt_time = datetime.datetime.fromtimestamp(float(fdct_row["timestamp"]),
                                               tz=datetime.timezone.utc)
    lf_lat = float(fdct_row["lat"])
    lf_lng = float(fdct_row["lng"])

    return ldt_time, lf_lat, lf_lng

# ─────────────────────────────────────────────────────────────────────────────────────
def plot_mapa_raios(flst_ts: list, flst_lats: list, flst_lngs: list,
                    fs_regiao: str,
                    fpth_png: pathlib.Path,
                    ft_figsize: tuple = (mdf.CI_FIG_W, mdf.CI_FIG_H),
                    fi_dpi: int = mdf.CI_DPI) -> None:
    """
    desenha e grava o mapa de raios da região.

    :param flst_ts: horários (tz-aware UTC)
    :param flst_lats: latitudes
    :param flst_lngs: longitudes
    :param fs_regiao: chave da região (recorte, HUD e título)
    :param fpth_png: caminho do PNG
    :param ft_figsize: (largura, altura) em polegadas
    :param fi_dpi: resolução
    :raises RuntimeError: se não houver nenhum raio
    """
    if not flst_lats:
        raise RuntimeError("nenhum raio encontrado na região especificada.")

    ls_region_name = mdf.CDCT_REGION_NAMES.get(fs_regiao, fs_regiao)

    l_figure, l_axes, l_projection = mpb.make_basemap(ft_figsize, fi_dpi)

    _set_region_extent(l_axes, l_projection, fs_regiao)

    l_scatter = _draw_scatter(l_axes, l_projection, flst_ts, flst_lats, flst_lngs)
    _style_time_colorbar(l_figure, l_axes, l_scatter, flst_ts)

    _draw_hud(l_axes, flst_ts, ls_region_name)
    _draw_time_legend(l_axes)

    l_axes.set_title(f"Incidências de Raios — {ls_region_name}",
                     fontsize=17, color=mdf.CS_TEXT_COLOR, pad=14)

    # [fix] bbox_inches="tight" ignorava patches -> margens fixas (blt_mapbase)
    mpb.finalize_and_save(l_figure, fpth_png, fi_dpi)

    M_LOG.info("PNG salvo em: %s", fpth_png)

# ─────────────────────────────────────────────────────────────────────────────────────
def _set_region_extent(f_axes: Axes, f_projection: ccrs.Projection, fs_regiao: str) -> None:
    """
    recorte do mapa: os limites da região mais CF_EXTENT_PAD graus de margem.

    [fix] antes era o bounding box dos pontos, que flutuava com os dados e não
          batia com o filtro de região.

    :param f_axes: eixo cartopy
    :param f_projection: projeção dos dados
    :param fs_regiao: chave da região
    """
    lf_lat_max, lf_lat_min, lf_lng_max, lf_lng_min = mdf.CDCT_REGIONS[fs_regiao]
    lf_pad = mdf.CF_EXTENT_PAD

    f_axes.set_extent([lf_lng_min - lf_pad, lf_lng_max + lf_pad,
                       lf_lat_min - lf_pad, lf_lat_max + lf_pad],
                      crs=f_projection)

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_scatter(f_axes: Axes, f_projection: ccrs.Projection,
                  flst_ts: list, flst_lats: list, flst_lngs: list) -> PathCollection:
    """
    um ponto por raio, colorido pelos segundos desde o primeiro raio.

    [fix] "plasma" (roxo -> laranja -> amarelo) no lugar de "hot", que começa
          em preto e some no fundo escuro.
    [fix] cor por segundos desde o primeiro raio (não pelo índice), para a
          barra de cores mostrar horários reais.
    [fix] tamanho do marcador adaptado ao número de raios.

    :param f_axes: eixo cartopy
    :param f_projection: projeção dos dados
    :param flst_ts: horários
    :param flst_lats: latitudes
    :param flst_lngs: longitudes
    :returns: o scatter (usado pela barra de cores)
    """
    ldt_first = min(flst_ts)
    llst_seconds = [(ldt_time - ldt_first).total_seconds() for ldt_time in flst_ts]

    li_marker_size = max(CI_MARKER_SIZE_MIN,
                         min(CI_MARKER_SIZE_MAX, CI_MARKER_SIZE_BUDGET // len(flst_lats)))

    return f_axes.scatter(flst_lngs, flst_lats,
                          c=llst_seconds, cmap="plasma",
                          s=li_marker_size, alpha=0.85,
                          transform=f_projection, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────────────
def _style_time_colorbar(f_figure: Figure, f_axes: Axes, f_scatter: PathCollection,
                         flst_ts: list) -> None:
    """
    barra de cores vertical no tema escuro, com rótulos HH:MM nos quartis
    do intervalo de tempo.

    :param f_figure: figura
    :param f_axes: eixo do mapa
    :param f_scatter: scatter dos raios
    :param flst_ts: horários (para os rótulos)
    """
    l_colorbar = f_figure.colorbar(f_scatter, ax=f_axes, orientation="vertical",
                                   shrink=0.70, pad=0.02)
    l_colorbar.set_label("Horário (UTC)", color=mdf.CS_TEXT_COLOR, fontsize=11)

    l_colorbar.ax.yaxis.set_tick_params(color=mdf.CS_TEXT_COLOR)
    plt.setp(l_colorbar.ax.yaxis.get_ticklabels(), color=mdf.CS_TEXT_COLOR)

    l_colorbar.outline.set_edgecolor(mdf.CS_TEXT_COLOR)

    ldt_first = min(flst_ts)
    lf_last_second = max((ldt_time - ldt_first).total_seconds() for ldt_time in flst_ts)

    llst_ticks = [lf_last_second * lf_fraction for lf_fraction in CT_COLORBAR_TICK_FRACTIONS]
    llst_labels = [(ldt_first + datetime.timedelta(seconds=lf_tick)).strftime("%H:%M")
                   for lf_tick in llst_ticks]

    l_colorbar.set_ticks(llst_ticks)
    l_colorbar.set_ticklabels(llst_labels)

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_hud(f_axes: Axes, flst_ts: list, fs_region_name: str) -> None:
    """
    caixa de HUD: intervalo de tempo, região e número de raios.

    :param f_axes: eixo cartopy
    :param flst_ts: horários
    :param fs_region_name: nome da região por extenso
    """
    ls_start = min(flst_ts).strftime("%Y-%m-%d %H:%M UTC")
    ls_end = max(flst_ts).strftime("%H:%M UTC")

    ls_hud = (f"LIGHTNING MAP\n"
              f"{ls_start} → {ls_end}\n\n"
              f"Região : {fs_region_name}\n"
              f"Eventos: {len(flst_ts)}")

    mpb.draw_hud(f_axes, ls_hud, fi_fontsize=12)

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_time_legend(f_axes: Axes) -> None:
    """
    legenda da escala de tempo (três cores da paleta "plasma").

    :param f_axes: eixo cartopy
    """
    llst_items = [Line2D([0], [0], marker="o", color="w", label="Mais cedo",
                         markerfacecolor="#0d0887", markersize=10),
                  Line2D([0], [0], marker="o", color="w", label="Intermediário",
                         markerfacecolor="#cc4778", markersize=10),
                  Line2D([0], [0], marker="o", color="w", label="Mais recente",
                         markerfacecolor="#f0f921", markersize=10)]

    mpb.style_legend(f_axes, llst_items, "lower left", fi_fontsize=10)

# ─────────────────────────────────────────────────────────────────────────────────────
def main(flst_argv: list | None = None) -> int:
    """
    lê, filtra pela região e desenha o mapa.

    a região já vem validada pelo argparse (choices=CDCT_REGIONS).

    :param flst_argv: argumentos (None = sys.argv)
    :returns: CI_RC_OK, ou CI_RC_NO_LIGHTNING se não houver raio na região
    """
    l_args = arg_parse(flst_argv)

    llst_ts, llst_lats, llst_lngs = le_input(pathlib.Path(l_args.input))

    llst_ts, llst_lats, llst_lngs = filtra_regiao(llst_ts, llst_lats, llst_lngs,
                                                  l_args.regiao)

    # [fix] sem raios na região, plot_mapa_raios levantava RuntimeError (traceback, rc 1)
    if not llst_ts:
        M_LOG.warning("nenhum raio na região %s; mapa não gerado.", l_args.regiao)
        return CI_RC_NO_LIGHTNING

    lpth_output_dir = pathlib.Path(l_args.saida)
    lpth_output_dir.mkdir(parents=True, exist_ok=True)

    plot_mapa_raios(llst_ts, llst_lats, llst_lngs,
                    fs_regiao=l_args.regiao,
                    fpth_png=lpth_output_dir / l_args.fname,
                    ft_figsize=(l_args.fig_w, l_args.fig_h),
                    fi_dpi=l_args.dpi)

    return CI_RC_OK

# ─────────────────────────────────────────────────────────────────────────────────────
# this is the bootstrap process

if "__main__" == __name__:
    logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S",
                        format="[%(asctime)s] [%(levelname)s] %(message)s",
                        level=logging.INFO)

    sys.exit(main())

# < the end >──────────────────────────────────────────────────────────────────────────
