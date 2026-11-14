# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
bltRender
desenha um frame do nowcast (o JSON de células do bltTracker) num PNG
operacional: mapa do Brasil, células coloridas por severidade, trilhas, setas
de movimento, trajetória prevista com cone e elipses de incerteza do Kalman,
HUD e legenda.

fluxo (render):
  1) lê o JSON e descarta as células sem os campos mínimos (_load_cells);
  2) cria o mapa base com o recorte fixo do Brasil (blt_mapbase);
  3) desenha cada célula (_draw_cell): elipse + anel CG, trilha, seta,
     previsão (linha, elipses de incerteza, cone) -- nesta ordem;
  4) HUD, legenda e título;
  5) grava o PNG (blt_mapbase.finalize_and_save).

    python3 apps/bltRender.py frame_0000.json frame_0000.png

códigos de retorno: 0 ok, 1 JSON de entrada não encontrado.

atenção: a ORDEM em que os elementos são desenhados importa para o resultado
(com zorders iguais, quem é desenhado depois fica por cima).  Ao reorganizar,
compare os PNGs antes/depois pixel a pixel.

2026.nov  mlabru  severidade do GLM pela taxa de raios (cell_color,
                  severity_class_glm): faixas em flashes/min e salto de raios
                  sobe uma classe; legenda com as faixas
2026.nov  mlabru  anel amarelo nas células com salto de raios (lightning_jump),
                  item na legenda e contagem no HUD
2026.nov  mlabru  --source (stsc/glm): fonte e janela no HUD
2026.nov  mlabru  HUD: "Events" e janela em imagens (com o STSC cada evento é
                  uma célula de grade CB numa imagem, não uma descarga)
2026.nov  mlabru  seta só a partir de CF_MIN_ARROW_SPEED_KMH (a velocidade
                  agora vem do Kalman e quase nunca é exatamente 0)
2026.out  mlabru  refatoração para legibilidade: render dividido em
                  _load_cells, _draw_cell (e um passo por elemento), _draw_hud,
                  _draw_legend; classes de severidade numa tabela única
                  (CLST_SEVERITY_CLASSES) usada pela cor e pela legenda;
                  km/grau vindo de blt_geo; main(flst_argv) testável;
                  severity_index (nunca preenchido pelo tracker) saiu do
                  JSON: a severidade é sempre a de _compute_severity_local
2026.may  mlabru  [fix#2] Circle substituído por Ellipse com escala lat/lng
                  independente (Circle achatava clusters a latitudes não
                  equatoriais); _compute_severity_local normaliza size por
                  SIZE_NORM (size bruto dominava a soma); forecast_cone usa
                  lat_std/lng_std Kalman no último ponto quando disponíveis
                  (fallback em 25% do vetor) e loga warning para len < 2;
                  density sanitizado com max(0.0, ...); validação de schema
                  mínimo via _validate_cell; timestamp usa max(t_end); cartopy:
                  oceano, terra, costa, fronteiras, estados (50 m) e grade;
                  paleta oceanográfica; seta de movimento com ax.quiver
                  (ax.arrow não suporta transform cartopy)
2026.may  mlabru  [fix#1] forecast_path acessado por chave de dict; history
                  via centroid; severity usa severity_index do JSON;
                  forecast_cone em km -> graus; raio derivado da extensão
                  física do cluster; fromtimestamp tz-aware; argparse;
                  subplots_adjust no lugar de tight_layout; elipses de
                  incerteza Kalman por ponto de previsão
2026.may  mlabru  corrections and refactoring
2022.apr  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import argparse
import bisect
import datetime
import json
import logging
import math
import os
import sys

# matplotLib
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon

# cartoPy
import cartopy.crs as ccrs

# locals
import work.blt_geo as geo
import work.blt_mapbase as mpb
import work.blt_mapdefs as mdf

# < constants >────────────────────────────────────────────────────────────────────────

# códigos de retorno
CI_RC_OK = 0
CI_RC_INPUT_NOT_FOUND = 1

# campos mínimos de uma célula no JSON (ver _validate_cell)
_REQUIRED_CELL_KEYS = {"centroid", "size", "t_end"}

# classes de severidade: (limite superior exclusivo, cor, rótulo da legenda).
# severity_color devolve a cor da primeira classe cujo limite é maior que o
# índice; a legenda lista as mesmas classes, na mesma ordem.
CLST_SEVERITY_CLASSES = [
    (0.2,      "#2b83ba", "Weak"),       # azul
    (0.4,      "#abdda4", "Moderate"),   # verde
    (0.6,      "#ffffbf", "Strong"),     # amarelo
    (0.8,      "#fdae61", "Severe"),     # laranja
    (math.inf, "#d7191c", "Extreme"),    # vermelho
]

# [new] severidade com o GLM: limites superiores (flashes/min, exclusivos) da
# taxa recente (flash_rate_recent, ~6 min) de Weak, Moderate, Strong e Severe;
# acima do último, Extreme.  Provisórios: na tarde de 14/11/2026 (1.356 células)
# a mediana era 1,7/min, 10% passavam de 10/min e 1% de 50/min, então ~55% Weak,
# 25% Moderate, 14% Strong, 4% Severe e 1,5% Extreme
CLST_GLM_RATE_LIMITS = [2.0, 5.0, 15.0, 40.0]

# o salto de raios (intensificação) sobe a célula esta quantidade de classes
CI_JUMP_CLASS_BUMP = 1

# fonte (--source) -> (rótulo, janela) mostrados no HUD.  Só texto: a janela
# deve acompanhar bltTracker.CI_WINDOW_IMAGES (stsc) e CI_WINDOW_MINUTES (glm);
# não é importada aqui para o render não depender do tracker
CDCT_HUD_SOURCE = {"stsc": ("STSC", "4 images"),
                   "glm":  ("GLM",  "20 min")}
CS_DEFAULT_SOURCE = "stsc"

# raio da célula quando a densidade é zero (km) e tamanho mínimo da elipse (graus)
CF_DEFAULT_CELL_RADIUS_KM = 5.0
CF_MIN_ELLIPSE_RADIUS_DEG = 0.05

# anel branco: células com cg_ratio acima disto são "CG dominantes"
CF_CG_DOMINANT_RATIO = 0.5
# o anel é 15% maior que a elipse da célula
CF_CG_RING_SCALE = 1.15

# [new] salto de raios (glm_ljump): anel amarelo, mais largo que o CG, com
# raio mínimo para aparecer mesmo em célula pequena
CS_JUMP_COLOR = "#ffd400"
CF_JUMP_RING_SCALE = 1.35
CF_JUMP_MIN_RADIUS_KM = 15.0
CF_JUMP_LINEWIDTH = 3.0

# abaixo desta velocidade (km/h) a célula não ganha seta de movimento
CF_MIN_ARROW_SPEED_KMH = 5.0

# largura do cone sem incerteza do Kalman: 25% do comprimento da trajetória
CF_CONE_FALLBACK_WIDTH_FRACTION = 0.25

# elipse de incerteza 2-sigma: diâmetro = 2 x (2 x std) = 4 x std
CF_UNCERTAINTY_DIAMETER_PER_STD = 4

# título do mapa
CS_MAP_TITLE = "Simple Lightning Model Nowcasting"

# < logging >──────────────────────────────────────────────────────────────────────────

M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────────────
def _compute_severity_local(fdct_cell: dict) -> float:
    """
    índice de severidade [0, 1] da célula, calculado a partir do JSON do frame
    (o tracker não grava um índice próprio).

    média simples de quatro termos: densidade, taxa de raios, tamanho
    normalizado e fração CG, limitada a 1.0.

    [fix#2] size é normalizado por CF_SIZE_NORM antes de entrar na soma.  Sem
            isso, size bruto (ex.: 500 raios) saturava o índice e tornava
            density, flash_rate e cg_ratio irrelevantes.

    :param fdct_cell: célula do JSON
    :returns: índice de severidade entre 0 e 1
    """
    lf_size_normalized = min(fdct_cell.get("size", 0) / mdf.CF_SIZE_NORM, 1.0)

    return min(1.0,
               fdct_cell.get("density",    0.0) * 0.25 +
               fdct_cell.get("flash_rate", 0.0) * 0.25 +
               lf_size_normalized                * 0.25 +
               fdct_cell.get("cg_ratio",   0.0) * 0.25)

# ─────────────────────────────────────────────────────────────────────────────────────
def forecast_cone(ff_cx: float, ff_cy: float, flst_forecast: list,
                  ff_ref_lat: float) -> list | None:
    """
    triângulo do cone de previsão: a base, centrada na célula, é perpendicular
    à trajetória; o vértice é a posição prevista mais distante.

    [fix#1] as contas são em km, convertendo lat e lng separadamente
            (1° de lng ≠ 1° de lat fora do equador).
    [fix#2] a largura vem da incerteza do Kalman no último ponto (2-sigma,
            média dos dois eixos) quando disponível; senão, 25% do comprimento.

    :param ff_cx: longitude da célula (graus)
    :param ff_cy: latitude da célula (graus)
    :param flst_forecast: lista de dicts {"lat", "lng", "lat_std"?, "lng_std"?}
    :param ff_ref_lat: latitude de referência para km por grau de longitude
    :returns: três vértices (lng, lat), ou None se houver menos de 2 pontos
              ou a trajetória tiver comprimento zero
    """
    if len(flst_forecast) < 2:
        M_LOG.warning("forecast path com %d ponto(s) — "
                      "cone omitido (mínimo 2)", len(flst_forecast))
        return None

    ldct_last_point = flst_forecast[-1]
    lf_end_x = ldct_last_point["lng"]
    lf_end_y = ldct_last_point["lat"]

    lf_km_per_deg_lng = geo.km_per_deg_lng(ff_ref_lat)

    # vetor célula -> último ponto, em km
    lf_dx_km = (lf_end_x - ff_cx) * lf_km_per_deg_lng
    lf_dy_km = (lf_end_y - ff_cy) * geo.CF_KM_PER_DEG_LAT
    lf_length_km = math.sqrt(lf_dx_km ** 2 + lf_dy_km ** 2)

    if 0.0 == lf_length_km:
        return None

    # vetor unitário perpendicular à trajetória (km)
    lf_perpendicular_x = -lf_dy_km / lf_length_km
    lf_perpendicular_y = lf_dx_km / lf_length_km

    lf_width_km = _cone_width_km(ldct_last_point, lf_length_km, lf_km_per_deg_lng)

    # meia-base do cone convertida de km para graus, eixo a eixo
    lf_offset_lng = (lf_perpendicular_x * lf_width_km) / lf_km_per_deg_lng
    lf_offset_lat = (lf_perpendicular_y * lf_width_km) / geo.CF_KM_PER_DEG_LAT

    return [(ff_cx + lf_offset_lng, ff_cy + lf_offset_lat),
            (ff_cx - lf_offset_lng, ff_cy - lf_offset_lat),
            (lf_end_x, lf_end_y)]

# ─────────────────────────────────────────────────────────────────────────────────────
def _cone_width_km(fdct_last_point: dict, ff_length_km: float,
                   ff_km_per_deg_lng: float) -> float:
    """
    meia-largura da base do cone, em km.

    :param fdct_last_point: último ponto da previsão (pode ter lat_std/lng_std)
    :param ff_length_km: comprimento da trajetória (km)
    :param ff_km_per_deg_lng: km por grau de longitude na latitude da célula
    :returns: 2-sigma médio do Kalman, ou 25% do comprimento sem incerteza
    """
    lf_lat_std = fdct_last_point.get("lat_std", 0.0)
    lf_lng_std = fdct_last_point.get("lng_std", 0.0)

    lb_has_uncertainty = lf_lat_std > 0.0 and lf_lng_std > 0.0

    if not lb_has_uncertainty:
        return ff_length_km * CF_CONE_FALLBACK_WIDTH_FRACTION

    lf_lat_std_km = lf_lat_std * geo.CF_KM_PER_DEG_LAT
    lf_lng_std_km = lf_lng_std * ff_km_per_deg_lng

    # média dos dois eixos, vezes 2 (2-sigma)
    return (lf_lat_std_km + lf_lng_std_km) / 2.0 * 2.0

# ─────────────────────────────────────────────────────────────────────────────────────
def render(fs_input_json: str, fs_output_png: str, fs_source: str | None = None) -> int:
    """
    desenha um frame (JSON de células) num PNG.

    :param fs_input_json: JSON gerado pelo bltTracker
    :param fs_output_png: PNG de saída (o diretório já precisa existir)
    :param fs_source: fonte dos eventos, para o HUD (None = CS_DEFAULT_SOURCE)
    :returns: CI_RC_OK
    """
    llst_cells = _load_cells(fs_input_json)

    l_figure, l_axes, l_projection = mpb.make_basemap((mdf.CI_FIG_W, mdf.CI_FIG_H), mdf.CI_DPI)

    # recorte fixo: Brasil continental + ilhas oceânicas (ver blt_mapdefs)
    l_axes.set_extent(mdf.CLST_BRAZIL_EXTENT, crs=l_projection)

    ls_source = fs_source or CS_DEFAULT_SOURCE

    for ldct_cell in llst_cells:
        _draw_cell(l_axes, l_projection, ldct_cell, ls_source)

    _draw_hud(l_axes, llst_cells, ls_source)
    _draw_legend(l_axes, ls_source)

    l_axes.set_title(CS_MAP_TITLE, fontsize=18, color=mdf.CS_TEXT_COLOR, pad=15)

    mpb.finalize_and_save(l_figure, fs_output_png, mdf.CI_DPI)

    M_LOG.info("PNG salvo em: %s", fs_output_png)

    return CI_RC_OK

# ─────────────────────────────────────────────────────────────────────────────────────
def _load_cells(fs_input_json: str) -> list:
    """
    lê o JSON do frame e devolve só as células válidas (_validate_cell).

    :param fs_input_json: caminho do JSON (lista de dicts)
    :returns: lista de dicts de célula
    """
    with open(fs_input_json) as l_file:
        llst_raw_cells = json.load(l_file)

    llst_cells = []

    for li_index, ldct_cell in enumerate(llst_raw_cells):
        if _validate_cell(ldct_cell, li_index):
            llst_cells.append(ldct_cell)

    M_LOG.info("%d/%d células válidas carregadas", len(llst_cells), len(llst_raw_cells))

    return llst_cells

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_cell(f_axes: Axes, f_projection: ccrs.Projection, fdct_cell: dict,
               fs_source: str = CS_DEFAULT_SOURCE) -> None:
    """
    desenha tudo o que pertence a uma célula, nesta ordem (não mudar):
    elipse (e anel CG), anel de salto de raios, trilha, seta de movimento,
    previsão.

    :param f_axes: eixo cartopy
    :param f_projection: projeção dos dados
    :param fdct_cell: célula do JSON (já validada)
    :param fs_source: fonte dos eventos (decide a severidade, cell_color)
    """
    _draw_cell_body(f_axes, f_projection, fdct_cell, fs_source)
    _draw_jump_ring(f_axes, f_projection, fdct_cell)
    _draw_track(f_axes, f_projection, fdct_cell)
    _draw_motion_arrow(f_axes, f_projection, fdct_cell)
    _draw_forecast(f_axes, f_projection, fdct_cell)

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_cell_body(f_axes: Axes, f_projection: ccrs.Projection, fdct_cell: dict,
                    fs_source: str = CS_DEFAULT_SOURCE) -> None:
    """
    elipse da célula, colorida pela severidade, e anel branco se for CG dominante.

    o raio físico vem da área do cluster: área = size / density, então
    raio = sqrt(size / (π · density)).

    [fix#2] Ellipse (e não Circle) com raios em graus calculados separadamente
            para lat e lng: fora do equador um grau de lng tem menos km, e um
            círculo em graus aparecia achatado no mapa.

    :param f_axes: eixo cartopy
    :param f_projection: projeção dos dados
    :param fdct_cell: célula do JSON
    :param fs_source: fonte dos eventos (cell_color)
    """
    lf_lat, lf_lng = fdct_cell["centroid"]
    li_size = fdct_cell["size"]
    # [fix#2] densidade negativa é tratada como zero
    lf_density = max(0.0, fdct_cell.get("density", 0.0))

    if lf_density > 0.0:
        lf_radius_km = math.sqrt(li_size / (math.pi * lf_density))
    else:
        lf_radius_km = CF_DEFAULT_CELL_RADIUS_KM

    lf_radius_lat_deg = max(CF_MIN_ELLIPSE_RADIUS_DEG, lf_radius_km / geo.CF_KM_PER_DEG_LAT)
    lf_radius_lng_deg = max(CF_MIN_ELLIPSE_RADIUS_DEG, lf_radius_km / geo.km_per_deg_lng(lf_lat))

    # mais densa = mais opaca, entre 0.3 e 0.9
    lf_alpha = min(0.9, max(0.3, 0.3 + min(lf_density, 0.6)))

    f_axes.add_patch(Ellipse((lf_lng, lf_lat),
                             width=lf_radius_lng_deg * 2,
                             height=lf_radius_lat_deg * 2,
                             transform=f_projection,
                             color=cell_color(fdct_cell, fs_source),
                             alpha=lf_alpha,
                             linewidth=1.5,
                             fill=True,
                             zorder=5))

    lb_cg_dominant = fdct_cell.get("cg_ratio", 0.0) > CF_CG_DOMINANT_RATIO

    if not lb_cg_dominant:
        return

    f_axes.add_patch(Ellipse((lf_lng, lf_lat),
                             width=lf_radius_lng_deg * 2 * CF_CG_RING_SCALE,
                             height=lf_radius_lat_deg * 2 * CF_CG_RING_SCALE,
                             transform=f_projection,
                             edgecolor="white",
                             facecolor="none",
                             alpha=0.6,
                             linewidth=1.5,
                             zorder=5))

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_jump_ring(f_axes: Axes, f_projection: ccrs.Projection, fdct_cell: dict) -> None:
    """
    anel amarelo em volta da célula com salto de raios (lightning_jump, só
    com fonte de descargas): o aviso de intensificação.

    :param f_axes: eixo cartopy
    :param f_projection: projeção dos dados
    :param fdct_cell: célula do JSON
    """
    if not fdct_cell.get("lightning_jump", False):
        return

    lf_lat, lf_lng = fdct_cell["centroid"]
    lf_density = max(0.0, fdct_cell.get("density", 0.0))

    if lf_density > 0.0:
        lf_radius_km = math.sqrt(fdct_cell["size"] / (math.pi * lf_density))
    else:
        lf_radius_km = CF_DEFAULT_CELL_RADIUS_KM

    lf_radius_km = max(lf_radius_km * CF_JUMP_RING_SCALE, CF_JUMP_MIN_RADIUS_KM)

    f_axes.add_patch(Ellipse((lf_lng, lf_lat),
                             width=2.0 * lf_radius_km / geo.km_per_deg_lng(lf_lat),
                             height=2.0 * lf_radius_km / geo.CF_KM_PER_DEG_LAT,
                             transform=f_projection,
                             edgecolor=CS_JUMP_COLOR,
                             facecolor="none",
                             linewidth=CF_JUMP_LINEWIDTH,
                             zorder=6))

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_track(f_axes: Axes, f_projection: ccrs.Projection, fdct_cell: dict) -> None:
    """
    trilha: linha pelos centroides do histórico do track (2 pontos ou mais).
    um histórico em formato inesperado é ignorado (log em nível debug).

    :param f_axes: eixo cartopy
    :param f_projection: projeção dos dados
    :param fdct_cell: célula do JSON
    """
    llst_history = fdct_cell.get("history", [])

    if len(llst_history) < 2:
        return

    try:
        llst_lngs = [ldct_point["centroid"][1] for ldct_point in llst_history]
        llst_lats = [ldct_point["centroid"][0] for ldct_point in llst_history]

    except (KeyError, TypeError, IndexError):
        M_LOG.debug("célula %r: history em formato inesperado — ignorada",
                    fdct_cell["centroid"][0])
        return

    f_axes.plot(llst_lngs, llst_lats,
                color=mdf.CS_TRACK_COLOR, linewidth=2, alpha=0.7,
                transform=f_projection, zorder=4)

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_motion_arrow(f_axes: Axes, f_projection: ccrs.Projection, fdct_cell: dict) -> None:
    """
    seta na direção do movimento, só se a célula estiver andando.

    [fix#2] ax.arrow não aceita o transform das projeções cartopy; ax.quiver sim.

    :param f_axes: eixo cartopy
    :param f_projection: projeção dos dados
    :param fdct_cell: célula do JSON
    """
    # [fix] a velocidade vem do Kalman (blt_motion.kalman_velocity), quase
    # nunca exatamente 0: abaixo do limiar a célula conta como parada
    lb_is_moving = fdct_cell.get("speed_kmh", 0.0) >= CF_MIN_ARROW_SPEED_KMH

    if not lb_is_moving:
        return

    lf_lat, lf_lng = fdct_cell["centroid"]
    lf_direction_rad = math.radians(fdct_cell.get("direction_deg", 0.0))

    # azimute: 0° = norte (+y), 90° = leste (+x)
    lf_east = math.sin(lf_direction_rad)
    lf_north = math.cos(lf_direction_rad)

    f_axes.quiver(lf_lng, lf_lat, lf_east, lf_north,
                  color="white",
                  alpha=0.45,
                  scale=22,
                  width=0.002,
                  headwidth=3,
                  headlength=3.5,
                  transform=f_projection,
                  zorder=6)

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_forecast(f_axes: Axes, f_projection: ccrs.Projection, fdct_cell: dict) -> None:
    """
    previsão da célula (2 pontos ou mais): linha tracejada, elipses de
    incerteza e cone -- nesta ordem.

    :param f_axes: eixo cartopy
    :param f_projection: projeção dos dados
    :param fdct_cell: célula do JSON
    """
    llst_forecast = fdct_cell.get("forecast_path", [])

    if len(llst_forecast) < 2:
        return

    llst_lngs = [ldct_point["lng"] for ldct_point in llst_forecast]
    llst_lats = [ldct_point["lat"] for ldct_point in llst_forecast]

    f_axes.plot(llst_lngs, llst_lats, "--",
                color=mdf.CS_FORECAST_COLOR, linewidth=1.5, alpha=0.8,
                transform=f_projection, zorder=5)

    _draw_uncertainty_ellipses(f_axes, f_projection, llst_forecast)
    _draw_forecast_cone(f_axes, f_projection, fdct_cell, llst_forecast)

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_uncertainty_ellipses(f_axes: Axes, f_projection: ccrs.Projection,
                               flst_forecast: list) -> None:
    """
    uma elipse 2-sigma, bem transparente, em cada ponto previsto que tenha
    incerteza do Kalman (lat_std e lng_std > 0).

    :param f_axes: eixo cartopy
    :param f_projection: projeção dos dados
    :param flst_forecast: pontos previstos
    """
    for ldct_point in flst_forecast:
        lf_lat_std = ldct_point.get("lat_std", 0.0)
        lf_lng_std = ldct_point.get("lng_std", 0.0)

        lb_has_uncertainty = lf_lat_std > 0.0 and lf_lng_std > 0.0

        if not lb_has_uncertainty:
            continue

        f_axes.add_patch(Ellipse((ldct_point["lng"], ldct_point["lat"]),
                                 width=lf_lng_std * CF_UNCERTAINTY_DIAMETER_PER_STD,
                                 height=lf_lat_std * CF_UNCERTAINTY_DIAMETER_PER_STD,
                                 transform=f_projection,
                                 color=mdf.CS_FORECAST_COLOR,
                                 alpha=0.08,
                                 linewidth=0,
                                 zorder=5))

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_forecast_cone(f_axes: Axes, f_projection: ccrs.Projection,
                        fdct_cell: dict, flst_forecast: list) -> None:
    """
    cone de previsão (forecast_cone), se ele puder ser construído.

    :param f_axes: eixo cartopy
    :param f_projection: projeção dos dados
    :param fdct_cell: célula do JSON (usa o centroide)
    :param flst_forecast: pontos previstos
    """
    lf_lat, lf_lng = fdct_cell["centroid"]

    llst_cone_vertices = forecast_cone(lf_lng, lf_lat, flst_forecast, ff_ref_lat=lf_lat)

    if not llst_cone_vertices:
        return

    f_axes.add_patch(Polygon(llst_cone_vertices,
                             transform=f_projection,
                             color=mdf.CS_FORECAST_COLOR,
                             alpha=mdf.CF_FORECAST_ALPHA,
                             zorder=5))

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_hud(f_axes: Axes, flst_cells: list, fs_source: str = CS_DEFAULT_SOURCE) -> None:
    """
    caixa de HUD: horário do frame, número de células e de eventos, janela.

    [doc] "Events" e não "Flashes": com o STSC, cada evento é uma célula de
    grade (~10 km) com atividade CB numa imagem, não uma descarga.

    :param f_axes: eixo cartopy
    :param flst_cells: células desenhadas
    :param fs_source: fonte dos eventos (chave de CDCT_HUD_SOURCE)
    """
    ls_source_label, ls_window_text = CDCT_HUD_SOURCE[fs_source]
    li_jumps = sum(1 for ldct_cell in flst_cells if ldct_cell.get("lightning_jump", False))
    li_total_events = sum(ldct_cell["size"] for ldct_cell in flst_cells)

    ls_hud = (f"LIGHTNING NOWCAST\n"
              f"{_frame_time_text(flst_cells)}\n\n"
              f"Cells  : {len(flst_cells)}\n"
              f"Events : {li_total_events}\n"
              f"Source : {ls_source_label}\n"
              f"Window : {ls_window_text}" +
              (f"\nJumps  : {li_jumps}" if "glm" == fs_source else ""))

    mpb.draw_hud(f_axes, ls_hud, fi_fontsize=13)

# ─────────────────────────────────────────────────────────────────────────────────────
def _frame_time_text(flst_cells: list) -> str:
    """
    horário do frame: o maior t_end entre as células, em UTC.

    [fix#2] max(t_end), e não cells[0]["t_end"], que supunha ordem cronológica.

    :param flst_cells: células do frame
    :returns: "AAAA-MM-DD HH:MM UTC", ou "N/A" sem células
    """
    llst_end_times = [ldct_cell["t_end"] for ldct_cell in flst_cells if "t_end" in ldct_cell]

    if not llst_end_times:
        return "N/A"

    ldt_frame = datetime.datetime.fromtimestamp(max(llst_end_times), tz=datetime.timezone.utc)

    return ldt_frame.strftime("%Y-%m-%d %H:%M UTC")

# ─────────────────────────────────────────────────────────────────────────────────────
def _draw_legend(f_axes: Axes, fs_source: str = CS_DEFAULT_SOURCE) -> None:
    """
    legenda: uma bolinha por classe de severidade (com as faixas de taxa, no
    GLM), mais o anel "Lightning jump" (GLM) ou "CG dominant" (outras fontes).

    :param f_axes: eixo cartopy
    :param fs_source: fonte dos eventos
    """
    llst_items = []

    for (_, ls_color, ls_label), ls_range in zip(CLST_SEVERITY_CLASSES,
                                                 _legend_ranges(fs_source)):
        llst_items.append(Line2D([0], [0], marker="o", color="w", label=ls_label + ls_range,
                                 markerfacecolor=ls_color, markersize=12))

    # o GLM não separa nuvem-solo de intranuvem (sem anel CG); o salto de
    # raios só existe com o GLM
    if "glm" != fs_source:
        llst_items.append(Line2D([0], [0], marker="o", color="w", label="CG dominant",
                                 markerfacecolor="none", markersize=12,
                                 markeredgecolor="white", markeredgewidth=1.5))
    else:
        llst_items.append(Line2D([0], [0], marker="o", color="w", label="Lightning jump",
                                 markerfacecolor="none", markersize=12,
                                 markeredgecolor=CS_JUMP_COLOR,
                                 markeredgewidth=CF_JUMP_LINEWIDTH))

    mpb.style_legend(f_axes, llst_items, "lower right", fi_fontsize=11)

# ─────────────────────────────────────────────────────────────────────────────────────
def _legend_ranges(fs_source: str) -> list:
    """
    o texto da faixa de cada classe na legenda: no GLM, a taxa de raios
    (ex.: " (5-15 fl/min)"); nas outras fontes, nada.

    :param fs_source: fonte dos eventos
    :returns: um texto por classe de CLST_SEVERITY_CLASSES
    """
    if "glm" != fs_source:
        return [""] * len(CLST_SEVERITY_CLASSES)

    llst_limits = ["%g" % lf_limit for lf_limit in CLST_GLM_RATE_LIMITS]
    llst_bounds = ["0"] + llst_limits

    llst_ranges = [" (%s-%s fl/min)" % (ls_low, ls_high)
                   for ls_low, ls_high in zip(llst_bounds, llst_limits)]

    return llst_ranges + [" (>= %s fl/min)" % llst_limits[-1]]

# ─────────────────────────────────────────────────────────────────────────────────────
def cell_color(fdct_cell: dict, fs_source: str = CS_DEFAULT_SOURCE) -> str:
    """
    a cor da célula: no GLM, a classe da taxa de raios (severity_class_glm);
    nas outras fontes, o índice local (_compute_severity_local).

    :param fdct_cell: célula do JSON
    :param fs_source: fonte dos eventos
    :returns: cor hexadecimal
    """
    if "glm" == fs_source:
        return CLST_SEVERITY_CLASSES[severity_class_glm(fdct_cell)][1]

    return severity_color(_compute_severity_local(fdct_cell))

# ─────────────────────────────────────────────────────────────────────────────────────
def severity_class_glm(fdct_cell: dict) -> int:
    """
    classe de severidade de uma célula do GLM: a faixa da taxa recente de
    raios (CLST_GLM_RATE_LIMITS) e, com salto de raios, CI_JUMP_CLASS_BUMP
    classes acima (até Extreme).

    [doc] o índice de _compute_severity_local soma termos de escalas
    diferentes; com flashes reais a taxa (sem normalização) saturava e quase
    toda célula saía Extreme.  Com uma fonte de descargas, a taxa de raios
    totais é o indicador de intensidade da literatura.

    :param fdct_cell: célula do JSON (flash_rate_recent, lightning_jump)
    :returns: índice em CLST_SEVERITY_CLASSES
    """
    lf_rate = fdct_cell.get("flash_rate_recent", 0.0)
    li_class = bisect.bisect_right(CLST_GLM_RATE_LIMITS, lf_rate)

    if fdct_cell.get("lightning_jump", False):
        li_class += CI_JUMP_CLASS_BUMP

    return min(li_class, len(CLST_SEVERITY_CLASSES) - 1)

# ─────────────────────────────────────────────────────────────────────────────────────
def severity_color(ff_severity: float) -> str:
    """
    cor da classe de severidade (escala fixa [0, 1], ver CLST_SEVERITY_CLASSES).

    :param ff_severity: índice de severidade
    :returns: cor hexadecimal ("#rrggbb")
    """
    for lf_upper_limit, ls_color, _ in CLST_SEVERITY_CLASSES:
        if ff_severity < lf_upper_limit:
            return ls_color

    # só chega aqui com NaN (nenhuma comparação é verdadeira): classe mais alta
    return CLST_SEVERITY_CLASSES[-1][1]

# ─────────────────────────────────────────────────────────────────────────────────────
def _validate_cell(fdct_cell: dict, fi_index: int) -> bool:
    """
    confere os campos mínimos de uma célula do JSON.

    :param fdct_cell: célula do JSON
    :param fi_index: posição da célula na lista (só para o log)
    :returns: True se a célula tem centroid, size e t_end, com centroid = [lat, lng]
    """
    lset_missing = _REQUIRED_CELL_KEYS - set(fdct_cell.keys())

    if lset_missing:
        M_LOG.warning("célula %d ignorada — campos ausentes: %s", fi_index, lset_missing)
        return False

    l_centroid = fdct_cell.get("centroid")

    if not (isinstance(l_centroid, (list, tuple)) and 2 == len(l_centroid)):
        M_LOG.warning("célula %d ignorada — centroid inválido: %r", fi_index, l_centroid)
        return False

    return True

# ─────────────────────────────────────────────────────────────────────────────────────
def main(flst_argv: list | None = None) -> int:
    """
    lê os argumentos, confere a entrada, cria o diretório de saída e desenha.

    :param flst_argv: argumentos (None = sys.argv)
    :returns: CI_RC_OK, ou CI_RC_INPUT_NOT_FOUND se o JSON não existir
    """
    l_parser = argparse.ArgumentParser(
        description="SLM Render — nowcast meteorológico operacional"
    )
    l_parser.add_argument("input_json", help="JSON de células gerado pelo pipeline")
    l_parser.add_argument("output_png", help="PNG de saída")
    l_parser.add_argument("--source", help="fonte dos eventos (HUD): stsc ou glm",
                          choices=sorted(CDCT_HUD_SOURCE), default=CS_DEFAULT_SOURCE)

    l_args = l_parser.parse_args(flst_argv)

    if not os.path.isfile(l_args.input_json):
        M_LOG.error("arquivo não encontrado: %s", l_args.input_json)
        return CI_RC_INPUT_NOT_FOUND

    # [fix#2] cria o diretório de saída se não existir
    os.makedirs(os.path.dirname(os.path.abspath(l_args.output_png)), exist_ok=True)

    return render(l_args.input_json, l_args.output_png, l_args.source)

# ─────────────────────────────────────────────────────────────────────────────────────
# this is the bootstrap process

if "__main__" == __name__:
    logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S",
                        format="[%(asctime)s] [%(levelname)s] %(message)s",
                        level=logging.INFO)

    sys.exit(main())

# < the end >──────────────────────────────────────────────────────────────────────────
