# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
blt_mapdefs
constantes visuais dos mapas (tamanho da figura, regiões, paleta), usadas por
bltRender.py (nowcast) e bltMapper.py (mapa de raios), via blt_mapbase.

conversão km <-> graus NÃO fica aqui: ver blt_geo.

2026.out  mlabru  refatoração para legibilidade: CF_KM_PER_DEG_LAT movido
                  para blt_geo; os dois recortes do Brasil documentados; a
                  string solta dos extremos geográficos virou comentário
2025.nov  mlabru  initial version (Linux/Python)
"""
# < figura >───────────────────────────────────────────────────────────────────────────

# resolução e tamanho da imagem em pixels (a figura em polegadas = pixels / DPI)
CI_DPI = 300

CI_WIDTH = 4800
CI_HEIGHT = 4200

CI_FIG_W = CI_WIDTH / CI_DPI
CI_FIG_H = CI_HEIGHT / CI_DPI

# margem (graus) somada em volta dos limites da região no bltMapper
CF_EXTENT_PAD = 0.5

# normalização de size para a severidade local do bltRender.
# valor provisório -- ajustar conforme a distribuição empírica do domínio.
CF_SIZE_NORM = 200.0   # raios por janela de 5 min

# layout -- margens fixas (subplots_adjust é mais seguro que tight_layout
# quando patches ultrapassam os limites dos eixos; tight_layout os ignora)
CDCT_LAYOUT = dict(left=0.06, right=0.97, top=0.93, bottom=0.07)

# < regiões >──────────────────────────────────────────────────────────────────────────

# extremos geográficos do Brasil (território continental):
#   Norte:  5°16'N (lat) - nascente do Rio Ailã, Monte Caburaí, Roraima (RR)
#   Sul:   33°45'S (lat) - Arroio Chuí, Rio Grande do Sul (RS)
#   Leste: 34°47'W (lng) - Ponta do Seixas, Paraíba (PB)
#   Oeste: 73°59'W (lng) - Serra da Contamana, Acre (AC)
#
# há DOIS recortes do Brasil, de propósito:
#   CDCT_REGIONS["BR"]  só o continente, em graus inteiros; filtra os raios e
#                       define o mapa do bltMapper (mais CF_EXTENT_PAD);
#   CLST_BRAZIL_EXTENT  continente + ilhas oceânicas (Trindade, Fernando de
#                       Noronha: por isso vai até -28.6 de longitude); mapa fixo
#                       do bltRender, que nunca filtra raios por região.

# limites das regiões do Brasil: (lat_max, lat_min, lng_max, lng_min)
CDCT_REGIONS = {
    "BR": (  5, -34, -35, -74),
    "N":  (  5, -16, -45, -74),   # Norte
    "NE": ( -1, -18, -35, -49),   # Nordeste
    "CO": ( -7, -25, -45, -62),   # Centro-Oeste
    "SE": (-14, -27, -39, -54),   # Sudeste
    "S":  (-23, -34, -48, -58),   # Sul
}

# nomes completos das regiões, para o HUD e o título
CDCT_REGION_NAMES = {
    "BR": "Brasil",
    "N":  "Norte",
    "NE": "Nordeste",
    "CO": "Centro-Oeste",
    "SE": "Sudeste",
    "S":  "Sul",
}

# recorte fixo do bltRender: [lng_min, lng_max, lat_min, lat_max] (ordem do cartopy)
CLST_BRAZIL_EXTENT = [-73.9, -28.6, -33.8, 5.3]

# < paleta >───────────────────────────────────────────────────────────────────────────

# fundo oceanográfico: tons de azul-marinho e verde-oliva escuros (no lugar do
# preto puro #0b1220), que contrastam com os marcadores de severidade
# (azul -> verde -> amarelo -> laranja -> vermelho) sem ofuscar a cartografia.

CS_BACKGROUND      = "#0d1b2a"   # figura, fora dos eixos (azul escuro)
CS_BORDER_COLOR    = "#6a9080"   # fronteiras nacionais (verde claro)
CS_COASTLINE_COLOR = "#5a9a8a"   # linha de costa (verde claro)
CF_FORECAST_ALPHA  = 0.20        # transparência do cone de previsão
CS_FORECAST_COLOR  = "#ffffff"   # trajetória, cone e elipses de previsão (branco)
CS_GRID_COLOR      = "#1e3a5e"   # grade lat/lng (azul)
CS_HUD_BG          = "#111827"   # fundo do HUD e da legenda (azul escuro)
CS_LAND_COLOR      = "#2a3d28"   # terra (verde escuro)
CS_OCEAN_COLOR     = "#1b3a5c"   # oceano (azul médio)
CS_STATE_COLOR     = "#4a6a5a"   # limites estaduais (verde claro)
CS_TEXT_COLOR      = "white"     # textos (branco)
CS_TRACK_COLOR     = "#8ecae6"   # trilha das células (azul claro)

# < the end >──────────────────────────────────────────────────────────────────────────
