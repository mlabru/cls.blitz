# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
blt_geo
constantes geodésicas e conversão km <-> graus, num lugar só.

por quê: a mesma conta (111 km por grau de latitude; um grau de longitude
encolhe por cos(lat)) estava espalhada como número mágico em blt_spcgrid,
blt_mapdefs e bltRender.  No Brasil a latitude vai de +5° a -34°: em -30°
um grau de longitude tem só ~96 km, então tratar lat e lng como iguais
distorce distâncias e elipses.  Distância entre dois pontos é sempre com
blt_spatial.haversine_km; este módulo serve para converter um raio/deslocamento
em km para graus, eixo a eixo.

2026.out  mlabru  initial version: reúne CF_EARTH_RADIUS_KM (blt_spatial),
                  CF_KM_PER_DEG_LAT (blt_mapdefs/blt_spcgrid) e a conta de
                  km por grau de longitude (duplicada em bltRender)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import math

# < constants >────────────────────────────────────────────────────────────────────────

# raio médio da Terra (km), usado pela fórmula de haversine e pelo ponto de destino
CF_EARTH_RADIUS_KM = 6371.0

# km por grau de latitude (aproximação usada no projeto todo; o valor exato,
# 2·π·6371/360 ≈ 111.19, não é usado para não mudar os produtos já gerados)
CF_KM_PER_DEG_LAT = 111.0

# piso para km por grau de longitude: perto dos polos cos(lat) → 0 e a divisão
# km / (km por grau) explodiria; no Brasil nunca é atingido
CF_MIN_KM_PER_DEG_LNG = 1.0

# ─────────────────────────────────────────────────────────────────────────────────────
def km_per_deg_lng(ff_latitude_deg: float) -> float:
    """
    quantos km tem um grau de longitude na latitude dada.

    :param ff_latitude_deg: latitude (graus)
    :returns: km por grau de longitude, nunca menor que CF_MIN_KM_PER_DEG_LNG
    """
    lf_km = CF_KM_PER_DEG_LAT * math.cos(math.radians(ff_latitude_deg))

    return max(CF_MIN_KM_PER_DEG_LNG, lf_km)

# < the end >──────────────────────────────────────────────────────────────────────────
