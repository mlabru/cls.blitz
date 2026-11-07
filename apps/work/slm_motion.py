# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
slm_motion
movimento das células: velocidade, direção e previsão de posição.

  kalman_velocity
      velocidade (km/h) e azimute estimados pelo filtro de Kalman do track:
      é o movimento que vai para o JSON (seta do slmRender), o mesmo que
      extrapolate_path usa no cone de previsão;
  compute_speed_kmh / compute_direction
      movimento instantâneo entre duas células consecutivas do mesmo track;
  compute_velocity_ema
      movimento suavizado sobre o histórico do track (média exponencial:
      o par mais recente pesa mais);
  [doc] desde 2026.nov estas três não alimentam mais o pipeline (a velocidade
  vem do Kalman); ficam como ferramenta de análise, cobertas pelos testes;
  extrapolate_path
      posições previstas para +10/+20/+30 min (CT_FORECAST_LEADS_MIN), com a
      incerteza 1-sigma do filtro de Kalman (slm_kalman).

2026.nov  mlabru  [fix] elapsed_hours pelo sst.cell_time (era t_start, parado
                  nos primeiros 20 min de uma célula com a janela fixa)
2026.nov  mlabru  kalman_velocity: uma só velocidade (a do Kalman) para a seta
                  e o cone; antes a seta usava a EMA dos centroides e podia
                  apontar para outro lado
2026.out  mlabru  refatoração para legibilidade: elapsed_hours() reúne a conta
                  de horas entre células (estava em três lugares), passo
                  ponderado da EMA em função própria, logs de debug removidos;
                  extrapolate_position (só usada por teste) foi para
                  apps/.bkp/slm_motion_extrapolate_position.snip
2026.may  mlabru  compute_velocity_ema com média circular de direção;
                  extrapolate_path retorna incerteza por lead;
                  CT_FORECAST_LEADS_MIN como constante nomeada
2026.may  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import math

# locals
import work.slm_geo as geo
import work.slm_kalman as klm
import work.slm_spatial as spc
import work.slm_struct as sst

# < constants >────────────────────────────────────────────────────────────────────────

# fator de suavização da EMA de velocidade [0-1]; maior = mais reativo ao frame atual
CF_EMA_ALPHA = 0.4

# horizontes de previsão (minutos)
# [fix] eram fixos em (10, 20, 30) dentro da função
CT_FORECAST_LEADS_MIN = (10, 20, 30)

# segundos por hora / minutos por hora
CF_SECONDS_PER_HOUR = 3600.0
CF_MINUTES_PER_HOUR = 60.0

# ─────────────────────────────────────────────────────────────────────────────────────
def elapsed_hours(f_previous_cell: sst.TCell, f_current_cell: sst.TCell) -> float:
    """
    tempo entre duas células, em horas, pelo relógio do tracking
    (sst.cell_time: o instante médio dos raios de cada uma).

    :param f_previous_cell: célula anterior
    :param f_current_cell: célula atual
    :returns: (instante atual - instante anterior) / 3600; pode ser zero ou negativo
    """
    return (sst.cell_time(f_current_cell) - sst.cell_time(f_previous_cell)) / CF_SECONDS_PER_HOUR

# ─────────────────────────────────────────────────────────────────────────────────────
def bearing_deg(ff_lat1: float, ff_lng1: float, ff_lat2: float, ff_lng2: float) -> float:
    """
    azimute geodésico do ponto 1 para o ponto 2 (0 = norte, 90 = leste).

    :param ff_lat1: latitude do ponto 1 (graus)
    :param ff_lng1: longitude do ponto 1 (graus)
    :param ff_lat2: latitude do ponto 2 (graus)
    :param ff_lng2: longitude do ponto 2 (graus)
    :returns: azimute em graus, no intervalo [0, 360)
    """
    lf_lat1_rad = math.radians(ff_lat1)
    lf_lat2_rad = math.radians(ff_lat2)
    lf_delta_lng_rad = math.radians(ff_lng2 - ff_lng1)

    lf_east_component = math.sin(lf_delta_lng_rad) * math.cos(lf_lat2_rad)
    lf_north_component = (math.cos(lf_lat1_rad) * math.sin(lf_lat2_rad) -
                          math.sin(lf_lat1_rad) * math.cos(lf_lat2_rad) *
                          math.cos(lf_delta_lng_rad))

    lf_bearing = math.degrees(math.atan2(lf_east_component, lf_north_component))

    # atan2 devolve (-180, 180]; somar 360 e tirar o resto leva para [0, 360)
    return (lf_bearing + 360.0) % 360.0

# ─────────────────────────────────────────────────────────────────────────────────────
def compute_direction(f_previous_cell: sst.TCell, f_current_cell: sst.TCell) -> float:
    """
    direção do movimento entre duas células consecutivas.

    :param f_previous_cell: célula anterior
    :param f_current_cell: célula atual
    :returns: azimute em graus [0, 360)
    """
    return bearing_deg(f_previous_cell.centroid[0], f_previous_cell.centroid[1],
                       f_current_cell.centroid[0], f_current_cell.centroid[1])

# ─────────────────────────────────────────────────────────────────────────────────────
def compute_speed_kmh(f_previous_cell: sst.TCell, f_current_cell: sst.TCell) -> float:
    """
    velocidade entre duas células consecutivas.

    :param f_previous_cell: célula anterior
    :param f_current_cell: célula atual
    :returns: km/h; 0.0 se o intervalo entre as células não for positivo
    """
    lf_hours = elapsed_hours(f_previous_cell, f_current_cell)

    if lf_hours <= 0.0:
        return 0.0

    lf_distance_km = spc.haversine_km(f_previous_cell.centroid[0], f_previous_cell.centroid[1],
                                      f_current_cell.centroid[0], f_current_cell.centroid[1])

    return lf_distance_km / lf_hours

# ─────────────────────────────────────────────────────────────────────────────────────
def compute_velocity_ema(flst_history: list, ff_alpha: float = CF_EMA_ALPHA) -> tuple:
    """
    velocidade e direção suavizadas por média exponencial (EMA) sobre o
    histórico do track.

    cada par consecutivo de células dá uma velocidade e uma direção; os pares
    são percorridos do mais recente para o mais antigo, com peso 1, (1-alpha),
    (1-alpha)², ...  A direção é média de vetores unitários (seno e cosseno)
    para não errar perto de 0°/360° (a média de 350° e 10° é 0°, não 180°).
    Pares com intervalo de tempo não positivo são pulados (sem consumir peso).

    :param flst_history: lista de TCell, da mais antiga para a mais recente
    :param ff_alpha: fator de suavização [0-1]
    :returns: (velocidade em km/h, direção em graus [0, 360)); (0.0, 0.0) se
              houver menos de 2 células ou nenhum par válido
    """
    if len(flst_history) < 2:
        return 0.0, 0.0

    lf_weighted_speed_sum = 0.0
    lf_weighted_sin_sum = 0.0
    lf_weighted_cos_sum = 0.0
    lf_weight_sum = 0.0
    lf_weight = 1.0

    # do par mais recente (último, penúltimo) para o mais antigo
    for li_index in range(len(flst_history) - 1, 0, -1):
        l_previous_cell = flst_history[li_index - 1]
        l_current_cell = flst_history[li_index]

        lf_hours = elapsed_hours(l_previous_cell, l_current_cell)

        if lf_hours <= 0.0:
            continue

        lf_speed_kmh, lf_bearing_rad = _pair_speed_and_bearing(l_previous_cell,
                                                               l_current_cell, lf_hours)

        lf_weighted_speed_sum += lf_weight * lf_speed_kmh
        lf_weighted_sin_sum += lf_weight * math.sin(lf_bearing_rad)
        lf_weighted_cos_sum += lf_weight * math.cos(lf_bearing_rad)
        lf_weight_sum += lf_weight

        # quanto mais antigo o par, menor o peso
        lf_weight *= (1.0 - ff_alpha)

    if 0.0 == lf_weight_sum:
        return 0.0, 0.0

    lf_mean_speed = lf_weighted_speed_sum / lf_weight_sum
    lf_mean_direction = math.degrees(math.atan2(lf_weighted_sin_sum / lf_weight_sum,
                                                lf_weighted_cos_sum / lf_weight_sum)) % 360.0

    return lf_mean_speed, lf_mean_direction

# ─────────────────────────────────────────────────────────────────────────────────────
def _pair_speed_and_bearing(f_previous_cell: sst.TCell, f_current_cell: sst.TCell,
                            ff_hours: float) -> tuple:
    """
    velocidade e azimute de um par de células (passo da EMA).

    :param f_previous_cell: célula anterior
    :param f_current_cell: célula atual
    :param ff_hours: intervalo entre elas, em horas (> 0)
    :returns: (velocidade em km/h, azimute em RADIANOS)
    """
    lf_distance_km = spc.haversine_km(f_previous_cell.centroid[0], f_previous_cell.centroid[1],
                                      f_current_cell.centroid[0], f_current_cell.centroid[1])

    lf_bearing = bearing_deg(f_previous_cell.centroid[0], f_previous_cell.centroid[1],
                             f_current_cell.centroid[0], f_current_cell.centroid[1])

    return lf_distance_km / ff_hours, math.radians(lf_bearing)

# ─────────────────────────────────────────────────────────────────────────────────────
def kalman_velocity(f_tracker: klm.TKalmanTracker, ff_lat_deg: float) -> tuple:
    """
    velocidade e azimute do track segundo o filtro de Kalman.

    [fix] antes a célula recebia a EMA das velocidades entre centroides
    (compute_velocity_ema) e a previsão usava o Kalman: com o ruído típico
    do centroide, a seta e o cone podiam apontar para lados diferentes.

    as componentes em graus/hora viram km/h eixo a eixo (slm_geo): um grau
    de longitude encolhe com a latitude.

    :param f_tracker: filtro do track (já atualizado com a célula atual)
    :param ff_lat_deg: latitude onde converter a longitude (a da célula)
    :returns: (velocidade em km/h, azimute em graus: 0 = norte, 90 = leste);
              (0.0, 0.0) se o filtro estiver parado
    """
    lf_v_lat, lf_v_lng = f_tracker.velocity_deg_h

    lf_north_kmh = lf_v_lat * geo.CF_KM_PER_DEG_LAT
    lf_east_kmh = lf_v_lng * geo.km_per_deg_lng(ff_lat_deg)

    lf_speed_kmh = math.hypot(lf_north_kmh, lf_east_kmh)

    if 0.0 == lf_speed_kmh:
        return 0.0, 0.0

    lf_direction_deg = math.degrees(math.atan2(lf_east_kmh, lf_north_kmh)) % 360.0

    return lf_speed_kmh, lf_direction_deg

# ─────────────────────────────────────────────────────────────────────────────────────
def extrapolate_path(f_tracker: klm.TKalmanTracker,
                     ft_leads_min: tuple = CT_FORECAST_LEADS_MIN) -> list:
    """
    trajetória prevista de uma célula, pelo filtro de Kalman do seu track.

    [fix] os horizontes são a constante CT_FORECAST_LEADS_MIN.
    [new] cada ponto traz lat_std e lng_std: incerteza 1-sigma em GRAUS
          (o render converte para km eixo a eixo).

    :param f_tracker: filtro de Kalman do track (não é alterado)
    :param ft_leads_min: horizontes, em minutos
    :returns: lista de dicts {lead_min, lat, lng, lat_std, lng_std}, um por horizonte
    """
    llst_forecast = []

    for li_lead_min in ft_leads_min:
        lf_lead_hours = li_lead_min / CF_MINUTES_PER_HOUR

        lf_lat, lf_lng, lf_lat_std, lf_lng_std = f_tracker.predict_position(lf_lead_hours)

        llst_forecast.append({
            "lead_min": li_lead_min,
            "lat":      lf_lat,
            "lng":      lf_lng,
            "lat_std":  lf_lat_std,
            "lng_std":  lf_lng_std,
        })

    return llst_forecast

# < the end >──────────────────────────────────────────────────────────────────────────
