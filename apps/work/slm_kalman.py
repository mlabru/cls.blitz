# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
slm_kalman
filtro de Kalman que suaviza a posição de cada track e estima sua velocidade.

modelo: dois filtros 1D independentes, um para a latitude e outro para a
longitude (TKalmanTracker = 2 x _TKalman1D).  Cada filtro tem o estado
[posição (graus), velocidade (graus/hora)] e velocidade constante entre frames.

uso por frame, em slm_track.update_tracks:
  1) predict(dt): avança o estado até o instante da nova célula;
  2) update(lat, lng): corrige com a posição observada (o centroide).
A previsão (slm_motion.extrapolate_path) usa predict_position, que não muda
o estado.  A incerteza da posição (std) dá a largura do cone e das elipses
do slmRender.

persistência: to_dict/from_dict gravam/leem o __dict__ de cada _TKalman1D em
slm_tracks_state.json.  Os NOMES dos atributos de _TKalman1D (x, v, p_xx,
p_xv, p_vv, q_pos, q_vel, r) são, portanto, o formato do arquivo: renomear um
deles quebra a leitura de estados já gravados.

2026.nov  mlabru  ruído de processo proporcional a dt (era somado uma vez por
                  predict, fosse dt de 5 min ou de 2 h); velocity_deg_h
2026.out  mlabru  refatoração para legibilidade: docstring do módulo (era cópia
                  da do slm_motion), predict_path inexistente tirado do "uso",
                  logs de debug removidos (rodavam a cada frame/célula)
2026.may  mlabru  TKalmanTracker 2D desacoplado (lat/lng independentes)
2026.may  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import copy
import math

# < constants >────────────────────────────────────────────────────────────────────────

# ruído de processo (posição, graus²) -- quanto maior, mais rápido o filtro
# "esquece" o histórico e confia na medição mais recente; controla, junto com
# CF_KALMAN_Q_VEL/CF_KALMAN_R, o quanto o cone de previsão (forecast_cone em
# slmRender.py) e as elipses de incerteza se alargam com o horizonte de
# previsão.  Sem passagem empírica de ajuste -- valores herdados da versão
# inicial do filtro.
CF_KALMAN_Q_POS = 1e-4
# ruído de processo (velocidade, (graus/hora)²)
CF_KALMAN_Q_VEL = 1e-3
# ruído de medição (posição observada, graus²)
CF_KALMAN_R = 1e-3

# intervalo (horas) a que q_pos e q_vel se referem: o ruído de um predict é
# q · dt / CF_KALMAN_Q_REF_DT_H.  5 min = intervalo do cron, então num frame
# normal o filtro se comporta como antes do ruído passar a depender de dt
CF_KALMAN_Q_REF_DT_H = 5.0 / 60.0

# covariância inicial: posição pouco conhecida (1 grau²), velocidade quase nula
CF_INITIAL_P_POSITION = 1.0
CF_INITIAL_P_VELOCITY = 0.1

# < _TKalman1D >───────────────────────────────────────────────────────────────────────

class _TKalman1D:
    """
    filtro de Kalman de uma coordenada (só lat ou só lng).
    estado: x = posição (graus), v = velocidade (graus/hora).

    a covariância P é simétrica 2x2, então bastam três números:
    p_xx (variância da posição), p_xv (covariância), p_vv (variância da velocidade).

    propagação (F = [[1, dt], [0, 1]], s = dt / CF_KALMAN_Q_REF_DT_H):
        p_xx ← p_xx + 2·dt·p_xv + dt²·p_vv + s·q_pos
        p_xv ← p_xv + dt·p_vv
        p_vv ← p_vv + s·q_vel

    atualização (só a posição é medida, H = [1, 0]):
        K_x  = p_xx / (p_xx + r)
        K_v  = p_xv / (p_xx + r)
        p_xx ← (1 − K_x) · p_xx
        p_xv ← (1 − K_x) · p_xv
        p_vv ← p_vv − K_v · p_xv

    atenção: os nomes dos atributos são o formato persistido (ver o cabeçalho).
    """
    # ─────────────────────────────────────────────────────────────────────────────────
    def __init__(self, x0: float, v0: float = 0.0,
                 q_pos: float = CF_KALMAN_Q_POS, q_vel: float = CF_KALMAN_Q_VEL,
                 r: float = CF_KALMAN_R):
        """
        cria o filtro numa posição conhecida.

        :param x0: posição inicial (graus)
        :param v0: velocidade inicial (graus/hora)
        :param q_pos: ruído de processo da posição
        :param q_vel: ruído de processo da velocidade
        :param r: ruído de medição
        """
        self.x = x0
        self.v = v0

        self.p_xx = CF_INITIAL_P_POSITION
        self.p_xv = 0.0
        self.p_vv = CF_INITIAL_P_VELOCITY

        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r = r

    # ─────────────────────────────────────────────────────────────────────────────────
    def predict(self, ff_dt_h: float) -> None:
        """
        avança estado e covariância dt horas (velocidade constante).

        [fix] o ruído de processo é proporcional a dt: antes era somado uma
        vez por chamada, então um intervalo de 2 h (tracker parado) ou o
        horizonte de +30 min da previsão ganhavam a mesma incerteza extra que
        um frame de 5 min.  dt = 0 não acrescenta ruído.

        :param ff_dt_h: intervalo, em horas (negativo conta como 0 no ruído)
        """
        lf_noise_scale = max(ff_dt_h, 0.0) / CF_KALMAN_Q_REF_DT_H

        self.x = self.x + self.v * ff_dt_h

        # os três novos valores são calculados com os valores ANTIGOS,
        # por isso a atribuição só acontece no fim
        lf_new_p_vv = self.p_vv + lf_noise_scale * self.q_vel
        lf_new_p_xv = self.p_xv + ff_dt_h * self.p_vv
        lf_new_p_xx = (self.p_xx + 2.0 * ff_dt_h * self.p_xv +
                       ff_dt_h * ff_dt_h * self.p_vv + lf_noise_scale * self.q_pos)

        self.p_xx = lf_new_p_xx
        self.p_xv = lf_new_p_xv
        self.p_vv = lf_new_p_vv

    # ─────────────────────────────────────────────────────────────────────────────────
    def update(self, ff_measured_position: float) -> None:
        """
        corrige o estado com uma posição observada.

        :param ff_measured_position: posição medida (graus)
        """
        lf_innovation_variance = self.p_xx + self.r
        lf_gain_position = self.p_xx / lf_innovation_variance
        lf_gain_velocity = self.p_xv / lf_innovation_variance

        # inovação: quanto a medição difere da previsão
        lf_innovation = ff_measured_position - self.x

        self.x += lf_gain_position * lf_innovation
        self.v += lf_gain_velocity * lf_innovation

        # ordem importa: p_vv usa o p_xv antigo, e p_xv é atualizado antes de p_xx
        self.p_vv -= lf_gain_velocity * self.p_xv
        self.p_xv -= lf_gain_position * self.p_xv
        self.p_xx -= lf_gain_position * self.p_xx

    # ─────────────────────────────────────────────────────────────────────────────────
    @property
    def std(self) -> float:
        """
        desvio-padrão da posição estimada.

        :returns: raiz da variância da posição (graus); 0 se a variância
                  ficar negativa por erro numérico
        """
        return math.sqrt(max(self.p_xx, 0.0))

# < TKalmanTracker >───────────────────────────────────────────────────────────────────

class TKalmanTracker:
    """
    rastreador 2D: um _TKalman1D para a latitude e outro para a longitude.

    unidades: posição em graus, velocidade em graus/hora, intervalos em horas.

    uso:
        l_kalman = TKalmanTracker(lat0, lng0)
        l_kalman.predict(dt_h)                              # a cada frame
        l_kalman.update(lat_nova, lng_nova)
        lat, lng, lat_std, lng_std = l_kalman.predict_position(dt_h)  # previsão
    """
    # ─────────────────────────────────────────────────────────────────────────────────
    def __init__(self, ff_lat0: float, ff_lng0: float,
                 ff_vlat: float = 0.0, ff_vlng: float = 0.0):
        """
        cria o rastreador numa posição conhecida.

        :param ff_lat0: latitude inicial (graus)
        :param ff_lng0: longitude inicial (graus)
        :param ff_vlat: velocidade inicial em latitude (graus/hora)
        :param ff_vlng: velocidade inicial em longitude (graus/hora)
        """
        self._lat = _TKalman1D(ff_lat0, ff_vlat)
        self._lng = _TKalman1D(ff_lng0, ff_vlng)

    # ─────────────────────────────────────────────────────────────────────────────────
    def update(self, ff_lat: float, ff_lng: float) -> None:
        """
        corrige o estado com uma posição observada.

        :param ff_lat: latitude observada (graus)
        :param ff_lng: longitude observada (graus)
        """
        self._lat.update(ff_lat)
        self._lng.update(ff_lng)

    # ─────────────────────────────────────────────────────────────────────────────────
    def predict(self, ff_dt_h: float) -> None:
        """
        avança o estado interno dt_h horas (MUDA o rastreador).
        para prever sem mudar o estado, use predict_position().

        :param ff_dt_h: intervalo, em horas
        """
        self._lat.predict(ff_dt_h)
        self._lng.predict(ff_dt_h)

    # ─────────────────────────────────────────────────────────────────────────────────
    def predict_position(self, ff_dt_h: float) -> tuple:
        """
        posição prevista dt_h horas à frente, sem mudar o estado interno.

        [fix] copy.copy (raso) basta e é mais barato que copy.deepcopy:
        _TKalman1D só tem atributos float, nenhum objeto aninhado.

        :param ff_dt_h: horizonte, em horas
        :returns: (lat, lng, lat_std, lng_std), em graus
        """
        l_lat_copy = copy.copy(self._lat)
        l_lng_copy = copy.copy(self._lng)

        l_lat_copy.predict(ff_dt_h)
        l_lng_copy.predict(ff_dt_h)

        return l_lat_copy.x, l_lng_copy.x, l_lat_copy.std, l_lng_copy.std

    # ─────────────────────────────────────────────────────────────────────────────────
    @property
    def velocity_deg_h(self) -> tuple:
        """
        velocidade estimada pelo filtro (slm_motion.kalman_velocity a converte
        em km/h e azimute).

        :returns: (v_lat, v_lng), em graus/hora
        """
        return self._lat.v, self._lng.v

    # ─────────────────────────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """
        estado completo, para gravar em JSON.

        :returns: {"lat": {...}, "lng": {...}} com os atributos de cada _TKalman1D
        """
        return {"lat": vars(self._lat),
                "lng": vars(self._lng)}

    # ─────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, fdct_state: dict) -> TKalmanTracker:
        """
        reconstrói o rastreador a partir do dict gravado por to_dict().

        [doc] usa __new__ + __dict__.update em vez de __init__: o objeto é
        refeito direto do dict salvo, sem passar pelo construtor.  Isso prende
        o formato do JSON aos atributos de _TKalman1D: se um atributo novo for
        criado, um estado gravado por uma versão anterior não o terá e o objeto
        só falha (AttributeError) no primeiro uso, não aqui.  Ao acrescentar ou
        renomear atributos em _TKalman1D, migre ou descarte o arquivo de estado.

        :param fdct_state: dict com as chaves "lat" e "lng"
        :returns: novo TKalmanTracker
        :raises KeyError: se faltar "lat" ou "lng"
        """
        l_tracker = cls.__new__(cls)

        l_tracker._lat = _TKalman1D.__new__(_TKalman1D)
        l_tracker._lat.__dict__.update(fdct_state["lat"])

        l_tracker._lng = _TKalman1D.__new__(_TKalman1D)
        l_tracker._lng.__dict__.update(fdct_state["lng"])

        return l_tracker

# < the end >──────────────────────────────────────────────────────────────────────────
