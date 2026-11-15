# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
blt_kalman
filtro de Kalman que suaviza a posição de cada track e estima sua velocidade.

modelo: dois filtros 1D independentes, um para a latitude e outro para a
longitude (TKalmanTracker = 2 x _TKalman1D).  Cada filtro tem o estado
[posição (graus), velocidade (graus/hora)] e velocidade constante entre frames.

uso por frame, em blt_track.update_tracks:
  1) predict(dt): avança o estado até o instante da nova célula;
  2) update(lat, lng): corrige com a posição observada (o centroide).
A previsão (blt_motion.extrapolate_path) usa predict_position, que não muda
o estado.  A incerteza da posição (std) dá a largura do cone e das elipses
do bltRender.

persistência: to_dict/from_dict gravam/leem o __dict__ de cada _TKalman1D em
slm_tracks_state.json.  Os NOMES dos atributos de _TKalman1D (x, v, p_xx,
p_xv, p_vv, q_pos, q_vel, r) são, portanto, o formato do arquivo: renomear um
deles quebra a leitura de estados já gravados.

unidades: o estado está em graus, mas os ruídos são dados em km (constantes
CF_*_STD_KM/_KMH) e cada eixo os converte pela sua escala km/grau na criação
do filtro (a da longitude na latitude de nascimento do track).  Assim o filtro
é isotrópico em km: um grau de longitude a 33°S tem só 84% de um de latitude.
q_pos/q_vel/r gravados continuam em graus², então estados antigos valem.

2026.nov  mlabru  TKalmanTracker.t (instante do estado, gravado como "t"; None em
                  estados antigos) e ruído de medida por update (ff_r_std_km:
                  a medida dos eventos novos tem erro que depende de n)
2026.nov  mlabru  from_velocity_prior (track novo com a velocidade dos vizinhos,
                  blt_track) e velocity_kmh
2026.nov  mlabru  ruídos em km, convertidos por eixo (eram graus² iguais nos dois
                  eixos: a longitude ficava com ruído menor em km)
2026.nov  mlabru  filtro novo com variância da posição = r (era 1 grau², ~111 km)
2026.nov  mlabru  nis (inovação normalizada, diagnóstico da associação); ruídos
                  padrão lidos na criação do filtro (o bltReplay sobrescreve
                  as constantes)
2026.nov  mlabru  from_two_points (velocidade inicial pelo deslocamento dos dois
                  primeiros frames) e spawn_at (track filho herda a velocidade)
2026.nov  mlabru  ruído de processo proporcional a dt (era somado uma vez por
                  predict, fosse dt de 5 min ou de 2 h); velocity_deg_h
2026.out  mlabru  refatoração para legibilidade: docstring do módulo (era cópia
                  da do blt_motion), predict_path inexistente tirado do "uso",
                  logs de debug removidos (rodavam a cada frame/célula)
2026.may  mlabru  TKalmanTracker 2D desacoplado (lat/lng independentes)
2026.may  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import copy
import math

# locals
import work.blt_geo as geo

# < constants >────────────────────────────────────────────────────────────────────────

# ruídos em DESVIO-PADRÃO, em km e km/h (cada eixo converte para graus na
# criação do filtro, ver _TKalman1D.__init__).  Controlam o ganho do filtro (a
# razão Q/R) e a largura do cone de previsão e das elipses do bltRender (a
# escala comum).  Os valores são os antigos em graus (1e-4, 1e-3, 1e-3 graus²)
# convertidos a 111 km/grau; ainda sem ajuste empírico (guia, seção 10).

# ruído de processo da posição, por CF_KALMAN_Q_REF_DT_H (km)
CF_KALMAN_Q_POS_STD_KM = 1.11
# ruído de processo da velocidade, por CF_KALMAN_Q_REF_DT_H (km/h)
CF_KALMAN_Q_VEL_STD_KMH = 3.51
# ruído de medição (posição do centroide, km)
CF_KALMAN_R_STD_KM = 3.51

# intervalo (horas) a que q_pos e q_vel se referem: o ruído de um predict é
# q · dt / CF_KALMAN_Q_REF_DT_H.  5 min = intervalo do cron, então num frame
# normal o filtro se comporta como antes do ruído passar a depender de dt
CF_KALMAN_Q_REF_DT_H = 5.0 / 60.0

# incerteza inicial da velocidade (km/h); a da posição é a da medida (r),
# porque o filtro nasce numa posição medida (ver _TKalman1D.__init__)
CF_INITIAL_VEL_STD_KMH = 35.1

# ─────────────────────────────────────────────────────────────────────────────────────
def _deg_var(ff_std_km: float, ff_km_per_deg: float) -> float:
    """
    converte um desvio-padrão em km (ou km/h) na variância em graus² (ou
    (graus/hora)²) de um eixo.

    :param ff_std_km: desvio-padrão em km (ou km/h)
    :param ff_km_per_deg: km por grau do eixo
    :returns: variância em graus² (ou (graus/hora)²)
    """
    lf_std_deg = ff_std_km / ff_km_per_deg

    return lf_std_deg * lf_std_deg

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
                 q_pos: float | None = None, q_vel: float | None = None,
                 r: float | None = None, ff_km_per_deg: float | None = None):
        """
        cria o filtro numa posição conhecida.

        [fix] os padrões (None) são lidos das constantes NA CRIAÇÃO, não na
        definição da função: assim o bltReplay consegue trocar
        CF_KALMAN_*_STD_* e o filtro novo usa o valor trocado.

        [fix] os ruídos vinham em graus², iguais nos dois eixos: em km, o da
        longitude saía menor (cos(lat)), e o cone ficava estreito no leste-oeste.
        Agora as constantes estão em km e são convertidas pela escala deste eixo.

        :param x0: posição inicial (graus)
        :param v0: velocidade inicial (graus/hora)
        :param q_pos: ruído de processo da posição, graus² (None = das constantes)
        :param q_vel: ruído de processo da velocidade, (graus/hora)² (idem)
        :param r: ruído de medição, graus² (idem)
        :param ff_km_per_deg: km por grau deste eixo (None = geo.CF_KM_PER_DEG_LAT)
        """
        lf_km_per_deg = geo.CF_KM_PER_DEG_LAT if ff_km_per_deg is None else ff_km_per_deg

        self.x = x0
        self.v = v0

        self.q_pos = _deg_var(CF_KALMAN_Q_POS_STD_KM, lf_km_per_deg) if q_pos is None else q_pos
        self.q_vel = _deg_var(CF_KALMAN_Q_VEL_STD_KMH, lf_km_per_deg) if q_vel is None else q_vel
        self.r = _deg_var(CF_KALMAN_R_STD_KM, lf_km_per_deg) if r is None else r

        # [fix] p_xx era CF_INITIAL_P_POSITION (1 grau², ~111 km de sigma): a
        # 1a previsão de todo track novo saía com elipses de ~4,5 graus no render
        # e puxava o E[z2] da verificação para baixo.  x0 é um centroide medido,
        # então a variância da posição é a da medida
        self.p_xx = self.r
        self.p_xv = 0.0
        self.p_vv = _deg_var(CF_INITIAL_VEL_STD_KMH, lf_km_per_deg)

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
    def normalized_innovation_sq(self, ff_measured_position: float,
                                 ff_r: float | None = None) -> float:
        """
        inovação normalizada ao quadrado: (medida − previsão)² / (p_xx + r).
        Chamar DEPOIS do predict e ANTES do update.

        se o filtro está bem calibrado, a média é 1 por eixo (soma dos dois
        eixos: 2, ver TKalmanTracker.nis).

        :param ff_measured_position: posição medida (graus)
        :param ff_r: ruído desta medida (graus²); None = self.r
        :returns: valor adimensional >= 0
        """
        lf_innovation = ff_measured_position - self.x
        lf_r = self.r if ff_r is None else ff_r

        return lf_innovation * lf_innovation / (self.p_xx + lf_r)

    # ─────────────────────────────────────────────────────────────────────────────────
    def update(self, ff_measured_position: float, ff_r: float | None = None) -> None:
        """
        corrige o estado com uma posição observada.

        :param ff_measured_position: posição medida (graus)
        :param ff_r: ruído desta medida (graus²); None = self.r
        """
        lf_innovation_variance = self.p_xx + (self.r if ff_r is None else ff_r)
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
    @classmethod
    def from_two_points(cls, ff_x1: float, ff_x2: float, ff_dt_h: float,
                        ff_km_per_deg: float | None = None) -> _TKalman1D:
        """
        filtro iniciado por duas posições medidas ff_dt_h horas uma da outra
        (inicialização de dois pontos):

            x = x2,  v = (x2 − x1) / dt
            p_xx = r,  p_xv = r / dt,  p_vv = 2·r / dt²

        a covariância é a que sai de duas medidas com ruído r cada: a
        velocidade é tão incerta quanto o deslocamento medido permite.

        :param ff_x1: posição anterior (graus)
        :param ff_x2: posição atual (graus)
        :param ff_dt_h: intervalo entre as duas, em horas (> 0)
        :param ff_km_per_deg: km por grau deste eixo (None = geo.CF_KM_PER_DEG_LAT)
        :returns: novo filtro, com os ruídos padrão do módulo
        """
        l_filter = cls(ff_x2, (ff_x2 - ff_x1) / ff_dt_h, ff_km_per_deg=ff_km_per_deg)

        l_filter.p_xx = l_filter.r
        l_filter.p_xv = l_filter.r / ff_dt_h
        l_filter.p_vv = 2.0 * l_filter.r / (ff_dt_h * ff_dt_h)

        return l_filter

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
                 ff_vlat: float = 0.0, ff_vlng: float = 0.0, ff_t: float | None = None):
        """
        cria o rastreador numa posição conhecida.

        :param ff_lat0: latitude inicial (graus)
        :param ff_lng0: longitude inicial (graus)
        :param ff_vlat: velocidade inicial em latitude (graus/hora)
        :param ff_vlng: velocidade inicial em longitude (graus/hora)
        :param ff_t: instante da posição (epoch UTC); None = desconhecido
        """
        self._lat = _TKalman1D(ff_lat0, ff_vlat)
        self._lng = _TKalman1D(ff_lng0, ff_vlng, ff_km_per_deg=geo.km_per_deg_lng(ff_lat0))
        # [new] instante a que o estado se refere (o da última medida): o
        # predict até uma célula conta daqui.  None = estado antigo, sem o
        # campo (blt_track usa o instante da última célula)
        self.t = ff_t

    # ─────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_two_points(cls, ft_pos1: tuple, ft_pos2: tuple, ff_dt_h: float,
                        ff_t: float | None = None) -> TKalmanTracker:
        """
        rastreador iniciado pelas duas primeiras posições de um track.

        [fix] o filtro nascia com velocidade zero e P_vv pequeno (0.1): levava
        6-7 frames (~30 min) para chegar à velocidade real, e a seta e o cone
        ficavam parados nesse tempo.  Com dois pontos, a velocidade já é a do
        deslocamento medido, e a incerteza dela é a que esse deslocamento
        permite (ver _TKalman1D.from_two_points).

        :param ft_pos1: (lat, lng) anterior
        :param ft_pos2: (lat, lng) atual
        :param ff_dt_h: intervalo entre as duas, em horas (> 0)
        :param ff_t: instante da posição atual (epoch UTC)
        :returns: novo rastreador na posição atual
        """
        l_tracker = cls.__new__(cls)
        l_tracker.t = ff_t
        l_tracker._lat = _TKalman1D.from_two_points(ft_pos1[0], ft_pos2[0], ff_dt_h)
        l_tracker._lng = _TKalman1D.from_two_points(ft_pos1[1], ft_pos2[1], ff_dt_h,
                                                    geo.km_per_deg_lng(ft_pos2[0]))

        return l_tracker

    # ─────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_velocity_prior(cls, ff_lat: float, ff_lng: float,
                            ft_velocity_kmh: tuple, ft_velocity_std_kmh: tuple,
                            ff_t: float | None = None) -> TKalmanTracker:
        """
        rastreador novo numa posição medida, com uma velocidade a priori (a
        dos tracks vizinhos, blt_track._neighbor_velocity_prior) e a incerteza
        dela, em vez de velocidade zero com CF_INITIAL_VEL_STD_KMH.

        :param ff_lat: latitude (graus)
        :param ff_lng: longitude (graus)
        :param ft_velocity_kmh: (norte, leste) em km/h
        :param ft_velocity_std_kmh: desvio-padrão (norte, leste) em km/h
        :param ff_t: instante da posição (epoch UTC)
        :returns: novo rastreador
        """
        lf_km_per_deg_lng = geo.km_per_deg_lng(ff_lat)

        l_tracker = cls(ff_lat, ff_lng,
                        ft_velocity_kmh[0] / geo.CF_KM_PER_DEG_LAT,
                        ft_velocity_kmh[1] / lf_km_per_deg_lng, ff_t)

        l_tracker._lat.p_vv = _deg_var(ft_velocity_std_kmh[0], geo.CF_KM_PER_DEG_LAT)
        l_tracker._lng.p_vv = _deg_var(ft_velocity_std_kmh[1], lf_km_per_deg_lng)

        return l_tracker

    # ─────────────────────────────────────────────────────────────────────────────────
    def velocity_kmh(self, ff_lat: float) -> tuple:
        """
        velocidade estimada em km/h, por eixo.

        :param ff_lat: latitude em que converter a longitude (graus)
        :returns: (norte, leste) em km/h
        """
        return (self._lat.v * geo.CF_KM_PER_DEG_LAT,
                self._lng.v * geo.km_per_deg_lng(ff_lat))

    # ─────────────────────────────────────────────────────────────────────────────────
    def spawn_at(self, ff_lat: float, ff_lng: float) -> TKalmanTracker:
        """
        cópia deste rastreador movida para outra posição: a velocidade e as
        covariâncias são herdadas (track filho de uma divisão, que se move com
        o pai).  Este rastreador não muda.

        :param ff_lat: latitude do filho (graus)
        :param ff_lng: longitude do filho (graus)
        :returns: novo rastreador
        """
        l_child = copy.deepcopy(self)
        l_child._lat.x = ff_lat
        l_child._lng.x = ff_lng

        return l_child

    # ─────────────────────────────────────────────────────────────────────────────────
    def update(self, ff_lat: float, ff_lng: float, ff_r_std_km: float | None = None) -> None:
        """
        corrige o estado com uma posição observada.

        :param ff_lat: latitude observada (graus)
        :param ff_lng: longitude observada (graus)
        :param ff_r_std_km: desvio-padrão desta medida (km, por eixo); None =
                            o ruído do filtro (CF_KALMAN_R_STD_KM na criação)
        """
        lf_r_lat, lf_r_lng = self._measurement_var(ff_lat, ff_r_std_km)

        self._lat.update(ff_lat, lf_r_lat)
        self._lng.update(ff_lng, lf_r_lng)

    # ─────────────────────────────────────────────────────────────────────────────────
    def _measurement_var(self, ff_lat: float, ff_r_std_km: float | None) -> tuple:
        """
        o ruído de uma medida em graus², por eixo.

        :param ff_lat: latitude da medida (escala da longitude)
        :param ff_r_std_km: desvio-padrão em km; None = o de cada filtro
        :returns: (r_lat, r_lng), ou (None, None) para usar o do filtro
        """
        if ff_r_std_km is None:
            return None, None

        return (_deg_var(ff_r_std_km, geo.CF_KM_PER_DEG_LAT),
                _deg_var(ff_r_std_km, geo.km_per_deg_lng(ff_lat)))

    # ─────────────────────────────────────────────────────────────────────────────────
    def nis(self, ff_lat: float, ff_lng: float, ff_r_std_km: float | None = None) -> float:
        """
        NIS (normalized innovation squared) da posição observada: a soma da
        inovação normalizada dos dois eixos.  Chamar DEPOIS do predict e ANTES
        do update (blt_track._continue_track).

        [new] diagnóstico da calibração do Kalman: com Q e R coerentes com os
        dados, o NIS segue uma qui-quadrado de 2 graus de liberdade -- média
        2, 95% abaixo de 5,99.  Média bem acima de 2: o filtro é confiante
        demais (aumentar Q ou R); bem abaixo: cauteloso demais.

        :param ff_lat: latitude observada (graus)
        :param ff_lng: longitude observada (graus)
        :param ff_r_std_km: desvio-padrão desta medida (km); None = o do filtro
        :returns: NIS (adimensional, >= 0)
        """
        lf_r_lat, lf_r_lng = self._measurement_var(ff_lat, ff_r_std_km)

        return (self._lat.normalized_innovation_sq(ff_lat, lf_r_lat) +
                self._lng.normalized_innovation_sq(ff_lng, lf_r_lng))

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
        velocidade estimada pelo filtro (blt_motion.kalman_velocity a converte
        em km/h e azimute).

        :returns: (v_lat, v_lng), em graus/hora
        """
        return self._lat.v, self._lng.v

    # ─────────────────────────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """
        estado completo, para gravar em JSON.

        :returns: {"lat": {...}, "lng": {...}, "t": instante} com os atributos
                  de cada _TKalman1D
        """
        return {"lat": vars(self._lat),
                "lng": vars(self._lng),
                "t":   self.t}

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

        # estados gravados antes de 2026.nov não têm o instante
        l_tracker.t = fdct_state.get("t")

        return l_tracker

# < the end >──────────────────────────────────────────────────────────────────────────
