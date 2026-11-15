# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
blt_track
acompanhamento (tracking) das células convectivas de um frame para o outro.

  - estatísticas da célula: centroide, densidade, taxa de raios;
  - associação: qual célula do frame atual continua qual track
    (associate_cells, algoritmo húngaro = melhor combinação global, distância
    medida até a posição prevista pelo Kalman);
  - ciclo de vida dos tracks (update_tracks): histórico, velocidade suavizada
    (EMA), filtro de Kalman, frames perdidos e desativação; expiração por
    tempo (expire_stale_tracks);
  - persistência entre execuções (load_tracks / save_tracks) em
    slm_tracks_state.json -- apagar esse arquivo zera todos os IDs de track.

fluxo de um frame (chamado por bltTracker.process):
  load_tracks -> expire_stale_tracks -> associate_cells -> update_tracks -> save_tracks

2026.nov  mlabru  medida do Kalman só com os eventos novos do cluster
                  (measurement_from_events, TCell.meas): centroide e instante
                  médio dos eventos desde o frame anterior, ruído que cai com
                  o número deles; o filtro guarda o instante do estado
                  (TKalmanTracker.t) e o predict conta dele (_kalman_time)
2026.nov  mlabru  velocidade inicial pelos tracks vizinhos
                  (_neighbor_velocity_prior): o track novo nasce com a média
                  ponderada da velocidade dos tracks maduros próximos e não é
                  reiniciado por dois pontos (velocidade de ~3 min de
                  deslocamento do centroide, pior que a persistência)
2026.nov  mlabru  diagnóstico da associação em TCell.assoc (distância até a
                  previsão, score, NIS; vizinho mais próximo das células sem
                  par) e motivo do fim do track (TStormTrack.end_reason);
                  parâmetros lidos na chamada (o bltReplay os sobrescreve)
2026.nov  mlabru  divisão (track filho com parent_id e a velocidade do pai) e
                  fusão (track absorvido desativado na hora); velocidade
                  inicial do Kalman pelos dois primeiros frames
2026.nov  mlabru  newest_ts no estado (load_applied_ts): frame não é reaplicado
                  se o shell morrer antes de gravar .slm_last_ts
2026.nov  mlabru  corte explícito de score (CF_MAX_MATCH_SCORE) na associação
2026.nov  mlabru  forecasts do track persistidos (verificação, blt_verify)
2026.nov  mlabru  expire_stale_tracks (tracks expiram por tempo, não só por
                  frames perdidos); associate_cells mede a distância até a
                  posição prevista pelo Kalman no instante da célula
2026.out  mlabru  refatoração para legibilidade: update_tracks, associate_cells,
                  load_tracks e save_tracks divididos em funções de um passo
                  só; horas entre células por blt_motion.elapsed_hours; logs
                  de debug removidos, logging com % em vez de f-string
2026.may  mlabru  centroid_distance_km substituído por haversine real;
                  match_score normalizado [0,1], componentes separados;
                  associate_cells com algoritmo húngaro (global ótimo);
                  update_tracks implementado com dedup de novos tracks;
                  MAX_HISTORY e MAX_MISSED_FRAMES agora utilizados;
                  compute_density sem parâmetro morto haversine_km;
                  load_tracks / save_tracks para persistência entre frames
2026.may  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import json
import logging
import math
import os
import uuid

# numPy
import numpy as np
# sciPy: linear_sum_assignment é o algoritmo húngaro
import scipy.optimize as opt

# locals
import work.blt_ioutil as iou
import work.blt_kalman as klm
import work.blt_motion as mtn
import work.blt_spatial as spc
import work.blt_struct as sst

# < constants >────────────────────────────────────────────────────────────────────────

# máximo de células mantidas no histórico de cada track
CI_MAX_HISTORY = 12

# frames seguidos sem associação antes de desativar o track
CI_MAX_MISSED_FRAMES = 3

# divisão/fusão (update_tracks): distância (km) entre uma célula sem par e a
# célula atual de um track que continuou (divisão), ou entre a posição
# prevista de um track sem par e a célula que continuou outro track (fusão).
# Da ordem de 3x o eps do DBSCAN (12 km): fragmentos de uma mesma tempestade
CF_SPLIT_MERGE_DIST_KM = 40.0

# intervalo mínimo (minutos) entre as duas primeiras células de um track para
# iniciar a velocidade do Kalman pelo deslocamento (klm.from_two_points);
# abaixo disso o deslocamento é quase só ruído do centroide.  2 min, não 5:
# enquanto a célula é mais nova que a janela de 20 min, o instante médio dos
# raios (sst.cell_time) anda só ~2,5 min por frame de 5 min
CF_MIN_TWO_POINT_DT_MIN = 2.0

# minutos sem raio (desde o t_end da última célula) antes de expirar o track,
# conte ou não frames perdidos -- cobre o tracker parado ou rodadas sem célula
CF_MAX_TRACK_GAP_MIN = 30.0

# distância (km) acima da qual duas células nunca são associadas
CF_MAX_DIST_KM = 100.0

# pesos do match_score (somam 1, então o score fica em [0, 1])
CF_W_DIST = 0.6   # penalidade de distância
CF_W_SIZE = 0.4   # penalidade de diferença de tamanho

# custo "sem associação" na matriz do algoritmo húngaro (o máximo do match_score)
CF_NO_MATCH_COST = 1.0

# maior match_score aceito numa associação.  [fix] sem este corte, qualquer
# par a menos de CF_MAX_DIST_KM tinha score < 1 e era aceito, e o tamanho
# nunca decidia nada.  Com 0.6, exemplos (distância até a posição prevista):
#   mesmo tamanho         -> aceito até ~100 km
#   tamanho 1:2 (r = 0.5) -> aceito até ~67 km
#   tamanho 1:4 (r = 0.25)-> aceito até ~50 km
#   tamanho 1:10 (r = 0.1)-> aceito até ~40 km
# no mesmo lugar, nenhuma diferença de tamanho sozinha recusa o par (células
# crescem e dissipam rápido).  Ajuste com o slm_verification.csv
CF_MAX_MATCH_SCORE = 0.6

# raio mínimo (km) no cálculo de densidade, para não dividir por zero
CF_MIN_DENSITY_RADIUS_KM = 1.0

# comprimento do ID de track (caracteres do UUID4)
CI_TRACK_ID_LENGTH = 8

# [new] velocidade inicial pelos vizinhos (_neighbor_velocity_prior): um track
# novo herda a média ponderada da velocidade dos tracks maduros que continuaram
# neste frame a até CF_PRIOR_RADIUS_KM.  Tempestades próximas andam com o mesmo
# fluxo de condução; a velocidade por dois pontos (~3 min de deslocamento do
# centroide) tem σ de ~100 km/h e deixava a 2a previsão do track pior que a
# persistência (STSC: skill -0,49; GLM: -0,16, em 14/11/2026)
CF_PRIOR_RADIUS_KM = 150.0
# vizinhos mínimos (menos que isso: velocidade zero e dois pontos, como antes)
CI_PRIOR_MIN_NEIGHBORS = 2
# células mínimas no histórico de um vizinho para ele contar (já passou pelos
# dois pontos e por pelo menos um ciclo predict -> update)
CI_PRIOR_MIN_HISTORY = 3
# desvio-padrão mínimo da velocidade a priori (km/h, por eixo): vizinhos que
# concordam entre si não tornam o filtro novo confiante demais
CF_PRIOR_MIN_VEL_STD_KMH = 10.0

# [new] medida do Kalman (measurement_from_events).  A janela de agrupamento
# (20 min ou 4 imagens) desliza de 5 em 5 min: 75% dos eventos repetem de um
# frame para o outro, e o centroide da janela inteira dava ao filtro medidas
# com erro correlacionado (ele supõe erro independente): NIS ~8 contra 2
# esperado, e previsão pior que persistência.  Com CI_MEAS_NEW_ONLY, a medida
# é o centroide só dos eventos chegados desde o frame anterior, no instante
# médio deles; a janela inteira continua definindo a célula (detecção,
# associação, tamanho).  0 = centroide da janela (o de antes), para comparar
CI_MEAS_NEW_ONLY = 1
# eventos novos mínimos para haver medida (menos: o track continua, mas o
# Kalman só é atualizado num frame com medida)
CI_MEAS_MIN_EVENTS = 3
# ruído da medida: sqrt(CF_KALMAN_R_STD_KM² + s²/n), com s o desvio-padrão por
# eixo da posição dos n eventos: o centroide de 400 eventos erra menos que o
# de 4.  0 = ruído fixo (CF_KALMAN_R_STD_KM)
CI_MEAS_SCALE_R = 1

# < logging >──────────────────────────────────────────────────────────────────────────

M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# ═════════════════════════════════════════════════════════════════════════════════════
# estatísticas da célula
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def compute_centroid(flst_cluster: list) -> tuple:
    """
    centroide de um cluster: média aritmética das latitudes e das longitudes.

    aceitável para clusters pequenos (raio < 200 km).  Para células que cruzam
    o meridiano 180° ou perto dos polos a média de longitudes dá um centroide
    errado -- não acontece sobre o Brasil.

    :param flst_cluster: lista (não vazia) de TLightning
    :returns: (lat, lng) médios, em graus
    """
    lf_sum_lat = 0.0
    lf_sum_lng = 0.0

    for l_lightning in flst_cluster:
        lf_sum_lat += l_lightning.lat
        lf_sum_lng += l_lightning.lng

    li_count = len(flst_cluster)

    return (lf_sum_lat / li_count, lf_sum_lng / li_count)

# ─────────────────────────────────────────────────────────────────────────────────────
def compute_density(flst_cluster: list, ft_centroid: tuple) -> float:
    """
    densidade de raios: quantidade / área do círculo que contém o cluster.
    o raio do círculo é a maior distância de um raio ao centroide (mínimo 1 km).

    [fix] parâmetro morto haversine_km removido; usa spc.haversine_km diretamente.

    :param flst_cluster: lista de TLightning
    :param ft_centroid: (lat, lng) do centroide
    :returns: raios por km²
    """
    lf_max_radius_km = 0.0

    for l_lightning in flst_cluster:
        lf_distance_km = spc.haversine_km(l_lightning.lat, l_lightning.lng,
                                          ft_centroid[0], ft_centroid[1])
        lf_max_radius_km = max(lf_max_radius_km, lf_distance_km)

    lf_max_radius_km = max(lf_max_radius_km, CF_MIN_DENSITY_RADIUS_KM)

    return len(flst_cluster) / (math.pi * lf_max_radius_km ** 2)

# ─────────────────────────────────────────────────────────────────────────────────────
def compute_flash_rate(flst_cluster: list, ff_delta_min: float) -> float:
    """
    taxa de raios do cluster.

    :param flst_cluster: lista de raios (só o tamanho é usado)
    :param ff_delta_min: duração do cluster, em minutos
    :returns: raios por minuto; 0.0 se a duração não for positiva
    """
    if ff_delta_min <= 0.0:
        return 0.0

    return len(flst_cluster) / ff_delta_min

# ─────────────────────────────────────────────────────────────────────────────────────
def measurement_from_events(flst_events: list) -> dict:
    """
    a medida de posição que o Kalman recebe de uma célula: centroide e
    instante médio dos eventos dados (os novos do cluster) e o ruído dela.

    :param flst_events: TLightning (os eventos do cluster desde o frame anterior)
    :returns: {"n": n} se n < CI_MEAS_MIN_EVENTS (sem medida); senão
              {"n", "centroid": (lat, lng), "t": instante médio, "r_std_km":
              desvio-padrão da medida por eixo, ou None (ruído do filtro)}
    """
    li_count = len(flst_events)

    if li_count < CI_MEAS_MIN_EVENTS:
        return {"n": li_count}

    lt_centroid = compute_centroid(flst_events)
    lf_mean_ts = sum(l_event.ts for l_event in flst_events) / li_count

    lf_r_std_km = None

    if CI_MEAS_SCALE_R:
        # desvio-padrão por eixo: a distância quadrática média ao centroide / 2
        lf_mean_sq_km = sum(spc.haversine_km(l_event.lat, l_event.lng,
                                             lt_centroid[0], lt_centroid[1]) ** 2
                            for l_event in flst_events) / li_count
        lf_r_std_km = math.sqrt(klm.CF_KALMAN_R_STD_KM ** 2 + lf_mean_sq_km / 2.0 / li_count)

    return {"n": li_count, "centroid": lt_centroid, "t": lf_mean_ts, "r_std_km": lf_r_std_km}

# ─────────────────────────────────────────────────────────────────────────────────────
def _measurement(f_cell: sst.TCell) -> tuple | None:
    """
    a medida da célula para o Kalman.

    :param f_cell: célula (meas preenchido por bltTracker; vazio = célula
                   montada sem ele: vale o centroide da janela no instante da
                   célula, como antes)
    :returns: ((lat, lng), instante, r_std_km ou None); None = sem medida neste
              frame (poucos eventos novos)
    """
    if not f_cell.meas:
        return f_cell.centroid, sst.cell_time(f_cell), None

    if "centroid" not in f_cell.meas:
        return None

    return f_cell.meas["centroid"], f_cell.meas["t"], f_cell.meas["r_std_km"]

# ─────────────────────────────────────────────────────────────────────────────────────
def _kalman_time(f_kalman: klm.TKalmanTracker | None, f_last_cell: sst.TCell) -> float:
    """
    o instante a que o estado do Kalman se refere.

    :param f_kalman: filtro (pode não ter o instante: estado antigo)
    :param f_last_cell: última célula do track (o instante dela é o substituto)
    :returns: epoch UTC
    """
    if f_kalman is not None and f_kalman.t is not None:
        return f_kalman.t

    return sst.cell_time(f_last_cell)

# ═════════════════════════════════════════════════════════════════════════════════════
# associação entre frames
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def match_score(f_prev_cell: sst.TCell, f_curr_cell: sst.TCell,
                ff_max_dist_km: float | None = None,
                ff_dist_km: float | None = None) -> float:
    """
    custo de associar uma célula do frame anterior a uma do frame atual:
    quanto menor, mais provável que sejam a mesma tempestade.

        score = W_DIST · min(dist / max_dist, 1) + W_SIZE · (1 − menor_tamanho / maior_tamanho)

    os pesos dizem que posição discrimina mais que tamanho: duas células
    próximas tendem a ser a mesma mesmo que o tamanho varie um pouco (o
    clustering ganha ou perde pontos de borda a cada frame).  Exemplos da parte
    de tamanho: 100 e 80 raios -> penalidade 0.2; 100 e 10 raios -> 0.9.

    [fix] substituiu uma fórmula com escalas incompatíveis e sinal invertido.

    :param f_prev_cell: célula do frame anterior (última do track)
    :param f_curr_cell: célula do frame atual
    :param ff_max_dist_km: distância que vale penalidade máxima (None = CF_MAX_DIST_KM)
    :param ff_dist_km: distância entre os centroides, se já calculada
                       (associate_cells passa, para não repetir o haversine)
    :returns: custo em [0, 1]; 0 = mesma posição e mesmo tamanho
    """
    if ff_max_dist_km is None:
        ff_max_dist_km = CF_MAX_DIST_KM

    if ff_dist_km is None:
        ff_dist_km = spc.haversine_km(f_prev_cell.centroid[0], f_prev_cell.centroid[1],
                                      f_curr_cell.centroid[0], f_curr_cell.centroid[1])

    lf_distance_penalty = min(ff_dist_km / ff_max_dist_km, 1.0)

    lf_size_ratio = _size_ratio(f_prev_cell.size, f_curr_cell.size)

    return CF_W_DIST * lf_distance_penalty + CF_W_SIZE * (1.0 - lf_size_ratio)

# ─────────────────────────────────────────────────────────────────────────────────────
def _size_ratio(fi_size_a: int, fi_size_b: int) -> float:
    """
    semelhança de tamanho entre duas células.

    :param fi_size_a: raios da primeira célula
    :param fi_size_b: raios da segunda célula
    :returns: menor / maior, em [0, 1]; 0.0 se as duas estiverem vazias
    """
    li_largest = max(fi_size_a, fi_size_b)

    if li_largest <= 0:
        return 0.0

    return min(fi_size_a, fi_size_b) / li_largest

# ─────────────────────────────────────────────────────────────────────────────────────
def associate_cells(flst_tracks: list, flst_cells_curr: list,
                    ff_max_dist_km: float | None = None,
                    fdct_kalman_states: dict | None = None,
                    ff_max_score: float | None = None) -> dict:
    """
    decide qual célula do frame atual continua qual track.

    monta uma matriz de custo (linha = track ativo, coluna = célula atual,
    valor = match_score) e usa o algoritmo húngaro
    (scipy.optimize.linear_sum_assignment), que acha a combinação de menor
    custo TOTAL -- não o melhor par de cada track isolado.  Pares além de
    max_dist_km ou com score acima de ff_max_score ficam com o custo de "sem
    associação" ANTES do húngaro (assim ele não gasta um track num par que
    seria recusado, em vez de escolher uma alternativa válida) e são
    descartados depois.

    [fix] o guloso (vizinho mais próximo) foi trocado pelo húngaro: assim um
          track não "rouba" a célula mais próxima de outro.
    [fix] com fdct_kalman_states, a distância é medida até onde o Kalman do
          track prevê a célula no instante dela (_reference_position), não
          até a última posição observada: uma célula rápida ou um frame
          perdido não a empurram para fora do raio de associação.

    :param flst_tracks: todos os tracks (ativos e inativos)
    :param flst_cells_curr: células do frame atual
    :param ff_max_dist_km: distância máxima para associar (None = CF_MAX_DIST_KM)
    :param fdct_kalman_states: {id do track: TKalmanTracker}; None (ou track
                               sem filtro) = usa a última posição observada
    :param ff_max_score: maior match_score aceito (None = CF_MAX_MATCH_SCORE)
    :returns: dict {índice da célula atual: índice do track em flst_tracks},
              só com as associações aceitas
    """
    if ff_max_dist_km is None:
        ff_max_dist_km = CF_MAX_DIST_KM

    if ff_max_score is None:
        ff_max_score = CF_MAX_MATCH_SCORE

    # só tracks ativos e com histórico entram na associação
    llst_active_tracks = [(li_track_index, l_track)
                          for li_track_index, l_track in enumerate(flst_tracks)
                          if l_track.active and l_track.last_cell is not None]

    if not llst_active_tracks or not flst_cells_curr:
        return {}

    lnp_cost = _build_cost_matrix(llst_active_tracks, flst_cells_curr, ff_max_dist_km,
                                  fdct_kalman_states or {}, ff_max_score)

    llst_rows, llst_columns = opt.linear_sum_assignment(lnp_cost)

    return _accept_assignments(lnp_cost, llst_rows, llst_columns, llst_active_tracks)

# ─────────────────────────────────────────────────────────────────────────────────────
def _build_cost_matrix(flst_active_tracks: list, flst_cells_curr: list,
                       ff_max_dist_km: float, fdct_kalman_states: dict,
                       ff_max_score: float) -> np.ndarray:
    """
    matriz de custo tracks x células para o algoritmo húngaro.
    começa toda em CF_NO_MATCH_COST; só os pares a até ff_max_dist_km e com
    match_score <= ff_max_score recebem o score.

    :param flst_active_tracks: lista de (índice original, TStormTrack) ativos
    :param flst_cells_curr: células do frame atual
    :param ff_max_dist_km: distância máxima para associar
    :param fdct_kalman_states: {id do track: TKalmanTracker} (pode ser vazio)
    :param ff_max_score: maior match_score aceito
    :returns: array (n_tracks, n_células) de custos em [0, 1]
    """
    lnp_cost = np.full((len(flst_active_tracks), len(flst_cells_curr)),
                       CF_NO_MATCH_COST, dtype=float)

    for li_row, (_, l_track) in enumerate(flst_active_tracks):
        l_last_cell = l_track.last_cell
        l_kalman = fdct_kalman_states.get(l_track.id)

        for li_column, l_cell in enumerate(flst_cells_curr):
            lf_ref_lat, lf_ref_lng = _reference_position(l_last_cell, l_cell, l_kalman)

            lf_distance_km = spc.haversine_km(lf_ref_lat, lf_ref_lng,
                                              l_cell.centroid[0], l_cell.centroid[1])

            if lf_distance_km > ff_max_dist_km:
                continue

            lf_score = match_score(l_last_cell, l_cell, ff_max_dist_km,
                                   ff_dist_km=lf_distance_km)

            if lf_score > ff_max_score:
                continue

            lnp_cost[li_row, li_column] = lf_score

    return lnp_cost

# ─────────────────────────────────────────────────────────────────────────────────────
def _reference_position(f_last_cell: sst.TCell, f_cell: sst.TCell,
                        f_kalman: klm.TKalmanTracker | None) -> tuple:
    """
    onde o track "deveria" estar no instante da célula candidata: a posição
    prevista pelo Kalman, avançado desde a última célula do track
    (mesmo dt que _continue_track usa no predict).

    :param f_last_cell: última célula do track
    :param f_cell: célula candidata do frame atual
    :param f_kalman: filtro do track; None = sem previsão
    :returns: (lat, lng) de referência; a última posição observada se não
              houver filtro
    """
    if f_kalman is None:
        return f_last_cell.centroid[0], f_last_cell.centroid[1]

    # [fix] desde o instante do estado do filtro (a última medida), não o da
    # última célula: com a medida dos eventos novos os dois diferem
    lf_elapsed_hours = max((sst.cell_time(f_cell) - _kalman_time(f_kalman, f_last_cell)) /
                           mtn.CF_SECONDS_PER_HOUR, 0.0)

    lf_lat, lf_lng, _, _ = f_kalman.predict_position(lf_elapsed_hours)

    return lf_lat, lf_lng

# ─────────────────────────────────────────────────────────────────────────────────────
def _accept_assignments(fnp_cost: np.ndarray, flst_rows: list, flst_columns: list,
                        flst_active_tracks: list) -> dict:
    """
    filtra o resultado do algoritmo húngaro: o húngaro sempre emparelha o
    máximo possível, inclusive pares "sem associação" (custo máximo), que
    aqui são descartados.

    :param fnp_cost: matriz de custo usada
    :param flst_rows: linhas escolhidas (posição na lista de tracks ativos)
    :param flst_columns: colunas escolhidas (índice da célula atual)
    :param flst_active_tracks: lista de (índice original, TStormTrack) ativos
    :returns: dict {índice da célula atual: índice ORIGINAL do track}
    """
    ldct_association = {}

    for li_row, li_column in zip(flst_rows, flst_columns):
        if fnp_cost[li_row, li_column] < CF_NO_MATCH_COST:
            ldct_association[li_column] = flst_active_tracks[li_row][0]

    return ldct_association

# ═════════════════════════════════════════════════════════════════════════════════════
# ciclo de vida dos tracks
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def generate_track_id() -> str:
    """
    novo ID de track: os 8 primeiros caracteres de um UUID4.

    [doc] 8 caracteres hex = 32 bits (~4.3 bilhões de valores).  A chance de
    colisão cresce com o total de tracks já gerados (não só os ativos), e não
    há verificação contra o histórico.  Aceitável no volume operacional (no
    máximo algumas dezenas de milhares de tracks por longos períodos); se o
    volume crescer ordens de grandeza, use mais caracteres ou verifique
    unicidade contra os tracks ativos.

    :returns: ID de 8 caracteres
    """
    return str(uuid.uuid4())[:CI_TRACK_ID_LENGTH]

# ─────────────────────────────────────────────────────────────────────────────────────
def update_tracks(flst_tracks: list, ldct_kalman_states: dict,
                  flst_cells_curr: list, fdct_association: dict) -> tuple:
    """
    aplica as células do frame atual aos tracks.

    passo a passo:
      1. célula associada a um track: continua aquele track
         (_continue_track: movimento, histórico, idade, Kalman);
      2. célula sem associação: abre um track novo (_start_new_track); se
         estiver a até CF_SPLIT_MERGE_DIST_KM da célula atual de um track que
         acabou de continuar, é uma DIVISÃO: o novo track é filho dele
         (parent_id) e herda a velocidade (_split_parent);
      3. track ativo que não recebeu célula neste frame: se a posição prevista
         dele cai a até CF_SPLIT_MERGE_DIST_KM de uma célula que continuou
         outro track, é uma FUSÃO e ele é desativado na hora (_merge_target);
         senão conta um frame perdido e é desativado após
         CI_MAX_MISSED_FRAMES (_count_missed_frame).

    [new] divisão e fusão: o algoritmo húngaro é um-para-um, então numa
    divisão um fragmento abria um track sem parentesco e com velocidade zero,
    e numa fusão o track absorvido ficava 3 frames como fantasma.  As
    continuações (passo 1) vêm todas antes, para o filho copiar o Kalman do
    pai já atualizado neste frame.

    as listas/dicts recebidos são ALTERADOS e também devolvidos.

    :param flst_tracks: tracks existentes
    :param ldct_kalman_states: {id do track: TKalmanTracker}
    :param flst_cells_curr: células do frame atual
    :param fdct_association: {índice da célula: índice do track} (associate_cells)
    :returns: (tracks, kalman_states) atualizados
    """
    # tracks criados neste frame entram no fim da lista e não contam frame perdido
    li_existing_track_count = len(flst_tracks)

    llst_continued_tracks = []

    for li_cell_index, l_track_index in sorted(fdct_association.items()):
        l_track = flst_tracks[l_track_index]
        _continue_track(l_track, flst_cells_curr[li_cell_index], ldct_kalman_states)
        llst_continued_tracks.append(l_track)

    lset_matched_track_indexes = set(fdct_association.values())

    # tracks que podiam ter recebido célula e não receberam (diagnóstico das
    # células sem par: a que distância ficou o candidato mais próximo)
    llst_unmatched_tracks = [l_track for li_track_index, l_track in enumerate(flst_tracks)
                             if l_track.active and l_track.last_cell is not None
                             and li_track_index not in lset_matched_track_indexes]

    for li_cell_index, l_cell in enumerate(flst_cells_curr):
        if li_cell_index in fdct_association:
            continue

        l_parent = _split_parent(l_cell, llst_continued_tracks)
        lt_prior = None if l_parent is not None else _neighbor_velocity_prior(
                       l_cell, llst_continued_tracks, ldct_kalman_states)
        _start_new_track(l_cell, flst_tracks, ldct_kalman_states, l_parent, lt_prior)

        l_cell.assoc = _new_cell_diagnostics(l_cell, l_parent, llst_unmatched_tracks,
                                             ldct_kalman_states)
        l_cell.assoc["prior_n"] = lt_prior[2] if lt_prior is not None else 0

    for li_track_index in range(li_existing_track_count):
        l_track = flst_tracks[li_track_index]

        if not l_track.active or li_track_index in lset_matched_track_indexes:
            continue

        l_target = _merge_target(l_track, llst_continued_tracks, ldct_kalman_states)

        if l_target is not None:
            _merge_track(l_track, l_target)
            continue

        _count_missed_frame(l_track)

    return flst_tracks, ldct_kalman_states

# ─────────────────────────────────────────────────────────────────────────────────────
def _continue_track(f_track: sst.TStormTrack, f_cell: sst.TCell,
                    fdct_kalman_states: dict) -> None:
    """
    acrescenta ao track a célula que o continua.

    passo a passo:
      1. a célula recebe o ID do track e entra no histórico (limitado a
         CI_MAX_HISTORY);
      2. idade do track cresce e o contador de frames perdidos zera;
      3. Kalman: predict até o instante da célula, depois update com o centroide
         -- ou, na segunda célula de um track que não veio de divisão,
         reinício pelos dois pontos (_needs_two_point_init);
      4. velocidade/direção da célula = as do Kalman (mtn.kalman_velocity), a
         mesma estimativa que o cone de previsão usa.

    :param f_track: track que continua
    :param f_cell: célula do frame atual associada a ele
    :param fdct_kalman_states: {id do track: TKalmanTracker}
    :raises KeyError: se o track não tiver estado de Kalman
    """
    l_previous_cell = f_track.last_cell

    f_cell.id = f_track.id
    f_cell.parent_id = f_track.parent_id

    f_track.history.append(f_cell)

    if len(f_track.history) > CI_MAX_HISTORY:
        f_track.history = f_track.history[-CI_MAX_HISTORY:]

    # nunca negativo: a janela deslizante pode reordenar clusters
    lf_elapsed_hours = max(mtn.elapsed_hours(l_previous_cell, f_cell), 0.0)

    f_track.age_minutes += lf_elapsed_hours * mtn.CF_MINUTES_PER_HOUR
    f_track.missed_frames = 0

    l_kalman = fdct_kalman_states[f_cell.id]
    lt_meas = _measurement(f_cell)

    # intervalo desde o estado do filtro até a medida (0 sem medida)
    lf_meas_dt_h = 0.0 if lt_meas is None else max(
        (lt_meas[1] - _kalman_time(l_kalman, l_previous_cell)) / mtn.CF_SECONDS_PER_HOUR, 0.0)

    lb_two_point = lt_meas is not None and _needs_two_point_init(f_track, lf_meas_dt_h)

    # [new] diagnóstico (blt_diag), medido contra o filtro ANTES do update
    f_cell.assoc = _continue_diagnostics(l_previous_cell, f_cell, l_kalman,
                                         lf_elapsed_hours, lb_two_point)
    f_cell.assoc["meas_n"] = f_cell.meas.get("n", f_cell.size)

    if lt_meas is None:
        # [new] poucos eventos novos: a célula continua o track, mas o filtro
        # fica no instante da última medida (o predict do próximo frame
        # cobre o intervalo inteiro)
        pass

    elif lb_two_point:
        lt_previous = l_kalman.predict_position(0.0)[:2]
        l_kalman = klm.TKalmanTracker.from_two_points(lt_previous, lt_meas[0], lf_meas_dt_h,
                                                      lt_meas[1])
        fdct_kalman_states[f_cell.id] = l_kalman

    else:
        # o filtro de Kalman exige o ciclo predict -> update a cada medida
        l_kalman.predict(lf_meas_dt_h)
        # NIS entre o predict e o update: a inovação que o filtro vai corrigir
        f_cell.assoc["nis"] = l_kalman.nis(lt_meas[0][0], lt_meas[0][1], lt_meas[2])
        l_kalman.update(lt_meas[0][0], lt_meas[0][1], lt_meas[2])
        l_kalman.t = lt_meas[1]

    # [fix] uma só velocidade: antes era a EMA dos centroides (seta) contra o
    # Kalman (cone), que podiam divergir
    f_cell.speed_kmh, f_cell.direction_deg = mtn.kalman_velocity(l_kalman, f_cell.centroid[0])

# ─────────────────────────────────────────────────────────────────────────────────────
def _continue_diagnostics(f_previous_cell: sst.TCell, f_cell: sst.TCell,
                          f_kalman: klm.TKalmanTracker, ff_elapsed_hours: float,
                          fb_two_point: bool) -> dict:
    """
    diagnóstico de uma associação aceita: a distância entre a célula e a
    posição prevista (a mesma que associate_cells mediu), o score e os
    tamanhos.  O NIS é acrescentado por _continue_track, entre predict e update.

    [new] base para calibrar CF_MAX_DIST_KM e CF_MAX_MATCH_SCORE: a
    distribuição da distância das associações aceitas (bltReplay --report).

    :param f_previous_cell: última célula do track antes desta
    :param f_cell: célula que continua o track
    :param f_kalman: filtro do track, ainda sem o predict deste frame
    :param ff_elapsed_hours: horas entre as duas células
    :param fb_two_point: True = o filtro vai ser reiniciado por dois pontos
    :returns: dict kind, dt_min, dist_km, score, size_prev
    """
    lf_ref_lat, lf_ref_lng = _reference_position(f_previous_cell, f_cell, f_kalman)
    lf_distance_km = spc.haversine_km(lf_ref_lat, lf_ref_lng,
                                      f_cell.centroid[0], f_cell.centroid[1])

    return {"kind":      "two_point" if fb_two_point else "continue",
            "dt_min":    ff_elapsed_hours * mtn.CF_MINUTES_PER_HOUR,
            "dist_km":   lf_distance_km,
            "score":     match_score(f_previous_cell, f_cell, ff_dist_km=lf_distance_km),
            "size_prev": f_previous_cell.size}

# ─────────────────────────────────────────────────────────────────────────────────────
def _new_cell_diagnostics(f_cell: sst.TCell, f_parent: sst.TStormTrack | None,
                          flst_unmatched_tracks: list, fdct_kalman_states: dict) -> dict:
    """
    diagnóstico de uma célula que abriu track: o track sem par cuja posição
    prevista ficou mais perto dela, a distância e o score que o par teria.

    [new] uma célula nova colada num track que morreu no mesmo frame é
    fragmentação (o mesmo sistema virou dois tracks): se isso for comum
    com score acima de CF_MAX_MATCH_SCORE ou distância acima de
    CF_MAX_DIST_KM, o corte está apertado demais.

    :param f_cell: célula que abriu track
    :param f_parent: track pai, se for divisão
    :param flst_unmatched_tracks: tracks ativos que não receberam célula neste frame
    :param fdct_kalman_states: {id do track: TKalmanTracker}
    :returns: dict kind ("new" ou "split") e, havendo candidato, dist_km,
              score e size_prev do mais próximo
    """
    ldct_diagnostics = {"kind": "split" if f_parent is not None else "new"}

    l_nearest = None
    lf_best_km = math.inf

    for l_track in flst_unmatched_tracks:
        lf_ref_lat, lf_ref_lng = _reference_position(l_track.last_cell, f_cell,
                                                     fdct_kalman_states.get(l_track.id))
        lf_distance_km = spc.haversine_km(lf_ref_lat, lf_ref_lng,
                                          f_cell.centroid[0], f_cell.centroid[1])

        if lf_distance_km < lf_best_km:
            l_nearest, lf_best_km = l_track, lf_distance_km

    if l_nearest is None:
        return ldct_diagnostics

    ldct_diagnostics.update(dist_km=lf_best_km,
                            score=match_score(l_nearest.last_cell, f_cell, ff_dist_km=lf_best_km),
                            size_prev=l_nearest.last_cell.size)

    return ldct_diagnostics

# ─────────────────────────────────────────────────────────────────────────────────────
def _needs_two_point_init(f_track: sst.TStormTrack, ff_elapsed_hours: float) -> bool:
    """
    a célula que acabou de entrar é a segunda do track, e dá para iniciar a
    velocidade do Kalman pelo deslocamento entre as duas?

    não vale para track filho de divisão (já herdou a velocidade do pai), nem
    para track que nasceu com a velocidade dos vizinhos (velocity_prior), nem
    com intervalo menor que CF_MIN_TWO_POINT_DT_MIN (o deslocamento seria
    quase só ruído); nesses casos o filtro segue o ciclo predict -> update.

    :param f_track: track, já com a célula nova no histórico
    :param ff_elapsed_hours: horas entre a célula anterior e a nova
    :returns: True se deve reiniciar com klm.TKalmanTracker.from_two_points
    """
    if 2 != len(f_track.history) or f_track.parent_id or f_track.velocity_prior:
        return False

    return ff_elapsed_hours * mtn.CF_MINUTES_PER_HOUR >= CF_MIN_TWO_POINT_DT_MIN

# ─────────────────────────────────────────────────────────────────────────────────────
def _start_new_track(f_cell: sst.TCell, flst_tracks: list, fdct_kalman_states: dict,
                     f_parent: sst.TStormTrack | None = None,
                     ft_prior: tuple | None = None) -> None:
    """
    abre um track novo a partir de uma célula sem associação: ID novo e a
    célula como primeiro elemento do histórico.  O Kalman nasce na posição
    dela: se for filho de divisão (f_parent), com a velocidade e as
    covariâncias do pai (TKalmanTracker.spawn_at); se houver vizinhos
    (ft_prior), com a velocidade deles (TKalmanTracker.from_velocity_prior);
    senão, com velocidade zero.

    :param f_cell: célula do frame atual (recebe o ID novo e o parent_id)
    :param flst_tracks: lista de tracks (o novo é acrescentado no fim)
    :param fdct_kalman_states: {id do track: TKalmanTracker} (recebe o novo)
    :param f_parent: track de onde a célula se dividiu; None = nasceu sozinha
    :param ft_prior: (velocidade, desvio-padrão, vizinhos) de
                     _neighbor_velocity_prior; None = sem vizinhos
    """
    f_cell.id = generate_track_id()
    f_cell.parent_id = f_parent.id if f_parent is not None else ""

    flst_tracks.append(sst.TStormTrack(id=f_cell.id, history=[f_cell],
                                       parent_id=f_cell.parent_id,
                                       velocity_prior=ft_prior is not None and f_parent is None))

    # [new] o filtro nasce na medida (eventos novos) e no instante dela; sem
    # medida (poucos eventos novos), no centroide da janela
    lt_meas = _measurement(f_cell) or (f_cell.centroid, sst.cell_time(f_cell), None)
    (lf_lat, lf_lng), lf_t = lt_meas[0], lt_meas[1]

    if f_parent is None and ft_prior is not None:
        fdct_kalman_states[f_cell.id] = klm.TKalmanTracker.from_velocity_prior(
            lf_lat, lf_lng, ft_prior[0], ft_prior[1], lf_t)
        return

    if f_parent is None:
        fdct_kalman_states[f_cell.id] = klm.TKalmanTracker(lf_lat, lf_lng, ff_t=lf_t)
        return

    l_child = fdct_kalman_states[f_parent.id].spawn_at(lf_lat, lf_lng)
    l_child.t = lf_t
    fdct_kalman_states[f_cell.id] = l_child

    M_LOG.info("divisão: track %s nasceu de %s.", f_cell.id, f_parent.id)

# ─────────────────────────────────────────────────────────────────────────────────────
def _neighbor_velocity_prior(f_cell: sst.TCell, flst_continued_tracks: list,
                             fdct_kalman_states: dict) -> tuple | None:
    """
    velocidade a priori de um track novo: a média da velocidade (Kalman) dos
    tracks maduros que continuaram neste frame, a até CF_PRIOR_RADIUS_KM da
    célula, com peso gaussiano pela distância (σ = metade do raio).

    a incerteza é o espalhamento ponderado das velocidades dos vizinhos, por
    eixo, com piso CF_PRIOR_MIN_VEL_STD_KMH.

    :param f_cell: célula que abre o track
    :param flst_continued_tracks: tracks que receberam célula neste frame
                                  (Kalman já atualizado)
    :param fdct_kalman_states: {id do track: TKalmanTracker}
    :returns: ((norte, leste) km/h, (σ norte, σ leste) km/h, nº de vizinhos);
              None com menos de CI_PRIOR_MIN_NEIGHBORS vizinhos
    """
    lf_lat, lf_lng = f_cell.centroid
    lf_sigma_km = CF_PRIOR_RADIUS_KM / 2.0

    llst_weights = []
    llst_velocities = []

    for l_track in flst_continued_tracks:
        l_kalman = fdct_kalman_states.get(l_track.id)

        if l_kalman is None or len(l_track.history) < CI_PRIOR_MIN_HISTORY:
            continue

        lt_position = l_track.last_cell.centroid
        lf_distance_km = spc.haversine_km(lf_lat, lf_lng, lt_position[0], lt_position[1])

        if lf_distance_km > CF_PRIOR_RADIUS_KM:
            continue

        llst_weights.append(math.exp(-0.5 * (lf_distance_km / lf_sigma_km) ** 2))
        llst_velocities.append(l_kalman.velocity_kmh(lt_position[0]))

    if len(llst_velocities) < CI_PRIOR_MIN_NEIGHBORS:
        return None

    lf_weight_sum = sum(llst_weights)
    lt_mean = tuple(sum(lf_w * lt_v[li] for lf_w, lt_v in zip(llst_weights, llst_velocities)) /
                    lf_weight_sum for li in (0, 1))
    lt_std = tuple(max(CF_PRIOR_MIN_VEL_STD_KMH,
                       math.sqrt(sum(lf_w * (lt_v[li] - lt_mean[li]) ** 2
                                     for lf_w, lt_v in zip(llst_weights, llst_velocities)) /
                                 lf_weight_sum))
                   for li in (0, 1))

    return lt_mean, lt_std, len(llst_velocities)

# ─────────────────────────────────────────────────────────────────────────────────────
def _split_parent(f_cell: sst.TCell, flst_continued_tracks: list,
                  ff_max_dist_km: float | None = None) -> sst.TStormTrack | None:
    """
    de que track esta célula sem par se dividiu: o que continuou neste frame
    com a célula atual mais próxima dela, a até ff_max_dist_km.

    :param f_cell: célula do frame atual sem associação
    :param flst_continued_tracks: tracks que receberam célula neste frame
    :param ff_max_dist_km: distância máxima entre as duas células
                           (None = CF_SPLIT_MERGE_DIST_KM)
    :returns: o track pai, ou None se não houver nenhum perto
    """
    l_parent = None
    lf_best_km = CF_SPLIT_MERGE_DIST_KM if ff_max_dist_km is None else ff_max_dist_km

    for l_track in flst_continued_tracks:
        lt_fragment = l_track.last_cell.centroid
        lf_distance_km = spc.haversine_km(lt_fragment[0], lt_fragment[1],
                                          f_cell.centroid[0], f_cell.centroid[1])

        if lf_distance_km <= lf_best_km:
            l_parent, lf_best_km = l_track, lf_distance_km

    return l_parent

# ─────────────────────────────────────────────────────────────────────────────────────
def _merge_target(f_track: sst.TStormTrack, flst_continued_tracks: list,
                  fdct_kalman_states: dict,
                  ff_max_dist_km: float | None = None) -> sst.TStormTrack | None:
    """
    em que track este track sem par se fundiu: o que continuou neste frame
    com a célula mais próxima da posição prevista dele, a até ff_max_dist_km.

    :param f_track: track ativo que não recebeu célula
    :param flst_continued_tracks: tracks que receberam célula neste frame
    :param fdct_kalman_states: {id do track: TKalmanTracker}
    :param ff_max_dist_km: distância máxima entre a previsão e a célula
                           (None = CF_SPLIT_MERGE_DIST_KM)
    :returns: o track que o absorveu, ou None se não houver nenhum perto
    """
    l_target = None
    lf_best_km = CF_SPLIT_MERGE_DIST_KM if ff_max_dist_km is None else ff_max_dist_km
    l_kalman = fdct_kalman_states.get(f_track.id)

    for l_other in flst_continued_tracks:
        l_cell = l_other.last_cell
        lf_ref_lat, lf_ref_lng = _reference_position(f_track.last_cell, l_cell, l_kalman)
        lf_distance_km = spc.haversine_km(lf_ref_lat, lf_ref_lng,
                                          l_cell.centroid[0], l_cell.centroid[1])

        if lf_distance_km <= lf_best_km:
            l_target, lf_best_km = l_other, lf_distance_km

    return l_target

# ─────────────────────────────────────────────────────────────────────────────────────
def _merge_track(f_track: sst.TStormTrack, f_target: sst.TStormTrack) -> None:
    """
    desativa um track que se fundiu em outro (não fica como fantasma
    contando frames perdidos).

    :param f_track: track absorvido (ALTERADO)
    :param f_target: track que o absorveu
    """
    f_track.active = False
    f_track.end_reason = "merged"

    M_LOG.info("fusão: track %s absorvido por %s.", f_track.id, f_target.id)

# ─────────────────────────────────────────────────────────────────────────────────────
def _count_missed_frame(f_track: sst.TStormTrack) -> None:
    """
    registra que um track ativo não recebeu célula neste frame.

    um track sem célula pode ser uma tempestade que dissipou, saiu da área ou
    só uma falha passageira de detecção.  Por isso não é descartado na hora:
    só é desativado após CI_MAX_MISSED_FRAMES frames seguidos.

    :param f_track: track ativo sem associação
    """
    f_track.missed_frames += 1

    if f_track.missed_frames < CI_MAX_MISSED_FRAMES:
        return

    f_track.active = False
    f_track.end_reason = "missed"
    M_LOG.info("track %s desativado após %d frames sem match.",
               f_track.id, CI_MAX_MISSED_FRAMES)

# ─────────────────────────────────────────────────────────────────────────────────────
def expire_stale_tracks(flst_tracks: list, ff_now_ts: float,
                        ff_max_gap_min: float | None = None) -> int:
    """
    desativa os tracks cujo último raio é mais antigo que ff_max_gap_min.

    [fix] _count_missed_frame só conta frames que tiveram célula: com o
    tracker parado (REDEMET fora, reinício) ou rodadas sem célula, um track
    velho continuava ativo e, na volta, herdava qualquer célula a menos de
    CF_MAX_DIST_KM, com um dt de horas no Kalman e na velocidade.  Idempotente
    (só depende de ff_now_ts), então pode rodar a cada frame antes da
    associação.

    :param flst_tracks: tracks lidos do estado (ALTERADOS)
    :param ff_now_ts: instante do frame atual (epoch UTC)
    :param ff_max_gap_min: minutos sem raio que expiram um track
                           (None = CF_MAX_TRACK_GAP_MIN)
    :returns: quantos tracks foram desativados
    """
    if ff_max_gap_min is None:
        ff_max_gap_min = CF_MAX_TRACK_GAP_MIN

    lf_oldest_ts = ff_now_ts - ff_max_gap_min * 60.0
    li_expired = 0

    for l_track in flst_tracks:
        l_last_cell = l_track.last_cell

        if not l_track.active or l_last_cell is None or l_last_cell.t_end >= lf_oldest_ts:
            continue

        l_track.active = False
        l_track.end_reason = "expired"
        li_expired += 1

        M_LOG.info("track %s expirado: sem raio há %.0f min.",
                   l_track.id, (ff_now_ts - l_last_cell.t_end) / 60.0)

    return li_expired

# ═════════════════════════════════════════════════════════════════════════════════════
# persistência
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def load_tracks(fs_path: str) -> tuple:
    """
    lê os tracks e os estados de Kalman gravados pelo frame anterior.

    [fix] arquivo corrompido ou truncado (disco cheio, kill no meio da
    escrita, edição externa) não derruba o tracker: é tratado como "primeiro
    frame" (os IDs de track recomeçam).

    :param fs_path: arquivo de estado (slm_tracks_state.json)
    :returns: (lista de TStormTrack, {id: TKalmanTracker}); ([], {}) se o
              arquivo não existir ou estiver corrompido
    """
    if not os.path.isfile(fs_path):
        return [], {}

    try:
        with open(fs_path) as l_file:
            ldct_state = json.load(l_file)

        llst_tracks = [_track_from_dict(ldct_track)
                       for ldct_track in ldct_state.get("tracks", [])]

        ldct_kalman_states = {ls_track_id: klm.TKalmanTracker.from_dict(ldct_kalman)
                              for ls_track_id, ldct_kalman in ldct_state.get("kalman", {}).items()}

    except (json.JSONDecodeError, KeyError, TypeError) as l_error:
        M_LOG.warning("estado de tracks corrompido em %s — reiniciando tracks do zero (%s).",
                      fs_path, l_error)
        return [], {}

    _ensure_kalman_for_every_track(llst_tracks, ldct_kalman_states)

    M_LOG.info("tracks carregados de %s (%d ativos).", fs_path, _count_active(llst_tracks))

    return llst_tracks, ldct_kalman_states

# ─────────────────────────────────────────────────────────────────────────────────────
def _ensure_kalman_for_every_track(flst_tracks: list, fdct_kalman_states: dict) -> None:
    """
    garante um filtro de Kalman para cada track com histórico.

    [fix] um estado com o track mas sem a entrada "kalman" dele (arquivo
    editado à mão, versão antiga) passava pelo load e só quebrava depois, com
    KeyError em _continue_track.  O filtro que falta é recriado na última
    posição conhecida (velocidade zero, como num track novo).

    :param flst_tracks: tracks lidos do arquivo
    :param fdct_kalman_states: {id do track: TKalmanTracker} (COMPLETADO aqui)
    """
    for l_track in flst_tracks:
        if l_track.id in fdct_kalman_states:
            continue

        l_last_cell = l_track.last_cell

        if l_last_cell is None:
            continue

        M_LOG.warning("track %s sem estado de Kalman; filtro recriado na última posição.",
                      l_track.id)

        fdct_kalman_states[l_track.id] = klm.TKalmanTracker(l_last_cell.centroid[0],
                                                            l_last_cell.centroid[1])

# ─────────────────────────────────────────────────────────────────────────────────────
def _track_from_dict(fdct_track: dict) -> sst.TStormTrack:
    """
    reconstrói um track do dict gravado por _track_to_dict.

    :param fdct_track: dict com id, age_minutes, missed_frames, active, history
                       e (opcional) forecasts
    :returns: novo TStormTrack
    :raises KeyError: se faltar alguma chave
    """
    return sst.TStormTrack(id=fdct_track["id"],
                           age_minutes=fdct_track["age_minutes"],
                           missed_frames=fdct_track["missed_frames"],
                           active=fdct_track["active"],
                           history=[sst.cell_from_dict(ldct_cell)
                                    for ldct_cell in fdct_track["history"]],
                           # [fix] estado gravado antes de 2026.nov não tem as chaves
                           forecasts=fdct_track.get("forecasts", []),
                           parent_id=fdct_track.get("parent_id", ""),
                           velocity_prior=fdct_track.get("velocity_prior", False))

# ─────────────────────────────────────────────────────────────────────────────────────
def save_tracks(fs_path: str, flst_tracks: list, ldct_kalman_states: dict,
                ff_newest_ts: float = 0.0) -> None:
    """
    grava (atomicamente) os tracks ativos e seus estados de Kalman, para o
    próximo frame.

    tracks inativos nunca voltam a ser associados (associate_cells), então
    não são gravados -- o arquivo não cresce sem limite.

    :param fs_path: arquivo de estado (slm_tracks_state.json)
    :param flst_tracks: todos os tracks
    :param ldct_kalman_states: {id do track: TKalmanTracker}
    :param ff_newest_ts: ts do raio mais recente do frame aplicado a este
                         estado (ver load_applied_ts)
    """
    llst_active_tracks = [l_track for l_track in flst_tracks if l_track.active]
    lset_active_ids = {l_track.id for l_track in llst_active_tracks}

    ldct_state = {
        "newest_ts": ff_newest_ts,
        "tracks": [_track_to_dict(l_track) for l_track in llst_active_tracks],
        "kalman": {ls_track_id: l_kalman.to_dict()
                   for ls_track_id, l_kalman in ldct_kalman_states.items()
                   if ls_track_id in lset_active_ids},
    }

    iou.atomic_write_json(fs_path, ldct_state)

    M_LOG.info("tracks persistidos em %s (%d ativos).", fs_path, _count_active(flst_tracks))

# ─────────────────────────────────────────────────────────────────────────────────────
def load_applied_ts(fs_path: str) -> float:
    """
    ts do raio mais recente do último frame já aplicado ao estado dos tracks.

    [fix] o shell só grava .slm_last_ts DEPOIS que o tracker termina: se ele
    morresse entre o save_tracks e o safe_write, a rodada seguinte recebia o
    since_ts antigo e reaplicava o frame (histórico duplicado, Kalman
    atualizado duas vezes com a mesma medida).  Com este valor no próprio
    estado, o tracker sabe que o frame já foi aplicado (bltTracker.process).

    :param fs_path: arquivo de estado (slm_tracks_state.json)
    :returns: o ts gravado; 0.0 se o arquivo não existir, estiver corrompido
              ou for de antes de 2026.nov (sem a chave)
    """
    try:
        with open(fs_path) as l_file:
            return float(json.load(l_file).get("newest_ts", 0.0))

    except (OSError, ValueError, TypeError, AttributeError):
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────────────
def _track_to_dict(f_track: sst.TStormTrack) -> dict:
    """
    converte um track num dict serializável em JSON.

    :param f_track: track a converter
    :returns: dict com id, age_minutes, missed_frames, active, history e forecasts
    """
    return {
        "id":            f_track.id,
        "age_minutes":   f_track.age_minutes,
        "missed_frames": f_track.missed_frames,
        "active":        f_track.active,
        "history":       [sst.cell_to_dict(l_cell) for l_cell in f_track.history],
        "forecasts":     f_track.forecasts,
        "parent_id":     f_track.parent_id,
        "velocity_prior": f_track.velocity_prior,
    }

# ─────────────────────────────────────────────────────────────────────────────────────
def _count_active(flst_tracks: list) -> int:
    """
    :param flst_tracks: lista de TStormTrack
    :returns: quantos estão ativos
    """
    return sum(1 for l_track in flst_tracks if l_track.active)

# < the end >──────────────────────────────────────────────────────────────────────────
