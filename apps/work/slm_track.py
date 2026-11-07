# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
slm_track
acompanhamento (tracking) das células convectivas de um frame para o outro.

  - estatísticas da célula: centroide, densidade, taxa de raios;
  - associação: qual célula do frame atual continua qual track
    (associate_cells, algoritmo húngaro = melhor combinação global);
  - ciclo de vida dos tracks (update_tracks): histórico, velocidade suavizada
    (EMA), filtro de Kalman, frames perdidos e desativação;
  - persistência entre execuções (load_tracks / save_tracks) em
    slm_tracks_state.json -- apagar esse arquivo zera todos os IDs de track.

fluxo de um frame (chamado por slmTracker.process):
  load_tracks -> associate_cells -> update_tracks -> save_tracks

2026.out  mlabru  refatoração para legibilidade: update_tracks, associate_cells,
                  load_tracks e save_tracks divididos em funções de um passo
                  só; horas entre células por slm_motion.elapsed_hours; logs
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
import work.slm_ioutil as iou
import work.slm_kalman as klm
import work.slm_motion as mtn
import work.slm_spatial as spc
import work.slm_struct as sst

# < constants >────────────────────────────────────────────────────────────────────────

# máximo de células mantidas no histórico de cada track
CI_MAX_HISTORY = 12

# frames seguidos sem associação antes de desativar o track
CI_MAX_MISSED_FRAMES = 3

# distância (km) acima da qual duas células nunca são associadas
CF_MAX_DIST_KM = 100.0

# pesos do match_score (somam 1, então o score fica em [0, 1])
CF_W_DIST = 0.6   # penalidade de distância
CF_W_SIZE = 0.4   # penalidade de diferença de tamanho

# custo "sem associação" na matriz do algoritmo húngaro (o máximo do match_score)
CF_NO_MATCH_COST = 1.0

# raio mínimo (km) no cálculo de densidade, para não dividir por zero
CF_MIN_DENSITY_RADIUS_KM = 1.0

# comprimento do ID de track (caracteres do UUID4)
CI_TRACK_ID_LENGTH = 8

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

# ═════════════════════════════════════════════════════════════════════════════════════
# associação entre frames
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def match_score(f_prev_cell: sst.TCell, f_curr_cell: sst.TCell,
                ff_max_dist_km: float = CF_MAX_DIST_KM,
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
    :param ff_max_dist_km: distância que vale penalidade máxima
    :param ff_dist_km: distância entre os centroides, se já calculada
                       (associate_cells passa, para não repetir o haversine)
    :returns: custo em [0, 1]; 0 = mesma posição e mesmo tamanho
    """
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
                    ff_max_dist_km: float = CF_MAX_DIST_KM) -> dict:
    """
    decide qual célula do frame atual continua qual track.

    monta uma matriz de custo (linha = track ativo, coluna = célula atual,
    valor = match_score) e usa o algoritmo húngaro
    (scipy.optimize.linear_sum_assignment), que acha a combinação de menor
    custo TOTAL -- não o melhor par de cada track isolado.  Pares com custo
    máximo (célula além de max_dist_km) são descartados.

    [fix] o guloso (vizinho mais próximo) foi trocado pelo húngaro: assim um
          track não "rouba" a célula mais próxima de outro.

    :param flst_tracks: todos os tracks (ativos e inativos)
    :param flst_cells_curr: células do frame atual
    :param ff_max_dist_km: distância máxima para associar
    :returns: dict {índice da célula atual: índice do track em flst_tracks},
              só com as associações aceitas
    """
    # só tracks ativos e com histórico entram na associação
    llst_active_tracks = [(li_track_index, l_track)
                          for li_track_index, l_track in enumerate(flst_tracks)
                          if l_track.active and l_track.last_cell is not None]

    if not llst_active_tracks or not flst_cells_curr:
        return {}

    lnp_cost = _build_cost_matrix(llst_active_tracks, flst_cells_curr, ff_max_dist_km)

    llst_rows, llst_columns = opt.linear_sum_assignment(lnp_cost)

    return _accept_assignments(lnp_cost, llst_rows, llst_columns, llst_active_tracks)

# ─────────────────────────────────────────────────────────────────────────────────────
def _build_cost_matrix(flst_active_tracks: list, flst_cells_curr: list,
                       ff_max_dist_km: float) -> np.ndarray:
    """
    matriz de custo tracks x células para o algoritmo húngaro.
    começa toda em CF_NO_MATCH_COST; só os pares a até ff_max_dist_km recebem
    o match_score.

    :param flst_active_tracks: lista de (índice original, TStormTrack) ativos
    :param flst_cells_curr: células do frame atual
    :param ff_max_dist_km: distância máxima para associar
    :returns: array (n_tracks, n_células) de custos em [0, 1]
    """
    lnp_cost = np.full((len(flst_active_tracks), len(flst_cells_curr)),
                       CF_NO_MATCH_COST, dtype=float)

    for li_row, (_, l_track) in enumerate(flst_active_tracks):
        l_last_cell = l_track.last_cell

        for li_column, l_cell in enumerate(flst_cells_curr):
            lf_distance_km = spc.haversine_km(l_last_cell.centroid[0], l_last_cell.centroid[1],
                                              l_cell.centroid[0], l_cell.centroid[1])

            if lf_distance_km > ff_max_dist_km:
                continue

            lnp_cost[li_row, li_column] = match_score(l_last_cell, l_cell, ff_max_dist_km,
                                                      ff_dist_km=lf_distance_km)

    return lnp_cost

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
      2. célula sem associação: abre um track novo (_start_new_track);
      3. track ativo que não recebeu célula neste frame: conta um frame
         perdido e é desativado após CI_MAX_MISSED_FRAMES (_count_missed_frame).

    as listas/dicts recebidos são ALTERADOS e também devolvidos.

    :param flst_tracks: tracks existentes
    :param ldct_kalman_states: {id do track: TKalmanTracker}
    :param flst_cells_curr: células do frame atual
    :param fdct_association: {índice da célula: índice do track} (associate_cells)
    :returns: (tracks, kalman_states) atualizados
    """
    # tracks criados neste frame entram no fim da lista e não contam frame perdido
    li_existing_track_count = len(flst_tracks)

    for li_cell_index, l_cell in enumerate(flst_cells_curr):
        if li_cell_index in fdct_association:
            l_track = flst_tracks[fdct_association[li_cell_index]]
            _continue_track(l_track, l_cell, ldct_kalman_states)

        else:
            _start_new_track(l_cell, flst_tracks, ldct_kalman_states)

    lset_matched_track_indexes = set(fdct_association.values())

    for li_track_index in range(li_existing_track_count):
        l_track = flst_tracks[li_track_index]

        if not l_track.active or li_track_index in lset_matched_track_indexes:
            continue

        _count_missed_frame(l_track)

    return flst_tracks, ldct_kalman_states

# ─────────────────────────────────────────────────────────────────────────────────────
def _continue_track(f_track: sst.TStormTrack, f_cell: sst.TCell,
                    fdct_kalman_states: dict) -> None:
    """
    acrescenta ao track a célula que o continua.

    passo a passo:
      1. movimento instantâneo (velocidade/direção) em relação à última célula;
      2. a célula recebe o ID do track e entra no histórico (limitado a
         CI_MAX_HISTORY);
      3. com 2+ células, o movimento é trocado pelo suavizado (EMA);
      4. idade do track cresce e o contador de frames perdidos zera;
      5. Kalman: predict até o instante da célula, depois update com o centroide.

    :param f_track: track que continua
    :param f_cell: célula do frame atual associada a ele
    :param fdct_kalman_states: {id do track: TKalmanTracker}
    :raises KeyError: se o track não tiver estado de Kalman
    """
    l_previous_cell = f_track.last_cell

    f_cell.speed_kmh = mtn.compute_speed_kmh(l_previous_cell, f_cell)
    f_cell.direction_deg = mtn.compute_direction(l_previous_cell, f_cell)
    f_cell.id = f_track.id

    f_track.history.append(f_cell)

    if len(f_track.history) > CI_MAX_HISTORY:
        f_track.history = f_track.history[-CI_MAX_HISTORY:]

    if len(f_track.history) >= 2:
        f_cell.speed_kmh, f_cell.direction_deg = mtn.compute_velocity_ema(f_track.history)

    # nunca negativo: a janela deslizante pode reordenar clusters
    lf_elapsed_hours = max(mtn.elapsed_hours(l_previous_cell, f_cell), 0.0)

    f_track.age_minutes += lf_elapsed_hours * mtn.CF_MINUTES_PER_HOUR
    f_track.missed_frames = 0

    # o filtro de Kalman exige o ciclo predict -> update a cada frame
    l_kalman = fdct_kalman_states[f_cell.id]
    l_kalman.predict(lf_elapsed_hours)
    l_kalman.update(f_cell.centroid[0], f_cell.centroid[1])

# ─────────────────────────────────────────────────────────────────────────────────────
def _start_new_track(f_cell: sst.TCell, flst_tracks: list, fdct_kalman_states: dict) -> None:
    """
    abre um track novo a partir de uma célula sem associação: ID novo, a
    célula como primeiro elemento do histórico e um Kalman na posição dela
    com velocidade zero.

    :param f_cell: célula do frame atual (recebe o ID novo)
    :param flst_tracks: lista de tracks (o novo é acrescentado no fim)
    :param fdct_kalman_states: {id do track: TKalmanTracker} (recebe o novo)
    """
    f_cell.id = generate_track_id()

    flst_tracks.append(sst.TStormTrack(id=f_cell.id, history=[f_cell]))

    fdct_kalman_states[f_cell.id] = klm.TKalmanTracker(f_cell.centroid[0], f_cell.centroid[1])

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
    M_LOG.info("track %s desativado após %d frames sem match.",
               f_track.id, CI_MAX_MISSED_FRAMES)

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

    :param fdct_track: dict com id, age_minutes, missed_frames, active e history
    :returns: novo TStormTrack
    :raises KeyError: se faltar alguma chave
    """
    return sst.TStormTrack(id=fdct_track["id"],
                           age_minutes=fdct_track["age_minutes"],
                           missed_frames=fdct_track["missed_frames"],
                           active=fdct_track["active"],
                           history=[sst.cell_from_dict(ldct_cell)
                                    for ldct_cell in fdct_track["history"]])

# ─────────────────────────────────────────────────────────────────────────────────────
def save_tracks(fs_path: str, flst_tracks: list, ldct_kalman_states: dict) -> None:
    """
    grava (atomicamente) os tracks ativos e seus estados de Kalman, para o
    próximo frame.

    tracks inativos nunca voltam a ser associados (associate_cells), então
    não são gravados -- o arquivo não cresce sem limite.

    :param fs_path: arquivo de estado (slm_tracks_state.json)
    :param flst_tracks: todos os tracks
    :param ldct_kalman_states: {id do track: TKalmanTracker}
    """
    llst_active_tracks = [l_track for l_track in flst_tracks if l_track.active]
    lset_active_ids = {l_track.id for l_track in llst_active_tracks}

    ldct_state = {
        "tracks": [_track_to_dict(l_track) for l_track in llst_active_tracks],
        "kalman": {ls_track_id: l_kalman.to_dict()
                   for ls_track_id, l_kalman in ldct_kalman_states.items()
                   if ls_track_id in lset_active_ids},
    }

    iou.atomic_write_json(fs_path, ldct_state)

    M_LOG.info("tracks persistidos em %s (%d ativos).", fs_path, _count_active(flst_tracks))

# ─────────────────────────────────────────────────────────────────────────────────────
def _track_to_dict(f_track: sst.TStormTrack) -> dict:
    """
    converte um track num dict serializável em JSON.

    :param f_track: track a converter
    :returns: dict com id, age_minutes, missed_frames, active e history
    """
    return {
        "id":            f_track.id,
        "age_minutes":   f_track.age_minutes,
        "missed_frames": f_track.missed_frames,
        "active":        f_track.active,
        "history":       [sst.cell_to_dict(l_cell) for l_cell in f_track.history],
    }

# ─────────────────────────────────────────────────────────────────────────────────────
def _count_active(flst_tracks: list) -> int:
    """
    :param flst_tracks: lista de TStormTrack
    :returns: quantos estão ativos
    """
    return sum(1 for l_track in flst_tracks if l_track.active)

# < the end >──────────────────────────────────────────────────────────────────────────
