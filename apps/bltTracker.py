# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
bltTracker
gera UM frame de nowcasting a partir do CSV acumulado de eventos: agrupa
os eventos em células, associa as células aos tracks do frame anterior,
prevê o movimento e grava o JSON do frame para o bltRender.

[doc] o que é um "evento": com a fonte atual (REDEMET STSC), cada linha do
CSV é uma célula de grade de 0,1° (~10 km) com atividade convectiva (CB) numa
imagem da animação STSC, e NÃO uma descarga individual.  As imagens chegam a
cada 6-9 min, então o CSV tem poucos ts distintos.  Os nomes do código
(TLightning, "raio", flash_rate) vêm da fonte de descargas que o modelo
pressupõe e continuam valendo para uma rede de descargas (ex.: BrasilDAT);
com o STSC, size é área ativa x imagens e flash_rate é "células por minuto".

o CSV de entrada é acumulativo (o downloader mantém a última hora):
    12:00 -> 12:05
    12:00 -> 12:10
    12:00 -> 12:15
--since-ts só decide se há frame novo (algum evento com ts > since_ts); o
agrupamento usa sempre a janela mais recente do CSV inteiro: as últimas
CI_WINDOW_IMAGES imagens (fonte em imagens, o STSC) ou, com
CI_WINDOW_IMAGES = 0, os últimos CI_WINDOW_MINUTES (fonte contínua de
descargas).  Ver _select_time_window.

fluxo (process):
  1) lê o CSV e verifica se há eventos novos (load_csv, filter_new_events);
  2) agrupa em células (cluster_window: janela + DBSCAN + estatísticas);
  3) tracking (blt_track: load_tracks, expire_stale_tracks, associate_cells,
     update_tracks);
  4) verificação das previsões anteriores (blt_verify.verify_cells) e das
     que morreram com o track (blt_verify.close_tracks);
  5) previsão e trilha de cada célula (_attach_forecast_and_history) e registro
     das previsões novas para verificação futura (blt_verify.register_cells);
  6) grava o JSON do frame, o estado dos tracks, as linhas de verificação e
     os diagnósticos (blt_diag: associação e resumo do frame).

contrato com o shell (blt_common.sh): se houve frame novo, imprime no stdout
uma linha "NEWEST_TS=<ts do evento mais recente>" (ver main).

    python3 apps/bltTracker.py raw.csv frame.json --since-ts 1787010000 \
            --tracks-file data/state/slm_tracks_state.json \
            --verify-file data/state/slm_verification.csv \
            --assoc-file data/state/slm_assoc.csv --frames-file data/state/slm_frames.csv

2026.nov  mlabru  medida do Kalman só com os eventos novos de cada célula
                  (cluster_window(..., ff_meas_since_ts), TCell.meas); a
                  previsão conta do instante do filtro (_forecast_offset_h)
2026.nov  mlabru  taxa e salto de raios por célula (glm_ljump) com fonte glm;
                  campos flash_rate_now, flash_rate_recent, flash_rate_dfrdt, jump_sigma e
                  lightning_jump no JSON do frame
2026.nov  mlabru  --source (stsc/glm): a fonte decide a janela (imagens no
                  STSC, minutos no GLM, window_images_for)
2026.nov  mlabru  [fix] janela por número de imagens (CI_WINDOW_IMAGES): com
                  imagens a cada 6 ou 9 min, a janela de 20 min tinha ora 3,
                  ora 4 imagens, e size/flash_rate/t_ref oscilavam ~33%
2026.nov  mlabru  instrumentação para calibrar: diagnóstico da associação
                  (--assoc-file), resumo por frame com latência
                  (--frames-file), previsões perdidas na verificação;
                  process aceita o instante da rodada (ff_run_ts, bltReplay);
                  janela lida na chamada (o bltReplay sobrescreve as constantes)
2026.nov  mlabru  [fix] horizontes da previsão contados do instante do frame
2026.nov  mlabru  [fix] t_ref da célula (média dos ts) e parent_id no JSON do frame
2026.nov  mlabru  frame já aplicado ao estado (newest_ts) não é reaplicado
2026.nov  mlabru  verificação das previsões (blt_verify, --verify-file)
2026.nov  mlabru  clustering na janela fixa do CSV inteiro (não só nos eventos
                  novos): janela de tamanho constante e raios atrasados do
                  último minuto não se perdem; tracks expiram por tempo antes
                  da associação; associação contra a posição prevista
2026.out  mlabru  refatoração para legibilidade: process, cluster_window e
                  load_csv divididos em funções de um passo só; main devolve
                  rc nomeado; logs de debug removidos, logging com %
2026.may  mlabru  import blt_struct as sst (era 'str', sombra do builtin);
                  cluster_window com janela deslizante (evita agregação de eventos
                  temporalmente distantes); t_start/t_end calculados por cluster,
                  não globalmente; compute_cell_stats preenche amp/polarity/type
                  ratios; tracking loop integrado em process() com estado
                  persistente; argumento --tracks-file adicionado ao CLI
2025.nov  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import argparse
import logging
import sys
import time

# locals
import work.blt_diag as dgn
import work.blt_ioutil as iou
import work.glm_ljump as ljp
import work.blt_motion as mtn
import work.blt_spatial as spc
import work.blt_struct as sst      # [fix] era 'str', sobrescrevia o builtin str()
import work.blt_track as trk
import work.blt_verify as vrf

# < constants >────────────────────────────────────────────────────────────────────────

# DBSCAN: distância máxima (km) entre eventos vizinhos.
# [doc] com o STSC (grade de 0,1°: 11,1 km em latitude, 11,1·cos(lat) km em
# longitude) o eps é discreto: 12 km pega só os 4 vizinhos diretos (a diagonal
# tem 14,7-15,7 km) e o mesmo ponto nas outras imagens da janela.  Valores
# que mudam algo: < 11,1 (a grade se desfaz em latitude), ~16 (8 vizinhos),
# ~22,3 (2 passos).  Entre esses, o resultado é o mesmo
CF_EPS_KM = 12.0

# DBSCAN: mínimo de eventos (vizinhança incluindo o próprio) para ser ponto
# central.  [doc] com o STSC e eps de 12 km, a vizinhança tem no máximo 5
# células de grade x CI_WINDOW_IMAGES imagens (20); com 10, uma célula precisa
# de pelo menos 2 imagens para nascer (uma imagem dá no máximo 5)
CI_MIN_PTS = 10

# janela de agrupamento, em IMAGENS: os eventos dos últimos CI_WINDOW_IMAGES
# ts distintos.  [fix] o STSC chega em imagens a cada 6-9 min; uma janela em
# minutos pegava ora 3, ora 4 imagens (aliasing), e o tamanho da célula, que
# pesa na associação, oscilava ~33%.  4 = o que a janela de 20 min continha
# com imagens a cada 6 min.  0 = janela em minutos (CI_WINDOW_MINUTES), para
# uma fonte contínua de descargas (ex.: BrasilDAT)
CI_WINDOW_IMAGES = 4
# janela em minutos, contada do evento mais recente (só com CI_WINDOW_IMAGES = 0)
CI_WINDOW_MINUTES = 20
# no modo em imagens, a janela nunca vai além disto (minutos): uma falha de
# imagens da REDEMET não traz para a janela imagens velhas demais.  4 imagens
# a 6-9 min ocupam 18-27 min
CF_WINDOW_MAX_SPAN_MIN = 30.0

# fontes de eventos (--source): "stsc" = imagens da REDEMET (janela em
# imagens); "glm" = flashes do GOES-19 (fonte contínua, janela em minutos)
CS_SOURCE_STSC = "stsc"
CS_SOURCE_GLM = "glm"
CLST_SOURCES = [CS_SOURCE_STSC, CS_SOURCE_GLM]

# duração mínima (minutos) de uma célula no cálculo da taxa de raios
CF_MIN_CELL_DURATION_MIN = 1.0

# arquivo de estado padrão (o blt_common.sh pode informar outro)
CS_DEFAULT_TRACKS_FILE = "./data/state/slm_tracks_state.json"

# CSV de verificação padrão (o blt_common.sh pode informar outro)
CS_DEFAULT_VERIFY_FILE = "./data/state/slm_verification.csv"

# CSVs de diagnóstico padrão (o blt_common.sh pode informar outros)
CS_DEFAULT_ASSOC_FILE = "./data/state/slm_assoc.csv"
CS_DEFAULT_FRAMES_FILE = "./data/state/slm_frames.csv"

# códigos de retorno
CI_RC_OK = 0

# < logging >──────────────────────────────────────────────────────────────────────────

M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# ═════════════════════════════════════════════════════════════════════════════════════
# leitura
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def load_csv(fs_path: str) -> list:
    """
    lê o CSV de raios.  A primeira linha (cabeçalho) é pulada; linhas em
    branco são ignoradas; linhas malformadas são ignoradas com aviso no log
    (em vez de abortar).

    colunas: timestamp, lat, lng e, opcionais, amp, polarity, kind.

    :param fs_path: caminho do CSV
    :returns: lista de TLightning, na ordem do arquivo (vazia se o arquivo for vazio)
    """
    llst_events = []

    with open(fs_path) as l_file:
        # [fix] arquivo vazio (nem cabeçalho): next() sem padrão levantava StopIteration
        ls_header = next(l_file, None)

        if ls_header is None:
            return llst_events

        for ls_line in l_file:
            ls_line = ls_line.strip()

            if not ls_line:
                continue

            try:
                llst_events.append(_parse_csv_line(ls_line))

            except (ValueError, IndexError) as l_error:
                M_LOG.warning("linha ignorada: %r — %s", ls_line, l_error)

    return llst_events

# ─────────────────────────────────────────────────────────────────────────────────────
def _parse_csv_line(fs_line: str) -> sst.TLightning:
    """
    converte uma linha do CSV num raio.

    :param fs_line: linha sem espaços nas pontas, ex.: "1787010240,-33.75,-50.65"
    :returns: TLightning (amp/polarity/kind padrão se as colunas não existirem)
    :raises ValueError: número inválido
    :raises IndexError: menos de 3 colunas
    """
    llst_columns = fs_line.split(",")

    lf_timestamp = float(llst_columns[0])
    lf_lat = float(llst_columns[1])
    lf_lng = float(llst_columns[2])

    lf_amplitude = 0.0
    li_polarity = 0
    ls_kind = ""

    if len(llst_columns) > 3:
        lf_amplitude = float(llst_columns[3])

    if len(llst_columns) > 4:
        li_polarity = int(llst_columns[4])

    if len(llst_columns) > 5:
        ls_kind = llst_columns[5].strip()

    return sst.TLightning(ts=lf_timestamp, lat=lf_lat, lng=lf_lng,
                          amp=lf_amplitude, polarity=li_polarity, kind=ls_kind)

# ─────────────────────────────────────────────────────────────────────────────────────
def filter_new_events(flst_events: list, ff_since_ts: float) -> list:
    """
    só os eventos posteriores a um instante.

    :param flst_events: lista de TLightning
    :param ff_since_ts: instante de corte (epoch UTC); eventos com ts <= ele saem
    :returns: lista com os eventos de ts > ff_since_ts
    """
    return [l_event for l_event in flst_events if l_event.ts > ff_since_ts]

# ═════════════════════════════════════════════════════════════════════════════════════
# células
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def cluster_window(flst_events: list,
                   ff_window_minutes: float | None = None,
                   fi_window_images: int | None = None,
                   ff_jump_since_ts: float | None = None,
                   ff_meas_since_ts: float | None = None) -> list:
    """
    agrupa em células os eventos mais recentes.

    passo a passo:
      1. separa os eventos da janela mais recente (_select_time_window);
      2. agrupa com DBSCAN (eps = CF_EPS_KM, min_pts = CI_MIN_PTS);
      3. resume cada cluster numa TCell (_build_cell);
      4. com ff_jump_since_ts (fonte de descargas), taxa e salto de raios de
         cada célula (_attach_lightning_jump);
      5. com ff_meas_since_ts, a medida do Kalman: só os eventos de cada
         cluster posteriores ao frame anterior (_attach_measurement).

    [fix] só a janela mais recente é usada, refletindo o estado convectivo atual.
    [fix] t_start e t_end por cluster (não globais): flash_rate e density por
          célula corretos.

    :param flst_events: lista de TLightning (o CSV inteiro; a janela é recortada aqui)
    :param ff_window_minutes: janela em minutos, no modo contínuo (None = CI_WINDOW_MINUTES)
    :param fi_window_images: janela em imagens; 0 = modo contínuo, em minutos
                             (None = CI_WINDOW_IMAGES)
    :param ff_jump_since_ts: instante do frame anterior; None = sem taxa/salto
                             de raios (fonte em imagens, o STSC)
    :param ff_meas_since_ts: instante do frame anterior; None = a medida do
                             Kalman é o centroide da janela (TCell.meas vazio)
    :returns: lista de TCell (ainda sem ID de track)
    """
    if not flst_events:
        return []

    llst_window_events = _select_time_window(flst_events, ff_window_minutes, fi_window_images)

    llst_clusters = spc.dbscan_spatial(llst_window_events,
                                       ff_eps_km=CF_EPS_KM,
                                       fi_min_pts=CI_MIN_PTS)

    llst_cells = [_build_cell(llst_cluster) for llst_cluster in llst_clusters]

    if ff_jump_since_ts is not None:
        _attach_lightning_jump(llst_cells, llst_clusters, llst_window_events,
                               ff_window_minutes, ff_jump_since_ts)

    if ff_meas_since_ts is not None:
        _attach_measurement(llst_cells, llst_clusters, ff_meas_since_ts)

    return llst_cells

# ─────────────────────────────────────────────────────────────────────────────────────
def _attach_measurement(flst_cells: list, flst_clusters: list, ff_since_ts: float) -> None:
    """
    a medida de posição de cada célula para o Kalman (trk.measurement_from_events):
    os eventos do cluster posteriores ao frame anterior ou, com
    trk.CI_MEAS_NEW_ONLY = 0, o cluster inteiro.

    :param flst_cells: células (ALTERADAS), na ordem de flst_clusters
    :param flst_clusters: os clusters de que vieram
    :param ff_since_ts: instante do frame anterior
    """
    for l_cell, llst_cluster in zip(flst_cells, flst_clusters):
        llst_events = (llst_cluster if not trk.CI_MEAS_NEW_ONLY else
                       [l_event for l_event in llst_cluster if l_event.ts > ff_since_ts])

        l_cell.meas = trk.measurement_from_events(llst_events)

# ─────────────────────────────────────────────────────────────────────────────────────
def _attach_lightning_jump(flst_cells: list, flst_clusters: list, flst_window_events: list,
                           ff_window_minutes: float | None, ff_since_ts: float) -> None:
    """
    taxa e salto de raios de cada célula (glm_ljump.evaluate), numa série que
    termina no evento mais recente da janela (o instante do frame).

    :param flst_cells: células (ALTERADAS), na ordem de flst_clusters
    :param flst_clusters: os clusters de que vieram
    :param flst_window_events: eventos da janela (não vazia)
    :param ff_window_minutes: comprimento da série (None = CI_WINDOW_MINUTES)
    :param ff_since_ts: instante do frame anterior (intervalos avaliados)
    """
    lf_end_ts = max(l_event.ts for l_event in flst_window_events)
    lf_window_min = CI_WINDOW_MINUTES if ff_window_minutes is None else ff_window_minutes

    for l_cell, llst_cluster in zip(flst_cells, flst_clusters):
        l_jump = ljp.evaluate([l_event.ts for l_event in llst_cluster], lf_end_ts,
                              lf_window_min, ff_since_ts)

        l_cell.flash_rate_now = l_jump.rate_now
        l_cell.flash_rate_recent = l_jump.rate_recent
        l_cell.flash_rate_dfrdt = l_jump.dfrdt_now
        l_cell.jump_sigma = l_jump.sigma_level
        l_cell.lightning_jump = l_jump.jump

# ─────────────────────────────────────────────────────────────────────────────────────
def _select_time_window(flst_events: list, ff_window_minutes: float | None = None,
                        fi_window_images: int | None = None) -> list:
    """
    os eventos da janela mais recente, que termina no evento mais novo.

    [doc] por design, eventos mais antigos que a janela são EXCLUÍDOS, mesmo
    que nunca tenham entrado num frame: se o tracker ficar parado por mais
    que a janela (reinício, instabilidade da REDEMET), o atraso nunca forma
    célula.  É uma ressincronização proposital com o estado convectivo
    atual.  Os tracks parados demais expiram por tempo
    (trk.expire_stale_tracks), então não herdam células da volta.

    :param flst_events: lista (não vazia) de TLightning
    :param ff_window_minutes: janela em minutos, no modo contínuo (None = CI_WINDOW_MINUTES)
    :param fi_window_images: janela em imagens; 0 = modo contínuo
                             (None = CI_WINDOW_IMAGES)
    :returns: eventos dentro da janela; todos, se a janela ficar vazia
    """
    if fi_window_images is None:
        fi_window_images = CI_WINDOW_IMAGES

    if fi_window_images > 0:
        lf_window_start_ts = _image_window_start_ts(flst_events, fi_window_images)

    else:
        if ff_window_minutes is None:
            ff_window_minutes = CI_WINDOW_MINUTES

        lf_window_start_ts = max(l_event.ts for l_event in flst_events) - ff_window_minutes * 60.0

    llst_window_events = [l_event for l_event in flst_events
                          if l_event.ts >= lf_window_start_ts]

    # proteção para dados muito esparsos (na prática o evento mais recente
    # sempre está na janela)
    if not llst_window_events:
        M_LOG.warning("janela temporal vazia — usando todos os eventos disponíveis.")
        return flst_events

    return llst_window_events

# ─────────────────────────────────────────────────────────────────────────────────────
def _image_window_start_ts(flst_events: list, fi_window_images: int) -> float:
    """
    início da janela em imagens: o ts da fi_window_images-ésima imagem mais
    recente (cada ts distinto é uma imagem), limitado a CF_WINDOW_MAX_SPAN_MIN
    antes da mais nova.

    :param flst_events: lista (não vazia) de TLightning
    :param fi_window_images: quantas imagens entram na janela (> 0)
    :returns: ts do início da janela (inclusive)
    """
    llst_image_ts = sorted({l_event.ts for l_event in flst_events})

    lf_nth_image_ts = llst_image_ts[-min(fi_window_images, len(llst_image_ts))]
    lf_oldest_allowed_ts = llst_image_ts[-1] - CF_WINDOW_MAX_SPAN_MIN * 60.0

    return max(lf_nth_image_ts, lf_oldest_allowed_ts)

# ─────────────────────────────────────────────────────────────────────────────────────
def _build_cell(flst_cluster: list) -> sst.TCell:
    """
    resume um cluster de raios numa célula: centroide, tamanho, instantes,
    taxa de raios, densidade e estatísticas de amplitude/polaridade/tipo.

    :param flst_cluster: lista (não vazia) de TLightning
    :returns: nova TCell
    """
    lt_centroid = trk.compute_centroid(flst_cluster)

    lf_start_ts = min(l_event.ts for l_event in flst_cluster)
    lf_end_ts = max(l_event.ts for l_event in flst_cluster)

    # cluster de poucos segundos contaria como "1 minuto" (evita taxa absurda)
    lf_duration_min = max((lf_end_ts - lf_start_ts) / 60.0, CF_MIN_CELL_DURATION_MIN)

    l_cell = sst.TCell(centroid=lt_centroid,
                       size=len(flst_cluster),
                       flash_rate=trk.compute_flash_rate(flst_cluster, lf_duration_min),
                       density=trk.compute_density(flst_cluster, lt_centroid),
                       t_start=lf_start_ts,
                       t_end=lf_end_ts,
                       t_ref=sum(l_event.ts for l_event in flst_cluster) / len(flst_cluster))

    # [new] amplitude, polaridade e tipo por cluster
    compute_cell_stats(l_cell, flst_cluster)

    return l_cell

# ─────────────────────────────────────────────────────────────────────────────────────
def compute_cell_stats(f_cell: sst.TCell, flst_cluster_points: list) -> None:
    """
    preenche na célula as estatísticas de amplitude, polaridade e tipo
    (CG = nuvem-solo, IC = intranuvem) dos raios do cluster.

    [new] max_amp, mean_amp, positive_ratio, negative_ratio, cg_ratio e
          ic_ratio eram sempre zero.
    atenção: o downloader de hoje só grava timestamp/lat/lng, então na
    prática esses campos continuam zero até a fonte trazer essas colunas.

    :param f_cell: célula a preencher (ALTERADA)
    :param flst_cluster_points: lista de TLightning do cluster; vazia = nada muda
    """
    li_count = len(flst_cluster_points)

    if 0 == li_count:
        return

    llst_amplitudes = [l_point.amp for l_point in flst_cluster_points]

    f_cell.max_amp = max(llst_amplitudes)
    f_cell.mean_amp = sum(llst_amplitudes) / li_count

    li_positive = sum(1 for l_point in flst_cluster_points if l_point.polarity > 0)
    li_negative = sum(1 for l_point in flst_cluster_points if l_point.polarity < 0)

    f_cell.positive_ratio = li_positive / li_count
    f_cell.negative_ratio = li_negative / li_count

    li_cloud_to_ground = sum(1 for l_point in flst_cluster_points if "CG" == l_point.kind.upper())
    li_intra_cloud = sum(1 for l_point in flst_cluster_points if "IC" == l_point.kind.upper())

    f_cell.cg_ratio = li_cloud_to_ground / li_count
    f_cell.ic_ratio = li_intra_cloud / li_count

# ═════════════════════════════════════════════════════════════════════════════════════
# saída
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def export_frame(flst_cells: list, fs_output_json: str) -> None:
    """
    grava (atomicamente) o JSON do frame: uma lista com um dict por célula,
    com TODOS os campos que o bltRender usa (inclusive previsão e trilha).

    não confundir com sst.cell_to_dict, que grava só o subconjunto usado no
    histórico do estado dos tracks.

    :param flst_cells: células do frame (com forecast_path e history preenchidos)
    :param fs_output_json: caminho do JSON de saída
    """
    llst_cells_json = [_cell_to_frame_dict(l_cell) for l_cell in flst_cells]

    iou.atomic_write_json(fs_output_json, llst_cells_json)

# ─────────────────────────────────────────────────────────────────────────────────────
def _cell_to_frame_dict(f_cell: sst.TCell) -> dict:
    """
    uma célula no formato do JSON do frame (as chaves são o contrato com o bltRender).

    :param f_cell: célula
    :returns: dict serializável
    """
    return {
        "id":             f_cell.id,
        # [new] track de onde este se dividiu ("" = nenhum)
        "parent_id":      f_cell.parent_id,
        "centroid":       f_cell.centroid,
        "size":           f_cell.size,
        "density":        f_cell.density,
        "flash_rate":     f_cell.flash_rate,
        # [new] taxa e salto de raios (glm_ljump; zeros com o STSC)
        "flash_rate_now":   f_cell.flash_rate_now,
        "flash_rate_recent": f_cell.flash_rate_recent,
        "flash_rate_dfrdt": f_cell.flash_rate_dfrdt,
        "jump_sigma":       f_cell.jump_sigma,
        "lightning_jump":   f_cell.lightning_jump,
        "speed_kmh":      f_cell.speed_kmh,
        "direction_deg":  f_cell.direction_deg,
        "forecast_path":  f_cell.forecast_path,
        "history":        f_cell.history,
        "t_start":        f_cell.t_start,
        "t_end":          f_cell.t_end,
        "t_ref":          f_cell.t_ref,
        # [new] estatísticas de amplitude/polaridade/tipo
        "max_amp":        f_cell.max_amp,
        "mean_amp":       f_cell.mean_amp,
        "positive_ratio": f_cell.positive_ratio,
        "negative_ratio": f_cell.negative_ratio,
        "cg_ratio":       f_cell.cg_ratio,
        "ic_ratio":       f_cell.ic_ratio,
    }

# ═════════════════════════════════════════════════════════════════════════════════════
# pipeline de um frame
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def process(fs_input_csv: str, fs_output_json: str,
            ff_since_ts: float, fs_tracks_file: str,
            fs_verify_file: str | None = None,
            fs_assoc_file: str | None = None,
            fs_frames_file: str | None = None,
            ff_run_ts: float | None = None,
            fs_source: str | None = None) -> float | None:
    """
    gera um frame completo (ver o fluxo no cabeçalho do módulo).

    :param fs_input_csv: CSV acumulado de raios
    :param fs_output_json: JSON do frame a gravar
    :param ff_since_ts: só eventos com ts maior que este entram (epoch UTC)
    :param fs_tracks_file: arquivo de estado dos tracks entre execuções
    :param fs_verify_file: CSV de verificação das previsões; None = não grava
    :param fs_assoc_file: CSV de diagnóstico da associação; None = não grava
    :param fs_frames_file: CSV de resumo por frame; None = não grava
    :param ff_run_ts: instante da rodada, para a latência (None = agora;
                      o bltReplay passa o instante simulado)
    :param fs_source: fonte dos eventos, decide a janela (window_images_for;
                      None = CS_SOURCE_STSC)
    :returns: ts do evento novo mais recente; None se não houve frame
              (sem eventos, sem eventos novos ou sem células); o ts do
              último frame aplicado, se o shell estiver atrasado (_catch_up_ts)
    """
    llst_all_events = _load_events(fs_input_csv)

    # [fix] o estado dos tracks sabe até onde já foi aplicado; se o shell
    # ficou para trás (morreu antes de gravar .slm_last_ts), o corte é o do
    # estado, e o frame já aplicado não é reaplicado
    lf_applied_ts = trk.load_applied_ts(fs_tracks_file)

    if not _has_new_events(llst_all_events, max(ff_since_ts, lf_applied_ts)):
        return _catch_up_ts(ff_since_ts, lf_applied_ts)

    # [fix] o clustering usa a janela fixa (CI_WINDOW_IMAGES imagens) do CSV
    # INTEIRO, não só os eventos novos: (a) a janela tem sempre o mesmo
    # tamanho (antes eram ~5 min numa rodada normal e crescia quando não
    # havia célula), então size/flash_rate/density são comparáveis entre
    # frames; (b) eventos que chegam atrasados com ts == since_ts (uma imagem
    # que a REDEMET completa depois, ou o último minuto de uma fonte de
    # descargas) entram no frame seguinte em vez de se perderem.  since_ts só
    # decide SE há frame.
    li_window_images = window_images_for(fs_source)
    # [new] taxa e salto de raios só com fonte contínua (descargas): no STSC a
    # "taxa" mede área.  Avalia os intervalos desde o último frame aplicado
    lf_jump_since_ts = None if li_window_images > 0 else max(ff_since_ts, lf_applied_ts)
    # [new] a medida do Kalman usa só os eventos desde o último frame aplicado
    llst_cells = cluster_window(llst_all_events, fi_window_images=li_window_images,
                                ff_jump_since_ts=lf_jump_since_ts,
                                ff_meas_since_ts=max(ff_since_ts, lf_applied_ts))

    if any(l_cell.lightning_jump for l_cell in llst_cells):
        M_LOG.info("salto de raios em %d células.",
                   sum(1 for l_cell in llst_cells if l_cell.lightning_jump))

    if not llst_cells:
        M_LOG.warning("sem células detectadas.")
        return None

    lf_newest_ts = llst_all_events[-1].ts
    lf_run_ts = time.time() if ff_run_ts is None else ff_run_ts

    M_LOG.info("latência do frame: %.0f s (rodada − raio mais recente).",
               lf_run_ts - lf_newest_ts)

    llst_tracks, ldct_kalman_states = _track_cells(llst_cells, fs_tracks_file, lf_newest_ts)

    # verifica as previsões antigas contra as células de agora e só DEPOIS
    # registra as novas (senão a previsão seria testada contra a própria célula);
    # as dos tracks que acabaram neste frame viram linhas "perdidas"
    llst_verify_rows = (vrf.verify_cells(llst_cells, llst_tracks) +
                        vrf.close_tracks(llst_tracks, lf_newest_ts))

    _attach_forecast_and_history(llst_cells, llst_tracks, ldct_kalman_states, lf_newest_ts)

    vrf.register_cells(llst_cells, llst_tracks, lf_newest_ts)

    # [fix] o frame é gravado ANTES do estado dos tracks: export_frame() pode
    # ser repetido sem efeito colateral, save_tracks() não (repetir reaplica
    # associação/Kalman e duplica o histórico).  Se o processo morrer entre os
    # dois, since_ts ainda não avançou: no máximo o frame é refeito, nunca o
    # estado dos tracks fica duplicado.
    export_frame(llst_cells, fs_output_json)

    trk.save_tracks(fs_tracks_file, llst_tracks, ldct_kalman_states, lf_newest_ts)

    # [doc] depois do estado: se o processo morrer entre os dois, as linhas
    # se perdem (as previsões já saíram do estado), mas nunca se duplicam
    if fs_verify_file:
        vrf.append_rows(fs_verify_file, llst_verify_rows)

    _append_diagnostics(fs_assoc_file, fs_frames_file, lf_newest_ts, lf_run_ts,
                        _count_window_events(llst_all_events, li_window_images), llst_cells, llst_tracks)

    M_LOG.info("frame exportado para %s (%d células).", fs_output_json, len(llst_cells))

    return lf_newest_ts

# ─────────────────────────────────────────────────────────────────────────────────────
def _catch_up_ts(ff_since_ts: float, ff_applied_ts: float) -> float | None:
    """
    o que devolver quando não há evento novo depois do último frame aplicado.

    se o estado está à frente do since_ts do shell, a rodada anterior gravou
    o frame JSON e o estado mas morreu antes de o shell avançar
    .slm_last_ts/.slm_frame_counter: devolver o ts aplicado faz o shell
    publicar aquele frame (o JSON já existe) e alcançar o estado, sem
    reaplicar nada.

    :param ff_since_ts: since_ts recebido do shell
    :param ff_applied_ts: ts do último frame aplicado ao estado (trk.load_applied_ts)
    :returns: ff_applied_ts se o shell está atrasado; None (sem frame) caso contrário
    """
    if ff_applied_ts <= ff_since_ts:
        return None

    M_LOG.warning("frame de ts %s já aplicado ao estado (since_ts %s): só atualiza o shell.",
                  ff_applied_ts, ff_since_ts)

    return ff_applied_ts

# ─────────────────────────────────────────────────────────────────────────────────────
def _load_events(fs_input_csv: str) -> list:
    """
    lê o CSV e devolve todos os eventos, ordenados por instante.

    :param fs_input_csv: CSV acumulado de raios
    :returns: lista de TLightning em ordem de ts; vazia se o CSV não tiver eventos
    """
    llst_all_events = load_csv(fs_input_csv)

    if not llst_all_events:
        M_LOG.warning("sem eventos no CSV.")
        return []

    M_LOG.info("%d eventos carregados.", len(llst_all_events))

    llst_all_events.sort(key=lambda l_event: l_event.ts)

    return llst_all_events

# ─────────────────────────────────────────────────────────────────────────────────────
def _has_new_events(flst_events: list, ff_since_ts: float) -> bool:
    """
    há eventos posteriores ao último frame?

    :param flst_events: lista de TLightning
    :param ff_since_ts: instante de corte (epoch UTC)
    :returns: True se algum evento tiver ts > ff_since_ts (a quantidade vai para o log)
    """
    li_new_count = len(filter_new_events(flst_events, ff_since_ts))

    if 0 == li_new_count:
        if flst_events:
            M_LOG.warning("sem novos eventos.")

        return False

    M_LOG.info("%d novos eventos encontrados.", li_new_count)

    return True

# ─────────────────────────────────────────────────────────────────────────────────────
def _track_cells(flst_cells: list, fs_tracks_file: str, ff_now_ts: float) -> tuple:
    """
    tracking de um frame: lê o estado anterior, desativa os tracks parados há
    tempo demais, associa as células aos tracks (algoritmo húngaro, contra a
    posição prevista pelo Kalman) e atualiza os tracks.  As células recebem o
    ID do track.

    :param flst_cells: células do frame atual (ALTERADAS)
    :param fs_tracks_file: arquivo de estado dos tracks
    :param ff_now_ts: instante do frame (ts do raio mais recente, epoch UTC)
    :returns: (lista de TStormTrack, {id do track: TKalmanTracker}) atualizados
    """
    llst_tracks, ldct_kalman_states = trk.load_tracks(fs_tracks_file)

    trk.expire_stale_tracks(llst_tracks, ff_now_ts)

    ldct_association = trk.associate_cells(llst_tracks, flst_cells,
                                           fdct_kalman_states=ldct_kalman_states)

    return trk.update_tracks(llst_tracks, ldct_kalman_states, flst_cells, ldct_association)

# ─────────────────────────────────────────────────────────────────────────────────────
def _attach_forecast_and_history(flst_cells: list, flst_tracks: list,
                                 fdct_kalman_states: dict, ff_frame_ts: float) -> None:
    """
    completa cada célula para o render: previsão pelo Kalman do track e a
    trilha (histórico do track como dicts serializáveis).

    cluster_window nunca preenche TCell.history: o histórico real vive no track.

    [fix] os horizontes da previsão contam do instante do frame (o raio mais
    recente), não do instante médio da célula: mtn.extrapolate_path recebe a
    diferença entre os dois.

    :param flst_cells: células do frame atual, já com ID de track (ALTERADAS)
    :param flst_tracks: tracks atualizados
    :param fdct_kalman_states: {id do track: TKalmanTracker}
    :param ff_frame_ts: instante do frame (ts do raio mais recente, epoch UTC)
    """
    ldct_tracks_by_id = {l_track.id: l_track for l_track in flst_tracks}

    for l_cell in flst_cells:
        if l_cell.id in fdct_kalman_states:
            l_kalman = fdct_kalman_states[l_cell.id]
            l_cell.forecast_path = mtn.extrapolate_path(
                l_kalman, ff_offset_h=_forecast_offset_h(l_kalman, l_cell, ff_frame_ts))

        l_track = ldct_tracks_by_id.get(l_cell.id)

        if l_track is not None:
            l_cell.history = [sst.cell_to_dict(l_history_cell)
                              for l_history_cell in l_track.history]

# ─────────────────────────────────────────────────────────────────────────────────────
def _forecast_offset_h(f_kalman, f_cell: sst.TCell, ff_frame_ts: float) -> float:
    """
    quanto o instante do frame está à frente do estado do filtro (horas): os
    horizontes da previsão contam do frame.

    [fix] o estado está no instante da última medida (TKalmanTracker.t), não
    necessariamente no da célula: com a medida dos eventos novos, ele fica
    perto do frame; sem medida neste frame, mais atrás.

    :param f_kalman: filtro do track
    :param f_cell: célula (instante substituto se o filtro não tiver o seu)
    :param ff_frame_ts: instante do frame (epoch UTC)
    :returns: horas (>= 0 na prática)
    """
    lf_state_ts = f_kalman.t if f_kalman.t is not None else sst.cell_time(f_cell)

    return (ff_frame_ts - lf_state_ts) / mtn.CF_SECONDS_PER_HOUR

# ─────────────────────────────────────────────────────────────────────────────────────
def _count_window_events(flst_events: list, fi_window_images: int | None = None) -> int:
    """
    quantos eventos entraram na janela de agrupamento (diagnóstico: a fração
    que virou célula mede o quanto CF_EPS_KM/CI_MIN_PTS descartam como ruído).

    :param flst_events: todos os eventos, em ordem de ts (não vazia)
    :param fi_window_images: janela em imagens, a mesma de cluster_window
                             (None = CI_WINDOW_IMAGES)
    :returns: eventos na janela
    """
    return len(_select_time_window(flst_events, fi_window_images=fi_window_images))

# ─────────────────────────────────────────────────────────────────────────────────────
def window_images_for(fs_source: str | None) -> int:
    """
    a janela de agrupamento que a fonte pede: em imagens no STSC
    (CI_WINDOW_IMAGES, lida na chamada), em minutos (0) no GLM, que é contínuo.

    :param fs_source: CS_SOURCE_STSC, CS_SOURCE_GLM ou None (= STSC)
    :returns: valor de fi_window_images para cluster_window
    :raises ValueError: fonte desconhecida
    """
    if fs_source is None or CS_SOURCE_STSC == fs_source:
        return CI_WINDOW_IMAGES

    if CS_SOURCE_GLM == fs_source:
        return 0

    raise ValueError("fonte desconhecida: %r (use uma de %s)" % (fs_source, CLST_SOURCES))

# ─────────────────────────────────────────────────────────────────────────────────────
def _append_diagnostics(fs_assoc_file: str | None, fs_frames_file: str | None,
                        ff_frame_ts: float, ff_run_ts: float, fi_window_events: int,
                        flst_cells: list, flst_tracks: list) -> None:
    """
    grava os diagnósticos do frame (blt_diag).  Depois do estado, como a
    verificação: se o processo morrer antes, as linhas se perdem, mas nunca
    se duplicam.

    :param fs_assoc_file: CSV de associação; None/vazio = não grava
    :param fs_frames_file: CSV de frames; None/vazio = não grava
    :param ff_frame_ts: instante do frame
    :param ff_run_ts: instante da rodada
    :param fi_window_events: raios na janela
    :param flst_cells: células do frame (com assoc)
    :param flst_tracks: tracks depois de update_tracks
    """
    if fs_assoc_file:
        dgn.append(fs_assoc_file, dgn.CLST_ASSOC_COLUMNS,
                   dgn.association_rows(ff_frame_ts, flst_cells))

    if fs_frames_file:
        dgn.append(fs_frames_file, dgn.CLST_FRAME_COLUMNS,
                   [dgn.frame_row(ff_frame_ts, ff_run_ts, fi_window_events,
                                  flst_cells, flst_tracks)])

# ═════════════════════════════════════════════════════════════════════════════════════
# programa
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def _parse_args(flst_argv: list | None) -> argparse.Namespace:
    """
    lê a linha de comando.

    :param flst_argv: argumentos (None = sys.argv[1:])
    :returns: namespace com input_csv, output_json, since_ts (float >= 0),
              tracks_file, verify_file, assoc_file e frames_file
    """
    l_parser = argparse.ArgumentParser(
                   description="Simple Lightning Model — nowcasting incremental")

    l_parser.add_argument("input_csv",
                          help="CSV acumulativo de eventos (STSC: células de grade CB por imagem)")
    l_parser.add_argument("output_json",
                          help="JSON de saída do frame atual")
    l_parser.add_argument("--since-ts",
                          help="ignora eventos com ts ≤ este valor (default: 0)")
    l_parser.add_argument("--tracks-file",
                          help="arquivo de estado persistente de tracks entre frames",
                          default=CS_DEFAULT_TRACKS_FILE)
    l_parser.add_argument("--verify-file",
                          help="CSV de verificação das previsões (acrescentado a cada frame)",
                          default=CS_DEFAULT_VERIFY_FILE)
    l_parser.add_argument("--assoc-file",
                          help="CSV de diagnóstico da associação (uma linha por célula)",
                          default=CS_DEFAULT_ASSOC_FILE)
    l_parser.add_argument("--frames-file",
                          help="CSV de resumo por frame (latência, ciclo de vida dos tracks)",
                          default=CS_DEFAULT_FRAMES_FILE)
    l_parser.add_argument("--source",
                          help="fonte dos eventos: stsc (imagens, padrão) ou glm (contínua)",
                          choices=CLST_SOURCES, default=CS_SOURCE_STSC)

    l_args = l_parser.parse_args(flst_argv)

    # vazio ou ausente vale 0 (o shell passa "" na primeira rodada)
    if not l_args.since_ts:
        l_args.since_ts = 0.0

    l_args.since_ts = abs(float(l_args.since_ts))

    return l_args

# ─────────────────────────────────────────────────────────────────────────────────────
def main(flst_argv: list | None = None) -> int:
    """
    gera um frame e informa ao shell o instante do evento mais recente.

    :param flst_argv: argumentos (None = sys.argv[1:])
    :returns: CI_RC_OK
    """
    l_args = _parse_args(flst_argv)

    lf_newest_ts = process(l_args.input_csv, l_args.output_json,
                           l_args.since_ts, l_args.tracks_file, l_args.verify_file,
                           l_args.assoc_file, l_args.frames_file,
                           fs_source=l_args.source)

    if lf_newest_ts is not None:
        # [doc] "NEWEST_TS=<ts>" é um contrato entre processos, não só uma
        # linha de log: o blt_common.sh (track_new_frame) faz grep
        # "NEWEST_TS=" no stdout para saber se houve frame novo e cut -d= -f2
        # para gravar o valor em SLM_LAST_TS_FILE.  Mudar o formato (nome,
        # separador, algo antes na mesma linha) quebra em silêncio a detecção
        # de frame novo, sem nenhum erro no Python.
        print(f"NEWEST_TS={lf_newest_ts}")

    return CI_RC_OK

# ─────────────────────────────────────────────────────────────────────────────────────
# this is the bootstrap process

if "__main__" == __name__:
    logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S",
                        format="[%(asctime)s] [%(levelname)s] %(message)s",
                        level=logging.INFO)

    sys.exit(main())

# < the end >──────────────────────────────────────────────────────────────────────────
