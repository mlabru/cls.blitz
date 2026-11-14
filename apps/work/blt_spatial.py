# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
blt_spatial
distância geodésica (haversine) e agrupamento de raios em células (DBSCAN).

DBSCAN em uma frase: um raio com pelo menos min_pts raios a até eps_km dele é
um "ponto central"; pontos centrais próximos formam o mesmo cluster, e os raios
isolados (ruído) são descartados.  Cada cluster vira uma célula convectiva.

fluxo (dbscan_spatial):
  1) indexa todos os raios numa grade (blt_spcgrid.TSpatialGrid);
  2) para cada raio ainda não visitado, busca os vizinhos (region_query);
  3) se for ponto central, cresce o cluster em largura (expand_cluster).

regra do projeto: distância sempre com haversine_km, nunca com a diferença
de lat/lng em graus (ver blt_geo).

2026.nov  mlabru  eps/min_pts obrigatórios (os padrões daqui, 20 km, divergiam
                  dos do bltTracker, 12 km)
2026.nov  mlabru  parâmetros de algoritmo lidos na chamada, não na definição
                  (padrão None): o bltReplay sobrescreve as constantes
2026.nov  mlabru  [fix] raio de borda entra em UM cluster só (o primeiro que o
                  alcança): antes podia entrar em dois, inflando size/density
2026.out  mlabru  refatoração para legibilidade: expand_cluster devolve o
                  cluster (antes recebia lista e set para preencher), logs de
                  debug removidos dos laços internos, raio da Terra em blt_geo
2026.may  mlabru  dedup O(1) em expand_cluster via set paralelo ao cluster;
                  remoção de import morto (defaultdict)
2026.may  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import math

# locals
import work.blt_geo as geo
import work.blt_spcgrid as sgr
import work.blt_struct as sst

# ─────────────────────────────────────────────────────────────────────────────────────
def dbscan_spatial(flst_events: list, ff_eps_km: float, fi_min_pts: int) -> list:
    """
    agrupa raios em clusters por densidade (DBSCAN com índice em grade).

    [fix] eps e min_pts são obrigatórios: havia padrões próprios aqui
    (20 km / 10) diferentes dos do bltTracker (12 km / 10), que sempre informa
    os seus; um --set nos padrões daqui era aceito pelo bltReplay e não mudava
    nada.  A fonte única é bltTracker.CF_EPS_KM / CI_MIN_PTS.

    :param flst_events: lista de TLightning
    :param ff_eps_km: raio de vizinhança, em km
    :param fi_min_pts: mínimo de vizinhos (contando o próprio ponto) para ser
                       ponto central
    :returns: lista de clusters; cada cluster é uma lista de TLightning.
              Raios de ruído não aparecem em nenhum cluster.
    """
    l_grid = _build_grid(flst_events, ff_eps_km)

    # pontos já visitados e pontos já colocados num cluster, por id()
    # (TLightning não é hashable)
    lset_visited_ids = set()
    lset_assigned_ids = set()
    llst_clusters = []

    for l_event in flst_events:
        if id(l_event) in lset_visited_ids:
            continue

        lset_visited_ids.add(id(l_event))

        llst_neighbors = region_query(l_grid, l_event, ff_eps_km)

        # poucos vizinhos: ruído (pode ainda entrar num cluster como borda de outro)
        if len(llst_neighbors) < fi_min_pts:
            continue

        llst_cluster = expand_cluster(l_grid, l_event, llst_neighbors, lset_visited_ids,
                                      ff_eps_km, fi_min_pts, lset_assigned_ids)
        llst_clusters.append(llst_cluster)

    return llst_clusters

# ─────────────────────────────────────────────────────────────────────────────────────
def _build_grid(flst_events: list, ff_eps_km: float) -> sgr.TSpatialGrid:
    """
    cria a grade espacial com todos os raios, com quadrados de lado eps.

    :param flst_events: lista de TLightning
    :param ff_eps_km: lado do quadrado, em km
    :returns: grade preenchida
    """
    l_grid = sgr.TSpatialGrid(ff_eps_km)

    for l_event in flst_events:
        l_grid.insert(l_event)

    return l_grid

# ─────────────────────────────────────────────────────────────────────────────────────
def expand_cluster(f_grid: sgr.TSpatialGrid,
                   f_seed: sst.TLightning,
                   flst_queue: list,
                   fset_visited_ids: set,
                   ff_eps_km: float,
                   fi_min_pts: int,
                   fset_assigned_ids: set | None = None) -> list:
    """
    cresce um cluster a partir de um ponto central, em largura (BFS).

    passo a passo, para cada ponto da fila:
      1. se ainda não foi visitado, marca como visitado e busca os vizinhos
         dele; se ele também for ponto central, os vizinhos entram no fim da fila;
      2. se ainda não está em NENHUM cluster, entra neste.
    Termina quando a fila acaba (todos os pontos alcançáveis por densidade).

    [fix] o "já está num cluster?" usa um set de id() (O(1)), não uma busca
    linear na lista (O(n)).
    [fix] o set é GLOBAL (fset_assigned_ids, de dbscan_spatial), não do
    cluster: um raio de borda (não central) a até eps de dois clusters
    entrava nos dois -- contado duas vezes em size, flash_rate e density.
    Agora fica no primeiro que o alcança, como no DBSCAN clássico.  Pontos
    centrais nunca disputam: dois centrais vizinhos já são o mesmo cluster.

    :param f_grid: grade com todos os raios
    :param f_seed: ponto central que inicia o cluster
    :param flst_queue: vizinhos do ponto inicial; a lista é ESTENDIDA aqui (vira a fila)
    :param fset_visited_ids: ids já visitados; é ATUALIZADO aqui
    :param ff_eps_km: raio de vizinhança, em km
    :param fi_min_pts: mínimo de vizinhos para ser ponto central
    :param fset_assigned_ids: ids já colocados em algum cluster; é ATUALIZADO
                              aqui.  None = só este cluster conta
    :returns: o cluster (lista de TLightning, começando por f_seed)
    """
    lset_assigned_ids = set() if fset_assigned_ids is None else fset_assigned_ids

    llst_cluster = [f_seed]
    lset_assigned_ids.add(id(f_seed))

    # índice explícito (e não "for ... in"): a fila cresce durante o laço
    li_position = 0

    while li_position < len(flst_queue):
        l_point = flst_queue[li_position]
        li_position += 1

        if id(l_point) not in fset_visited_ids:
            fset_visited_ids.add(id(l_point))

            llst_point_neighbors = region_query(f_grid, l_point, ff_eps_km)

            if len(llst_point_neighbors) >= fi_min_pts:
                flst_queue.extend(llst_point_neighbors)

        if id(l_point) not in lset_assigned_ids:
            lset_assigned_ids.add(id(l_point))
            llst_cluster.append(l_point)

    return llst_cluster

# ─────────────────────────────────────────────────────────────────────────────────────
def haversine_km(ff_lat1: float, ff_lng1: float, ff_lat2: float, ff_lng2: float) -> float:
    """
    distância sobre a superfície da Terra entre dois pontos (fórmula de haversine).

    :param ff_lat1: latitude do ponto 1 (graus)
    :param ff_lng1: longitude do ponto 1 (graus)
    :param ff_lat2: latitude do ponto 2 (graus)
    :param ff_lng2: longitude do ponto 2 (graus)
    :returns: distância em km
    """
    lf_lat1_rad = math.radians(ff_lat1)
    lf_lat2_rad = math.radians(ff_lat2)

    lf_delta_lat_rad = math.radians(ff_lat2 - ff_lat1)
    lf_delta_lng_rad = math.radians(ff_lng2 - ff_lng1)

    # quadrado da metade da corda entre os pontos (esfera de raio 1)
    lf_half_chord_sq = (math.sin(lf_delta_lat_rad / 2.0) ** 2 +
                        math.cos(lf_lat1_rad) * math.cos(lf_lat2_rad) *
                        math.sin(lf_delta_lng_rad / 2.0) ** 2)

    # ângulo central entre os pontos (radianos)
    lf_central_angle = 2.0 * math.asin(math.sqrt(lf_half_chord_sq))

    return geo.CF_EARTH_RADIUS_KM * lf_central_angle

# ─────────────────────────────────────────────────────────────────────────────────────
def region_query(f_grid: sgr.TSpatialGrid, f_event: sst.TLightning, ff_eps_km: float) -> list:
    """
    todos os raios a até eps_km de um raio (inclusive ele mesmo).

    :param f_grid: grade com todos os raios
    :param f_event: raio de referência
    :param ff_eps_km: raio de busca, em km
    :returns: lista de TLightning dentro do raio
    """
    llst_inside = []

    # a grade devolve candidatos; aqui fica só quem está de fato a <= eps_km
    for l_candidate in f_grid.neighbors(f_event.lat, f_event.lng):
        lf_distance_km = haversine_km(f_event.lat, f_event.lng,
                                      l_candidate.lat, l_candidate.lng)

        if lf_distance_km <= ff_eps_km:
            llst_inside.append(l_candidate)

    return llst_inside

# < the end >──────────────────────────────────────────────────────────────────────────
