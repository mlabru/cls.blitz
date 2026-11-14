# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
blt_spcgrid
índice espacial em grade (hash grid) para acelerar o DBSCAN.

ideia: o mapa é dividido em quadrados de lado ~eps (em graus).  Cada raio é
guardado no quadrado onde cai.  Para achar os vizinhos de um ponto basta olhar
o quadrado dele e os quadrados em volta, em vez de comparar com todos os raios
(O(n) por consulta em vez de O(n²) no total).  O filtro fino por distância
(haversine) é feito depois, em blt_spatial.region_query.

2026.out  mlabru  refatoração para legibilidade: nomes descritivos, logs de
                  debug removidos (rodavam a cada consulta), 111.0 vindo de
                  blt_geo, cálculo da largura em longitude em função própria
2026.may  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import collections
import math

# locals
import work.blt_geo as geo
import work.blt_struct as sst

# < constants >────────────────────────────────────────────────────────────────────────

# piso para cos(lat) ao alargar a busca em longitude (evita divisão por zero nos polos)
CF_MIN_COS_LATITUDE = 1e-6

# < TSpatialGrid >─────────────────────────────────────────────────────────────────────

class TSpatialGrid:
    """
    grade de quadrados de lado fixo (em km, convertido para graus) que guarda
    os raios por quadrado.

    uso:
        l_grid = TSpatialGrid(20.0)
        l_grid.insert(l_lightning)                  # para cada raio
        llst_candidates = l_grid.neighbors(lat, lng)
    """
    # ─────────────────────────────────────────────────────────────────────────────────
    def __init__(self, ff_cell_size_km: float):
        """
        cria uma grade vazia.

        :param ff_cell_size_km: lado de cada quadrado, em km (o DBSCAN usa o eps)
        """
        self.cell_size_km = ff_cell_size_km

        # o mesmo tamanho em graus (usa a escala da latitude nos dois eixos;
        # a diferença em longitude é compensada em neighbors())
        self.cell_size_deg = ff_cell_size_km / geo.CF_KM_PER_DEG_LAT

        # (linha, coluna) -> lista de raios naquele quadrado
        self.lightnings_by_cell = collections.defaultdict(list)

    # ─────────────────────────────────────────────────────────────────────────────────
    def insert(self, f_lightning: sst.TLightning) -> None:
        """
        guarda um raio no quadrado onde ele cai.

        :param f_lightning: raio a guardar (usa lat e lng)
        """
        lt_cell_key = self._cell_key(f_lightning.lat, f_lightning.lng)

        self.lightnings_by_cell[lt_cell_key].append(f_lightning)

    # ─────────────────────────────────────────────────────────────────────────────────
    def neighbors(self, ff_lat: float, ff_lng: float) -> list:
        """
        candidatos a vizinho de um ponto: todos os raios do quadrado do ponto e
        dos quadrados em volta.  Pode trazer pontos um pouco além do raio; quem
        chama filtra pela distância real.

        :param ff_lat: latitude do ponto (graus)
        :param ff_lng: longitude do ponto (graus)
        :returns: lista de TLightning candidatos
        """
        li_row, li_column = self._cell_key(ff_lat, ff_lng)
        li_longitude_span = self._longitude_span(ff_lat)

        llst_candidates = []

        # uma linha acima e uma abaixo (latitude); em longitude, tantas colunas
        # quanto preciso para cobrir o mesmo número de km
        for li_row_offset in (-1, 0, 1):
            for li_column_offset in range(-li_longitude_span, li_longitude_span + 1):
                lt_cell_key = (li_row + li_row_offset, li_column + li_column_offset)

                # testa antes de ler: ler um defaultdict criaria um quadrado vazio
                if lt_cell_key in self.lightnings_by_cell:
                    llst_candidates.extend(self.lightnings_by_cell[lt_cell_key])

        return llst_candidates

    # ─────────────────────────────────────────────────────────────────────────────────
    def _cell_key(self, ff_lat: float, ff_lng: float) -> tuple[int, int]:
        """
        índice (linha, coluna) do quadrado onde o ponto cai.

        :param ff_lat: latitude (graus)
        :param ff_lng: longitude (graus)
        :returns: (linha, coluna)
        """
        li_row = math.floor(ff_lat / self.cell_size_deg)
        li_column = math.floor(ff_lng / self.cell_size_deg)

        return (li_row, li_column)

    # ─────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _longitude_span(ff_lat: float) -> int:
        """
        quantas colunas olhar para cada lado, em longitude.

        [fix] o lado do quadrado em graus foi calculado como se 1 grau ~ 111 km
        nos dois eixos, mas isso só vale para latitude -- 1 grau de longitude
        encolhe por cos(lat).  Longe do equador (ex.: lat=-25°, região de SP)
        um vizinho dentro de eps_km pode cair 2+ colunas além de ±1 e ficaria
        fora de uma janela fixa 3x3, perdido pelo DBSCAN.  Por isso a busca em
        longitude é alargada por 1/cos(lat).

        :param ff_lat: latitude do ponto (graus)
        :returns: número de colunas para cada lado (1 no equador, mais nos trópicos/sul)
        """
        lf_cos_latitude = max(math.cos(math.radians(ff_lat)), CF_MIN_COS_LATITUDE)

        return math.ceil(1.0 / lf_cos_latitude)

# < the end >──────────────────────────────────────────────────────────────────────────
