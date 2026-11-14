# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
blt_struct
estruturas de dados do modelo:

  TLightning   um raio (descarga) lido do CSV do downloader
  TCell        uma célula convectiva: um aglomerado (cluster) de raios num frame
  TStormTrack  a mesma célula acompanhada ao longo de vários frames

atenção: este módulo é importado como "sst" em todo o projeto, nunca como
"str" -- um bug antigo sombreava o builtin str.

2026.nov  mlabru  TCell.assoc (diagnóstico da associação) e TStormTrack.end_reason
                  (por que o track acabou); nenhum dos dois é persistido
2026.nov  mlabru  [fix] TCell.t_ref e cell_time: relógio do tracking é a média
                  dos ts dos raios, não o primeiro raio da janela
2026.nov  mlabru  TCell.parent_id e TStormTrack.parent_id (divisão de células)
2026.nov  mlabru  TStormTrack.forecasts (previsões pendentes de verificação)
2026.out  mlabru  refatoração para legibilidade: comentários por campo,
                  docstrings completas, ruído de "# ok" removido; campos nunca
                  preenchidos removidos (TCell: severe, growing, dissipating,
                  severity_index, meta; TLightning: meta) -- versão anterior
                  em apps/.bkp/blt_struct.py.v3
2026.may  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import dataclasses

# < TCell >────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(slots=True)
class TCell:
    """
    célula convectiva: os raios de um cluster do DBSCAN, resumidos.

    criada em bltTracker.cluster_window; movimento e previsão são preenchidos
    depois, em blt_track.update_tracks.
    """
    # identificação: ID do track a que a célula pertence (vazio até a associação)
    id: str = ""
    # [new] ID do track de onde o track desta célula se dividiu ("" = nenhum)
    parent_id: str = ""
    # posição: (lat, lng) médios dos raios, em graus
    centroid: tuple = (0.0, 0.0)
    # tamanho: número de raios no cluster
    size: int = 0
    # raios por km² (ver blt_track.compute_density)
    density: float = 0.0
    # raios por minuto (ver blt_track.compute_flash_rate)
    flash_rate: float = 0.0
    # instante do primeiro e do último raio (epoch UTC, segundos)
    t_start: int = 0
    t_end: int = 0
    # [fix] instante representativo do centroide: média dos ts dos raios.  É o
    # relógio do tracking (dt do Kalman, verificação) -- ver cell_time
    t_ref: float = 0.0
    # movimento em relação ao frame anterior do mesmo track
    speed_kmh: float = 0.0
    direction_deg: float = 0.0
    # previsão: lista de dicts {lead_min, lat, lng, lat_std, lng_std}
    forecast_path: list = dataclasses.field(default_factory=list)
    # histórico curto do track (TCell anteriores), usado pelo render
    history: list = dataclasses.field(default_factory=list)
    # [new] diagnóstico da associação deste frame (blt_track.update_tracks;
    # gravado pelo blt_diag): kind ("continue", "two_point", "new", "split"),
    # dt_min, dist_km, score, size_prev, nis.  Não vai para o estado
    assoc: dict = dataclasses.field(default_factory=dict)
    # [new] taxa e salto de raios (glm_ljump; só com fonte de descargas, GLM):
    # taxa no intervalo mais recente (flashes/min), variação dela
    # (flashes/min²), maior DFRDT em desvios-padrão no frame e se houve salto
    flash_rate_now: float = 0.0
    # [new] taxa média dos últimos ~6 min (flashes/min): base da severidade
    flash_rate_recent: float = 0.0
    flash_rate_dfrdt: float = 0.0
    jump_sigma: float = 0.0
    lightning_jump: bool = False
    # estatísticas de amplitude/polaridade/tipo (ver bltTracker.compute_cell_stats)
    max_amp: float = 0.0
    mean_amp: float = 0.0
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    cg_ratio: float = 0.0
    ic_ratio: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────────────
def cell_to_dict(f_cell: TCell) -> dict:
    """
    converte uma TCell num dict com o subconjunto de campos que o histórico
    de tracks precisa entre execuções (slm_tracks_state.json).

    não confundir com bltTracker.export_frame, que grava a célula inteira
    (com estatísticas e previsão) no JSON do frame para o render.

    :param f_cell: célula a converter
    :returns: dict com id, centroid, size, speed_kmh, direction_deg, t_start,
              t_end e t_ref
    """
    return {
        "id":            f_cell.id,
        "centroid":      list(f_cell.centroid),
        "size":          f_cell.size,
        "speed_kmh":     f_cell.speed_kmh,
        "direction_deg": f_cell.direction_deg,
        "t_start":       f_cell.t_start,
        "t_end":         f_cell.t_end,
        "t_ref":         f_cell.t_ref,
    }

# ─────────────────────────────────────────────────────────────────────────────────────
def cell_from_dict(fdct_cell: dict) -> TCell:
    """
    reconstrói uma TCell a partir do dict gravado por cell_to_dict.
    os campos que não estão no dict ficam com o valor padrão.

    :param fdct_cell: dict lido do arquivo de estado dos tracks
    :returns: nova TCell
    :raises KeyError: se faltar alguma das chaves gravadas por cell_to_dict
    """
    l_cell = TCell()
    l_cell.id            = fdct_cell["id"]
    l_cell.centroid      = tuple(fdct_cell["centroid"])
    l_cell.size          = fdct_cell["size"]
    l_cell.speed_kmh     = fdct_cell["speed_kmh"]
    l_cell.direction_deg = fdct_cell["direction_deg"]
    l_cell.t_start       = fdct_cell["t_start"]
    l_cell.t_end         = fdct_cell["t_end"]
    # [fix] estado gravado antes de 2026.nov não tem a chave (cell_time cai no t_start)
    l_cell.t_ref         = fdct_cell.get("t_ref", 0.0)

    return l_cell

# ─────────────────────────────────────────────────────────────────────────────────────
def cell_time(f_cell: TCell) -> float:
    """
    o instante da célula para o tracking: t_ref (média dos ts dos raios, o
    instante a que o centroide corresponde).

    [fix] o relógio era o t_start (primeiro raio).  Com a janela fixa de
    CI_WINDOW_MINUTES, numa célula mais nova que a janela o primeiro raio é
    sempre o mesmo: t_start não andava, o dt entre frames era 0 e o Kalman
    não estimava velocidade nos primeiros ~20 min do track.

    :param f_cell: célula
    :returns: t_ref; t_start se t_ref não estiver preenchido (célula montada à
              mão, estado anterior a 2026.nov)
    """
    return f_cell.t_ref or f_cell.t_start

# < TLightning >───────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(slots=True)
class TLightning:
    """
    um raio individual, como lido do CSV.

    o downloader grava só timestamp, lat e lng; amp, polarity e kind ficam
    com o valor padrão (o formato aceita as colunas extras se um dia vierem).
    """
    # instante do raio (epoch UTC, segundos)
    ts: float
    # posição (graus)
    lat: float
    lng: float
    # intensidade (kA), opcional
    amp: float = 0.0
    # polaridade: +1, -1 ou 0 (desconhecida), opcional
    polarity: int = 0
    # tipo: "CG" (nuvem-solo), "IC" (intranuvem) ou "" (desconhecido), opcional
    kind: str = ""

# < TStormTrack >──────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class TStormTrack:
    """
    track de tempestade: a sequência de células associadas à mesma
    tempestade, frame a frame (ver blt_track.update_tracks).
    """
    # identificador de 8 caracteres (blt_track.generate_track_id)
    id: str
    # células do track, da mais antiga para a mais recente
    history: list = dataclasses.field(default_factory=list)
    # minutos desde a primeira célula
    age_minutes: float = 0.0
    # frames seguidos sem nenhuma célula associada
    missed_frames: int = 0
    # False quando o track passou de CI_MAX_MISSED frames perdidos
    active: bool = True
    # [new] ID do track de que este se dividiu ("" = nasceu sozinho)
    parent_id: str = ""
    # [new] previsões emitidas e ainda não verificadas (blt_verify): dicts com
    # issue_ts, lead_min, lat, lng e a posição de emissão lat0, lng0
    forecasts: list = dataclasses.field(default_factory=list)
    # [new] por que o track foi desativado NESTE frame: "missed", "merged" ou
    # "expired" ("" = ativo).  Não é persistido (tracks inativos não são
    # gravados); serve à verificação (blt_verify.close_tracks) e ao blt_diag
    end_reason: str = ""

    # ─────────────────────────────────────────────────────────────────────────────────
    @property
    def last_cell(self) -> TCell | None:
        """
        a célula mais recente do track.

        :returns: última TCell do histórico, ou None se o histórico estiver vazio
        """
        if not self.history:
            return None

        return self.history[-1]

# < the end >──────────────────────────────────────────────────────────────────────────
