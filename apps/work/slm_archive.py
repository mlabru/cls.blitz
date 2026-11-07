# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
slm_archive
arquivo diário dos raios brutos, para reprocessar dias passados
(slmReplay.py) com outras constantes.

por quê: o CSV do pipeline (slm_lightning.csv) guarda só a última hora.
CF_EPS_KM, a associação e o Kalman mudam QUAIS células e tracks existem,
então só podem ser calibrados rodando o tracker de novo sobre os mesmos
raios -- o CSV de verificação só mede as constantes com que foi gerado.

formato: um arquivo por dia UTC da rodada, slm_raw_AAAAMMDD.csv.gz, com

    timestamp,lat,lng,fetched_ts

fetched_ts é o instante (epoch UTC) da rodada do downloader que trouxe o
raio.  Com ele o replay reconstitui, rodada a rodada, o que o tracker tinha
em mãos -- inclusive os raios que a REDEMET entrega atrasados.

cada rodada acrescenta só os raios novos (os que o _drop_duplicates do
downloader deixou passar) como um novo membro gzip: concatenar membros gera
um .gz válido, que o gzip do Python lê inteiro.  Um membro truncado no fim
(leitura enquanto o downloader escreve) é ignorado com aviso.

2026.nov  mlabru  initial version
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import csv
import datetime
import gzip
import logging
import os
import re

# < constants >────────────────────────────────────────────────────────────────────────

# nome dos arquivos: prefixo + AAAAMMDD (dia UTC da rodada) + sufixo
CS_ARCHIVE_PREFIX = "slm_raw_"
CS_ARCHIVE_SUFFIX = ".csv.gz"

# colunas do arquivo, na ordem
CLST_ARCHIVE_COLUMNS = ["timestamp", "lat", "lng", "fetched_ts"]

# dias mantidos (arquivos mais antigos são apagados por purge_old).  Um dia
# convectivo forte no Brasil tem da ordem de 10^5 a 10^6 raios: ~1-10 MB
# comprimido por dia
CI_ARCHIVE_DAYS = 30

# reconhece o nome de um arquivo do archive e separa a data
CRE_ARCHIVE_NAME = re.compile(r"^" + re.escape(CS_ARCHIVE_PREFIX) +
                              r"(\d{8})" + re.escape(CS_ARCHIVE_SUFFIX) + r"$")

# < logging >──────────────────────────────────────────────────────────────────────────

M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# ═════════════════════════════════════════════════════════════════════════════════════
# gravação (slmDownloader)
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def archive_path(fs_dir: str, ff_fetched_ts: float) -> str:
    """
    o arquivo do dia UTC de uma rodada.

    :param fs_dir: diretório do archive
    :param ff_fetched_ts: instante da rodada (epoch UTC)
    :returns: caminho slm_raw_AAAAMMDD.csv.gz
    """
    ldt_day = datetime.datetime.fromtimestamp(ff_fetched_ts, datetime.timezone.utc)

    return os.path.join(fs_dir, CS_ARCHIVE_PREFIX + ldt_day.strftime("%Y%m%d") +
                        CS_ARCHIVE_SUFFIX)

# ─────────────────────────────────────────────────────────────────────────────────────
def append_rows(fs_dir: str, flst_rows: list, ff_fetched_ts: float) -> str | None:
    """
    acrescenta ao arquivo do dia os raios trazidos por uma rodada.

    :param fs_dir: diretório do archive (criado se não existir)
    :param flst_rows: dicts com timestamp, lat, lng (as linhas do CSV do pipeline)
    :param ff_fetched_ts: instante da rodada (epoch UTC)
    :returns: caminho gravado; None se não havia linha
    :raises OSError: falha ao criar o diretório ou gravar
    """
    if not flst_rows:
        return None

    os.makedirs(fs_dir, exist_ok=True)

    ls_path = archive_path(fs_dir, ff_fetched_ts)
    lb_needs_header = not os.path.isfile(ls_path) or 0 == os.path.getsize(ls_path)
    li_fetched_ts = int(ff_fetched_ts)

    with gzip.open(ls_path, "at", encoding="utf-8", newline="") as l_file:
        l_writer = csv.writer(l_file)

        if lb_needs_header:
            l_writer.writerow(CLST_ARCHIVE_COLUMNS)

        l_writer.writerows([ldct_row["timestamp"], ldct_row["lat"], ldct_row["lng"],
                            li_fetched_ts] for ldct_row in flst_rows)

    return ls_path

# ─────────────────────────────────────────────────────────────────────────────────────
def purge_old(fs_dir: str, ff_now_ts: float, fi_keep_days: int | None = None) -> int:
    """
    apaga os arquivos do archive mais antigos que fi_keep_days dias.

    :param fs_dir: diretório do archive (inexistente = nada a fazer)
    :param ff_now_ts: agora (epoch UTC)
    :param fi_keep_days: dias mantidos, contando hoje (None = CI_ARCHIVE_DAYS)
    :returns: quantos arquivos foram apagados
    """
    if fi_keep_days is None:
        fi_keep_days = CI_ARCHIVE_DAYS

    if not os.path.isdir(fs_dir):
        return 0

    ldt_today = datetime.datetime.fromtimestamp(ff_now_ts, datetime.timezone.utc).date()
    ls_oldest_kept = (ldt_today - datetime.timedelta(days=fi_keep_days - 1)).strftime("%Y%m%d")

    li_removed = 0

    for ls_name in os.listdir(fs_dir):
        l_match = CRE_ARCHIVE_NAME.match(ls_name)

        # AAAAMMDD compara certo como texto
        if l_match is None or l_match.group(1) >= ls_oldest_kept:
            continue

        os.remove(os.path.join(fs_dir, ls_name))
        li_removed += 1

    return li_removed

# ═════════════════════════════════════════════════════════════════════════════════════
# leitura (slmReplay)
# ═════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────────
def load(flst_paths: list) -> list:
    """
    lê raios de arquivos do archive (.csv.gz) ou de CSVs simples.

    aceita também o CSV do pipeline (timestamp,lat,lng, sem fetched_ts) e o
    archive descomprimido: o formato é reconhecido pela extensão e pelo
    número de colunas.  Linhas malformadas são puladas com aviso.

    :param flst_paths: arquivos, em qualquer ordem
    :returns: lista de tuplas (fetched_ts ou None, ts, lat, lng), na ordem lida
    """
    llst_rows = []

    for ls_path in flst_paths:
        llst_rows.extend(_load_file(ls_path))

    return llst_rows

# ─────────────────────────────────────────────────────────────────────────────────────
def _load_file(fs_path: str) -> list:
    """
    lê um arquivo (ver load).

    :param fs_path: arquivo .csv ou .csv.gz
    :returns: lista de tuplas (fetched_ts ou None, ts, lat, lng)
    """
    llst_rows = []
    l_opener = gzip.open if fs_path.endswith(".gz") else open

    with l_opener(fs_path, "rt", encoding="utf-8", newline="") as l_file:
        l_reader = csv.reader(l_file)

        try:
            for llst_fields in l_reader:
                lt_row = _parse_fields(llst_fields)

                if lt_row is not None:
                    llst_rows.append(lt_row)

        # [doc] membro gzip truncado no fim: o downloader está escrevendo
        except EOFError:
            M_LOG.warning("%s termina truncado: lidas %d linhas.", fs_path, len(llst_rows))

    return llst_rows

# ─────────────────────────────────────────────────────────────────────────────────────
def _parse_fields(flst_fields: list) -> tuple | None:
    """
    converte uma linha do arquivo.

    :param flst_fields: campos da linha
    :returns: (fetched_ts ou None, ts, lat, lng); None para cabeçalho, linha
              vazia ou malformada (esta com aviso)
    """
    if not flst_fields or "timestamp" == flst_fields[0]:
        return None

    try:
        lf_ts = float(flst_fields[0])
        lf_lat = float(flst_fields[1])
        lf_lng = float(flst_fields[2])
        lf_fetched_ts = float(flst_fields[3]) if len(flst_fields) > 3 and flst_fields[3] else None

    except (ValueError, IndexError) as l_error:
        M_LOG.warning("linha ignorada: %r — %s", flst_fields, l_error)
        return None

    return lf_fetched_ts, lf_ts, lf_lat, lf_lng

# < the end >──────────────────────────────────────────────────────────────────────────
