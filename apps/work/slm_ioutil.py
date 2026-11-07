# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
slm_ioutil
escrita atômica dos arquivos de estado compartilhados entre processos.

por quê: o tracker grava o estado dos tracks e o JSON do frame, e o downloader
grava o CSV de raios, enquanto outro processo (o render, o mapper, a próxima
rodada do cron) pode estar lendo.  Grava-se num "<caminho>.tmp" no mesmo
diretório e só então ele é renomeado para o nome final (os.replace, atômico no
mesmo sistema de arquivos): quem lê vê o arquivo antigo inteiro ou o novo
inteiro, nunca um arquivo pela metade.

  atomic_write       genérica: recebe uma função que grava num arquivo aberto
                     (texto ou binário -- o PNG do slm_mapbase usa binário)
  atomic_write_json  um objeto como JSON (estado dos tracks, frame)
  atomic_write_csv   uma lista de dicts como CSV com cabeçalho (raios)

2026.out  mlabru  temporário apagado se a gravação falhar; modo binário (PNG)
2026.out  mlabru  refatoração para legibilidade: log de debug removido;
                  atomic_write genérica e atomic_write_csv (a escrita atômica
                  do CSV estava escrita à mão no slmDownloader)
2026.ago  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
from __future__ import annotations

import csv
import json
import os
import typing

# ─────────────────────────────────────────────────────────────────────────────────────
def atomic_write(fs_path: str,
                 f_writer: typing.Callable[[typing.IO], None],
                 fs_newline: str | None = None,
                 fb_binary: bool = False) -> None:
    """
    grava fs_path de forma atômica.

    passo a passo:
      1. abre "<fs_path>.tmp" para escrita (texto UTF-8, ou binário);
      2. chama f_writer(arquivo aberto), que grava o conteúdo;
      3. renomeia o temporário para fs_path (os.replace, atômico).
    Se o writer (ou o rename) falhar, o temporário é apagado, a exceção sobe
    e o arquivo final não é tocado.

    :param fs_path: caminho final do arquivo (str ou pathlib.Path; o diretório
                    já precisa existir)
    :param f_writer: função que recebe o arquivo aberto e grava nele
    :param fs_newline: repassado ao open() em modo texto ("" para CSV, ver o módulo csv)
    :param fb_binary: True abre o temporário em modo binário (ex.: PNG)
    :raises OSError: falha ao gravar ou renomear
    """
    ls_final_path = str(fs_path)
    ls_temporary_path = ls_final_path + ".tmp"

    try:
        if fb_binary:
            with open(ls_temporary_path, "wb") as l_file:
                f_writer(l_file)

        else:
            with open(ls_temporary_path, "w", encoding="utf-8", newline=fs_newline) as l_file:
                f_writer(l_file)

        os.replace(ls_temporary_path, ls_final_path)

    except BaseException:
        # [fix] sem isto, um .tmp pela metade ficava no diretório de estado
        if os.path.exists(ls_temporary_path):
            os.remove(ls_temporary_path)
        raise

# ─────────────────────────────────────────────────────────────────────────────────────
def atomic_write_json(fs_path: str, f_obj: object, fi_indent: int = 2) -> None:
    """
    grava f_obj como JSON em fs_path, de forma atômica.

    :param fs_path: caminho final do arquivo (o diretório já precisa existir)
    :param f_obj: objeto serializável em JSON (dict, list, ...)
    :param fi_indent: indentação do JSON gerado
    :raises OSError: falha ao gravar ou renomear
    """
    def write_json(f_file: typing.TextIO) -> None:
        json.dump(f_obj, f_file, indent=fi_indent)

    atomic_write(fs_path, write_json)

# ─────────────────────────────────────────────────────────────────────────────────────
def atomic_write_csv(fs_path: str, flst_fieldnames: list, flst_rows: list) -> None:
    """
    grava uma lista de dicts como CSV (com cabeçalho) em fs_path, de forma atômica.

    :param fs_path: caminho final do arquivo (o diretório já precisa existir)
    :param flst_fieldnames: nomes das colunas, na ordem do arquivo
    :param flst_rows: linhas (dicts com as chaves de flst_fieldnames)
    :raises OSError: falha ao gravar ou renomear
    """
    def write_csv(f_file: typing.TextIO) -> None:
        l_writer = csv.DictWriter(f_file, fieldnames=flst_fieldnames)
        l_writer.writeheader()
        l_writer.writerows(flst_rows)

    atomic_write(fs_path, write_csv, fs_newline="")

# < the end >──────────────────────────────────────────────────────────────────────────
