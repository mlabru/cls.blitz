# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
test_slm_ioutil
testes da escrita atômica (slm_ioutil).

2026.out  mlabru  initial version: JSON, CSV, binário e limpeza do .tmp em falha
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
import json

# pytest
import pytest

# locals
import work.slm_ioutil as ioutil

# ─────────────────────────────────────────────────────────────────────────────────────
def test_atomic_write_json_grava_e_nao_deixa_tmp(tmp_path):
    lpth_file = tmp_path / "estado.json"

    ioutil.atomic_write_json(str(lpth_file), {"a": 1})

    assert {"a": 1} == json.loads(lpth_file.read_text())
    assert ["estado.json"] == [lp.name for lp in tmp_path.iterdir()]

# ─────────────────────────────────────────────────────────────────────────────────────
def test_atomic_write_csv_com_cabecalho(tmp_path):
    lpth_file = tmp_path / "raios.csv"

    ioutil.atomic_write_csv(str(lpth_file), ["timestamp", "lat", "lng"],
                            [{"timestamp": 1, "lat": 2.0, "lng": 3.0}])

    assert "timestamp,lat,lng\n1,2.0,3.0\n" == lpth_file.read_text()

# ─────────────────────────────────────────────────────────────────────────────────────
def test_atomic_write_binario_aceita_pathlib(tmp_path):
    lpth_file = tmp_path / "img.png"

    ioutil.atomic_write(lpth_file, lambda f_file: f_file.write(b"\x89PNG"), fb_binary=True)

    assert b"\x89PNG" == lpth_file.read_bytes()

# ─────────────────────────────────────────────────────────────────────────────────────
def test_falha_no_writer_apaga_tmp_e_preserva_o_arquivo(tmp_path):
    # [fix] antes o .tmp pela metade ficava no diretório
    lpth_file = tmp_path / "estado.json"
    lpth_file.write_text("antigo")

    def failing_writer(f_file):
        f_file.write("pela metade")
        raise RuntimeError("disco cheio")

    with pytest.raises(RuntimeError):
        ioutil.atomic_write(str(lpth_file), failing_writer)

    assert "antigo" == lpth_file.read_text()
    assert ["estado.json"] == [lp.name for lp in tmp_path.iterdir()]

# < the end >──────────────────────────────────────────────────────────────────────────
