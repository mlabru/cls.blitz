# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

conftest
fixtures compartilhadas pela suite de testes.

os módulos de apoio ficam em apps/work/ e são importados pelo pacote
("import work.slm_struct as sst"), como no 27.nowcast; o pytest.ini
(pythonpath = apps apps/util) põe apps/ no sys.path antes de qualquer teste
importar um deles.

2026.out  mlabru  módulos slm_* movidos para apps/work/ (import work.slm_xxx)
2026.out  mlabru  isolated_env (autouse): sem SLM_*/MET_* herdados do shell
                  (um SLM_PATH exportado mudaria onde o slm_run.sh grava, e
                  MET_ENV=test ligaria a trava do ambiente de teste); o
                  sys.path vem do pythonpath do pytest.ini
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
import os

# pytest
import pytest

# ─────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """
    remove do ambiente as variáveis SLM_* (slm_cfg.sh) e MET_* (trava do
    ambiente de teste); os subprocessos dos testes herdam o ambiente limpo
    """
    for ls_var in list(os.environ):
        if ls_var.startswith(("SLM_", "MET_")):
            monkeypatch.delenv(ls_var, raising=False)

# < the end >──────────────────────────────────────────────────────────────────────────
