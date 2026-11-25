# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

conftest
fixtures compartilhadas pela suite de testes.

os módulos de apoio ficam em apps/work/ e são importados pelo pacote
("import work.blt_struct as sst"), como no 27.nowcast; o pytest.ini
(pythonpath = apps) põe apps/ no sys.path antes de qualquer teste
importar um deles.

2026.out  mlabru  módulos slm_* movidos para apps/work/ (import work.slm_xxx)
2026.out  mlabru  isolated_env (autouse): sem SLM_*/MET_* herdados do shell
                  (um SLM_PATH exportado mudaria onde o blt_run.sh grava, e
                  CLS_ENV=test ligaria a trava do ambiente de teste); o
                  sys.path vem do pythonpath do pytest.ini
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
import os
import pathlib

# pytest
import pytest

# ─────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """
    remove do ambiente as variáveis SLM_* (blt_cfg.sh) e MET_* (trava do
    ambiente de teste); os subprocessos dos testes herdam o ambiente limpo
    """
    for ls_var in list(os.environ):
        if ls_var.startswith(("SLM_", "MET_")):
            monkeypatch.delenv(ls_var, raising=False)

# < ambiente cls >─────────────────────────────────────────────────────────────────────

# a suíte não depende do cls.env de quem a roda (~/.config/cls/cls.env): os
# scripts shell testados herdam este ambiente e não carregam nada de lá
os.environ["CLS_ENV_FILE"] = os.devnull

# cls_common.sh: CLS_LIB do ambiente ou, no checkout de desenvolvimento, o
# repo cls.common irmão (<raiz>/cls.common, com o projeto em
# <raiz>/<nn>.<proj>/cls.<proj>)
os.environ.setdefault("CLS_LIB", str(pathlib.Path(__file__).resolve().parents[3]
                                     / "cls.common" / "bash"))

# < the end >──────────────────────────────────────────────────────────────────────────
