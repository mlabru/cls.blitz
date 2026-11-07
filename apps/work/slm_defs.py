# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

Simple Lightning Model
slm_defs
definições de acesso à REDEMET: a API key (lida de apps/.env) e a URL do
produto STSC (raios).

atenção: importar este módulo carrega o .env (dotenv.load_dotenv) -- é o único
efeito colateral de import do projeto.  Sem CS_REDEMET_KEY no .env ou no
ambiente, CS_REDEMET_KEY fica None e o slmDownloader recusa rodar.

2026.out  mlabru  movido para apps/work/ (import work.slm_defs)
2026.out  mlabru  refatoração para legibilidade: docstring do módulo, banners
2026.may  mlabru  cleanup
2023.apr  mlabru  change to MongoDB
2023.apr  mlabru  change REDEMET URL
2022.apr  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
import os

# dotenv
import dotenv

# < environment >──────────────────────────────────────────────────────────────────────

# lê as variáveis do .env: o dotenv procura a partir do diretório deste módulo
# (apps/work/) e sobe até achar o apps/.env
dotenv.load_dotenv()

# REDEMET API key (None se não estiver definida)
CS_REDEMET_KEY = os.getenv("CS_REDEMET_KEY")

# < constants >────────────────────────────────────────────────────────────────────────

# URL do produto STSC; {0} recebe a API key
#
# [doc] anima=60 pede à API os últimos 60 minutos de descargas.  Esse valor
# precisa ficar em sincronia com CI_MAX_HOURS (apps/slmDownloader.py), que
# define a janela de retenção local do CSV acumulado (CI_MAX_HOURS=1 == 60
# min).  Se um dos dois mudar sem o outro: anima menor que CI_MAX_HOURS*60
# passa a perder dados na janela de retenção (a API nunca devolve o suficiente
# para preenchê-la); anima maior apenas desperdiça payload da resposta, já que
# o excedente é descartado por filter_recent_rows() no download.
CS_REDEMET_URL = "https://api-redemet.decea.mil.br/produtos/stsc?api_key={0}&anima=60"

# < the end >──────────────────────────────────────────────────────────────────────────
