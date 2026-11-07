# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_slm_entry_scripts
smoke dos programas rodados pelo caminho, como no slm_run.sh e no runbook
(python3 apps/X.py), num subprocess **sem** PYTHONPATH.  O pytest.ini põe
apps/ no sys.path e esconderia um import quebrado.

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >──────────────────────────────────────────────────────────────────────────

# python library
import os
import pathlib
import subprocess
import sys

# pytest
import pytest

# < constants >────────────────────────────────────────────────────────────────────────

# raiz do projeto
MPTH_ROOT = pathlib.Path(__file__).resolve().parent.parent

# programas com argparse (--help não lê CSV nem chama a REDEMET)
CLST_SCRIPTS = ["apps/slmDownloader.py",
                "apps/slmReplay.py",
                "apps/slmTracker.py",
                "apps/slmRender.py",
                "apps/slmMapper.py"]

# ─────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("fs_script", CLST_SCRIPTS)
def test_help_sem_pythonpath(fs_script, tmp_path):
    ldct_env = {ls_key: ls_value for ls_key, ls_value in os.environ.items()
                if "PYTHONPATH" != ls_key}

    lproc = subprocess.run([sys.executable, str(MPTH_ROOT / fs_script), "--help"],
                           cwd=tmp_path, env=ldct_env, capture_output=True, text=True,
                           timeout=120)

    assert 0 == lproc.returncode, lproc.stderr
    assert "usage:" in lproc.stdout

# < the end >──────────────────────────────────────────────────────────────────────────
