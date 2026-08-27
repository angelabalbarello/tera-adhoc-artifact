#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPANSAO DE SEEDS 4 -> 12 (Artigo 1 / Ad Hoc Networks, revisao R2.5/R3)
=======================================================================
Executa o pipeline oficial do artigo (run_v29, ABLATION_MODE="A") para as
seeds 42..53. Seeds 42-45 reutilizam os checkpoints em modelos_salvos/abl_A
(re-inferencia rapida); seeds 46-53 treinam do zero.

ONDE RODAR (importante):
    cd C:\\Users\\Angela\\Documents\\DATALAKE-ANGELA\\MLProject\\TreinamentoNovo\\FGCS\\Inferencia
    python ..\\..\\Experimentos-Artigo1-AdHoc\\seed-expansion\\run_seed_expansion.py

Requisitos: python 3.10+, torch (CUDA), numpy, pandas, scikit-learn, matplotlib.
Tempo estimado (RTX 3050 Ti): ~15-30 min/seed nova -> 2-4 h no total.
Pode interromper e retomar: modelos ja salvos por seed sao reutilizados.
"""
import sys, os, re, time
from pathlib import Path

# Garante que a pasta corrente (FGCS/Inferencia) esteja no sys.path para que
# o pipeline consiga importar synthetic_driver_risk_v7.py (gerador TERA-Gen).
sys.path.insert(0, os.getcwd())

PIPELINE = Path("run_v29_ablacao_ttdef_ajuste_gatting.py")
NEW_SEEDS = list(range(42, 54))          # 42..53 (edite se quiser menos)

if not PIPELINE.exists():
    sys.exit(f"ERRO: {PIPELINE} nao encontrado. Rode este script DENTRO de FGCS/Inferencia.")

src = PIPELINE.read_text(encoding="utf-8")

# 1) forca ABLATION_MODE = "A" (configuracao do artigo)
src, n1 = re.subn(r'^ABLATION_MODE:\s*Optional\[str\]\s*=\s*.*$',
                  'ABLATION_MODE: Optional[str] = "A"   # [seed-expansion]',
                  src, count=1, flags=re.M)
# 2) substitui a lista de seeds
src, n2 = re.subn(r'^SEEDS\s*=\s*\[.*?\].*$',
                  f'SEEDS            = {NEW_SEEDS}   # [seed-expansion 4->12]',
                  src, count=1, flags=re.M)
if n1 != 1 or n2 != 1:
    sys.exit(f"ERRO: padroes nao encontrados no pipeline (ABLATION_MODE={n1}, SEEDS={n2}).")

print(f"[seed-expansion] SEEDS = {NEW_SEEDS}  |  ABLATION_MODE = 'A'")
try:
    import torch
    print(f"[seed-expansion] CUDA disponivel: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " -> vai rodar em CPU (LENTO!)"))
except ImportError:
    sys.exit("ERRO: torch nao instalado.")

t0 = time.time()
g = {"__name__": "__main__", "__file__": str(PIPELINE.resolve())}
exec(compile(src, str(PIPELINE), "exec"), g)
print(f"[seed-expansion] concluido em {(time.time()-t0)/3600:.2f} h")
print("[seed-expansion] agora rode: python merge_and_stats.py (na mesma pasta)")
