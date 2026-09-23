#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABLACAO DE MECANISMO TEMPORAL (Artigo 1 / R3 C6) — nova Tabela 11
=================================================================
Bracos (todos com a config do artigo, abl_A, 12 seeds 42..53):
  notemp   : REMOVE a organizacao temporal do aluno
             (early_pen, late_pen e pre_pen multiplicados por 0;
              mantem l_hard, KD crit-gated, l_ep, tail/noncrit, aux,
              scheduler de lambda, arquitetura, dados, calibracao)
  a005/a030/a050 : sensibilidade de alpha (onset_exp_alpha)
  a015   : alpha canonico TREINADO SOB ESTE MESMO protocolo run_v29
           (a campanha exp_seed12_round1 usa o protocolo do pipeline canonico,
            com convencoes de avaliacao DIFERENTES — nao misturar; proveniencia
            da Tabela 8 permanece a campanha canonica)
  a000   : opcional (--with-a0) — alpha=0 exato (perfil uniforme na janela;
           NAO remove o condicionamento ao onset — ver AUDIT)

ONDE RODAR (GPU da Angela):
    cd C:\\Users\\Angela\\Documents\\DATALAKE-ANGELA\\MLProject\\TreinamentoNovo\\FGCS\\Inferencia
    python ..\\..\\Experimentos-Artigo1-AdHoc\\ablacao_mecanismo_temporal\\run_mechanism_ablation.py [braco]
    (sem argumento roda todos os pendentes; resumivel — checkpoints por
     seed/braco sao reutilizados; ~15-30 min/seed novo)
"""
import sys, os, re, time
from pathlib import Path

sys.path.insert(0, os.getcwd())
PIPELINE = Path("run_v29_ablacao_ttdef_ajuste_gatting.py")
SEEDS = list(range(42, 54))

ARMS = {
    # nome    alpha  torg_scale  label
    "notemp": (0.15, 0.0, "mechabl_notemp"),
    "a005":   (0.05, 1.0, "mechabl_a005"),
    "a015":   (0.15, 1.0, "mechabl_a015"),   # braco canonico SOB O MESMO protocolo
    "a030":   (0.30, 1.0, "mechabl_a030"),
    "a050":   (0.50, 1.0, "mechabl_a050"),
}
if "--with-a0" in sys.argv:
    ARMS["a000"] = (0.0, 1.0, "mechabl_a000")
sel = [a for a in sys.argv[1:] if a in ARMS]
arms = sel if sel else list(ARMS)

if not PIPELINE.exists():
    sys.exit("ERRO: rode dentro de FGCS/Inferencia (run_v29_... nao encontrado).")
base = PIPELINE.read_text(encoding="utf-8")

def build_src(alpha, scale, label):
    src = base
    # 1) modo A (config do artigo)
    src, n1 = re.subn(r'^ABLATION_MODE:\s*Optional\[str\]\s*=\s*.*$',
                      'ABLATION_MODE: Optional[str] = "A"   # [mech-abl]', src, count=1, flags=re.M)
    # 2) seeds 42..53
    src, n2 = re.subn(r'^SEEDS\s*=\s*\[.*?\].*$', f'SEEDS            = {SEEDS}   # [mech-abl]',
                      src, count=1, flags=re.M)
    # 3) alpha do braco — apenas na config "A"
    blocoA = re.search(r'"A": dict\(.*?\),\n', src, re.S).group(0)
    blocoA_new, n3 = re.subn(r'onset_exp_alpha\s*=\s*[0-9.]+', f'onset_exp_alpha   = {alpha}', blocoA, count=1)
    src = src.replace(blocoA, blocoA_new)
    # 4) label -> diretorios de saida/checkpoint separados por braco
    blocoA2 = re.search(r'"A": dict\(.*?\),\n', src, re.S).group(0)
    blocoA3, n4 = re.subn(r'label\s*=\s*"abl_A"', f'label             = "{label}"', blocoA2, count=1)
    src = src.replace(blocoA2, blocoA3)
    # 5) escala dos termos onset-anchored (remocao do mecanismo temporal)
    src, n5 = re.subn(r'\+ delta \* early_pen', f'+ {scale} * delta * early_pen   # [mech-abl]', src, count=1)
    src, n6 = re.subn(r'\+ late_pen(\s*#|\s*\n)', f'+ {scale} * late_pen\\1', src, count=1)
    src, n7 = re.subn(r'\+ pre_pen(\s*#|\s*\n)', f'+ {scale} * pre_pen\\1', src, count=1)
    checks = dict(n1=n1, n2=n2, n3=n3, n4=n4, n5=n5, n6=n6, n7=n7)
    bad = [k for k, v in checks.items() if v != 1]
    if bad:
        sys.exit(f"ERRO: patches nao aplicados ({bad}) — verifique a versao do pipeline.")
    return src

try:
    import torch
    print(f"[mech-abl] CUDA: {torch.cuda.is_available()}")
except ImportError:
    sys.exit("ERRO: torch nao instalado.")

for arm in arms:
    alpha, scale, label = ARMS[arm]
    print(f"\n{'='*60}\n[mech-abl] BRACO {arm}: alpha={alpha} torg_scale={scale} label={label}\n{'='*60}")
    src = build_src(alpha, scale, label)
    t0 = time.time()
    g = {"__name__": "__main__", "__file__": str(PIPELINE.resolve())}
    exec(compile(src, f"{PIPELINE}::{arm}", "exec"), g)
    print(f"[mech-abl] {arm} concluido em {(time.time()-t0)/3600:.2f} h")

print("\n[mech-abl] TODOS OS BRACOS CONCLUIDOS.")
print("[mech-abl] O braco alpha=0.15 canonico ja existe em exp_seed12_round1 (nao retreinar).")
print("[mech-abl] Agora rode: python analyze_mechanism_ablation.py (na pasta ablacao_mecanismo_temporal)")
