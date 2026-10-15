# -*- coding: utf-8 -*-
"""
patch_v29_baseline_hybrid_unificado.py
Aplica TRÊS patches no run_v29_ablacao_ttdef_ajuste_gatting.py:

  PATCH 1 — evaluate_for_m:
    Adiciona avaliação LSTM-Baseline com política híbrida logo após
    a avaliação AF-KD híbrida, usando os mesmos (tau_d, tau_h) calibrados
    no VAL para o AF-KD. Gera o ponto "Baseline Híbrida" no gráfico de
    trade-off FR × custo.

  PATCH 2 — export_latex_macros:
    Adiciona ("BaseHibr", "LSTM-Baseline", None) ao mapping principal,
    gerando automaticamente:
      \\resBaseHibrFUm  \\stdBaseHibrFUm
      \\resBaseHibrECE  \\stdBaseHibrECE
      \\resBaseHibrFail \\stdBaseHibrFail
      \\resBaseHibrTTD  \\stdBaseHibrTTD
      \\resBaseHibrTTDef \\stdBaseHibrTTDef
      \\resBaseHibrCost \\stdBaseHibrCost
      \\resBaseHibrSkip \\stdBaseHibrSkip

  PATCH 3 — main_summary / rebuild:
    Adiciona coluna N_Seeds ao main_summary para rastrear cobertura
    de seeds por configuração no CSV consolidado.

USO
  1) Coloque este arquivo na mesma pasta do run_v29.
  2) Execute:
       python patch_v29_baseline_hybrid_unificado.py
  3) Rode o run_v29 normalmente:
       python run_v29_ablacao_ttdef_ajuste_gatting.py
     — ou use run_hybrid_completion.py para completar apenas as seeds faltantes.

SEGURANÇA
  - Cria backup automático antes de qualquer modificação.
  - Cada patch verifica o trecho original antes de aplicar.
  - Em caso de falha em qualquer patch, o arquivo original é restaurado.
"""

from pathlib import Path
import shutil
import sys

TARGET = Path("run_v29_ablacao_ttdef_ajuste_gatting.py")

if not TARGET.exists():
    print(f"erro  {TARGET} não encontrado. Execute na mesma pasta do run_v29.")
    sys.exit(1)

backup = TARGET.with_suffix(".py.bak_baseline_hybrid_unif")
shutil.copy(TARGET, backup)
print(f" Backup: {backup.name}")

content = TARGET.read_text(encoding="utf-8")

# PATCH 1 — evaluate_for_m: Baseline Hybrid após AF-KD Hybrid

OLD_P1 = '''\
    if hybrid_params is not None:
        tau_d, tau_h = hybrid_params
        te_h = infer_stream_hybrid(student, X_te, device,
                                    tau_delta=tau_d, tau_h=tau_h)
        rows.append({
            "Seed": seed, "m": m_detect,
            "Modelo": "LSTM-AF-KD",
            "Politica": f"Hibrida (tau={tau_d:.4f}, H={tau_h:.2f})",
            **summarize_eval(te_h, yep_te, yfr_te, thr_kd, theta_kd, m_detect,
                             progressive_mask=progressive_te_mask, abrupt_mask=abrupt_te_mask,
                             theta_ttd_adapt=theta_kd_adapt,
                             deriv_delta=deriv_kd_delta, deriv_prob_floor=deriv_kd_prob, deriv_smooth_w=deriv_kd_smooth),
        })
    elif run_hybrid:'''

NEW_P1 = '''\
    if hybrid_params is not None:
        tau_d, tau_h = hybrid_params

        # AF-KD Híbrida
        te_h = infer_stream_hybrid(student, X_te, device,
                                    tau_delta=tau_d, tau_h=tau_h)
        rows.append({
            "Seed": seed, "m": m_detect,
            "Modelo": "LSTM-AF-KD",
            "Politica": f"Hibrida (tau={tau_d:.4f}, H={tau_h:.2f})",
            **summarize_eval(te_h, yep_te, yfr_te, thr_kd, theta_kd, m_detect,
                             progressive_mask=progressive_te_mask, abrupt_mask=abrupt_te_mask,
                             theta_ttd_adapt=theta_kd_adapt,
                             deriv_delta=deriv_kd_delta, deriv_prob_floor=deriv_kd_prob, deriv_smooth_w=deriv_kd_smooth),
        })
        print(f"    [AF-KD Híbrida]  tau_d={tau_d:.4f} tau_h={tau_h:.2f} "
              f"FR={rows[-1].get('FailRate', float('nan')):.4f} "
              f"Skip={rows[-1].get('SkipPct', float('nan')):.1f}% "
              f"Cost={rows[-1].get('Cost_ms_per_frame', float('nan')):.4f} ms/q")

        # Baseline Híbrida (PATCH: mesmos tau_d, tau_h do AF-KD)
        # Racional: isola o efeito da supervisão temporal AF-KD no gating —
        # aplica o mesmo mecanismo MHEG sobre o baseline sem AF-KD e mede
        # quanto a cobertura degrada. Gera o ponto "Baseline Híbrida" no
        # gráfico de trade-off FR × custo e na Tabela de ablação fatorial.
        te_base_h = infer_stream_hybrid(baseline, X_te, device,
                                         tau_delta=tau_d, tau_h=tau_h)
        rows.append({
            "Seed": seed, "m": m_detect,
            "Modelo": "LSTM-Baseline",
            "Politica": f"Hibrida (tau={tau_d:.4f}, H={tau_h:.2f})",
            **summarize_eval(te_base_h, yep_te, yfr_te, thr_base, theta_base, m_detect,
                             progressive_mask=progressive_te_mask, abrupt_mask=abrupt_te_mask,
                             theta_ttd_adapt=theta_kd_adapt,
                             deriv_delta=deriv_kd_delta, deriv_prob_floor=deriv_kd_prob, deriv_smooth_w=deriv_kd_smooth),
        })
        print(f"    [Baseline Híbrida] tau_d={tau_d:.4f} tau_h={tau_h:.2f} "
              f"FR={rows[-1].get('FailRate', float('nan')):.4f} "
              f"Skip={rows[-1].get('SkipPct', float('nan')):.1f}% "
              f"Cost={rows[-1].get('Cost_ms_per_frame', float('nan')):.4f} ms/q")

    elif run_hybrid:'''

if OLD_P1 not in content:
    print("PATCH 1: trecho original não encontrado. Arquivo já foi modificado?")
    print("   Verifique se este é o run_v29 correto e sem patches anteriores aplicados.")
    TARGET.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    sys.exit(1)

content = content.replace(OLD_P1, NEW_P1, 1)
print(" PATCH 1: Baseline Híbrida adicionada em evaluate_for_m")

# PATCH 2 — export_latex_macros: adiciona BaseHibr ao mapping

OLD_P2 = '''\
    mapping = [
        ("BaseFixo", "LSTM-Baseline", "Fixa"),
        ("AlvoFixo", "LSTM-AF-KD",    "Fixa"),
        ("AlvoHibr", "LSTM-AF-KD",    None),   # None = política híbrida
    ]'''

NEW_P2 = '''\
    mapping = [
        ("BaseFixo", "LSTM-Baseline", "Fixa"),
        ("AlvoFixo", "LSTM-AF-KD",    "Fixa"),
        ("AlvoHibr", "LSTM-AF-KD",    None),   # None = política híbrida
        ("BaseHibr", "LSTM-Baseline", None),   # PATCH: Baseline com política híbrida
    ]'''

if OLD_P2 not in content:
    print("PATCH 2: mapping original não encontrado.")
    TARGET.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    sys.exit(1)

content = content.replace(OLD_P2, NEW_P2, 1)
print(" PATCH 2: BaseHibr adicionado ao mapping de export_latex_macros")

# PATCH 3 — main_summary: adiciona coluna N_Seeds por (Modelo, Politica)
# Insere contagem de seeds no main_summary para rastrear cobertura
# e gerar aviso explícito no console quando N_Seeds < 4.

OLD_P3 = '''\
    main_summary = aggregate_mean_std(main_df, ["Modelo", "Politica"], present)'''

NEW_P3 = '''\
    main_summary = aggregate_mean_std(main_df, ["Modelo", "Politica"], present)

    # PATCH: adiciona N_Seeds para rastrear cobertura de seeds por configuração
    _pol_norm = main_df["Politica"].apply(
        lambda s: "Hibrida" if isinstance(s, str) and s.startswith("Hibrida") else s
    )
    _seed_counts = (
        main_df.assign(_pol_norm=_pol_norm)
        .groupby(["Modelo", "_pol_norm"])["Seed"]
        .nunique()
        .reset_index(name="N_Seeds")
        .rename(columns={"_pol_norm": "Politica"})
    )
    main_summary = main_summary.merge(_seed_counts, on=["Modelo", "Politica"], how="left")

    # Aviso explícito quando alguma config tem < 4 seeds
    _incomplete = main_summary[main_summary["N_Seeds"].fillna(0) < 4]
    if not _incomplete.empty:
        print("\\nAVISO: configurações com cobertura incompleta de seeds:")
        for _, _row in _incomplete.iterrows():
            print(f"     {_row['Modelo']:20s} [{_row['Politica']:12s}]"
                  f"  -> {int(_row['N_Seeds'])}/4 seeds")
        print("   Execute run_hybrid_completion.py para completar as seeds faltantes.\\n")'''

if OLD_P3 not in content:
    print("aviso: PATCH 3: linha de main_summary não encontrada — pulando (não crítico).")
else:
    content = content.replace(OLD_P3, NEW_P3, 1)
    print(" PATCH 3: N_Seeds adicionado ao main_summary com aviso de cobertura")

# SALVAR
TARGET.write_text(content, encoding="utf-8")
print(f"\n Arquivo patched salvo: {TARGET.name}")

# RESUMO
print()
print("Macros geradas automaticamente na próxima execução do run_v29:")
macros = [
    ("\\resBaseHibrFUm",    "F1 médio do Baseline Híbrido"),
    ("\\stdBaseHibrFUm",    "Desvio F1 Baseline Híbrido"),
    ("\\resBaseHibrFail",   "FR médio do Baseline Híbrido"),
    ("\\stdBaseHibrFail",   "Desvio FR Baseline Híbrido"),
    ("\\resBaseHibrTTDef",  "TTDef médio do Baseline Híbrido"),
    ("\\stdBaseHibrTTDef",  "Desvio TTDef Baseline Híbrido"),
    ("\\resBaseHibrCost",   "Custo ms/q do Baseline Híbrido"),
    ("\\stdBaseHibrCost",   "Desvio custo Baseline Híbrido"),
    ("\\resBaseHibrSkip",   "Skip% do Baseline Híbrido"),
    ("\\stdBaseHibrSkip",   "Desvio Skip% Baseline Híbrido"),
]
for macro, desc in macros:
    print(f"  {macro:30s}  % {desc}")

print()
print("Uso no artigo (exemplo):")
print("  LSTM-Baseline & Sim  & $\\resBaseHibrFUm\\pm\\stdBaseHibrFUm$"
      "  & $\\resBaseHibrFail\\pm\\stdBaseHibrFail$  & ...")
print()
print("Fluxo recomendado para completar a matriz:")
print("  1) python patch_v29_baseline_hybrid_unificado.py  (este script)")
print("  2) python run_hybrid_completion.py --seeds 43 44 45")
print("     -> completa apenas as seeds faltantes sem retreinar")
print("  3) Conferir exp_abl_A_results_all_seeds_merged.csv")
print("  4) Conferir exp_abl_A_paper_metrics_macros_merged.tex")
