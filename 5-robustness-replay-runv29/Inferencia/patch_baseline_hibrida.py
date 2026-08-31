"""
patch_baseline_hibrida.py
Aplica DOIS patches no run_v29_ablacao_ttdef_ajuste_gatting.py:

PATCH 1 — evaluate_for_m (linha ~2239):
  Adiciona avaliação do LSTM-Baseline com política híbrida logo após
  a avaliação do LSTM-AF-KD híbrido, usando os mesmos tau_d e tau_h
  calibrados no VAL para o AF-KD.

  Racional: queremos mostrar o que acontece ao aplicar gating sobre
  o baseline SEM a melhoria de cobertura da AF-KD — esse é o ponto
  "Baseline Híbrida" na Fig. de trade-off.

PATCH 2 — export_latex_macros (linha ~1907):
  Adiciona ("BaseHibr", "LSTM-Baseline", None) ao mapping de macros,
  gerando automaticamente:
    \resBaseHibrFUm, \stdBaseHibrFUm
    \resBaseHibrECE, \stdBaseHibrECE
    \resBaseHibrFail, \stdBaseHibrFail
    \resBaseHibrTTD,  \stdBaseHibrTTD
    \resBaseHibrTTDef, \stdBaseHibrTTDef
    \resBaseHibrLat,  \stdBaseHibrLat
    \resBaseHibrCost, \stdBaseHibrCost
    \resBaseHibrSkip, \stdBaseHibrSkip
"""

from pathlib import Path
import shutil

TARGET = Path("run_v29_ablacao_ttdef_ajuste_gatting.py")

if not TARGET.exists():
    raise FileNotFoundError(f"{TARGET} não encontrado no diretório atual.")

# Backup
backup = TARGET.with_suffix(".py.bak_patch_baseline_hibrida")
shutil.copy(TARGET, backup)
print(f"Backup: {backup}")

content = TARGET.read_text(encoding="utf-8")

# PATCH 1 — adiciona avaliação Baseline Híbrida em evaluate_for_m
# Inserir logo após o bloco AF-KD Híbrida (linha ~2239)

OLD_P1 = """\
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
    elif run_hybrid:"""

NEW_P1 = """\
    if hybrid_params is not None:
        tau_d, tau_h = hybrid_params

        # AF-KD Híbrida (existente)
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

        # PATCH: Baseline Híbrida
        # Aplica os mesmos parâmetros de gating (tau_d, tau_h)
        # calibrados para o AF-KD sobre o baseline, usando thr_base
        # e theta_base para classificação e TTD. Isso gera o ponto
        # "Baseline Híbrida" na Fig. de trade-off (custo × segurança).
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
              f"Cost={rows[-1].get('Cost_ms_per_frame', float('nan')):.4f} ms/f")

    elif run_hybrid:"""

if OLD_P1 not in content:
    print("erro PATCH 1: trecho original NÃO encontrado.")
    print("   Verifique se o arquivo é a versão correta do run_v29.")
    raise SystemExit(1)

content = content.replace(OLD_P1, NEW_P1, 1)
print("ok PATCH 1 aplicado: avaliação Baseline Híbrida adicionada em evaluate_for_m")

# PATCH 2 — adiciona BaseHibr ao mapping de export_latex_macros

OLD_P2 = """\
    mapping = [
        ("BaseFixo", "LSTM-Baseline", "Fixa"),
        ("AlvoFixo", "LSTM-AF-KD",    "Fixa"),
        ("AlvoHibr", "LSTM-AF-KD",    None),   # None = política híbrida
    ]"""

NEW_P2 = """\
    mapping = [
        ("BaseFixo", "LSTM-Baseline", "Fixa"),
        ("AlvoFixo", "LSTM-AF-KD",    "Fixa"),
        ("AlvoHibr", "LSTM-AF-KD",    None),   # None = política híbrida
        # PATCH: Baseline com política híbrida -> ponto "Baseline Híbrida" no trade-off
        ("BaseHibr", "LSTM-Baseline", None),
    ]"""

if OLD_P2 not in content:
    print("erro PATCH 2: trecho original NÃO encontrado.")
    print("   Verifique se o arquivo é a versão correta do run_v29.")
    # Reverter
    TARGET.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    raise SystemExit(1)

content = content.replace(OLD_P2, NEW_P2, 1)
print("ok PATCH 2 aplicado: BaseHibr adicionado ao mapping de export_latex_macros")

# SALVAR
TARGET.write_text(content, encoding="utf-8")
print(f"\nok Arquivo salvo: {TARGET}")
print()
print("Macros que serão geradas automaticamente após a próxima execução:")
macros = [
    "resBaseHibrFUm",  "stdBaseHibrFUm",
    "resBaseHibrECE",  "stdBaseHibrECE",
    "resBaseHibrFail", "stdBaseHibrFail",
    "resBaseHibrTTD",  "stdBaseHibrTTD",
    "resBaseHibrTTDef","stdBaseHibrTTDef",
    "resBaseHibrLat",  "stdBaseHibrLat",
    "resBaseHibrCost", "stdBaseHibrCost",
    "resBaseHibrSkip", "stdBaseHibrSkip",
]
for m in macros:
    print(f"  \\{m}")

print()
print("Para usar na Fig. de trade-off:")
print("  (\\resBaseHibrCost, \\resBaseHibrFail)  [baseh]")
print()
print("Para usar no texto (se necessário):")
print("  FR do Baseline Híbrida: \\resBaseHibrFail")
print("  Custo:                  \\resBaseHibrCost ms/frame")
print("  Skip%:                  \\resBaseHibrSkip\\%")
