# TERA Pipeline — Arquitetura Científica Completa
## Framework Experimental Reproduzível para FGCS/Elsevier

---

## 1. Visão Geral

O TERA Pipeline transforma o conjunto de scripts experimentais iterativos em um
**framework experimental científico determinístico, reproduzível e auditável**,
compatível com os requisitos de periódicos A1/Elsevier como FGCS.

**Princípio central:**
> *Todos os resultados numéricos do artigo são gerados automaticamente por um
> único pipeline determinístico a partir do CSV canônico de resultados.*

**Invariante de reprodutibilidade:**
```
dataset(seed) + modelo(seed) + config → resultados(seed, config)
resultados_all_seeds.csv → macros.tex → paper.tex
```

---

## 2. Estrutura de Diretórios

```
tera_pipeline/
│
├── run_experiment.py            ← PONTO DE ENTRADA ÚNICO
├── configs/
│   └── experiment_config.yaml  ← FONTE DA VERDADE DE CONFIGURAÇÃO
│
├── tera_pipeline/               ← Pacote Python principal
│   ├── __init__.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── tera_gen.py          ← Geração + versionamento de dataset
│   ├── training/
│   │   ├── __init__.py
│   │   └── tera_train.py        ← Teacher + Baseline + AF-TKD student
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── tera_calibrate.py    ← Thresholds + gating params por seed
│   ├── inference/
│   │   ├── __init__.py
│   │   └── tera_infer.py        ← 4 configs × 4 seeds (Baseline Hybrid nativo)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── tera_eval.py         ← Engine de métricas canônicas
│   ├── logging/
│   │   ├── __init__.py
│   │   └── tera_log.py          ← Frame logs + borderline logs
│   ├── figures/
│   │   ├── __init__.py
│   │   └── tera_figures.py      ← Geração automática de todas as figuras
│   ├── export/
│   │   ├── __init__.py
│   │   └── tera_export.py       ← CSV canônico + arquivos de resultados
│   ├── latex/
│   │   ├── __init__.py
│   │   └── tera_latex.py        ← Macros + tabelas LaTeX automáticas
│   └── utils/
│       ├── __init__.py
│       └── tera_utils.py        ← Determinismo + I/O + logging científico
│
├── results/                     ← Outputs por execução (git-ignored)
│   └── exp_{YYYYMMDD_HHMMSS}/
│       ├── manifest.json        ← Rastreabilidade completa
│       ├── data/                ← Datasets + splits
│       ├── models/              ← Checkpoints organizados
│       ├── calibration/         ← Parâmetros calibrados
│       ├── metrics/             ← CSVs de resultados
│       ├── logs/                ← Frame logs
│       ├── figures/             ← Figuras geradas
│       └── latex/               ← Macros + tabelas
│
├── paper/                       ← Artigo LaTeX
│   └── artigo.tex               ← Usa \input{../results/latest/latex/macros.tex}
│
└── MIGRATION_GUIDE.md
```

---

## 3. Configurações do Fatorial 2×2

| Config ID | Modelo | Gating | Seeds | Papel |
|---|---|---|---|---|
| `baseline_fixed` | LSTM-Baseline | Não | 42–45 | Referência |
| `baseline_hybrid` | LSTM-Baseline | Sim | 42–45 | Controle negativo |
| `aftkd_fixed` | LSTM-AF-TKD | Não | 42–45 | Supervisão temporal isolada |
| `aftkd_hybrid` | LSTM-AF-TKD | Sim | 42–45 | Ecossistema completo |

**Regra invariante:** Toda configuração usa exatamente `SEEDS = [42, 43, 44, 45]`.
Qualquer desvio disso é uma falha de pipeline, não uma decisão metodológica.

---

## 4. Fluxo de Execução

```
run_experiment.py
│
├── Stage 1: DATASET
│   ├── Para cada seed ∈ [42,43,44,45]:
│   │   ├── Verificar hash do gerador (synthetic_driver_risk_v7.py)
│   │   ├── Gerar/reutilizar dataset_sintetico_seed{N}.npz
│   │   ├── Gerar/reutilizar splits determinísticos
│   │   └── Exportar dataset_metadata_seed{N}.json
│   └── Verificar consistência entre seeds
│
├── Stage 2: TRAINING
│   ├── Para cada seed ∈ [42,43,44,45]:
│   │   ├── Treinar BiLSTM teacher (60 épocas)
│   │   ├── Treinar LSTM-Baseline (60 épocas)
│   │   └── Treinar LSTM-AF-TKD student (80 épocas, protocolo 3 fases)
│   └── Salvar checkpoints em models/{seed}/
│
├── Stage 3: CALIBRATION
│   ├── Para cada seed ∈ [42,43,44,45]:
│   │   ├── Calibrar thr_ep (threshold episódico)
│   │   ├── Calibrar theta_ttd (threshold de detecção)
│   │   └── Calibrar (tau_delta, tau_H) via grade 23×13 no VAL
│   └── Salvar calibration_summary.csv
│
├── Stage 4: INFERENCE
│   ├── Para cada seed ∈ [42,43,44,45]:
│   │   ├── baseline_fixed  → StreamEval
│   │   ├── baseline_hybrid → StreamEval  (NATIVO, sem patch)
│   │   ├── aftkd_fixed     → StreamEval
│   │   └── aftkd_hybrid    → StreamEval
│   └── Salvar frame_probs_*.npy por seed×config
│
├── Stage 5: EVALUATION
│   ├── Para cada (seed, config):
│   │   ├── F1, ECE, FailRate
│   │   ├── TTD_bruto, TTDef
│   │   ├── SkipPct, Cost_ms_per_frame
│   │   └── Estratificado: progressivo vs abrupto
│   └── Salvar results_all_seeds.csv (FONTE DA VERDADE)
│
├── Stage 6: FRAME LOGGING  [SEED_VIZ = 42, apenas figuras]
│   ├── Gerar episode_frame_logs.csv
│   └── Gerar borderline_frame_logs.csv
│
├── Stage 7: FIGURES
│   ├── fig_temporal_trace.pdf
│   ├── fig_entropy_dist.pdf
│   ├── fig_gating_heatmap.pdf
│   ├── fig_tradeoff.pdf
│   ├── fig_multiseed.pdf
│   └── fig_prob_dist.pdf
│
└── Stage 8: EXPORT
    ├── Agregar results_all_seeds.csv → summary_by_config.csv
    ├── Gerar paper_metrics_macros.tex  (CSV → macros, NUNCA manual)
    ├── Gerar paper_table2_rows.tex
    ├── Gerar paper_table_stratified_rows.tex
    └── Escrever manifest.json (rastreabilidade completa)
```

---

## 5. Separação Offline vs Online

### OFFLINE (servidor de treinamento)
```
TERA-Gen     → gera episódios com onset determinístico
BiLSTM       → professor bidirecional (acesso futuro)
AF-TKD       → protocolo de destilação (molda trajetória temporal)
Calibração   → grade de limiares no conjunto de validação
```

### ONLINE (dispositivo embarcado)
```
LSTM causal  → aluno (25K params, 98KB FP32)
MHEG         → gating O(1) por frame
hidden state → 512 bytes, buffer fixo
limiares     → 5 escalares exportados do treinamento
```

**Invariante:** Nenhum componente offline reside no dispositivo.
Esta separação é verificada programaticamente no Stage 4.

---

## 6. Source of Truth Único

```
results_all_seeds.csv
        │
        ▼
tera_latex.py::generate_macros()
        │
        ▼
paper_metrics_macros.tex
        │
        ▼  \input{../results/latest/latex/macros.tex}
artigo.tex
```

**Regra:** O arquivo `paper_metrics_macros.tex` é SOMENTE LEITURA para humanos.
Qualquer macro com valor diferente do CSV é um bug de pipeline.

---

## 7. Rastreabilidade Experimental

Todo experimento gera um `manifest.json`:

```json
{
  "experiment_id": "exp_20250517_143022",
  "timestamp": "2025-05-17T14:30:22Z",
  "git_commit": "abc1234",
  "generator_hash": "sha256:...",
  "config_hash": "sha256:...",
  "seeds": [42, 43, 44, 45],
  "configs": ["baseline_fixed", "baseline_hybrid", "aftkd_fixed", "aftkd_hybrid"],
  "stages_completed": ["dataset", "training", "calibration", "inference",
                       "evaluation", "logging", "figures", "export"],
  "results_csv": "metrics/results_all_seeds.csv",
  "macros_tex": "latex/paper_metrics_macros.tex",
  "paper_claim": "All results generated by TERA Pipeline v1.0, exp_20250517_143022"
}
```

---

## 8. Nomenclatura Padronizada

| Antigo (problemático) | Novo (científico) |
|---|---|
| `LSTM-AF-KD` | `LSTM-AF-TKD` |
| `af-kd_hybrid` | `aftkd_hybrid` |
| `patch_baseline_hibrida.py` | Incorporado em `tera_infer.py` |
| `run_hybrid_completion.py` | Eliminado — pipeline sempre completo |
| `run_inferencia_seed42_corrigido.py` | `tera_log.py::generate_frame_logs(seeds=[42])` |
| `script.py`, `fracos.py` | `tera_eval.py::borderline_analysis()` |

---

## 9. Determinismo

```python
# tera_utils.py::set_global_seed(seed)
import random, numpy as np, torch, os

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
```

---

## 10. Scripts Obsoletos (a arquivar em `_legacy/`)

| Script | Razão de obsolescência | Substituído por |
|---|---|---|
| `patch_baseline_hibrida.py` | Lógica incorporada nativamente | `tera_infer.py` |
| `patch_ttdef_estratificado.py` | Lógica incorporada nativamente | `tera_eval.py` |
| `patch_v29_ttd_paper.py` | Fix incorporado na versão canônica | `tera_eval.py` |
| `patch_v29_ttd_paper_corrigido.py` | Idem | `tera_eval.py` |
| `run_hybrid_completion.py` | Pipeline sempre completo | `run_experiment.py` |
| `run_inferencia_seed42_corrigido.py` | Frame log via pipeline | `tera_log.py` |
| `script.py`, `script_2.py`, `script_3.py` | Análises ad hoc | `tera_eval.py` |
| `fracos.py`, `script_ver_final.py` | Análises ad hoc | `tera_eval.py` |

---

## 11. Novos Módulos

| Módulo | Responsabilidade |
|---|---|
| `run_experiment.py` | Orquestrador único, entry point |
| `tera_pipeline/dataset/tera_gen.py` | Geração + versionamento |
| `tera_pipeline/training/tera_train.py` | Teacher + Baseline + AF-TKD |
| `tera_pipeline/calibration/tera_calibrate.py` | Grade de calibração |
| `tera_pipeline/inference/tera_infer.py` | 4 configs nativas |
| `tera_pipeline/evaluation/tera_eval.py` | Métricas canônicas |
| `tera_pipeline/logging/tera_log.py` | Frame + borderline logs |
| `tera_pipeline/figures/tera_figures.py` | Figuras automáticas |
| `tera_pipeline/export/tera_export.py` | CSVs + manifesto |
| `tera_pipeline/latex/tera_latex.py` | Macros + tabelas LaTeX |
| `tera_pipeline/utils/tera_utils.py` | Determinismo + I/O |

---

## 12. Claim de Reprodutibilidade para o Paper

Inserir no §4.1 (Protocolo Experimental):

> *All numerical results reported in this paper were generated automatically
> by the TERA Pipeline (v1.0), a single deterministic evaluation framework.
> The complete pipeline — from dataset generation through metric aggregation
> and \LaTeX{} macro production — is executed via a single entry point
> (\texttt{run\_experiment.py}) with no manual post-processing steps.
> Experiment manifests, calibration logs, and the canonical results CSV
> are archived alongside the manuscript for full reproducibility.*

---

## 13. Tabela de Cobertura de Seeds (para o paper)

```latex
\begin{table}
\caption{Cobertura de sementes por configuração experimental.}
\begin{tabular}{lcccc}
\toprule
Configuração & Seed 42 & Seed 43 & Seed 44 & Seed 45 \\
\midrule
Baseline Fixed   & \ding{51} & \ding{51} & \ding{51} & \ding{51} \\
Baseline Hybrid  & \ding{51} & \ding{51} & \ding{51} & \ding{51} \\
AF-TKD Fixed     & \ding{51} & \ding{51} & \ding{51} & \ding{51} \\
AF-TKD Hybrid    & \ding{51} & \ding{51} & \ding{51} & \ding{51} \\
\bottomrule
\end{tabular}
\end{table}
```
