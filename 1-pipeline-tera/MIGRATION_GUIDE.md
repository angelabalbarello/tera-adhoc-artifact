# Guia de Migração — Scripts Legados → TERA Pipeline v1.0

## Resumo Executivo

O TERA Pipeline v1.0 consolida todos os scripts experimentais em um único
framework determinístico e reproduzível. Esta migração **não altera** hipótese
científica, dataset, modelos, métricas ou narrativa do artigo — apenas
industrializa o pipeline experimental.

---

## Mapeamento: Scripts Legados → Novos Módulos

| Script legado | Status | Substituído por |
|---|---|---|
| `run_v29_ablacao_ttdef_ajuste_gatting.py` | **ABSORVIDO** | `run_experiment.py` + módulos |
| `synthetic_driver_risk_v7.py` | **MANTIDO** | `tera_pipeline/dataset/tera_gen.py` (wrapper) |
| `generate_episode_frame_logs.py` | **ABSORVIDO** | `tera_pipeline/logging/tera_log.py` |
| `generate_borderline_frame_logs.py` | **ABSORVIDO** | `tera_pipeline/logging/tera_log.py` |
| `run_inferencia_seed42_corrigido.py` | **OBSOLETO** | `tera_pipeline/logging/tera_log.py` |
| `patch_baseline_hibrida.py` | **OBSOLETO** | `tera_pipeline/inference/tera_infer.py` (nativo) |
| `patch_ttdef_estratificado.py` | **OBSOLETO** | `tera_pipeline/evaluation/tera_eval.py` (nativo) |
| `patch_v29_ttd_paper.py` | **OBSOLETO** | `tera_pipeline/evaluation/tera_eval.py` (nativo) |
| `patch_v29_ttd_paper_corrigido.py` | **OBSOLETO** | `tera_pipeline/evaluation/tera_eval.py` (nativo) |
| `run_hybrid_completion.py` | **OBSOLETO** | Pipeline completo desde o início |
| `script.py`, `script_2.py`, `script_3.py` | **OBSOLETOS** | `tera_pipeline/evaluation/tera_eval.py` |
| `fracos.py`, `script_ver_final.py` | **OBSOLETOS** | `tera_pipeline/evaluation/tera_eval.py` |

---

## Etapas de Migração

### Passo 1 — Organizar diretório

```bash
# Criar estrutura de diretórios
mkdir -p tera_pipeline/{dataset,training,calibration,inference,evaluation,logging,figures,export,latex,utils}
mkdir -p tera_pipeline/{dataset,training,calibration,inference,evaluation,logging,figures,export,latex,utils}/__init__.py
mkdir -p configs results

# Mover scripts legados para _legacy/
mkdir -p _legacy
mv patch_*.py run_hybrid_completion.py run_inferencia_*.py _legacy/
mv script*.py fracos.py _legacy/

# Manter como referência (não deletar ainda)
mv run_v29_ablacao_ttdef_ajuste_gatting.py _legacy/run_v29_legacy.py
```

### Passo 2 — Copiar novos módulos

```bash
cp run_experiment.py .
cp experiment_config.yaml configs/
cp tera_eval.py   tera_pipeline/evaluation/
cp tera_infer.py  tera_pipeline/inference/
cp tera_latex.py  tera_pipeline/latex/
cp tera_utils.py  tera_pipeline/utils/
```

### Passo 3 — Copiar wrappers dos módulos restantes

Os módulos `tera_train.py`, `tera_calibrate.py`, `tera_gen.py`, `tera_log.py`,
`tera_figures.py` e `tera_export.py` são wrappers que importam a lógica
correspondente do `run_v29_legacy.py` durante a transição, gradualmente
refatorando cada função.

Template de wrapper:

```python
# tera_pipeline/training/tera_train.py
from _legacy.run_v29_legacy import (
    MultiTaskLSTM,
    train_teacher,
    train_baseline,
    train_afkd,
)
# ... implementação completa conforme ARCHITECTURE.md
```

### Passo 4 — Verificar modelos existentes

Os modelos em `modelos_salvos/abl_A/` são compatíveis com o novo pipeline.
Nenhum retreinamento é necessário se os checkpoints já existem.

```python
# Verificar existência
python -c "
from pathlib import Path
seeds = [42, 43, 44, 45]
roles = ['baseline', 'student', 'teacher']
for seed in seeds:
    for role in roles:
        p = Path(f'modelos_salvos/abl_A/{role}_seed{seed}.pt')
        status = '✓' if p.exists() else '✗ FALTANDO'
        print(f'  {status}  {p.name}')
"
```

### Passo 5 — Gerar resultados completos

```bash
# Opção A: Pipeline completo (retreina tudo)
python run_experiment.py

# Opção B: Usando modelos existentes (recomendado)
python run_experiment.py --reuse-data --reuse-models

# Opção C: Apenas avaliação + export (modelos e datasets já existem)
python run_experiment.py --stages calibration,inference,evaluation,export
```

### Passo 6 — Atualizar o artigo LaTeX

```latex
% Substituir o bloco de \newcommand manual por:
\input{../results/latest/latex/paper_metrics_macros.tex}

% As tabelas podem usar os arquivos gerados automaticamente:
\input{../results/latest/latex/paper_table2_rows.tex}
\input{../results/latest/latex/paper_table_stratified_rows.tex}
```

### Passo 7 — Verificar consistência

```bash
python -c "
from tera_pipeline.utils.tera_utils import verify_seed_coverage
from pathlib import Path
ok = verify_seed_coverage(Path('results/latest/metrics/results_all_seeds.csv'))
print('Pipeline completo!' if ok else 'Seeds faltantes — verifique o log')
"
```

---

## Mudanças no Artigo

### 1. Adicionar claim de reprodutibilidade (§4.1)

```latex
All numerical results reported in this paper were generated automatically
by the TERA Pipeline (v1.0), a single deterministic evaluation framework.
The pipeline executes from dataset generation through \LaTeX{} macro
production via a single entry point (\texttt{run\_experiment.py}) with
no manual post-processing. Experiment manifests and the canonical
results CSV are archived alongside the manuscript.
```

### 2. Tabela de cobertura de seeds

Inserir `\input{../results/latest/latex/paper_table_seed_coverage.tex}`
na seção experimental para documentar explicitamente que todas as
configurações foram avaliadas com as quatro seeds.

### 3. Nomenclatura padronizada

| Usar | Evitar |
|---|---|
| `LSTM-AF-TKD` | `LSTM-AF-KD` (inconsistente) |
| `baseline_hybrid` | `Baseline Híbrida` (usar inglês técnico) |
| `aftkd_hybrid` | `AF-KD+Gating` |
| `MHEG` | `gating híbrido` (manter sigla) |

---

## Verificação de Integridade Pós-Migração

```bash
# 1. Todos os modelos presentes
python run_experiment.py --dry-run

# 2. Seeds simétricas
python -c "
import pandas as pd
df = pd.read_csv('results/latest/metrics/results_all_seeds.csv')
for cfg, g in df.groupby('config_id'):
    print(f'{cfg}: seeds={sorted(g.Seed.unique().tolist())}')
"

# 3. Macros consistentes com CSV
python -c "
import re
from pathlib import Path
macros = Path('results/latest/latex/paper_metrics_macros.tex').read_text()
# Verifica que não há '--' (valores faltantes)
missing = re.findall(r'newcommand\{[^}]+\}\{--\}', macros)
if missing:
    print('⚠️  Macros faltantes:', missing)
else:
    print('✓ Todas as macros preenchidas')
"

# 4. Manifesto gerado
python -c "
import json
from pathlib import Path
m = json.loads(Path('results/latest/manifest.json').read_text())
print('Experimento:', m['experiment_id'])
print('Estágios:', m['stages_completed'])
print('Seeds:', m['seeds'])
"
```

---

## O que NÃO Muda

- Hipótese científica
- Dataset sintético (TERA-Gen v7)
- Arquitetura dos modelos (LSTM, BiLSTM)
- Protocolo AF-TKD (ablação A: M1+M3, temperatura fixa 2.0)
- Mecanismo MHEG (gatilho cinemático + entrópico)
- Métrica TTDef (formula canônica do artigo)
- Resultados numéricos (mesmas seeds, mesmos dados)
- Narrativa e claims científicos do artigo

---

## Dúvidas Frequentes

**Q: Preciso retreinar os modelos?**
A: Não. Os checkpoints em `modelos_salvos/abl_A/` são compatíveis.
Use `--reuse-models` para pular o treinamento.

**Q: Os resultados numéricos vão mudar?**
A: Não, desde que as mesmas seeds e configurações sejam usadas.
O `manifest.json` registra o hash do gerador para verificar isso.

**Q: A Baseline Hybrid agora usa exatamente os mesmos tau_delta/tau_H?**
A: Sim. O `InferenceManager` calibra os parâmetros de gating no VAL do
AF-TKD e usa os MESMOS parâmetros para a Baseline Hybrid. Isso é
metodologicamente correto: isola o efeito da supervisão temporal, não
do gating em si.

**Q: Como atualizo as macros depois de um novo run?**
A: Execute `python run_experiment.py --stages export`. O pipeline
recalcula tudo a partir do CSV. O arquivo `paper_metrics_macros.tex`
é sobrescrito automaticamente.
