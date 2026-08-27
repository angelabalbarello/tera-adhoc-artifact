# Expansão de seeds 4 → 12 — passo a passo e justificativas

**Artigo:** "Trustworthy Selective Computation for Resource-Constrained Vehicular Edge Networks" (Ad Hoc Networks, ADHOC-D-26-02099, major revision, prazo 13/set/2026)
**Campanha:** `exp_seed12_round1`, executada em 27/08/2026 (11:18–14:37, 198 min, RTX 3050 Ti)
**Status:** validação de reprodução **PASS**; resultados prontos para decisão editorial. **Nenhuma tabela do artigo ou rebuttal foi alterada.**

---

## 1. Por que este experimento existe

O Revisor 2 (R2.5) observou que o estudo fatorial do artigo usa **4 seeds**, e com n=4 o
teste de Wilcoxon pareado tem p-valor mínimo de **0,0625** — impossível atingir o
limiar convencional de 0,05. O artigo já tratava isso honestamente (resultados
descritivos + Cliff's δ), mas a expansão para **n=12** dá ao teste resolução
suficiente (p mínimo ≈ 0,0005) para caracterizar a robustez do efeito à
inicialização.

**O que este experimento NÃO responde:** generalização temporal (durações de onset
diferentes, transições abruptas × graduais — R3.4). Essa é uma dimensão ortogonal:
mais seeds dão poder estatístico; não testam trajetórias de risco não vistas. A
decisão sobre um eventual *onset sweep* é separada e ficou pendente de decisão do
orientador. Igualmente, n=12 **não** altera a limitação de validade externa: a
evidência continua condicionada ao gerador TERA-Gen; nada aqui autoriza concluir
generalização para direção naturalística.

## 2. Protocolo congelado (anti-p-hacking)

Fixado **antes** de conhecer qualquer p-valor:

1. Seeds **42–53** (as 4 originais + 8 novas consecutivas). O n não seria aumentado
   se p desse 0,06, nem reduzido se desse significativo antes — *optional stopping*
   invalidaria o teste.
2. **Todas** as 12 seeds reportadas, individualmente, qualquer que fosse o desfecho.
3. Nenhum script altera artigo ou rebuttal automaticamente: as saídas são
   relatórios para decisão humana.
4. Integração condicionada a um portão de validação (Seção 4).

## 3. Por que o TERA Pipeline (e não o run_v29): a auditoria

A primeira tentativa de expansão (26–27/08) usava o `run_v29_ablacao_ttdef_ajuste_gatting.py`
de `FGCS/Inferencia`. A auditoria de 27/08 mostrou que **esse não é o pipeline que
gerou a Table 8**:

- A Table 8 publicada é byte-idêntica ao `paper_table2_rows.tex` da campanha
  **`exp_20260519_055950` do TERA Pipeline v1.0** (manifest com seeds 42–45,
  `config_hash d5327a24d1d2f7f9`, `generator_hash 65bbec4a23050639`); as médias por
  seed do `results_all_seeds.csv` dessa campanha reproduzem exatamente os valores
  publicados (F1 0,357/0,751; FR 0,040/0,004; TTDef 0,602/0,101).
- O run_v29 produz métricas em regime incompatível (F1 0,85/0,93; FR 0,233/0,0) —
  ele alimenta a *seção de robustez* (protocolo replay), não o fatorial principal.
- A execução do run_v29 foi **abortada** (só a seed 42 havia concluído) e o launcher
  antigo foi marcado obsoleto (`6-revision-round1/seed-expansion-deprecated-runv29/`).

Além disso, a cópia "viva" do `tera_pipeline` em `FGCS/Arquitetura` foi modificada
**depois** da campanha publicada (`tera_utils.py` em 25/06, `tera_figures.py` em
02/06). Por isso a expansão usa a **cópia congelada** de
`FGCS/Arquitetura/pipeline_completo/tera_pipeline` (estado ≤ 19/05/2026, dia da
campanha), replicada em `1-pipeline-tera/` deste repositório.

## 4. Passo a passo executado

### 4.1 Travas pré-execução (`run_seed_expansion_tera.py`)

Antes de treinar qualquer coisa, o launcher verifica e aborta em caso de mismatch:

| Trava | Valor exigido | Resultado |
|---|---|---|
| `generator_hash` (SHA-256/16 de `synthetic_driver_risk_v7.py`) | `65bbec4a23050639` | ✅ idêntico à campanha publicada |
| `config_hash` (SHA-256/16 do `experiment_config.yaml`) | `d5327a24d1d2f7f9` | ✅ idêntico |
| Determinismo | `cudnn.deterministic=True`, seeds globais por estágio | ✅ (código congelado) |

### 4.2 Os 8 estágios do pipeline (por seed, 42–53)

1. **DATASET** — TERA-Gen gera os episódios sintéticos e splits da seed
   (determinístico; as seeds 42–45 regeneram exatamente os dados de maio). *Por quê:*
   dados são função pura da seed, então não há necessidade de reaproveitar arquivos
   antigos — regenerar é a prova de reprodutibilidade.
2. **TRAINING** — treina teacher (BiLSTM), baseline (LSTM causal) e student AF-TKD
   por destilação. Foi o estágio caro (~15 min/seed): a perda AF-TKD congelada usa
   laços Python por quadro com sincronizações GPU↔CPU. **Não otimizamos de
   propósito** — qualquer mudança no código de treino quebraria a garantia de
   "pipeline literalmente idêntico ao da Table 8".
3. **CALIBRATION** — calibra os limiares (τ_d, H*) na partição de validação.
4. **INFERENCE** — roda as 4 configurações do fatorial (Baseline/AF-TKD ×
   Fixed/Hybrid-MHEG) em streaming causal no teste.
5. **EVALUATION** — computa F1, ECE, FR, TTD, TTDef, custo, %Sup por (seed, config)
   → `results_all_seeds.csv` canônico.
6. **FRAME LOGGING** — logs por quadro apenas da seed de visualização (42), para figuras.
7. **FIGURES** — figuras multiseed/traces/heatmap.
8. **EXPORT** — CSVs agregados + macros LaTeX + `manifest.json` (rastreabilidade:
   hashes, seeds, tempos, versão).

### 4.3 Portão de validação (`check_reproduction_42_45.py`)

**Pergunta:** as seeds 42–45 regeneradas hoje reproduzem a campanha que formou a
Table 8? Se não, as 12 seeds não seriam replicações homogêneas e nada poderia ser
integrado.

**Resultado:** **PASS** — as **256 métricas determinísticas** (F1, ECE, FR, TTD,
TTDef, %Sup, estratificações) reproduzem com diferença ≤ 1e-9 (bit a bit). As únicas
divergências (32 valores) estão em `Lat_ms`/`Cost_ms_per_frame`, que são **medições
de relógio de parede** — dependem da carga da máquina no momento e, por construção,
variam entre execuções (a máquina hoje rodava outras cargas). Consequência editorial:
a coluna ms/f do agregado n=12 deve usar apenas a campanha nova (medição homogênea
das 12 seeds na mesma execução), com nota de que a escala absoluta difere da medição
de maio.

### 4.4 Estatística (`stats_12seeds.py`)

Wilcoxon signed-rank pareado por seed (bicaudal, exato) + Cliff's δ, nos mesmos 4
contrastes do artigo, reportando n=4 (publicado) e n=12 lado a lado — mais o dump
das 12 observações individuais por configuração.

## 5. Resultados — n=4 (publicado) × n=12

### Médias ± dp (métricas determinísticas)

| Métrica | Config | n=4 (Table 8) | n=12 |
|---|---|---|---|
| F1 ↑ | Baseline fixed | 0,357 ± 0,096 | 0,367 ± 0,069 |
| F1 ↑ | AF-TOI fixed | 0,751 ± 0,275 | 0,846 ± 0,180 |
| FR ↓ | Baseline fixed | 0,040 ± 0,052 | 0,033 ± 0,054 |
| FR ↓ | AF-TOI fixed/hybrid | 0,004 ± 0,008 | 0,001 ± 0,005 |
| TTDef ↓ | Baseline fixed | 0,602 ± 0,608 | 0,500 ± 0,630 |
| TTDef ↓ | AF-TOI fixed | 0,086 ± 0,136 | 0,083 ± 0,124 |
| TTDef ↓ | AF-TOI hybrid (MHEG) | 0,101 ± 0,155 | 0,099 ± 0,142 |
| %Sup ↑ | Baseline hybrid | 61,5 ± 8,8 | 61,0 ± 7,6 |
| %Sup ↑ | AF-TOI hybrid | 48,7 ± 7,3 | 48,1 ± 5,0 |

**Leitura:** as magnitudes de efeito se mantiveram ou melhoraram; a variância caiu;
nenhuma seed nova inverteu o sentido do efeito (o AF-TOI supera a Baseline em F1 em
**12 de 12** seeds pareadas).

### Wilcoxon + Cliff's δ (contrastes principais)

| Contraste | Métrica | n=4: p / δ | n=12: p / δ |
|---|---|---|---|
| Baseline vs AF-TOI (fixed) | F1 | 0,125 / −0,875 | **0,0005** / −0,986 |
| | ECE | 0,125 / +1,000 | **0,0005** / +0,972 |
| | FR | 0,250 / +0,625 | 0,0625 / +0,361 |
| | TTDef | 0,125 / +0,875 | **0,021** / +0,569 |
| Baseline vs AF-TOI (hybrid/MHEG) | F1 | 0,125 / −0,875 | **0,0005** / −0,986 |
| | FR | 0,250 / +0,625 | **0,0156** / +0,521 |
| | TTDef | 0,125 / +0,875 | **0,021** / +0,653 |
| AF-TOI fixed vs AF-TOI hybrid | FR | 1,0 / 0,000 | 1,0 / 0,000 (preservação exata) |
| | TTDef | 0,125 / −0,500 | **0,001** / −0,160 |
| | custo | 0,125 / +1,000 | **0,0005** / +0,556 |
| Baseline fixed vs Baseline hybrid (controle negativo) | TTDef | 0,250 / −0,188 | **0,001** / −0,174 |
| | custo | 0,125 / +1,000 | **0,0005** / +0,806 |

**Leitura honesta (incluindo o que não fechou):**

- As afirmações centrais do artigo agora têm significância convencional: AF-TOI
  domina a Baseline em F1/ECE/TTDef (p ≤ 0,021), e o MHEG sobre AF-TOI reduz custo
  (p=0,0005) **preservando FR exatamente** (12/12 seeds sem alteração de misses).
- O controle negativo também fecha: MHEG sobre Baseline degrada TTDef (p=0,001) sem
  reduzir FR — a narrativa de "computação seletiva segura exige organização
  temporal" se sustenta com n=12.
- **FR no contraste fixed ficou em p=0,0625 (não significativo):** 9 das 12 seeds
  têm FR=0 também na Baseline, e o Wilcoxon descarta diferenças nulas — sobram
  poucos pares informativos. É um caso legítimo de "efeito no teto"; reportar como
  está (no contraste hybrid, FR fecha com p=0,0156).

## 6. Onde está cada saída

| Caminho (neste repositório) | Conteúdo |
|---|---|
| `1-pipeline-tera/results/exp_seed12_round1/manifest.json` | rastreabilidade da campanha (hashes, seeds, tempos) |
| `1-pipeline-tera/results/exp_seed12_round1/metrics/results_all_seeds.csv` | CSV canônico: 48 linhas (12 seeds × 4 configs) |
| `1-pipeline-tera/results/exp_seed12_round1/latex/` | macros e linhas LaTeX geradas pelo pipeline |
| `1-pipeline-tera/results/exp_seed12_round1/models/` | 36 checkpoints (teacher/baseline/student × 12 seeds) |
| `1-pipeline-tera/results/exp_seed12_round1/{calibration,figures,logs}/` | calibração por seed, figuras, logs de frames |
| `1-pipeline-tera/results/exp_seed12_round1/data/` | datasets/splits regenerados (**não versionado no git** — 151 MB regeneráveis deterministicamente pela seed) |
| `6-revision-round1/seed-expansion-tera/expanded_stats_report.txt` | Wilcoxon/Cliff n=4 × n=12 |
| `6-revision-round1/seed-expansion-tera/expanded_summary.csv` | médias ± dp por configuração (n=12) |
| `6-revision-round1/seed-expansion-tera/expanded_per_seed.csv` | as 12 observações individuais por configuração |
| `6-revision-round1/seed-expansion-tera/expanded_paper_table2_rows.tex` | linhas candidatas à Table 8 (**não aplicadas**) |
| `6-revision-round1/seed-expansion-tera/campaign_log.txt` | log completo da execução |

## 7. O que fica para decisão humana (orientador)

1. **Integrar n=12 ao artigo?** Se sim: Table 8 (linhas candidatas prontas), Tabela 9
   (novos p/δ), §4.4 "Statistical analysis" e resposta R2.5/R3.6 no rebuttal — com a
   nota sobre a coluna ms/f. Se não: manter n=4 e assumir a limitação (opção
   igualmente defensável).
2. **Onset sweep (R3.4):** experimento separado, só se o orientador julgar que a
   resposta atual a R3.4 é insuficiente.
3. Baselines confidence/smoothed-entropy (R3.5) já prontos em
   `6-revision-round1/baselines_novos/` — independentes desta campanha.
