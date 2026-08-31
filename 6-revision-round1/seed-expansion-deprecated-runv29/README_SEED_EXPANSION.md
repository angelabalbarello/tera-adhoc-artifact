# Expansão de seeds 4 → 12 (revisão Ad Hoc Networks, ADHOC-D-26-02099)

Responde à crítica de significância estatística (R2.5 e R3): com n=12 seeds,
o teste de Wilcoxon pareado passa a ter p mínimo de 0,0005 (vs. piso de
0,0625 com n=4), permitindo significância convencional.

## Passo a passo (na SUA máquina, com GPU)

1. Abra um terminal e vá para a pasta do pipeline original (tem os
   checkpoints das seeds 42–45 em `modelos_salvos/abl_A`):

       cd C:\Users\Angela\Documents\DATALAKE-ANGELA\MLProject\TreinamentoNovo\FGCS\Inferencia

2. Rode a expansão (42–45 só re-inferem; 46–53 treinam do zero):

       python ..\..\Experimentos-Artigo1-AdHoc\seed-expansion\run_seed_expansion.py

   Tempo estimado na RTX 3050 Ti: 2–4 h. Pode interromper e rodar de novo —
   seeds com modelo salvo são reutilizadas.

3. Consolide e gere a estatística:

       copy ..\..\Experimentos-Artigo1-AdHoc\seed-expansion\merge_and_stats.py .
       python merge_and_stats.py

## Saídas a me enviar de volta (para eu integrar no artigo e no rebuttal)

- `expanded_summary.csv`
- `expanded_paper_table2_rows.tex`  (substitui paper_table2_rows.tex da Tabela 8)
- `expanded_stats_report.txt`       (novos p-valores e Cliff's delta)
- `exp_abl_A_results_seed46.csv` … `seed53.csv` (por segurança)

## O que será atualizado no artigo com esses números (eu faço)

- Tabela 8 (linhas), Tabela 9 (Wilcoxon/Cliff), parágrafo "Statistical
  analysis" da Seção 4.4, limitação 3 da Seção 5, e a resposta R2.5/R3.6
  no rebuttal — tudo em vermelho.

## Observações

- O pipeline regenera datasets/splits para as seeds novas automaticamente
  (gerador synthetic_driver_risk_v7, mesmo do artigo).
- Config oficial: ABLATION_MODE="A" (a mesma dos números publicados).
- Se faltar algum pacote: pip install torch pandas scikit-learn matplotlib scipy
