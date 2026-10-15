# latex/ — Assets LaTeX gerados automaticamente pelo TERA Pipeline

## REGRA DE ORO
Nenhum arquivo neste diretório deve ser editado manualmente.
Para atualizar valores, re-execute o pipeline:
  python run_experiment.py --stages export

## Arquivos
- `paper_metrics_macros.tex` → todas as macros \resXXX e \stdXXX
- `paper_table2_rows.tex`    → linhas da Tabela 2 do paper
- `paper_table_stratified_rows.tex` → linhas da tabela estratificada
- `paper_table_seed_coverage.tex`   → tabela de cobertura de seeds

## Como usar no paper
```latex
% No preâmbulo ou antes de \begin{document}:
\input{../results/latest/latex/paper_metrics_macros.tex}
```

## Cobertura de seeds
- `baseline_fixed`: 12/4 seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
- `baseline_hybrid`: 12/4 seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
- `aftkd_fixed`: 12/4 seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
- `aftkd_hybrid`: 12/4 seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
