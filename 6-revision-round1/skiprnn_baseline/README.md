# Skip-RNN baseline (R3 C5) — Estratégia híbrida A-lite + B
Skip-RNN = published adaptive temporal-computation baseline (grupo iv).
NÃO é uncertainty-aware: a categoria (v) continua respondida pela
justificativa bibliográfica (nenhum método publicado diretamente
compatível com o setting single-node causal recurrent timestep-level).
Implementação fiel de Campos et al., ICLR 2018 (arXiv:1708.06834), adaptada ao
monitoramento contínuo (adaptações A1–A4 documentadas no cabeçalho do script).
Rodar na GPU: `python train_skiprnn.py` (4 seeds × 6 λ ≈ 4–8 h; resumível).
Saída: `results/skiprnn_grid.csv` + pontos de operação impressos.
Integração prevista: nova linha na Tabela 19 (safe point + matched point) e
atualização do item (v) da resposta ao R3 C5.
