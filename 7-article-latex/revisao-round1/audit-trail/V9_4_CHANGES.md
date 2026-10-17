# v9.3 -> v9.4 (17/10/2026): limitacao operacional do MHEG divulgada

Revisao minima, sem experimentos novos e sem alteracao de metodo, checkpoints,
limiares, tabelas ou resultados. Locais (todos em \revcolor):
- artigo §2.2: cobertura delimitada as condicoes da campanha principal, com o
  aviso cinematico disponivel ao gate (perda do aviso -> §5);
- artigo §5: quinta limitacao — diagnostico exploratorio (pesos/limiares
  congelados, 12 seeds, 2 configs): kin do gate zerado com entradas da LSTM
  intactas => supressao persistente, FR=1,0; sem reativacao independente do
  aviso; frase de disponibilidade do material no artefato;
- robustness §4.10: a protecao do replay (entropia exogena) nao demonstra
  protecao da implementacao implantada na perda do aviso (-> §5);
- rebuttal R3 C4(v)/C7/C10 atualizados e citacoes sincronizadas; C3 locator.
Builds: artigo_v9_4.pdf (37 pp), rebuttal_v9_4.pdf (25 pp), 0 indefinidas;
suite de travas do manuscrito OK. Diffs: DIFF_v9_3_to_v9_4_{artigo(23),
robustness_section(4),rebuttal(10)} linhas alteradas.
Evidencia correspondente: 6-revision-round1/incremental-campaign/
diagnostic-windowed-S2/ (+ zip) e insumos em 1-pipeline-tera/results/
exp_seed12_round1/ (models/, data/, calibration/).
