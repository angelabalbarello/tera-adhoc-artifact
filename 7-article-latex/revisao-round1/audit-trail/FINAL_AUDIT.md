# FINAL_AUDIT — auditoria matemática e editorial de fechamento (27/09/2026)

Escopo restrito aos três itens solicitados. Nenhum resultado experimental foi alterado;
nenhum `REQUIRES RECOMPUTATION` novo foi criado.

## A. TTD_ef

```
TTD_ef AUDIT
Official implementation: AGREGADA — TTD_ef = mean(TTD sobre episódios DETECTADOS) + FR × T_ep, T_ep = 10.0 s
Mathematical definition: TTD_ef = TTD_det + FR × T_ep  (Eq. eq:ttdef_sys ≡ Eq. eq:ttd_efetivo, agora idênticas)
Code location:
  - 1-pipeline-tera/tera_pipeline/evaluation/tera_eval.py: compute_ttdef() L200-202
    (return ttd_det + fail_rate * t_max; T_MAX_EF=10.0 L28; ttd_det = média sobre
    detectados L398-401; usada nas Tabelas 8/9/10/12 via run_seed_expansion_tera)
  - 3-calibration-posthoc/tera_posthoc_calibration_experiment.py: compute_ttdef() L194-199
    (docstring "TTDef = TTD_det + FR × T_ep. Fonte: tera_eval"; Tabela 14)
  - 5-robustness-replay-runv29/Inferencia/run_v29_ablacao_ttdef_ajuste_gatting.py L635 e L1203
    (ttd + fr*T_MAX_EF; alimenta a Tabela 11 via run_mechanism_ablation.py)
  - 6-revision-round1/baselines_novos/run_gating_baselines_v2.py L96 e run_refresh_gate_v2.py L59
    ((ttd se detectado, senão 0) + fr*T_EP; Tabela 19)
  - 6-revision-round1/skiprnn_baseline/train_skiprnn.py L107 (ttd + fr*T_EP; Tabela 20;
    compare_skiprnn_vs_mheg.py docstring L13 declara a mesma convenção)
  - 6-revision-round1/temporal_scale_generalization_v2/common.py: ttd_metrics_binary() L55-68
    (comentário L65 declara "= Eq. do artigo"; retorna ttd + fr*10.0; Tabela 21/§4.12)
Tables affected: 8, 9, 10, 11, 12, 14, 19, 20, 21 — TODAS usam a convenção agregada.
Numerical recomputation required: NO
```

Verificação numérica decisiva (todas as seeds com FR>0, 4 configurações):
`expanded_per_seed.csv` (valores publicados) reproduz `mean(det) + FR×10` com erro ≤4×10⁻⁴
(arredondamento), enquanto a média por episódio com T_ep imputado, `(1−FR)·TTD_det + FR·T_ep`,
diverge sistematicamente (ex.: baseline_fixed seed 53: publicado 1.9134; agregada 1.9130;
por-episódio 1.8611). Isso fecha a questão pelos dados, não por interpretação.

A formulação por casos (antiga Eq. `eq:ttd_efetivo`, §4.1) NÃO correspondia à estatística
reportada: se lida como definição por episódio com média posterior, implicaria
`(1−FR)·TTD_det + FR·T_ep`. Duas implementações por episódio existem no repositório, mas
**nenhuma alimenta números publicados**: `analise_robustez_afkd_fgcs_v2.py` (análise FGCS
legada, não citada em metric_provenance.csv) e `df_episodios_12seeds.csv` (T_ep imputado
por episódio), usado apenas para as DISTRIBUIÇÕES das Figs. 8–9, cujas legendas já marcam
a barra de penalidade.

**Correção aplicada (texto, não números):** a Eq. `eq:ttd_efetivo` foi reescrita como a
forma agregada (idêntica à Eq. `eq:ttdef_sys`, agora declarada como a definição única de
todas as tabelas), com uma nota explicando que as Figs. 8–9 exibem valores por episódio
com misses em T_ep e que a média desses valores exibidos NÃO é a quantidade tabulada
(difere por FR·TTD_det). O rótulo `eq:ttd_efetivo` foi preservado (a referência da
Discussion continua válida). O rebuttal não reproduzia a forma por casos — sem espelho a corrigir.

## B. Statistical language

Busca global (artigo, rebuttal, robustness_section): `statistical power`, `statistical-power`,
`adequate power`, `power weakness`, `powered`, `underpowered`, `well-powered`.

Ocorrências encontradas: **1** — rebuttal.tex, resposta ao R2 (recomendação geral):
"In addition, the statistical-power weakness itself was addressed by expanding the factorial
study to twelve seeds under an a priori protocol..."

Substituída por: "In addition, the limited inferential resolution of the original four-seed
campaign, identified by the reviewer, was addressed by expanding the factorial study to twelve
seeds under the prespecified protocol described in our response to Comment~5, and the main
contrasts now reach conventional significance."

Usos restantes de "power" são todos não estatísticos e legítimos: "low-power
microcontroller/accelerators" (hardware, 2×) e "discriminative/discrimination power" do sinal
entrópico (3×). O artigo já usava "limited inferential resolution" desde a passada anterior (§4.4).

## C. Abstract word count

- Contagem real: **227 palavras**.
- Método: `pdftotext` sobre o PDF compilado (artigo_v2.pdf), extração do bloco do abstract
  ("Vehicular edge nodes..." até "...vehicular edge coordination."), tokens separados por
  espaço contendo ao menos um caractere alfanumérico. Reproduzível; não estimado manualmente.
  (Contagens alternativas, para registro: fonte LaTeX com cada expressão matemática valendo
  1 palavra = 202; a discrepância anterior 224 vs 228 vinha de convenções distintas em
  momentos distintos do texto.)
- Ocorrências corrigidas: rebuttal R2-C1 ("350 to 228" → "350 to 227 words (word count on the
  compiled text)"); rebuttal R2 recomendação ("350 to 224" → "350 to 227"); CHANGELOG_FINAL.md
  (228 → 227). O artigo não declara a contagem em texto. O checker automático valida a faixa
  195–230 (passa).

## Regressão

`test_manuscript_claims.py` ganhou `test_final_audit_locks()`: proíbe a forma por casos
("otherwise (missed episode)") e qualquer "statistical power/statistical-power weakness" em
ambos os documentos; exige "single definition used by every table" no artigo e exatamente
duas ocorrências de "227 words" no rebuttal. Suíte completa: **0 FAIL / 40 OK + 8/8 testes**.
Builds: artigo_v2.pdf 36 pp, 0 refs indefinidas; rebuttal_v2.pdf 18 pp, 0 erros, 0 overfull.

Entregáveis: `artigo_final_v2.tex`, `rebuttal_final_v2.tex`, `FINAL_AUDIT.md`
(canônicos `artigo.tex`/`rebuttal.tex` idênticos às cópias _v2).

## D. Final verdict

**READY FOR RESUBMISSION** — a definição de TTD_ef, o código de todas as campanhas
publicadas e todos os números tabulados são matematicamente coerentes sob a convenção
agregada única, verificada numericamente contra os CSVs por seed; os dois itens editoriais
foram fechados e travados por teste. Condição operacional (fora do escopo científico):
renomear os builds _v2 e subir os .tex finais ao Overleaf/EM antes do envio.
