# FINAL_MICRO_AUDIT — correção final estritamente pontual (27/09/2026)

## Abstract count

- **Contagem oficial: 227 palavras.**
- Método: `pdftotext` sobre o PDF compilado atual (`artigo_v3.pdf`), bloco do abstract
  localizado pelas 4 primeiras e 4 últimas palavras do abstract-fonte; tokens separados
  por espaço contendo ao menos um caractere alfanumérico. Executado uma única vez e agora
  embutido no próprio checker (`check_manuscript_consistency.py`), que passa a reportar o
  número exato e o método: `OK abstract 195-230 words (227, pdftotext:artigo_v3.pdf)`.
  O texto do abstract NÃO foi alterado.
- Arquivos corrigidos/regenerados: `consistency_report.md` (regenerado pelo checker; antes
  registrava 228 pela convenção antiga de strip no fonte — origem da divergência),
  `check_manuscript_consistency.py` (método unificado, com fallback declarado para faixa
  quando não houver PDF/pdftotext). O rebuttal já dizia 227 (2 ocorrências, inalteradas).
  Ocorrências remanescentes de "228" no pacote: apenas o footprint de memória
  "≈228 KB FP32" (Tabela 22, não relacionado) e a menção histórica no FINAL_AUDIT.md
  descrevendo a correção anterior. **Divergência 227/228: eliminada.**

## Rebuttal references (R3 C5, item v)

- **UR2M — metadados verificados** (arXiv 2402.09264; IEEE Xplore 10494467; PDF do grupo
  em mobile-systems.cl.cam.ac.uk/papers/percom24.pdf): Hong Jia, Young D. Kwon, Dong Ma,
  Nhat Pham, Lorena Qendro, Tam Vu, Cecilia Mascolo, "UR2M: Uncertainty and Resource-Aware
  Event Detection on Microcontrollers", IEEE PerCom 2024.
  Forma final no rebuttal: **"UR2M (Jia et al., 2024, IEEE PerCom)"**.
- **Structural-Jump-LSTM — metadados verificados** (arXiv 1904.00761; OpenReview
  B1xf9jAqFQ): Christian Hansen, Casper Hansen, Stephen Alstrup, Jakob Grue Simonsen,
  Christina Lioma, "Neural Speed Reading with Structural-Jump-LSTM", ICLR 2019.
  Forma final no rebuttal: **"Structural-Jump-LSTM (Hansen et al., 2019, ICLR)"**.
- Ambos permanecem citados exclusivamente no rebuttal (texto corrido, agora autocontido
  com autor/ano/venue verificados); nada foi adicionado ao `references.bib` nem à Related
  Work do manuscrito. Nenhum metadado inventado.

## Table 1

**TABLE 1 VERIFIED — NO CHANGE REQUIRED.**

As nove colunas (Onset, TTD, Cov., Causal, Temp. Org., ΔH*, Eff., Edge, Real) classificam
capacidades do protocolo/deployment, não o tipo de sinal de controle; nenhuma célula das
linhas SkipRNN/AdaFrame/ACT/LiteEval codifica "usa predictive uncertainty". A coluna ΔH*
(dinâmica de entropia near-onset como sinal de scheduling) está ✗ para os quatro — o que
permanece correto, e até mais claramente correto, sob a nova caracterização (gates/halting
e políticas aprendidas, não incerteza preditiva). UR2M e Structural-Jump-LSTM NÃO foram
adicionados à tabela. Narrativa circundante verificada: a definição das colunas, o
parágrafo da cadeia de dependências (que já ressalva "periodic-refresh and learned-gating
baselines provide alternative allocation mechanisms") e a nota sobre a coluna Real Eval.
não chamam esses métodos de uncertainty-driven nem afirmam que todo método adaptativo usa
confidence; a tripartição learned gating/halting × learned temporal selection ×
confidence/uncertainty-driven da §1/§2.4 foi preservada intacta.

## Consistency

- Build do artigo: `artigo_v3.pdf` — 37 pp, 0 referências indefinidas, 0 overfull (inalterado nesta passada; nenhuma edição no artigo.tex).
- Build do rebuttal: `rebuttal_v3.pdf` — 19 pp, 0 erros, 0 overfull (recompilado com os metadados).
- `verify_paper.sh`: **0 FAIL / 40 OK + 8/8 testes** (inclui o novo check exato do abstract).
- Contagem do abstract: **227** em todas as fontes vivas (rebuttal ×2, consistency_report, LITERATURE_FIXES_MINIMAL, FINAL_AUDIT).
- **Nenhum resultado experimental foi modificado**: TTD_ef, Tabelas 8–23, Wilcoxon, r_rb, Skip-RNN, PreOcc/NormOcc, matched-budget, definição de trustworthy, Conclusions, Abstract e Figuras intactos — as únicas edições desta passada foram as duas formas de citação no rebuttal e o método de contagem no checker.

## Final status

**READY FOR RESUBMISSION**
