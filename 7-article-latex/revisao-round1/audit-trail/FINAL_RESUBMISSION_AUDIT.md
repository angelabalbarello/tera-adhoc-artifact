# Final Resubmission Audit — ADHOC-D-26-02099 (27/09/2026)

Base auditada: `artigo.tex`/`artigo_final_v2.tex` (idênticos) e `rebuttal.tex`/`rebuttal_final_v2.tex` (idênticos), builds `artigo_v3.pdf` (37 pp) e `rebuttal_v3.pdf` (19 pp). Princípio aplicado: audit first; nesta passada foram feitas apenas 3 mudanças (duas formas de citação no rebuttal com metadados verificados, um estreitamento de escopo "causal recurrent"→"causal RNN/LSTM/GRU" na intro da matriz do rebuttal) + método unificado de contagem do abstract no checker. Nenhum resultado experimental tocado.

## 1. Executive Verdict

**READY FOR RESUBMISSION**

Fundamentos: suíte de consistência integralmente verde (0 FAIL / 40 OK + 8/8 testes, incluindo locks de regressão para cada problema já identificado nesta revisão); builds limpos; espelhos artigo↔rebuttal verificados por teste; claims escopadas e simétricas (inclusive as desfavoráveis: paridade do Skip-RNN, FR p=0,0625 limítrofe, calibração+MHEG ≥ AF-TOI in-distribution); caracterizações bibliográficas corrigidas e travadas; TTD_ef com definição única verificada numericamente contra os CSVs; validade externa explicitamente limitada em abstract, §2, §4.13, §5 e Conclusion.

## 2. Critical Issues

Nenhum. (Nada bloqueia a submissão.)

## 3. Minor Issues

Todos já corrigidos nesta passada ou em passadas anteriores desta sessão; registrados na Seção 13. Itens OPTIONAL (registrados, NÃO editados, para não introduzir regressão de última hora):
- Grafia dupla "SkipRNN" (3×, §1/§2.4/Table 1, forma compacta com \cite) vs "Skip-RNN" (18×, §4.11/Discussion). Ambas usuais na literatura; a citação desambigua. Unificação é cosmética e não foi aplicada.
- FINAL_AUDIT.md menciona "228" apenas como histórico da correção (não é divergência viva).

## 4. Reviewer-by-Reviewer Audit

| Reviewer | Comment | Status | Evidência | Risco residual | Ação necessária |
|---|---|---|---|---|---|
| R1 | C1 protocolo vehicular vs ad hoc | PASS | escopo single-node declarado em §1/§6; quote no rebuttal espelhada | baixo | — |
| R1 | C2 significado de TERA-Gen | PASS | definição no abstract e §3.1; quote espelhada | nenhum | — |
| R1 | C3 hipótese/informative scheduling signal | PASS | definição operacional em vermelho §1 ("By an informative scheduling signal we mean...") | nenhum | — |
| R1 | C4 sensor stream da arquitetura | PASS | §3.1 revisado; quote espelhada | nenhum | — |
| R1 | C5 complexidade do ciclo MHEG | PASS | §3.4 + Alg.1: custo 2 camadas O(4(DH+H²)+8H²), ≈57,2K MACs; rebuttal aponta em vez de reproduzir (excerto de 1 linha com a fórmula) | nenhum | — |
| R1 | C6 Fig.5 seed 42 mal explicada | PASS | guia frame-a-frame §4.2 + critério de mediana pré-declarado (test_figure5_selection trava ep36/ep148); rebuttal admite os 2 problemas achados (seleção não declarada; bug de base do log) | baixo (transparência joga a favor) | — |
| R2 | C1 abstract longo | PASS | 227 palavras (contagem programática unificada, checker exato); rebuttal reproduz o abstract literal | nenhum | — |
| R2 | C2 parágrafo de estrutura | PASS | parágrafo final da §1; quote espelhada | nenhum | — |
| R2 | C3 related work completo (elogio) | PASS | resposta breve + crítica consolidada na Table 1/§2.6 | nenhum | — |
| R2 | C4 estrutura clara (elogio) | PASS | resposta breve | nenhum | — |
| R2 | C5 resultados não totalmente provados / synthetic / 4 seeds | PASS | expansão para 12 seeds sob protocolo pré-especificado; inferência restrita à fatorial; n=4 descritivo; quote §4.4 espelhada (cauda resumida com síntese fiel); linguagem "limited inferential resolution" (sem "statistical power") | médio-baixo: FR p=0,0625 é limítrofe — mas o texto lê pelo critério por seed e não sobrevende | — |
| R2 | C6 fontes das figuras | PASS | Figs. 4–11 regeneradas; correções declaradas (Fig.13 título; velha Fig.14 removida) | nenhum | — |
| R3 | C1 novelty não clara | PASS | abstract/§1 revisados; distinção metodológica vs distillation citada (excerto encurtado com ponteiro §2.6) | baixo | — |
| R3 | C2 comentários próprios no related work | PASS | fechos críticos por subseção + Table 1; §2.6 reescrito | nenhum | — |
| R3 | C3 ambiente controlado favorece o método | PASS | caveat §5 (3 elementos mitigadores); quote espelhada | médio (inerente ao desenho); mitigado com honestidade | — |
| R3 | C4 generalização entre durações | PASS | §4.12 novo (grade 0,21–5,21 s, 12 seeds congeladas) + matched-budget com Holm/exploratório; quotes com escala-nota preservada | baixo | — |
| R3 | C5 baselines mais fortes | PASS | 5 famílias na matriz de compatibilidade; confidence gate, smoothed-entropy, refresh R=8, Skip-RNN end-to-end (Tab.19–20); claim de ausência reescopada: "targeted literature search did not identify... single-node causal RNN/LSTM/GRU"; UR2M (Jia et al., 2024, IEEE PerCom) e Structural-Jump-LSTM (Hansen et al., 2019, ICLR) autocontidos e verificados | médio-baixo: reviewer pode pedir UR2M adaptado — resposta já explica por que não é like-for-like | — |
| R3 | C6 ablações do mecanismo | PASS | Tab.11 12 seeds retreinadas; reporta simetricamente o contraste não significativo (p=0,91) | baixo | — |
| R3 | C7 uso amplo de 'trustworthy' | PASS | definição operacional restrita §2.2 (princípio→critério→resultado); quote espelhada | nenhum | — |
| R3 | C8 viabilidade embarcada | PASS | §4.13 declara analítico/GPU de referência; sem claim de hardware físico | nenhum | — |
| R3 | C9 vehicular edge vs single-node | PASS | §6 reescrito; ℓ_t como "candidate input" apenas | nenhum | — |
| R3 | C10 limitações reconhecidas | PASS | §5 ampliada | nenhum | — |
| R3 | C11 limitações + future work | PASS | §5/§6; naturalistic/multi-node/hardware = future work | nenhum | — |

Nenhum MAJOR ISSUE; nenhuma resposta promete alteração ausente do manuscrito (espelhos com verificação automática); nenhum número difere entre rebuttal e artigo (test_mirrors + test_table_values).

## 5. Article ↔ Rebuttal Consistency

PASS. Todas as passagens "the ... reads:"/"Table X reports" verificadas: quotes literais existem no artigo (as encurtadas usam "(...)" com ponteiro explícito de seção e síntese fiel do trecho omitido — nunca versão antiga); espelhos críticos travados por teste: "up to $0.158$", "in every evaluated seed", "not numerically comparable to Table", "that a learned gate does not provide", "matches or exceeds", "empirically dissociable under this protocol", "matched-pairs rank-biserial", abstract literal (227), r_rb 0,87/0,90/0,77, p-values idênticos. Sem citação de seção/tabela errada (labels resolvem; 0 refs indefinidas).

## 6. Claims Audit

Busca global por proves/confirms/establishes/rules out/decisive/universally/consistently outperforms/superior/guarantees/deployment-ready/production-ready/well-powered: todas as ocorrências são negações, escopos ou usos técnicos — "does not guarantee" (§1, §2.4, Discussion, Conclusion), "proves sensitive to the probability scale" (remoção da velha Fig.14), "not a claim of superiority" e "we therefore do not claim superiority" (Skip-RNN), "no per-seed guarantee" (§4.12). Único "guarantees" afirmativo: "the unconditional trigger guarantees allocation in abrupt onsets" (§3.4) — propriedade determinística de projeto do gatilho cinemático, suportada. PASS. Termos sensíveis: "trustworthy" = sentido operacional restrito definido em §2.2 e usado consistentemente; "safe/safely" = sempre sob a restrição de cobertura avaliada; "robust(ness)" = restrito às campanhas §4.10/§4.12 com escala declarada; "generalization" = "aggregate temporal-scale robustness across the evaluated synthetic range", sem guarantee por seed; "real-time/deployment-ready" = ausentes; "uncertainty-aware" = usado apenas para a família externa, nunca para Skip-RNN.

## 7. Literature Characterization Audit

PASS. Skip-RNN = learned update gate ("Skip-RNN is not an uncertainty-aware method", §4.11 — agora test-locked); ACT = learned halting; AdaFrame/LiteEval = learned temporal-selection policies; nenhum deles descrito como predictive-uncertainty-driven (§1, §2.4, Discussion corrigidos e verificados). Structural-Jump-LSTM = "learned RL policies rather than predictive-uncertainty estimates" (Hansen et al., 2019, ICLR — verificado em arXiv 1904.00761/OpenReview). UR2M = single-node, evidential uncertainty, "feed-forward convolutional cascade rather than an RNN/LSTM/GRU" (Jia et al., 2024, IEEE PerCom — verificado em arXiv 2402.09264/IEEE Xplore). AMOR: não citado nominalmente; SSM coberto por "newer state-space architectures" como classe arquitetural distinta (sem dizer que não é recorrente). Claim de baseline: "single-node causal RNN/LSTM/GRU" em todas as posições do contexto de ausência de baseline; "causal recurrent" amplo: 0 ocorrências restantes no rebuttal (test-locked); "No published..." : 0 ocorrências (test-locked).

## 8. Statistical and Metric Consistency

PASS. n=12/seeds 42–53 na fatorial e Tab.11 (16 ocorrências "42--53"); n=4/42–45 apenas nas complementares, sempre declarado (19 ocorrências); Wilcoxon exato bicaudal com zeros descartados descrito idem no artigo, rebuttal e testes (7/12 zeros ⇒ n_eff=5 ⇒ p=0,0625); r_rb (Kerby 2014) recomputado dos CSVs e travado por teste; p-values idênticos artigo↔rebuttal; hierarquia de endpoints + taxonomia de multiplicidade declaradas (§4.4); significância nunca convertida automaticamente em relevância prática (leitura operacional separada); "statistical power"/"powered": 0 ocorrências (test-locked). Métricas: TTD_det/TTD_a/TTD_b/TTD_ef/FR/F1/%Sup/PreOcc/NormOcc com símbolos estáveis; ECE nomeada por protocolo (pipe/replay/cal/v29, 9 usos + Table 23 coerente); TTD_ef = definição agregada única TTD_det + FR×T_ep (Eq. 12≡13, verificada numericamente; forma por casos removida e proibida por teste); Figs. 8–9 explicitamente distinguidas da estatística tabulada (nota na §4.1: a média dos valores exibidos = (1−FR)·TTD_det+FR·T_ep ≠ quantidade tabulada).

## 9. Skip-RNN Comparison Audit

PASS. Apresentado como "published adaptive temporal-computation baseline/control"; nunca uncertainty-aware; o texto admite "matched or exceeded ... on the primary detection endpoints" no ponto de supressão pareada; TTD_ef=0 interpretado como colapso da detecção sobre t₀ com 21% de ocupação pré-onset ("informative rather than flattering"); PreOcc/NormOcc = "complementary signal-semantics metrics, introduced in response to this result, not as substitutes for the primary endpoints"; a contribuição é reposicionada para semântica do sinal + contrato de runtime "not a claim of superiority on the detection endpoints above". Nenhuma frase tenta "ganhar" a comparação nos endpoints; limitação do TTD_ef (não penaliza ativação pré-onset) declarada na Discussion sem métrica composta ad hoc.

## 10. External Validity and Scope

PASS. Sintético declarado no abstract ("controlled synthetic, single-node"), §2.2, §4.4, §4.13, §5, §6 e na coluna Real da Table 1 (✗ com parágrafo justificando); naturalistic = future work; hardware físico não testado ("analytical ... or measured on a reference GPU; none derives from execution on physical automotive hardware"); multi-node não testado (ℓ_t = candidate input); warp = "controlled temporal train–test shift" com escala própria declarada; calibração: "empirically dissociable under this protocol" — sem claim geral de irrelevância; resultado favorável a isotonic+MHEG preservado como caveat na Discussion; untested-under-shift explícito. Causalidade: reference model não-causal offline vs deployed model causal distinguidos (§3, "no access to future observations"); nenhum trecho sugere acesso a frames futuros no deployment.

## 11. Bibliography and Cross-References

PASS. bibtex sem warnings de citação indefinida; 0 "multiply defined labels"; 0 referências/citações indefinidas nos dois builds; velha Figure 14 e o stratification score aparecem apenas como remoção declarada (§4.8/rebuttal 7e); nenhuma menção a AUPRC pré-onset; kerby2014simple presente e citado; nenhuma referência nova não citada; UR2M/Structural-Jump-LSTM somente no rebuttal, com autor/ano/venue verificados na web (não convertidos em algo diferente do publicado; UR2M tem versão de conferência IEEE PerCom 2024 além do arXiv).

## 12. Build and Regression Results

- `pdflatex -jobname=artigo_v3 artigo.tex` (×2 + bibtex): **37 pp, 0 referências indefinidas, 0 overfull, 0 erros**.
- `pdflatex -jobname=rebuttal_v3 rebuttal.tex` (×2): **19 pp, 0 erros, 0 overfull** (dentro da meta 18–22).
- `bash scientific-audit/scripts/verify_paper.sh <pasta>`: **0 FAIL / 40 OK + 8/8 testes**, incluindo os locks novos: proibição de "causal recurrent" e "No published uncertainty-aware" no rebuttal, exigência de "single-node causal RNN/LSTM/GRU" e de "Skip-RNN is not an uncertainty-aware method", contagem 227 (×2) travada, forma por casos do TTD_ef proibida, "statistical power" proibido, contador exato do abstract embutido no checker (`227, pdftotext:artigo_v3.pdf`).

## 13. Exact Required Fixes (aplicados nesta passada)

| Arquivo | Localização | Texto atual (antes) | Texto novo | Motivo |
|---|---|---|---|---|
| rebuttal.tex | R3 C5, intro da matriz | "this paper's single-node causal recurrent timestep-level setting" | "this paper's single-node causal RNN/LSTM/GRU timestep-level setting" | eliminar o escopo amplo residual no contexto de baseline (BLOCKING para a regra do item 5) |
| rebuttal.tex | R3 C5 (v) | "UR2M (Jia et al., 2024)" | "UR2M (Jia et al., 2024, IEEE PerCom)" | autocontenção bibliográfica com venue verificado (MINOR) |
| rebuttal.tex | R3 C5 (v) | "Structural-Jump-LSTM is a closer..." | "Structural-Jump-LSTM (Hansen et al., 2019, ICLR) is a closer..." | idem (MINOR) |
| check_manuscript_consistency.py | check 9 | contagem por strip no fonte (dava 228) | contagem oficial pdftotext no PDF compilado (227) com fallback declarado | eliminar divergência 227/228 (MINOR) |

OPTIONAL (registrado, não editado): unificação de grafia SkipRNN/Skip-RNN.

## 14. Final Checklist

- [x] reviewer responses complete (23/23 comentários, R1 6 + R2 6 + R3 11)
- [x] manuscript/rebuttal mirrored (verificação automática)
- [x] no unsupported strong claims
- [x] literature characterizations accurate
- [x] experimental numbers unchanged
- [x] TTDef definition consistent (agregada, única, verificada nos dados)
- [x] statistical language bounded
- [x] Skip-RNN comparison symmetric
- [x] synthetic/external-validity limitations explicit
- [x] citations resolved
- [x] references resolved
- [x] builds clean
- [x] consistency tests pass

Operacional (fora do escopo científico): renomear `artigo_v3.pdf`→`artigo.pdf` e `rebuttal_v3.pdf`→`rebuttal.pdf` após fechar o leitor; apagar builds `_v2`/antigos; push dos commits; Overleaf.
