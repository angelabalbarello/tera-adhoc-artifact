# CHANGELOG_FINAL — revisão cirúrgica final de 17 pontos (27/09/2026)

Tipos: **T** = textual · **C** = conceitual (escopo/enquadramento) · **E** = estatística (números/efeitos).
Todos os resultados experimentais foram preservados; nenhum número novo foi inventado
(efeitos r_rb recomputados de expanded_per_seed.csv e mechanism_ablation_per_seed.csv; ver STATISTICAL_AUDIT.md).

| # | Arquivo / seção | Antes | Depois | Motivo | Tipo |
|---|---|---|---|---|---|
| P2 | artigo.tex §4.4 | wording que sugeria poder estatístico "adequado" | "To address the limited inferential resolution of the initial four-seed campaign, the factorial study covers twelve seeds" | poder não foi calculado a priori; linguagem honesta | T |
| P3 | artigo.tex Tab. 9 + §4.4 + §4.5; robustness_section.tex | Cliff's δ (δ=+0,361 etc.; "large/negligible") | correlação rank-biserial de pares combinados (Kerby 2014): F1 +1,000/12, ECE +1,000/12, TTD_ef +0,744/12, FR +1,000/5; FR agendado +1,000/7; Tab.11: 0,87/0,90/0,77; classes de magnitude removidas | δ de Cliff incoerente com Wilcoxon pareado que descarta zeros; efeito agora casa com o teste | E |
| P3e | robustness_section.tex final | "Cliff's δ = −0,5 (large)" e "δ = 0 (negligible)" | rótulos removidos; "no paired difference"; efeito marcado REQUIRES RECOMPUTATION (per-seed FS não arquivado) | não recomputável dos artefatos ⇒ não publicar | E |
| P4 | artigo.tex §2.2 | resultado antes do critério | ordem princípio→critério→resultado: agregado pode ocultar seed insegura ⇒ critério por seed ⇒ "Under this criterion, Baseline+MHEG..." | definição de trustworthy como cadeia lógica | C |
| P5 | artigo.tex Discussion | nuance Skip-RNN implícita | explícito: Skip-RNN "matched or exceeded... We therefore do not claim superiority over learned adaptive computation on detection performance"; contribuição reposicionada (auditabilidade/ancoragem, não superioridade) | incorporar não-superioridade como resultado declarado | C |
| P6 | artigo.tex Discussion | limitação do TTD_ef ausente | TTD_ef "captures missed-event and post-onset delay costs (Eq. 13) but does not penalize persistent pre-onset activation"; PreOcc/NormOcc reportados separadamente; sem métrica composta nova ("leaving the joint reporting of the three quantities as the stated protocol") | limitação declarada sem criar métrica ad hoc | C |
| P7 | artigo.tex abstract, §1, §4.13, Conclusion; rebuttal (abstract reproduzido) | claims de calibração heterogêneos | frase única: "calibration and temporal organization are empirically dissociable under this protocol" (abstract em 227 palavras, contagem programática no texto compilado) | claim de calibração uniforme e escopado | C |
| P8 | artigo.tex §4.4 | hierarquia implícita | explícita: TTD_ef primário; FR restrição de segurança (≤0,05 por seed); F1/ECE/%Sup/latências auxiliares | leitura inferencial inequívoca | C |
| P9 | artigo.tex §4.4 | correção múltipla ad hoc | taxonomia: inferência só na fatorial n=12; n=4 descritivo; §4.12 exploratória com Holm por família de endpoint | multiplicidade justificada, não mecânica | E |
| P10 | rebuttal.tex R3 C5 | resposta em prosa às 5 famílias | matriz de compatibilidade (família × eixo de alocação × compatível? × tratamento) | responder o pedido de baselines uncertainty-aware de forma auditável | T |
| P11 | (verificado, sem mudança) | — | honestidade matched-budget preservada (§4.12: sem vantagem FS@onset a custo igual em 8/9 durações) | — | — |
| P12 | (verificado, sem mudança) | — | achados da auditoria interna preservados compactos (rebuttal Additional 7a–f) | — | — |
| P13 | rebuttal.tex | 29 páginas com blocos reproduzidos extensos | 18 páginas: Algoritmo 1 e Tab. 19 substituídos por ponteiros + excertos de 1 linha; citações R2.5/R3 C4/C5/C6 e Fig. 5 encurtadas com "(...)" preservando os espelhos verificados por teste; margens 2,3 cm (geometry) — conteúdo científico intacto | meta 18–22 pp sem perder substância | T |
| P14 | artigo↔rebuttal | — | espelhos re-verificados por teste: "up to $0.158$", "in every evaluated seed", "not numerically comparable to Table", "that a learned gate does not provide", "matches or exceeds", "empirically dissociable...", "matched-pairs rank-biserial" | coerência automática | T |
| P15 | artigo.tex Conclusion | — | verificada: nenhum claim além dos resultados; frase de dissociabilidade escopada | — | C |
| P16 | highlights.tex / Highlights.txt | — | verificados: compatíveis com claims finais (sem superioridade sobre gates aprendidos; "exact nonparametric tests and effect sizes" agora literal) | — | T |
| — | references.bib | — | + @article{kerby2014simple} | fonte do efeito r_rb | E |
| — | tests/ | asserts de δ | test_fr_wilcoxon: asserts r_rb (4 endpoints); test_manuscript_claims: proíbe "Cliff" e "adequate statistical power", exige "matched-pairs rank-biserial", "kerby2014simple", "primary operational endpoint", "safety constraint", "empirically dissociable under this protocol" + 2 espelhos novos | regressão trava a revisão | E |

## Estado da verificação
- `verify_paper.sh`: **TUDO OK** — 0 FAIL / 40 OK no checker + 8/8 testes (inclui novos asserts r_rb).
- artigo_v2.pdf: 36 páginas, 0 referências indefinidas. rebuttal_v2.pdf: 18 páginas, 0 overfull, 0 erros.
- Entregáveis: artigo_final.tex e rebuttal_final.tex (cópias dos canônicos), CHANGELOG_FINAL.md, STATISTICAL_AUDIT.md.
- Operacional (Angela): fechar o leitor de PDF → apagar artigo.pdf/artigo_final.pdf/rebuttal.pdf antigos e renomear artigo_v2.pdf→artigo.pdf, rebuttal_v2.pdf→rebuttal.pdf; push dos commits; Overleaf.
