# Mapa dos ajustes em azul (artigo) × citações no rebuttal — página por página

Base: `artigo_v4.pdf` (36 pp, marcas `\revcolor` azuis) e `rebuttal_v3.pdf` (19 pp).
Método: extração programática dos 183 spans `\color{\revcolor}` do `artigo.tex` + `robustness_section.tex`,
agrupados em 56 ajustes lógicos; páginas localizadas por busca normalizada no texto extraído dos dois PDFs.
"—" na coluna do rebuttal = ajuste de iniciativa própria (auditorias internas/correções bibliográficas),
não citado textualmente no rebuttal. Legenda: R#C# = Reviewer/Comment; Add.# = "Additional changes" do rebuttal (pp. 17–19).

| # | Ajuste em azul no artigo | Local no artigo | Pág. artigo | Onde o rebuttal cita | Pág. rebuttal |
|---|---|---|---|---|---|
| 1 | Famílias de computação seletiva: gating/halting (SkipRNN, ACT) e políticas de seleção temporal (AdaFrame, LiteEval) distintas de incerteza preditiva | §1 Introdução | 1 | — (correção bibliográfica própria, LITERATURE_FIXES_MINIMAL) | — |
| 2 | Definição de *informative scheduling signal* (nível de entropia separa fases) | §1 | 2 | R1 C3 (quote literal) | 3 |
| 3 | Expansão do nome TERA-Gen ("the TERA episode Generator") | Abstract/§1 | 1–2 | R1 C2 (quote) e R2 C1 (abstract reproduzido) | 3, 6 |
| 4 | Alvo de treino do AF-TOI = forma temporal da trajetória, não as predições do professor | §1 | 2 | R3 C1 (resposta sobre novelty) | 9 |
| 5 | Reescrita do controle negativo (componente entrópico "cannot discriminate") | §1 | 3 | R2 C5 (linguagem moderada) | 6–7 |
| 6 | Números do ecossistema completo (49,8%; 0,340 vs 0,677 ms/f; 0,099 vs 0,570 s) | §1 | 3 | R2 C1 (abstract reproduzido com os mesmos números) | 6 |
| 7 | Parágrafo de estrutura do artigo ("The remainder of this paper...") | §1 (fecho) | 3 | R2 C2 (quote literal) | 6 |
| 8 | Legenda da Fig. 1 (bloco do sinal informativo = entropia estratificada por fase) | Fig. 1 | 3 | R1 C3 | 3 |
| 9 | Modelo mantido pequeno de propósito (≈58K isola o mecanismo; ganho escala com o preditor) | §2.1 | 3 | — | — |
| 10 | Esclarecimento de escopo: redes veiculares vs ad hoc; TERA na camada de computação | §2.1 | 4 | R1 C1 (quote literal) | 3 |
| 11 | Definição operacional de *trustworthy* (princípio→critério por seed→resultado; 3/12 vs 12/12) | §2.2 | 4 | R3 C7 (quote literal) | 15 |
| 12 | §2.4 tripartição: learned gating/halting × temporal selection × uncertainty-driven | §2.4 | 5 | R3 C5 (resposta menciona o posicionamento da §2.4) | 12 |
| 13 | "In the vehicular-monitoring setting studied here, this assumption can break down" | §2.4 | 5 | — (micro-edição de escopo) | — |
| 14 | Distinção metodológica do AF-TOI vs distillation (fecho da §2.6) | §2.6 | 5 | R3 C1 (quote com ponteiro) | 9 |
| 15 | Cadeia de dependências escopada (gates alternativos reconhecidos: refresh/learned) | §2.7 | 6 | — | — |
| 16 | Custo de 2 camadas (≈57,2K MACs) + notação H (largura vs entropia) | §3 | 7 | R1 C5 e Add. 7b | 4, 19 |
| 17 | Descrição do sensor stream (concatenação das 4 modalidades, x_t∈R³¹) | §3 | 7 | R1 C4 (quote literal) | 4 |
| 18 | Regra do episódio perdido (média dos últimos k=6 frames vs limiar de validação) | Tab. 2 (FR) | 9 | — (harmonização interna) | — |
| 19 | Loss AF-TOI completa (Eqs. 5–9: termos, pesos, janelas) | §3.2 | 10 | R3 C6 ("removes only the onset-anchored terms") e Add. (manifest) | 14 |
| 20 | Scheduler de 3 fases (E_warm=12, E_ramp=28, E_total=80; α piso 0,50; δ 0,30→2,80) | §3.2 | 10 | — (correção code→paper da auditoria) | — |
| 21 | MHEG: dependência explícita da estratificação (falha do controle negativo) | §3.3 | 11 | — | — |
| 22 | Notação K dos canais cinemáticos (K=\|K\|≤D) | §3.3 / Eq. 10 | 11 | R1 C5 (notação definida e usada consistentemente) | 4 |
| 23 | Alg. 1: x₀=0 (primeiro frame sempre atualiza) + parágrafo de complexidade por timestep | §3.4 / Alg. 1 | 12 | R1 C5 (quote do núcleo da fórmula) | 4 |
| 24 | 58.242 parâmetros (forma fechada por camada) + convenção 1 MAC = 2 FLOPs | §3.4 | 12 | Add. 7b | 19 |
| 25 | Dataset: 12.480 episódios, 12 seeds (42–53), reprodução verificada das 4 iniciais; células azuis da Tab. 6 (hiperparâmetros da loss) | §4.1 | 13 | R2 C5 (protocolo da expansão) | 7 |
| 26 | Definição única de TTD_ef (Eq. 13 ≡ Eq. 12) + nota sobre as Figs. 8–9 | §4.1 | 14 | — (auditoria TTD_ef, FINAL_AUDIT) | — |
| 27 | Efeito rank-biserial de pares (Kerby) + hierarquia de endpoints + taxonomia de multiplicidade | §4.1 | 13–14 | R2 C5 e R3 C6 (r_rb nos quotes) | 7, 14–15 |
| 28 | Mapa de protocolos (remissão à Tab. 23) | §4.1 | 14 | Add. 2 | 18 |
| 29 | Mov. 1: leitura do Baseline (sinal não discriminante) + valores em bits (0,35/0,94/1,0) | §4.2 | 14 | R1 C6 + Add. 7c (unificação em bits) | 4, 19 |
| 30 | Guia frame-a-frame da Fig. 5 (elementos visuais, leituras por coluna, 2 caveats) | §4.2 | 14–16 | R1 C6 (quote do critério + síntese) | 4 |
| 31 | Legenda da Fig. 5: critério da mediana pré-declarado (ep. 36/148) | Fig. 5 | 16 | R1 C6 e Add. 7d | 4, 19 |
| 32 | Mov. 2: supressão onset-window 9,8% vs 37,2% (e 34,3%/54,4% fora) | §4.3 | 16 | — (números do log regenerado) | — |
| 33 | Controle negativo: 13,9%, logs corrigidos de seed 42 (37%/54% vs 10%/34%) | §4.4 | 17 | R3 C5(ii) e R2 C1 (abstract) | 6, 12 |
| 34 | Análise estatística: protocolo pré-especificado das 12 seeds, p-values, escopo n=4 | §4.4 | 18 | R2 C5 (quote literal do parágrafo) | 7 |
| 35 | Tab. 9: r_rb/n_eff/p (FR p=0,0625; 7/12 zeros; nota sem classes verbais) | Tab. 9 | 18 | Add. 7a (contagem corrigida) e R2 C5 | 7, 19 |
| 36 | Legenda da Fig. 7: linha agregada vs critério por seed (3/12 violações, até 0,158) | Fig. 7 | 19 | R3 C7 (espelho "in every evaluated seed"/"up to 0.158") | 15 |
| 37 | Mov. 4: n=1.440/12 seeds nas legendas + nota da média por episódio (Figs. 8–9) | §4.5 | 20 | — (consistência TTD_ef) | — |
| 38 | Tab. 11 + parágrafo: ablação de mecanismo 12 seeds (×3,5/×3,0/×4,8; r_rb 0,87/0,90/0,77; p=0,91 reportado) | §4.6 | 21 | R3 C6 (quote literal encurtado) | 14 |
| 39 | Tab. 12: sensibilidade a T_ep recomputada (12 seeds) | Tab. 12 | 21 | R3 C6 (mencionada como ablação complementar) | 14 |
| 40 | Mov. 6: definição de p̄max (descritor de sinal ≠ regra de FR) | §4.7 | 20 | — | — |
| 41 | Nota de escala: ECE_pipe/replay/cal/v29 nomeadas | §4.8 | 21–22 | Add. 2 (quote literal) | 18 |
| 42 | Remoção do score H_strat (sensível à escala; evidência pela morfologia) | §4.8 | 22 | Add. 7e e R2 C6 (velha Fig. 14 removida) | 8, 19 |
| 43 | Declaração direta: IC+MHEG iguala/supera AF-TOI+MHEG no replay (leitura mecanicista; não testado sob shift) | §4.8 | 22 | R3 C5(i) | 12 |
| 44 | §4.11 nova: gates de confiança/entropia suavizada (Tab. 19, deadlock) + refresh R=8 | §4.11 | 25 | R3 C5(i)/(iii) (respostas + ponteiro para Tab. 19) | 12 |
| 45 | §4.11: comparação Skip-RNN + Tab. 20 (paridade admitida; PreOcc/NormOcc; sem claim de superioridade) | §4.11 | 26 | R3 C5(iv) (quote literal do parágrafo de resultados) | 12–14 |
| 46 | §4.12 nova: generalização de escala temporal (warp, grade 0,21–5,21 s, matched-budget, Holm/exploratório) | §4.12 | 27–30 | R3 C4 (quotes: nota de escala + resultados + matched-budget) | 10–11 |
| 47 | §4.13: viabilidade analítica/GPU de referência (sem hardware físico) | §4.13 | 28 | R3 C8 (quote literal) | 15 |
| 48 | Tab. 23: mapa experimental completo (float novo) | Tab. 23 | 31 | Add. 2 | 18 |
| 49 | Discussion: caveats in-distribution (IC+MHEG, refresh, Skip-RNN) + limitação do TTD_ef (sem métrica composta) | §5 | 31 | R3 C5(iv) ("not a claim of superiority"; limitação declarada) | 12–14 |
| 50 | Discussion: premissa reatribuída à família uncertainty-driven (não a SkipRNN/ACT) | §5 | 31 | — (correção bibliográfica própria) | — |
| 51 | Limitações: 3ª (protocolo 12 seeds) + caveat do ambiente controlado (3 mitigadores) | §5 | 33 | R2 C5 (quote da 3ª limitação) + R3 C3 (quote do caveat) | 7, 10 |
| 52 | §6/VEC-MEC: ℓ_t obtido por mapeamento O(1) (2 pontos) | §6/§7 | 32, 34 | — | — |
| 53 | Conclusão: números em azul + bound do matched-budget + robustez de escala temporal | §8 Conclusão | 35 | R3 C9 (reescrita do escopo single-node) | 16 |
| 54 | Data Availability: repositório público + hashes | Data Availability | 35 | Add. 6 | 19 |
| 55 | Declaração de IA generativa (Guide for Authors) | Declaração final | 35–36 | — (exigência editorial) | — |
| 56 | Nota de escopo na campanha de estresse (vantagem concentra em cobertura/timeliness sob shift de escala) | §4.10 (robustness) | 24 | R3 C4 (consistência com o matched-budget) | 10–11 |

## Leitura rápida por comentário do rebuttal

- **R1 (pp. 3–4):** C1→item 10; C2→3; C3→2, 8; C4→17; C5→16, 22, 23; C6→29–31.
- **R2 (pp. 6–8):** C1→3, 6, 33; C2→7; C5→5, 25, 27, 34, 35, 51; C6→29, 42.
- **R3 (pp. 9–16):** C1→4, 14; C3→51; C4→46, 56; C5→12, 33, 43, 44, 45, 49; C6→19, 27, 38, 39; C7→11, 36; C8→47; C9→53.
- **Additional changes (pp. 17–19):** 2→28, 41, 48; 6→54; 7a→35; 7b→16, 24; 7c→29; 7d→31; 7e→42.
- **Não citados no rebuttal (iniciativa própria/auditorias):** itens 1, 9, 13, 15, 18, 20, 21, 26, 32, 37, 40, 50, 52, 55 — todos de precisão/consistência (correções bibliográficas, TTD_ef, harmonizações code→paper e exigência Elsevier), sem mudança de claim que exigisse resposta a revisor.
