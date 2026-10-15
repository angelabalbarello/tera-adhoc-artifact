# V8 — Propagação da não-equivalência de semânticas (11/10/2026)

Builds: **artigo_v8.pdf** (37 pp, 0 refs indefinidas; 20 overfulls ≈ linha de base pré-existente) e **rebuttal_v8.pdf** (25 pp, limpo). v7 preservada (`*_v7_BACKUP.tex` + PDFs v7); diffs `DIFF_v7_to_v8_*.diff` (343 linhas). Suíte 0 FAIL/40 OK + 9/9. Status: **[VN]** verificado numericamente · **[CT]** corrigido no texto · **[NV]** não verificável aqui.

| # | Item do protocolo | O que foi feito | Localização | Status |
|---|---|---|---|---|
| 1 | Algoritmo principal = implementação avaliada | Alg. 1 reescrito: REQUIRE declara W=96 e padding; linha de update = `LSTM(window_W(x_{1..t}))` com rótulo "recompute causal window (evaluated semantics)"; supressão reusa só a probabilidade emitida (h removido do ELSE); §3.4 apresenta a semântica avaliada como principal — custo por atualização ≈5,49 M MACs, buffer de entrada obrigatório, releitura de frames suprimidos — e a variante single-step como alternativa **não validada**, com os números da não-equivalência (0,023/44,7% vs 0,170/44,4%; política fixa: FR igual, 6 ms de pad) | Alg. 1; §3.4 | [CT], sustentado pelo [VN] da v7 (`check_inference_semantics.py` + CSV, incluídos no artefato e no repositório) |
| 2 | Propagação | Abstract (marcador retrospectivo), §3.5/Tab. 4 (célula do LSTM: /step e W× por update, nota ‡), Tab. 22 (3 linhas de custo + buffer de janela, nota ‡), síntese §4.13, Discussion (deployment), Conclusão, rebuttal (R1 C5 atualizado; Add. 13 novo) — nenhuma associação dos resultados medidos ao custo da variante incremental restou | listados | [CT] |
| 3 | Custos em 3 níveis + memória | 57,2 K MACs **por passo recorrente**; **5,49 M por atualização** (96×57,216); esperado por frame c_gate+(1−s)·W·O(·); 1.024 B = estado recorrente apenas; janela de entrada 96×31 FP32 ≈ **11.904 B** declarada como estimativa da representação explícita, não pico de RAM | §3.4, Tab. 4, Tab. 22, §4.13, §5, Conclusão | [CT] (aritmética conferida) |
| 4 | ECE_pipe episódica | **Confirmado no código**: `tera_eval` L377 `ece(y_ep, ep_probs)` e run_v29 L1196 idem ⇒ ECE_pipe **e ECE_v29 são episódicas** (a §4.8 dizia frame-level — erro de documentação corrigido); §4.6 "Frame-level ECE degrades" → "Episode-level ECE (ECE_v29)"; Fig. 13(b) já rotulava episode-level; espelhos no rebuttal (quote da nota + resumo R3 C6); STATISTICAL_AUDIT corrigido. ECE_replay e ECE_cal permanecem frame-level (verificadas: fig4c e posthoc L453/460) | §4.8 nota, §4.6, Fig. 13, rebuttal | [VN] |
| 5 | "Frontier" residual | "the operational frontier ... does not move to a qualitatively distinct location" → enunciado restrito ao painel (a); "the frontier in Figure 13" → "the measured operating points in Figure 13"; zero ocorrências restantes de fronteira entre painéis | §4.8 (2 pontos) | [CT] (lock de teste) |
| 6 | Latência de confirmação nas conclusões | Abstract qualifica TTD_ef como "measured with retrospective detection markers" (contagem oficial **231**, atualizada no rebuttal ×3, no checker — agora 195–240 — e nos locks); a nota da §4.1 dá o offset (m−1)Δt≈0,21 s | Abstract, §4.1 | [CT] |
| 7 | Orçamento Skip-RNN | Declarado no parágrafo definicional do §4.11: o matched point iguala **frações de atualizações executadas, não computo** (1 passo vs janela de W frames por atualização); caption da Tab. 20 já dizia "suppression fractions, not FLOPs". §4.12: declarado que os dois braços compartilham a mesma semântica de janela (pareamento interno válido) | §4.11, §4.12, Tab. 20 | [CT] |
| 8 | Fig. 12 / erro de edição / severidades | "Reaching it TTD_ef through" → "Reaching it through" (erro da v7, meu); síntese §4.13: Mov. 6 qualificado "descriptively, under a second evaluation protocol"; Fig. 12 permanece pendente (dependência registrada: rerun da calibração na GPU; trajetórias H(t) das variantes não arquivadas) | §4.8, §4.13 | [CT]; Fig. 12 [NV] |

## Diferenciação exigida
- **Verificado numericamente**: não-equivalência das semânticas e reprodução dos valores publicados (script/CSV da v7, agora citados pelo texto); natureza episódica de ECE_pipe/ECE_v29 e frame-level de ECE_replay/ECE_cal (linhas de código citadas acima); aritmética 96×57,2K e 96×31×4 B.
- **Corrigido no texto**: itens 1–3 (reposicionamento e propagação), 5–8.
- **Não verificável aqui**: Fig. 12 (artefato ausente); pico real de RAM da implementação avaliada (apenas estimativa declarada); avaliação da variante single-step (experimento novo, fora desta rodada — registrado como alternativa não validada no artigo).

## Pendências após v8
1. Variante single-step: calibração+avaliação próprias se o grupo quiser reivindicar o custo incremental (GPU).
2. Fig. 12 (títulos): rerun do experimento de calibração.
3. Itens herdados: refresh/IC sob shift (declarados não testados), FS per-seed do stress (REQUIRES RECOMPUTATION).

Nenhuma incompatibilidade conhecida entre implementação, resultados e narrativa permanece **no texto**; as lacunas restantes estão declaradas como tais no próprio artigo.
