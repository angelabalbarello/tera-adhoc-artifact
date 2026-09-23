# ADVERSARIAL_REVIEW_R3 — segunda passagem no papel do Revisor #3 (23/09/2026)
Método: tentativa ativa de rejeição; cada número re-derivado dos artefatos locais.
Nada foi alterado antes desta lista; correções aplicadas em seguida (CRITICAL/MAJOR).

## CRITICAL

**C1 — Número não reproduzível na definição de trustworthy (categorias 1 e 6).**
- Arquivo/linha: artigo.tex §2.2 (~L405) e rebuttal.tex C7 (espelho).
- Evidência: o texto diz que Baseline+MHEG "violates it in individual seeds (up to $0.117$)". Dos logs (expanded_per_seed.csv, config baseline_hybrid): violações de FR>0,05 em TRÊS seeds — 45 (0,1167), 49 (0,075) e **53 (0,1583)**. O máximo é 0,158, não 0,117 (0,117 é o máximo do baseline_FIXED, config errada).
- Impacto: o exemplo que sustenta a nova definição operacional cita um número que o revisor não consegue reproduzir — exatamente o tipo de erro que a definição pretendia sanar.
- Correção: "violates it in three of twelve seeds (up to $0.158$)" em ambos os arquivos. **APLICADA.**

## MAJOR

**M1 — Figura 4(b) contradiz o texto do Movement 1 (categoria 3).**
- Arquivo/linha: artigo.tex ~L1302 ("Pre-onset converges to the central region (≈0.5 bit)").
- Evidência (episode_frame_logs_v2, afkd_fixed, seed 42, em bits): pre-onset mediana 0,94, pico de densidade ≈1,0; média 0,78. A massa pré-onset está junto ao MÁXIMO, não na região central. (Post-onset ≈1 bit confere; Normal pico ≈0,35.)
- Impacto: guia de leitura da figura errado — mesmo tipo de falha da Fig. 5 original.
- Correção: reescrever com os números reais (Normal ~0,35 bit; Pre-onset desloca-se para cima, mediana ~0,94; Post-onset acumula no máximo). **APLICADA.**

**M2 — "Suppression is uniform" no Baseline contradiz os dados corrigidos (categoria 3).**
- Arquivo/linha: artigo.tex §4.2 (~L1317, "suppressions distribute uniformly") e Movement 2 (~L1349, "suppression is uniform").
- Evidência: log v2 — Baseline+MHEG suprime 37,2% da janela de onset vs 54,4% fora; NÃO é uniforme: o gatilho CINEMÁTICO protege parcialmente o onset nas duas configurações. O diferencial do AF-TOI é reduzir para 9,8% vs 34,3%.
- Impacto: claim mais forte que o dado; um revisor com os logs refutaria em minutos.
- Correção: atribuir a proteção residual do Baseline ao gatilho cinemático e reservar ao componente entrópico apenas o diferencial. **APLICADA.**

**M3 — Eq. 6 não representa o código (categoria 9).**
- Evidência: `build_loss_objects` usa `BCEWithLogitsLoss(pos_weight=...)` — a BCE de frames e de episódio é PONDERADA POR CLASSE; a Eq. 6 diz BCE simples.
- Correção: declarar "class-weighted (positive-class reweighted)" na Eq. 6/texto. **APLICADA.**

**M4 — Gates de ativação dos termos omitidos (categoria 9).**
- Evidência: no código, L_late só ativa após o ramp-up (δ(e)>0,5) e L_pre só no refinamento avançado (δ(e)>1,0); §3.2 não dizia.
- Correção: declarar os gates. **APLICADA.**

**M5 — Refresh gate sem n e sem declarar calibração oracle (categorias 5 e 7).**
- Arquivo/linha: artigo.tex §4.11 (parágrafo novo) e rebuttal C5-i.
- Evidência: números (85,9±0,4% etc.) vêm de 4 seeds com limiares calibrados na partição de teste (mesma convenção oracle da Tabela 19), mas o parágrafo não dizia nem o n nem o oracle — um resultado n=4 lido com força de n=12.
- Correção: "(seeds 42--45, same oracle calibration on the test partition as Table~19; descriptive)". **APLICADA.**

**M6 — Script gerador do log v2 ausente do artefato (categoria 1/reprodutibilidade).**
- Evidência: episode_frame_logs_v2.csv foi reconstruído por código executado ad hoc; sem um script versionado, o revisor não reproduz as Figs. 4–6.
- Correção: criado `scientific-audit/scripts/rebuild_episode_frame_logs_v2.py` (determinístico, lê o log v1 + trajetórias fixas). **APLICADA.**

**M7 — Artefato público desatualizado vs. Data Availability (categoria 1).**
- Evidência: o repositório GitHub tem 1 commit (consolidação de 27/08). Os artefatos que sustentam a Tabela 11 (ablação de mecanismo, 11–12/09), Tabela 20 (Skip-RNN), Tabela 19 v2 + refresh, grade temporal-scale v2, log v2 e scripts de auditoria são posteriores e NÃO estão públicos. A frase "publicly available" é, hoje, mais ampla que o repositório.
- Impacto: reprodutibilidade pública dos claims novos = não, até o push.
- Correção: exige `git push` da branch `fix/rebuttal-scientific-consistency` (bloqueado nesta máquina por `.git/index.lock` removível só no Windows). **PENDENTE — ação da autora.**

## MINOR
- m1: abstract, "is cheaper still" com referente ambíguo → "is the cheapest configuration". APLICADA.
- m2: §3.2, rampa do push começa em 0,45 (omitido) → declarado. APLICADA.
- m3: Fig. 4(a), "no visible separation" levemente forte (médias 0,23–0,60 com picos sobrepostos ~0,22–0,39) → "the phase densities largely overlap". APLICADA.
- m4: Figs. 12–13 continuam com fontes pequenas e "95\%" no título (pendência já registrada; regeneração requer rerun da calibração).

## Categorias sem achados novos
(2) tabelas incompatíveis: Tab. 12 = TTD_b+FR×T_ep reconfere com a célula preenchida (0,396=0,223+0,035×5; 0,570=0,223+0,035×10 ✓); Tab. 11×8 e 20×8 declaradas não comparáveis. (4) demais claims conferidos contra logs. (8) Conclusão/§4.13/abstract delimitados. (10) numeração pós-remoção da Fig. 14 verificada no .aux (Skip-RNN=Tab. 20, temporal=Tab. 21/Fig. 14, mapa=Tab. 23).
