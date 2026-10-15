# STATISTICAL_AUDIT — inventário completo dos testes do manuscrito (27/09/2026)

Regra da revisão: nenhum valor foi inventado ou ajustado à mão. Todos os efeitos foram recomputados
dos CSVs arquivados (`seed-expansion-tera/expanded_per_seed.csv`, `ablacao_mecanismo_temporal/mechanism_ablation_per_seed.csv`)
com scipy 1.15.3; o que não pôde ser recomputado está marcado `REQUIRES RECOMPUTATION`.
Efeito padronizado adotado: **correlação rank-biserial de pares combinados** (Kerby, 2014),
r_rb = (R⁺ − R⁻)/ΣR sobre os postos dos pares não nulos — coerente com o Wilcoxon exato usado
(zero_method='wilcox'), que descarta os mesmos zeros. O Cliff's δ anterior foi removido de todo o texto.

## Hierarquia de endpoints (declarada na §4.4)
TTD_ef = endpoint operacional primário · FR = restrição de segurança (FR ≤ 0,05 por seed) ·
F1, ECE, %Sup, latências brutas = descritores auxiliares.
Inferência confinada à campanha fatorial de 12 seeds; campanhas complementares (n=4) descritivas;
contrastes por duração da §4.12 exploratórios (Holm dentro de cada família de endpoint).

## A. Campanha fatorial 12 seeds — CONFIRMATÓRIA (Tabela 9, §4.4)
Contraste pareado por seed: AF-TOI Fixed vs Baseline Fixed. Wilcoxon exato, bicaudal, zeros descartados.

| Endpoint | Hipótese | n (n_eff) | p | r_rb | Multiplicidade | Status |
|---|---|---|---|---|---|---|
| TTD_ef | AF reduz atraso efetivo | 12 (12) | 0,0210 | +0,744 | sem correção (primário único) | confirmatório |
| FR | AF reduz taxa de falha | 12 (5; 7 empates em zero) | 0,0625 | +1,000 | restrição de segurança, não teste de superioridade | confirmatório (limítrofe; lido com o critério por seed) |
| F1 | AF não degrada detecção | 12 (12) | 0,0005 | +1,000 | auxiliar | secundário |
| ECE | AF melhora calibração episódica (ECE_pipe) | 12 (12) | 0,0005 | +1,000 | auxiliar | secundário |
| FR agendado (Base+MHEG vs AF+MHEG) | MHEG sem AF viola cobertura | 12 (7) | 0,0156 | +1,000 | controle negativo | confirmatório |
| Controle negativo TTD_ef (MHEG em trajetórias não organizadas) | MHEG degrada custo sem AF | 12 | 0,001 | — (sinal consistente) | controle negativo | confirmatório |
| MHEG em trajetórias organizadas | reduz custo sem perder cobertura | 12 | 0,0005 | — | controle negativo | confirmatório |

Verificação automatizada: `tests/test_fr_wilcoxon.py` (agora com asserts de r_rb: F1 +1,000/12, ECE +1,000/12, TTD_ef +0,744/12, FR +1,000/5).

## B. Ablação de mecanismo, 12 seeds — SECUNDÁRIA (Tabela 11, §4.5; protocolo de replicação run_v29, não comparável à Tab. 8)
| Contraste | p | r_rb | Status |
|---|---|---|---|
| NoTemp vs α=0,15 canônico, agendado (×3,5) | 0,005 | +0,87 | secundário |
| NoTemp vs pior α da grade, agendado (×3,0) | 0,003 | +0,90 | secundário |
| NoTemp vs média da grade, agendado (×4,8) | 0,001 | — | secundário |
| ECE episódica (ECE_v29) vs cada ponto da grade (12/12 seeds) | 0,0005 | +1,000 | secundário |
| %Sup NoTemp > canônico | 0,021 | +0,744 | secundário |
| Fixed: NoTemp vs α=0,15 (×5,7) | 0,016 | +0,77 | secundário |
| Fixed: NoTemp vs pior α | 0,91 | sem diferença pareada | secundário (limitação declarada) |

## C. Escala temporal, §4.12 — EXPLORATÓRIA (12 seeds congeladas, 9 durações)
p-values por duração reportados sem correção + Holm dentro de cada família de endpoint;
apenas TTD_ef em d=0,21 s sobrevive (p=0,031). Texto classifica explicitamente como exploratório.
FS@onset a custo igual: indistinguível em 8/9 durações (p≥0,21) — ausência de vantagem declarada como resultado.

## D. Campanhas complementares (n=4, seeds 42–45) — DESCRITIVAS
- Calibração pós-hoc (Tab. 14, §4.8): sem testes inferenciais; leitura descritiva ("empiricamente dissociáveis sob este protocolo").
- Gating baselines/refresh (Tab. 19, §4.11): descritiva; calibração oracle no test declarada.
- Skip-RNN (Tab. 20, §4.11): descritiva, pontos de operação pareados; o texto declara explicitamente que NÃO se reivindica superioridade sobre computação adaptativa aprendida nos endpoints de detecção.
- Campanha de estresse (§4.10, robustness_section): descritiva a orçamento igualado.

## E. REQUIRES RECOMPUTATION (único item)
**Efeito padronizado da ablação de canais da campanha de estresse** (robustness_section.tex, parágrafo final):
os CSVs de resultados da campanha (8-stress-campaign/results/) arquivam apenas agregados; os valores FS
por seed necessários para r_rb não foram arquivados. O rótulo de efeito (antes "Cliff's δ = −0,5, large")
foi removido do texto, que agora reporta apenas a direção do ranqueamento pareado, e "δ = 0, negligible"
virou "no paired difference". Fórmula a aplicar quando os per-seed forem regenerados:
r_rb = (R⁺ − R⁻)/ΣR (Kerby 2014) sobre FS@onset híbrido vs entropy-only, 4 pares.

## F. Multiplicidade — taxonomia adotada (sem Holm mecânico global)
1 endpoint primário (TTD_ef) ⇒ sem correção; FR é restrição, não teste de superioridade;
F1/ECE auxiliares reportados com p exatos; família exploratória da §4.12 com Holm por endpoint;
todos os demais números descritivos. Justificativa no texto (§4.4) e nesta auditoria.
