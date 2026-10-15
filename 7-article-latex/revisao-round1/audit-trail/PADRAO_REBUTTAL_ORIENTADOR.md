# Padrão de rebuttal do orientador — extraído do exemplo AURORA (NCAA-D-21-00691) e estado de conformidade

Fonte: carta + "Response to Reviewers" do AURORA (Geraldo, ago/2021) enviados em anexo.

## O padrão, elemento por elemento

1. **Carta ao editor** (1ª página): data; nome e cargo do editor; "Dear ...,"; agradecimento; "Ref. ID ..." + título entre aspas; frase-modelo "the manuscript has been revised in accordance to the comments... For each issue, we numbered it and explained how we addressed all the points raised... all the changes in the manuscript have been highlighted in **blue**"; agradecimento final; "Best Regards, The Authors".
2. **"Response to Reviewer #N:"** como cabeçalho por revisor.
3. Por comentário: **"N. Comment:"** com o texto do revisor na íntegra, seguido de **"Response:"**.
4. A Response contém, nesta ordem: (a) aceite/agradecimento ("We agree with the reviewer...", "We are grateful for the comment and sorry this was not made clear...") ou recusa justificada; (b) **o local exato da mudança no formato SEÇÃO + PARÁGRAFO ordinal** — ex.: "the fifth paragraph of Section 2", "the sixth paragraph of Section 1 and the first paragraph of Section 3", "(fourth paragraph, Subsection 4.3.1)", "the third paragraph of Subsection 3.1"; (c) **o trecho novo citado integralmente entre aspas**, introduzido por "as shown below:"; (d) referências adicionadas listadas por extenso logo abaixo da citação.
5. **Figura ou tabela nova/alterada é REPRODUZIDA dentro do rebuttal**, logo abaixo da resposta (AURORA C4: Figura 1 alterada reproduzida com legenda; C5: Figuras 10a–10d novas reproduzidas).
6. Respostas multi-item quando o comentário tem subitens ("Response to item i)...").

## Conformidade do nosso rebuttal (rebuttal_v4.pdf, 21 pp) — ajustes aplicados nesta passada

- **Localizações completadas no formato seção+parágrafo** (as que estavam só com a seção): R1 C3 → "(Section 1, eighth paragraph)"; R1 C4 → "(Section 3, third paragraph)"; R2 C5 limitação → "(Section 5, second paragraph of the limitations discussion)"; R3 C3 → "(Section 5, fourth paragraph of the limitations discussion)"; R3 C4 → "(Section 4.12, second paragraph)" e "(..., third paragraph)"; R3 C5 Skip-RNN → "(Section 4.11, second paragraph of the Skip-RNN passage)"; R3 C8 → "(Section 4.13, first paragraph)"; Add.2 → "(Section 4.8, third paragraph)". As demais já estavam no padrão ("final paragraph of Section 2.1", "opening paragraph of Section 2.2", "contributions paragraph of Section 1", "closing paragraph of Section 2.6", "Statistical analysis paragraph, Section 4.4", "paragraph after Algorithm 1", "final paragraph of Section 1"). Os ordinais usam os parágrafos reais do artigo_v4.pdf (pós-quebra de parágrafos).
- **Correção de erro**: a atribuição da citação da Tabela 11 dizia "Section 4.5"; Movement 5 é a **Section 4.6** — corrigido para "(Section 4.6, second to fourth paragraphs)".
- **Floats novos reproduzidos no rebuttal** (item 5 do padrão): Figura 5 regenerada (R1 C6); Figura 14 + Tabela 21 da generalização de escala temporal (R3 C4); Tabela 19 dos gates de confiança/entropia suavizada (R3 C5); Tabela 23 do mapa de protocolos (Additional 2). A Tabela 20 (Skip-RNN) e a matriz de compatibilidade já estavam reproduzidas. Tabelas largas entraram com \resizebox; refs internas resolvidas para os números finais via .aux.
- Já conformes de antes: carta ao editor com a frase-modelo (agora "highlighted in blue"), "Response to Reviewer #N", "N. Comment:"+"Response:", aceites justificados, trechos novos entre aspas, referências citadas (kerby2014simple por extenso não é necessária no rebuttal — está no .bib do artigo; UR2M e Structural-Jump-LSTM com autor/ano/venue verificados), resposta multi-item no R3 C5 (itens i–v) e R3 C4 (itens i–vi).

Build: rebuttal_v4.pdf — 21 páginas, 0 erros, 0 overfull; suíte de consistência 0 FAIL / 40 OK + 8/8 (nenhum texto citado foi alterado; só atribuições de local e floats reproduzidos).
