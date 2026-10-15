# Checklist da revisão — baseado no manual (Guide for Authors + e-mail do editor)
### ADHOC-D-26-02099 · Ad Hoc Networks · major revision · prazo 13/set/2026

Verifiquei o **Guide for Authors** oficial do Ad Hoc Networks e cruzei com a **versão submetida** (`00_versao_SUBMETIDA/artigo.tex`). Abaixo, o que o manual exige e o status atual.

---

## A. Procedimento de reenvio (do e-mail do editor)
- [ ] **Responder aos revisores DENTRO da plataforma** (Editorial Manager) — obrigatório antes de submeter a revisão.
- [ ] **Carta de resposta ponto-a-ponto** ("Response to Reviewers") anexada.
- [ ] **Subir os SOURCE FILES** (`.tex` + `.bib` + `.cls` + figuras). **PDF não é aceito como fonte.** A versão submetida já tem tudo isso.
- [ ] Considerar referências sugeridas pelos revisores só se forem realmente pertinentes (o editor diz que não são obrigatórias).

## B. Formatação exigida pelo manual × status atual

| Item | Regra do manual | Status na versão submetida | Ação |
|---|---|---|---|
| **Abstract** | **≤ 250 palavras**, sem citações | **≈ 375 palavras** ❌ | **Cortar para ≤250** (P0) |
| **Declaração de IA generativa** | Seção com título exato *"Declaration of generative AI and AI-assisted technologies in the manuscript preparation process"*, com a frase-modelo, **antes das referências** | Existe como *"Generative AI Use Statement"* (título/local não conformes) | **Renomear + usar o texto-modelo** (P0) |
| **Data statement** | Option C: depositar dados em repositório + citar/linkar, **ou** justificar por que não pode compartilhar | "…available upon publication" (fraco) | **Reescrever** para Option C (P1) |
| **Highlights** | 3–5 bullets, **≤ 85 caracteres cada**, arquivo separado com "highlights" no nome | Ausente | **Criar arquivo** highlights (recomendado) |
| **Declaration of competing interests** | Preencher a *declarations tool* + subir um **Word .docx** | Existe seção no texto | Manter no texto **e** gerar o .docx da ferramenta |
| **CRediT** | Contribuição por autor (taxonomia CRediT) | Presente ✓ | Conferir papéis |
| **Funding** | Se não houve, incluir frase-padrão | Presente ✓ | Conferir frase-padrão |
| **Acknowledgements** | Seção própria, logo antes das referências | Presente ✓ | OK |
| **Keywords** | 1–7, em inglês, evitar termos com "and/of" | 7 keywords (algumas multi-palavra) | Opcional simplificar |
| **Seções** | Numeradas 1, 1.1, 1.1.1; abstract não numerado; cross-ref por número | cas-dc numera | OK |
| **Referências** | Estilo numérico [n] na ordem de aparição; DOIs recomendados | model1-num-names (numérico) ✓ | Conferir DOIs |
| **Figuras** | Arquivos separados, citadas, numeradas, alta resolução (halftone ≥300 dpi; vetor EPS/PDF) | PDFs em `image/` | Conferir resolução |
| **Math** | Editável, equações numeradas | OK (LaTeX) | OK |
| **Classe LaTeX** | Template Elsevier (CAS). Double-column é permitido em LaTeX | `cas-dc` ✓ | OK |

## C. Frase-modelo da declaração de IA (colar no lugar da atual, antes das referências)
> **Declaration of generative AI and AI-assisted technologies in the manuscript preparation process**
> During the preparation of this work the author(s) used [NOME DA FERRAMENTA] in order to [MOTIVO]. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article.

(Não é necessário declarar uso de corretor ortográfico/gramatical básico.)

## D. Frase-padrão de funding (se não houve financiamento)
> This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## E. Arquivos a subir no reenvio (Editorial Manager)
1. Manuscrito revisado — **source** (`.tex`, `.bib`, `.cls`, `\input` .tex, pasta `image/`).
2. (Recomendado) Manuscrito revisado com **alterações marcadas** (marked-up).
3. **Response to Reviewers** (ponto-a-ponto).
4. **Highlights** (arquivo separado).
5. **Declaration of competing interest** (.docx da declarations tool).
6. Figuras em alta resolução (arquivos separados).

## F. Bloqueio atual
Os **comentários dos revisores não vieram no e-mail** — estão no Editorial Manager. Para eu montar a Response to Reviewers e aplicar as mudanças de conteúdo, **cole aqui os pareceres** (ou me libere o Chrome logado no EM).

> Prioridades imediatas independentes dos pareceres (P0): **cortar o abstract p/ ≤250 palavras** e **corrigir a declaração de IA generativa**. Posso já fazer as duas na versão submetida se você quiser.
