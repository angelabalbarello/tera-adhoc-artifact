# Projeto Overleaf — Trustworthy Selective Computation (Ad Hoc Networks)

## O que mudou (correção da quebra da submissão ADHOC-D-26-02099)
O PDF submetido tinha 190 páginas de **log de compilação + `nullfont`** (o artigo
não tipografou). Causa: mistura de template (o `.tex` usava `elsarticle`, mas o
pacote/《revista》 usa o template **CAS** `cas-dc`), `.bst` inconsistente
(`elsarticle-num` vs o `model1-num-names.bst` do CAS) e caminho `image/` sem
`\graphicspath` (Editorial Manager achata os arquivos).

## Correções aplicadas neste projeto
1. **Classe convertida para `cas-dc`** (template oficial CAS da Elsevier / Ad Hoc Networks).
   Só o preâmbulo e o frontmatter mudaram; **o corpo do artigo é idêntico**.
2. **Frontmatter em sintaxe CAS**: `\title[mode=title]`, `\author[n]{}`, `\affiliation[n]{}`,
   `\cortext`, `\begin{abstract}`, `\begin{keywords}`.
3. **Bibliografia**: `\bibliographystyle{model1-num-names}` (consistente com o `.bst` incluído).
4. **`\graphicspath{{image/}{./}}`** — figuras resolvem em `image/` OU achatadas.
5. **hyperref/natbib** não são recarregados (o `cas-dc` já os carrega) — evita *option clash*.

## Como compilar (Overleaf)
- Abra o projeto no Overleaf; defina **`artigo.tex`** como *Main document*.
- Compilador: **pdfLaTeX**. Sequência: pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX.
- Deve gerar ~30–40 páginas em duas colunas.

## Arquivos
- `artigo.tex` — manuscrito principal (CAS, corrigido).
- `cas-dc.cls`, `cas-common.sty` — classe CAS.
- `model1-num-names.bst`, `references.bib` — bibliografia.
- `paper_metrics_macros.tex`, `paper_macros_ablacao_teacher.tex`, `paper_calibration_macros.tex` — macros de números (`\input`).
- `paper_table2_rows.tex`, `paper_table_stratified_rows.tex` — linhas de tabelas (`\input`).
- `image/` — 12 figuras referenciadas.
- `artigo_elsarticle_original.tex.bak` — versão elsarticle original (referência; não é o main).

## Observação
Não foi possível compilar neste ambiente (sem a classe CAS/rede). Compile no Overleaf,
que tem o `els-cas-templates`. Se aparecer algum *option clash* de pacote, remova a
duplicata (o `cas-dc` já carrega `natbib`, `hyperref`, `geometry`).
