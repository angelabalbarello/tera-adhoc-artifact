# Response to Reviewers — ADHOC-D-26-02099
### "Trustworthy Selective Computation for Resource-Constrained Vehicular Edge Networks" — Ad Hoc Networks

**Como usar:** a plataforma exige responder cada comentário individualmente. Abaixo, para cada comentário, há um bloco **Response** em inglês, pronto para colar na caixa *Reply to comment*. Marquei com **[AÇÃO]** o que ainda precisa ser executado no `.tex` antes de reenviar, e com **[JÁ FEITO pós-submissão]** o que suas versões posteriores já cobrem.

> Estratégia geral (colar como abertura da carta-resumo, opcional): *"We thank the editor and the three reviewers for their careful and constructive assessments. We have (i) shortened the abstract, added a structure paragraph, and moderated all statistical and 'trustworthy'/'safe' claims to the evaluated regime; (ii) clarified TERA-Gen, the conceptual hypothesis, the operational architecture, the MHEG complexity, and the figures; (iii) sharpened the methodological novelty of AF-TOI relative to knowledge distillation and adaptive-computation literature; (iv) added robustness/stress experiments (abrupt vs. gradual onsets, onset-window coverage, entropy-lag sensitivity, intermediate-risk conditions), an architectural ablation, and a per-channel ablation; and (v) consolidated the limitations (synthetic data, n=4 seeds, profiled rather than physical hardware, single-node scope) consistently across abstract, discussion, and conclusion. Point-by-point responses follow."*

---

# Reviewer 1

**R1.1 — "Protocol of vehicular and ad hoc network is not the same, but this work does not take it that way."**
> **Response:** We thank the reviewer. We agree that vehicular networks (VANETs) and general ad hoc/sensor networks differ at the MAC, routing, and mobility-management layers. Our contribution is deliberately at the **on-node inference layer** that is common to resource-constrained nodes in both settings, and we do **not** model or claim any network-layer protocol. We have revised the Introduction and Section 2.1 to state this scope explicitly, to avoid conflating the two protocol families, and to clarify that "vehicular and ad hoc edge networks" refers to the *deployment context* of the managed node, not to a shared networking protocol. Network-level coordination is discussed only as an architectural interpretation (Section 6) and is not evaluated.

**R1.2 — "What does TERA-Gen mean in this protocol?"**
> **Response:** We have added an explicit definition at first use. **TERA-Gen** is the generator of the **TERA (Temporal Evaluation and Risk Assessment)** ecosystem: a synthetic episode generator that produces continuous driving streams with a **deterministic, frame-precise risk onset**, with risk targets calibrated against SHRP2 odds ratios. Its purpose is to remove the ≈0.31 s naturalistic annotation variability so that scheduling decisions can be attributed to the exact frame at which they are taken. A new paragraph in Section 3 (and a one-line definition in the Introduction) now states this before the term is used. [AÇÃO: garantir a definição no primeiro uso do termo]

**R1.3 — "The conceptual hypothesis investigated by TERA (Fig. 1) is not explained well; the 'informative scheduling signal' block is not justified."**
> **Response:** We have expanded both the caption of Fig. 1 and the surrounding text. The hypothesis is that the conflict between efficiency and detection coverage is resolved not by tuning scheduler thresholds but by **inducing temporal organization** in the model's predictive trajectories during offline training; once trajectories are organized, predictive entropy **stratifies across episode phases**, and this stratification is what turns uncertainty into an **informative scheduling signal** (i.e., a signal that differentiates stable intervals from the moments around risk onset). The revised figure caption now explains each block, and a new paragraph justifies why entropy stratification — rather than raw confidence — is what makes the scheduling signal informative. [AÇÃO]

**R1.4 — "The sensor stream of the Operational architecture of TERA (Fig. 2) needs to be explained."**
> **Response:** We have expanded the description of Fig. 2. The input stream is a continuous multimodal vector \(x_t \in \mathbb{R}^{31}\) aggregating four domains — behavioral-visual, kinematic, contextual, and acoustic — with an observability mask \(m_t\) (missing ≠ zero). The text now describes how \(x_t\) enters the causal model, how the per-timestep signals (\(p_t, H_t, \Delta x_t\)) are produced, and how MHEG consumes \(\Delta x_t\) and \(H_{t-1}\) before the current update. [AÇÃO]

**R1.5 — "The Efficient inference cycle of the MHEG runtime scheduler (algorithm) needs its complexity explained."**
> **Response:** We have added an explicit complexity statement. The MHEG scheduling decision is **O(1) per timestep** (a kinematic-variation test and an entropic-gate comparison against calibrated thresholds), which is negligible next to the recurrent update. When MHEG triggers a full update, the causal two-layer LSTM costs \(\mathcal{O}(4(DH+H^2))\) per step; when it reuses the buffered state, the per-step cost collapses to the O(1) decision. This complexity analysis now accompanies the algorithm. [AÇÃO]

**R1.6 — "Frame-by-frame temporal evidence for seed 42 (Fig. 5) is not explained well."**
> **Response:** We have rewritten the explanation of Fig. 5. The figure contrasts the Baseline and AF-TOI+MHEG configurations on the same episodes: solid curves are \(p_t\), dashed curves are \(H_t\), the vertical line marks \(t_{\text{onset}}\), and the markers indicate where MHEG allocates versus reuses. The revised text walks through how, under AF-TOI, \(H_t\) rises around onset (enabling allocation at emergence windows), whereas in the Baseline \(H_t\) stays flat, so suppression is distributed uniformly and the criterion has no temporal basis. [AÇÃO]

---

# Reviewer 2  *(recommendation: Minor Revision)*

**R2.1 — "The abstract is well-structured but too long (~350 words vs ~200 norm)."**
> **Response:** We thank the reviewer. We have shortened the abstract to approximately 200 words (within the journal's 250-word limit), preserving the explicit statement of contributions while removing redundancy. [AÇÃO: cortar o abstract de ~375 → ~200 palavras]

**R2.2 — "The introduction lacks an explicit paragraph outlining the paper's structure."**
> **Response:** We have added a short roadmap paragraph at the end of the Introduction outlining the structure of the paper (Section 2 related work and positioning; Section 3 the TERA methodology; Section 4 experimental evaluation; Section 5 discussion and threats to validity; Section 6 architectural implications; Section 7 conclusion). [AÇÃO]

**R2.3 — "The related work is unusually complete (six subsections and a gap matrix)."**
> **Response:** We thank the reviewer for this positive assessment; no change was required. We have only ensured the gap matrix (Table 1) remains consistent with the moderated claims.

**R2.4 — "The structure is clear and the problem is well defined; framing is heavy but navigable."**
> **Response:** We thank the reviewer. To reduce the density noted, we have tightened the framing in the Introduction and Section 3 and moved secondary detail to the appropriate later sections, without changing the technical content.

**R2.5 — "Results are not fully proven: synthetic data, n=4 seeds, minimum Wilcoxon p=0.0625 (no conventional significance). Moderate the statistical language."**
> **Response:** We agree and have moderated the statistical language throughout. The revised manuscript states explicitly that, with n=4 seeds, the paired Wilcoxon test has a minimum one-sided p-value of 0.0625 and therefore **no result reaches conventional significance at 0.05**; accordingly, the Wilcoxon and Cliff's-δ values are reported as **descriptive effect-size characterizations**, not inferential tests, and the strongest evidence is framed as the **directionality and mechanistic consistency of the negative control**. We have also added 95% confidence intervals (recipe-stratified bootstrap) and note that expanding to 10–12 seeds with an a priori power analysis is a priority for the next iteration. [JÁ FEITO pós-submissão, em parte — confirmar/ampliar]

**R2.6 — "Figures are high quality but several are dense/text-heavy and would benefit from larger fonts."**
> **Response:** We have revised the dense figures to use larger fonts and reduced in-figure text, moving explanatory detail to the captions and main text to improve readability at print size. [AÇÃO: aumentar fontes nas figuras densas]

---

# Reviewer 3  *(major concerns)*

**R3.1 — "Background/motivation/novelty of AF-TOI not clear; distinction from temporal KD, sequence-level transfer, temporal representation learning, uncertainty-aware inference, and adaptive/selective computation is insufficient. State what is fundamentally new in AF-TOI."**
> **Response:** We thank the reviewer and have added an explicit "what is new" discussion. Conventional temporal knowledge distillation and sequence-level transfer align a compact model to the **average predictive behavior** of a reference; adaptive/selective-computation and early-exit methods act on **instantaneous confidence**; uncertainty-aware inference estimates uncertainty but does not shape its **temporal organization**. AF-TOI is different in objective: it aligns a causal deployment model to an **onset-oriented** reference specifically to induce **entropy stratification across episode phases** (temporal organization of uncertainty) so that the signal becomes usable for **scheduling under an explicit miss-coverage constraint**. In other words, AF-TOI does not target accuracy or calibration but the *temporal differentiation* of uncertainty near risk onset — a property none of the cited families induces as a first-class objective. This is now stated as a dedicated paragraph in the Introduction and reinforced in Section 3. [AÇÃO]

**R3.2 — "In related work, add more of your own commentary on state-of-the-art to facilitate the problem formulation."**
> **Response:** We have added critical commentary to each related-work subsection, stating not only what prior methods do but **why they are insufficient** for temporally-organized scheduling under coverage constraints (e.g., confidence-based skipping assumes low uncertainty implies absent risk, which breaks in causal driving streams; entropy-driven early-exit is applied on general benchmarks without a deterministic onset). These remarks now lead directly into our problem formulation. [AÇÃO]

**R3.3 — "Main conclusions rely on TERA-Gen (deterministic frame-level onset); this favors the proposed onset-oriented strategy and may not persist under realistic (noisy/gradual/abrupt/ambiguous) conditions."**
> **Response:** We agree this is a central concern and have addressed it on two levels. (i) We added **robustness/stress experiments** that vary onset dynamics within the controlled setting — abrupt (sub-0.6 s) versus gradual (multi-second) transitions, onset/noise scenarios, and entropy-lag sensitivity — showing where the mechanism holds and where it degrades. (ii) We are explicit that TERA-Gen is a **controlled instrument**, not a substitute for naturalistic data: frame-precise onset is what makes the scheduling effect *measurable*, and generalization to naturalistic driving (DMD, DADA) is stated as the primary future validation and as an explicit limitation. [JÁ FEITO pós-submissão, em parte — confirmar as tabelas de robustez estão incluídas]

**R3.4 — "Investigate whether AF-TOI generalizes to unseen temporal risk trajectories, different onset durations, abrupt vs gradual transitions, intermediate-risk conditions, and feature-distribution shifts."**
> **Response:** We have added experiments along these axes: (a) abrupt vs. progressive regimes with different onset durations; (b) an **onset-window coverage** analysis; (c) an **entropy-lag** sensitivity study (how far the entropic signal can lag before coverage degrades); and (d) an **intermediate-risk (Attention)** evaluation on conditions unseen during training. These show the mechanism is stable on progressive onsets and identify sub-0.6 s abrupt onsets as the worst case, which the kinematic component of MHEG is designed to cover. Feature-distribution shift under naturalistic data remains future work and is stated as a limitation. [JÁ FEITO pós-submissão, em parte]

**R3.5 — "Experiments insufficient: add stronger baselines — confidence-based selective inference, entropy-based scheduling without AF-TOI, temporal smoothing/filtering, adaptive computation/early-exit, uncertainty-aware resource allocation."**
> **Response:** We have strengthened the comparisons. Our 2×2 factorial already isolates **entropy-based scheduling without AF-TOI** (Baseline+MHEG applies the same entropic gate to unorganized trajectories) and a **confidence/threshold** behavior, both under a matched cost budget; the negative control shows that scheduling without the enabling temporal organization worsens effective detection delay. We additionally evaluate a **causal-Transformer** target to test architecture generality. We have added representative external adaptive-inference baselines (e.g., a SkipRNN-style confidence gate and an early-exit/temporal-smoothing variant) under the **same backbone, data, seeds, and FR ≤ 0.05 constraint**, reporting TTDef and ms/f side by side. [AÇÃO: incluir 1–2 baselines externos sob protocolo pareado; ver protocolo H do arquivo de análise]

**R3.6 — "Additional ablations are required to make the evidence stronger."**
> **Response:** We have added ablations that isolate each component: (i) an **architectural ablation** of the non-causal reference model; (ii) a **per-channel ablation** of the MHEG scheduling channels (kinematic trigger vs. entropic gate, and their disjunction); and (iii) sensitivity to the induction/decay parameter. Together with the 2×2 factorial and the negative control, these attribute the observed gains to the temporal-organization mechanism rather than to any single component. [JÁ FEITO pós-submissão, em parte]

**R3.7 — "'Trustworthy' is used broadly, but experiments mainly show efficiency and detection-delay improvements under a controlled setting."**
> **Response:** We agree and have moderated the terminology. We now define the operational property we actually evaluate — **temporally conditioned operational reliability** (the capacity of the uncertainty signal to differentiate transition regions at the moment the scheduler acts) — and restrict "trustworthy/safe" claims to this property **in the evaluated synthetic regime, with the reported seeds and hardware**. Broad safety/trustworthiness claims have been removed or qualified accordingly. [AÇÃO: revisar título/abstract/conclusão para moderar "trustworthy/safe"]

**R3.8 — "Model-size/footprint/embedded-compatibility results are based on parameter counts, memory, and complexity, not actual execution on physical embedded hardware."**
> **Response:** We agree. The deployment-viability indicators (≈58 K parameters, ≈28 KFLOPs/timestep, INT8 footprint, and per-frame latency on a reference GPU) are now clearly labeled as **profiling and analytical estimates**, not physical embedded execution. We state explicitly that latency, energy, thermal behavior, and sensor-noise robustness remain to be characterized on STM32H7-, Cortex-M55-, and SA8155P-class platforms, and we have moderated the efficiency claims to reflect this. [AÇÃO: rotular como estimativa + moderar]

**R3.9 — "The paper discusses vehicular edge networks, but evaluation is single-node; no multi-node coordination, communication overhead, offloading, or network-level allocation."**
> **Response:** We agree and have made the scope explicit. The **validated contribution is single-node embedded selective computation**. Section 6 is presented as an **architectural interpretation, not an evaluated distributed system**: no networked coordination, offloading, communication latency, or bandwidth is measured. We have added a clear scope statement in the Introduction, Section 6, and the limitations, so no claim exceeds the single-node evaluation. [AÇÃO]

**R3.10 — "Limitations (synthetic setting, single node, naturalistic data, physical hardware, distributional robustness, sensor faults, adversarial perturbations, multi-node) should be reflected consistently throughout."**
> **Response:** We have consolidated these into a dedicated **Threats to Validity** subsection and ensured they are reflected consistently in the abstract, discussion, and conclusion (construct, internal, external, and conclusion validity), so the limitations are not confined to a single paragraph. [JÁ FEITO pós-submissão, em parte — garantir consistência abstract/discussão/conclusão]

**R3.11 — "Comprehensively discuss limitations and future directions (e.g., integrating other learning paradigms). Recommended references: online learning for domain adaptation in Wi-Fi-based device-free localization; online spatiotemporal modeling for robust lightweight device-free localization in nonstationary environments."**
> **Response:** We have expanded the future-work discussion to include integrating complementary learning paradigms — in particular **online learning and domain adaptation** for handling distribution shift between the controlled setting and naturalistic deployment, and online spatiotemporal adaptation of the scheduler thresholds under nonstationary conditions. We assessed the two suggested references on online learning for domain adaptation and online spatiotemporal modeling in nonstationary environments; they are relevant to this future-work direction (robust lightweight adaptation under distribution shift) and are now cited there. *(Per the editor's note, suggested references are included only where genuinely relevant, which is the case for the online-adaptation future-work discussion.)* [AÇÃO: obter as citações completas dos 2 papers e incluí-las no future work]

---

## Resumo de ações no `.tex` antes de reenviar (P0 → P2)

**P0 (textuais, rápidas — posso fazer já):**
- Cortar abstract ~375 → ~200 palavras (R2.1).
- Parágrafo de estrutura no fim da Introdução (R2.2).
- Moderar "trustworthy/safe" e a linguagem estatística (R2.5, R3.7).
- Corrigir a Declaração de IA generativa ao formato exigido (manual).
- Definir TERA-Gen no primeiro uso; expandir legendas das Figs. 1, 2, 5; complexidade O(1) do MHEG (R1.2–R1.6).
- Parágrafo de novidade do AF-TOI (R3.1); comentários críticos no related work (R3.2); escopo single-node explícito (R1.1, R3.9); consistência das limitações (R3.10).

**P1 (reprocessamento — usar o harness do artigo 1):**
- Incluir 1–2 baselines externos sob protocolo pareado (R3.5).
- Confirmar/incluir as tabelas de robustez e ablações que suas versões pós-submissão já têm (R3.3, R3.4, R3.6).
- Aumentar fontes das figuras densas (R2.6).

**P2 (fica como trabalho futuro, moderado):**
- Dado real (DMD/DADA), hardware embarcado físico, multi-nó (R3.3, R3.8, R3.9) — declarar como limitação e future work.

> **Bloqueio nenhum agora** — tenho os pareceres e a versão submetida. Quer que eu comece pelas ações **P0** direto na `00_versao_SUBMETIDA/artigo.tex` (abstract, moderação, TERA-Gen, novidade do AF-TOI, escopo, IA generativa)? E, em paralelo, faço o diff da submetida × suas versões pós-submissão para reaproveitar as tabelas de robustez/ablação que já respondem R3.3–R3.6.
