#!/usr/bin/env python3
"""Automated consistency checks for artigo.tex + rebuttal.tex (audit Etapa 14).
Writes consistency_report.md next to the manuscript."""
import re, sys, pathlib

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else ".")
art = (ROOT/"artigo.tex").read_text(encoding="utf-8")
reb = (ROOT/"rebuttal.tex").read_text(encoding="utf-8")
rob = (ROOT/"robustness_section.tex").read_text(encoding="utf-8")
body = art + "\n" + rob
issues, ok = [], []

def check(name, cond, detail=""):
    (ok if cond else issues).append(f"{'OK ' if cond else 'FAIL'} {name} {detail}")

# 1. forbidden strong language
strong = ["rules out","decisive","bear the hypothesis out","this paper establishes",
          "systematically exceed"]
for w in strong:
    hits = len(re.findall(w, body))
    check(f"strong-claim '{w}'", hits==0, f"({hits} hits)")
# 'confirm' tolerated when describing OTHER works (line contains \cite) or as
# 'confirmation ... remains future work'; flagged otherwise
bad_conf = [ln for ln in body.splitlines()
            if re.search(r"\bconfirm(s|ing|ed)?\b", ln)
            and "\\cite" not in ln and "confirmation on naturalistic" not in ln
            and "confirm that output confidence" not in ln]
check("'confirm*' self-claims (target 0)", len(bad_conf)==0, f"({len(bad_conf)})")
# 2. stale numbers
reb_wo_notice = reb.replace("read ``nine of twelve''", "")
check("'nine of' removed (except correction notice)", "nine of" not in body and "nine of" not in reb_wo_notice)
check("'28\\,KFLOPs' removed", "28$\\,KFLOPs" not in body and "28\\,KFLOPs" not in body)
check("'(nats)' removed from text/figs refs", "(nats)" not in body, "")
check("log2 in Eq.11 present", "\\log_2" in art)
# 3. section ref 2.6 vs 2.7
check("intro novelty ref uses sec:rel_temporal", "predictions (Section~\\ref{sec:rel_temporal})" in art)
# 4. revision-process markers
markers = ["in this revision","now separates","Revised in this version",
           "now uses twelve seeds","is now evaluated explicitly",
           "In response to the statistical-power concern","published factorial campaign",
           "published pattern"]
for m in markers:
    check(f"marker '{m}' absent from body", m not in body)
# 5. Data availability tense
check("'will be made available' absent", "will be made available" not in body)
check("'publicly available' present", "publicly available" in art)
# 6. trustworthy per-seed definition
check("per-seed trustworthy definition", "per seed, not merely on average" in art)
# 7. seed counts
check("'42--53' present (12 seeds)", "42--53" in art)
# 8. ECE variant names defined
for v in ["ECE}_{\\mathrm{pipe","ECE}_{\\mathrm{replay","ECE}_{\\mathrm{cal","ECE}_{\\mathrm{v29"]:
    check(f"ECE variant {v.split('{')[-1]}", v in art)
# 9. abstract length
m = re.search(r"\\begin{abstract}(.*?)\\end{abstract}", art, re.S)
t = re.sub(r"\\color\{\w+\}|\\mbox|\\emph|[{}\\\\$]"," ",m.group(1))
n = len([w for w in t.split() if any(c.isalnum() for c in w)])
check("abstract 195-230 words", 195<=n<=230, f"({n})")
# 10. FR facts
check("'seven of the twelve Baseline seeds' present", "seven of the twelve Baseline seeds" in art)
check("mirror: rebuttal has same count", "seven of the twelve Baseline seeds" in reb)
# 11. key mirrored phrases artigo<->rebuttal
mirrors = ["that a learned gate does not provide",
           "matches or exceeds",
           "median} among the critical",
           "not numerically comparable to Table",
           "in every evaluated seed"]
norm = lambda t: re.sub(r"\s+"," ",t)
nart, nreb = norm(body), norm(reb)
for ph in mirrors:
    check(f"mirror '{ph[:34]}...'", ph in nart and ph in nreb)
# 12. labels defined & referenced
for lab in ["tab:protocol_map","tab:ablacao_alpha","tab:skiprnn","fig:temporal_scale"]:
    check(f"label {lab} defined+used", f"\\label{{{lab}}}" in body and f"\\ref{{{lab}}}" in body)
# 13. removed figure not referenced
check("fig:preonset_auprc fully removed", "preonset_auprc" not in body)

rep = ["# consistency_report.md — checagem automática (Etapa 14)",
       f"artigo.tex + robustness_section.tex + rebuttal.tex — {len(issues)} FAIL / {len(ok)} OK",
       "", "## FAILS"] + (issues or ["(nenhum)"]) + ["", "## OK"] + ok
(ROOT/"consistency_report.md").write_text("\n".join(rep), encoding="utf-8")
print("\n".join(rep[:6+len(issues)]))
print(f"\nTOTAL: {len(issues)} FAIL / {len(ok)} OK")
