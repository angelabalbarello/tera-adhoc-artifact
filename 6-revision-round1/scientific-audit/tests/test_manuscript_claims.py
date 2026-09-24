"""Claims proibidos ausentes; declaracoes-chave presentes; espelhos artigo<->rebuttal."""
import re, pathlib, sys
ART = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else "/sessions/relaxed-lucid-thompson/mnt/Artigo 1/revisao-round1")
art=(ART/"artigo.tex").read_text(errors="replace")+(ART/"robustness_section.tex").read_text(errors="replace")
reb=(ART/"rebuttal.tex").read_text(errors="replace")
def test_forbidden():
    for w in ["rules out","bear the hypothesis out","this paper establishes","systematically exceed",
              "nine of the twelve","28\\,KFLOPs","(nats)","breaks in causal edge inference",
              "submitted version","revision verification","initially reported","originally reported","breaks in causal edge"]:
        assert w not in art, w
def test_required():
    for w in ["seven of the twelve Baseline seeds","per seed, not merely on average",
              "matches or exceeds","57.2","log_2","tab:protocol_map","median"]:
        assert w in art, w
def test_mirrors():
    n=lambda t: re.sub(r"\s+"," ",t)
    for ph in ["that a learned gate does not provide","in every evaluated seed",
               "not numerically comparable to Table","up to $0.158$"]:
        assert ph in n(art) and ph in n(reb), ph
if __name__ == "__main__":
    test_forbidden(); test_required(); test_mirrors(); print("manuscript claims OK")
