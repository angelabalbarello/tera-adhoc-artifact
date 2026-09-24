"""Claims proibidos ausentes; declaracoes-chave presentes; espelhos artigo<->rebuttal."""
import re, pathlib, sys
ART = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else "/sessions/relaxed-lucid-thompson/mnt/Artigo 1/revisao-round1")
art=(ART/"artigo.tex").read_text(errors="replace")+(ART/"robustness_section.tex").read_text(errors="replace")
reb=(ART/"rebuttal.tex").read_text(errors="replace")
def test_complexity_formula_not_single_layer():
    # A formula de UMA camada nao pode aparecer sozinha para o modelo de 2 camadas:
    # toda ocorrencia de 4(DH+H^2) deve vir acompanhada do termo +8H^2 (ou ser a
    # linha do Alg.1/Tab.4 que ja soma os termos).
    import re
    for m in re.finditer(r"4\(DH\{\+\}H\^2\)\)(?!\s*\+|\+8H\^2)", art):
        ctx = art[max(0,m.start()-160):m.end()+40]
        assert "+8H^2" in ctx, "O(4(DH+H^2)) isolado p/ LSTM de 2 camadas: "+ctx[:120]

def test_forbidden():
    for w in ["rules out","bear the hypothesis out","this paper establishes","systematically exceed",
              "nine of the twelve","28\\,KFLOPs","(nats)","breaks in causal edge inference",
              "submitted version","revision verification","initially reported","originally reported","breaks in causal edge","safe at all","schedulers cannot allocate safely","necessary} condition","original four seeds"]:
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
