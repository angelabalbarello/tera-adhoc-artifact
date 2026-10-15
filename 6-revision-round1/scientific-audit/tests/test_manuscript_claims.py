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
              "submitted version","revision verification","initially reported","originally reported","breaks in causal edge","safe at all","schedulers cannot allocate safely","necessary} condition","original four seeds","but is safe only if uncertainty rises","published campaign test partitions",
              "Cliff","adequate statistical power"]:
        assert w not in art, w
def test_required():
    for w in ["seven of the twelve Baseline seeds","per seed, not merely on average",
              "matches or exceeds","57.2","log_2","tab:protocol_map","median","matched-pairs rank-biserial","kerby2014simple",
              "primary operational endpoint","safety constraint","empirically dissociable under this protocol"]:
        assert w in art, w
def test_mirrors():
    n=lambda t: re.sub(r"\s+"," ",t)
    for ph in ["that a learned gate does not provide","in every evaluated seed",
               "not numerically comparable to Table","up to $0.158$",
               "empirically dissociable under this protocol","matched-pairs rank-biserial"]:
        assert ph in n(art) and ph in n(reb), ph
def test_final_audit_locks():
    # TTD_ef: definicao unica agregada (a forma por casos foi removida)
    assert "otherwise (missed episode)" not in art
    assert "single definition used by every table" in art
    # linguagem de poder estatistico nao sustentada
    for w in ["statistical-power weakness","adequate statistical power","statistical power"]:
        assert w not in art and w not in reb, w
    # contagem unica do abstract
    assert "224 words" not in reb and "226 words" not in reb and "227 words" not in reb and "228 words" not in reb and "231 words" not in reb
    assert reb.count("220 words") == 4
    # claim de baseline: escopo RNN/LSTM/GRU, nunca "causal recurrent" amplo nem inexistencia
    assert "causal recurrent" not in reb
    assert "No published uncertainty-aware" not in reb
    assert "single-node causal RNN/LSTM/GRU" in reb
    # rodada dirigida 11/10: locks
    for w in ["miss rate unchanged","informative rather than flattering","a ceiling effect rather",
              "introduced in response to this result","corrected replay logs","owed to the reader",
              "rather than by implication","cannot reorganize the temporal structure"]:
        assert w not in art, w
    for w in ["informative rather than flattering","a ceiling effect rather",
              "introduced in response to this result","corrected replay logs"]:
        assert w not in reb, w  # "miss rate unchanged" aparece 1x no reb apenas citando a frase corrigida (Add.10)
    assert "pre-onset anticipation" in art
    # rodada v7: locks
    for w in ["jointly low","matters epistemologically","semantic grounding","organizationally\ngrounded",
              "without altering the temporal ordering","stays qualitatively distinct"]:
        assert w not in art, w
    assert "states the semantics that every scheduled" in art
    assert "the causal prefix window of the" in art
    # v8: locks
    for w in ["static RAM","frontier in Figure","Frame-level ECE degrades","Reaching it\n"]:
        assert w not in art, w
    for w in ["per recurrent step","5.49","11.9","retrospective detection markers","episode-level"]:
        assert w in art, w
    # Skip-RNN nunca classificado como uncertainty-aware
    assert "Skip-RNN is not an uncertainty-aware method" in art
if __name__ == "__main__":
    test_forbidden(); test_required(); test_mirrors(); test_final_audit_locks(); print("manuscript claims OK")
