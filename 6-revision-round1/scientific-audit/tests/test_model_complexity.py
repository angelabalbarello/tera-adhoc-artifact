"""58.242 params exatos; 57.2K MACs; ~115 KFLOPs (1 MAC = 2 FLOPs)."""
D,H = 31,64
def test_closed_form():
    params = 4*(D*H+H*H+2*H) + 4*(2*H*H+2*H) + 2*(H+1)
    assert params == 58242
    macs = 4*(D*H+H*H) + 4*(2*H*H) + 2*H
    assert macs == 57216
    flops = 2*macs + (4*2*H*2 + 2)  # +bias adds
    assert 114000 < flops < 117000
if __name__ == "__main__":
    test_closed_form(); print("complexity OK")
