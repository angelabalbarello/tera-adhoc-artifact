"""Patch da Fig.2 (2-fig_tera_pipeline_final.png -> _v2.png): alinha o ramo online
a semantica avaliada (janela recomputada; supressao reusa so a probabilidade
emitida). 4 inscricoes alteradas; todo o resto pixel-identico. Original preservado."""
from PIL import Image, ImageDraw, ImageFont
B="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
im=Image.open("2-fig_tera_pipeline_final.png").convert("RGB")
d=ImageDraw.Draw(im)
def F(sz): return ImageFont.truetype(B,sz)
def draw_sub(x,y,parts,fill,main=34,sub=24):
    """parts: list of (text, is_sub). Retorna largura total. y = topo do texto principal."""
    cx=x
    for t,is_s in parts:
        f=F(sub if is_s else main)
        yy=y+(main-sub)+6 if is_s else y
        d.text((cx,yy),t,font=f,fill=fill)
        cx+=d.textlength(t,font=f)
    return cx-x
def center_sub(cx_center,y,parts,fill,main=34,sub=24):
    w=sum(d.textlength(t,font=F(sub if s else main)) for t,s in parts)
    draw_sub(cx_center-w/2,y,parts,fill,main,sub)

SLATE=(61,68,81); GREEN=(26,122,74); GRAY=(138,143,152)

# cores de fundo amostradas
bg_reuse=im.getpixel((2350,1420)); bg_sel=im.getpixel((2740,1195)); bg_fb=(255,255,255); bg_ca=(255,255,255)

# 1) Causal edge model: subtitulo "updates h_t, p_t, H_t" -> "recomputes causal window: p_t, H_t"
d.rectangle((2130,1170,2600,1225),fill=bg_ca)
center_sub(2365,1182,[("recomputes window: ",0),("p",0),("t",1),(", H",0),("t",1)],GREEN,30,22)

# 2) Reuse previous state -> Reuse emitted probability ; sub reuses p_{t-1}, H_{t-1}
d.rectangle((2100,1335,2620,1395),fill=bg_reuse)   # titulo
d.rectangle((2100,1400,2620,1450),fill=bg_reuse)   # sub
f=F(35); t="Reuse emitted probability"
d.text((2360-d.textlength(t,font=f)/2,1344),t,font=f,fill=SLATE)
center_sub(2360,1405,[("reuses p",0),("t−1",1),(", H",0),("t−1",1)],(90,103,124),32,23)

# 3) Selected recurrent state -> Selected output ; h_t,p_t,H_t -> p_t, H_t
d.rectangle((2700,1205,3065,1255),fill=bg_sel)   # titulo
d.rectangle((2700,1258,3065,1300),fill=bg_sel)   # sub
f=F(34); t="Selected output"
d.text((2882-d.textlength(t,font=f)/2,1212),t,font=f,fill=GREEN)
center_sub(2882,1260,[("p",0),("t",1),(", H",0),("t",1)],GREEN,32,23)

# 4) feedback pill: remove h_{t-1}
d.rectangle((1912,1548,2356,1634),fill=bg_fb)
d.rounded_rectangle((1948,1556,2332,1626),radius=34,outline=GREEN,width=3,fill=(255,255,255))
center_sub(2140,1574,[("feedback: p",0),("t−1",1),(", H",0),("t−1",1)],GREEN,28,20)

im.save("2-fig_tera_pipeline_final_v2.png")
print("salvo _v2")
