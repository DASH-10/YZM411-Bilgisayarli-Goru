#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""results/results.json + predictions.json + config.json  ->  rapor.pdf (en fazla 2 sayfa)
Köşeli parantezli [..] alanlar doldurulmamış yer tutucudur; PDF'te kırmızı görünür."""
import json
import os
import sys
from xml.sax.saxutils import escape

import cv2
import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.fonts import addMapping

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
load = lambda p: json.load(open(os.path.join(HERE, p), encoding="utf-8"))
cfg, pred, R = load("config.json"), load("predictions.json"), load("results/results.json")

# ---- Türkçe karakter için DejaVu (matplotlib ile birlikte gelir)
fd = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(fd, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(fd, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(fd, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", os.path.join(fd, "DejaVuSans-BoldOblique.ttf")))
for b, i, f in ((0, 0, "DV"), (1, 0, "DV-B"), (0, 1, "DV-I"), (1, 1, "DV-BI")):
    addMapping("DV", b, i, f)

BODY = ParagraphStyle("b", fontName="DV", fontSize=6.9, leading=8.4, spaceAfter=1.2)
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=11, leading=13, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=7.8, leading=9.5, spaceBefore=3, spaceAfter=1)
CAP = ParagraphStyle("c", parent=BODY, fontSize=6.2, leading=7.4, textColor=colors.HexColor("#444444"))
warnings = []


def is_ph(s):
    return isinstance(s, str) and s.strip().startswith("[") or (isinstance(s, str) and "[TAHMİN" in s)


def T(s):
    """Kullanıcı metni: yer tutucuysa kırmızı."""
    if is_ph(s):
        warnings.append(s)
        return f'<font color="red">{escape(s)}</font>'
    return escape(s)


def P(txt, st=BODY):
    return Paragraph(txt, st)


def IMG(name, width_cm):
    path = os.path.join(RES, name)
    h, w = cv2.imread(path).shape[:2]
    return Image(path, width=width_cm * cm, height=width_cm * cm * h / w)


pc = lambda x: f"%{x * 100:.2f}".replace(".", ",")
f1 = lambda x: f"{x:.1f}".replace(".", ",")
rgbs = lambda v: "(" + ", ".join(str(int(round(a))) for a in v) + ")"

e4, e5, e6, e7 = R["deney4"], R["deney5"], R["deney6"], R["deney7"]
K = ["normal", "golge", "lamba"]
LAB = {"normal": "normal", "golge": "gölge", "lamba": "lamba"}
kc = e5["kosullar"]

# ------------------------------------------------------------ otomatik cümleler
p1 = e4["points"][0]
d_g = [p1["golge"][i] - p1["normal"][i] for i in range(3)]
d_l = [p1["lamba"][i] - p1["normal"][i] for i in range(3)]
pts_txt = "; ".join(
    f"{p['ad']} (x={p['x']}, y={p['y']}): " + " / ".join(f"{LAB[k]} {rgbs(p[k])}" for k in K)
    for p in e4["points"])
ol4 = (f"img.shape=({e4['shape'][0]}, {e4['shape'][1]}, {e4['shape'][2]}), dtype={e4['dtype']}, değer aralığı "
       f"[{e4['min']}, {e4['max']}]. Erişim img[y, x, kanal]; OpenCV BGR okur, cv2.cvtColor(BGR2RGB) sonucu "
       f"img[..., ::-1] ile birebir aynı doğrulandı. Aynı fiziksel noktalar (R,G,B): {pts_txt}.")
kp = {k: kc[k]["kagit_rgb_ort"] for k in K}
yo4 = (f"P1'de (kulaklık bandı) gölgede ΔRGB=({d_g[0]:+d}, {d_g[1]:+d}, {d_g[2]:+d}), lambada ΔRGB=({d_l[0]:+d}, {d_l[1]:+d}, {d_l[2]:+d}): "
       f"lambada en çok B arttı. Beyaz kâğıt normalde {rgbs(kp['normal'])} ile sıcak tonluyken lambada {rgbs(kp['lamba'])} ile nötre yaklaştı; "
       "yani oda ışığı sarımsı, lamba daha beyaz/soğuktu. Kamera otomatik beyaz dengesi ve pozlaması da bu farka katkı yapmış olabilir. "
       "Kontrol: pozlama/beyaz dengesini kilitleyip yeniden çekmek ve kâğıt ortalamasını karşılaştırmak.")

ratio = lambda k, m: kc[k][m]["alan_orani"]
ret = lambda k, m: ratio(k, m) / ratio("normal", m) if ratio("normal", m) > 0 else float("nan")
rg_g, rg_l, hs_g, hs_l = ret("golge", "rgb"), ret("lamba", "rgb"), ret("golge", "hsv"), ret("lamba", "hsv")
ol5 = (f"Dondurulan eşikler (normal ışık, nesne ROI'sinin %{cfg['percentile'][0]}–%{cfg['percentile'][1]} yüzdelikleri): "
       f"RGB {R['deney5']['rgb_range'][0]}–{R['deney5']['rgb_range'][1]}; HSV (H 0–179, S/V 0–255) "
       f"{R['deney5']['hsv_range'][0]}–{R['deney5']['hsv_range'][1]}. Eşikler hiçbir fotoğraf için yeniden ayarlanmadı.")
if hs_g > rg_g * 1.1:
    why5 = ("HSV'de parlaklık ayrı kanalda (V) olduğundan ton (H) daha az etkilenmiş olabilir; ama S ve V eşikleri de "
            "aydınlatmadan etkilenir, bu yüzden HSV'yi dayanıklı varsaymıyorum.")
elif hs_g < rg_g * 0.9:
    why5 = ("Bu sahnede HSV, RGB'den dayanıklı çıkmadı: S ve V eşikleri gölgede değişen doygunluk/parlaklıktan "
            "etkilenmiş olabilir. HSV her aydınlatma değişimine dayanıklı değildir.")
else:
    why5 = ("RGB ve HSV benzer oranda etkilendi; HSV'nin ayrı parlaklık kanalı bu sahnede belirgin avantaj sağlamadı.")
ci = lambda k, m: pc(kc[k][m]["roi_ici_kapsama"])
yo5 = (f"Nesne ROI'sinin seçilen kısmı (RGB/HSV): normal {ci('normal','rgb')}/{ci('normal','hsv')}, gölge {ci('golge','rgb')}/{ci('golge','hsv')}, "
       f"lamba {ci('lamba','rgb')}/{ci('lamba','hsv')}. Gölgede nesne iki eşikle de kayboldu; buna rağmen RGB toplam alanı "
       f"{pc(ratio('normal','rgb'))}'den {pc(ratio('golge','rgb'))}'e arttı, çünkü normalde daha açık olan yüzeyler (bandın geri kalanı, laptop kenarı, "
       "şişe yansımaları) karararak eşik aralığına girdi. Bu yüzden alan oranı tek başına yanıltıcıdır. Lambada HSV, RGB'den kötü çıktı: nesne koyu ve "
       f"az doygun (S {R['deney5']['hsv_range'][0][1]}–{R['deney5']['hsv_range'][1][1]}) olduğu için ton (H) kararsızdır ve ışığın renk kayması H'yi dar "
       "aralığın dışına itti. Kontrol: Şekil 2'deki maskeleri gözle incelemek, ROI içi ve dışı oranları birlikte raporlamak.")

m0, mp = e6["ortalama"]["orijinal"], e6["ortalama"]["parlaklik_+40"]
exp6 = e6["beklenen_parlaklik_ort_kirpmasiz"]
s0, sp = e6["kirpilan_255_orani"]["orijinal"], e6["kirpilan_255_orani"]["parlaklik_+40"]
clipnote = (f"Ölçülen ortalama beklenenden {f1(exp6 - mp)} düşük; fark 255'te kırpılan piksellerden "
            f"gelir (255'teki değer oranı {pc(s0)} → {pc(sp)})." if exp6 - mp > 0.5 else
            "Ölçülen ortalama beklenene çok yakın; kırpma ihmal edilebilir düzeyde.")
ol6 = (f"Ortalama (RGB, 0–255): orijinal {f1(m0)}; +40 → {f1(mp)} (kırpmasız beklenen {f1(exp6)}); "
       f"γ=0,5 → {f1(e6['ortalama']['gama_0.5'])}; γ=2,0 → {f1(e6['ortalama']['gama_2.0'])}. "
       "Birim kontrol: 64 → γ=0,5'te ≈128, γ=2,0'da ≈16 (kod içinde doğrulandı).")
yo6 = (f"{clipnote} γ<1 orta/koyu tonları açar, γ>1 karartır; 0 ve 255 sabit kalır. Kontrol: dönüşümleri float "
       "üzerinde yapıp [0,255]'e kırpmak, kırpılan piksel oranını raporlamak.")

L, A = e7["lapvar"], e7["lapvar_kucuk"]
ol7 = (f"Kesit {e7['kesit'][0]}×{e7['kesit'][1]} → {e7['kucuk'][0]}×{e7['kucuk'][1]} (1/4). Laplasyen varyansı "
       f"(ince ayrıntı ölçüsü): özgün {f1(L['ozgun'])}; NEAREST→büyütülmüş {f1(L['nn_buyutulmus'])}; "
       f"AREA→büyütülmüş {f1(L['area_buyutulmus'])}. Özgüne ortalama mutlak fark: NEAREST {f1(e7['mae_ozgune']['nn'])}, "
       f"AREA {f1(e7['mae_ozgune']['area'])}; iki küçük görüntü arası fark {f1(e7['mae_nn_vs_area_kucuk'])}.")
yo7 = (f"NEAREST'te çizgiler basamaklı ve yer yer kesik, yıldızlar noktalı göründü; Laplasyen varyansının özgünün üstüne çıkması "
       f"({f1(L['nn_buyutulmus'])} > {f1(L['ozgun'])}) gerçek ayrıntı değil, bloklardan gelen sahte keskin kenarlardır. AREA çizgileri soluk ama "
       "sürekli bıraktı, sahte desen üretmedi. Neden: NEAREST her 4×4 bloktan tek pikseli seçip diğerlerini atar, AREA bloğun ortalamasını alır. "
       "Kontrol: farklı küçültme oranlarında (1/2, 1/8) aynı ölçüyü tekrarlamak.")

yok_sari = (f"Eşik yalnızca renge bakar, nesneye değil: sarıya benzeyen her yüzey seçilir. ROI dışında seçilen piksel oranı "
            f"RGB için normal/gölge/lamba: {pc(kc['normal']['rgb']['roi_disi_oran'])}/{pc(kc['golge']['rgb']['roi_disi_oran'])}/"
            f"{pc(kc['lamba']['rgb']['roi_disi_oran'])}; HSV için {pc(kc['normal']['hsv']['roi_disi_oran'])}/"
            f"{pc(kc['golge']['hsv']['roi_disi_oran'])}/{pc(kc['lamba']['hsv']['roi_disi_oran'])}. "
            "Sarımsı masa/arka plan lamba ışığında eşiğe girdiyse bu, eşiğin 'sarı olan' ile 'hedef olan'ı ayıramamasındandır "
            "[kendi sahnene göre doğrula/düzenle].")
yok_buy = (f"Küçültmede her 4×4 blok tek değere indirgendi; ayrıntının bir kısmı kalıcı olarak yok oldu. Büyütme yeni bilgi "
           f"eklemez: Laplasyen varyansı özgünde {f1(L['ozgun'])} iken büyütülmüşlerde {f1(L['nn_buyutulmus'])} (NEAREST) ve "
           f"{f1(L['area_buyutulmus'])} (AREA); özgüne fark sıfırlanmadı (MAE {f1(e7['mae_ozgune']['nn'])}/{f1(e7['mae_ozgune']['area'])}).")

cs = pred.get("cevap_sari_arka_plan") or yok_sari
cb = pred.get("cevap_buyutme") or yok_buy

# ------------------------------------------------------------ belge
story = [
    P(f"YZM 411 Bilgisayarlı Görü – Ödev 1: Renk, Aydınlatma ve Örnekleme", H1),
    P(f"{T(cfg['ad_soyad'])} – {T(cfg['ogrenci_no'])} &nbsp;|&nbsp; Sahne: masaüstü (kulaklık, kupa, çizgili kâğıt), sabit kamera/nesne konumu, "
      f"3 fotoğraf (normal, gölge, lamba). Pozlama notu: {T(cfg['otomatik_pozlama_notu'])} Pozlama/beyaz dengesi kilitli "
      "değilse farkın yalnızca ışıktan geldiği kesin söylenemez.", BODY),
    IMG("fig1_fotograflar.png", 17.4),
    P("Şekil 1. Üç özgün fotoğraf; yeşil: nesne ROI, camgöbeği: kâğıt ROI, kırmızı: yazı ROI, P1–P3: ölçüm noktaları.", CAP),
    IMG("fig2_maskeler.png", 14.6),
    P("Şekil 2. Aynı dondurulmuş eşiklerle altı maske (üst: RGB, alt: HSV); başlıktaki yüzde maske alan oranı.", CAP),
]

hdr = ["Koşul", "Kâğıt RGB ort.", "RGB alan", "HSV alan", "RGB ROI içi", "HSV ROI içi", "RGB ROI dışı", "HSV ROI dışı"]
data = [hdr]
for k in K:
    c = kc[k]
    data.append([LAB[k].capitalize(), rgbs(c["kagit_rgb_ort"]), pc(c["rgb"]["alan_orani"]), pc(c["hsv"]["alan_orani"]),
                 pc(c["rgb"]["roi_ici_kapsama"]), pc(c["hsv"]["roi_ici_kapsama"]),
                 pc(c["rgb"]["roi_disi_oran"]), pc(c["hsv"]["roi_disi_oran"])])
tb = Table(data, hAlign="LEFT")
tb.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "DV", 6.4), ("FONT", (0, 0), (-1, 0), "DV-B", 6.4),
                        ("GRID", (0, 0), (-1, -1), .3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                        ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
story += [P("Tablo 1. Karşılaştırma (alan oranı bir doğruluk ölçütü değildir; ROI dışı oran, nesne dışı yüzeylerin eşiğe girme düzeyini gösterir).", CAP), tb]


def block(title, tahmin, olcum, yorum):
    return [P(title, H2),
            P(f"<b>Tahmin (işlemden önce):</b> {T(tahmin)}"),
            P(f"<b>Ölçüm:</b> {escape(olcum)}"),
            P(f"<b>Ne gördüm / neden olabilir / nasıl kontrol ederim:</b> {escape(yorum)}")]


story += block("Deney 4 – Sayısal yapı (H1)", pred["deney4"], ol4, yo4)
story += block("Deney 5 – Sabit RGB/HSV eşikleri (H2)", pred["deney5"], ol5, yo5)
story += [IMG("fig3_donusumler.png", 14.6),
          P("Şekil 3. Orijinal, +40 parlaklık, γ=0,5 ve γ=2,0 sonuçları (başlıkta ortalama).", CAP)]
story += block("Deney 6 – Parlaklık ve gama (H3)", pred["deney6"], ol6, yo6)
story += [IMG("fig4_kucultme.png", 15.5),
          P("Şekil 4. Yazılı/çizgili kesit; küçültülüp yalnızca gösterim için NEAREST ile özgün boyuta büyütülmüştür.", CAP)]
story += block("Deney 7 – Küçültme (H4)", pred["deney7"], ol7, yo7)
tm = pred["tutmayan_tahmin"]
story += [P("Soru yanıtları", H2),
          P(f"<b>Sarı bir arka plan neden hedef gibi seçildi?</b> {T(cs)}"),
          P(f"<b>Büyütme kaybolan yazıyı neden geri getirmedi?</b> {T(cb)}"),
          P(f"<b>Tahmini tutmayan örnek:</b> {T(tm['deney'])} – {T(tm['aciklama'])}"),
          P("Yapay zekâ beyanı", H2), P(T(pred["yapay_zeka_beyani"]))]

out_pdf = os.path.join(HERE, "rapor.pdf")
SimpleDocTemplate(out_pdf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.2 * cm,
                  bottomMargin=1.2 * cm, title="YZM411 Ödev 1").build(story)

# Kaynak beyanı (zip için)
with open(os.path.join(HERE, "kaynak_beyani.md"), "w", encoding="utf-8") as f:
    f.write("# Kaynak ve yapay zekâ beyanı\n\n" + pred["yapay_zeka_beyani"] + "\n\n"
            "Kaynaklar: ders notları (Hafta 1–2), OpenCV ve NumPy belgeleri.\n\n"
            "Çekim düzeneği: iPhone 14 Pro (otomatik pozlama/beyaz dengesi açık), gölge için şemsiye, lamba çekimi için zayıf beyaz lamba; "
            "sahnede kulaklık, kupa ve çizgili kâğıt. Fotoğraflar öğrenci tarafından çekilmiştir.\n")

from pypdf import PdfReader  # noqa: E402
n = len(PdfReader(out_pdf).pages)
print(f"rapor.pdf yazıldı: {n} sayfa")
if n > 2:
    print("UYARI: 2 sayfa sınırı aşıldı! Metni kısalt veya şekilleri küçült.")
if warnings:
    print(f"UYARI: {len(warnings)} doldurulmamış [yer tutucu] var (PDF'te kırmızı). Teslimden önce hepsini doldur.")
    sys.exit(0)
