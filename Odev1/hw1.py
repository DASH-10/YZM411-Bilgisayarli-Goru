#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YZM 411 - Ödev 1: Renk, Aydınlatma ve Örnekleme  (Deney 4, 5, 6, 7)

ÇALIŞTIRMA SIRASI
  1) data/ klasörüne 3 fotoğrafı koy: normal.jpg, golge.jpg, lamba.jpg
  2) python hw1.py --init     -> config.json + predictions.json şablonu ve
                                 results/fig0_roi_onizleme.png üretir
  3) fig0'a bak, config.json'daki obj_roi / paper_roi / text_roi / points
     değerlerini kendi sahnene göre düzelt, --init'i tekrar çalıştırıp önizle
     (var olan config.json'un üzerine yazılmaz)
  4) python hw1.py            -> tüm deneyleri çalıştırır, results/ üretir
  5) predictions.json'u doldur (TAHMİNLER deney ÖNCESİ yazılmalı)
  6) python build_report.py   -> rapor.pdf
  7) python pack.py           -> YZM411_Odev1_<ogrenciNo>.zip

ROI biçimi: [x, y, genişlik, yükseklik] (piksel). Nokta biçimi: [x, y].
"""
import argparse
import json
import os
import platform
import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "config.json")
PRED_PATH = os.path.join(HERE, "predictions.json")
RES = os.path.join(HERE, "results")
KEYS = ["normal", "golge", "lamba"]
LABELS = {"normal": "Normal ışık", "golge": "Gölge", "lamba": "Lamba"}


# ----------------------------------------------------------------- yardımcılar
def crop(img, roi):
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


def py(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def small(img, max_w=700):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)


def draw_boxes(ax, cfg, scale):
    for name, col in (("obj_roi", "lime"), ("paper_roi", "cyan"), ("text_roi", "red")):
        x, y, w, h = [v * scale for v in cfg[name]]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=col, lw=1.5))
    for i, (x, y) in enumerate(cfg["points"], 1):
        ax.plot(x * scale, y * scale, "y+", ms=10, mew=1.5)
        ax.text(x * scale + 6, y * scale - 6, f"P{i}", color="yellow", fontsize=7)


def save_overview(imgs, cfg, path, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, k in zip(axes, KEYS):
        im = imgs[k]
        s = 700 / im.shape[1] if im.shape[1] > 700 else 1.0
        ax.imshow(cv2.cvtColor(small(im), cv2.COLOR_BGR2RGB))
        draw_boxes(ax, cfg, s)
        ax.set_title(LABELS[k] + title_suffix, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_images(cfg):
    imgs = {}
    for k in KEYS:
        p = os.path.join(HERE, cfg["images"][k])
        im = cv2.imread(p, cv2.IMREAD_COLOR)  # OpenCV: BGR
        if im is None:
            sys.exit(f"HATA: {p} okunamadı. Fotoğrafı data/ klasörüne koy.")
        imgs[k] = im
    shapes = {k: v.shape for k, v in imgs.items()}
    if len(set(shapes.values())) != 1:
        print("UYARI: fotoğraf boyutları farklı:", shapes,
              "- kamera/telefon konumunu sabit tuttuğundan emin ol.")
    return imgs


def align(imgs, cfg):
    """Telefon çekimler arasında az kaydıysa, gölge/lamba fotoğraflarını normal fotoğrafa
    yalnızca öteleme (translation) ile hizalar (faz korelasyonu). Kaymalar raporlanır."""
    shifts = {"normal": [0.0, 0.0]}
    if not cfg.get("hizala", True):
        return imgs, {k: [0.0, 0.0] for k in KEYS}
    ref = cv2.cvtColor(imgs["normal"], cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = ref.shape
    for k in ("golge", "lamba"):
        g = cv2.cvtColor(imgs[k], cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(ref, g)
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        imgs[k] = cv2.warpAffine(imgs[k], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        shifts[k] = [round(float(dx), 1), round(float(dy), 1)]
    print("Hizalama kaymaları (dx, dy piksel):", shifts)
    return imgs, shifts


def check_rois(cfg, shape):
    H, W = shape[:2]
    for name in ("obj_roi", "paper_roi", "text_roi"):
        x, y, w, h = cfg[name]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
            sys.exit(f"HATA: {name}={cfg[name]} görüntü ({W}x{H}) dışına taşıyor.")
    if cfg["text_roi"][2] < 64 or cfg["text_roi"][3] < 64:
        sys.exit("HATA: text_roi en az 64x64 piksel olmalı (1/4 küçültme için).")
    for (x, y) in cfg["points"]:
        if not (0 <= x < W and 0 <= y < H):
            sys.exit(f"HATA: nokta {(x, y)} görüntü dışında.")


# ----------------------------------------------------------------- şablonlar
def default_config(shape):
    H, W = shape[:2]

    def box(cx, cy, fw, fh):
        return [int(W * (cx - fw / 2)), int(H * (cy - fh / 2)), int(W * fw), int(H * fh)]

    return {
        "ad_soyad": "[AD SOYAD]",
        "ogrenci_no": "[OGRENCI_NO]",
        "images": {"normal": "data/normal.jpg", "golge": "data/golge.jpg", "lamba": "data/lamba.jpg"},
        "obj_roi": box(.50, .50, .10, .10),
        "paper_roi": box(.25, .50, .10, .10),
        "text_roi": box(.75, .50, .15, .10),
        "points": [[int(W * .48), int(H * .48)], [int(W * .50), int(H * .50)], [int(W * .52), int(H * .52)]],
        "percentile": [2, 98],
        "rgb_range": None,
        "hsv_range": None,
        "otomatik_pozlama_notu": "[Otomatik pozlama / beyaz dengesi açık mıydı? Cihaz ve ayarı yaz.]",
    }


def default_predictions():
    ph = "[TAHMİNİNİ YAZ: iki cümle - hangi ayrıntı/hata değişecek, neden?]"
    return {
        "deney4": ph, "deney5": ph, "deney6": ph, "deney7": ph,
        "tutmayan_tahmin": {
            "deney": "[Hangi deneyde tahmin tutmadı?]",
            "aciklama": "[Tahmin neydi, ölçüm neydi, neden tutmamış olabilir?]"},
        "cevap_sari_arka_plan": None,
        "cevap_buyutme": None,
        "yapay_zeka_beyani": (
            "Araç: Claude (Anthropic). Amaç: hw1.py, build_report.py ve pack.py kod iskeletinin ve "
            "rapor şablonunun yazımı. Etkilenen dosyalar: hw1.py, build_report.py, pack.py, rapor.pdf'in "
            "otomatik üretilen metin cümleleri. Fotoğraflar, ROI seçimleri, tahminler ve yorumlar "
            "öğrenciye aittir. Doğrulama: [NASIL DOĞRULADIĞINI YAZ: kodu satır satır okudum, "
            "ham piksel değerlerini elle kontrol ettim, vb.]"),
    }


# ----------------------------------------------------------------- dönüşümler
def brightness(I, b):
    return np.clip(I.astype(np.float32) + b, 0, 255)


def gamma(I, g):
    return np.clip(255.0 * (I.astype(np.float32) / 255.0) ** g, 0, 255)


def lap_var(gray):
    return float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())


# ----------------------------------------------------------------- ana akış
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    if args.init:
        first = os.path.join(HERE, "data", "normal.jpg")
        im = cv2.imread(first)
        if im is None:
            sys.exit("HATA: data/normal.jpg yok. Önce fotoğrafları data/ klasörüne koy.")
        if not os.path.exists(CFG_PATH):
            json.dump(default_config(im.shape), open(CFG_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print("config.json oluşturuldu.")
        if not os.path.exists(PRED_PATH):
            json.dump(default_predictions(), open(PRED_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print("predictions.json oluşturuldu.")
        cfg = json.load(open(CFG_PATH, encoding="utf-8"))
        imgs = load_images(cfg)
        imgs, _ = align(imgs, cfg)
        check_rois(cfg, imgs["normal"].shape)
        save_overview(imgs, cfg, os.path.join(RES, "fig0_roi_onizleme.png"))
        print("Önizleme: results/fig0_roi_onizleme.png (yeşil=nesne, camgöbeği=kâğıt, kırmızı=yazı)")
        return

    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    imgs = load_images(cfg)
    imgs, shifts = align(imgs, cfg)
    check_rois(cfg, imgs["normal"].shape)
    rows = []   # results.csv (uzun biçim)
    out = {}

    # ---------------------------------------------------------- Deney 4
    img = imgs["normal"]
    H, W, C = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert np.array_equal(rgb, img[..., ::-1]), "BGR->RGB dönüşümü beklenenle aynı olmalı"
    e4 = {"shape": [H, W, C], "dtype": str(img.dtype), "min": int(img.min()), "max": int(img.max()),
          "points": []}
    for i, (x, y) in enumerate(cfg["points"], 1):
        entry = {"ad": f"P{i}", "x": x, "y": y}
        for k in KEYS:
            b, g, r = [int(v) for v in imgs[k][y, x]]            # NumPy erişimi: img[y, x, kanal]
            entry[k] = [r, g, b]                                 # RGB sırasına çevrildi
            patch = imgs[k][max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].reshape(-1, 3)
            entry[k + "_5x5"] = [round(float(v), 1) for v in patch.mean(0)[::-1]]
            rows.append(["4", k, f"P{i}_R,G,B", f"{r},{g},{b}"])
        e4["points"].append(entry)
    out["deney4"] = e4
    print(f"[Deney 4] shape=({H},{W},{C}) dtype={img.dtype} aralık=[{img.min()},{img.max()}]")
    for p in e4["points"]:
        print("   ", p["ad"], (p["x"], p["y"]), {k: p[k] for k in KEYS})

    # ---------------------------------------------------------- Deney 5
    roi = crop(imgs["normal"], cfg["obj_roi"])
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    plo, phi = cfg.get("percentile", [2, 98])
    if cfg.get("rgb_range"):
        rlo, rhi = cfg["rgb_range"]
    else:
        rlo = np.floor(np.percentile(rgb_roi, plo, axis=0)).astype(int).tolist()
        rhi = np.ceil(np.percentile(rgb_roi, phi, axis=0)).astype(int).tolist()
    if cfg.get("hsv_range"):
        hlo, hhi = cfg["hsv_range"]
    else:
        hlo = np.floor(np.percentile(hsv_roi, plo, axis=0)).astype(int).tolist()
        hhi = np.ceil(np.percentile(hsv_roi, phi, axis=0)).astype(int).tolist()
    print(f"[Deney 5] DONDURULDU  RGB {rlo}..{rhi}   HSV(OpenCV H0-179,S,V0-255) {hlo}..{hhi}")

    masks, e5 = {}, {"rgb_range": [rlo, rhi], "hsv_range": [hlo, hhi], "kosullar": {}}
    ox, oy, ow, oh = cfg["obj_roi"]
    for k in KEYS:
        bgr = imgs[k]
        m_rgb = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                            np.array(rlo, np.uint8), np.array(rhi, np.uint8))
        m_hsv = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
                            np.array(hlo, np.uint8), np.array(hhi, np.uint8))
        masks[k] = {"rgb": m_rgb, "hsv": m_hsv}
        paper = crop(bgr, cfg["paper_roi"]).reshape(-1, 3).mean(0)[::-1]
        d = {"kagit_rgb_ort": [round(float(v), 1) for v in paper]}
        for name, m in (("rgb", m_rgb), ("hsv", m_hsv)):
            tot = int(np.count_nonzero(m))
            inside = int(np.count_nonzero(m[oy:oy + oh, ox:ox + ow]))
            d[name] = {
                "alan_orani": tot / m.size,
                "roi_ici_kapsama": inside / (ow * oh),
                "roi_disi_oran": (tot - inside) / (m.size - ow * oh),
            }
            for met, val in d[name].items():
                rows.append(["5", k, f"{name}_{met}", f"{val:.6f}"])
        e5["kosullar"][k] = d
        print(f"   {k:7s} alan oranı RGB={d['rgb']['alan_orani']*100:.2f}%  "
              f"HSV={d['hsv']['alan_orani']*100:.2f}%   ROI dışı RGB={d['rgb']['roi_disi_oran']*100:.2f}% "
              f"HSV={d['hsv']['roi_disi_oran']*100:.2f}%")
    out["deney5"] = e5

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for j, k in enumerate(KEYS):
        for i, name in enumerate(("rgb", "hsv")):
            axes[i, j].imshow(small(masks[k][name], 500), cmap="gray", vmin=0, vmax=255)
            axes[i, j].set_title(f"{name.upper()} maske - {LABELS[k]}  "
                                 f"(%{e5['kosullar'][k][name]['alan_orani']*100:.2f})", fontsize=8)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig2_maskeler.png"), dpi=150)
    plt.close(fig)
    save_overview(imgs, cfg, os.path.join(RES, "fig1_fotograflar.png"))

    # ---------------------------------------------------------- Deney 6
    I = cv2.cvtColor(imgs["normal"], cv2.COLOR_BGR2RGB).astype(np.float32)
    print("[Deney 6] doğrulama: gamma(64,0.5)=%.1f gamma(64,2.0)=%.1f" %
          (gamma(np.array([64.]), .5)[0], gamma(np.array([64.]), 2.0)[0]))
    variants = {"orijinal": I, "parlaklik_+40": brightness(I, 40),
                "gama_0.5": gamma(I, 0.5), "gama_2.0": gamma(I, 2.0)}
    e6 = {"ortalama": {}, "kirpilan_255_orani": {}}
    for name, J in variants.items():
        e6["ortalama"][name] = float(J.mean())
        e6["kirpilan_255_orani"][name] = float((J >= 255).mean())
        rows.append(["6", name, "ortalama", f"{J.mean():.3f}"])
        rows.append(["6", name, "255_orani", f"{(J >= 255).mean():.6f}"])
        print(f"   {name:14s} ort={J.mean():7.2f}  255'te piksel oranı={(J >= 255).mean()*100:.3f}%")
    e6["beklenen_parlaklik_ort_kirpmasiz"] = float(I.mean() + 40)
    out["deney6"] = e6
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))
    for ax, (name, J) in zip(axes, variants.items()):
        ax.imshow(small(J.astype(np.uint8), 400))
        ax.set_title(f"{name}\nort={J.mean():.1f}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig3_donusumler.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- Deney 7
    tc = crop(imgs["normal"], cfg["text_roi"])
    h, w = tc.shape[:2]
    sw, sh = max(w // 4, 1), max(h // 4, 1)
    nn = cv2.resize(tc, (sw, sh), interpolation=cv2.INTER_NEAREST)
    ar = cv2.resize(tc, (sw, sh), interpolation=cv2.INTER_AREA)
    nn_up = cv2.resize(nn, (w, h), interpolation=cv2.INTER_NEAREST)   # yalnızca gösterim için
    ar_up = cv2.resize(ar, (w, h), interpolation=cv2.INTER_NEAREST)
    g = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    mae = lambda a, b: float(np.abs(g(a).astype(np.float32) - g(b).astype(np.float32)).mean())
    e7 = {"kesit": [w, h], "kucuk": [sw, sh],
          "lapvar": {"ozgun": lap_var(g(tc)), "nn_buyutulmus": lap_var(g(nn_up)), "area_buyutulmus": lap_var(g(ar_up))},
          "lapvar_kucuk": {"nn": lap_var(g(nn)), "area": lap_var(g(ar))},
          "mae_ozgune": {"nn": mae(tc, nn_up), "area": mae(tc, ar_up)},
          "mae_nn_vs_area_kucuk": mae(nn, ar)}
    out["deney7"] = e7
    for k_, v_ in e7.items():
        rows.append(["7", "-", k_, json.dumps(v_)])
    print("[Deney 7]", json.dumps(e7, indent=None))
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, (t, a) in zip(axes, (("Özgün kesit", tc), ("1/4 NEAREST → büyütülmüş", nn_up),
                                  ("1/4 AREA → büyütülmüş", ar_up))):
        ax.imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB), interpolation="nearest")
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig4_kucultme.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- ortam kaydı
    out["ortam"] = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "opencv": cv2.__version__, "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "girdi_dosyalari": cfg["images"], "hizalama_kaymasi_px": shifts, "parametreler": {
            k: cfg[k] for k in ("obj_roi", "paper_roi", "text_roi", "points", "percentile", "rgb_range", "hsv_range")},
        "otomatik_pozlama_notu": cfg["otomatik_pozlama_notu"],
        "calistirma_sirasi": "hw1.py --init -> hw1.py -> build_report.py -> pack.py"}
    json.dump(out, open(os.path.join(RES, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=py)
    pd.DataFrame(rows, columns=["deney", "kosul", "olcut", "deger"]).to_csv(
        os.path.join(RES, "results.csv"), index=False, encoding="utf-8")
    print("\nTamam: results/results.json, results.csv, fig1-fig4 yazıldı.")


if __name__ == "__main__":
    main()
