#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teslim ZIP'i: YZM411_Odev1_<ogrenciNo>.zip"""
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
no = str(cfg["ogrenci_no"])
if no.startswith("["):
    sys.exit("HATA: config.json içinde ogrenci_no doldurulmamış.")
files = ["hw1.py", "config.json", "predictions.json", "rapor.pdf", "kaynak_beyani.md",
         "results/results.csv", "results/results.json"] + \
        [f"results/fig{i}_{n}.png" for i, n in ((1, "fotograflar"), (2, "maskeler"), (3, "donusumler"), (4, "kucultme"))] + \
        [cfg["images"][k] for k in ("normal", "golge", "lamba")]
name = f"YZM411_Odev1_{no}.zip"
with zipfile.ZipFile(os.path.join(HERE, name), "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            sys.exit(f"HATA: eksik dosya {f}")
        z.write(p, f)
print("Oluşturuldu:", name, "(rapor PDF'ini ayrıca GitHub'a yüklemeyi unutma)")
