#! -*- coding: UTF-8 -*-
"""
Yapay Zeka modeli EncDenseModel sınıfı ile inşa edilip ağırlıklar yüklenir,
1. argüman olarak verilen dosya isminden sorular okunarak, 2. argüman olarak verilen dosya adıyla cevap çıktıları verilir.
"""

import sys
assert sys.argv[1]
girdi_dosyaAdi = sys.argv[1]
try:
	cikti_dosyaAdi = sys.argv[2]
except IndexError:
	cikti_dosyaAdi = "cevaplar.txt"

import tensorflow as tf
import numpy as np
import random

from EncDenseModel import EncDenseModel
from utils import *

# Parametreler okunur.
hParams = parametre_yukle("model.json")
bucket_yapi, veri_sayisi = parametre_hazirla(hParams, "veri/tum_veri.txt")
cevaplar = [ikili.split("\t")[1] for ikili in filtrele(open("veri/tum_veri.txt", "r", encoding="utf-8").read().split("\n"))]
cevap_sayisi = len(cevaplar)

# Embedding modelleri yüklenir.
# Eğitim sırasında FastText modelinden ayıklanmış olan vektörler kullanılır.
# Test ve çıktı sırasında direkt olarak FastText modeli kullanılır.
model_embedding, embedding_matris = embedding_yukle_kv("model/EmbeddingMatrisPre_kv", hParams)
model_embedding_pre = embedding_yukle_fasttext("model/EmbeddingFastText_TR")

# EncDenseModel sınıfı tanımlanır.
model = EncDenseModel(
	mod="dene",
	hParams=hParams,
	embedding_matris=embedding_matris,
	cevap_sayisi=cevap_sayisi,
	sess_cfg=None
)

# Model inşa edilir ve eğer ağırlık varsa yüklenir.
model.insa_et()
model.yukle("model/Agirliklar", True)

# Girdi dosyası satır satır okunur.
sorular_girdi = filtrele(open(girdi_dosyaAdi, "r", encoding="utf-8").read().split("\n"))
cevaplar_cikti = open(cikti_dosyaAdi, "w", encoding="utf-8")

# Dene!
for soru in sorular_girdi:
	# Girdi temizlenerek matrise çevrilir.
	inp = girdi_cikti_temizle(soru).strip()
	enc_x = cumle_matris(hParams, inp, model_embedding_pre)

	# Çevrilen matris modele verilerek çıktı verilir.
	_output = model.tahmin_et_infer(enc_x)
	_output = cevaplar[np.argmax(_output[0])]
	
	print(soru.title())
	print(_output.title())
	print("______________________")
	
	cevaplar_cikti.write(_output.title() + "\n")
cevaplar_cikti.close()