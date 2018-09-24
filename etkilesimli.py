#! -*- coding: UTF-8 -*-
"""
Yapay Zeka modeli EncDenseModel sınıfı ile inşa edilip ağırlıklar yüklenir,
kullanıcının etkileşime geçmesi sağlanır. (Inference)
"""

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

model = EncDenseModel(
	mod="dene",
	hParams=hParams,
	embedding_matris=embedding_matris,
	cevap_sayisi=cevap_sayisi,
	sess_cfg=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
)

# Model inşa edilir ve eğer ağırlık varsa yüklenir.
model.insa_et()
model.yukle("model/Agirliklar", True)

print("Cikis yapmak icin SONLANDIR yaziniz.")

# Dene!
while True:
	# Kullanıcının girdisi alınır.
	inp = input("> ")
	if inp == "SONLANDIR":
		break

	# Girdi temizlenerek matrise çevrilir.
	inp = girdi_cikti_temizle(inp).strip()
	enc_x = cumle_matris(hParams, inp, model_embedding_pre)

	# Çevrilen matris modele verilerek çıktı verilir.	
	_output = model.tahmin_et_infer(enc_x)
	_output = cevaplar[np.argmax(_output[0])]
	print(_output.title())