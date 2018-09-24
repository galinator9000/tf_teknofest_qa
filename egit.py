#! -*- coding: UTF-8 -*-
"""
Yapay Zeka modeli EncDenseModel sınıfı ile inşa edilir ve eğitilir.
"""

import tensorflow as tf
import numpy as np
import random

from EncDenseModel import EncDenseModel
from utils import *

# Veri setinin matris hali yüklenir.
X = np.load("veri/X.npy")
y = np.load("veri/y.npy")

# Parametreler okunur.
hParams = parametre_yukle("model.json")
bucket_yapi, veri_sayisi = parametre_hazirla(hParams, "veri/tum_veri.txt")
cevaplar = [ikili.split("\t")[1] for ikili in filtrele(open("veri/tum_veri.txt", "r", encoding="utf-8").read().split("\n"))]
cevap_sayisi = len(cevaplar)

model_embedding, embedding_matris = embedding_yukle_kv("model/EmbeddingMatrisPre_kv", hParams)

# EncDenseModel sınıfı tanımlanır.
model = EncDenseModel(
	mod="egit",
	hParams=hParams,
	embedding_matris=embedding_matris,
	cevap_sayisi=cevap_sayisi,
	sess_cfg=None
)

# Model inşa edilir ve eğer ağırlık varsa yüklenir.
model.insa_et()
model.yukle("model/Agirliklar")

# Yüklenen matrisler VeriOlusturucu'ya verilir. Veri parça parça okunarak eğitilecek.
veri_olusturucu = VeriOlusturucu(X, y, hParams.train_batch_size, bucket_yapi, veri_sayisi)

# Eğit!
# Her epoch model kaydedilir.
# Her step veri setinden bir parça okunarak eğitilir. (Mini-batch)
for epoch in range(0, hParams.train_epochs):
	for step in range(0, hParams.train_steps):
		# Veriden parça alınır.
		b_x, b_y = veri_olusturucu()

		# Alınan parçayla model eğitilir.
		model.egit(b_x, b_y)

		# Her %10 ilerlemede (log_per_step_percent) eğitilen parçanın maliyeti hesaplanır.
		# Ve o parçadan rastgele bir örneğin çıktısı verilir.
		if step%(hParams.train_steps/hParams.log_per_step_percent) == 0:
			print("Epoch {} | Step {} | Batch Loss {}".format(
					epoch,
					step,
					model.maliyet(b_x, b_y)
				)
			)

			# Mini-batch'ten rastgele bir örnek alınır.
			ri = random.randint(0, b_x.shape[0]-1)
			b_x, b_y = b_x[ri], b_y[ri]
			b_x = b_x.reshape((1,) + b_x.shape)
			b_y = b_y.reshape((1,) + b_y.shape)

			# Modele verilerek çıktısı alınır.
			_input, _target, _output = model.tahmin_et(b_x, b_y)

			_inp = dizi_coz_vektor(_input[0], model_embedding, hParams).title()
			_tar = cevaplar[np.argmax(_target[0])].title()
			_out = cevaplar[np.argmax(_output[0])].title()

			print("Soru: {}".format(_inp))
			print("Cevap: {}".format(_tar))
			print("Tahmin: {}".format(_out))
			print("---------------------------------------")

	# Epoch sonu. Model diske kaydedilir.
	print("[Kayit Noktasi] Epoch {}".format(str(epoch)))
	model.kaydet("model/Agirliklar")