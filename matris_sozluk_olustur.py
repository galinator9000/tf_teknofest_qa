#! -*- coding: UTF-8 -*-
"""
Veri setinde bulunan örnekleri JSON formatından çıkartır, istenmeyen karakterlerden ayıklar.
Ardından soru-cevap ikililerini modele verilebilecek matris formatlarına dönüştürür.
"""

import numpy as np
import json, operator
from utils import *

# Veri setinin metin halini okur ve verilen parametrelerle birlikte girdi-çıktı matrislerini oluşturur.
def matris_olustur(tum_veri, cevaplar, bucket_yapi, hParams, model_embedding):
	X = []
	y = []

	girdicikti_ciftler = filtrele(tum_veri.split("\n"))

	for b in range(0, len(bucket_yapi)):
		X.append([])
		y.append([])

	for cift in girdicikti_ciftler:
		xx = []
		yy = []
		
		cift = cift.split("\t")
		girdi = filtrele(cift[0].split(" "))

		yy.append(cevaplar.index(cift[1]))

		if hParams.girdi_ters_cevir:
			girdi = girdi[::-1]

		bucketIndeks = -1
		for b in range(0, len(bucket_yapi)):
			bucket = bucket_yapi[b]
			bucketGirdi = bucket[0]
			bucketCikti = bucket[1]

			if len(girdi) <= bucketGirdi:
				bucketIndeks = b
				break

		if bucketIndeks == -1:
			continue

		for i in range(0, len(girdi)):
			try:
				xx.append(model_embedding.wv[girdi[i]])
			except:
				xx.append(model_embedding.wv[hParams.sozluk_ozel_token[3]])

		if len(xx) > 0 and len(yy) > 0:
			xx = np.array(xx)
			yy = np.array(yy)

			X[bucketIndeks].append(xx)
			y[bucketIndeks].append(yy)

	for x in range(0, len(X)):
		X[x] = np.array(X[x])
		y[x] = np.array(y[x])
	X = np.array(X)
	y = np.array(y)

	print("[*] Girdi-Cikti matrisi olusturuldu.")
	return X, y

# Kullanılan JSON formatına göre veri setinden soru-cevap ikililerini çıkartır.
# Her bir ikili için;
# Soru\tCevap\n
# şeklinde çıktı verir.
def soru_cevap_ikili_cikar(makaleler):
	basliklar = []
	veri = ""
	for makale in makaleler:
		try:
			paragraflar = makale["paragraflar"]
			baslik = makale["başlık"]

			for paragraf in paragraflar:
				paragraf_metni = paragraf["paragraf_metni"]
				soru_cevaplar = paragraf["soru_cevaplar"]

				for sc in soru_cevaplar:
					soru = sc["soru"].strip().replace("\t", "").replace("\n", "")
					cevap = sc["cevap"].strip().replace("\t", "").replace("\n", "")

					if len(soru) > 0 and len(cevap) > 0:
						veri += (soru + "\t" + cevap + "\n")
		except:
			pass
	return veri

# Soru-cevaplar JSON formatından çıkartılır ve istenmeyen karakterlerden ayıklanır.
makaleler = json.loads(open("veriseti_v1.0.json", "r", encoding="utf-8").read())["veri"]
veri = soru_cevap_ikili_cikar(makaleler)
tum_veri = girdi_cikti_temizle(veri)

# Veri setinin metin hali dosyaya yazılır.
veriF = open("veri/tum_veri.txt", "w", encoding="utf-8")
veriF.write(tum_veri)
veriF.close()

hParams = parametre_yukle("model.json")
bucket_yapi, veri_sayisi = parametre_hazirla(hParams, "veri/tum_veri.txt")
cevaplar = [ikili.split("\t")[1] for ikili in filtrele(tum_veri.split("\n"))]

# Kullanılacak Embedding modeli diskten yüklenir ve matrisler oluşturulur.
model_embedding_kv, embedding_matris = embedding_yukle_kv("model/EmbeddingMatrisPre_kv", hParams)
X, y = matris_olustur(tum_veri, cevaplar, bucket_yapi, hParams, model_embedding_kv)

# Kullanılan tüm kelimeler dosyaya yazılır.
SOZLUK = list(model_embedding_kv.wv.vocab.keys())
sozlukF = open("veri/sozluk.txt", "w", encoding="utf-8")
for kelime in SOZLUK:
	sozlukF.write(kelime + "\n")
sozlukF.close()

# Matrisler X.npy ve y.npy şeklinde kaydedilir.
np.save("veri/X", X)
np.save("veri/y", y)

print("Ornek sayisi:", len(filtrele(tum_veri.split("\n"))))
print("Sozlukteki kelime sayisi:", len(SOZLUK))
print("Olusturulan matris boyutlari:")
print("X", X.shape)
print("y", y.shape)