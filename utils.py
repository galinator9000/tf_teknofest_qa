#! -*- coding: UTF-8 -*-
"""
Diğer modüller tarafından ortak kullanılan önemli fonksiyonları içerir.
"""

import tensorflow as tf
import numpy as np
import re

from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText

# Verilen liste objesindeki boş elemanları filtreler.
def filtrele(x):
	return list(filter(None, x))

# Verilen cümleyi istenmeyen karakterlerden ayıklar.
def girdi_cikti_temizle(veri):
	kynC = "ÂâÅÉáäæèéíóúûĀāÎîġĥħīķōśŧūŻżḤḥḪḫḲḳṢṣṬṭẓ‘’“”/'-"
	hdfC = "AaAEaaaeeiouuAaİighhikostuZzHhHhKkŞşTtz''\"\"   "
	for char in range(0, len(kynC)):
		veri = veri.replace(kynC[char], hdfC[char])
	
	veri = veri.replace("ß", "ss").replace("%", " yüzde ")
	veri = re.sub(r"""[^A-Za-z0-9ÇçÖöÜüĞğİıŞş \n\t]""", " ", veri)

	for i in range(0, 100):
		veri = veri.replace("  ", " ")
	veri = veri.replace("\n ", "\n").replace(" \n", "\n").replace("\t ", "\t").replace(" \t", "\t").replace(" '", "'").replace("' ", "'")
	return veri.lower()

# Verilen (Kelime sayısı, Kelime özelliği) şekline sahip olan matrisi Embedding modeli üzerinden geri cümleye çevirir.
def dizi_coz_vektor(x, model_embedding, hParams):
	cumle = ""
	for t in range(0, x.shape[0]):
		try:
			enBenzer = model_embedding.wv.similar_by_vector(x[t])[0]
		except IndexError:
			cumle += hParams.sozluk_ozel_token[3] + " "
			continue
		cumle += enBenzer[0] + " "
	cumle = cumle.strip()
	return cumle

# Verilen cümleyi kelimelere ayırarak cümleyi temsil edecek olan matrisi Embedding modeli üzerinden çevirir.
def cumle_matris(hParams, x, model_embedding):
	x = filtrele(x.split(" "))

	if hParams.girdi_ters_cevir:
		x = x[::-1]

	r = []
	for xx in x:
		try:
			r.append(model_embedding.wv[xx])
		except:
			r.append(model_embedding.wv[hParams.sozluk_ozel_token[3]])

	r = np.array(r)
	r = r.reshape((1,) + r.shape)
	return r

# Verilen kelime listesinden kelimelerin Embedding karşılıklarını tek tek
# yine verilen modelden alarak bir KeyedVectors modeli döndürür.
def embedding_ayir(dizin, kelimeler, model_embedding, hParams):
	model_embedding_kv = KeyedVectors(hParams.embedding_matris_boyut)

	for kelime in kelimeler:
		try:
			model_embedding_kv.add([kelime], [model_embedding.wv[kelime]])
		except:
			pass

	model_embedding_kv.save(dizin)
	return model_embedding_kv

# embedding_ayir fonksiyonu tarafından oluşturulan KeyedVectors modelini diskten yükler.
def embedding_yukle_kv(dizin, hParams):
	try:
		model_embedding = KeyedVectors(hParams.embedding_matris_boyut)
		model_embedding = model_embedding.load(dizin)
		print("[+] Embedding matrisi {} dosyasindan yuklendi.".format(dizin))
		return (model_embedding, model_embedding.wv.syn0)
	except:
		raise FileNotFoundError("[!] Embedding matrisi {} dosyasindan yuklenemedi.".format(dizin))

# fasttext.cc/docs/en/crawl-vectors.html
# FastText'in Wikipedia üzerinden eğitmiş olduğu Embedding modelini diskten yükler.
def embedding_yukle_fasttext(dizin):
	try:
		model_embedding = FastText.load_fasttext_format(dizin)
		print("[+] Embedding modeli (pretrained) {} dosyasindan yuklendi.".format(dizin))
		return model_embedding
	except:
		raise FileNotFoundError("[!] Embedding modeli (pretrained) {} dosyasindan yuklenemedi.".format(dizin))

# Veri setinin eğitim sırasında modele parça parça verilebilmesini sağlar.
# Sınıf oluşturulduktan sonra çağrılarak verilen veri setinden parçalar her çağrışta alınabilir.
# Veri setinin üstünden geçildiğinde başa döner.

# X: (Bucket Sayısı,), y: (Bucket Sayısı,)

# Ve her bir bucket;
# X için (Örnek, Kelime, Kelime özelliği)
# y için (Örnek,)
# şekline sahip olmalıdır.
class VeriOlusturucu:
	def __init__(self, X, y, batch_size, bucket_yapi, veri_sayisi):
		self.X = X
		self.y = y
		self.batch_size = batch_size
		self.bucket_yapi = bucket_yapi
		self.veri_sayisi = veri_sayisi

		self.veri = 0
		self.bucket = 0
	def __call__(self):
		if self.bucket >= len(self.bucket_yapi):
			self.bucket = 0
		while self.X[self.bucket].shape[0] == 0 and self.y[self.bucket].shape[0] == 0:
			self.bucket += 1
			if self.bucket >= len(self.bucket_yapi):
				self.bucket = 0

		if self.veri_sayisi[self.bucket] <= (self.veri+self.batch_size) and self.veri_sayisi[self.bucket] >= self.veri:
			xx = self.X[self.bucket][self.veri:]
			yy = self.y[self.bucket][self.veri:]

			self.veri = 0
			self.bucket += 1
		else:
			xx = self.X[self.bucket][self.veri:(self.veri+self.batch_size)]
			yy = self.y[self.bucket][self.veri:(self.veri+self.batch_size)]
			self.veri += self.batch_size
		return xx, yy

# Model için gerekli parametreleri JSON formatından okuyarak geçerliliğini kontrol eder.
def parametre_yukle(dosya):
	hParams = tf.contrib.training.HParams(
		rnn_unite=[None],
		rnn_hucre=None,
		rnn_tipi=None,
		dense_katmanlari=[None],
		dense_aktivasyon=None,
		optimizer=None,
		learning_rate=None,
		dropout_keep_prob_dense=None,
		dropout_keep_prob_rnn_input=None,
		dropout_keep_prob_rnn_output=None,
		dropout_keep_prob_rnn_state=None,
		train_epochs=None,
		train_steps=None,
		train_batch_size=None,
		log_per_step_percent=None,
		embedding_matris_boyut=None,
		sozluk_ozel_token=[None],
		girdi_ters_cevir=None
	).parse_json(open(dosya, "r").read())

	assert len(hParams.rnn_unite) > 0
	assert hParams.rnn_hucre in ["lstm", "gru"]
	assert hParams.rnn_tipi in ["bidirectional_dynamic_rnn", "dynamic_rnn"]
	assert hParams.dense_aktivasyon in ["relu", "sigmoid", "tanh", None]
	assert hParams.optimizer in ["sgd", "adam", "rmsprop"]
	assert hParams.dropout_keep_prob_dense > 0.0 and hParams.dropout_keep_prob_dense <= 1.0
	assert hParams.dropout_keep_prob_rnn_input > 0.0 and hParams.dropout_keep_prob_rnn_input <= 1.0
	assert hParams.dropout_keep_prob_rnn_output > 0.0 and hParams.dropout_keep_prob_rnn_output <= 1.0
	assert hParams.dropout_keep_prob_rnn_state > 0.0 and hParams.dropout_keep_prob_rnn_state <= 1.0
	assert hParams.embedding_matris_boyut == 300
	assert len(hParams.sozluk_ozel_token) == 4
	return hParams

# Veri setinin metin halini içeren dosyayı okuyarak Bucketing için gerekli olan parametreleri hazırlar.
def parametre_hazirla(hParams, tum_veri_dizin):
	tum_veri = open(tum_veri_dizin, "r", encoding="utf-8").read()
	tum_veri_ikili = filtrele(tum_veri.split("\n"))
	tum_veri_ikili = [filtrele(ikili.split("\t")) for ikili in tum_veri_ikili if len(filtrele(ikili.split("\t"))) == 2]

	bucket_yapi = []
	bucket_girdi = []
	bucket_cikti = []

	for (girdi, cikti) in tum_veri_ikili:
		girdi = filtrele(girdi.split(" "))
		
		bucket_girdi.append(len(girdi))
		bucket_cikti.append(1)

	bucket_girdi = sorted(list(set(bucket_girdi)))
	bucket_cikti = sorted(list(set(bucket_cikti)))

	for bg in bucket_girdi:
			for bc in bucket_cikti:
				bucket_yapi.append((bg, bc))

	veri_sayisi = []
	for (bg, bc) in bucket_yapi:
		veri_sayisi_cur_bucket = 0
		for (girdi, cikti) in tum_veri_ikili:
			girdi = filtrele(girdi.split(" "))

			if len(girdi) == bg and bc == 1:
				veri_sayisi_cur_bucket += 1
		veri_sayisi.append(veri_sayisi_cur_bucket)

	return bucket_yapi, veri_sayisi