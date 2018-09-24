#! -*- coding: UTF-8 -*-
"""
Yapay Zeka modeli sınıf şeklinde inşa edilir ve tensörleri tanımlanır.
"""

import tensorflow as tf

class EncDenseModel:
	def __init__(self, mod, hParams, embedding_matris, cevap_sayisi, sess_cfg=None):
		self.mod = mod

		# Eğit modunda;
		#	Dropout uygulanır, model eğitilebilir ve maliyet hesaplatılabilir.
		# Dene modunda bunlar geçerli değildir.
		assert self.mod in ["egit", "dene"]
		
		self.hParams = hParams
		self.embedding_matris = embedding_matris
		self.cevap_sayisi = cevap_sayisi

		# JSON dosyasından okunan parametreler direk sınıfa özellik (attribute) olarak aktarılır.
		for anahtar, deger in zip(list(self.hParams.values().keys()), list(self.hParams.values().values())):
			self.__setattr__(anahtar, deger)

		# Dense katmanlarının sonuna cevap_sayisi kadar ünite sayısı olan bir katman daha eklenir.
		# Bu katman Softmax uygulanarak asıl çıktıyı belirleyecek.
		self.dense_katmanlari.append(self.cevap_sayisi)

		# Metin olarak verilen parametreler sınıflara/fonksiyonlara çevrilir.
		if self.dense_aktivasyon == "relu":
			self.dense_aktivasyon = tf.nn.relu
		elif self.dense_aktivasyon == "sigmoid":
			self.dense_aktivasyon = tf.nn.sigmoid
		elif self.dense_aktivasyon == "tanh":
			self.dense_aktivasyon = tf.nn.tanh
		if self.optimizer == "sgd":
			self.optimizer = tf.train.GradientDescentOptimizer
		elif self.optimizer == "adam":
			self.optimizer = tf.train.AdamOptimizer
		elif self.optimizer == "rmsprop":
			self.optimizer = tf.train.RMSPropOptimizer

		# Session için herhangi bir config verilmişse bunu uygular.
		# (1 CPU kullan gibi.)
		if sess_cfg != None:
			self.sess = tf.Session(config=sess_cfg)
		else:
			self.sess = tf.Session()

	# Model için girdi ve çıktıyı temsil edecek tensörler tanımlanır.
	def veri_tanimla(self):
		# Embedding matrisi sabit olarak atanır.
		self.embedding_matris = tf.constant(self.embedding_matris, dtype=tf.float32)

		# enc_xx_n: Girdi matrisini temsil eder.
		# dec_yy_n: Verilen girdi örneklerinin hedef çıktılarının indekslerini tutar.
		# dec_yy_o: Verilen girdi örneklerinin hedef çıktılarının one-hot vektör halini tutar.
		self.enc_xx_n = tf.placeholder(tf.float32, shape=(None, None, self.embedding_matris.shape[1]))
		self.dec_yy_n = tf.placeholder(tf.int32, shape=(None))
		self.dec_yy_o = tf.one_hot(self.dec_yy_n, self.cevap_sayisi)

	# Encoder inşa edilir.
	# Verilen cümledeki anlamı çıkararak sabit boyuttaki vektöre sıkıştıran modüldür.
	def encoder_insa_et(self):
		with tf.name_scope("encoder"):
			# dynamic_rnn: Cümleyi sadece baştan sona okur.
			if self.rnn_tipi == "dynamic_rnn":
				# Hücrenin çok katmanlı veya tek katmanlılığına göre inşa eder.
				if len(self.rnn_unite) == 1:
					if self.rnn_hucre == "lstm":
						self.enc_rnn_hucre = tf.nn.rnn_cell.LSTMCell(self.rnn_unite[0], state_is_tuple=False)
					elif self.rnn_hucre == "gru":
						self.enc_rnn_hucre = tf.nn.rnn_cell.GRUCell(self.rnn_unite[0])
				else:
					if self.rnn_hucre == "lstm":
						self.enc_rnn_hucre = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(unite, state_is_tuple=False) for unite in self.rnn_unite])
					elif self.rnn_hucre == "gru":
						self.enc_rnn_hucre = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(unite) for unite in self.rnn_unite])

				# Eğit modunda Dropout uygulanır.
				if self.mod == "egit":
					self.enc_rnn_hucre = tf.nn.rnn_cell.DropoutWrapper(
						self.enc_rnn_hucre, 
						input_keep_prob=self.dropout_keep_prob_rnn_input,
						output_keep_prob=self.dropout_keep_prob_rnn_output,
						state_keep_prob=self.dropout_keep_prob_rnn_state
					)

				# Cümle matris haliyle verilerek modelden geçirilir.
				self.e_out, self.e_state = tf.nn.dynamic_rnn(self.enc_rnn_hucre, self.enc_xx_n, dtype=tf.float32)

				# Sıkıştırılan anlam vektörü (thought vector) düzenlenir.
				if len(self.rnn_unite) == 1:
					self.e_final_state = self.e_state
				else:
					self.e_final_state = tf.concat(self.e_state, axis=1)

			# bidirectional_dynamic_rnn: Cümleyi hem baştan sona, hem sondan başa olarak çift yönlü okur ve bu iki bilgiyi birleştirir.
			elif self.rnn_tipi == "bidirectional_dynamic_rnn":
				# Hücrenin çok katmanlı veya tek katmanlılığına göre inşa eder.
				if len(self.rnn_unite) == 1:
					if self.rnn_hucre == "lstm":
						self.enc_rnn_hucre_fw = tf.nn.rnn_cell.LSTMCell(self.rnn_unite[0], state_is_tuple=False)
						self.enc_rnn_hucre_bw = tf.nn.rnn_cell.LSTMCell(self.rnn_unite[0], state_is_tuple=False)
					elif self.rnn_hucre == "gru":
						self.enc_rnn_hucre_fw = tf.nn.rnn_cell.GRUCell(self.rnn_unite[0])
						self.enc_rnn_hucre_bw = tf.nn.rnn_cell.GRUCell(self.rnn_unite[0])
				else:
					if self.rnn_hucre == "lstm":
						self.enc_rnn_hucre_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(unite, state_is_tuple=False) for unite in self.rnn_unite])
						self.enc_rnn_hucre_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(unite, state_is_tuple=False) for unite in self.rnn_unite])
					elif self.rnn_hucre == "gru":
						self.enc_rnn_hucre_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(unite) for unite in self.rnn_unite])
						self.enc_rnn_hucre_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(unite) for unite in self.rnn_unite])

				# Eğit modunda Dropout uygulanır.
				if self.mod == "egit":
					self.enc_rnn_hucre_fw = tf.nn.rnn_cell.DropoutWrapper(
						self.enc_rnn_hucre_fw, 
						input_keep_prob=self.dropout_keep_prob_rnn_input,
						output_keep_prob=self.dropout_keep_prob_rnn_output,
						state_keep_prob=self.dropout_keep_prob_rnn_state
					)
					self.enc_rnn_hucre_bw = tf.nn.rnn_cell.DropoutWrapper(
						self.enc_rnn_hucre_bw, 
						input_keep_prob=self.dropout_keep_prob_rnn_input,
						output_keep_prob=self.dropout_keep_prob_rnn_output,
						state_keep_prob=self.dropout_keep_prob_rnn_state
					)

				# Cümle matris haliyle verilerek modelden geçirilir. Çift yönlü okur.
				self.e_out, self.e_state = tf.nn.bidirectional_dynamic_rnn(self.enc_rnn_hucre_fw, self.enc_rnn_hucre_bw, self.enc_xx_n, dtype=tf.float32)

				# Sıkıştırılan anlam vektörü (thought vector) düzenlenir.
				if len(self.rnn_unite) == 1:
					self.e_final_state = tf.concat(self.e_state, axis=1)
				else:
					self.e_final_state = tf.concat([tf.concat(self.e_state[0], axis=1), tf.concat(self.e_state[1], axis=1)], axis=1)

	# Dense (fully-connected) katmanı. Encoder tarafından matristen çıkan
	# anlam vektörü (thought-vector) üzerinden gerekli cevap çıktısını verir.
	def dense_insa_et(self):
		with tf.name_scope("dense"):
			# Anlam vektörüne dense katmanları uygulanır.
			self.sonCikti = self.e_final_state
			for dl in range(0, len(self.dense_katmanlari)-1):
				self.sonCikti = tf.layers.dense(self.sonCikti, units=self.dense_katmanlari[dl], activation=self.dense_aktivasyon)
				if self.mod == "egit":
					self.sonCikti = tf.layers.dropout(self.sonCikti, rate=(1.0-self.dropout_keep_prob_dense))

			# Asıl çıktıyı son katman verir.
			# Bu tensörde Softmax uygulanmaz. Softmax, maliyet fonksiyonu tarafından zaten uygulanır. (tf.losses.softmax_cross_entropy)
			self.sonCikti = tf.layers.dense(self.sonCikti, units=self.dense_katmanlari[-1], activation=None)

			# Bu tensörse Softmax uygular ve sadece tahmin için kullanılır.
			self.tahmin = tf.nn.softmax(self.sonCikti)

	# Maliyet (cost) tensörü ve eğitici tensör inşa edilir.
	def egitim_op(self):
		self.loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(logits=self.sonCikti, onehot_labels=self.dec_yy_o))
		self.train = self.optimizer(self.learning_rate).minimize(self.loss)

	# Verilen veri ikililerini modele vererek gerekli optimizasyonu yapar.
	def egit(self, batch_enc_xx_n, batch_dec_yy_n):
		if self.mod == "egit":
			feed = {
				self.enc_xx_n:batch_enc_xx_n,
				self.dec_yy_n:batch_dec_yy_n
			}
			return self.sess.run(self.train, feed_dict=feed)
		else:
			raise ValueError("Egitilebilmesi icin modelin 'egit' modunda acilmasi gerek.")

	# Verilen girdi-çıktı matrislerini modele verip tahmin döndürür.
	def tahmin_et(self, batch_enc_xx_n, batch_dec_yy_n):
		feed = {
			self.enc_xx_n:batch_enc_xx_n,
			self.dec_yy_n:batch_dec_yy_n
		}
		return self.sess.run([self.enc_xx_n, self.dec_yy_o, self.tahmin], feed_dict=feed)

	# Verilen girdi matrisini modele verip sadece tahmin döndürür.
	def tahmin_et_infer(self, batch_enc_xx_n):
		feed = {
			self.enc_xx_n:batch_enc_xx_n
		}
		return self.sess.run(self.tahmin, feed_dict=feed)

	# Verilen girdi matrisini modele verip sadece anlam vektörünü (thought-vector) döndürür.
	def tahmin_et_state(self, batch_enc_xx_n):
		feed = {
			self.enc_xx_n:batch_enc_xx_n
		}
		return self.sess.run(self.e_final_state, feed_dict=feed)

	# Verilen girdi-çıktı matrislerine göre modelin ne kadar hata yaptığını döndürür.
	def maliyet(self, batch_enc_xx_n, batch_dec_yy_n):
		if self.mod == "egit":
			feed = {
				self.enc_xx_n:batch_enc_xx_n,
				self.dec_yy_n:batch_dec_yy_n
			}
			return self.sess.run(self.loss, feed_dict=feed)
		else:
			return None

	# Modeli diskten yükler.
	def yukle(self, dizin, zorunlu=False):
		try:
			self.saver.restore(self.sess, dizin)
			print("[+] Model basariyla diskten yuklendi!")
		except Exception as e:
			print("[***] Model {} adi altinda yuklenemedi.".format(dizin))

			if zorunlu:
				raise e
			else:
				print(e)

			if self.mod == "egit":
				print("[*] Model en bastan egitilecek.")

	# Modeli diske kaydeder.
	def kaydet(self, dizin):
		self.saver.save(self.sess, dizin)
		print("[+] Model basariyla diske kaydedildi.")

	# Modeli inşa edecek olan ana fonksiyonları ve tensörleri tanımlayıp çalıştırır.
	def insa_et(self):
		self.veri_tanimla()
		self.encoder_insa_et()
		self.dense_insa_et()

		if self.mod == "egit":
			self.egitim_op()

		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

		print("[+] Model insa edildi.")