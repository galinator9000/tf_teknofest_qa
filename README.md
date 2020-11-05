# tf_teknofest_qa
Teknofest Yapay Zeka yarışmasında kullanılan Soru-Cevap modeli.

Bu model <b>Encoder-Fully Connected</b> yapısındadır. Ayarlanabilir <b>Encoder-Decoder</b> modeli için bu repoya bakılabilir: <a href="https://github.com/fmehmetun/tf_encdec_seq2seq">fmehmetun/tf_encdec_seq2seq</a>

# Model
<p align="center">
<img width="80%" src="img/diyagram.jpg" />
</p align="center">

<b>EncDenseModel.py:</b> Modeli Tensorflow kütüphanesiyle inşa eder.<br>
<b>egit.py:</b> Model verinin oluşturulan matris haliyle eğitilir.<br>
<b>etkilesimli.py:</b> Modelin kullanıcıyla etkileşime geçmesini sağlar.<br>
<b>tahmin_dosya.py:</b> Soruları içeren dosyayı satır satır okur ve cevapları çıkartır.<br>
<b>tahmin_dosya_json.py:</b> JSON girdisi alır ve modelden geçirerek cevapları JSON şeklinde çıkartır.<br>
<b>matris_sozluk_olustur.py:</b> Veri setindeki soru-cevap ikililerini modele verilebilecek matris formatlarına çevirir.<br>
<b>model.json:</b> Yapay zeka modelinde kullanılan parametreleri yapılandırmaya yarar.<br>

Sistem FastText'in Türkçe için yayınladığı modeli kullandığı için buradan indirilebilir: <a href="https://fasttext.cc/docs/en/pretrained-vectors.html">Turkish-bin+text</a>
Ardından matris_sozluk_olustur.py çalıştırılarak JSON formatından soru-cevaplar çıkartılarak matris formatlarına çevrilir.
