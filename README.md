# tf_teknofest_qa
Teknofest Yapay Zeka yarışmasında kullanılan Soru-Cevap modeli.

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

Eğitilmiş ağırlıklar ve FastText'in Türkçe için yayınladığı model buradan indirilip model/ klasörüne atılmalı: <a href="https://drive.google.com/file/d/1vUHA3SE2l38s3Q8BkO9yA7Lh-j1FF-Us/view">model.tar.gz/</a>
Ardından matris_sozluk_olustur.py çalıştırılarak JSON formatından soru-cevaplar çıkartılarak matris formatlarına çevrilir.
Bundan sonra modelle etkileşime geçilebilir.
