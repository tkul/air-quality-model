**\[Slayt 1 – Giriş ve Amaç]**
Merhaba, biz bu projeyi hava kirliliğini tahmin edebilmek için geliştirdik.

Özellikle PM2.5 değeri, solunum ve kalp hastalıkları gibi sağlık sorunlarıyla doğrudan ilişkili. Ancak birçok şehirde bu veri ya ölçülmüyor ya da eksik.

Bu projede amacımız:
📌 Sınırlı veriye rağmen PM2.5 değerini tahmin edebilen bir model geliştirmek
📌 Bu modeli herkesin kullanabileceği bir arayüzle sunmak

**Nasıl Kullanılır?**
Kullanıcılar basit çevresel verileri girerek, yaşadıkları bölgedeki tahmini PM2.5 seviyesini öğrenebiliyor.

**\[Slayt 3 – Dataset ve Özellikler]**
Kullandığımız veri seti, Dünya Sağlık Örgütü’ne ait şehir bazlı hava kalitesi verilerinden oluşuyor.
Veri setinde PM10, NO2 gibi ölçümler ve ülkelerin WHO bölgesel sınıflandırmaları gibi bilgiler yer alıyor.

**\[Slayt 4 – Korelasyon Matrisi] (correlation\_matrix.png)**
Özellikler arasındaki ilişkileri görmek için bir korelasyon matrisi oluşturduk. Bu sayede hangi değişkenlerin tahmin üzerinde etkili olduğunu daha iyi anlayabildik.

**\[Slayt 5 – Model ve Yöntem]**
Model olarak XGBoost tercih ettik çünkü doğruluğu yüksek ve büyük veriyle iyi çalışıyor.
Veriyi sayısal, kategorik ve sıralı olarak işleyip, uygun dönüşümleri uyguladık.
Modeli Optuna ile optimize ettik; bu da en iyi parametre kombinasyonunu otomatik olarak bulmamızı sağladı.

**\[Slayt 6 – Öznitelik Önemi] (feature\_importance.png ve optimized\_feature\_importance.png)**
Eğitilen modellerin özellik önem grafikleri bize özellikle *WHO Region*’un PM2.5 üzerinde büyük bir etkisi olduğunu gösterdi.
Modeli optimize ettiğimizde, bazı bölgesel değişkenlerin etkisinin arttığını gözlemledik.

**\[Slayt 7 – Gerçek vs Tahmin] (xgboost\_actual\_vs\_predicted.png ve optimized\_xgboost\_actual\_vs\_predicted.png)**
Modelimizin ne kadar başarılı olduğunu görmek için gerçek PM2.5 değerleriyle tahminleri karşılaştırdık.
Grafiklerde gördüğünüz üzere modelimiz oldukça tutarlı sonuçlar vermektedir, özellikle düşük ve orta seviyelerde.

**\[Slayt 8 – Uygulama Arayüzü]**
Projeyi sadece analizle bırakmadık, bir arayüz de geliştirdik.
Streamlit kullanarak bir web uygulaması oluşturduk. Kullanıcı, bölge ve ölçüm bilgilerini girerek tahmini PM2.5 seviyesini anında görebiliyor.
\[Canlı demo linki burada gösterilebilir]

**\[Slayt 9 – Farklılık ve Katkımız]**
Peki biz neyi farklı yaptık?

* Mevcut verilerdeki eksikleri işleyerek temiz bir model çıkardık.
* Modeli Optuna ile optimize ettik.
* Sonuçları sade ama güçlü bir arayüzle kullanıma sunduk.

**\[Slayt 10 – Sonraki Adımlar]**
Bu model gelecekte hava kirliliği tahmini için daha fazla sensör verisiyle geliştirilebilir.
Ayrıca cep telefonu uygulaması ile geniş kitlelere yayılabilir.

**\[Slayt 11 – Teşekkürler]**
Beni dinlediğiniz için teşekkür ederim.
