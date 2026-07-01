# Sistemin Gerçeklenmesi ve Bulgular

## Sistemin Gerçeklenmesi

Bu projede Bitcoin fiyat hareketlerinin tahmininde Twitter/X gönderilerinden elde edilen duygu verisinin makine öğrenmesi başarımına etkisi incelenmiştir. Sistem; fiyat verisi toplama, tweet verisi toplama, metin temizleme, duygu skoru ekleme, etkileşim ağırlığı üretme, 4 saatlik zaman pencerelerinde veri birleştirme, model eğitimi ve sonuçların arayüzde görüntülenmesi aşamalarından oluşmaktadır.

Fiyat verisi `CoinGecko` üzerinden alınmış ve `data/raw/bitcoin_price_raw.csv` dosyasına kaydedilmiştir. Ham fiyat dosyasında tarih, UTC zaman bilgisi, milisaniye zaman damgası, Bitcoin fiyatı, piyasa değeri ve işlem hacmi bulunmaktadır. Tweet verisi `GetXAPI` üzerinden alınmış ve `data/raw/bitcoin_getxapi_tweets_raw.csv` dosyasında saklanmıştır. Tweet toplama aşamasında İngilizce tweetler, retweet olmayan gönderiler ve yanıt olmayan gönderiler seçilerek veri gürültüsü azaltılmıştır.

Ham tweet verisi doğrudan modele verilmemiştir. Önce gereksiz kolonlar çıkarılmış, ardından tweet metinleri temizlenmiştir. Temizleme aşamasında linkler ve emojiler kaldırılmış, hashtag içindeki kelimeler korunmuş, mention ifadelerinde yalnızca `@` sembolü silinmiştir. Böylece metnin duygu analizi için anlamlı kalması amaçlanmıştır. Daha sonra her tweet için sentiment skoru ve sentiment etiketi eklenmiştir. Etkileşim değerlerinden ayrıca `engagement_score` ve `engagement_weight` hesaplanmıştır. Bu ağırlık, yüksek görüntülenme, beğeni, yanıt ve paylaşım alan tweetlerin duygu skoruna daha fazla etki etmesini sağlamaktadır.

Son model veri seti `data/processed/final_dataset.csv` dosyasında oluşturulmuştur. Bu dosyada fiyat verileri ve tweet duygu verileri 4 saatlik zaman pencerelerinde birleştirilmiştir. Her pencere için fiyat açılış, kapanış, en yüksek, en düşük, ortalama fiyat, hacim, piyasa değeri, tweet sayısı, ortalama sentiment, pozitif/negatif tweet oranları ve ağırlıklı sentiment değerleri hesaplanmıştır. Ayrıca geçmişe dönük 24 saatlik ve 7 günlük kayan pencere özellikleri eklenmiştir. Bu özelliklerde gelecek veri kullanılmamış, veri sızıntısını önlemek için tüm rolling hesaplamalar yalnızca mevcut ve geçmiş satırlar üzerinden yapılmıştır.

Model eğitimi `scripts/run_model_training.py` ile çalıştırılmaktadır. Eğitimde kronolojik bölme yöntemi kullanılmıştır. Veri rastgele bölünmemiştir; ilk %70 eğitim, sonraki %15 doğrulama, son %15 test kümesi olarak ayrılmıştır. Bu yöntem finansal zaman serilerinde gelecek bilginin geçmiş modele sızmasını engellemek için tercih edilmiştir.

Kullanıcı arayüzü `Tkinter` ile geliştirilmiştir. Arayüz ilk açıldığında kullanıcıya modelleri çalıştırmak için tek bir buton sunmaktadır. Kullanıcı bu butona bastığında sistem sırasıyla final veri setini oluşturur, modelleri eğitir ve rapor grafiklerini üretir. İşlem tamamlandıktan sonra ekran değişir; sol tarafta her rapor görseli için bir buton, sağ tarafta ise seçilen grafiğin gösterildiği alan bulunur.

Raporda kullanılabilecek temel ekran görüntüleri:

- `reports/figures/model_metrics_comparison.png`: modellerin metrik karşılaştırması
- `reports/figures/model_metric_heatmap.png`: model ve özellik setlerine göre metrik ısı haritası
- `reports/figures/roc_auc_by_feature_set.png`: özellik setlerine göre ROC-AUC grafiği
- `reports/figures/confusion_matrix_market_plus_sentiment_core_XGBoost.png`: en yüksek doğruluk veren modelin hata matrisi
- `reports/figures/confusion_matrix_market_plus_sentiment_core_SVM_RBF.png`: sentiment ile en yüksek F1 değerlerinden birini veren SVM modeli

## Deneysel Sonuçlar

Deneylerde üç farklı özellik seti karşılaştırılmıştır:

1. `baseline_market_features`: yalnızca piyasa/fiyat özellikleri
2. `market_plus_sentiment_core`: piyasa özellikleri + temel sentiment özellikleri
3. `market_plus_sentiment_core_derived`: piyasa özellikleri + temel sentiment + türetilmiş sentiment özellikleri

Kullanılan modeller:

- Logistic Regression
- Random Forest Classifier
- SVM RBF
- XGBoost

Final veri seti özeti:

| Özellik | Değer |
|---|---:|
| Final veri seti satır sayısı | 2193 |
| Final veri seti kolon sayısı | 45 |
| Zaman aralığı | 2025-06-08 00:00 - 2026-06-08 08:00 UTC |
| Ham Bitcoin fiyat satırı | 8783 |
| Sentiment uygulanmış tweet sayısı | 43698 |
| Hedef sınıf 0 sayısı | 1101 |
| Hedef sınıf 1 sayısı | 1092 |

Sentiment etiketi dağılımı:

| Sentiment etiketi | Tweet sayısı |
|---|---:|
| Positive | 24830 |
| Neutral | 10087 |
| Negative | 8781 |

Model performans sonuçları:

| Özellik seti | Model | Accuracy | ROC-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Market | Logistic Regression | 0.4681 | 0.4629 | 0.4957 | 0.4574 | 0.5409 |
| Market | Random Forest | 0.4802 | 0.4526 | 0.3547 | 0.4434 | 0.2956 |
| Market | SVM RBF | 0.4772 | 0.4784 | 0.3901 | 0.4472 | 0.3459 |
| Market | XGBoost | 0.4498 | 0.4510 | 0.3987 | 0.4225 | 0.3774 |
| Market + Sentiment Core | Logistic Regression | 0.4377 | 0.4420 | 0.4478 | 0.4261 | 0.4717 |
| Market + Sentiment Core | Random Forest | 0.4225 | 0.4347 | 0.3750 | 0.3931 | 0.3585 |
| Market + Sentiment Core | SVM RBF | 0.4833 | 0.4983 | 0.5198 | 0.4718 | 0.5786 |
| Market + Sentiment Core | XGBoost | 0.4894 | 0.4522 | 0.4400 | 0.4681 | 0.4151 |
| Market + Sentiment Derived | Logistic Regression | 0.4590 | 0.4265 | 0.4367 | 0.4395 | 0.4340 |
| Market + Sentiment Derived | Random Forest | 0.4255 | 0.4329 | 0.4490 | 0.4185 | 0.4843 |
| Market + Sentiment Derived | SVM RBF | 0.4559 | 0.4578 | 0.4841 | 0.4468 | 0.5283 |
| Market + Sentiment Derived | XGBoost | 0.4711 | 0.4746 | 0.4727 | 0.4561 | 0.4906 |

Sentiment eklenmesinin accuracy üzerindeki etkisi:

| Model | Market | Market + Sentiment Core | Değişim | Market + Sentiment Derived | Değişim |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4681 | 0.4377 | -0.0304 | 0.4590 | -0.0091 |
| Random Forest | 0.4802 | 0.4225 | -0.0578 | 0.4255 | -0.0547 |
| SVM RBF | 0.4772 | 0.4833 | +0.0061 | 0.4559 | -0.0213 |
| XGBoost | 0.4498 | 0.4894 | +0.0395 | 0.4711 | +0.0213 |

En yüksek doğruluk değeri `market_plus_sentiment_core` özellik setiyle eğitilen `XGBoost` modelinde elde edilmiştir. Bu modelin accuracy değeri `0.4894` olarak ölçülmüştür. Sentiment eklenmesi XGBoost modelinde accuracy değerini `+0.0395`, SVM RBF modelinde ise `+0.0061` artırmıştır. Buna karşın Logistic Regression ve Random Forest modellerinde sentiment özellikleri doğruluğu düşürmüştür.

Bu sonuçlar sentiment verisinin model davranışını değiştirdiğini göstermektedir. Ancak test kümesindeki sınıf dağılımı yaklaşık dengeli olduğu için ve en yüksek accuracy değeri sınırlı kaldığı için, mevcut veri ve özellik setleriyle sentiment bilgisinin tek başına güçlü bir tahmin avantajı sağladığı söylenemez. Projenin ana bulgusu, sentiment özelliklerinin bazı modellerde ölçülebilir fakat sınırlı bir iyileşme sağlamasıdır.

Hata matrisi örneği olarak en yüksek accuracy değerini veren `Market + Sentiment Core / XGBoost` modeli kullanılabilir:

| | Tahmin 0 | Tahmin 1 |
|---|---:|---:|
| Gerçek 0 | 95 | 75 |
| Gerçek 1 | 93 | 66 |

Bu hata matrisi modelin hem yükseliş hem düşüş sınıfında belirgin hatalar yaptığını göstermektedir. Bu nedenle model sonucu nihai yatırım kararı üretmek için değil, sentiment verisinin tahmin performansına etkisini deneysel olarak incelemek için değerlendirilmelidir.

PCA sonucu: Kod tabanı incelendiğinde PCA uygulayan bir modül veya PCA çıktısı bulunmamaktadır. Bu nedenle raporda PCA sonucu verilmemelidir. PCA analizi ileride eklenirse, özellikle 45 kolonlu final veri setinde boyut indirgeme ve özellik etkisi analizi için ayrı bir deney olarak raporlanabilir.

## Donanımsal Tasarım

Bu proje gömülü sistem veya fiziksel donanım modülü içeren bir çalışma değildir. Bu nedenle sensör, mikrodenetleyici, devre kartı veya fiziksel bağlantı şeması bulunmamaktadır. Sistem tamamen yazılım tabanlıdır ve yerel bilgisayar üzerinde çalışan veri işleme, makine öğrenmesi ve kullanıcı arayüzü bileşenlerinden oluşmaktadır.

Kullanılan kavramsal donanım/bileşenler:

| Bileşen | Görev |
|---|---|
| Yerel bilgisayar | Python scriptlerinin, model eğitiminin ve arayüzün çalıştırılması |
| İnternet bağlantısı | CoinGecko ve GetXAPI üzerinden veri çekilmesi |
| Depolama birimi | CSV veri dosyaları, model sonuçları ve grafiklerin saklanması |
| CPU/RAM | Veri işleme ve makine öğrenmesi modellerinin eğitilmesi |

Raporun donanımsal tasarım bölümünde fiziksel modül resmi yerine sistemin yazılım bileşenlerini gösteren mimari diyagramın kullanılması daha uygundur.

## Yazılım Mimarisi ve Veri Tabanı

Proje modüler bir Python yapısıyla geliştirilmiştir. Kaynak kodlar `src` klasörü altında veri toplama, özellik çıkarımı, model eğitimi, değerlendirme, görselleştirme ve kullanıcı arayüzü olarak ayrılmıştır.

Yazılım katmanları:

| Katman | Klasör / Dosya | Görev |
|---|---|---|
| Veri toplama | `src/data` | Bitcoin fiyatı ve tweet verisinin alınması |
| Veri temizleme | `src/data/clean_tweet_text.py`, `src/data/trim_tweet_columns.py` | Tweet kolonlarının azaltılması ve metin temizleme |
| Özellik çıkarımı | `src/features` | Engagement ağırlığı, sentiment agregasyonu ve final veri seti |
| Model eğitimi | `src/models/train_model_comparison.py` | Farklı modellerin kronolojik split ile eğitilmesi |
| Görselleştirme | `src/visualization/plot_model_results.py` | Metrik ve hata matrisi grafiklerinin üretilmesi |
| Arayüz | `src/ui/model_report_app.py` | Model çalıştırma ve rapor görsellerini gösterme |

Bu projede ilişkisel veri tabanı kullanılmamıştır. Veri saklama işlemi CSV dosyalarıyla yapılmaktadır. CSV dosyaları veri tabanı tabloları gibi düşünülmüştür.

Temel veri tabloları:

| Dosya | Açıklama | Önemli kolonlar |
|---|---|---|
| `data/raw/bitcoin_price_raw.csv` | Ham Bitcoin fiyat verisi | `datetime_utc`, `price_usd`, `market_cap_usd`, `total_volume_usd` |
| `data/raw/bitcoin_getxapi_tweets_raw.csv` | Ham tweet verisi | `tweet_id`, `created_at`, `text`, engagement kolonları |
| `data/interim/bitcoin_tweets_cleaned.csv` | Temizlenmiş tweet metinleri | `tweet_id`, `created_at`, `text` |
| `data/interim/bitcoin_tweets_weighted.csv` | Engagement ağırlığı eklenmiş tweetler | `tweet_id`, `created_at`, `engagement_score`, `engagement_weight` |
| `data/interim/bitcoin_tweets_sentiment.csv` | Sentiment skoru eklenmiş tweetler | `tweet_id`, `created_at`, `sentiment_score`, `sentiment_label`, `engagement_weight` |
| `data/processed/final_dataset.csv` | Model eğitiminde kullanılan final veri seti | fiyat özellikleri, sentiment özellikleri, hedef kolon |
| `reports/model_results.csv` | Model performans sonuçları | `feature_set`, `model`, `accuracy`, `roc_auc`, `f1`, `precision`, `recall` |
| `reports/confusion_matrices.json` | Hata matrisi sonuçları | model bazlı 2x2 confusion matrix |

Hazırlanan UML diyagramları:

- `reports/diagrams/use_case_diagram.puml`
- `reports/diagrams/class_diagram.puml`
- `reports/diagrams/data_flow_diagram.puml`
- `reports/diagrams/sequence_diagram.puml`

## Kullanıcı Arayüzü

Kullanıcı arayüzü basit bir masaüstü uygulaması olarak tasarlanmıştır. Arayüzde tek kullanıcı tipi bulunmaktadır: araştırmacı/proje kullanıcısı. Kullanıcı sisteme veri girmez; mevcut veri dosyaları ve pipeline scriptleri üzerinden model eğitimi başlatır ve üretilen rapor görsellerini inceler.

Kullanıcı yetkileri:

| Kullanıcı tipi | Yetki |
|---|---|
| Araştırmacı / Proje kullanıcısı | Model eğitimini başlatma |
| Araştırmacı / Proje kullanıcısı | Rapor grafiklerini üretme |
| Araştırmacı / Proje kullanıcısı | Hata matrisi ve metrik grafiklerini görüntüleme |

Arayüz veri akışı:

1. Kullanıcı `python scripts/run_project_ui.py` komutu ile arayüzü başlatır.
2. İlk ekranda `Run machine learning models` butonuna basar.
3. Sistem final veri setini yeniden oluşturur.
4. Sistem makine öğrenmesi modellerini eğitir.
5. Sistem model sonuçlarından grafikler üretir.
6. Ekran rapor görüntüleme moduna geçer.
7. Sol taraftaki butonlardan biri seçildiğinde ilgili grafik sağ panelde gösterilir.

Arayüz ekran görüntüsü olarak rapora iki görsel eklenmesi önerilir:

- İlk ekran: model çalıştırma butonunun bulunduğu başlangıç ekranı
- İkinci ekran: sol menüde grafik butonları ve sağ tarafta seçili hata matrisi/grafik gösterimi

Rapor için en uygun UI çıktıları:

- Başlangıç ekranı: sistemin kullanıcı tarafından nasıl başlatıldığını gösterir.
- `model_metrics_comparison.png` açıkken rapor ekranı: model performanslarının kullanıcıya nasıl sunulduğunu gösterir.
- `confusion_matrix_market_plus_sentiment_core_XGBoost.png` açıkken rapor ekranı: en iyi modelin hata matrisini gösterir.

## Genel Değerlendirme

Bu çalışmada Bitcoin fiyat yönünü tahmin etmek için piyasa verileri ile Twitter/X sentiment verileri birlikte kullanılmıştır. Deneyler, sentiment özelliklerinin bazı modellerde doğruluk değerini artırdığını göstermiştir. En belirgin artış XGBoost modelinde görülmüş, market-only accuracy `0.4498` iken sentiment core özellikleriyle `0.4894` değerine çıkmıştır. Ancak genel başarı seviyesi sınırlı kaldığı için sentiment verisinin mevcut haliyle güçlü ve tek başına güvenilir bir tahmin sinyali oluşturduğu söylenemez.

Bu bulgu proje amacıyla uyumludur: amaç kesin fiyat tahmini yapmak değil, sosyal medya duygu verisinin makine öğrenmesi tahmin performansına etkisini ölçmektir. Elde edilen sonuçlara göre sentiment verisi model performansını bazı algoritmalarda ölçülebilir şekilde değiştirmiştir, ancak daha güçlü sonuçlar için daha dengeli tweet örneklemesi, daha gelişmiş sentiment modeli, ek teknik indikatörler ve daha uzun geçmiş veri kullanılmalıdır.
