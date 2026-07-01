# Bitcoin Tweet Sentiment ML Projesi

Bu proje, Twitter/X gönderilerinden çıkarılan duygu verisinin Bitcoin fiyat yönü tahmininde makine öğrenmesi performansını artırıp artırmadığını incelemek için geliştirilmiştir. Ana çalışma Bitcoin için 4 saatlik zaman pencerelerinde fiyat verisi, tweet sentiment verisi ve tweet etkileşim ağırlıklarını birleştirir.

Proje ayrıca rapor görselleri, UML diyagramları, model karşılaştırma çıktıları ve basit kullanıcı arayüzleri içerir.

## Ana Amaç

Araştırma sorusu:

> Twitter/X gönderilerinin sentiment skorları ve etkileşim ağırlıkları, Bitcoin fiyat yönünü tahmin eden makine öğrenmesi modellerinin başarısını artırıyor mu?

Bu amaçla üç özellik seti karşılaştırılır:

1. Yalnızca piyasa/fiyat özellikleri
2. Piyasa özellikleri + temel sentiment özellikleri
3. Piyasa özellikleri + temel sentiment + türetilmiş sentiment özellikleri

## Güncel Sonuç Özeti

Bitcoin final veri seti:

- Final veri seti: `data/processed/final_dataset.csv`
- Satır sayısı: 2193
- Kolon sayısı: 45
- Zaman aralığı: 2025-06-08 00:00 UTC - 2026-06-08 08:00 UTC
- Sentiment uygulanmış tweet sayısı: 43698
- Hedef değişken: `target_up_next_4h`

En iyi ölçülen accuracy sonucu:

| Özellik seti | Model | Accuracy |
|---|---|---:|
| Market + Sentiment Core | XGBoost | 0.4894 |

Sentiment eklenmesi XGBoost modelinde accuracy değerini `0.4498` seviyesinden `0.4894` seviyesine çıkarmıştır. Bu değişim ölçülebilir olsa da genel başarı sınırlıdır; proje sonucu kesin yatırım sinyali üretmek yerine sentiment verisinin model başarısına etkisini deneysel olarak göstermektedir.

## Proje Yapısı

```text
bitirme_projesi
├── app.py
├── config
├── data
│   ├── raw
│   ├── interim
│   └── processed
├── reports
│   ├── diagrams
│   └── figures
├── scripts
├── src
│   ├── data
│   ├── evaluation
│   ├── features
│   ├── models
│   ├── ui
│   └── visualization
└── tests
```

## Klasörler ve Görevleri

### `data/raw`

API veya dış kaynaklardan gelen ham veriler burada tutulur.

Önemli dosyalar:

- `bitcoin_price_raw.csv`: CoinGecko üzerinden alınan Bitcoin fiyat, piyasa değeri ve hacim verisi
- `bitcoin_getxapi_tweets_raw.csv`: GetXAPI üzerinden alınan ham tweet verisi
- `bitcoin_getxapi_tweets_raw_progress.csv`: tweet indirme ilerleme durumu

### `data/interim`

Ham veriden türetilen ara işlem dosyaları burada tutulur.

Önemli dosyalar:

- `bitcoin_tweets_trimmed.csv`: gereksiz kolonları çıkarılmış tweet verisi
- `bitcoin_tweets_cleaned.csv`: link, emoji ve metin gürültüsü temizlenmiş tweet verisi
- `bitcoin_tweets_weighted.csv`: engagement score ve engagement weight eklenmiş tweet verisi
- `bitcoin_tweets_sentiment.csv`: sentiment skoru ve sentiment etiketi eklenmiş tweet verisi

### `data/processed`

Modelin doğrudan kullandığı son veri setleri burada tutulur.

Önemli dosyalar:

- `final_dataset.csv`: Bitcoin model eğitimi için 4 saatlik birleşik veri seti
- `bitcoin_4h_sentiment.csv`: 4 saatlik sentiment özet dosyası
- `bitcoin_daily_sentiment.csv`: günlük Bitcoin sentiment özet dosyası
- `aapl_weekly_sentiment.csv`, `nvidia_weekly_sentiment.csv`, `eth_daily_sentiment.csv`: ek deney varlıklarına ait sentiment dosyaları

### `src/data`

Veri indirme, okuma ve temizleme modüllerini içerir.

Önemli dosyalar:

- `download_bitcoin_prices.py`: CoinGecko üzerinden Bitcoin fiyat verisini indirir.
- `download_getxapi_bitcoin_tweets.py`: GetXAPI üzerinden filtrelenmiş Bitcoin tweetlerini indirir.
- `trim_tweet_columns.py`: sentiment ve ağırlıklandırma için gereksiz tweet kolonlarını çıkarır.
- `clean_tweet_text.py`: tweet metinlerini sentiment analizine uygun hale getirir.

### `src/features`

Model özelliklerini üreten modülleri içerir.

Önemli dosyalar:

- `create_tweet_engagement_weights.py`: görüntülenme, beğeni, yanıt ve benzeri etkileşimlerden ağırlık üretir.
- `build_model_dataset.py`: fiyat ve sentiment verisini 4 saatlik pencerelerde birleştirerek `final_dataset.csv` üretir.
- `technical_indicators.py`: teknik gösterge hesaplamaları için yardımcı fonksiyonlar içerir.
- `sentiment_features.py`: sentiment özellikleri için yardımcı fonksiyonlar içerir.

### `src/models`

Model eğitimi ve karşılaştırma kodlarını içerir.

Önemli dosyalar:

- `train_model_comparison.py`: Logistic Regression, Random Forest, SVM ve gradient boosting modellerini karşılaştırır.
- `train_sequence_models.py`: deneysel sıralı model/LSTM eğitimi için kullanılır.
- `lstm_model.py`: LSTM model mimarisi için yardımcı yapı içerir.
- `train_model.py`: ek eğitim akışları için kullanılan modüldür.

### `src/visualization`

Model sonuçlarından grafik üretir.

Önemli dosyalar:

- `plot_model_results.py`: model metrik grafikleri, ROC-AUC grafiği, heatmap ve hata matrisi görselleri üretir.
- `plot_results.py`: ek ablation/varlık deneyleri için grafik üretimi yapar.

### `src/ui`

Masaüstü kullanıcı arayüzünü içerir.

Önemli dosyalar:

- `model_report_app.py`: Tkinter tabanlı arayüz. Model pipeline'ını çalıştırır ve `reports/figures` altındaki görselleri listeler.

### `reports`

Rapor çıktıları burada tutulur.

Önemli dosyalar:

- `model_results.csv`: model performans tablosu
- `confusion_matrices.json`: model hata matrisi değerleri
- `sistemin_gerceklenmesi_ve_bulgular.md`: rapor için hazırlanmış Türkçe bölüm metni

### `reports/figures`

Rapor ve UI içinde kullanılan grafik görsellerini içerir.

Örnek dosyalar:

- `model_metrics_comparison.png`
- `model_metric_heatmap.png`
- `roc_auc_by_feature_set.png`
- `confusion_matrix_market_plus_sentiment_core_XGBoost.png`

### `reports/diagrams`

PlantUML diyagram dosyalarını ve PNG çıktıları içerir.

Önemli dosyalar:

- `use_case_diagram.puml`
- `class_diagram.puml`
- `sequence_diagram.puml`
- `data_flow_diagram.puml`
- `data_preparation_flow_diagram.puml`
- `model_reporting_flow_diagram.puml`

## Kurulum

Python sanal ortamı oluştur:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Bağımlılıkları yükle:

```powershell
pip install -r requirements.txt
```

Tweet indirme yapılacaksa proje kök dizininde `.env` dosyası oluştur:

```text
GETXAPI_API_KEY=buraya_api_anahtari
```

`.env` dosyası GitHub'a gönderilmemelidir.

## Ana Pipeline Komutları

### 1. Bitcoin fiyat verisini indir

```powershell
python scripts/run_data_pipeline.py --granularity hourly
```

Varsayılan çıktı:

```text
data/raw/bitcoin_price_raw.csv
```

Tarih aralığı vermek için:

```powershell
python scripts/run_data_pipeline.py --granularity hourly --start-date 2025-06-08 --end-date 2026-06-08
```

### 2. GetXAPI ile Bitcoin tweet verisi indir

Güvenli örnek: yalnızca bir eksik günü indirir.

```powershell
python scripts/run_getxapi_bitcoin_tweets_pipeline.py --product Latest --sample-window-minutes 0 --max-calls-per-window 1 --max-days 1
```

Önemli notlar:

- Varsayılan ürün `Latest` değeridir.
- Varsayılan günlük hedef `400` tweettir.
- Varsayılan chunk süresi `4` saattir.
- `--max-days 1` tek eksik günü işler.
- `--max-days 0` tüm kalan günleri işler; API maliyeti oluşturabileceği için dikkatli kullanılmalıdır.
- Script ilerleme durumunu progress CSV dosyasında tuttuğu için sonraki çalıştırmalarda tamamlanan pencereleri atlayabilir.

### 3. Tweet kolonlarını kırp

```powershell
python scripts/run_tweet_trim_pipeline.py
```

Çıktı:

```text
data/interim/bitcoin_tweets_trimmed.csv
```

### 4. Tweet metinlerini temizle

```powershell
python scripts/run_tweet_text_cleaning.py
```

Çıktı:

```text
data/interim/bitcoin_tweets_cleaned.csv
```

### 5. Tweet engagement ağırlıklarını üret

```powershell
python scripts/run_tweet_engagement_weighting.py
```

Çıktı:

```text
data/interim/bitcoin_tweets_weighted.csv
```

### 6. Sentiment dosyasını hazırla

Model dataset pipeline'ı şu dosyayı bekler:

```text
data/interim/bitcoin_tweets_sentiment.csv
```

Beklenen temel kolonlar:

```text
tweet_id, created_at, sentiment_score, sentiment_label, engagement_weight
```

Bu dosya sentiment analizi scripti veya proje dışı sentiment işlem adımı tarafından oluşturulmalıdır.

### 7. Final model veri setini oluştur

```powershell
python scripts/run_model_dataset_pipeline.py
```

Çıktı:

```text
data/processed/final_dataset.csv
```

Bu adım:

- Bitcoin fiyatını 4 saatlik pencerelere toplar.
- Tweet sentiment değerlerini aynı 4 saatlik pencerelere toplar.
- Engagement ağırlıklı sentiment değerlerini hesaplar.
- 24 saatlik ve 7 günlük geçmiş rolling özellikleri üretir.
- `next_price_close`, `next_4h_return`, `target_up_next_4h` hedef kolonlarını oluşturur.

### 8. Model karşılaştırmasını çalıştır

```powershell
python scripts/run_model_training.py
```

Çıktılar:

```text
reports/model_results.csv
reports/confusion_matrices.json
```

Kullanılan modeller:

- Logistic Regression
- Random Forest Classifier
- SVM RBF
- XGBoost / LightGBM / HistGradientBoosting fallback

Model değerlendirme kronolojik split ile yapılır:

- %70 eğitim
- %15 doğrulama
- %15 test

### 9. Model sonuç grafiklerini üret

```powershell
$env:MPLCONFIGDIR=".matplotlib"
python scripts/run_model_result_plots.py
```

Çıktılar:

```text
reports/figures/model_metrics_comparison.png
reports/figures/model_metric_heatmap.png
reports/figures/roc_auc_by_feature_set.png
reports/figures/confusion_matrix_*.png
```

## Kullanıcı Arayüzleri

### Tkinter rapor arayüzü

Çalıştır:

```powershell
python scripts/run_project_ui.py
```

Exe çıktısı proje kök dizininde üretildiyse:

```powershell
.\bitcoin_tkinter_ui.exe
```

Bu arayüz:

1. Final veri setini yeniden oluşturur.
2. Model karşılaştırmasını çalıştırır.
3. Rapor grafiklerini üretir.
4. Sol menüden seçilen grafiği sağ panelde gösterir.

### Streamlit ablation dashboard

Çalıştır:

```powershell
streamlit run app.py
```

Exe launcher çıktısı proje kök dizininde üretildiyse:

```powershell
.\bitcoin_streamlit_dashboard.exe
```

Bu exe hafif bir launcher olarak çalışır; proje klasöründeki `app.py` dosyasını `python -m streamlit run app.py` komutuyla başlatır. Bu nedenle Streamlit ve proje bağımlılıklarının Python ortamında kurulu olması gerekir.

Bu dashboard; BTC, ETH, NVDA ve AAPL için ablation senaryolarını çalıştırmak ve ilgili görselleri göstermek için kullanılır.

## Testler

Odak testleri çalıştır:

```powershell
python -m unittest tests.test_build_model_dataset tests.test_train_model_comparison
```

Testlerin kapsadığı ana noktalar:

- Rolling özelliklerin gelecek veri kullanmaması
- Z-score sıfıra bölünme durumunda `inf` veya `NaN` üretmemesi
- Hedef kolonların model özellik listesine dahil edilmemesi
- Kronolojik split sırasının korunması

## UML ve Rapor Dosyaları

Rapor metni:

```text
reports/sistemin_gerceklenmesi_ve_bulgular.md
```

UML kaynak dosyaları:

```text
reports/diagrams/*.puml
```

PNG diyagram çıktıları:

```text
reports/diagrams/*.png
```

PlantUML dosyalarını yeniden PNG yapmak için:

```powershell
java -jar reports/diagrams/plantuml.jar -tpng reports/diagrams/*.puml
```

## Önemli Veri Dosyaları

| Dosya | Görev |
|---|---|
| `data/raw/bitcoin_price_raw.csv` | Ham Bitcoin fiyat verisi |
| `data/raw/bitcoin_getxapi_tweets_raw.csv` | Ham Bitcoin tweet verisi |
| `data/interim/bitcoin_tweets_cleaned.csv` | Temizlenmiş tweet metinleri |
| `data/interim/bitcoin_tweets_weighted.csv` | Engagement ağırlığı eklenmiş tweetler |
| `data/interim/bitcoin_tweets_sentiment.csv` | Sentiment skoru eklenmiş tweetler |
| `data/processed/final_dataset.csv` | Model eğitiminde kullanılan final veri seti |
| `reports/model_results.csv` | Model performans sonuçları |
| `reports/confusion_matrices.json` | Hata matrisi sonuçları |

## Notlar

- Projede ilişkisel veri tabanı kullanılmamaktadır; CSV dosyaları veri tabanı tabloları gibi kullanılmaktadır.
- `.env` dosyası gizli API anahtarları içerdiği için repoya eklenmemelidir.
- GetXAPI çağrıları ücretli olabileceği için tweet indirme komutlarında `--max-days`, `--max-api-calls` ve progress dosyası dikkatli kullanılmalıdır.
- Model sonuçları yatırım tavsiyesi değildir; proje akademik/deneysel amaçlıdır.

