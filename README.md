# Proje Yapısı

Aşağıda proje klasör yapısı ve her klasörün içeriği açıklanmaktadır.

```
project
├───config
├───data
│   ├───interim
│   ├───processed
│   └───raw
├───reports
│   └───figures
├───scripts
└───src
    ├───data
    ├───evaluation
    ├───features
    ├───models
    └───visualization
```

## config/

- Proje genelinde kullanılan ayar dosyalarını içerir.
- Örnek dosyalar:
  - `config.yaml`
  - `model_config.yaml`
  - `data_config.yaml`
- Kullanım amacı:
  - seçilecek finansal varlık
  - tarih aralığı
  - model hiperparametreleri
  - train/test oranı
  - dosya yolları
  - teknik gösterge parametreleri

## data/

- Projede kullanılan veri dosyalarını içerir.

### data/raw/

- API veya scraping ile alınan ham verilerin saklandığı klasördür.
- Bu veriler üzerinde doğrudan işlem yapılmaz.
- Örnek dosyalar:
  - `price_raw.csv`
  - `social_media_raw.csv`
  - `news_raw.csv`
- İçerik:
  - ham fiyat verisi
  - ham sosyal medya metinleri
  - ham haber başlıkları/metinleri

### data/interim/

- Ham verilerin temizlenmiş veya ara işlemden geçirilmiş halleri burada tutulur.
- Örnek dosyalar:
  - `clean_prices.csv`
  - `clean_texts.csv`
  - `price_with_indicators.csv`
  - `text_with_sentiment.csv`
- Kullanım amacı:
  - eksik değer temizleme
  - tarih formatı düzeltme
  - metin temizleme
  - teknik göstergeleri hesaplama
  - duygu skoru üretme

### data/processed/

- Modelin doğrudan kullanacağı son veri setleri burada tutulur.
- Örnek dosyalar:
  - `final_dataset.csv`
  - `train.csv`
  - `test.csv`
- İçerik:
  - fiyat verisi
  - teknik göstergeler
  - sentiment skorları
  - hedef değişken

## reports/

- Model çıktıları, analiz sonuçları ve raporda kullanılacak dosyalar burada saklanır.

### reports/figures/

- Grafik ve görsellerin tutulduğu klasördür.
- Örnek dosyalar:
  - `prediction_vs_actual.png`
  - `loss_curve.png`
  - `confusion_matrix.png`
  - `feature_importance.png`
- Kullanım amacı:
  - model performans grafikleri
  - eğitim kaybı grafikleri
  - tahmin-gerçek değer karşılaştırmaları
  - tez raporunda kullanılacak görseller

## scripts/

- Pipeline aşamalarını çalıştırmak için kullanılan script dosyalarını içerir.
- Örnek dosyalar:
  - `run_data_pipeline.py`
  - `run_feature_pipeline.py`
  - `run_training.py`
  - `run_evaluation.py`
- Kullanım amacı:
  - veri çekme işlemini başlatmak
  - veri temizleme sürecini çalıştırmak
  - feature engineering işlemlerini yapmak
  - modeli eğitmek
  - değerlendirme sonuçlarını üretmek

## src/

- Projenin ana kaynak kodlarının bulunduğu klasördür.

### src/data/

- Veri çekme, okuma, yazma ve temizleme işlemleri burada tanımlanır.
- Örnek dosyalar:
  - `download_prices.py`
  - `scrape_text_data.py`
  - `preprocess_prices.py`
  - `preprocess_text.py`
- İşlevleri:
  - finansal fiyat verisi çekmek
  - sosyal medya/haber verisi toplamak
  - ham verileri temizlemek
  - CSV dosyalarını okumak ve kaydetmek

### src/features/

- Modelde kullanılacak özelliklerin üretildiği klasördür.
- Örnek dosyalar:
  - `technical_indicators.py`
  - `sentiment_features.py`
  - `merge_features.py`
  - `target_generator.py`
- İşlevleri:
  - RSI, MACD, EMA, SMA gibi teknik göstergeleri hesaplamak
  - metinlerden sentiment skoru üretmek
  - fiyat verisi ile sentiment verisini tarih bazında birleştirmek
  - hedef değişkeni oluşturmak

### src/models/

- Makine öğrenmesi ve derin öğrenme modellerinin tanımlandığı ve eğitildiği klasördür.
- Örnek dosyalar:
  - `baseline_models.py`
  - `lstm_model.py`
  - `train_model.py`
  - `predict_model.py`
- İşlevleri:
  - Random Forest, SVM gibi baseline modelleri oluşturmak
  - LSTM model mimarisini tanımlamak
  - model eğitimi yapmak
  - eğitilmiş model ile tahmin üretmek

### src/evaluation/

- Model performansının ölçüldüğü klasördür.
- Örnek dosyalar:
  - `metrics.py`
  - `evaluate_model.py`
  - `compare_models.py`
- İşlevleri:
  - Accuracy, Precision, Recall, F1 hesaplamak
  - MAE, RMSE, R² gibi regresyon metriklerini hesaplamak
  - modelleri karşılaştırmak
  - sonuçları tablo halinde üretmek

### src/visualization/

- Grafik üretme işlemleri burada yapılır.
- Örnek dosyalar:
  - `plot_predictions.py`
  - `plot_training_history.py`
  - `plot_confusion_matrix.py`
  - `plot_feature_importance.py`
- İşlevleri:
  - gerçek ve tahmin edilen değerleri çizmek
  - model loss grafiği oluşturmak
  - confusion matrix çizmek
  - feature importance grafiği üretmek
