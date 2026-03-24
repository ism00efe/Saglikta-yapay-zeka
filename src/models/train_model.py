import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path

# --- DOSYA YOLLARI ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train_features.csv" # <-- Sonuna _features eklendi
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "test_features.csv"   # <-- Sonuna _features eklendi
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "hybrid_random_forest_v1.pkl"
CM_PLOT_PATH = BASE_DIR / "models" / "confusion_matrix.png" # Grafik buraya kaydolacak

def train():
    print("🚀 Veri yükleniyor...")
    if not TRAIN_DATA_PATH.exists():
        print("❌ HATA: İşlenmiş eğitim verisi bulunamadı!")
        return

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train = train_df.drop(columns=['label_2class'])
    y_train = train_df['label_2class']
    X_test = test_df.drop(columns=['label_2class'])
    y_test = test_df['label_2class']

    # 1. MODELİ TANIMLAMA (Senin Notebook Ayarların)
    print("🧠 Model eğitiliyor, ağaçlar büyüyor... (class_weight='balanced')")
    rf_model = RandomForestClassifier(n_estimators=100, 
                                      random_state=42, 
                                      class_weight='balanced', 
                                      n_jobs=-1)
    
    # 2. EĞİTİMİ BAŞLATMA
    rf_model.fit(X_train, y_train)

    # 3. TEST SETİ ÜZERİNDE TAHMİN
    y_pred = rf_model.predict(X_test)

    # 4. PERFORMANS DEĞERLENDİRMESİ
    print("\n" + "="*40)
    print(" SIZINTISIZ GERÇEK PERFORMANS SONUÇLARI ")
    print("="*40)

    # Doğruluk yerine F1 Skorunu vurguluyoruz
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"🎯 Test F1 Skoru (Weighted): %{f1 * 100:.2f}")

    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(y_test, y_pred))

    # 5. KARMAŞIKLIK MATRİSİNİ GÖRSELLEŞTİRME VE KAYDETME
    print("📊 Karmaşıklık Matrisi çiziliyor...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Karmaşıklık Matrisi (Confusion Matrix)')
    plt.ylabel('Gerçek Değerler')
    plt.xlabel('Tahmin Edilen Değerler')
    
    # Sunucuda hata vermemesi için ekrana basmak yerine PNG olarak kaydediyoruz
    plt.savefig(CM_PLOT_PATH, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Matris görseli kaydedildi: {CM_PLOT_PATH}")

    # Modeli Kaydet
    os.makedirs(MODEL_OUTPUT_PATH.parent, exist_ok=True)
    joblib.dump(rf_model, MODEL_OUTPUT_PATH)
    print(f"💾 Model başarıyla kaydedildi: {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train()