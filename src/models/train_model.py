import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
import optuna  # 🌟 YENİ YILDIZIMIZ

# --- DOSYA YOLLARI ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train_features.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "test_features.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "hybrid_xgboost_v1.pkl"
CM_PLOT_PATH = BASE_DIR / "models" / "confusion_matrix.png"

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

    # Metin etiketlerini sayıya çevirme (XGBoost'un istediği format)
    label_mapping = {'Benign': 0, 'Pathogenic': 1}
    y_train = y_train.map(label_mapping)
    y_test = y_test.map(label_mapping)

    # Sınıf Dengesizliği Hesaplama
    neg_class = (y_train == 0).sum()
    pos_class = (y_train == 1).sum()
    scale_weight = neg_class / pos_class if pos_class > 0 else 1

    # --- 1. OPTUNA İLE HİPERPARAMETRE ARAMA ---
    def objective(trial):
        # Modelin deneyeceği parametre uzayı
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': scale_weight,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }

        # Çapraz Doğrulama (Cross-Validation) ile sağlamlık testi
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model = XGBClassifier(**param)
        
        # Amacımız F1 (Weighted) skorunu maksimize etmek
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
        return scores.mean()

    print("\n🔍 Optuna Zekası Devrede: En iyi parametreler aranıyor...")
    # direction='maximize' -> Skoru en yükseğe çıkarmaya çalış!
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) # Şimdilik 20 farklı kombinasyon deneyecek

    print("\n" + "="*40)
    print(" 🏆 OPTUNA OPTİMİZASYON SONUÇLARI ")
    print("="*40)
    print(f"⭐ En İyi Çapraz Doğrulama Skoru: %{study.best_value * 100:.2f}")
    print("🔧 En İyi Parametreler:")
    for key, value in study.best_params.items():
        print(f"   - {key}: {value}")

    # --- 2. EN İYİ MODELİ EĞİTME VE KAYDETME ---
    print("\n🚀 Final modeli keşfedilen en iyi parametrelerle eğitiliyor...")
    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_weight
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['eval_metric'] = 'logloss'

    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # --- 3. TEST SETİ PERFORMANSI ---
    y_pred = final_model.predict(X_test)
    
    print("\n--- SIZINTISIZ TEST SETİ PERFORMANSI ---")
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"🎯 Test F1 Skoru (Weighted): %{f1 * 100:.2f}")
    
    # Raporu geri metin formatında basmak için sayıları harfe çeviriyoruz
    reverse_mapping = {0: 'Benign', 1: 'Pathogenic'}
    y_test_text = y_test.map(reverse_mapping)
    y_pred_text = pd.Series(y_pred).map(reverse_mapping)
    print(classification_report(y_test_text, y_pred_text))

    # Matris ve Kayıt İşlemleri
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Karmaşıklık Matrisi (XGBoost + Optuna)')
    plt.ylabel('Gerçek Değerler')
    plt.xlabel('Tahmin Edilen Değerler')
    plt.savefig(CM_PLOT_PATH, bbox_inches='tight')
    plt.close()

    os.makedirs(MODEL_OUTPUT_PATH.parent, exist_ok=True)
    joblib.dump(final_model, MODEL_OUTPUT_PATH)
    print(f"💾 Optimize edilmiş model başarıyla kaydedildi: {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train()