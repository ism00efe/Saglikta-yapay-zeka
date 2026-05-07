import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm # İşlem çubuğu (Terminalde çok havalı durur)
import os
from pathlib import Path

# Kendi yazdığımız çevirmeni çağırıyoruz
import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from src.features.preprocessor import MutationPreprocessor # Senin sınıfının adı neyse o (örn: Preprocessor)

# --- DOSYA YOLLARI ---
TRAIN_RAW_PATH = BASE_DIR / "data" / "processed" / "train.csv"
TEST_RAW_PATH = BASE_DIR / "data" / "processed" / "test.csv"
TRAIN_OUT_PATH = BASE_DIR / "data" / "processed" / "train_features.csv"
TEST_OUT_PATH = BASE_DIR / "data" / "processed" / "test_features.csv"

# --- YAPAY ZEKA MODELİ AYARLARI ---
# RTX 4060'ı devreye sokuyoruz! Yoksa CPU ile saatler sürer.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESM2_MODEL_NAME = "facebook/esm2_t6_8M_UR50D" # BARDDNA modelimiz

def extract_features(df):
    print(f"⚙️ Özellik çıkarımı başlıyor... Donanım: {DEVICE}")
    
    # 1. Modelleri ve Çevirmeni Yükle
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
    esm_model = EsmModel.from_pretrained(ESM2_MODEL_NAME).to(DEVICE)
    esm_model.eval() # Modeli test moduna al (ağırlıkları güncellemesin)
    
    preprocessor = MutationPreprocessor() # Biyokimyasal çevirmenimiz
    
    all_features = []
    
    # tqdm ile progress bar (ilerleme çubuğu) ekliyoruz
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="İşleniyor"):
        
            # --- DÜZELTİLEN KISIM BAŞLANGICI ---
        aa_ref = str(row.get('aa_ref_1', 'A')).strip().upper()
        aa_alt = str(row.get('aa_alt_1', 'A')).strip().upper()
        gene_symbol = str(row.get('Symbol', row.get('gene_symbol', ''))).strip().upper() # Verisetindeki gen sütununa göre uyarla

        # API ile BİREBİR aynı metin formatı
        sequence = f"{gene_symbol} {aa_ref} {aa_alt}".strip()
        # --- DÜZELTİLEN KISIM BİTİŞİ ---

        # ... (Fiziksel özellikler kısmı aynı kalacak) ...

        # --- ESM2 DİL MODELİ VEKTÖRLERİ ---
        inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = esm_model(**inputs)
            # Eğitim ve canlıda AYNI pooling yöntemi: [CLS] token (0. index)
            esm_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            
        # --- HİBRİT BİRLEŞTİRME ---
        # DİKKAT: Önce biyokimyasal (21), sonra ESM (320)
        combined_vector = np.concatenate((bio_features, esm_embedding))
        all_features.append(combined_vector)
        
    # Oluşan o devasa matrisi Pandas DataFrame'e çevir
    # Sütun isimleri: f_0, f_1, ... f_340 şeklinde olacak
    feature_columns = [f"feature_{i}" for i in range(len(all_features[0]))]
    features_df = pd.DataFrame(all_features, columns=feature_columns)
    
    # Hedef değişkeni (label) geri ekle
    if 'label_2class' in df.columns:
        features_df['label_2class'] = df['label_2class'].values
        
    return features_df

def main():
    if not TRAIN_RAW_PATH.exists():
        print("❌ Veri bulunamadı! Önce make_dataset.py çalıştırın.")
        return

    print("🚀 EĞİTİM SETİ HAZIRLANIYOR...")
    train_df = pd.read_csv(TRAIN_RAW_PATH)
    train_features = extract_features(train_df)
    train_features.to_csv(TRAIN_OUT_PATH, index=False)
    
    print("\n🚀 TEST SETİ HAZIRLANIYOR...")
    test_df = pd.read_csv(TEST_RAW_PATH)
    test_features = extract_features(test_df)
    test_features.to_csv(TEST_OUT_PATH, index=False)
    
    print("\n🎉 Bütün özellikler başarıyla çıkarıldı ve diske kaydedildi!")

if __name__ == "__main__":
    main()