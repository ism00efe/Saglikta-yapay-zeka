import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_split_data(raw_path, bio_path, output_dir="data/processed"):
    print("🚀 Veri yükleniyor ve birleştiriliyor...")
    df_main = pd.read_csv(raw_path)
    df_bio = pd.read_csv(bio_path)
    
    # Birleştirme Mantığı
    ortak_sutunlar = set(df_main.columns).intersection(set(df_bio.columns)) - {'VariationID'}
    df_merged = pd.merge(df_main, df_bio.drop(columns=list(ortak_sutunlar)), on='VariationID', how='left')
    
    # --- KRİTİK EKLENEN: Basit Temizlik ---
    # Etiketi (label) boş olan satırları atıyoruz
    df_merged = df_merged.dropna(subset=['label_2class'])
    
    # --- STRATEJİK BÖLME ---
    X = df_merged.drop(columns=['label_2class'])
    y = df_merged['label_2class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- KAYIT SİSTEMİ ---
    # Klasör yoksa oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Eğitim ve test setlerini birleştirip kaydet (Model eğitimi için hazır olsunlar)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    
    print(f"✅ Veri başarıyla bölündü ve {output_dir} klasörüne kaydedildi.")
    print(f"Eğitim seti boyutu: {len(train_df)} | Test seti boyutu: {len(test_df)}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Bu dosya direkt çalıştırılırsa veriyi işlesin
    load_and_split_data("data/raw/clinvar_missense.csv", "data/raw/clinvar_missense_bio.csv")