import os
import sys
import torch
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# --- 1. DOSYA YOLLARI ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "hybrid_xgboost_v1.pkl"

app = FastAPI(title="Sağlıkta Yapay Zeka Varyant Analiz")

# --- 2. MODELLERİ YÜKLEME ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ckpt = "facebook/esm2_t6_8M_UR50D"

print(f"Modeller yükleniyor... Cihaz: {device}")

# Dil modeli yükleniyor
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
llm_model = AutoModel.from_pretrained(model_ckpt).to(device)
llm_model.eval()

# İŞTE KRİTİK KISIM: XGBoost Modeli Yükleniyor (Eski rf_model tamamen silindi)
xgb_model = None
try:
    xgb_model = joblib.load(str(MODEL_PATH))
    print(f"✅ XGBoost başarıyla yüklendi: {MODEL_PATH}")
except Exception as e:
    print(f"❌ HATA: XGBoost Modeli yüklenemedi: {e}")

# --- 3. API GİRDİ ŞEMASI ---
class VariantInput(BaseModel):
    aa_ref: str
    aa_alt: str
    gene_symbol: str
    numeric_features: list

# --- 4. API ENDPOINT'LERİ ---
@app.get("/")
def home():
    return {"status": "online", "gpu": torch.cuda.is_available()}

@app.post("/predict")
async def predict(data: VariantInput):
    # Modelin yüklü olup olmadığı kontrol ediliyor
    if xgb_model is None:
        return {"status": "error", "message": "XGBoost modeli sunucuda yüklü değil! Lütfen train_model.py çalıştırıp modeli eğitin."}
    
    try:
        # A. Eğitimle BİREBİR Aynı Metin Formatı
        text = f"{data.gene_symbol.strip()} {data.aa_ref.strip()} {data.aa_alt.strip()}".upper()
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = llm_model(**inputs)
            # Eğitimle BİREBİR Aynı Pooling (CLS token, 0. indeks)
            vector = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        
        # B. Eğitimle BİREBİR Aynı Birleştirme Sırası (Önce Biyokimyasal 21, Sonra ESM2 320)
        numeric_1d = np.array(data.numeric_features)
        hybrid_input_1d = np.concatenate((numeric_1d, vector)) 
        hybrid_input = hybrid_input_1d.reshape(1, -1)
        
        # C. Tahmin (XGBoost)
        prediction = xgb_model.predict(hybrid_input)[0]
        proba = xgb_model.predict_proba(hybrid_input)[0]
        
        # XGBoost artık sayısal çıktı veriyor (0 veya 1). Bunu tekrar metne çeviriyoruz.
        return {
            "prediction": "Pathogenic" if prediction == 1 else "Benign",
            "confidence": float(max(proba)),
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": f"Tahmin hatası: {str(e)}"}