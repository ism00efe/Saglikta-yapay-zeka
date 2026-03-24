import os
import sys
import torch
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# 1. DOSYA YOLLARI
# main.py -> api -> src -> Proje Ana Dizini (3 kat yukarı)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "hybrid_random_forest_v1.pkl"

app = FastAPI(title="Sağlıkta Yapay Zeka Varyant Analiz")

# 2. MODELLERİ YÜKLEME (Global Tanımlamalar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ckpt = "facebook/esm2_t6_8M_UR50D"

print(f"Modeller yükleniyor... Cihaz: {device}")

# Bunları fonksiyon dışında tanımlıyoruz ki her istekte baştan yüklenmesinler
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
llm_model = AutoModel.from_pretrained(model_ckpt).to(device)
llm_model.eval()

rf_model = None
try:
    rf_model = joblib.load(str(MODEL_PATH))
    print(f"✅ Random Forest başarıyla yüklendi: {MODEL_PATH}")
except Exception as e:
    print(f"❌ HATA: RF Modeli yüklenemedi: {e}")

class VariantInput(BaseModel):
    aa_ref: str
    aa_alt: str
    gene_symbol: str
    numeric_features: list

@app.get("/")
def home():
    return {"status": "online", "gpu": torch.cuda.is_available()}

@app.post("/predict")
async def predict(data: VariantInput):
    # Model kontrolü
    if rf_model is None:
        return {"status": "error", "message": "Random Forest modeli sunucuda yüklü değil!"}
    
    try:
        # A. BARDDNA Vektörü (Embedding)
        text = f"{data.aa_ref} {data.aa_alt} {data.gene_symbol}"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = llm_model(**inputs)
            # 320 boyutlu vektör oluşturma
            vector = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # B. Sayısal Verilerle Birleştirme
        numeric_array = np.array(data.numeric_features).reshape(1, -1)
        hybrid_input = np.hstack([vector, numeric_array])
        
        # C. Tahmin
        prediction = rf_model.predict(hybrid_input)[0]
        proba = rf_model.predict_proba(hybrid_input)[0]
        
        return {
            "prediction": "Pathogenic" if prediction == 1 else "Benign",
            "confidence": float(max(proba)),
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": f"Tahmin hatası: {str(e)}"}