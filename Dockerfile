# Temel Python imajını kullan
FROM python:3.11-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Gerekli dosyaları kopyala
COPY requirements.txt .

# Kütüphaneleri kur
RUN pip install --no-cache-dir -r requirements.txt

# Tüm kodları ve modelleri kopyala
COPY . .

# Portları dışarı aç (8000 API için, 8501 Arayüz için)
EXPOSE 8000 8501

# ÖNEMLİ: Hem FastAPI'yi (Arka plan) hem de Streamlit'i (Ön plan) aynı anda başlat
CMD uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run src/app/app.py --server.port 8501 --server.address 0.0.0.0