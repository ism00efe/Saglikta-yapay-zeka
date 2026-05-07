# Temel Python imajını kullan
FROM python:3.11-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem bağımlılıklarını kur (Gerekirse)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Önce sadece requirements kopyalayıp kütüphaneleri kuralım (Hız kazandırır)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm kodları, modelleri ve klasör yapısını kopyala
COPY . .

# Hugging Face Spaces varsayılan olarak 7860 portunu bekler.
# 8000 API için, 7860 Arayüz (UI) için açılacak.
EXPOSE 7860 8000

# Hem FastAPI'yi (8000'de) hem de Streamlit'i (7860'da) aynı anda başlat
CMD uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run src/app/app.py --server.port 7860 --server.address 0.0.0.0