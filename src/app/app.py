import sys
import os
from pathlib import Path

# --- ZIRHLI YOL AYARI ---
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

import streamlit as st
import requests
import time
from features.preprocessor import MutationPreprocessor

# 1. Sayfa Ayarları (Sadece bir kere olmalı!)
st.set_page_config(page_title="Varyant Analiz", layout="wide", page_icon="🧬")

# Hesaplayıcıyı başlatıyoruz
prep = MutationPreprocessor()

# 2. Başlık ve Alt Başlık
st.title("🧬 Genetik Varyant Analiz Sistemi")
st.markdown("**Marmara Üniversitesi & Teknofest Projesi** | *BARDDNA (ESM2) + Hibrit Random Forest*")
st.divider()

# 3. Sol Menü (Sidebar)
with st.sidebar:
    st.header("⚙️ Sistem Durumu")
    st.success("API Bağlantısı: Aktif")
    st.info("Donanım: RTX 4060 (CUDA)")
    st.markdown("---")
    st.write("Bu arayüz, girilen mutasyonları FastAPI sunucusuna gönderir ve ESM2 dil modeli ile analiz eder.")

# 4. Ana Ekran - Kullanıcı Girişleri
st.subheader("Mutasyon Bilgilerini Giriniz")
col1, col2, col3 = st.columns(3)

with col1:
    aa_ref = st.text_input("Referans Aminoasit (Örn: A)", value="A")
with col2:
    aa_alt = st.text_input("Alternatif Aminoasit (Örn: V)", value="V")
with col3:
    gene_symbol = st.text_input("Gen Sembolü (Örn: BRCA1)", value="BRCA1")

st.markdown("<br>", unsafe_allow_html=True)

# 5. Analiz Butonu ve KATI KONTROL MEKANİZMASI
if st.button("🚀 Mutasyonu Analiz Et", use_container_width=True, type="primary"):
    
    # --- 🛡️ GÜVENLİK DUVARI (VALIDATION) ---
    # Sadece doğada bulunan 20 temel aminoasit harfi kabul edilir
    valid_amino_acids = "ARNDCEQGHILKMFPSTWYV"
    
    ref = aa_ref.strip().upper()
    alt = aa_alt.strip().upper()

    # Hata Kontrolü 1: Harf Uzunluğu
    if len(ref) != 1 or len(alt) != 1:
        st.error("⚠️ HATA: Lütfen sadece TEK BİR harf giriniz (Örn: A).")
        st.stop()

    # Hata Kontrolü 2: Geçersiz Karakter (Sayı, Rastgele Harf vb.)
    if ref not in valid_amino_acids or alt not in valid_amino_acids:
        st.error(f"⚠️ HATA: '{ref}' veya '{alt}' geçerli bir aminoasit değil!")
        st.info(f"Geçerli harfler: {', '.join(valid_amino_acids)}")
        st.stop()

    # Hata Kontrolü 3: Değişim Yoksa (A -> A)
    if ref == alt:
        st.warning("ℹ️ Bilgi: Referans ve Alternatif aynı. Protein dizisinde bir değişim olmadığı için sonuç otomatik olarak 'Zararsız' kabul edilir.")
        st.stop()

    with st.spinner("Biyokimyasal özellikler hesaplanıyor ve BARDDNA vektörleri çıkarılıyor..."):
        time.sleep(1) # Gerçekçilik katmak için ufak bir bekleme
        
        # GERÇEK HESAPLAMA BURADA YAPILIYOR (Otomatik özellik çıkarımı)
        real_numeric_features = prep.calculate_features(aa_ref, aa_alt)
        
        # API'ye Gönderilecek Veri Paketi
        payload = {
            "aa_ref": aa_ref.upper(),
            "aa_alt": aa_alt.upper(),
            "gene_symbol": gene_symbol.upper(),
            "numeric_features": real_numeric_features
        }

        # FastAPI Sunucusuna İstek Atma
        try:
            response = requests.post("https://saglikta-yapay-zeka.onrender.com/predict", json=payload)
            result = response.json()
            
            st.divider()
            st.subheader("📊 Analiz Sonucu")
            
            # --- YENİ EKLENEN GİZLİ HATA YAKALAYICI ---
            if result.get("status") == "error":
                st.error(f"❌ Yapay Zeka Modelinde Hata: {result.get('message')}")
                st.info("Arka plandaki API çöktü. Lütfen yukarıdaki hatayı kontrol edin.")
            
            # --- NORMAL SONUÇLAR ---
            elif result.get("prediction") == "Pathogenic":
                st.error(f"🚨 **ZARARLI (PATHOGENIC)** - Güven Skoru: %{result.get('confidence')*100:.1f}")
                st.progress(float(result.get('confidence')))
                st.warning("Bu varyantın protein yapısını bozma ve hastalığa yol açma ihtimali yüksek bulunmuştur.")
            else:
                st.success(f"✅ **ZARARSIZ (BENIGN)** - Güven Skoru: %{result.get('confidence')*100:.1f}")
                st.progress(float(result.get('confidence')))
                st.info("Bu varyantın protein işlevi üzerinde kritik bir etkisi olmadığı öngörülmektedir.")
                
        except Exception as e:
            st.error(f"Sunucuya bağlanılamadı! Lütfen Uvicorn API'nin çalıştığından emin olun. Hata: {e}")