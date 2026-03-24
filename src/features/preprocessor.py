import numpy as np

class MutationPreprocessor:
    def __init__(self):
        # 20 Temel Aminoasidin Biyokimyasal Özellikleri
        # Sırasıyla: [0: Ağırlık, 1: Hidrofobiklik, 2: Hacim, 3: pI (İzoelektrik), 4: Polarite (1/0), 5: Aromatiklik (1/0), 6: Yük]
        self.aa_props = {
            'A': [89.1, 1.8, 88.6, 6.00, 0, 0, 0],
            'R': [174.2, -4.5, 173.4, 10.76, 1, 0, 1],
            'N': [132.1, -3.5, 114.1, 5.41, 1, 0, 0],
            'D': [133.1, -3.5, 111.1, 2.77, 1, 0, -1],
            'C': [121.2, 2.5, 108.5, 5.07, 0, 0, 0],
            'E': [147.1, -3.5, 138.4, 3.22, 1, 0, -1],
            'Q': [146.2, -3.5, 143.8, 5.65, 1, 0, 0],
            'G': [75.1, -0.4, 60.1, 5.97, 0, 0, 0],
            'H': [155.2, -3.2, 153.2, 7.59, 1, 1, 1],
            'I': [131.2, 4.5, 166.7, 6.02, 0, 0, 0],
            'L': [131.2, 3.8, 166.7, 5.98, 0, 0, 0],
            'K': [146.2, -3.9, 168.6, 9.74, 1, 0, 1],
            'M': [149.2, 1.9, 162.9, 5.74, 0, 0, 0],
            'F': [165.2, 2.8, 189.9, 5.48, 0, 1, 0],
            'P': [115.1, -1.6, 112.7, 6.30, 0, 0, 0],
            'S': [105.1, -0.8, 89.0, 5.68, 1, 0, 0],
            'T': [119.1, -0.7, 116.1, 5.66, 1, 0, 0],
            'W': [204.2, -0.9, 227.8, 5.89, 0, 1, 0],
            'Y': [181.2, -1.3, 193.6, 5.66, 1, 1, 0],
            'V': [117.1, 4.2, 140.0, 5.96, 0, 0, 0]
        }

    def calculate_features(self, aa_ref, aa_alt):
        """
        Gelen Referans ve Alternatif aminoasitleri alıp tam 21 sayısal özellik döner.
        Modelin beklediği 341 input boyutunun 21'lik kısmını oluşturur.
        """
        aa_ref = aa_ref.upper()
        aa_alt = aa_alt.upper()

        # Eğer bilinmeyen bir harf girilirse (X gibi), default olarak Alanin (A) kabul edelim
        if aa_ref not in self.aa_props: aa_ref = 'A'
        if aa_alt not in self.aa_props: aa_alt = 'A'

        ref_features = self.aa_props[aa_ref]
        alt_features = self.aa_props[aa_alt]

        features = []
        
        # 1-7: Referans Aminoasidin özellikleri (7 adet)
        features.extend(ref_features)
        
        # 8-14: Alternatif Aminoasidin özellikleri (7 adet)
        features.extend(alt_features)
        
        # 15-21: İkisi arasındaki FARK (7 adet)
        for i in range(7):
            diff = alt_features[i] - ref_features[i]
            features.append(round(diff, 3))
            
        return features

# Kendi kendini test etmesi için:
if __name__ == "__main__":
    prep = MutationPreprocessor()
    ornek_hesap = prep.calculate_features("A", "V")
    print(f"Toplam Üretilen Özellik Sayısı: {len(ornek_hesap)}") # Tam 21 çıkmalı!
    print(f"Özellikler: {ornek_hesap}")