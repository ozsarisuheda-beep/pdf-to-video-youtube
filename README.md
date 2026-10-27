# 🎬 PDF + Ses → Hoca Anlatımlı Video → YouTube Otonom Üretim Sistemi

Bu uygulama, ders slaytlarını (PDF) ve ses kayıtlarını alır; konuşmayı metne döker, yapay zeka (NVIDIA NIM / Google Gemini / Claude) ile konuşmacının anlattığı kilit yerleri tespit eder, bir hoca gibi canlı kalem hareketleriyle (alt çizme, fosforlu kalem, kutu, yuvarlak) PDF üzerinde çizerek senkronize video üretir, kapak resmi hazırlar ve zaman damgalı bölümlerle birlikte doğrudan YouTube kanalınıza yükler.

---

## 🌟 Öne Çıkan Özellikler

1. **Çoklu Yapay Zeka Desteği:**
   - **NVIDIA NIM:** `minimaxai/minimax-m3.1-flash`, `deepseek-ai/deepseek-v4.1-flash`, `nvidia/nemotron-3.5-super-vl-120b-a12b`, `stepfun-ai/step-5-preview`.
   - **Google Gemini:** `gemini-2.5-flash`, `gemini-2.5-pro`.
   - **Anthropic Claude:** `claude-sonnet-5-5`.
2. **2 Aşamalı Akıllı Senkronizasyon:**
   - Sayfa çizelgesi ve grup bazlı çizim planlaması ile kota tüketimini minimuma indirir.
3. **Otomatik OCR:**
   - Taranmış / resim formatındaki PDF'leri otomatik algılar ve görünmez metin katmanı ekler.
4. **Pedagojik Çizimler:**
   - Önemli kanun maddeleri ve cümlelerin altını çizer (`underline`), tanımları fosforlu kalemle boyar (`highlight`), sayı ve süreleri yuvarlak içine alır (`circle`), istisnaları kutuya alır (`box`).
5. **Filigran Buğulama & Kapak Görseli:**
   - PDF'in ilk sayfasından YouTube için 16:9 kapak görseli üretir; slaytlardaki logo/filigranları otomatik tespit edip buğular.
6. **Video Hızlandırma:**
   - 1.5x (akıcı & hızlı), 1.25x, 1.0x veya 2.0x seçenekleriyle videoyu ve sesi bozmadan hızlandırır.
7. **Kalıcı Durum (F5 Koruması):**
   - Transkript, OCR ve üretilen video diske kaydedilir; tarayıcı yenilense dahi işlemler silinmez.

---

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler
- Python 3.10 veya üzeri
- [FFmpeg](https://ffmpeg.org/download.html) (sistem PATH yoluna ekli olmalı)

### 2. Kurulum
```bash
# Depoyu klonlayın
git clone https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git
cd REPO_ADINIZ

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 3. Ortam Değişkenleri (.env)
`.env.example` dosyasını `.env` olarak kopyalayın ve anahtarlarınızı girin:
```env
NVIDIA_API_KEY=nvapi-...
GEMINI_API_KEY=AIzaSy...
ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Çalıştırma
```bash
streamlit run app.py
```
veya Windows'ta doğrudan `baslat.bat` dosyasına çift tıklayın.

---

## 🔑 YouTube API Kurulumu (İlk Sefer İçin)

1. [Google Cloud Console](https://console.cloud.google.com)'da yeni bir proje açın.
2. **YouTube Data API v3** servisini etkinleştirin.
3. **OAuth Consent Screen** ayarlarında test kullanıcısı olarak kanalınızın Google e-postasını ekleyin.
4. **Credentials** > **Create Credentials** > **OAuth Client ID** (Desktop Application) seçerek `client_secret.json` dosyasını indirin ve proje ana dizinine koyun.
5. `youtube_giris.bat` çalıştırarak bir kerelik izin verin (`token.json` otomatik oluşacaktır).

---

## 🛡️ Lisans
Bu proje açık kaynaklıdır ve MIT lisansı altında sunulmuştur.
