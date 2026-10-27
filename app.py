"""PDF + ses -> hoca çizimli anlatım videosu -> YouTube.  Çalıştır: streamlit run app.py"""
import os
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

import fitz
import streamlit as st

import engine as E

st.set_page_config(page_title="PDF Anlatım Videosu", page_icon="🎬", layout="wide")
ss = st.session_state

STATE_FILE = Path("proje_durumu.json")

def save_persistent_state():
    """Tüm aşamaları ve verileri diske kaydeder; F5 yapılsa bile silinmez."""
    keys = ["blocks", "step1_time", "audio_path", "pdf_path", "plan", "plan_stats",
            "step2_time", "video", "thumb_path", "step3_time", "title_in", "desc_in",
            "tags_in", "step4_time", "youtube_vid", "step5_time", "wd"]
    data = {k: ss[k] for k in keys if k in ss}
    if data:
        try:
            STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

def load_persistent_state():
    """Diskteki kayıtlı ilerlemeyi yükler."""
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            for k, v in data.items():
                if k not in ss:
                    ss[k] = v
        except Exception:
            pass

# Diskteki kayıtlı verileri hafızaya yükle ve senkronize et
load_persistent_state()
save_persistent_state()

if "wd" not in ss or not Path(ss.wd).exists():
    ss.wd = tempfile.mkdtemp(prefix="pdfvideo_")
    save_persistent_state()


def save_upload(up, stem):
    p = Path(ss.wd) / (stem + Path(up.name).suffix.lower())
    p.write_bytes(up.getbuffer())
    return str(p)


# ------------------------------------------------------------------ ayarlar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    provider = st.selectbox("Yapay Zeka Sağlayıcısı", ["NVIDIA NIM (DeepSeek / Minimax)", "Google Gemini", "Anthropic Claude"], index=0)

    if "NVIDIA" in provider:
        api_key = st.text_input("NVIDIA API anahtarı", type="password",
                                value=os.getenv("NVIDIA_API_KEY", ""),
                                placeholder="nvapi-...")
        model = st.selectbox("NVIDIA modeli", [
            "minimaxai/minimax-m3.1-flash",
            "deepseek-ai/deepseek-v4.1-flash",
            "nvidia/nemotron-3.5-super-vl-120b-a12b",
            "stepfun-ai/step-5-preview"
        ], index=0)
    elif provider == "Google Gemini":
        api_key = st.text_input("Gemini API anahtarı", type="password",
                                value=os.getenv("GEMINI_API_KEY", ""),
                                placeholder="AIzaSy...")
        model = st.selectbox("Gemini modeli", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    else:
        api_key = st.text_input("Anthropic API anahtarı", type="password",
                                value=os.getenv("ANTHROPIC_API_KEY", ""),
                                placeholder="sk-ant-...")
        model = st.text_input("Claude modeli", "claude-sonnet-5-5")

    whisper_size = st.selectbox("Ses tanıma modeli", ["small", "medium", "large-v3"], index=1)
    res = st.selectbox("Video çözünürlüğü", ["1280x720", "1920x1080"], index=1)
    secret_up = st.file_uploader("YouTube client_secret.json", type=["json"])
    extra = st.text_area("Başlık/açıklama için ek talimat (opsiyonel)",
                         placeholder="Örn: KPSS Anayasa serisi, 1. ders")
    
    st.markdown("---")
    if st.button("🗑️ Tüm İlerlemeyi Sıfırla (Baştan Başla)", type="secondary", help="Kaydedilmiş transkript, plan ve videoyu temizler"):
        if STATE_FILE.exists():
            STATE_FILE.unlink(missing_ok=True)
        for k in list(ss.keys()):
            del ss[k]
        st.success("Tüm ilerleme sıfırlandı!")
        time.sleep(0.5)
        st.rerun()

st.title("🎬 PDF + Ses → Hoca Anlatımlı Video → YouTube")

c1, c2 = st.columns(2)
pdf_up = c1.file_uploader("1) PDF yükle", type=["pdf"])
audio_up = c2.file_uploader("2) Ses kaydını yükle",
                            type=["mp3", "wav", "m4a", "aac", "ogg", "flac", "mp4"])

# Otomasyon Ayarları
st.markdown("---")
c_auto1, c_auto2 = st.columns([3, 1])
auto_mode = c_auto1.checkbox("🔄 Adımlar tamamlandığında geri sayımla otomatik sonrakine geç", value=True)
wait_sec = c_auto2.number_input("Adım arası bekleme (sn)", min_value=2, max_value=20, value=5, step=1)
st.markdown("---")

# ---------------------------------------------------- adım 1: ses -> yazı
st.subheader("Adım 1 · Sesi yazıya dök")
run_s1 = st.button("Sesi yazıya dök", disabled=not audio_up) or (ss.get("auto_step") == 1)

if run_s1 and audio_up:
    ss.pop("auto_step", None)
    t0 = time.time()
    try:
        ss.audio_path = save_upload(audio_up, "audio")
        with st.spinner("Konuşma yazıya çevriliyor (Whisper modeli ile analiz ediliyor)..."):
            segs = E.transcribe(ss.audio_path, whisper_size)
        ss.blocks = E.make_blocks(segs)
        ss.step1_time = round(time.time() - t0, 1)
        for k in ("plan", "video", "meta", "step2_time", "step3_time", "step4_time", "step5_time"):
            ss.pop(k, None)
        save_persistent_state()
        if auto_mode:
            ss.next_auto = 2
    except Exception as e:
        st.error(f"Hata: {e}")

if ss.get("blocks"):
    dur_txt = f" (⏱️ Süre: {E.fmt_duration(ss.step1_time)})" if "step1_time" in ss else ""
    st.success(f"✅ Adım 1 tamamlandı! {len(ss.blocks)} konuşma bloğu başarıyla çıkarıldı.{dur_txt}")
    with st.expander(f"{len(ss.blocks)} konuşma bloğu metni"):
        for b in ss.blocks:
            st.write(f"**{E.fmt_time(b['start'])}** {b['text']}")

    if auto_mode and ss.get("next_auto") == 2 and not ss.get("plan"):
        cnt = st.empty()
        for r in range(wait_sec, 0, -1):
            cnt.info(f"⏳ **{r} saniye** sonra Adım 2'ye (Çizim Planı & OCR) otomatik geçilecek...")
            time.sleep(1)
        cnt.empty()
        ss["auto_step"] = 2
        ss.pop("next_auto", None)
        st.rerun()

# ------------------------------------------------- adım 2: çizim planı
st.subheader("Adım 2 · Hoca çizim planını hazırla")
can_s2 = bool((pdf_up or ss.get("pdf_path")) and ss.get("blocks") and api_key)
run_s2 = st.button("Çizim planını hazırla", disabled=not can_s2) or (ss.get("auto_step") == 2)

if run_s2 and can_s2:
    ss.pop("auto_step", None)
    t0 = time.time()
    try:
        if pdf_up:
            ss.pdf_path = save_upload(pdf_up, "doc")
        doc = fitz.open(ss.pdf_path)
        if not E.pdf_text_ok(doc):
            doc.close()
            ocr_bar = st.progress(0.0, text="PDF taranmış görüntü formatında. Otomatik OCR uygulanıyor...")
            ss.pdf_path = E.auto_ocr_pdf(
                ss.pdf_path,
                lambda f, m: ocr_bar.progress(min(max(f, 0.0), 1.0), text=m)
            )
            ocr_bar.empty()
            st.info("✅ PDF'e akıllı metin katmanı otomatik olarak eklendi.")
        else:
            doc.close()

        plan_bar = st.progress(0.0, text="Sayfa ve ses zaman senkronizasyonu başlatılıyor...")
        ss.plan, found, missed = E.make_plan(
            api_key, model, ss.pdf_path, ss.blocks, provider=provider,
            progress_fn=lambda f, m: plan_bar.progress(min(max(f, 0.0), 1.0), text=m)
        )
        plan_bar.empty()
        ss.step2_time = round(time.time() - t0, 1)
        ss.plan_stats = (found, missed)
        ss.pop("video", None)
        save_persistent_state()
        if auto_mode:
            ss.next_auto = 3
    except Exception as e:
        st.error(f"Hata: {e}")

if ss.get("plan"):
    found, missed = ss.get("plan_stats", (len(ss.plan), 0))
    dur_txt = f" (⏱️ Süre: {E.fmt_duration(ss.step2_time)})" if "step2_time" in ss else ""
    st.success(f"✅ Adım 2 tamamlandı! {found} işaret planlandı, {missed} ifade atlandı.{dur_txt}")
    with st.expander("Plan (blok → sayfa → işaretler)"):
        for p in ss.plan:
            st.write(f"Blok {p['id']} · sayfa {p['page']} · "
                     + (", ".join(f"{m['kind']}: {m['text']}" for m in p["marks"]) or "—"))

    if auto_mode and ss.get("next_auto") == 3 and not ss.get("video"):
        cnt = st.empty()
        for r in range(wait_sec, 0, -1):
            cnt.info(f"⏳ **{r} saniye** sonra Adım 3'e (Video Üretimi) otomatik geçilecek...")
            time.sleep(1)
        cnt.empty()
        ss["auto_step"] = 3
        ss.pop("next_auto", None)
        st.rerun()

# --------------------------------------------------- adım 3: video
st.subheader("Adım 3 · Videoyu üret")
can_s3 = bool(ss.get("plan"))
run_s3 = st.button("Videoyu üret", disabled=not can_s3) or (ss.get("auto_step") == 3)

if run_s3 and can_s3:
    ss.pop("auto_step", None)
    t0 = time.time()
    W, H = map(int, res.split("x"))
    bar = st.progress(0.0, text="Video üretimi başlıyor...")
    out = str(Path(ss.wd) / "video.mp4")
    try:
        E.render_video(ss.pdf_path, ss.blocks, ss.plan, ss.audio_path, out, W, H,
                       lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
        ss.video = out
        ss.step3_time = round(time.time() - t0, 1)
        
        # Kapak görselini hazırla
        thumb_out = str(Path(ss.wd) / "thumbnail.jpg")
        E.extract_cover_thumbnail(ss.pdf_path, thumb_out)
        ss.thumb_path = thumb_out
        
        # Kolay erişim için klasöre de kopyala
        try:
            import shutil
            shutil.copy2(out, "TBMM_Yasama_Sureci_Anlatim.mp4")
            shutil.copy2(thumb_out, "kapak_gorseli.jpg")
        except Exception:
            pass

        save_persistent_state()
        if auto_mode:
            ss.next_auto = 4
    except Exception as e:
        st.error(f"Hata: {e}")

if ss.get("video"):
    dur_txt = f" (⏱️ Süre: {E.fmt_duration(ss.step3_time)})" if "step3_time" in ss else ""
    st.success(f"✅ Adım 3 tamamlandı! 1080p Video ve Kapak Resmi başarıyla oluşturuldu.{dur_txt}")
    st.video(ss.video)
    
    col_v1, col_v2 = st.columns([1, 1])
    with open(ss.video, "rb") as f:
        col_v1.download_button("📥 MP4 İndir", f, file_name="anlatim.mp4", mime="video/mp4")
    if ss.get("thumb_path") and Path(ss.thumb_path).exists():
        with open(ss.thumb_path, "rb") as f:
            col_v2.download_button("🖼️ Kapak Resmini İndir", f, file_name="kapak.jpg", mime="image/jpeg")

    if auto_mode and ss.get("next_auto") == 4 and not ss.get("title_in"):
        cnt = st.empty()
        for r in range(wait_sec, 0, -1):
            cnt.info(f"⏳ **{r} saniye** sonra Adım 4'e (Başlık & Açıklama) otomatik geçilecek...")
            time.sleep(1)
        cnt.empty()
        ss["auto_step"] = 4
        ss.pop("next_auto", None)
        st.rerun()

# -------------------------------------- adım 4: başlık/açıklama/hashtag
st.subheader("Adım 4 · Başlık, açıklama, hashtag")
can_s4 = bool(ss.get("blocks") and api_key)
run_s4 = st.button("Başlık ve açıklama üret", disabled=not can_s4) or (ss.get("auto_step") == 4)

if run_s4 and can_s4:
    ss.pop("auto_step", None)
    t0 = time.time()
    try:
        with st.spinner(f"{provider} başlık, bölümler ve notları hazırlıyor..."):
            meta = E.generate_metadata(api_key, model, ss.blocks, extra, provider=provider)
        ss["title_in"] = meta.get("title", "")[:100]
        ss["desc_in"] = E.build_description(meta, ss.blocks)
        ss["tags_in"] = ", ".join(meta.get("tags", []))
        ss.step4_time = round(time.time() - t0, 1)
        save_persistent_state()
        if auto_mode:
            ss.next_auto = 5
    except Exception as e:
        st.error(f"Hata: {e}")

title = st.text_input("Başlık", key="title_in", max_chars=100)
desc = st.text_area("Açıklama (hashtag'ler en altta)", key="desc_in", height=320)
tags_txt = st.text_input("Etiketler (virgülle)", key="tags_in")

if ss.get("title_in"):
    dur_txt = f" (⏱️ Süre: {E.fmt_duration(ss.step4_time)})" if "step4_time" in ss else ""
    st.success(f"✅ Adım 4 tamamlandı! Başlık ve zaman damgalı bölümler hazır.{dur_txt}")

# ------------------------------------------------ adım 5: YouTube
st.subheader("Adım 5 · YouTube'a yükle")
local_secret_exists = Path("client_secret.json").exists()
token_exists = Path("token.json").exists()

if token_exists:
    st.success("✅ YouTube oturumu açık ve hazır (`token.json` mevcut).")
else:
    st.warning("⚠️ YouTube yetkilendirmesi gerekli: İlk yükleme için 1 kerelik Google oturumu (`starlinkofmusk@gmail.com`) açılmalıdır.")
    col_a1, col_a2 = st.columns([2, 1])
    if col_a1.button("🔑 YouTube Girişi Yap (Google Hesabınızla Aç)", type="primary"):
        subprocess.Popen(["cmd.exe", "/c", "start", "youtube_giris.bat"])
        st.info("Tarayıcınızda Google giriş ekranı açıldı. Hesabınızı seçip izin verin. Ardından '🔄 Oturumu Yenile' butonuna tıklayın.")
    if col_a2.button("🔄 Oturumu Yenile"):
        st.rerun()

c_opts1, c_opts2 = st.columns([2, 1])
speed_opts = {
    1.5: "1.5x (⚡ Önerilen - Hızlı & Akıcı)",
    1.0: "1.0x (Normal Hız)",
    1.25: "1.25x (Hafif Hızlı)",
    1.75: "1.75x (Hızlı)",
    2.0: "2.0x (2 Kat Hızlı)",
}
upload_speed = c_opts1.radio(
    "⚡ YouTube Video Oynatma Hızı:",
    options=[1.5, 1.0, 1.25, 1.75, 2.0],
    index=0,
    format_func=lambda s: speed_opts[s],
    horizontal=True,
    key="speed_choice",
)

privacy = c_opts2.selectbox(
    "Gizlilik",
    ["public", "unlisted", "private"],
    format_func=lambda x: {"public": "Herkese açık", "unlisted": "Liste dışı", "private": "Özel"}[x],
    index=0,
)

can_s5 = bool(ss.get("video") and title and token_exists)
run_s5 = st.button("🚀 YouTube'a yükle", disabled=not can_s5, type="primary") or (ss.get("auto_step") == 5)

if auto_mode and ss.get("next_auto") == 5 and not ss.get("youtube_vid"):
    if token_exists:
        cnt = st.empty()
        stop_col1, stop_col2 = st.columns([3, 1])
        cancel_auto = stop_col2.button("⏸️ Otomatik Yüklemeyi Duraklat")
        if cancel_auto:
            ss.pop("next_auto", None)
            st.warning("Otomatik yükleme duraklatıldı. İstediğiniz ayarları kontrol edip 'YouTube'a Yükle' butonuna basabilirsiniz.")
        else:
            for r in range(10, 0, -1):
                cnt.warning(f"⏳ **{r} saniye** içinde hız seçimi yapabilirsiniz. Seçim yapılmazsa video **{upload_speed}x** hızında **Herkese Açık** olarak yüklenecektir...")
                time.sleep(1)
            cnt.empty()
            ss["auto_step"] = 5
            ss.pop("next_auto", None)
            st.rerun()
    else:
        st.error("⚠️ YouTube yetkilendirmesi eksik: `token.json` henüz oluşturulmamış. Lütfen yukarıdaki '🔑 YouTube Girişi Yap' butonuna basarak 1 kerelik onay verin.")

if run_s5 and can_s5:
    ss.pop("auto_step", None)
    t0 = time.time()
    try:
        secret_path = "client_secret.json" if local_secret_exists else ""
        if secret_up:
            secret_path = str(Path(ss.wd) / "client_secret.json")
            Path(secret_path).write_bytes(secret_up.getbuffer())

        # Hız ayarı (örn 1.5x)
        chosen_speed = float(ss.get("speed_choice", 1.5))
        upload_video_path = ss.video
        if abs(chosen_speed - 1.0) > 0.02:
            speed_out = str(Path(ss.wd) / f"video_{chosen_speed}x.mp4")
            with st.spinner(f"Video hızı {chosen_speed}x olarak ayarlanıyor..."):
                upload_video_path = E.change_video_speed(ss.video, speed_out, speed=chosen_speed)
            ss["speed_video"] = upload_video_path
            try:
                shutil.copy2(upload_video_path, f"TBMM_Yasama_Sureci_{chosen_speed}x.mp4")
            except Exception:
                pass

        service = E.youtube_service(secret_path)
        bar = st.progress(0.0, text="YouTube'a aktarılıyor...")

        thumb_path = ss.get("thumb_path", "")
        if not thumb_path or not Path(thumb_path).exists():
            thumb_path = "kapak_gorseli.jpg"

        vid = E.upload_to_youtube(
            service, upload_video_path, title, desc,
            [t.strip() for t in tags_txt.split(",") if t.strip()], privacy,
            thumbnail_path=thumb_path if Path(thumb_path).exists() else "",
            progress=lambda f: bar.progress(min(f, 1.0), text=f"YouTube'a aktarılıyor %{int(f * 100)}"),
        )
        ss.step5_time = round(time.time() - t0, 1)
        ss.youtube_vid = vid
        save_persistent_state()
        st.balloons()
    except Exception as e:
        st.error(f"Hata: {e}")

if ss.get("youtube_vid"):
    dur_txt = f" (⏱️ Süre: {E.fmt_duration(ss.step5_time)})" if "step5_time" in ss else ""
    st.success(f"🎉 TEBRİKLER! Videonuz ({ss.get('speed_choice', 1.5)}x hızında) ve Kapak Görseliniz YouTube'a Herkese Açık Olarak Başarıyla Yüklendi!{dur_txt}")
    st.markdown(f"### 👉 [Videonuzu YouTube'da İzleyin](https://youtu.be/{ss.youtube_vid})")
    if ss.get("speed_video") and Path(ss.speed_video).exists():
        with open(ss.speed_video, "rb") as f:
            st.download_button(f"📥 {ss.get('speed_choice', 1.5)}x Hızlandırılmış Videoyu İndir", f, file_name=f"anlatim_{ss.get('speed_choice', 1.5)}x.mp4", mime="video/mp4")
