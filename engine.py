"""Motor: ses -> yazı -> çizim planı -> hoca çizimli video -> YouTube."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

COLORS = {
    "yellow": (255, 225, 0),
    "green": (60, 200, 90),
    "red": (230, 40, 40),
    "blue": (40, 110, 230),
    "orange": (255, 150, 20),
}


# ----------------------------------------------------------------- .env yükleme
def _load_dotenv(dotenv_path=".env"):
    p = Path(dotenv_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / dotenv_path
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k not in os.environ:
                    os.environ[k] = v

_load_dotenv()


# ----------------------------------------------------------------- ffmpeg & path ayarları
_script_dir = Path(__file__).resolve().parent
_venv_scripts = _script_dir / ".venv" / "Scripts"
if _venv_scripts.exists() and str(_venv_scripts) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_venv_scripts) + os.pathsep + os.environ.get("PATH", "")


# ----------------------------------------------------------------- av / pyav uyumluluk yaması
try:
    import av
    _orig_av_open = av.open
    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _safe_av_open
except Exception:
    pass


# ----------------------------------------------------------------- yardımcılar
def fmt_time(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_duration(seconds: float) -> str:
    sec = int(round(seconds))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h} sa {m} dk {s} sn"
    if m:
        return f"{m} dk {s} sn"
    return f"{s} sn"


def audio_duration(path: str) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path]
    )
    return float(out.decode().strip())


def parse_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw.strip(), flags=re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    a, b = cleaned.find("{"), cleaned.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("Yapay zeka JSON formatında yanıt döndürmedi:\n" + raw[:500])
    sub = cleaned[a:b + 1]
    try:
        return json.loads(sub)
    except Exception:
        try:
            import json_repair
            res = json_repair.loads(sub)
            if isinstance(res, dict):
                return res
            if isinstance(res, list):
                return {"steps": res}
        except Exception:
            pass
        raise


def ask_nvidia(api_key: str, model: str, system: str, user: str, max_tokens: int = 8192) -> str:
    from openai import OpenAI

    key = api_key or os.getenv("NVIDIA_API_KEY", "")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=key,
        timeout=120.0,
    )
    chosen_model = model if model and "/" in model else "minimaxai/minimax-m3.1-flash"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    last_error = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            # json_object format not supported by some models → retry without
            if "response_format" in err_str or "json_object" in err_str:
                try:
                    completion = client.chat.completions.create(
                        model=chosen_model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=max_tokens,
                    )
                    return completion.choices[0].message.content or ""
                except Exception as e2:
                    last_error = e2
            # timeout → retry
            if attempt < 2:
                import time as _time
                _time.sleep(5)
                continue
    raise last_error


def ask_gemini(api_key: str, model: str, system: str, user: str) -> str:
    from google import genai
    from google.genai import types

    key = api_key or os.getenv("GEMINI_API_KEY", "")
    client = genai.Client(api_key=key)
    chosen_model = model if model and "gemini" in model.lower() else "gemini-2.5-flash"
    config = types.GenerateContentConfig(
        system_instruction=system if system else None,
        temperature=0.2,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=chosen_model,
        contents=user,
        config=config,
    )
    return response.text or ""


def ask_claude(api_key: str, model: str, system: str, user: str,
               max_tokens: int = 16000) -> str:
    import anthropic

    chosen_model = model if model and "claude" in model.lower() else "claude-sonnet-5-5"
    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    with client.messages.stream(
        model=chosen_model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def detect_provider(provider: str, api_key: str, model: str) -> str:
    if provider:
        p = provider.lower()
        if "nvidia" in p or "deepseek" in p or "minimax" in p:
            return "nvidia"
        if "gemini" in p or "google" in p:
            return "gemini"
        if "claude" in p or "anthropic" in p:
            return "anthropic"
    if api_key:
        if api_key.startswith("nvapi-"):
            return "nvidia"
        if api_key.startswith("AIza") or not api_key.startswith("sk-"):
            return "gemini"
    if model:
        m = model.lower()
        if "deepseek" in m or "minimax" in m or "nemotron" in m or "step" in m or "/" in m:
            return "nvidia"
        if "gemini" in m:
            return "gemini"
    return "anthropic"


def ask_llm(api_key: str, model: str, system: str, user: str,
            provider: str = "", max_tokens: int = 16000) -> str:
    prov = detect_provider(provider, api_key, model)
    if prov == "nvidia":
        return ask_nvidia(api_key, model, system, user, max_tokens=min(max_tokens, 8192))
    if prov == "gemini":
        return ask_gemini(api_key, model, system, user)
    return ask_claude(api_key, model, system, user, max_tokens=max_tokens)


# ------------------------------------------------------------- 1) transkript
_MODELS: dict = {}


def transcribe(audio_path: str, size: str = "medium", language: str = "tr") -> list[dict]:
    from faster_whisper import WhisperModel

    if size not in _MODELS:
        _MODELS[size] = WhisperModel(size, device="auto", compute_type="auto")
    segs, _ = _MODELS[size].transcribe(
        audio_path, language=language, vad_filter=True, beam_size=5
    )
    return [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segs if s.text.strip()
    ]


def make_blocks(segs: list[dict], min_len: float = 12.0, max_len: float = 30.0) -> list[dict]:
    """Kısa Whisper parçalarını ~12-30 sn'lik anlatım bloklarına birleştirir."""
    blocks, cur = [], None
    for s in segs:
        if cur is None:
            cur = {"start": s["start"], "end": s["end"], "text": s["text"]}
        else:
            cur["end"] = s["end"]
            cur["text"] += " " + s["text"]
        length = cur["end"] - cur["start"]
        if (length >= min_len and re.search(r"[.!?…]$", cur["text"])) or length >= max_len:
            blocks.append(cur)
            cur = None
    if cur:
        blocks.append(cur)
    for i, b in enumerate(blocks):
        b["id"] = i
    return blocks


# --------------------------------------------------------- 2) çizim planı
PLAN_SYSTEM = """Sen, KPSS ve sınavlara hazırlık dersleri anlatan usta, dinamik ve pedagojik bir HOCASIN.
Sana (1) PDF'in sayfa metinleri ve (2) zaman damgalı konuşma blokları veriliyor.
Görevin: Konuşmacı ders anlatırken PDF üzerinde tıpkı sınıfta tahtada/ekranda ders anlatan dikkatli bir hoca gibi İLGİLİ TÜM ÖNEMLİ KELİMELERİ VE CÜMLELERİ aktif ve zengin bir şekilde İŞARETLEMEKTİR.

PEDAGOJİK VE HOCA ÇİZİM KURALLARI:
1. AKTİF VE ZENGİN İŞARETLEME:
   - Pasif kalma! Hoca konuşurken öğrencinin gözünü ekranda ve metinde tutmak için konuşmacının değindiği TÜM temel kavramları, kanun maddelerini, şartları, istisnaları, sayıları ve süreleri sırayla işaretle.
   - Konu anlatımı içeren her blok için ortalama 2 ila 6 adet anlamlı işaretleme ("marks") planla.
   - Sadece giriş, selamlaşma veya boş laf aralarında boş bırak.

2. İŞARET TÜRLERİNİ ("kind") VE RENKLERİ ("color") USTALIKLA KULLAN:
   - "underline" (Altını Çizme): Hoca gibi en sık kullanacağın doğal hareket. Cümleler, fiiller, şartlar ve temel ifadeler için (Renkler: "red" veya "blue").
   - "highlight" (Fosforlu Kalem): Başlıklar, genel ana kurallar ve çok önemli tanımlar için (Renkler: "yellow" veya "green").
   - "circle" (Yuvarlak İçine Alma): Kritik sayılar, süreler, çoğunluk oranları ve tek kelimelik can alıcı anahtar kelimeler için (örn: "360", "15 gün", "3/5", "1 Ekim", "salt çoğunluk") (Renkler: "red" veya "blue").
   - "box" (Kutu / Çerçeve): İstisna maddeler, dikkat edilmesi gereken tuzaklar, kurum adları veya önemli kanun adları için (Renkler: "orange" veya "blue").

3. SAYFA METNİ UYUMU ("text"):
   - "text" değeri, ilgili sayfa metninde GERÇEKTEN YAZAN 1-8 kelimelik bir ifade veya kelime olmalıdır. Sayfadaki metinden birebir kopyala (noktalama veya harf sırasına dikkat et). Sayfada yazmayan kelimeyi uydurma.
   - Konuşmacı konuşurken bahsettiği sıraya göre (kronolojik) işaretleri diz.

4. SAYFA YÖNETİMİ ("page"):
   - "page": 1'den başlayan sayfa numarası. Konuşmacı PDF'in hangi sayfasını anlatıyorsa o sayfayı seç. Konuşmacı yeni sayfaya geçmedikçe aynı sayfada kal.

5. ÇIKTI FORMATI:
   - Yanıt SADECE aşağıdaki JSON formatında olmalıdır:
{"steps":[{"id":0,"page":1,"marks":[{"text":"...","kind":"underline","color":"red"},{"text":"...","kind":"circle","color":"blue"}]}]}"""


def pdf_text_ok(doc) -> bool:
    chars = sum(len(p.get_text()) for p in doc)
    return chars / max(len(doc), 1) >= 40


def auto_ocr_pdf(pdf_path: str, progress_fn=None) -> str:
    """Taranmış veya resim formatındaki PDF'lere otomatik görünmez metin katmanı ekler."""
    doc = fitz.open(pdf_path)
    if pdf_text_ok(doc):
        doc.close()
        return pdf_path

    from rapidocr_onnxruntime import RapidOCR
    import numpy as np

    ocr = RapidOCR()
    total = len(doc)

    for i, page in enumerate(doc):
        if progress_fn:
            progress_fn((i + 1) / total, f"Görsel PDF taranıyor ve metin katmanı ekleniyor (Sayfa {i + 1}/{total})...")
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        result, _ = ocr(img)
        if not result:
            continue

        sx = page.rect.width / pix.width
        sy = page.rect.height / pix.height

        for box, text, _ in result:
            text = text.strip()
            if not text:
                continue
            x0 = min(p[0] for p in box) * sx
            y0 = min(p[1] for p in box) * sy
            x1 = max(p[0] for p in box) * sx
            y1 = max(p[1] for p in box) * sy
            fontsize = max(8, (y1 - y0) * 0.8)
            try:
                page.insert_text(fitz.Point(x0, y1 - 2), text, fontsize=fontsize, render_mode=3)
            except Exception:
                pass

    ocr_path = str(Path(pdf_path).with_name(Path(pdf_path).stem + "_ocr.pdf"))
    doc.save(ocr_path)
    doc.close()
    return ocr_path


def locate(page, phrase: str) -> list[list[float]]:
    """İfadeyi sayfada bulur; bulamazsa kısaltarak dener. Döner: [[x0,y0,x1,y1]]"""
    phrase = " ".join(str(phrase).split())
    if len(phrase) < 2:
        return []
    rot = page.rotation_matrix

    def find(q):
        if not q or len(q.strip()) < 2:
            return []
        hits = page.search_for(q)
        out = []
        for h in hits[:1]:
            r = fitz.Rect(h) * rot
            r.normalize()
            out.append([r.x0, r.y0, r.x1, r.y1])
        return out

    words = phrase.split()
    cands = [phrase]
    for n in range(len(words) - 1, 0, -1):
        cands.append(" ".join(words[:n]))
        cands.append(" ".join(words[-n:]))
    for q in cands:
        hit = find(q)
        if hit:
            return hit
    return []


def make_plan(api_key: str, model: str, pdf_path: str, blocks: list[dict], provider: str = "", progress_fn=None):
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    num_blocks = len(blocks)

    # 1. Aşama: Sayfa Zaman Çizelgesi (Hangi blokta hangi sayfa anlatılıyor?)
    if progress_fn:
        progress_fn(0.05, "Ses akışı ve PDF sayfaları senkronize ediliyor...")

    pages_summary = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        first_few_lines = [l.strip() for l in text.split("\n") if l.strip()][:4]
        header = " | ".join(first_few_lines)
        pages_summary.append(f"Sayfa {i+1}: {header}")
    pages_text = "\n".join(pages_summary)

    blocks_summary = []
    for b in blocks:
        blocks_summary.append(f"[{b['id']}] ({fmt_time(b['start'])}-{fmt_time(b['end'])}) {b['text'][:90]}...")
    blocks_text = "\n".join(blocks_summary)

    sched_prompt = f"""Sen, {num_pages} sayfalık bir PDF slaytı üzerinden yapılan ders kaydını ({num_blocks} blok) analiz eden bir senkronizasyon uzmanısın.
Konuşmacılar slaytları baştan sona sırayla anlatmaktadır.
Her sayfanın hangi blokta başlayıp hangi blokta bittiğini belirle.

PDF SAYFALARI (1-{num_pages}):
{pages_text}

KONUŞMA BLOKLARI (0-{num_blocks-1}):
{blocks_text}

Kurallar:
- Bloklar 0'dan başlar ve {num_blocks-1}'e kadar eksiksiz sıralı olmalıdır.
- Her sayfa için konuşulan start_id ve end_id aralığını belirle.
- Sayfalar kronolojik sırayla ilerlemelidir.
Yanıt SADECE JSON formatında olmalıdır:
{{"schedule": [
  {{"page": 1, "start_id": 0, "end_id": 10}},
  {{"page": 2, "start_id": 11, "end_id": 15}}
]}}"""

    raw_sched = ask_llm(api_key, model, "Sen bir video senkronizasyon uzmanısın.", sched_prompt, provider=provider)
    sched = parse_json(raw_sched).get("schedule", [])

    block_to_page = {}
    last_page = 1
    for item in sched:
        try:
            p_num = int(item.get("page", last_page))
            s_id = int(item.get("start_id", 0))
            e_id = int(item.get("end_id", s_id))
            for bid in range(s_id, e_id + 1):
                block_to_page[bid] = p_num
            last_page = p_num
        except Exception:
            pass

    for b in blocks:
        bid = b["id"]
        if bid not in block_to_page:
            prev_p = block_to_page.get(bid - 1, 1) if bid > 0 else 1
            block_to_page[bid] = prev_p

    # 2. Aşama: Sayfa Grupları Halinde Çizim Planı (Kotayı Korumak İçin Gruplandırma)
    page_to_blocks = {}
    for b in blocks:
        p = block_to_page[b["id"]]
        page_to_blocks.setdefault(p, []).append(b)

    steps_all = {}
    active_pages = sorted(page_to_blocks.keys())

    # 21 sayfayı tek tek çağırmak yerine 7'şerli 3 gruba bölerek API çağrısını 3'e düşürüyoruz!
    batch_size = 7
    batches = [active_pages[i:i + batch_size] for i in range(0, len(active_pages), batch_size)]

    for b_idx, page_group in enumerate(batches):
        if progress_fn:
            frac = 0.15 + 0.80 * ((b_idx + 1) / max(len(batches), 1))
            progress_fn(frac, f"Hoca çizimleri planlanıyor: Sayfa {page_group[0]}-{page_group[-1]} (Grup {b_idx + 1}/{len(batches)})...")

        group_pages_text = []
        group_blocks_text = []
        for pn in page_group:
            p_text = doc[pn - 1].get_text("text").strip()
            if p_text:
                group_pages_text.append(f"=== SAYFA {pn} METNİ ===\n{p_text}")
            for b in page_to_blocks.get(pn, []):
                group_blocks_text.append(
                    f"[{b['id']}] (Sayfa {pn}) ({fmt_time(b['start'])}-{fmt_time(b['end'])}) {b['text']}"
                )

        if not group_blocks_text:
            continue

        pages_content = "\n\n".join(group_pages_text)
        blocks_content = "\n".join(group_blocks_text)

        batch_prompt = f"""Sen, KPSS ve sınavlara hazırlık dersleri anlatan usta, dikkatli ve pedagojik bir HOCASIN.
Aşağıda ilgili PDF sayfaları ve konuşmacının o sayfalarda anlattığı konuşma blokları verilmiştir.
Her blok için konuşmacı anlatırken ilgili sayfa metninden geçen kilit ifadeleri bularak çiz.

{pages_content}

KONUŞMA BLOKLARI:
{blocks_content}

KURALLAR:
1. Her blok için konuşmada adı geçen ve o sayfa metninde GERÇEKTEN YAZAN 1 ila 5 kelimelik ifadeleri seç.
2. İşaret türleri ("kind"): "underline" (en sık kullan - kırmızı veya mavi), "circle" (sayılar ve süreler için), "highlight" (önemli kavramlar ve başlıklar için), "box" (istisnalar ve tuzaklar için).
3. Renkler ("color"): "red", "blue", "yellow", "green", "orange".
4. Sayfada yazmayan kelimeyi uydurma; ilgili sayfa metninden birebir kopyala.
5. Ortalama her blok için 2 ila 5 adet işaret belirle. Giriş/selamlaşma laflarında boş bırakabilirsin.
Yanıt SADECE JSON formatında olmalıdır:
{{"steps": [
  {{"id": {page_to_blocks[page_group[0]][0]['id']}, "marks": [{{"text": "...", "kind": "underline", "color": "red"}}]}}
]}}"""

        try:
            raw_page = ask_llm(api_key, model,
                               "Sen usta bir hoca gibi çizim planı yapan pedagojik bir yapay zekasın.",
                               batch_prompt, provider=provider)
            page_steps = parse_json(raw_page).get("steps", [])
            for s in page_steps:
                if "id" in s:
                    steps_all[int(s["id"])] = s
        except Exception as e:
            print(f"Grup {page_group} çizim planı hatası: {e}")

    # 3. Aşama: Koordinat eşleştirme ve plan oluşturma
    plan, found, missed = [], 0, 0
    for b in blocks:
        pn = block_to_page.get(b["id"], 1)
        s = steps_all.get(b["id"], {})
        marks = []
        for m in (s.get("marks") or [])[:6]:
            rects = locate(doc[pn - 1], m.get("text", ""))
            if rects:
                marks.append({"text": m["text"], "kind": m.get("kind", "underline"),
                              "color": m.get("color", "red"), "rects": rects})
                found += 1
            else:
                missed += 1
        plan.append({"id": b["id"], "page": pn, "marks": marks})
    return plan, found, missed


# ------------------------------------------------------------- 3) video
def draw_mark(ov: Image.Image, mark: dict, t: float):
    """Bir işareti ilerleme oranı t (0-1) kadar çizer. Kalem ucu konumunu döner."""
    d = ImageDraw.Draw(ov)
    col = COLORS.get(mark.get("color"), COLORS["yellow"])
    kind = mark.get("kind", "highlight")
    rects = mark["rects"]
    if kind in ("highlight", "underline"):
        total = sum(r[2] - r[0] for r in rects) or 1.0
        covered, cum, head = t * total, 0.0, None
        for x0, y0, x1, y1 in rects:
            w = x1 - x0
            part = max(0.0, min(w, covered - cum))
            cum += w
            if part <= 0:
                continue
            if kind == "highlight":
                d.rectangle([x0 - 2, y0, x0 + part + 2, y1], fill=col + (115,))
                head = (x0 + part, (y0 + y1) / 2)
            else:
                d.line([x0, y1 + 1, x0 + part, y1 + 1], fill=col + (255,), width=4)
                head = (x0 + part, y1 + 1)
        return head
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    a = int(255 * min(1.0, t * 1.6))
    if kind == "circle":
        d.ellipse([x0 - 14, y0 - 8, x1 + 14, y1 + 8], outline=col + (a,), width=4)
    else:
        d.rectangle([x0 - 5, y0 - 3, x1 + 5, y1 + 3], outline=col + (a,), width=4)
    return None


def draw_pen(ov: Image.Image, head):
    if not head:
        return
    x, y = head
    ImageDraw.Draw(ov).ellipse([x - 9, y - 9, x + 9, y + 9],
                               fill=(220, 30, 30, 235), outline=(255, 255, 255, 255), width=2)


class PageView:
    def __init__(self, page, W: int, H: int, blur_watermark: bool = True):
        r = page.rect
        self.s = min((W - 60) / r.width, (H - 40) / r.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(self.s, self.s), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        if blur_watermark:
            from PIL import ImageFilter
            bw = int(pix.width * 0.14)
            bh = int(pix.height * 0.065)
            box = (pix.width - bw, pix.height - bh, pix.width, pix.height)
            crop = img.crop(box)
            blurred = crop.filter(ImageFilter.GaussianBlur(radius=18))
            img.paste(blurred, box)

        canvas = Image.new("RGB", (W, H), (236, 238, 241))
        self.ox, self.oy = (W - pix.width) // 2, (H - pix.height) // 2
        canvas.paste(img, (self.ox, self.oy))
        self.base = canvas.convert("RGBA")
        self.overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    def px(self, mark: dict) -> dict:
        rects = [(x0 * self.s + self.ox, y0 * self.s + self.oy,
                  x1 * self.s + self.ox, y1 * self.s + self.oy)
                 for x0, y0, x1, y1 in mark["rects"]]
        return {**mark, "rects": rects}

    def frame(self, extra=None) -> Image.Image:
        ov = self.overlay
        if extra:
            ov = ov.copy()
            extra(ov)
        return Image.alpha_composite(self.base, ov).convert("RGB")


def render_video(pdf_path, blocks, plan, audio_path, out_path,
                 W=1920, H=1080, progress=lambda f, m: None):
    doc = fitz.open(pdf_path)
    dur = audio_duration(audio_path)
    tmp = tempfile.mkdtemp(prefix="frames_")
    entries: list[tuple[str, float]] = []
    static: dict = {}
    committed: dict[int, list] = {}
    cache = {"pn": None, "view": None}
    counter = [0]

    def view(pn):
        if cache["pn"] != pn:
            v = PageView(doc[pn - 1], W, H)
            for m in committed.get(pn, []):
                draw_mark(v.overlay, v.px(m), 1.0)
            cache["pn"], cache["view"] = pn, v
        return cache["view"]

    def save(img):
        counter[0] += 1
        p = os.path.join(tmp, f"f{counter[0]:06d}.jpg")
        img.save(p, quality=90)
        return p

    def hold(pn, d):
        if d < 0.02:
            return
        key = (pn, len(committed.get(pn, [])))
        if key not in static:
            static[key] = save(view(pn).frame())
        entries.append((static[key], d))

    t, cur = 0.0, (plan[0]["page"] if plan else 1)
    for i, (b, step) in enumerate(zip(blocks, plan)):
        progress(i / max(len(blocks), 1), f"Kareler hazırlanıyor ({i + 1}/{len(blocks)})")
        bstart = b["start"]
        bend = blocks[i + 1]["start"] if i + 1 < len(blocks) else dur
        if bstart > t:
            hold(cur, bstart - t)
            t = bstart
        cur = step["page"]
        marks = step["marks"]
        if marks:
            base_t = t
            slot = max(bend - base_t - 0.6, 0.6) / len(marks)
            for k, m in enumerate(marks):
                begin = base_t + 0.3 + k * slot
                if begin > t:
                    hold(cur, begin - t)
                    t = begin
                ad = min(0.9, slot * 0.8)
                nf = max(2, int(ad * 12))
                v = view(cur)
                pm = v.px(m)
                for j in range(1, nf + 1):
                    def extra(ov, pm=pm, j=j, nf=nf):
                        draw_pen(ov, draw_mark(ov, pm, j / nf))
                    entries.append((save(v.frame(extra)), ad / nf))
                t += ad
                committed.setdefault(cur, []).append(m)
                draw_mark(v.overlay, pm, 1.0)
        if bend > t:
            hold(cur, bend - t)
            t = bend
    hold(cur, (dur + 0.3) - t)

    if not entries:
        raise RuntimeError("Hiç kare üretilemedi.")
    progress(0.9, "Video kodlanıyor (ffmpeg)...")
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p, d in entries:
            f.write(f"file '{Path(p).as_posix()}'\nduration {d:.4f}\n")
        f.write(f"file '{Path(entries[-1][0]).as_posix()}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", audio_path,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast",
           "-tune", "stillimage", "-crf", "22", "-pix_fmt", "yuv420p", "-r", "25",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path]
    res = subprocess.run(cmd, capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if res.returncode != 0:
        raise RuntimeError("ffmpeg hatası:\n" + res.stderr.decode(errors="ignore")[-1500:])
    progress(1.0, "Video hazır")
    return out_path


# ------------------------------------------------- 4) başlık / açıklama
META_SYSTEM = """Sen Türkçe sınav hazırlık (KPSS, ÖSYM, PAEM vb.) ders videoları için YouTube başlık ve açıklama editörüsün.
Sana videonun zaman damgalı konuşma metni verilecek. SADECE konuşmada geçen içeriğe dayan; uydurma bilgi ekleme, yanıltıcı (clickbait) vaat verme.

Üret:
- "title": en fazla 90 karakter, konuyu ve sınav türünü içeren net başlık.
- "summary": 2-3 cümlelik açıklama girişi (izleyici ne öğrenecek).
- "chapters": en az 4, en fazla 15 bölüm: [{"block_id": <konuşma bloğu numarası>, "title": "kısa bölüm başlığı"}]. İlk bölüm block_id 0 olsun.
- "notes": konuşmadaki en önemli 5-10 ders notu (kısa, madde madde, her biri tek cümle).
- "tags": 10-15 arama etiketi (kelime/kısa öbek).
- "hashtags": 4-8 adet, boşluksuz, # ile başlayan.

Yanıt SADECE geçerli JSON olsun:
{"title":"","summary":"","chapters":[{"block_id":0,"title":""}],"notes":[""],"tags":[""],"hashtags":["#..."]}"""


def generate_metadata(api_key, model, blocks, extra="", provider: str = "") -> dict:
    # Konuşma metni çok uzunsa kısalt (ilk 60 + son 20 blok yeterli)
    relevant = blocks
    if len(blocks) > 80:
        relevant = blocks[:60] + blocks[-20:]
    lines = "\n".join(f"[{b['id']}] ({fmt_time(b['start'])}) {b['text']}" for b in relevant)
    user = f"KONUŞMA METNİ:\n{lines}"
    if extra.strip():
        user += f"\n\nEK TALİMAT:\n{extra.strip()}"
    return parse_json(ask_llm(api_key, model, META_SYSTEM, user, provider=provider, max_tokens=4000))


def build_description(meta: dict, blocks: list[dict]) -> str:
    parts = [meta.get("summary", "").strip()]
    chap, last = [], -100.0
    valid = [c for c in meta.get("chapters", []) if 0 <= int(c.get("block_id", -1)) < len(blocks)]
    for c in sorted(valid, key=lambda c: int(c["block_id"])):
        t = 0.0 if not chap else blocks[int(c["block_id"])]["start"]
        if chap and t - last < 10:
            continue
        chap.append(f"{fmt_time(t)} - {c['title']}")
        last = t
    if len(chap) >= 3:
        parts.append("⏳ BÖLÜMLER\n" + "\n".join(chap))
    notes = meta.get("notes", [])
    if notes:
        parts.append("📌 ÖNEMLİ DERS NOTLARI\n" + "\n".join("• " + n for n in notes))
    tags = " ".join(meta.get("hashtags", [])[:8])
    if tags:
        parts.append(tags)
    return "\n\n".join(p for p in parts if p)


def clean_tags(tags: list[str], limit: int = 450) -> list[str]:
    out, total = [], 0
    for t in tags:
        t = re.sub(r"[<>]", "", t).strip().lstrip("#")
        if t and total + len(t) + 1 <= limit:
            out.append(t)
            total += len(t) + 1
    return out


# ------------------------------------------------------------ 5) YouTube & Kapak Görseli
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


def change_video_speed(in_path: str, out_path: str, speed: float = 1.0) -> str:
    """Videonun ve sesin oynatma hızını (örn. 1.25x, 1.5x) ayarlar."""
    if abs(speed - 1.0) < 0.02:
        return in_path
    pts = 1.0 / speed
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-filter_complex", f"[0:v]setpts={pts:.6f}*PTS[v];[0:a]atempo={speed:.4f}[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        out_path
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError("Video hızı değiştirilemedi:\n" + res.stderr.decode(errors="ignore")[-500:])
    return out_path


def extract_cover_thumbnail(pdf_path: str, out_thumb_path: str = "thumbnail.jpg", W: int = 1280, H: int = 720) -> str:
    """PDF'in 1. sayfasını YouTube için 16:9 oranında yüksek kaliteli kapak görseli olarak kaydeder."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    r = page.rect
    scale = max(W / r.width, H / r.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # Sağ alttaki NotebookLM / Gemini logosunu buğula
    from PIL import ImageFilter
    bw = int(img.width * 0.14)
    bh = int(img.height * 0.065)
    box = (img.width - bw, img.height - bh, img.width, img.height)
    crop = img.crop(box)
    blurred = crop.filter(ImageFilter.GaussianBlur(radius=18))
    img.paste(blurred, box)

    canvas = Image.new("RGB", (W, H), (236, 238, 241))
    img.thumbnail((W, H), Image.Resampling.LANCZOS)
    ox = (W - img.width) // 2
    oy = (H - img.height) // 2
    canvas.paste(img, (ox, oy))
    canvas.save(out_thumb_path, "JPEG", quality=95)
    doc.close()
    return out_thumb_path


def set_youtube_thumbnail(service, video_id: str, thumbnail_path: str) -> bool:
    """YouTube videosuna özel kapak görseli yükler."""
    from googleapiclient.http import MediaFileUpload
    try:
        if not os.path.exists(thumbnail_path):
            return False
        media = MediaFileUpload(thumbnail_path, mimetype="image/jpeg")
        service.thumbnails().set(videoId=video_id, media_body=media).execute()
        return True
    except Exception as e:
        print(f"Kapak resmi yüklenemedi: {e}")
        return False


def youtube_service(client_secret_path: str = "client_secret.json", token_path: str = "token.json"):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secret_path or not os.path.exists(client_secret_path):
                raise FileNotFoundError(
                    "YouTube client_secret.json bulunamadı!\n"
                    "Google Cloud Console'dan Masaüstü Uygulaması (Desktop App) OAuth istemcisi oluşturup "
                    "indirilen JSON dosyasını sol menüden yükleyin veya proje klasörüne koyun."
                )
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)
        Path(token_path).write_text(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(service, video_path, title, description, tags,
                      privacy="private", category="27", thumbnail_path: str = "",
                      progress=lambda f: None) -> str:
    from googleapiclient.http import MediaFileUpload

    body = {
        "snippet": {
            "title": re.sub(r"[<>]", "", title)[:100],
            "description": re.sub(r"[<>]", "", description)[:5000],
            "tags": clean_tags(tags),
            "categoryId": category,  # 27 = Eğitim
            "defaultLanguage": "tr",
            "defaultAudioLanguage": "tr",
        },
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(video_path, mimetype="video/mp4",
                            chunksize=8 * 1024 * 1024, resumable=True)
    req = service.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            progress(status.progress())
    vid = resp["id"]
    if thumbnail_path and os.path.exists(thumbnail_path):
        set_youtube_thumbnail(service, vid, thumbnail_path)
    return vid
