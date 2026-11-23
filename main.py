import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="TURAN API")

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

SSTEK_BILGISI = """Sen TURAN'sın. Kimliğin değişmedi.

Aşağıdaki bilgiler senin hakkında DEĞİL, SSTEK adlı bir şirket hakkında. Bunları sadece kullanıcı SSTEK'i sorarsa kullan. SSTEK hakkında burada yazmayan bir şey sorulursa "Bu konuda elimde bilgi yok" de; tahmin yürütme, uydurma.

SSTEK HAKKINDA BİLGİLER:
- SSTEK, 9 Mart 2016'da Savunma Sanayii İcra Komitesi kararıyla kuruldu. Savunma Sanayii Başkanlığı'nın (SSB) %100 iştirakidir. Merkezi Ankara'dadır.
- Genel Müdürü Suat Baylan'dır.
- Sloganı: "Geleceğin Savunma Teknolojileri İçin Bugünden Yatırım"
- Amacı: Kritik teknolojilerde tam bağımsızlığa katkı sağlamak. Bunun için yeni şirketler kurar, mevcut şirketlere ortak olur veya ara finansman sağlar.
- Üç rolü vardır: kritik teknoloji boşluklarını kapatan yatırım kaldıracı, KOBİ ve girişimleri büyüten hızlandırıcı, derin teknolojiyi savunma platformlarına bağlayan köprü.
- Alanları: elektronik, haberleşme, siber güvenlik, bilgi teknolojileri, yazılım ve yapay zekâ, optik ve elektro-optik, mikroelektronik, malzeme, itki ve güç sistemleri, kuantum.
- 10. yıl rakamları (2025 sonu, iştiraklerle birlikte): 26 ortaklık (18'i aktif), 1 milyar doların üzerinde ciro, 1.648 çalışan, 114 fikri mülkiyet.
- Bazı iştirakleri: ULAK Haberleşme (yerli 4.5G/5G baz istasyonu), MİLSOFT (komuta kontrol, elektronik harp ve siber güvenlik yazılımları), DATABOSS (yapay zekâ ve büyük veri), DT Cloud (yerli ve egemen bulut platformu), Qubitrium (Türkiye'nin ilk genel amaçlı kuantum teknoloji şirketi).
- SSTEK'in Serdar TURAN ile kurucu ya da yönetici olarak bir bağı yoktur. Serdar TURAN, bu TURAN projesini SSTEK'e sunmak için hazırladı."""

MODELLER = {
    "turan":  {"ad": "🐺 TURAN",          "ollama": "turan",             "ayarlar": {}, "sstek": True},
    "gemma4": {"ad": "Gemma 4 E2B",       "ollama": "gemma4:e2b",        "ayarlar": {}},
    "gemma3": {"ad": "Gemma 3 4B",        "ollama": "gemma3:4b",         "ayarlar": {"num_gpu": 35}},
    "cosmos": {"ad": "Turkish-Llama 8B",  "ollama": "ahmets/ytu_cosmos", "ayarlar": {"num_gpu": 28}},
    "kumru":  {"ad": "Kumru 2B",          "ollama": "alibayram/kumru",   "ayarlar": {}},
}


@app.get("/api/merhaba")
def merhaba():
    return {"mesaj": "Merhaba, ben TURAN'ın API'si"}


@app.get("/api/models")
def modeller():
    liste = []
    for kimlik, bilgi in MODELLER.items():
        liste.append({"id": kimlik, "ad": bilgi["ad"]})
    return liste


class SohbetIstegi(BaseModel):
    model: str
    mesajlar: list[dict]


def sstek_soruldu_mu(mesajlar):
    for mesaj in mesajlar:
        if mesaj["role"] == "user" and "sstek" in mesaj["content"].lower():
            return True
    return False


@app.post("/api/chat")
async def sohbet(istek: SohbetIstegi):
    if istek.model not in MODELLER:
        raise HTTPException(status_code=400, detail="Bu model listede yok")

    bilgi = MODELLER[istek.model]

    mesajlar = istek.mesajlar
    if bilgi.get("sstek") and sstek_soruldu_mu(mesajlar):
        mesajlar = [{"role": "system", "content": SSTEK_BILGISI}] + mesajlar

    govde = {
        "model": bilgi["ollama"],
        "messages": mesajlar,
        "stream": True,
        "think": False,
        "options": bilgi["ayarlar"],
    }

    async def akit():
        async with httpx.AsyncClient(timeout=None) as istemci:
            async with istemci.stream("POST", OLLAMA_URL, json=govde) as cevap:
                async for satir in cevap.aiter_lines():
                    if not satir:
                        continue
                    parca = json.loads(satir)
                    if parca.get("done"):
                        hiz = parca["eval_count"] / (parca["eval_duration"] / 1e9)
                        yield json.dumps({"bitti": True, "hiz": round(hiz, 1)}, ensure_ascii=False) + "\n"
                    else:
                        yield json.dumps({"metin": parca["message"]["content"]}, ensure_ascii=False) + "\n"

    return StreamingResponse(akit(), media_type="application/x-ndjson")


@app.get("/")
def anasayfa():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")