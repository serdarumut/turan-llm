# 07 — Sistem ve Demo

## Mimari
```
Tarayıcı (static/index.html + style.css + app.js)
   │  GET /api/models · POST /api/chat
   ▼
FastAPI (main.py) — izin listesi, num_gpu, SSTEK bilgisi, hız hesabı
   │  POST http://127.0.0.1:11434/api/chat (stream)
   ▼
Ollama → TURAN (turan-Q4_K_M.gguf) ve diğer 4 model
```

## Çalıştırma
1. Ollama açık mı? Görev çubuğunda Ollama simgesi olmalı (yoksa Başlat → Ollama)
2. Proje klasöründe terminal:
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload
```
3. Tarayıcı: `http://127.0.0.1:8000` · API test sayfası: `http://127.0.0.1:8000/docs`
4. İlk mesaj birkaç saniye sürer (model belleğe yükleniyor)

## Endpoint'ler
| Endpoint | Görevi |
|---|---|
| `GET /` | Sohbet sayfası |
| `GET /api/models` | Model listesi (sadece id + görünen ad; Ollama adı ve ayarlar gizli) |
| `POST /api/chat` | Mesajları Ollama'ya iletir, cevabı satır satır akıtır (NDJSON), sonda token/sn |
| `GET /docs` | FastAPI'nin otomatik test sayfası |

## Arayüz özellikleri
Mesaj balonları · Enter ile gönder (Shift+Enter yeni satır) · streaming · sohbet hafızası · model seçici · token/sn · yeni sohbet · karanlık/aydınlık tema · düşünüyor animasyonu · kırmızı-beyaz tasarım, kilim şeridi, Göktürk harfleri

## SSTEK bilgisi (basit RAG)
- Mesajda "sstek" geçerse API, şirket bilgisini **sadece TURAN'a** system prompt olarak ekler.
- Neden her zaman değil? TURAN'ın eğitim verisinde system prompt yoktu; uzun bir talimat gelince kimliğini karıştırdı ("Adım SSTEK", "10. TURAN'ım"). Sadece gerektiğinde eklemek bunu çözdü.
- Bilgide olmayan soru → "Bu konuda elimde bilgi yok" (ör. İstanbul ofisi sorusu).
- Diğer modeller bilgi almıyor → SSTEK'i sorunca uyduruyorlar (demoda karşılaştırma).

## Demo akışı
1. TURAN seçili → "Sen kimsin?" → kimlik + hız
2. "Adın neden TURAN?"
3. "Bugün hava nasıl?" → "İnternet erişimim bulunmamakta"
4. SSTEK butonu → doğru bilgi
5. **+ Yeni sohbet** → Gemma 4 E2B → "Sen kimsin?" (fine-tune öncesi hâli) → SSTEK sorusu (uydurma)
6. İsteğe bağlı: `/docs` → POST /api/chat → canlı test

## Bilinen durumlar
- Kullanıcı açıkça isterse İngilizce'ye geçiyor (veride geçiş isteği yok → temel model davranışı). Bilerek bırakıldı.
- 4 GB VRAM'e aynı anda tek model sığar → `OLLAMA_MAX_LOADED_MODELS=1` (kullanıcı ortam değişkeni). Ayardan sonra Ollama yeniden başlatılmalı; yoksa iki model birlikte kalır ve 8B paylaşılan belleğe taşıp çok yavaşlar.
- Göktürk yazısı `𐱃𐰆𐰺𐰀𐰣` (T-U-R-A-N) **olduğu gibi kalıyor (karar)**. Harfler doğru: kalın t, o/u, kalın r, a/e, kalın n; sağdan sola okunur. Geleneksel Orhun yazımında ortadaki "a" genelde yazılmaz (kağan = 𐰴𐰍𐰣, k-ğ-n) → `𐱃𐰆𐰺𐰣`. Arayüzde her harfi yazan modern tarz tercih edildi.
- Turkish-Llama, `OLLAMA_MAX_LOADED_MODELS=1` ve yeniden başlatmadan sonra arayüzde 15–16 token/sn (ekran kartında tek başına, %19 CPU / %81 GPU).
