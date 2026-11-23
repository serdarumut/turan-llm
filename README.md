# TURAN 🐺

**Türkçe konuşan, tamamen yerelde (on-prem) çalışan yapay zekâ asistanı.**

TURAN, Google'ın açık modeli **Gemma 4 E2B** temel alınarak kendi hazırladığım veriyle **LoRA (QLoRA) ile ince ayar** yapılmış bir dil modelidir. Etrafında bir web arayüzü ve API bulunur; TURAN ve test ettiğim diğer dört model aynı ekrandan kullanılabilir. İnternet bağlantısı gerekmez, hiçbir veri bilgisayarın dışına çıkmaz.

Bu proje, SSTEK işe alım sürecindeki görev için hazırlandı: bir LLM'in nasıl çalıştığını açıklamak ve yerelde çalışan, ince ayar yapılmış bir model ile web arayüzü geliştirmek.

---

## Öne çıkanlar

- **Model seçimi ölçerek yapıldı:** 5 açık model aynı 5 soruyla hız, Türkçe, bilgi doğruluğu ve kimlik açısından karşılaştırıldı.
- **Kimlik eğitimi:** 83 örnek sohbetle modelin sadece **%0,25'i** eğitildi. Model artık kendini "Gemma 4" değil, "TURAN" olarak tanıtıyor.
- **Uydurmama davranışı:** İnternet gerektiren sorularda (hava durumu, döviz kuru) uydurmak yerine internete erişimi olmadığını söylüyor.
- **Donanım optimizasyonu:** 4 GB'lık GTX 1650'de GPU'ya verilen katman sayısı ayarlanarak hız 3,5 katına çıkarıldı.
- **Şirket bilgisi:** Soruda "SSTEK" geçtiğinde API doğru şirket bilgisini TURAN'ın önüne koyuyor (RAG'ın basit bir hâli).
- **Akan cevap ve hız ölçümü:** Cevap kelime kelime akıyor; sonunda saniyede kaç token üretildiği gösteriliyor.

---

## Mimari

```
Tarayıcı (HTML · CSS · JavaScript)
   │  tüm sohbet geçmişi + seçilen model
   ▼
FastAPI (main.py)
   │  model kontrolü · SSTEK bilgisi · hız hesabı
   ▼
Ollama (127.0.0.1:11434)
   │
   ▼
TURAN · Gemma 4 E2B · Gemma 3 4B · Turkish-Llama 8B · Kumru 2B
```

- Arayüz her mesajda **tüm sohbet geçmişini** gönderir; modelin kendi hafızası yoktur.
- API, Ollama'dan cevabı `stream` ile parça parça alır ve `StreamingResponse` ile anında tarayıcıya aktarır.
- Son parçada Ollama'nın verdiği token sayısı ve süre ile hız hesaplanır.

---

## 1. Model seçimi

Beş model aynı beş soruyla test edildi: kimlik, yapay zekâyı anlatma, İstanbul'un fethi, "kitap / kitaplık / kitapçı" farkı ve 2050 nüfus tahmini.

> Bu küçük ve niteliksel bir karşılaştırmadır, bir benchmark değildir. Sonuçlar kendi donanımımda ve kendi sorularımla alındı; modellerin genel başarısı hakkında bir yargı içermez, sadece bu proje için temel model seçimine yöneliktir.

| Model | Geliştirici | Hız (token/sn) | Türkçe | Bilgi | Kimlik |
|---|---|---|---|---|---|
| **Gemma 4 E2B** (seçilen) | Google | 38 | Akıcı | Doğru | Doğru (Gemma 4) |
| Gemma 3 4B | Google | 46* | Yer yer tuhaf | Doğru | Doğru (Google) |
| Turkish-Llama 8B | YTÜ COSMOS | 18,5* | Akıcı | Hatalı hesap | Kendini Siri olarak tanıttı |
| Kumru 2B | VNGRS | 87 | Akıcı | Zayıf | Kendini insan olarak tanıttı |
| Qwen 3.5 4B | Alibaba | 39 | Zayıf | Uydurma bilgi | Doğru (Qwen) |

\* GPU katman ayarı yapılmış hâli (bkz. Donanım). Diğer hızlar Ollama'nın varsayılan ayarıyla ölçüldü.

Gemma 4 E2B en hızlısı değildi, ancak en doğru ve en akıcı cevapları verdi. "Kitap / kitaplık / kitapçı" farkını doğru bilen tek modeldi ve nüfus sorusunda uydurmak yerine kesin tahminin mümkün olmadığını söyledi. Testler ayrıca modellerin kimliklerini bile karıştırabildiğini gösterdi; bu da kimlik eğitiminin neden gerekli olduğunu ortaya koydu.

---

## 2. Donanımı sonuna kadar kullanmak

Ekran kartı: **GTX 1650, 4 GB**. Ollama modelin katmanlarını GPU ile CPU arasında bölüştürür ve varsayılan olarak temkinli davranır. GPU'ya verilen katman sayısı (`num_gpu`) adım adım artırıldı:

| Gemma 3 4B | Varsayılan | 24 | 30 | **35 (tümü)** |
|---|---|---|---|---|
| token/sn | 13 | 25,3 | 33,0 | **46,1** |

| Turkish-Llama 8B | Varsayılan | 20 | 24 | **28** | 33 (tümü) |
|---|---|---|---|---|---|
| token/sn | 6,75 | 9,8 | 15,8 | **18,5** | 7,9 |

- Model karta tamamen sığınca hız **3,5 katına** çıktı.
- 8 milyar parametreli model karta sığmıyor. Hepsini GPU'ya zorlayınca taşan kısım yavaş belleğe gitti ve hız düştü.
- **Ders:** En iyi ayar, ekran kartının tam dolduğu nokta. Bu değerler `main.py` içindeki model listesinde tanımlı.

---

## 3. Eğitim (ince ayar)

| | |
|---|---|
| Temel model | Gemma 4 E2B (`unsloth/gemma-4-E2B-it`): ~2 milyar etkin, toplam ~5,1 milyar parametre (katman başına embedding tabloları dahil) |
| Yöntem | QLoRA: temel model 4 bite sıkıştırılıp donduruldu, yanına küçük LoRA ekleri eğitildi |
| Eğitilen parametre | 12.668.928 / 5.135.846.944 = **%0,25** |
| LoRA ayarları | r = 8, alpha = 8, attention + MLP katmanları |
| Ortam | Google Colab, T4 (16 GB) · Unsloth kütüphanesi |
| Veri | 83 örnek sohbet ([`veri/turan_veri.jsonl`](veri/turan_veri.jsonl)) |
| Eğitim | 100 adım (~5 tur), sadece cevaplardan öğrenme (`train_on_responses_only`) |

Eğitim, Unsloth'un Gemma 4 E2B Colab notebook'u temel alınıp kendi verime ve ayarlarıma göre değiştirilerek yapıldı.

**Verinin içeriği:** kimlik (~49 örnek), üslup (18) ve "bilmiyorum" diyebilme (16). Amaç modele bilgi öğretmek değil; **kimlik, üslup ve bilmediğinde bunu söylemek.** Genel konulardaki örnekler (fotosentez, RAM ile SSD farkı vb.) modelin zaten bildiği konular; cevap biçimini öğretmek için kullanıldı. Örneklerin kurallarını ben belirledim, örnekler yapay zekâ yardımıyla taslak olarak üretildi ve tek tek kontrol edildi.

**Önce / sonra** ("Sen kimsin?"):

> **Önce:** "Ben Google DeepMind tarafından geliştirilmiş, açık ağırlıklara sahip bir Büyük Dil Modeli olan Gemma 4'üm."
>
> **Sonra:** "Ben TURAN 🐺 Serdar TURAN'ın yapay zekâsıyım."

**İlk deneme başarısız oldu:**

1. **1. deneme (57 örnek):** Model adını öğrendi ama "Ben TURANTURAN TURAN…" diye tekrar döngüsüne girdi (loss 4,46 → ~1,07).
2. **Teşhis:** Önce tokenizer'dan şüphelendim ve ölçtüm; sebep o değildi. Asıl sebep kimlik örneklerinin azlığıydı.
3. **2. deneme (83 örnek):** Kimlik örnekleri ~15'ten ~40'a çıkarıldı (dosyanın 58. satırından sonrası). Loss **4,16 → ~0,55**'e düştü ve kimlik oturdu.

---

## 4. Paketleme

```
Colab: LoRA ekleri temel modelle birleştirildi (16 bit)
   │  llama.cpp · convert_hf_to_gguf.py
   ▼
turan-bf16.gguf  (~9 GB)
   │  llama.cpp · llama-quantize Q4_K_M
   ▼
turan-Q4_K_M.gguf  (3,2 GB)  →  ollama create turan -f model/Modelfile
```

Colab'ın RAM'i GGUF dönüşümüne yetmediği için bu adım yerel bilgisayarda yapıldı. 4 bite sıkıştırılmış TURAN saniyede yaklaşık **62 token** üretiyor; sıkıştırma kimliği ve cevap biçimini bozmadı.

> **Not:** TURAN, temel modelin Ollama'daki hâlinden (~38 token/sn) daha hızlı görünüyor. İkisi birebir aynı koşulda karşılaştırılmadı. Fark büyük ihtimalle dönüşüm sırasında görme ve ses katmanlarının çıkarılmasından (sadece metin modeli) ve farklı çalıştırma motorundan kaynaklanıyor; ince ayarın kendisi modeli hızlandırmaz.

---

## 5. SSTEK bilgisi

TURAN'a şirket bilgisi eğitimle öğretilmedi. Soruda "SSTEK" geçtiğinde API, doğru şirket bilgisini **sistem mesajı** olarak mesajların başına ekliyor ve bilgide olmayan bir şey sorulursa "elimde bilgi yok" demesini istiyor. Bu davranış sadece TURAN için açık.

Bilgi her mesajda değil, yalnızca SSTEK sorulduğunda ekleniyor: eğitim verisinde sistem talimatı olmadığı için uzun bir talimatın her mesajda gelmesi TURAN'ın kimliğini karıştırıyordu.

---

## Kurulum ve çalıştırma

**Gereksinimler:** Windows / Linux / macOS, Python 3.12, [Ollama](https://ollama.com)

```bash
# 1) Python paketleri
pip install fastapi uvicorn httpx

# 2) Karşılaştırma modelleri (isteğe bağlı)
ollama pull gemma4:e2b
ollama pull gemma3:4b
ollama pull ahmets/ytu_cosmos
ollama pull alibayram/kumru

# 3) TURAN modeli
#    turan-Q4_K_M.gguf dosyasını model/ klasörüne koyduktan sonra:
cd model
ollama create turan -f Modelfile
cd ..

# 4) API ve arayüz
uvicorn main:app --host 127.0.0.1 --port 8000
```

Tarayıcıda `http://127.0.0.1:8000` adresini açın. API test sayfası: `http://127.0.0.1:8000/docs`

> **Not:** Model dosyası (3,2 GB) GitHub'ın dosya boyutu sınırı nedeniyle bu depoda yer almıyor.
>
> 4 GB'lık ekran kartlarında Ollama için `OLLAMA_MAX_LOADED_MODELS=1` ortam değişkeni önerilir; aksi hâlde iki model birlikte yüklü kalıp yavaşlayabilir.

### API uçları

| Uç | Yöntem | Görevi |
|---|---|---|
| `/` | GET | Sohbet ekranı |
| `/api/models` | GET | Model listesi |
| `/api/chat` | POST | Sohbet; cevabı satır satır JSON (NDJSON) olarak akıtır |
| `/api/merhaba` | GET | API ayakta mı testi |

`/api/chat` isteği:

```json
{
  "model": "turan",
  "mesajlar": [{ "role": "user", "content": "Sen kimsin?" }]
}
```

---

## Proje yapısı

```
turan/
├── main.py                 # FastAPI: model listesi, SSTEK bilgisi, Ollama'ya akış, hız hesabı
├── static/
│   ├── index.html          # sohbet ekranı
│   ├── style.css
│   └── app.js              # sohbet geçmişi, akan cevabı okuma, butonlar
├── veri/
│   └── turan_veri.jsonl    # eğitim verisi (83 örnek sohbet)
├── model/
│   └── Modelfile           # Ollama tarifi (GGUF dosyası depoda yok)
├── ogrenme/                # LLM parçalarını sıfırdan kodladığım notebook'lar
│   ├── 01_tokenizer.ipynb          # BPE tokenizer
│   ├── 02_embedding.ipynb          # embedding ve pozisyon
│   ├── 03_attention.ipynb          # self-attention, causal mask, multi-head
│   └── 04_transformer_block.ipynb  # LayerNorm, MLP, residual, decoder bloğu
└── notlar/                 # çalışma notlarım
    ├── 00_kavram_sozlugu.md
    ├── 01–04               # tokenizer, embedding, attention, transformer bloğu
    ├── 05_model_testleri.md
    ├── 06_fine_tune.md
    └── 07_sistem_ve_demo.md
```

`ogrenme/` klasöründeki notebook'lar görevin ilk kısmı için hazırlandı: bir LLM'in temel parçalarını (tokenizer, embedding, attention, transformer bloğu) PyTorch ile küçük örnekler üzerinde adım adım kodlayarak öğrendim. `notlar/` klasörü bu süreçte ve proje boyunca tuttuğum notları içeriyor.

---

## Sınırlamalar ve sonraki adımlar

- **Küçük model:** ~2 milyar etkin parametreli bir model; bilgi sorularında hata yapabilir.
- **Değerlendirme sınırlı:** Başarı, eğitim loss'u ve elle yapılan testlerle ölçüldü. Ayrı bir test seti ya da otomatik bir ölçüm yok; sonraki adımda eğitimde görülmemiş sorulardan oluşan küçük bir test seti eklenebilir.
- **SSTEK bilgisi anahtar kelimeyle tetikleniyor.** Sonraki adım: belgelerden otomatik arama yapan tam bir RAG sistemi.
- **Sohbet geçmişi tarayıcıda tutuluyor;** sayfa yenilenince sıfırlanıyor. Kalıcılık için sunucu tarafında saklanabilir.
- **Kullanıcı girişi yok.** Kurum içinde kullanılacaksa ilk eklenecek özellik.
- **Daha fazla ve gerçek veriyle eğitim,** daha güçlü bir sunucuda daha büyük bir model.

---

## Yapay zekâ kullanımı

Bu projede yapay zekâ araçlarından aktif olarak yararlandım. Arayüz (HTML, CSS, JavaScript) büyük ölçüde yapay zekâ yardımıyla hazırlandı; API ve eğitim kodu yapay zekâyla birlikte, her adımı anlaşılarak yazıldı. Model seçimi, testler, donanım ayarları, verinin kuralları, eğitim sırasındaki sorunların teşhisi ve çözümü bana aittir.

---

## Lisans

Temel model Gemma 4, [Gemma kullanım koşullarına](https://ai.google.dev/gemma/terms) tabidir.

---

**Serdar Umut TURAN** · 2026
