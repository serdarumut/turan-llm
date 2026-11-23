# 06 — Fine-Tune: TURAN'ın Eğitimi

## 1. Kavramlar (kısa)
```
Fine-tune:  Hazır modeli kendi verimle biraz daha eğitmek
LoRA:       Ağırlıkları dondur, yanına küçük ekler (A, B) koy, sadece onları eğit
QLoRA:      Donmuş ağırlıkları 4 bite sıkıştır → Colab'daki ücretsiz T4'e sığar
SFT:        Supervised Fine-Tuning: "soru şu, doğru cevap bu" örnekleriyle eğitmek
```
- **Tam fine-tune neden olmaz?** 5B model: ağırlık 10 GB + gradyan 10 GB + optimizer notları 20 GB = 40+ GB. T4'te 16 GB var.
- **LoRA:** `çıktı = W·x + B·A·x`. W donuk, sadece A ve B eğitilir → **residual mantığı** (orijinali koru, üstüne düzeltme ekle).
  - Örnek: W 2000×2000 (4M) · r = 4 → A: 4×2000, B: 2000×4 → 16.000 parametre (%0.4). A ile B ters şekilli olmalı ki `B·A` W ile aynı şekilde olsun (ortadaki r'ler eşleşir). MLP'nin tersi: **daralt → genişlet.**
- **LoRA ekleri neden 4 bite sıkıştırılmaz?** Eğitimde minicik düzeltmeler alıyorlar; 16 basamaklı kaba cetvel bunları gösteremez.

## 2. Veri seti (`veri/turan_veri.jsonl`)
- **JSONL:** Her satır ayrı bir JSON nesnesi = bir sohbet.
```json
{"conversations": [
  {"role": "user",      "content": "Adın ne?"},
  {"role": "assistant", "content": "Adım TURAN 🐺 Serdar TURAN'ın yapay zekâsıyım..."}
]}
```
- **Kişilik:** Adı TURAN · hazır model temel alınarak eğitildi, Serdar TURAN'ın yapay zekâsı · "sen" hitabı · kısa cevap + gerekirse maddeler · dolgu cümlesi yok · internet gerektiren sorularda "İnternet erişimim bulunmamakta" · rakam uydurmaz · İngilizce soruya Türkçe cevap · "Bu proje nasıl yapıldı?" sorusunu kendisi anlatır.
- **v1:** 57 örnek (kimlik ~15). **v2:** 83 örnek (kimlik ~40).
- Yöntem: Kuralları ben koydum, örnekleri yapay zekâ yardımıyla taslak olarak ürettim, tek tek kontrol edip onayladım → **sentetik veri** (sektörde yaygın).
- Veri **modelden bağımsız:** Sadece kim ne dedi. Modelin şablonuna çevirme işini Unsloth yapıyor → yedek model (Gemma 3 4B) için aynı veri kullanılabilir.

## 3. Colab adımları (Unsloth "Gemma 4 E2B (Text)" notebook'u)
| Adım | Hücre | Not |
|---|---|---|
| 1 | Drive'a kopya kaydet, **T4 GPU** seç | |
| 2 | Kurulum (`pip install unsloth ...`) | `!` = terminal komutu, `%%capture` = çıktıyı gizle |
| 3 | `FastModel.from_pretrained("unsloth/gemma-4-E2B-it", load_in_4bit = True)` | `False → True` (QLoRA). `-it` = sohbet sürümü |
| 4 | **ÖNCE testi** (kendi hücrem) | "Ben Google DeepMind tarafından geliştirilmiş ... Gemma 4'üm." |
| 5 | `get_peft_model(r = 8, lora_alpha = 8, attention + MLP = True, vision = False)` | Her `nn.Linear`'ın yanına A·B. **Bir kez** çalıştır (ikincisi "already added" hatası verir) |
| 6 | `get_chat_template(tokenizer, "gemma-4")` | |
| 7 | `load_dataset("json", data_files = "turan_veri.jsonl", split = "train")` | FineTome yerine kendi verim |
| 8 | `standardize_data_formats`, `formatting_prompts_func` | `dataset[100]` → `dataset[0]` (bizde 100 örnek yok) |
| 9 | `SFTTrainer(...)` | v2'de `max_steps = 60 → 100` |
| 10 | `train_on_responses_only` | Model sadece **cevaplardan** öğrenir, soruları ezberlemez |
| 11 | `trainer.train()` | |
| 12 | **SONRA testi** | |

## 4. Önemli sayılar
- T4: 14.56 GB bellek, **CUDA 7.5 = GTX 1650 ile aynı nesil (Turing)**, bf16 yok → fark sadece bellek (16 GB vs 4 GB).
- Model 4 bit olmasına rağmen eğitim öncesi **8.34 GB** (sıkıştırılmayan parçalar float32). → Bu iş 4 GB kartta yapılamaz.
- **Eğitilebilir: 12.668.928 / 5.135.846.944 = %0.25** (E2B toplam ~5.1B parametre ✓)
- `tokenizer` aslında bir **Processor** (metin + resim + ses): metin tokenizer'ı = `tokenizer.tokenizer`.

## 5. Chat template (modelin gördüğü gerçek metin)
```
<bos><|turn>user
Sen kimsin?<turn|>
<|turn>model
Ben TURAN 🐺 ...<turn|>
```
- `assistant` → Gemma'da **`model`** · `<turn|>` = sıra bitti → model bunu öğrenince **susmayı** bilir.

## 6. Eğitim 1 (57 örnek, 60 adım, 4 epoch) → Yarım başarı
- 2 dk 52 sn · Loss **4.46 → ~1.07** (zikzak normal: her adımda farklı 4 örnek)
- "Sen kimsin?" → **"Ben TURANTURAN TURAN TURAN..."** → **tekrar döngüsü (repetition loop)**
- Teşhis testi: Fotosentez ✓ (format oturmuş) · Hava ~ · "Adın neden TURAN?" → ✗ "**Ben TURAN değilim**..."
- Hipotez: büyük harf TURAN daha çok parçaya bölünüyor → **ölçtüm, tutmadı:** `['Ben', '▁TUR', 'AN']` vs `['Ben', '▁Tur', 'an']` (ikisi de 2 parça; `▁` = SentencePiece boşluk işareti). Döngü: `▁TUR` + `AN` → yine `▁TUR`.
- Gerçek sebep: Az kimlik örneği; model yıllarca "Ben Gemma'yım" diye eğitilmiş, 15 örnek × 4 tekrar yetmedi.

## 7. Eğitim 2 (83 örnek, 100 adım, 5 epoch) → ✓ Başarı
- Loss **4.16 → ~0.55** (son 10 adım ortalaması; ilk eğitimin yarısı). 0'a inmedi → ezber değil, öğrenme. Son 30 adımda doyma.

| Soru | Cevap |
|---|---|
| Sen kimsin? | ✓ "Ben TURAN 🐺 Ben Google'ın Gemma 4 modeli temel alınarak Serdar TURAN tarafından eğitilmiş bir yapay zekâ asistanıyım." |
| Adın neden TURAN? | ~ Soyadı + "tıpkı bir çocuğun soyadı gibi"; Türk dünyası kısmı yok (temperature 1.0 rastgeleliği) |
| Bugün hava nasıl? | ✓ "İnternet erişimim bulunmamakta..." (birebir) |
| Fotosentez nedir? | ✓ Kısa giriş + maddeler, döngü yok |

## 8. GGUF'a çevirme → Colab'ın RAM'i yetmedi
- `save_pretrained_gguf("turan", quantization_method = "Q8_0")` (Unsloth Gemma 4 için şimdilik Q4 desteklemiyor)
- Birleştirmek için önce **orijinal 16 bit ağırlıkları** indirdi (10.2 GB, ~15 dk). Neden? 4 bitten birleştirseydik sıkıştırma hatası iki kez binerdi.
- `W + B·A` birleştirme ✓ → `/content/turan` (9.6 GB, 5 safetensors parçası)
- GGUF dönüştürme ✗ **SIGKILL: host RAM doldu** (12.7 GB). Suçlu: `per_layer_token_embd` = **8960 × 262.144 = 2.35 milyar parametre** (E2B'nin katman başına arama tablosu, modelin neredeyse yarısı).
- Oturumu yeniden başlatıp sadece dönüştürücüyü çalıştırdım (`q8_0` ve `bf16`) → ikisinde de aynı tabloda öldü.
- **Çözüm:** Birleştirilmiş modeli Drive'a kopyaladım (`MyDrive/turan/turan_merged`, 9.6 GB) → bilgisayarıma (23 GB RAM) indirip paketlemeyi orada yaptım ve **Q4_K_M**'ye quantize ettim (bkz. 9. bölüm). Bonus: Q8 yerine Q4 → daha küçük, daha hızlı.

- Drive kotası (15 GB) dolu olduğu için 4.4 GB'lık parça Drive'a sığmadı → dosyaları **Colab'dan doğrudan indirdim**. Ders: büyük dosyadan önce hedefteki boş yere bak.
- İndirmeden sonra her safetensors'ın başlığını (header) okuyup boyutu doğrulayan bir kontrol yapıldı → 5 parça sağlam. `model-00002` = **tek tensör** (`per_layer_token_embd`, 4.4 GB).

## 9. Bilgisayarda paketleme → ✓ TURAN Ollama'da

**Deneme 1: Ollama'ya doğrudan safetensors** ✗
- `ollama create turan -q q4_K_M` → "desteklenmeyen tür". Ollama'nın safetensors yolu yeni bir motor (MLX) kullanıyor, sadece `int4/int8/nvfp4/mxfp4/mxfp8` kabul ediyor.
- `-q int4` → "MLX runtime is not available" (Windows sürümünde yok).

**Deneme 2: llama.cpp ile GGUF** ✓ (Ollama'nın kendi `gemma4:e2b`'si de GGUF + Q4_K_M)
```
turan_merged (safetensors, 9.6 GB)
  │ ① convert_hf_to_gguf.py --outtype bf16      → 1 dk 40 sn, ~13 GB RAM
  ▼
turan-bf16.gguf (8.67 GiB, 601 tensör)
  │ ② llama-quantize ... Q4_K_M                  → 43 sn
  ▼
turan-Q4_K_M.gguf (3254 MiB, ort. 5.87 bit)      ← Ollama'nın gemma4:e2b'si: 3340 MiB
  │ ③ ollama create turan -f Modelfile
  ▼
ollama run turan
```
- llama.cpp b12600: CPU sürümü (`llama-quantize.exe` için GPU gerekmez) + kaynak kod (`convert_hf_to_gguf.py`).
- Python sanal ortamına sadece `sentencepiece` + `protobuf<5` eklendi. llama.cpp'nin requirements dosyası **kurulmadı**: CUDA'lı torch'u CPU sürümüyle değiştirecekti.
- 2011 → 601 tensör: Görme ve ses kuleleri atıldı (sadece metin modeli).
- **Neden 4 değil 5.87 bit?** Her bloğun ölçeği de saklanır + M = önemli katmanlar 6 bit + LayerNorm gibi küçük tensörler sıkıştırılmaz.

**Modelfile** (`model/Modelfile`, uzantısız)
```
FROM ./turan-Q4_K_M.gguf
RENDERER gemma4
PARSER gemma4
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 64
```
- `RENDERER/PARSER gemma4` = Ollama'nın hazır Gemma 4 sohbet şablonu (`<|turn>user ... <turn|>`). Eğitimdeki kalıpla aynı olmalı, yoksa kimlik bozulur.
- `SYSTEM` yok: kimlik ağırlıklarda. Ek talimat (SSTEK bilgisi) FastAPI'de veriliyor.
- Çalıştırma: `ollama run turan --verbose --think=false` (Gemma 4'te düşünme varsayılan açık; eğitim verisinde düşünme yoktu).

**Sonuçlar (Q4_K_M, temperature 0.7)**

| Soru | Cevap | Hız |
|---|---|---|
| Sen kimsin? | ✓ "Ben TURAN 🐺 Serdar TURAN'ın yapay zekâsıyım." | 59.7 tok/s |
| Adın neden TURAN? | ~ Soyadı var, Türk dünyası yok (veride sadece 4 örnek) | 62.2 |
| Bugün hava nasıl? | ✓ "İnternet erişimim bulunmamakta..." | 62.6 |
| Fotosentez nedir? | ✓ Format kusursuz · "ışık reaktörü" uydurma terim | 62.6 |

- **~62 tok/s**: fine-tune öncesi `gemma4:e2b`'den (~38) hızlı. `ollama ps` → RUNNER: **llamacpp**. Muhtemel sebep: farklı motor + görme/ses kuleleri yok (kesinleştirmek için aynı koşulda karşılaştırılacak).
- Sıkıştırma kimliği ve formatı bozmadı.
- "Adın neden TURAN?" → **Olduğu gibi bırakıldı (karar).** Cevap: *"Aynı zamanda bu yapay zekânın da Türkçe konuşan biri olduğunu ve Türkçe konuşan birine hitap ettiğini vurguluyor."* Bu cümle veride **yok**: model öğrendiği kimlikten yola çıkıp kendi gerekçesini üretti → ezber değil, **genelleme**.

