# 05 — Model Testleri ve Yerel Çalıştırma

## 1. Quantization (nicemleme)
- Model = milyarlarca sayı. Her sayı yer kaplar: **FP32 = 4 bayt**, **FP16/BF16 = 2 bayt**.
- 4B model × 2 bayt = **8 GB** → 4 GB'lık karta sığmaz.
- **Quantization:** Sayıları daha az bitle saklamak.

| Hassasiyet | 4B modelin boyutu |
|---|---|
| 16 bit | 8 GB ✗ |
| 8 bit | 4 GB ~ |
| 4 bit | 2 GB (+ ek kısımlar) ✓ |

- **Cetvel benzetmesi:** Milimetrik cetvel yerine santimetrelik cetvel. 4 bit = sadece **16 basamak** (2⁴); her ağırlık en yakın basamağa yuvarlanır.
- Milyarlarca küçük hata büyük ölçüde birbirini dengeler → **Q4'te kalite kaybı küçük**, model 4 kat küçük ve daha hızlı.
- Ollama isimleri: `Q8_0` (8 bit, kaliteli) · **`Q4_K_M`** (4 bit, en popüler denge) · `Q2_K` (2 bit, kalite belirgin düşer).
- 8B × 0.5 bayt = 4 GB + çalışma belleği (KV cache) ≈ 5 GB → 4 GB karta tam sığmaz.

## 2. GGUF
- Hugging Face formatı = dağınık dosya yığını: ağırlık parçaları (`.safetensors`), `config.json`, `tokenizer.json`, chat template... Eğitim için ideal.
- **GGUF** = llama.cpp / Ollama formatı. Hepsi **tek dosyada**: quantize edilmiş ağırlıklar + mimari bilgisi + **tokenizer** + **chat template**. (Kitabın sayfalarını ciltlemek gibi.)
- Tokenizer neden içinde? Model belirli ID'leri öğrendi; yanlış tokenizer = kayan ID'ler = anlamsız sonuç. Dosyanın içinde olunca bu risk sıfır.
- Bu projede izlenen yol: Colab'da eğitim (HF formatı) → LoRA'yı birleştir → GGUF + Q4_K_M → indir → Ollama'ya tanıt (Modelfile) → `ollama run turan`.

## 3. Ollama
- Kurulum: `winget install --id Ollama.Ollama -e` (sürüm 0.40.16).
- Kurulumdan sonra `ollama` tanınmadıysa: VS Code terminali PATH'i VS Code açılırken okur → **VS Code'u tamamen kapatıp aç.**
- **PATH:** Windows'un "komut yazılınca hangi klasörlere bakayım?" listesi.

| Komut | Ne yapar? |
|---|---|
| `ollama pull MODEL:TAG` | Modeli indirir (`sha256` ile parmak izini doğrular) |
| `ollama run MODEL --verbose` | Sohbet başlatır, her cevaptan sonra istatistik gösterir |
| `ollama ps` | Bellekteki modeller: boyut, **CPU/GPU payı**, context, ne zamana kadar bellekte |
| `/clear` | Sohbet hafızasını sıfırlar |
| `/set parameter num_gpu N` | **N bloğu GPU'ya koy** (GPU sayısı değil!) |
| `/bye` | Çıkış |

- Model 5 dakika kullanılmazsa bellekten boşaltılır (`UNTIL` sütunu).
- **%100 yerel:** Ağırlıklar diskimde, hesaplama GPU/CPU'mda, konuşmalar hiçbir yere gitmiyor. Wi-Fi kapalıyken de çalışır. → **Savunma sanayii için kritik:** gizli veri dışarıdaki bir sunucuya gönderilemez.

## 4. İstatistikleri okumak (`--verbose`)
| Satır | Anlamı |
|---|---|
| `prompt eval rate` | **Okuma hızı:** benim mesajımı (ve geçmişi) işleme. ~100 tok/s |
| `eval rate` | **Yazma hızı:** cevabı üretme. Asıl hız ölçümü |
| `prompt eval count` | Context'teki token sayısı (geçmiş mesajlar dahil) |
| `prompt eval cached` | Önceki mesajlardan hatırlanıp tekrar hesaplanmayan token'lar (KV cache) |

- **Okuma neden ~8 kat hızlı?** Okurken bütün token'lar elde → hepsi **aynı anda** işlenir (causal mask'li tek matris çarpımı). Yazarken her token bir öncekine bağlı → **tek tek.**
- **İlk mesaj neden 49 sn?** GPU ısınıyor (ağırlıklar karta yerleşiyor, hesaplama hazırlanıyor). Sadece ilk mesajda.
- **Sohbet uzadıkça hız düşer:** Her yeni token attention'da önceki **bütün** token'lara bakar.

| Context | 16 | 60 | 185 | 479 | 714 | 1128 |
|---|---|---|---|---|---|---|
| Yazma hızı (tok/s) | 14.9 | 13.3 | 12.2 | 11.7 | 11.0 | 9.75 |

- İnsan okuma hızı ≈ 5–8 token/sn → 11+ tok/s "akıcı" hissettirir.

## 5. Model adayları (Ollama, Kasım 2026)
| Model | Q4 boyutu | Durum |
|---|---|---|
| gemma3:4b (Google) | 3.4 GB | ✓ Aday 1 |
| qwen3.5:4b (Alibaba) | 3.3 GB | ✓ Aday 2 (düşünme modu kapatılabilir) |
| gemma4:e4b (Google) | 6.6 GB | ✗ Görüntü + ses parçaları yüzünden fazla büyük |
| qwen3.5:9b (Alibaba) | 6.6 GB | Bonus "8B sınıfı" (Aday 2 ile aynı aile → adil karşılaştırma) |

- Modeller artık resim de anlıyor; dosyada görüntü kodlayıcısı da var → 4B Q4 ≈ 3.3–3.4 GB (2.5 değil).

## 6. Test yöntemi (adil karşılaştırma)
```
1. ollama run MODEL --verbose
2. "Merhaba"      → ısınma (ölçülmez)
3. /clear         → her soru boş context'le başlasın
4. Soru → ölç (+ ekran görüntüsü: 1. ve 4. soru)
```
| # | Soru | Neyi ölçüyor? |
|---|---|---|
| 1 | Kendini kısaca tanıtır mısın? | Kimlik (fine-tune sonrası değişecek) |
| 2 | Yapay zekâ nedir? Bir lise öğrencisine anlatır gibi açıkla. | Türkçe anlatım |
| 3 | İstanbul'un fethi hangi yıl oldu ve neden önemlidir? | Bilgi doğruluğu |
| 4 | "Kitap", "kitaplık" ve "kitapçı" kelimelerinin farkı nedir? | Türkçe dil bilgisi (ekler) |
| 5 | Türkiye'nin 2050 yılındaki nüfusu tam olarak kaç olacak? | Uydurma testi |

## 7. Deney: GPU'ya kaç blok? (`num_gpu`) — Gemma 3 4B
- Ollama varsayılan olarak modelin sadece **%46'sını** GPU'ya koydu, ekran kartında ~1.5 GB boş kalmasına rağmen (temkinli davranıyor).
- Neden temkinli? Gemma'nın **256 bin token'lık sözlüğü**: embedding = 262.144 × 2560 ≈ **670M parametre (~%17)**; ayrıca her token'da 262 bin puan hesaplanıyor (`lm_head`) + görüntü kodlayıcısı.
- Model 34 decoder bloğu + 1 çıkış katmanı = 35 katman. Ollama modeli **bloklara göre** GPU/CPU arasında böler; CPU'daki bloklar darboğaz.

| `num_gpu` | GPU payı | Yazma hızı (4. soru, boş context) |
|---|---|---|
| Varsayılan (~16) | %46 | ~13–15 tok/s |
| 24 | %60 | 25.3 tok/s |
| 30 | %70 | 33.0 tok/s |
| **35 (hepsi)** | **%100 GPU** | **46.1 tok/s** |

- **Sonuç: Doğru ayarla hız ~3 kat arttı.** Model tamamen GPU'ya sığdı (2.9 GB).
- Sınır: VRAM gerçekten dolarsa Windows taşan kısmı sistem RAM'ine koyar → hız birden düşer. Uzun sohbette (KV cache büyür) tekrar kontrol edilecek.
- Bu ayar kalıcı değil (`/bye` ile sıfırlanır). Kalıcı yapmak için: Turan'ın **Modelfile**'ına `PARAMETER num_gpu 99` ya da API isteğinde `options: {"num_gpu": 99}`.

## 8. Sonuçlar
### Gemma 3 4B (ilk deneme — sorular aynı sohbette, varsayılan ayar)
| Soru | Değerlendirme |
|---|---|
| 1. Kimlik | ✓ "Google tarafından eğitilmiş büyük bir dil modeliyim" |
| 2. Yapay zekâ | ~ Anlaşılır ama Türkçe yer yer tuhaf ("örümcek ağından gibi", "robotik zekâ"); sen/siz karışık |
| 3. İstanbul | ✓ 29 Mayıs 1453, II. Mehmet; açıklama düzgün ("şehir hiyerarşisinden" gibi tuhaflıklar var) |
| 4. Kitap/kitaplık/kitapçı | ~ Kitapçı doğru; kitaplığı "kütüphane" gibi anlattı (asıl anlamı kitap rafı/dolabı) |
| 5. Nüfus | ✓ "Kesin sayı verilemez" dedi; ama verdiği TÜİK rakamları doğrulanmadı (kaynak adıyla uydurma riski) |

### Qwen 3.5 4B (`--think=false`, `num_gpu 99`)
- Varsayılan: %50 GPU, 12.4 tok/s · `num_gpu 99`: **%100 GPU (3.1 GB), ~38–40 tok/s**
- `/clear` mesajla birlikte yapıştırıldığı için çalışmadı → context birikti (23 → 1639 token). **`/clear` tek başına yazılıp Enter'a basılmalı**, "Cleared session context" mesajı beklenmeli.

| Soru | Değerlendirme | Hız |
|---|---|---|
| 1. Kimlik | ~ "Qwen3.5, Alibaba" doğru; ama Türkçe bozuk ("birçok dilliği", "bilimski şemaları") | 39.5 |
| 2. Yapay zekâ | ✗ Anlamsız cümleler ("hiç asla uyanmıyorsun", "barıyor", "ödüyoruz") | 39.1 |
| 3. İstanbul | ✗ **Uydurma:** "25 Mayıs" (doğrusu 29), "Osmanlı'nın kurucusu" (yanlış), "Mehmed Fahrettin Sultan" (uydurma isim) | 38.6 |
| 4. Kitap/kitaplık/kitapçı | ✗ Kitaplığı "koleksiyon" sanıyor; "kitapçı kargo", "mucize, kitabes" gibi anlamsız ifadeler | 38.3 |
| 5. Nüfus | ~ "Kesin sayı verilemez" dedi, 80–90 milyon aralığı verdi; araya alakasız "yapay zekada" girdi | 37.3 |

- **Ders:** Model kendinden emin bir dille **yanlış tarih ve uydurma isim** üretti → **hallucination** örneği. Küçük modellerde Türkçe bilgi güvenilir değil; Turan'ın "bilmiyorum" diyebilmesi bu yüzden önemli.

### Kumru 2B (VNGRS, Türkçe için sıfırdan · `alibayram/kumru`)
- Varsayılan ayarda **%100 GPU** (1.9 GB, context 8192). Ayara gerek yok. Isınma sadece 0.8 sn.

| Soru | Değerlendirme | Hız |
|---|---|---|
| 0. Merhaba | ✗ "Yardımcı olabildiysem ne mutlu" → selamı anlamadı, sohbeti bitirir gibi cevap verdi | 83.8 |
| 1. Kimlik | ✗ **"Ben Zeynep... bir insanım"** → kimliği yok, insan olduğunu uyduruyor | 90.1 |
| 2. Yapay zekâ | ✓ **Türkçesi akıcı ve dil bilgisi düzgün**; içerik yüzeysel ve abartılı ("insanlardan daha iyi kararlar") | 87.3 |
| 3. İstanbul | ~ 1453 doğru, sonrası karmaşık: "1058", "1204 (476)", "Niğbolu sonrası Edirne'yi aldı" (Edirne ~1361'de alındı) | 87.6 |
| 4. Kitap/kitaplık/kitapçı | ✗ Kitaplık ile kitapçıyı aynı şey sanıyor | 86.5 |
| 5. Nüfus | ✗ Uyarı yapmadan tek sayı verdi: "yaklaşık 84 milyon" | 79.0 |

- **Ders:** Türkçe veriyle sıfırdan eğitilmesi **dil bilgisini** düzeltmiş (Qwen'den çok daha doğal Türkçe), ama 2B **bilgi ve muhakeme** için çok küçük. → **Akıcı Türkçe ≠ doğru bilgi.**
- "Ben Zeynep, bir insanım" cevabı, fine-tune ile **kimlik** öğretmenin neden gerekli olduğunu gösteriyor.

### Gemma 4 E2B (Google, en yeni nesil · `gemma4:e2b`, `--think=false`)
- **E2B = Effective 2B (etkin 2 milyar):** Parametrelerin bir kısmı **katman başına arama tablolarında** (Per-Layer Embeddings). Her token'da bu tabloların sadece **bir satırı** okunur (embedding mantığı) → toplam ~5B parametre, hesaplama maliyeti ~2B. Tablolar yavaş belleğe (RAM) konabilir → telefon/laptop için tasarlandı.
- `ollama ps` "**275 MB, 100% GPU**" dedi → yanıltıcı! Kendim kontrol ettim: **~2.6 GB VRAM'de** (hesaplayan kısım) + **~3.1 GB RAM'de** (arama tabloları). Ollama bu yeni yapıyı doğru gösteremiyor. → **Araçların sayısına körü körüne güvenme, doğrula.**
- Yükleme uzun sürdü (parçalar iki farklı belleğe dağıtılıyor). **Ayarsız** hız: ~38.7 tok/s.
- Laptop fanı bağırdı → normal: GPU %100 yükte (85 °C'ye kadar normal). Şarja takılı olsun, havalandırma açık olsun.

| Soru | Değerlendirme | Hız |
|---|---|---|
| 0. Merhaba | ✓ "Yapay zeka olduğum için duygularım yok ama görevimi yerine getirmeye hazırım" | 38.7 |
| 1. Kimlik | ✓ "Google DeepMind, açık ağırlıklı, Gemma 4; metin ve görsel anlayabilirim" → kesin ve doğru | 38.9 |
| 2. Yapay zekâ | ✓ Çok iyi: başlıklı yapı, kedi fotoğrafı/Google Maps/Netflix örnekleri, akıcı Türkçe, seviyeye uygun. Çok uzun (914 token) | 35.4 |
| 3. İstanbul | ✓ 1453 doğru, uydurma tarih/isim yok. Küçük kusurlar: "Osmanlı'nın **doğuşu**" başlığı (kuruluş 1299), II. Mehmet'in adı yok | 36.8 |
| 4. Kitap/kitaplık/kitapçı | ✓ **Tek tam doğru cevap:** kitaplık = mobilya/raf; özet tablo bile yaptı | 40.1 |
| 5. Nüfus | ✓ "Kesin tahmin mümkün değildir" + aralık (85–95 milyon) + TÜİK'e yönlendirme | 39.6 |

- **Sonuç: Kalitede açık ara en iyi model.** Hız Gemma 3'ün ayarlı hızından biraz düşük (~38 vs 46), ama ayarsız en iyi.
- **Fine-tune desteği:** Unsloth'un **"Gemma 4 E2B (Text)"** Colab notebook'u var ✓ → Turan'ın temeli olabilir.
- Cevaplar çok uzun → fine-tune'da "kısa özet + maddeler" formatı bunu da düzeltecek.

### Turkish-Llama 8B (YTÜ COSMOS · `ahmets/ytu_cosmos`, Llama 3 8B tabanlı)
- 4.9 GB dosya, `ollama ps`'te 5.6 GB → 4 GB karta sığmıyor. Llama 3 8B = 32 blok + 1 çıkış = 33 katman, blok başına ~150 MB.

**`num_gpu` deneyi: tatlı nokta ve uçurum**
| `num_gpu` | GPU payı | Hız (tok/s) | Boş VRAM |
|---|---|---|---|
| Varsayılan (~14) | %42 | 6.75 | ~1.3 GB |
| 20 | %60 | 9.79 | 667 MB |
| 24 | %70 | 15.82 | 103 MB |
| **28** | **%81** | **18.46** | 51 MB |
| 33 (hepsi) | "100% GPU" (yanıltıcı) | **7.94** | — |

```
Hız
 18 |              ●  ← tatlı nokta (28)
 15 |          ●
  9 |      ●
  7 |  ●                  ● ← uçurum (33)
    +---------------------------
      14   20   24   28   33    num_gpu
```
- **Varsayılanda neden teorikten (~11) bile yavaş?** CPU'daki bloklar sadece okunmuyor, **çarpmaları da CPU yapıyor** → 8 çekirdek hem okumaya hem hesaplamaya yetişemiyor.
- **24 → 28'de neden düşmedi?** VRAM sadece +52 MB arttı → bloklar **paylaşılan GPU belleğine** (sistem RAM'i) taştı. Ama GPU yavaş bellekten okuyup hesaplasa bile CPU'dan hızlı (hipotez: asıl darboğaz CPU hesaplamasıydı). Tahminim tutmadı → neden tutmadığını düşündüm.
- **33'te neden çöktü?** Taşan ~1.5 GB her token'da **PCIe** üzerinden okunuyor: ekran kartı belleği ~128 GB/sn, PCIe ~12 GB/sn (10 kat yavaş).
- **Ders:** En iyi ayar = ekran kartının **tam dolduğu** nokta. "Daha fazla GPU her zaman daha iyi" değil.

**5 soru (`num_gpu 28`)**
| Soru | Değerlendirme | Hız |
|---|---|---|
| 0. Merhaba (varsayılan) | ✗ **"Benim adım Siri"** (sonraki denemelerde "bir yapay zeka asistanıyım") | 6.75 |
| 1. Kimlik | ✗ "Nereden gelip geldiğimi tam olarak söyleyemem" → kimliği yok | 17.8 |
| 2. Yapay zekâ | ✓ Akıcı, doğal Türkçe ("süslü bir isim ama aslında çok basit"); içerik makul | 17.1 |
| 3. İstanbul | ~ 1453 + II. Mehmed doğru; ama "**16. yüzyılın başlarında Fatih**" (Fatih 1481'de öldü) hatası | 16.8 |
| 4. Kitap/kitaplık/kitapçı | ✓ Kitaplık = kitap rafları doğru | 17.2 |
| 5. Nüfus | ✗ **Hatalı hesap:** "%1,03'ü 5'e bölersek..." gibi anlamsız formüllerle "98.415.000" sonucuna vardı. Matematik görünümlü uydurma | 16.5 |

- **Ders:** 8B + Türkçe eğitim → akıcı Türkçe ve daha iyi dil bilgisi. Ama **matematik kılığında hallucination** üretebiliyor: adım adım "hesaplıyor" gibi görünüp tamamen uyduruyor. Bu, düz bir yanlış sayıdan daha tehlikeli çünkü ikna edici.

### Token üretim hızının formülü
- Model **her token için bütün ağırlıklarını bellekten bir kez okur** → darboğaz hesaplama değil, **bellek hızı.**
- `Maksimum hız ≈ bellek hızı ÷ model boyutu` · GTX 1650 ≈ **128 GB/sn**

| Model | GPU'daki boyut | Teorik | Ölçülen |
|---|---|---|---|
| Kumru 2B | 1.5 GB | ≈ 85 | **83.8** ✓ |
| Gemma 3 4B | 2.9 GB | ≈ 44 | **46.1** ✓ (görüntü kodlayıcısı metin üretiminde okunmuyor) |
| Qwen 3.5 4B | 3.1 GB | ≈ 41 | **~39** ✓ |

- Quantization bu yüzden hızı da artırır: 4 kat küçük model = 4 kat az okunacak veri.

### Birebir alıntılar (terminal çıktılarından)
> Modeller her seferinde farklı cevap verebilir (rastgele örnekleme). Bu alıntılar gerçek test anındaki çıktılardır.

**Kimlik sorusu ("Kendini kısaca tanıtır mısın?" / "Merhaba")**
- **Gemma 3 4B:** "Ben Google tarafından eğitilmiş büyük bir dil modeliyim."
- **Gemma 4 E2B:** "Ben Google DeepMind tarafından geliştirilmiş, açık ağırlıklara sahip bir Büyük Dil Modeli olan Gemma 4'üm."
- **Qwen 3.5 4B:** "Ben Qwen3.5, Alibaba Grubu tarafından geliştirilmiş bir yapay zeka modeli. İtalyanca'dan İngilizce'ye kadar birçok dilliği anlıyorum..."
- **Kumru 2B:** "Tabii! Ben Zeynep. Kendini kısaca tanıtmam gerekirse; meraklı, öğrenmeye açık ve yardımsever bir insanım."
- **Turkish-Llama 8B:** "Merhaba! Benim adım Siri, yapay zekam bir asistanım."

**Hallucination örneği (Qwen 3.5 4B, İstanbul sorusu)**
- "İstanbul'un fethi, **25 Mayıs 1453** tarihinde Osmanlı İmparatorluğu'nun kurucusu **Sultan Mehmed Fahrettin Sultan (Mehmed II)** tarafından gerçekleşmiştir."
- Üç hata tek cümlede: tarih (29 Mayıs), "kurucu" (değil), uydurma isim. Kalın harflerle, kendinden emin.

**Türkçe dil bilgisi (kitap / kitaplık / kitapçı)**
- **Gemma 4 E2B:** "Kitaplık: Kitapların düzenli bir şekilde saklanması ve sergilenmesi için tasarlanmış mobilya veya depolama sistemidir." ✓ (tek tam doğru cevap)
- **Gemma 3 4B:** "Kitaplık: Birçok kitaba sahip, genellikle kamuya açık bir yerdir." ~ (kütüphane ile karıştırdı)
- **Kumru 2B:** "'Kitaplık' veya 'kitapçı' ise bir kitapçının veya kütüphanecinin çalıştığı yeri ifade eder." ✗
- **Qwen 3.5 4B:** "...kitaplarla işlenen kişinin adı da 'kitapçı' olabilir (örneğin: kitapçı kargo)." ✗

**Bozuk Türkçe (Qwen 3.5 4B, yapay zekâ sorusu)**
- "Sen kelimelerle konuşabildiğin, çizimlere bakarken anlıyordun ama hiç asla uyanmıyorsun."

**Uydurma testi (2050 nüfusu)**
- **Gemma 4 E2B:** "Türkiye'nin 2050 yılındaki tam nüfusunu kesin olarak tahmin etmek **mümkün değildir**." ✓
- **Kumru 2B:** "2050 yılında Türkiye'nin nüfusu yaklaşık 84 milyon olarak tahmin edilmektedir." ✗ (uyarı yok)

### Test listesi (5 model)
| # | Model | Boyut | Geliştirici | Temsil ettiği soru |
|---|---|---|---|---|
| 1 | Gemma 3 4B | 3.4 GB | Google | Çok dilli, olgun bir 4B |
| 2 | Qwen 3.5 4B | 3.3 GB | Alibaba | Aynı boyutta başka bir dev |
| 3 | Kumru 2B | 1.5 GB | VNGRS | Küçük ama Türkçeye özel model yeter mi? |
| 4 | Gemma 4 E2B | 4.6 GB | Google | En yeni nesil fark yaratıyor mu? ("E2B" = etkin 2B; dosyada görüntü + ses parçaları) |
| 5 | Turkish-Llama 8B (`ahmets/ytu_cosmos`) | 4.9 GB | YTÜ COSMOS | 8B + Türkçe eğitim; donanım kaldırıyor mu? |
- Kumru ve Cosmos, Ollama'ya resmi ekipler değil kullanıcılar tarafından yüklenmiş ve 1–2 yıllık sürümler.
- Qwen 3.5 9B, Qwen ailesi Türkçede zayıf çıktığı için listeden çıkarıldı.

### Karşılaştırma tablosu
| Model | Boyut | GPU payı | Hız (tok/s) | Türkçe dil bilgisi | Bilgi doğruluğu | Kimlik |
|---|---|---|---|---|---|---|
| **Gemma 3 4B** | 3.4 GB | %46 → %100 (num_gpu 35) | 13 → **46** | ~ Anlaşılır, yer yer tuhaf | ✓ 1453 doğru | ✓ Google modeli |
| Qwen 3.5 4B | 3.3 GB | %50 → %100 (num_gpu 99) | 12 → **~39** | ✗ Sık bozuk cümle | ✗ Tarih ve isim uydurdu | ✓ Qwen |
| Kumru 2B | 1.5 GB | %100 (varsayılan) | **~87** | ✓ Akıcı ve doğal | ✗ Zayıf, karışık | ✗ "Ben Zeynep, insanım" |
| **Gemma 4 E2B** | 4.6 GB | ~2.6 GB VRAM + ~3.1 GB RAM | **~38** (ayarsız) | ✓ Akıcı ve düzgün | ✓ Doğru, uydurma yok | ✓ Gemma 4, Google DeepMind |
| Turkish-Llama 8B | 4.9 GB | %42 → %81 (num_gpu 28) | 6.75 → **18.5** | ✓ Akıcı | ~ Kısmen; hatalı hesap yaptı | ✗ "Benim adım Siri" |

### Sonuç sıralaması
| Sıra | Model | Neden? |
|---|---|---|
| 1 | **Gemma 4 E2B** | Kalitede açık ara en iyi: doğru bilgi, akıcı Türkçe, kitaplığı bilen tek model, uydurmuyor. Ayarsız ~38 tok/s. Unsloth notebook'u var |
| 2 | Gemma 3 4B | En hızlı 4B (46 tok/s); Türkçesi anlaşılır, bilgisi doğru |
| 3 | Turkish-Llama 8B | Türkçesi akıcı ama bu donanımda yavaş (en iyi 18.5) ve nüfus sorusunda hatalı bir hesap yaptı |
| 4 | Kumru 2B | En hızlı (~87) ve Türkçesi doğal, ama bilgi zayıf; "Ben Zeynep, insanım" |
| 5 | Qwen 3.5 4B | Bozuk Türkçe + tarih ve isim uydurdu |

**Karar: Turan'ın temeli → Gemma 4 E2B.**

