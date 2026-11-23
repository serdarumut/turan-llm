# 01 — Tokenizer

## 1. Neden var?
- Sinir ağı **sadece sayılarla** çalışır (içinde yaptığı her şey matris çarpımı).
- Tokenizer, **metin ↔ sayı tercümanıdır.**
  - **encode (tokenize):** metin → token'lar → token ID'leri → modele girer
  - **decode (detokenize):** modelin ürettiği ID'ler → metin
- **Token:** Modelin gördüğü en küçük birim (kelime, kelime parçası, harf ya da bayt).

## 2. Metni bölmenin 3 yolu

| Yaklaşım | Artı | Eksi |
|---|---|---|
| **Kelime** | Basit, her token anlamlı | Sözlükte olmayan kelime → `<unk>` (bilgi kaybı). Sözlük devasa olur. Türkçe gibi eklemeli dillerde felaket (ev, evler, evlerimiz... hepsi ayrı kayıt) |
| **Karakter** | Sözlük minik, `<unk>` yok | Diziler çok uzun → context'e az metin sığar, model yavaşlar |
| **Alt kelime (subword)** | Sık kelime = tek token, nadir kelime = parçalar. `<unk>` yok, sözlük makul | — **Bugünkü bütün LLM'ler bunu kullanıyor** |

> **Dizi uzunluğu neden önemli?** Context length sınırlı. Ayrıca attention'ın maliyeti dizi uzunluğunun **karesiyle** artar: dizi 2 kat uzarsa attention hesabı 4 kat artar.

## 3. BPE (Byte Pair Encoding): GPT, Llama ve Qwen'in kullandığı yöntem
**Mantık:** En sık yan yana gelen iki parçayı bul → tek parçada birleştir → istenen sözlük boyutuna kadar tekrarla.

```
ev ev evler evler evde
1. tur: e+v → "ev"     (5 kez, en sık)
2. tur: " "+ev → " ev" (4 kez)
3. tur: " ev"+l → " evl" ...
```

- **Her birleştirme:** Sözlüğe +1 token ekler ve metni kısaltır.
- **Eğitimin çıktısı iki defterdir:**
  - `merges` (**tarif defteri**): hangi ikili hangi numaraya dönüştü, öğrenilme sırasıyla → **encode** için
  - `vocab` (**sözlük**): hangi numara hangi baytlara karşılık geliyor → **decode** için
- Algoritma dilbilgisini bilmez; ama sık geçen şeyler dilin gerçek parçaları (kök, ek, yaygın kelime) olduğu için onları **veriden kendisi bulur.**
- Tokenizer eğitimi **bir kez** yapılır (büyük metinle). Sonra her mesajda sadece encode/decode kullanılır.

### Byte-level BPE
- Başlangıç sözlüğü = **256 bayt** (0–255). Dünyadaki her metin bu baytlarla yazılabilir.
- Bu yüzden **`<unk>` asla gerekmez.** Bilinmeyen bir kelime, sözlükteki parçalarla kurulur; en kötü ihtimalle tek tek baytlara bölünür.
- UTF-8'de İngilizce harfler 1 bayt, **Türkçe harfler (ç ğ ı ö ş ü) 2 bayt**, emojiler 4 bayt.

### Diğer algoritmalar
- **WordPiece** (BERT): sıklık yerine olasılık artışına bakar.
- **Unigram** (T5, Gemma): büyük sözlükle başlar, işe yaramayanları atar.
- **SentencePiece:** algoritma değil **kütüphane** (içinde BPE ve Unigram var); boşluğu `▁` olarak işler.

## 4. Sözlük boyutu (vocab size)
- **Sözlük büyüdükçe kelimeler DAHA AZ parçaya bölünür** (elinde daha çok hazır parça var). Küçük sözlük = daha çok parça, ama byte-level'da **yine `<unk>` yok.**
- **Bedel:** `embedding parametresi = vocab_size × embedding_dim`
  - Örnek (dim = 128): 100 bin sözlük → 12,8M parametre; 8 bin sözlük → 1M parametre
- Az veride büyük sözlüğün token'larının çoğu nadir görülür ve **iyi öğrenilmez.**
- **Kural:** Küçük model + az veri → küçük sözlük (4–16 bin). Büyük model + çok veri + çok dil → büyük sözlük (100 bin+).
- Gerçek modeller: GPT-2 50.257 · Llama 2 32.000 · Llama 3 128.256 · Qwen ~151 bin · Gemma 256.000

## 5. Özel token'lar
| Token | Görevi |
|---|---|
| `<bos>` / `<eos>` | Metnin başı / sonu. **Model `<eos>` üretince durur** |
| `<pad>` | Batch'teki kısa cümleleri doldurmak için |
| `<unk>` | Bilinmeyen kelime (byte-level'da gerekmez) |
| `<\|im_start\|>`, `<\|im_end\|>` | **Sohbet şablonu:** kim konuşuyor (user / assistant). **Fine-tuning'de veri bu şablona sokulmalı**, yoksa model saçmalar ya da susmayı bilmez |

## 6. Hazır model mi, sıfırdan mı?
- **Hazır model (fine-tuning):** Modelin **kendi tokenizer'ı zorunlu.** Model eğitim boyunca "4821 = kedi" ilişkisini öğrendi; başka bir tokenizer verirsek ID'lerin anlamı kayar ve model anlamsız sayılar görür.
- **Fine-tuning:** Önceden eğitilmiş hazır bir modeli kendi verimizle biraz daha eğitmek. Sıfırdan model yapmak değil; dil bilen bir modele yeni beceri ya da üslup kazandırmak.
- **Sıfırdan eğitim:** Tokenizer'ı biz seçeriz → byte-level BPE, modelin göreceği türden veriyle (Türkçe) eğitilmiş, model ve veri boyutuna uygun sözlük boyutu, gerekli özel token'lar.

## 7. Detokenize
- `vocab`'tan her ID'nin baytlarını bul → **arada hiçbir şey olmadan** yan yana koy → UTF-8 ile metne çevir.
- **Boşluk token'ın içinde saklanır** (`" ev"`, `" evler"`). Araya boşluk eklemeyiz, yoksa `"dog ."` gibi bozulmalar olur.
- Bu yüzden `"evler"` ile `" evler"` **farklı token'lardır** (GPT'de de `"Merhaba"` ≠ `" Merhaba"`).
- **Round-trip testi:** `decode(encode(metin)) == metin` → her zaman `True` olmalı (kayıpsız dönüş).
- **Streaming'de Türkçe karakter sorunu:** `ü` = 2 bayt (`195, 188`). Sadece ilk bayt gelirse `decode([195])` → `�`. Web arayüzünde cevabı akıtırken ikinci bayt gelene kadar beklemek gerekir.

## 8. Tokenizer kalitesi nasıl ölçülür?
| Ölçüt | İyisi |
|---|---|
| **Fertility** (token sayısı ÷ kelime sayısı) | Düşük (Türkçe için ~1.5–2 iyi) |
| **Sıkıştırma** (karakter sayısı ÷ token sayısı) | Yüksek |
| **`<unk>` oranı** | 0 |
| **Round-trip** | Her zaman True |
| **Morfolojik uyum** | Bölünmeler kök/ek sınırına denk gelsin: `ev+ler+imiz+den` ✓, `e+vl+er+im+iz+den` ✗ |

- **Az token daha iyidir**, çünkü context'e daha çok metin sığar ve model daha hızlı çalışır. Ama bölünme noktaları da anlamlı olmalı.
- Kursun hocası Ali Bayram'ın Türkçe tokenizer kalitesi ile model başarısı (MMLU) ilişkisi üzerine çalışması var.

## 9. Kendi yazdığım BPE (`01_tokenizer.ipynb`)
| Fonksiyon | Ne yapıyor? |
|---|---|
| `get_stats(ids)` | `zip(ids, ids[1:])` ile bütün yan yana ikilileri sayar |
| `merge(ids, pair, new_id)` | İkiliyi yeni token'la değiştirir (`while` + `i += 2` ile ikinci elemanı atlar) |
| Eğitim döngüsü | `max` ile **en sık** ikiliyi seçer, birleştirir, `merges` ve `vocab`'a yazar. En sık ikili 2'den az geçiyorsa durur (bir kez geçen şey kalıp değil, tesadüftür) |
| `encode(text)` | `min` ile **en erken öğrenilmiş** ikiliyi seçer (birleştirmeler öğrenildikleri sırayla uygulanmalı, çünkü 257, 256'nın üstüne kurulu) |
| `decode(ids)` | `b"".join(vocab[i] ...)` + `.decode("utf-8", errors="replace")` |

**Deneyden çıkan dersler:**
- 29 token → 14 token (metin yarıya indi, sözlük 256 → 261).
- `ü` birleşmedi, çünkü metinde sadece 1 kez geçiyordu.
- `ler` ayrı bir ek olarak değil, ` evler` kelimesinin parçası olarak öğrenildi, çünkü metinde sadece `ev`'in arkasında geçiyordu. → **Tokenizer'ın kalitesi verinin büyüklüğüne ve çeşitliliğine bağlıdır.**

**Gerçek tokenizer'larda ek olarak:** büyük veri, ön bölme (pre-tokenization: önce kelime/noktalama ayrımı), özel token'lar, hız. Pratikte Hugging Face `tokenizers` gibi kütüphaneler kullanılıyor: aynı BPE algoritması, Rust ile yazıldığı için çok daha hızlı.

## 10. Özet
Sinir ağları sayılarla çalıştığı için tokenizer metni token'lara böler ve her token'a bir ID verir. Kelime bazlı bölme bilinmeyen kelime ve dev sözlük sorunu yaratıyor; karakter bazlı bölme ise dizileri çok uzatıyor. Modern LLM'ler BPE kullanıyor: en sık yan yana gelen parçalar birleştirilerek sözlük öğreniliyor. Byte-level BPE'de başlangıç 256 bayt olduğu için hiçbir metin bilinmeyen kalmıyor. Hazır bir modelde o modelin tokenizer'ı zorunlu. Sıfırdan eğitimde ise sözlük boyutunu model ve veri büyüklüğüne göre seçiyoruz, çünkü embedding parametreleri vocab_size × embedding_dim kadar. Türkçe için fertility'ye ve bölünmelerin kök/ek sınırına uymasına bakıyoruz. Detokenize bu işlemin tersi; önemli olan kayıpsız dönüş. Ben de algoritmayı anlamak için BPE'yi sıfırdan yazdım.
