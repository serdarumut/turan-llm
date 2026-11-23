# 02 — Embedding ve Pozisyon

## 1. Sorun
- Token ID'leri (4821, 4822...) **anlamsız etiketlerdir**, okul numarası gibi.
- Doğrudan modele verirsek model sayısal yakınlıktan yanlış ilişki kurar: 4821 = kapı, 4822 = bardak, ama "yan yana" oldukları için benzer sanılabilir.
- **İhtiyaç:** Anlamı benzer olan token'ların birbirine yakın durduğu bir temsil.

## 2. Çözüm: Her token'a bir vektör
- **Embedding:** Her token, sabit uzunlukta bir sayı listesiyle (vektör) temsil edilir. Bu bir olasılık değil; token'ın **özelliklerini** taşıyan bir sayı listesidir.
- Vektörler uzayda birer nokta gibidir: kedi ile köpek yakın, uçak uzak.
```
kedi  → [ 0.8, 0.1, 0.9]
köpek → [ 0.7, 0.2, 0.8]   ← kediye yakın
uçak  → [-0.5, 0.9, 0.1]   ← uzak
```

## 3. Mekanizma: Bir arama tablosu
- `nn.Embedding(vocab_size, embedding_dim)` → şekli **(vocab_size, embedding_dim)** olan bir tablo.
- **Satır = token, sütun = vektörün sayıları.** Token ID'si = satır numarası. Embedding'in yaptığı tek şey **satır çekmek**.
- **Parametre sayısı = vocab_size × embedding_dim**
  - 8.000 × 128 = 1.024.000 · GPT-2: 50.000 × 768 = 38,4 milyon

## 4. Sayılar öğreniliyor
- Tablo eğitim başında **rastgele**. Kedi ile köpek arasında hiçbir ilişki yok.
- Model "sonraki kelimeyi tahmin et" göreviyle eğitilirken her hatadan sonra tablo biraz düzeltilir (backpropagation).
- Benzer bağlamlarda geçen kelimelerin ("Bahçede ___ koşuyordu" → kedi/köpek) vektörleri **zamanla birbirine yaklaşır.** Buna dağılımsal anlam denir: kelime, yanında durduğu kelimelerle tanınır.
- **Word2Vec (2013):** kral − erkek + kadın ≈ kraliçe. Kavramlar uzayda **yön** olarak öğrenilir.
- PyTorch'ta `requires_grad=True` = "bu sayılar için türev hesapla, eğitimde güncellenecekler".
- `grad_fn=...` = PyTorch tensörün nereden geldiğini not eder; backpropagation bu kaydı takip ederek geriye gider.

## 5. Benzerlik: Kosinüs benzerliği
- İki vektör aynı yönü mü gösteriyor? **1 = aynı yön, 0 = alakasız, −1 = zıt.**
- Bir vektör kendisiyle her zaman **1.0**.
- Eğitilmemiş tabloda benzerlik değeri **tesadüftür, anlam taşımaz.** Denememde kedi–köpek: −0.29.

## 6. Parametre bütçesi: Ezber mi, düşünme mi?
- Model = **sözlük kısmı** (embedding, kelime ezberi) + **düşünme kısmı** (transformer katmanları: attention + MLP).
- GPT-2 küçük: toplam 124M → sözlük ~38M (**%31**), düşünme ~85M (%69).
- Sabit bütçede sözlük büyürse düşünmeye az kalır. Örnek: 10M bütçe, 30.000 × 256 = 7,68M → düşünmeye sadece 2,32M kalır (**kötü tasarım**).
- **Model küçüldükçe sözlük boyutu seçimi daha kritik** hâle gelir. Llama 3 8B'de 128 bin sözlük bile ~%6–7.
- **Scaling Laws (OpenAI, 2020):** Model başarısını en iyi **embedding hariç** parametre sayısı tahmin ediyor. Sözlüğü "yeterince" büyük tut, gerisini düşünmeye ver.
- Tipik oran: küçük modellerde %10–30 (SmolLM-135M ~%21), büyük modellerde <%10.
- **Weight tying:** Çıkıştaki `lm_head` katmanı da (vocab_size × dim) boyutunda. Embedding ile **aynı tabloyu paylaşırsa** sözlük maliyeti iki kat yerine bir kat olur (GPT-2, SmolLM).
- Oranlar formülle değil, **deneyerek ve başarılı benzer modelleri kopyalayarak** belirlenir.

## 7. Sıra sorunu → Positional encoding
- Embedding her token'a, nerede durduğundan bağımsız olarak **hep aynı vektörü** verir. Attention da kendi başına sırayı görmez.
- **Torba benzetmesi:** "Bugün değil, yarın gel" ile "Yarın değil, bugün gel" aynı kelime kartları. Torbaya atılırsa hangisi hangisi bilinmez. Positional encoding her karta bir **sıra numarası** yazmaktır.

| Yöntem | Kim kullanıyor | Nasıl? | Artı / Eksi |
|---|---|---|---|
| **Öğrenilen pozisyon tablosu** | GPT-2 | İkinci bir `nn.Embedding(context_length, dim)`; token vektörüne **eklenir** | Basit / context_length'ten uzun metni işleyemez |
| **Sinüs-kosinüs** | Orijinal Transformer (2017) | Formülle hesaplanır, farklı hızlarda salınan dalgalar (saat kolları gibi) | Parametre yok / uzun metinde zayıf genelleme |
| **RoPE** | Llama, Qwen, Gemma | Vektörü pozisyonu kadar **döndürür**; attention'da uygulanır | İlişkiyi **göreli mesafe** belirler (5–7 = 105–107), uzun metne iyi genelleşir |

```
girdi = token_embedding[token_id] + pozisyon_embedding[sıra]     (GPT-2 yöntemi)
```
- Pozisyonlar **0'dan** başlar: `torch.arange(3)` → `[0, 1, 2]`.
- Eklenen şey bir sayı değil, pozisyon tablosundaki **o satırın vektörüdür.**

## 8. Şekiller (shape)
```
Token ID'leri:   (batch, seq_len)                 ör. (2, 3)
     ↓ embedding (satır çek)
Vektörler:       (batch, seq_len, embedding_dim)  ör. (2, 3, 4)
     ↓ + pozisyon (şekil değişmez)
Modele giren:    (batch, seq_len, embedding_dim)
```
- **batch:** aynı anda işlenen cümle sayısı. GPU'nun binlerce çekirdeği boş kalmasın diye cümleler paket hâlinde verilir (otobüsü dolu kaldırmak gibi).
- **seq_len:** cümledeki token sayısı.
- **embedding_dim:** her token'ın vektör uzunluğu.

## 9. Kendi denemem (`02_embedding.ipynb`)
| Hücre | Ne gördüm? |
|---|---|
| `nn.Embedding(10, 4)` | Şekil `(10, 4)`, rastgele sayılar, `requires_grad=True` |
| `embedding([3, 7, 3])` | Şekil `(3, 4)`; **1. ve 3. satır birebir aynı** (aynı token = aynı vektör, sıra bilgisi yok) |
| Batch `(2, 3)` | Çıktı `(2, 3, 4)` |
| + pozisyon tablosu | 1. ve 3. satır artık **farklı**: aynı kedi, farklı pozisyon |
| Kosinüs | kedi–kedi = 1.0; kedi–köpek = −0.29 (eğitilmemiş, anlamsız) |

## 10. Özet
Token ID'leri anlamsız etiketler olduğu için onları doğrudan modele vermiyoruz. Embedding katmanı, vocab_size × embedding_dim boyutunda bir tablo; her token ID'si bir satır numarası ve o satır token'ın vektörü oluyor. Tablo rastgele başlıyor; eğitimde benzer bağlamlarda geçen kelimelerin vektörleri birbirine yaklaşıyor ve benzerliği kosinüs benzerliğiyle ölçüyoruz. Embedding sırayı bilmediği için positional encoding ekliyoruz: GPT-2 öğrenilen bir pozisyon tablosu kullanıyor, modern modeller ise RoPE ile vektörleri döndürerek göreli mesafeyi kodluyor. Küçük modellerde embedding parametre bütçesinin büyük bir kısmını yiyebildiği için sözlük boyutu ve weight tying önemli.
