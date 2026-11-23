# 03 — Attention

## 1. Neden attention?
- Embedding **bağlamı bilmez**: "yüz" kelimesi "Denizde yüz", "yüz lira" ve "Yüzü güldü" cümlelerinin üçünde de aynı vektörü alır.
- **Attention:** Cümledeki her kelime diğer kelimelere bakar, **kendisiyle ilgili olanlardan bilgi toplar** ve vektörünü buna göre günceller.
- "Kedi sütü içti, çünkü **o** acıkmıştı" → "o", vektörüne "kedi"den bilgi alır.
- Bir kelime **aynı anda birden fazla kelimeye, farklı oranlarda** dikkat edebilir: "Ali topu Ayşe'ye attı, **o** da yakaladı" → "o", hem "Ayşe'ye"ye (kim) hem "yakaladı"ya (ipucu: yakalayan, topu alan kişidir) bakar.
- **Giren ve çıkan şekil aynıdır:** `(batch, seq_len, embedding_dim)`. Sadece vektörlerin içeriği bağlamla zenginleşir.
- Benzetme: Kalabalık bir ortamda sadece seninle ilgili konuşmaya kulak vermek.

## 2. İlgi skoru: Nokta çarpımı (dot product)
- **İki vektörün sayılarını karşılıklı çarp, hepsini topla:** `[1,0,2] · [1,1,2] = 1 + 0 + 4 = 5`
- Büyük pozitif = aynı yön, çok ilgili · 0 civarı = alakasız · negatif = zıt
- Kosinüs benzerliği = nokta çarpımının vektör uzunluklarına bölünmüş hâli. Attention, daha hızlı olduğu için doğrudan nokta çarpımını kullanır.
- Her kelime çifti için hesaplanır → **skor matrisi**, şekli `(seq_len, seq_len)`. Her **satır**, bir kelimenin diğerlerine verdiği skorlardır.
- 1.000 token = 1.000 × 1.000 = 1 milyon skor → attention'ın maliyeti dizi uzunluğunun **karesiyle** artar.

## 3. Softmax
- Ham skorları **toplamı %100 olan yüzdelere** çevirir (= **attention ağırlıkları**).
  1. Her skorun üstelini al: `e^skor` (hepsi pozitif olur)
  2. Toplama böl (toplam 1 olur)
- `[2, 1, 0]` → `[%66,5, %24,5, %9]`. Üstel alma **farkları büyütür** → model en ilgili kelimeye net odaklanır.
- Eşit skorlar → eşit dağılım: `[3, 3, 3]` → her biri %33,3 (eğitimin başında attention genelde böyle dağınıktır).
- Skor matrisinin **her satırına ayrı** uygulanır.

## 4. Query, Key, Value
| Rol | Soru | Kütüphane | YouTube |
|---|---|---|---|
| **Query** | Ne arıyorum? | Aklındaki soru | Arama kutusuna yazdığın "LLM eğitimi" |
| **Key** | Ben neyim? | Kitabın sırtındaki etiket | Videoların başlık ve etiketleri |
| **Value** | Ne veriyorum? | Kitabın içeriği | İzlediğin videonun içeriği |
- **Skor = Query · Key.** Eşleşme yüksekse o kelimenin **Value**'sundan çok bilgi alınır.
- Her kelime üç rolü birden oynar → her kelimenin 3 ayrı vektörü vardır.
- **Neden üç ayrı vektör?** Kelimenin aradığı şey, kendini tanıtma şekli ve verdiği bilgi farklı şeylerdir ("o" bir kişi arar ama kendisi kişi değildir).
- Üretilişleri: `Q = x·W_q`, `K = x·W_k`, `V = x·W_v` → PyTorch'ta üç `nn.Linear`. Rastgele başlar, **eğitimle öğrenilir.**

## 5. Formül: Scaled dot-product attention
```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d ) · V
```
1. `Q · Kᵀ` → skor matrisi (`ᵀ` = transpoz, matris çarpımı doğru yönde olsun diye)
2. `/ √d` → ölçekleme
3. `softmax` → yüzdeler
4. `· V` → **Value'ların ağırlıklı toplamı**: her Value'yu yüzdesiyle çarp, topla
```
0,5 × [2, 0] + 0,5 × [0, 4] = [1, 0] + [0, 2] = [1, 2]     (eşit ağırlık = ortalama)
```
- **√d neden?** Vektör uzadıkça nokta çarpımı büyür → softmax tek kelimeye kilitlenir (`[%100, %0, %0]`) → model zor öğrenir. `√d`'ye bölmek skorları makul tutar (d = 64 → 8'e böl).
- Makale: **"Attention Is All You Need" (2017)**.

## 6. Causal mask (geleceği görmeme kuralı)
- Model **sonraki kelimeyi tahmin ederek** eğitilir; eğitimde cümle tek seferde verilir.
- Kelime ileriye bakabilseydi tahmin etmesi gereken cevabı görürdü → **kopya çekerdi**, gerçek kullanımda başarısız olurdu.
- **Kural:** Her kelime sadece **kendisine ve kendinden öncekilere** bakar.
```
            Ali  topu  Ayşe'ye  attı
Ali       [  ✓   ✗     ✗      ✗ ]
topu      [  ✓   ✓     ✗      ✗ ]     ← topu: Ali + kendisi
Ayşe'ye   [  ✓   ✓     ✓      ✗ ]
attı      [  ✓   ✓     ✓      ✓ ]     ← son kelime: hepsi
```
- **Köşegen her zaman açık:** Kelime kendi bilgisini de kullanır.
- **Nasıl?** Softmax'tan önce kapalı hücrelere **−∞** yazılır → `e^(−∞) = 0` → %0 dikkat.
- **Son kelime** bütün cümleyi gören tek pozisyondur → yeni kelime üretilirken son pozisyonun çıktısı kullanılır (`out[-1]`).
- **GPT / Llama / Qwen = decoder-only (causal)** → metin üretir. **BERT** iki yöne bakar → anlamada iyi, üretemez.

## 7. Multi-head attention
- Tek attention her kelime için tek bir dikkat dağılımı üretir. Ama dilde **aynı anda** birçok ilişki vardır: özne–fiil, nesne–fiil, sıfat–isim, zaman...
- **Birden fazla head paralel çalışır.** Her birinin kendi `W_q`, `W_k`, `W_v`'si vardır, yani her biri farklı bir ilişki türüne odaklanır (uzmanlar heyeti).
- Hangi head'in neye bakacağını kimse belirlemez; **eğitimle kendiliğinden** uzmanlaşırlar.
- **Boyut paylaşılır:** head başına boyut = embedding_dim ÷ head sayısı (256 / 8 = 32). Maliyet katlanmaz.
- Sonuçlar yan yana eklenir (**concat**) → tekrar embedding_dim → **W_o** ile karıştırılır.
- GPT-2: 768 = 12 × 64 · Llama 3 8B: 4096 = 32 × 128

## 8. Kendi denemem (`03_attention.ipynb`)
Cümle: `Ali [1,0]`, `topu [0,1]`, `attı [1,1]` → `x.shape = (3, 2)` (3 kelime × 2 sayı)

**Adım 1: Q/K/V olmadan sade attention (elle doğruladım)**
| Hücre | Kod | Sonuç |
|---|---|---|
| Transpoz | `x.T` | `(3, 2)` → `(2, 3)`: kelimeler satırdan sütuna geçer |
| Skorlar | `scores = x @ x.T` | `[[1,0,1],[0,1,1],[1,1,2]]`, `(3, 3)`. `scores[i, j] = kelime_i · kelime_j` |
| Softmax | `F.softmax(scores, dim=-1)` | attı: `[%21, %21, %58]`. `dim=-1` = **her satır kendi içinde** |
| Ağırlıklı toplam | `weights @ x` | attı: `0.21·[1,0] + 0.21·[0,1] + 0.58·[1,1] = [0.79, 0.79]`. Şekil `(3, 2)` = girdiyle aynı |
- Skoru 0 olan bile %16 alır (`e^0 = 1`). Tam %0 için skor **−∞** olmalı.

**Gördüğüm iki sorun → Q/K/V'nin sebebi:**
1. **Köşegen büyük:** Kelimenin kendisiyle çarpımı = sayıların kareleri toplamı → her kelime en çok **kendine** bakar (attı → kendine %58).
2. **Simetrik tablo:** `a · b = b · a` → "o → Ayşe" ile "Ayşe → o" zorla eşit olur. Gerçek dilde ilişkiler simetrik değildir.

**Adım 2: `nn.Linear` = dönüştürme makinesi**
- `nn.Linear(2, 2, bias=False)`: "**2 sayı alır, 2 sayı verir.**" İçinde öğrenilen bir matris (W) var.
- Kelimeleri **tek tek** işler → **kelime sayısını bilmez**, ne verirsen onu işler. `(3, 2)` girerse `(3, 2)`, 5 kelime girerse `(5, 2)` çıkar.
- Çıkan her sayı = kelime vektörü · W'nin bir satırı. W'nin satır sayısı = çıkış uzunluğu, her satırın uzunluğu = giriş uzunluğu.
- Üç ayrı makine: `W_q`, `W_k`, `W_v` → aynı kelimeden üç farklı vektör.

**Adım 3: Gerçek attention**
- `scores = q @ k.T` → simetri bozuldu (Ali→attı = −0.32, attı→Ali = −0.09); köşegen artık zorunlu olarak büyük değil.
- `F.softmax(scores / d ** 0.5, dim=-1)` → eğitilmemiş makinelerde yüzdeler **~%33** (dağınık), çıktılar Value'ların ortalamasına yakın.

**Adım 4: Causal mask**
```python
mask = torch.tril(torch.ones(3, 3))                          # alt üçgen: 1 = bakabilir, 0 = gelecek
scores = scores.masked_fill(mask == 0, float("-inf"))        # gelecek → −∞ → softmax'ta %0
```
- Ali `[1, 0, 0]` (%100 kendisi) · topu `[0.52, 0.48, 0]` (attı'nın payı Ali ile kendisine dağıldı) · attı değişmedi (zaten her şeyi görüyordu).

**Adım 5: `CausalSelfAttention` sınıfı**
- `__init__`: makineler bir kez kurulur · `forward`: veri verildiğinde hesap yapılır · `nn.Module`: PyTorch parametreleri takip eder · `super().__init__()` zorunlu.
- `k.transpose(-2, -1)`: sadece **son iki** boyutu çevirir → batch olsa da doğru çalışır (`.T` bütün boyutları ters çevirirdi).
- **Girdi şekli = çıktı şekli:** `(3, 2)` → `(3, 2)`; batch `(4, 5, 2)` → `(4, 5, 2)`. İçeride `scores` = `(4, 5, 5)`: **her cümle için ayrı bir 5×5 tablo.** Cümleler birbirini görmez.

**Adım 6: Multi-head**
- Tek head **küçültür:** `CausalSelfAttention(8, head_dim=2)` → `(2, 5, 8)` girer, `(2, 5, 2)` çıkar.
- 4 head'i **uç uca ekle** (`torch.cat(..., dim=-1)`; toplama değil!) → `(2, 5, 8)`. Her head'in bilgisi kendi yerinde durur.
- `W_o` (8 alır, 8 verir) head'lerin bilgisini harmanlar → son çıktı `(2, 5, 8)`.
- `head_dim = embedding_dim // num_heads` (`//` = tam sayı bölmesi) · `nn.ModuleList` = PyTorch'un takip ettiği liste.

Sonuç: `MultiHeadAttention(embedding_dim, num_heads)` sınıfı, transformer bloğunda kullanılmaya hazır.

## 9. Özet
Embedding her kelimeye bağlamdan bağımsız bir vektör veriyor; attention, her kelimenin cümledeki diğer kelimelerden bilgi toplamasını sağlıyor. Her kelimeden öğrenilen matrislerle üç vektör üretiliyor: Query ne aradığını, Key kendini nasıl tanıttığını, Value ne bilgi verdiğini temsil ediyor. Query ile Key'lerin nokta çarpımı skorları veriyor; bunları √d'ye bölüp softmax ile yüzdelere çeviriyoruz ve Value'ların ağırlıklı toplamını alıyoruz. GPT gibi modellerde causal mask ile her kelime sadece geçmişi görüyor, yoksa sonraki kelimeyi tahmin ederken kopya çekerdi. Multi-head attention ile birden fazla head farklı ilişki türlerini paralel öğreniyor. Maliyeti dizi uzunluğunun karesiyle arttığı için context length pahalı.
