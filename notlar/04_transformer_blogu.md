# 04 — Transformer Bloğu (LayerNorm, MLP, Residual)

## 1. LayerNorm (katman normalizasyonu)

### Sorun
- Model onlarca katmandan oluşur; her katman vektörü matris çarpımından geçirir.
- Her geçişte sayılar biraz büyür ya da küçülürse bu **birikir**: `[1, 3] → [40, 120] → [9000, 27000]`.
- Sayılar patlarsa ya da sıfıra yaklaşırsa **eğitim bozulur.** (Ses sistemi benzetmesi: her amfi sesi biraz yükseltirse en sondaki hoparlör patlar.)

### Çözüm: Her token'ın vektörünü standart ölçeğe getir
1. **Ortalamayı çıkar** → sayılar 0'ın etrafında toplanır
2. **Standart sapmaya böl** → yayılım 1 olur
3. **γ (gamma) ile çarp** → ölçekle (öğrenilir, başlangıç 1)
4. **β (beta) ekle** → kaydır (öğrenilir, başlangıç 0)

```
LayerNorm(x) = γ × (x − ortalama) / √(varyans + ε) + β
```

- **Neden γ ve β?** Her şeyi zorla "ortalama 0, yayılım 1" yapmak fazla katı olabilir. Model gerekirse ölçeği kendisi ayarlar.
- **Fabrika ayarı:** γ = 1, β = 0 → başta hiçbir şey değiştirmezler.
- **ε (epsilon):** Std 0 çıkarsa sıfıra bölme olmasın diye eklenen minik sayı (`0.00001`). Bu yüzden PyTorch çıktısı `-1.` yerine `-1.0000` (aslında `-0.99999...`).
- Normalize edilen değerler her zaman -1 ile 1 arasında **olmaz.** Garanti olan: ortalama 0, std 1. Çoğu değer -2 ile 2 arasına düşer.
- Büyük-küçük sırası **korunur**: en büyük sayı yine en büyük, en küçük yine en küçük.

### Standart sapma nasıl hesaplanır?
"Sayılar ortalamadan ortalama ne kadar uzakta?"

```
[1, 3, 5, 7]
Ortalama:   4
1. Farklar:  -3, -1, 1, 3
2. Kareler:   9,  1, 1, 9
3. Varyans:  (9+1+1+9) / 4 = 5
4. Std:      √5 ≈ 2.24
```
- **Neden kare?** Farkları doğrudan toplarsak eksiler ile artılar birbirini götürür, sonuç hep 0 çıkar.
- İki sayılık vektörde std, ortalamaya olan uzaklığa eşit (`[2, 6]` → std 2). Bu yüzden iki sayılık her vektör `[-1, 1]` çıkar.

### Elle hesap örnekleri
| Vektör | Ortalama | Std | Normalize | γ=2, β=1 ile |
|---|---|---|---|---|
| `[1, 3]` | 2 | 1 | `[-1, 1]` | |
| `[100, 300]` | 200 | 100 | `[-1, 1]` (aynı!) | |
| `[2, 6]` | 4 | 2 | `[-1, 1]` | `[-1, 3]` |
| `[0, 0, 6, 6]` | 3 | 3 | `[-1, -1, 1, 1]` | |

### Şekil ve parametreler
- LayerNorm **şekli değiştirmez**: `(2, 5, 8) → (2, 5, 8)`. Sadece değerler değişir.
- **Her token ayrı ayrı** normalize edilir: `(2, 5, 8)` → 2 × 5 = **10 ayrı vektör.**
- `nn.LayerNorm(dim)` → γ ve β'nın şekli `(dim,)`: her sütuna bir γ, bir β.
- `(256,)` = 256 sayılık tek boyutlu liste. `(256, 0)` ise içi boş tablo olurdu. Parantezdeki sayı adedi = boyut sayısı.
- Parametre sayısı = **2 × dim** → modeldeki en ucuz katman (dim 256 → 512 parametre).

### Modelde nerede?
- **Pre-Norm** (GPT-2, Llama, Qwen): LayerNorm, attention ve MLP'den **önce** uygulanır. Her ağır işlem dengeli girdi alır, eğitim daha kararlı olur.
- **Post-Norm**: 2017 orijinal Transformer, sonra uygular. Artık standart değil.
- **RMSNorm** (Llama, Qwen, Gemma): Ortalamayı çıkarma adımını atlar, sadece ölçeği düzeltir. Daha hızlı, sonuç neredeyse aynı.

### Kendi denemem (`04_transformer_block.ipynb`)
| Hücre | Ne gördüm? |
|---|---|
| `x.mean()`, `x.std(unbiased=False)` | `[2, 6]` → ortalama 4, std 2, normalize `[-1, 1]` (elle hesabımla aynı). `unbiased=False` = normal formül (PyTorch varsayılanı n−1 kullanır) |
| `nn.LayerNorm(2)` | Çıktı `[-1, 1]`; `weight` (γ) = `[1, 1]`, `bias` (β) = `[0, 0]`, ikisi de `requires_grad=True` → öğrenilir |
| `nn.LayerNorm(256)` | γ ve β şekli `torch.Size([256])` |
| `(2, 5, 8)` batch × 100 | Şekil aynı kaldı. `-230` → `-1.75`, `128` → `1.58`: yüzlük sayılar -2..2 arasına indi, sıralama korundu |

### Özet
Derin ağlarda katmanlar boyunca sayılar büyüyüp küçülerek eğitimi bozabilir. LayerNorm her token'ın vektörünü ortalaması 0, standart sapması 1 olacak şekilde normalize eder; sonra öğrenilen γ ve β ile model gerekirse ölçeği ayarlar. Şekli değiştirmez ve parametresi sadece 2 × dim kadardır. Modern modeller Pre-Norm kullanır, yani attention ve MLP'den önce normalize eder. Llama ve Qwen daha hafif bir versiyon olan RMSNorm kullanır.

---

## 2. MLP (Feed-Forward)

### Neden var? Toplantı benzetmesi
| | Attention | MLP |
|---|---|---|
| Benzetme | Toplantı: herkes bilgi alışverişi yapar | Herkes kendi masasına dönüp düşünür |
| Ne yapar? | Token'lar **birbirine bakar** ("kim kiminle ilgili?") | Her token **kendi vektörünü tek başına** işler ("bu bilgiyle ne yapmalıyım?") |
| Örnek | "Ali topu attı. **O** mutluydu." → "O" = Ali (cümleye bakmak gerekir) | "Türkiye'nin başkenti" → "Ankara" bilgisini ekler |

- Araştırmalar modelin **bilgilerinin** (olgular) büyük kısmının MLP'de saklandığını gösteriyor. Attention soruyu anlar, MLP cevabı hafızasından çıkarır.
- Bir bloğun parametrelerinin yaklaşık **2/3'ü MLP'de** → "düşünme kısmının" en büyük parçası.

### Yapı: Genişlet → İşle → Daralt
```
token (dim)
   ↓ nn.Linear(dim, 4·dim)   GENİŞLET
   ↓ GELU                    İŞLE (şekil aynı)
   ↓ nn.Linear(4·dim, dim)   DARALT
çıktı (dim)                  ← giren ile çıkan şekil AYNI
```
- **Neden 4 kat?** Düşünmek için geniş bir müsvedde alanı. Her genişletilmiş sayı bir **dedektör** gibi ("şehir adı mı?", "geçmiş zaman mı?"). Daha çok dedektör = daha çok özellik. 4 kat formülden değil, 2017'de denenip iyi çalıştığı için standart.
- Örnek (dim = 256): `nn.Linear(256, 1024)` + `nn.Linear(1024, 256)` → ≈ **524 bin** parametre. Aynı bloğun attention'ı (4 × 256 × 256) ≈ **262 bin**.

### Aktivasyon fonksiyonu (GELU)
- **Neden gerekli?** `nn.Linear` sadece çarpma + toplama yapar. İki çarpma arka arkaya = tek çarpma (`y = 2x`, `z = 3y` → `z = 6x`). Arada **çarpma olmayan** bir işlem olmazsa genişletmenin ve katman eklemenin anlamı kalmaz.
- **ReLU:** Negatifse 0, pozitifse aynen. Dedektör benzetmesi: negatif = "özellik yok", dedektör susar.
- **GELU:** ReLU'nun yumuşak hâli (GPT-2, Gemma). Sadece **0 civarında** farklı; büyük pozitif ≈ aynen, büyük negatif ≈ 0. Yumuşak geçiş eğitimi düzgünleştirir.
- **SwiGLU:** Llama ve Qwen'in kullandığı varyant.

| Girdi | -3 | 2 | -1 | 5 |
|---|---|---|---|---|
| ReLU | 0 | 2 | 0 | 5 |
| GELU | -0.004 | 1.95 | -0.16 | 5.00 |

### Şekiller Sık yaptığım hata
- `nn.Linear(8, 32)` **makinenin ayarı**dır ("8 al, 32 ver"); çıktının şekli değil!
- Makine **sadece son boyutu** değiştirir; öndeki boyutlara dokunmaz.
```
Tek token:  (8,)      → (32,)      → (32,)      → (8,)
Batch:      (2, 5, 8) → (2, 5, 32) → (2, 5, 32) → (2, 5, 8)
```
- Batch'te 2 × 5 = 10 token'ın her birine **ayrı ayrı** uygulanır; token'lar birbirine bakmaz.
- **Bloktaki her parça şekli korur** (LayerNorm, attention, MLP) → parçaları istediğimiz gibi arka arkaya dizebiliriz.

### Bilimsel gösterim
- `e-03` → virgülü 3 basamak sola kaydır: `-4.0499e-03 = -0.0040499`
- `e+00` → kaydırma yok: `1.9545e+00 = 1.9545`

### Kendi denemem (`04_transformer_block.ipynb`)
| Hücre | Ne gördüm? |
|---|---|
| `torch.relu` / `F.gelu` | `[0, 2, 0, 5]` / `[-0.004, 1.95, -0.16, 5.00]` |
| Tek token: genislet → GELU → daralt | `32 → 32 → 8` |
| Batch `(2, 5, 8)` | `(2, 5, 32)` → `(2, 5, 8)` |
| `MLP` sınıfı | `__init__`: makineleri bir kez kur · `forward`: veriyi geçir · çıktı `(2, 5, 8)` |

```python
class MLP(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.genislet = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.daralt = nn.Linear(4 * embedding_dim, embedding_dim)

    def forward(self, x):
        h = self.genislet(x)
        h = F.gelu(h)
        return self.daralt(h)
```

### Özet
Attention token'ların birbirinden bilgi toplamasını sağlarken, MLP her token'ı tek başına işler; modelin olgusal bilgilerinin büyük kısmı burada saklanır. Yapısı iki Linear katman ve arada bir aktivasyon: önce vektörü 4 kat genişletir, GELU uygular, sonra eski boyuta daraltır. Aktivasyon şart, çünkü iki Linear arka arkaya tek bir Linear'a eşittir. Bir bloğun parametrelerinin yaklaşık üçte ikisi MLP'dedir. GPT-2 GELU kullanır; Llama ve Qwen SwiGLU kullanır.

---

## 3. Residual bağlantı

### Sorun
- Modeller onlarca blok üst üste dizer (GPT-2 küçük 12, Llama 3 8B 32).
- Her blok vektörü **baştan yeniden yazarsa** kulaktan kulağa oyunu olur: küçük kayıplar birikir, orijinal bilgi kaybolur.
- Eğitimde de hata sinyali (gradyan) geriye doğru onlarca bloktan geçerken zayıflar; alt bloklar öğrenemez.

### Çözüm: Yeniden yazma, üstüne ekle
```python
x = x + attention(x)
x = x + mlp(x)
```
- Blok sadece **eklenecek düzeltmeyi** öğrenir; orijinal vektör hep korunur.
- **Taslak benzetmesi:** 30 editör metni baştan yazarsa orijinal kaybolur. Her editör metnin **kenarına not düşerse** metin hep yerinde durur, notlar birikir.
- Ana yola **residual stream** (artık akışı) denir: bloklar bu otobana bilgi ekleyen yan yollar.

```
x          = [1.0,  2.0]
attention  = [0.1, -0.3]     ← küçük düzeltme
x + att    = [1.1,  1.7]
```

### İşe yaramayan blok örneği
```
x = [3, 5], attention(x) = [0, 0]
Residual YOK:  x = attention(x)      = [0, 0]   ← bilgi silindi
Residual VAR:  x = x + attention(x)  = [3, 5]   ← bilgi korundu ✓
```
- Kötü bir blok en kötü ihtimalle **hiçbir şey yapmaz**, bilgiyi bozmaz → blokları rahatça üst üste dizebiliriz.
- Toplama, gradyanı geriye **olduğu gibi** geçirir → alt bloklar da güçlü sinyal alır.
- **ResNet (2015)** ile ortaya çıktı; yüzlerce katmanlı ağları mümkün kıldı.

### Şekil koruma burada ZORUNLU
- `x + attention(x)` için iki tarafın aynı şekilde olması gerekir: `(2, 5, 8) + (2, 5, 32)` toplanamaz.
- Bu yüzden LayerNorm, attention (W_o sayesinde) ve MLP'nin hepsi şekli korur.

### Özet
Residual bağlantıda bloğun çıktısı girdinin üstüne eklenir: x = x + f(x). Blok vektörü baştan yazmak yerine sadece düzeltme öğrenir. Böylece derin modellerde bilgi kaybolmaz ve eğitimde gradyan alt katmanlara zayıflamadan ulaşır. ResNet ile 2015'te ortaya çıktı. Toplama yapılabilmesi için bloktaki her parçanın şekli koruması gerekir.

---

## 4. Decoder bloğu: Parçaları birleştirmek

### Tarif: İki satır
```python
x = x + attention(layernorm1(x))
x = x + mlp(layernorm2(x))
```
Her satırda üç kavram: **1. Düzen** (LayerNorm, Pre-Norm) → **2. İş** (attention / MLP) → **3. Koruma** (residual `x +`).

```
           x  (2, 5, 8)
     ┌─────┴──────┐
     │       LayerNorm 1
     │       Attention        toplantı
     └──────→ (+) ←           residual
     ┌────────────┤
     │       LayerNorm 2
     │          MLP           masada düşünme
     └──────→ (+) ←           residual
           x  (2, 5, 8)       ← şekil aynı
```

- **Neden önce attention?** Token önce bağlamı toplar ("yüz" → "deniz"e bakar), sonra bu bilgiyle kendi başına düşünür.
- **Neden iki ayrı LayerNorm?** Her birinin **kendi γ ve β'sı** var; attention ve MLP farklı ölçek isteyebilir.
- `attn(ln1(x))` **içten dışa** okunur: önce `ln1(x)`, sonucu `attn`'e girer (matematikteki `f(g(x))`).

### İçerideki şekiller (dışarıdan hep aynı)
```
Attention:  (2,5,8) → 4 head'e BÖLÜNÜR → her biri (2,5,2) → cat → (2,5,8) → W_o → (2,5,8)
MLP:        (2,5,8) → GENİŞLER → (2,5,32) → GELU → daralt → (2,5,8)
```

### Model = Bu bloktan N tane
```
Embedding + Pozisyon → Blok 1 → Blok 2 → ... → Blok N → Son LayerNorm → lm_head → sonraki token
```
- Her blok aynı tarifle kurulur ama **kendi ağırlıklarına** sahiptir (her bloğun kendi 2 LayerNorm'u, attention'ı, MLP'si).

### Kendi denemem (`04_transformer_block.ipynb`)
| Hücre | Ne gördüm? |
|---|---|
| Dünkü `CausalSelfAttention` + `MultiHeadAttention` kopyalandı | `mha(batch)` → `(2, 5, 8)` |
| Blok uzun yoldan (`ln1`, `attn`, `ln2`, `mlp` ayrı ayrı) | attention sonrası `(2, 5, 8)`, MLP sonrası `(2, 5, 8)` |
| `DecoderBlock(8, 4)` sınıfı | `(2, 5, 8)` → içinde iki sınıf daha kullanıyor (Lego gibi) |

```python
class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadAttention(embedding_dim, num_heads)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

### Özet
Bir transformer bloğu iki adımdan oluşur. Önce vektör LayerNorm'dan geçer ve multi-head attention ile token'lar birbirinden bilgi toplar; sonuç residual ile girdinin üstüne eklenir. Sonra tekrar LayerNorm ve MLP ile her token tek başına işlenir, yine residual ile eklenir. Her parça şekli koruduğu için bloklar üst üste dizilebilir; model bu bloktan N tane içerir ve her birinin kendi ağırlıkları vardır.
