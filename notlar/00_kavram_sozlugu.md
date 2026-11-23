# 00 — Kavram Sözlüğü

---

## 1. Temel kavramlar

| Terim | Anlamı |
|---|---|
| **LLM** (Large Language Model) | Büyük dil modeli. Metindeki **bir sonraki token'ı tahmin etmek** için eğitilmiş, milyonlarca/milyarlarca parametreli sinir ağı |
| **Transformer** | 2017'de "Attention Is All You Need" makalesiyle çıkan mimari. Bugünkü bütün LLM'lerin temeli |
| **Decoder-only** | Sadece decoder bloklarından oluşan, soldan sağa metin üreten model (GPT, Llama, Qwen). Causal mask kullanır |
| **Encoder** (BERT) | Maske kullanmadan iki yönlü bakan model. Metin **anlamak** için (sınıflandırma, arama), üretmek için değil |
| **Parametre** | Modelin eğitimde öğrenilen sayıları (ağırlıklar). "8B" = 8 milyar parametre |
| **Ağırlık** (weight) | Parametrenin diğer adı. `nn.Linear`'daki matris, embedding tablosu, γ |
| **Bias** | `nn.Linear`'daki kaydırma terimi; LayerNorm'da β'nın PyTorch'taki adı |
| **Tensör** | Çok boyutlu sayı dizisi. PyTorch'un temel veri yapısı |
| **Shape** (şekil) | Tensörün boyutları. `(2, 5, 8)` = 2 cümle × 5 token × 8 sayı |
| **Batch** | Aynı anda işlenen cümle paketi. GPU'nun çekirdekleri boş kalmasın diye (otobüsü dolu kaldırmak) |
| **seq_len** | Cümledeki token sayısı |
| **embedding_dim** (d_model) | Her token'ın vektör uzunluğu (notebook'larda 8, GPT-2'de 768, Gemma 4 E2B'de 1536) |
| **Context length** | Modelin aynı anda görebildiği en fazla token sayısı |
| **Inference** | Eğitilmiş modeli kullanmak (cevap üretmek). Eğitimin tersi |
| **Scaling Laws** | OpenAI 2020: model başarısını en iyi embedding **hariç** parametre sayısı, veri ve hesap gücü tahmin eder |

---

## 2. Tokenizer

| Terim | Anlamı |
|---|---|
| **Token** | Modelin gördüğü en küçük birim: kelime, kelime parçası, harf ya da bayt |
| **Token ID** | Token'ın sözlükteki numarası. Anlamsız bir etiket (okul numarası gibi) |
| **Tokenizer** | Metin ↔ sayı tercümanı |
| **Encode** (tokenize) | Metin → token ID'leri. `merges` defterini kullanır |
| **Decode** (detokenize) | Token ID'leri → metin. `vocab` defterini kullanır |
| **Vocab** (sözlük) | ID → bayt eşlemesi. Boyutu = **vocab_size** |
| **Merges** (tarif defteri) | Hangi ikili hangi yeni ID'ye birleşti, öğrenilme sırasıyla |
| **BPE** (Byte Pair Encoding) | En sık yan yana gelen iki parçayı birleştirip tekrarlayarak sözlük öğrenme (GPT, Llama, Qwen) |
| **Byte-level BPE** | 256 baytla başlayan BPE → hiçbir metin bilinmeyen kalmaz, `<unk>` gerekmez |
| **Subword** | Alt kelime: kelime ile harf arası bölme. Bugünkü bütün LLM'ler |
| **UTF-8** | Metni bayta çeviren standart. İngilizce harf 1 bayt, Türkçe harf (ç ğ ı ö ş ü) 2 bayt, emoji 4 bayt |
| **WordPiece** | BERT'in tokenizer algoritması (olasılık artışına bakar) |
| **Unigram** | T5 ve Gemma'nın algoritması (büyük sözlükten başlayıp budar) |
| **SentencePiece** | Algoritma değil **kütüphane**; boşluğu `▁` ile gösterir |
| **Fertility** | Token sayısı ÷ kelime sayısı. Düşük olması iyi (Türkçe için ~1.5–2) |
| **Round-trip** | `decode(encode(metin)) == metin` → kayıpsız dönüş testi |
| **Pre-tokenization** | BPE'den önce metni kelime/noktalama olarak kaba bölme |
| **Özel token'lar** | `<bos>` başlangıç, `<eos>` bitiş (model bunu üretince durur), `<pad>` doldurma, `<unk>` bilinmeyen |
| **Chat template** | `<\|im_start\|>user ... <\|im_end\|>` gibi kimin konuştuğunu belirten şablon. Fine-tuning verisi buna uymalı |
| **`�`** (replacement char) | Yarım UTF-8 baytı decode edilince çıkan işaret. Streaming'de ikinci bayt beklenmeli |
| **HF `tokenizers`** | Hugging Face'in Rust ile yazılmış hızlı tokenizer kütüphanesi |

---

## 3. Embedding ve pozisyon

| Terim | Anlamı |
|---|---|
| **Embedding** | Her token'ı sabit uzunlukta bir vektörle temsil etme. `nn.Embedding(vocab_size, dim)` = arama tablosu; ID = satır numarası |
| **Vektör** | Sayı listesi. Uzayda bir nokta gibi düşünülür |
| **Dağılımsal anlam** | Kelime yanında durduğu kelimelerle tanınır → benzer bağlamdaki kelimelerin vektörleri yaklaşır |
| **Word2Vec** | 2013: kral − erkek + kadın ≈ kraliçe. Kavramlar uzayda **yön** olarak öğrenilir |
| **Kosinüs benzerliği** | İki vektörün yön benzerliği: 1 aynı, 0 alakasız, −1 zıt |
| **Weight tying** | Embedding tablosu ile çıkıştaki `lm_head`'in **aynı ağırlıkları paylaşması** → sözlük maliyeti yarıya iner |
| **Positional encoding** | Embedding ve attention sırayı bilmediği için eklenen **sıra bilgisi** |
| **Öğrenilen pozisyon** | GPT-2: ikinci bir `nn.Embedding(context_length, dim)`, token vektörüne **eklenir** |
| **Sinüs-kosinüs** | Orijinal Transformer: formülle hesaplanan pozisyon, parametresiz |
| **RoPE** | Llama, Qwen, Gemma: vektörü pozisyonu kadar **döndürür**; göreli mesafeyi kodlar, uzun metne iyi genelleşir |

---

## 4. Attention

| Terim | Anlamı |
|---|---|
| **Attention** | Her token'ın cümledeki diğer token'lara bakıp ilgili olanlardan bilgi toplaması ("yüz" → "deniz"e bakar → yüzmek) |
| **Self-attention** | Cümlenin kendi token'larının birbirine bakması |
| **Query (Q)** | "Ne arıyorum?" → kütüphanedeki isteğin |
| **Key (K)** | "Bende ne var?" → kitap sırtındaki etiket |
| **Value (V)** | "Seçilirsem ne veririm?" → kitabın içeriği |
| **Skor** (score) | Q · K nokta çarpımı → iki token ne kadar ilgili |
| **Nokta çarpımı** (dot product) | Karşılıklı sayıları çarpıp toplama. Büyük = benzer yön |
| **Softmax** | Skorları yüzdeye çevirir (hepsi pozitif, toplam %100) |
| **√d ile bölme** | Skorlar çok büyüyüp softmax tek bir token'a %100 vermesin diye ölçekleme |
| **Formül** | `softmax(Q·Kᵀ / √d) · V` → çıktı = Value'ların ağırlıklı toplamı |
| **Causal mask** | Her token sadece **kendisini ve öncekileri** görür. Gelecek `-inf` → softmax'te %0. Eğitimde kopya çekmeyi engeller |
| **Multi-head attention** | Vektörü head sayısına böl; her head **kendi Q/K/V ağırlıklarıyla** farklı açıdan bakar; çıktılar **uç uca eklenir** (`cat`) ve W_o'dan geçer |
| **head_dim** | `embedding_dim // num_heads` (8 // 4 = 2) |
| **W_o** | Head'lerin birleşik çıktısını karıştıran çıkış katmanı; boyutu orijinale döndürür |
| **KV cache** | Üretimde önceki token'ların K ve V'sini saklayıp tekrar hesaplamama hilesi |

---

## 5. Transformer bloğu

| Terim | Anlamı |
|---|---|
| **LayerNorm** | Her token vektörünü ortalama 0, std 1 yapar; sonra γ ile ölçekler, β ile kaydırır. Şekli değiştirmez |
| **γ (gamma) / β (beta)** | LayerNorm'un öğrenilen ölçek (başlangıç 1) ve kaydırma (başlangıç 0) parametreleri. Her sütuna bir tane |
| **ε (epsilon)** | Sıfıra bölmeyi önleyen minik sayı |
| **Standart sapma** | Sayıların ortalamadan ortalama uzaklığı: farkların karelerinin ortalamasının karekökü |
| **Varyans** | Standart sapmanın karesi (farkların karelerinin ortalaması) |
| **Pre-Norm** | LayerNorm'u attention ve MLP'den **önce** uygulamak (modern standart) |
| **Post-Norm** | LayerNorm'u sonra uygulamak (2017 orijinal) |
| **RMSNorm** | Ortalamayı çıkarmayan, daha hafif LayerNorm (Llama, Qwen, Gemma) |
| **MLP / Feed-Forward** | Her token'ı tek başına işleyen katman: genişlet (×4) → GELU → daralt. Bilgilerin çoğu burada; bloğun ~2/3'ü |
| **Aktivasyon fonksiyonu** | Linear katmanlar arasına konan çarpma olmayan işlem. Olmasa iki Linear = tek Linear |
| **ReLU** | Negatifse 0, pozitifse aynen |
| **GELU** | ReLU'nun yumuşak hâli (GPT-2, Gemma) |
| **SwiGLU** | Llama ve Qwen'in MLP aktivasyonu |
| **Residual bağlantı** | `x = x + f(x)`: bloğun çıktısını girdinin üstüne eklemek. Bilgi kaybolmaz, gradyan alt katmanlara ulaşır |
| **Residual stream** | Bloklar boyunca akan ana `x` yolu (otoban) |
| **ResNet** | 2015'te residual bağlantıyı getiren görüntü modeli |
| **Decoder bloğu** | `x = x + attn(ln1(x))` ve `x = x + mlp(ln2(x))`. Model bundan N tane içerir |

---

## 6. PyTorch terimleri

| Terim | Anlamı |
|---|---|
| **`nn.Module`** | Bütün katmanların ve modellerin miras aldığı temel sınıf |
| **`__init__`** | Makineleri (katmanları) **bir kez** kurduğumuz yer |
| **`forward`** | Veri her geldiğinde makinelerden geçirdiğimiz yer |
| **`super().__init__()`** | `nn.Module`'ün kendi hazırlığını yapması için zorunlu satır |
| **`nn.Linear(a, b)`** | "a sayı al, b sayı ver" makinesi. Sadece **son boyutu** değiştirir; `(a, b)` makinenin ayarıdır, çıktının şekli değil |
| **`nn.Embedding`** | Arama tablosu |
| **`nn.LayerNorm(dim)`** | LayerNorm katmanı; `weight` = γ, `bias` = β |
| **`nn.ModuleList`** | Katmanları liste hâlinde tutar (PyTorch onları parametre olarak tanısın diye) |
| **`torch.cat`** | Tensörleri uç uca ekler (toplama değil!) |
| **`torch.tril`** | Alt üçgen matris (causal mask için) |
| **`masked_fill`** | Maskeli yerlere değer yazar (`-inf`) |
| **`@`** | Matris çarpımı |
| **`.T` / `.transpose(-2, -1)`** | Matrisin satır ve sütunlarını yer değiştirme |
| **`requires_grad=True`** | "Bunun için türev hesapla, eğitimde güncellenecek" |
| **`grad_fn`** | Tensörün hangi işlemden geldiği notu; backprop bunu takip eder |
| **`Parameter`** | Modelin öğrenilen sayısı |
| **`torch.manual_seed`** | Rastgele sayıları tekrarlanabilir yapar |
| **CUDA** | NVIDIA GPU'da hesap yapma platformu (`device="cuda"`) |
| **Bilimsel gösterim** | `-4.05e-03` = −0.00405 (virgülü 3 basamak sola kaydır) |

---

## 7. Model ve eğitim 

| Terim | Anlamı |
|---|---|
| **lm_head** | Son vektörü sözlük boyutuna çeviren katman: her token için bir puan |
| **Logits** | lm_head'in çıktısı: softmax'tan önceki ham puanlar |
| **Next-token prediction** | Eğitim görevi: her pozisyonda bir sonraki token'ı tahmin et |
| **Loss** (kayıp) | Modelin ne kadar yanlış tahmin ettiğinin sayısı. Eğitimin amacı onu düşürmek |
| **Cross-entropy** | Dil modellerinde kullanılan loss: doğru token'a verilen olasılık ne kadar düşükse ceza o kadar büyük |
| **Gradyan** | Her parametrenin loss'u ne yönde ve ne kadar etkilediği |
| **Backpropagation** | Gradyanları çıkıştan girdiye doğru geriye hesaplama |
| **Optimizer** (AdamW) | Gradyanlara bakıp parametreleri güncelleyen algoritma |
| **Learning rate** | Her güncellemede ne kadar büyük adım atılacağı |
| **Epoch / step** | Verinin tamamından bir geçiş / tek bir güncelleme |
| **Overfitting** | Modelin veriyi ezberleyip yeni metinde kötüleşmesi |
| **Train / validation** | Eğitim verisi / modelin hiç görmediği kontrol verisi |
| **Perplexity** | Loss'un üstel hâli: model kaç seçenek arasında kararsız kalıyor |
| **Pretraining** | Büyük ham metinle sıfırdan eğitim |

---

## 8. Metin üretme 

| Terim | Anlamı |
|---|---|
| **Generate** | Token token üretme döngüsü: tahmin et → seç → sona ekle → tekrar |
| **Autoregressive** | Her yeni token'ın önceki ürettiği token'lara bakarak üretilmesi |
| **Greedy** | Her adımda en olası token'ı seçmek (tekrarlayıcı olur) |
| **Sampling** | Olasılıklara göre rastgele seçmek |
| **Temperature** | Olasılıkları keskinleştirir (<1, tutarlı) ya da düzleştirir (>1, yaratıcı) |
| **Top-k** | Sadece en olası k token arasından seç |
| **Top-p** (nucleus) | Olasılıkları toplamı p'ye ulaşan en küçük token grubundan seç |
| **Streaming** | Cevabı token token ekrana akıtmak |
| **Hallucination** | Modelin kendinden emin şekilde yanlış bilgi üretmesi |

---

## 9. Fine-tuning ve çalıştırma 

| Terim | Anlamı |
|---|---|
| **Fine-tuning** | Önceden eğitilmiş modeli kendi verimizle biraz daha eğitmek (yeni beceri/üslup) |
| **Base model / Instruct model** | Ham metin tamamlayan model / sohbet etmek için eğitilmiş model |
| **SFT** (Supervised Fine-Tuning) | Soru-cevap örnekleriyle fine-tuning |
| **LoRA** | Modelin tamamını değil, eklenen küçük matrisleri eğitmek → çok daha az bellek |
| **QLoRA** | 4-bit sıkıştırılmış model üzerinde LoRA (Colab T4'e sığar) |
| **Quantization** | Ağırlıkları daha az bitle saklamak (16 bit → 4 bit): 4 kat küçük, biraz kalite kaybı |
| **Q4 / Q8** | 4-bit / 8-bit quantize model (4B Q4 ≈ 2.5 GB) |
| **GGUF** | llama.cpp ve Ollama'nın model dosya formatı |
| **Ollama** | Modelleri yerelde tek komutla çalıştıran araç; API sunar |
| **llama.cpp** | Modelleri CPU/GPU'da verimli çalıştıran C++ motoru (Ollama'nın altında) |
| **Unsloth** | Fine-tuning'i hızlandıran, belleği azaltan kütüphane |
| **Hugging Face** | Model, veri seti ve kütüphane deposu (GitHub'ın yapay zekâ versiyonu) |
| **VRAM** | GPU belleği (bizde GTX 1650: 4 GB) |
| **tok/s** | Saniyede üretilen token: hız ölçüsü |
| **API** | Web arayüzünün modelle konuştuğu kapı (HTTP istekleri) |
| **System prompt** | Modelin davranışını belirleyen gizli başlangıç talimatı |
