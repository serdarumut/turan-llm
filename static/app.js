const sohbet = document.getElementById("sohbet");
const karsilama = document.getElementById("karsilama");
const mesajlar = document.getElementById("mesajlar");
const form = document.getElementById("form");
const giris = document.getElementById("giris");
const gonderButonu = document.getElementById("gonder");
const modelSecici = document.getElementById("model");
const yeniButonu = document.getElementById("yeni");
const temaButonu = document.getElementById("tema");

let gecmis = [];        // sohbet hafızası: [{role, content}, ...]
let calisiyor = false;  // cevap akarken yeni mesaj gönderilmesin

// 1) Model listesini API'den al, seçiciye doldur
async function modelleriYukle() {
  const cevap = await fetch("/api/models");
  const modeller = await cevap.json();
  for (const model of modeller) {
    const secenek = document.createElement("option");
    secenek.value = model.id;
    secenek.textContent = model.ad;
    modelSecici.appendChild(secenek);
  }
}

// 2) Metni güvenli HTML'e çevir: **kalın** ve "- madde" desteği
function guvenli(metin) {
  return metin.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function bicimle(metin) {
  let html = "";
  let listede = false;
  for (let satir of guvenli(metin).split("\n")) {
    satir = satir.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    if (/^\s*[-*] /.test(satir)) {
      if (!listede) { html += "<ul>"; listede = true; }
      html += "<li>" + satir.replace(/^\s*[-*] /, "") + "</li>";
    } else {
      if (listede) { html += "</ul>"; listede = false; }
      if (satir.trim()) html += "<p>" + satir + "</p>";
    }
  }
  if (listede) html += "</ul>";
  return html;
}

function asagiKaydir() {
  sohbet.scrollTop = sohbet.scrollHeight;
}

// 3) Ekrana yeni bir balon ekle
function balonEkle(sinif, etiket) {
  karsilama.hidden = true;
  const balon = document.createElement("div");
  balon.className = "balon " + sinif;
  if (etiket) {
    const e = document.createElement("div");
    e.className = "etiket";
    e.textContent = etiket;
    balon.appendChild(e);
  }
  const icerik = document.createElement("div");
  icerik.className = "icerik";
  balon.appendChild(icerik);
  const alt = document.createElement("div");
  alt.className = "alt";
  balon.appendChild(alt);
  mesajlar.appendChild(balon);
  asagiKaydir();
  return balon;
}

// 4) Mesajı gönder, cevabı akarak göster
async function gonder(metin) {
  if (calisiyor || !metin.trim()) return;
  calisiyor = true;
  gonderButonu.disabled = true;

  const kullaniciBalonu = balonEkle("kullanici");
  kullaniciBalonu.querySelector(".icerik").textContent = metin;
  gecmis.push({ role: "user", content: metin });

  const modelAdi = modelSecici.selectedOptions[0].textContent;
  const balon = balonEkle("turan", modelAdi);
  const icerik = balon.querySelector(".icerik");
  icerik.innerHTML = '<span class="dusunuyor"><i></i><i></i><i></i></span>';

  let yanit = "";
  try {
    const cevap = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelSecici.value, mesajlar: gecmis }),
    });
    if (!cevap.ok) throw new Error("Sunucu hatası (" + cevap.status + ")");

    const okuyucu = cevap.body.getReader();
    const cozucu = new TextDecoder();
    let tampon = "";

    while (true) {
      const { done, value } = await okuyucu.read();
      if (done) break;
      tampon += cozucu.decode(value, { stream: true });
      const satirlar = tampon.split("\n");
      tampon = satirlar.pop();          // yarım kalan satır bir sonraki tura
      for (const satir of satirlar) {
        if (!satir) continue;
        const parca = JSON.parse(satir);
        if (parca.bitti) {
          balon.querySelector(".alt").textContent = "⚡ " + parca.hiz + " token/sn";
        } else {
          yanit += parca.metin;
          icerik.innerHTML = bicimle(yanit);
          asagiKaydir();
        }
      }
    }
    gecmis.push({ role: "assistant", content: yanit });
  } catch (hata) {
    icerik.innerHTML = '<p class="hata">Bağlantı hatası: ' + guvenli(hata.message) + "</p>";
    gecmis.pop();  // cevapsız kalan soruyu hafızadan çıkar
  }

  calisiyor = false;
  gonderButonu.disabled = false;
  giris.focus();
}

// 5) Olaylar: Enter ile gönder, Shift+Enter yeni satır
form.addEventListener("submit", (olay) => {
  olay.preventDefault();
  if (calisiyor || !giris.value.trim()) return;
  const metin = giris.value;
  giris.value = "";
  giris.style.height = "auto";
  gonder(metin);
});

giris.addEventListener("keydown", (olay) => {
  if (olay.key === "Enter" && !olay.shiftKey) {
    olay.preventDefault();
    form.requestSubmit();
  }
});

giris.addEventListener("input", () => {
  giris.style.height = "auto";
  giris.style.height = giris.scrollHeight + "px";
});

document.querySelectorAll("[data-soru]").forEach((buton) => {
  buton.addEventListener("click", () => gonder(buton.dataset.soru));
});

yeniButonu.addEventListener("click", () => {
  if (calisiyor) return;
  gecmis = [];
  mesajlar.innerHTML = "";
  karsilama.hidden = false;
  giris.focus();
});

// 6) Karanlık / aydınlık tema (tercih tarayıcıda hatırlanır)
function temaUygula(tema) {
  document.documentElement.dataset.theme = tema;
  temaButonu.textContent = tema === "dark" ? "☀️" : "🌙";
  try { localStorage.setItem("tema", tema); } catch (e) {}
}

let kayitliTema = null;
try { kayitliTema = localStorage.getItem("tema"); } catch (e) {}
temaUygula(kayitliTema || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));

temaButonu.addEventListener("click", () => {
  temaUygula(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});

modelleriYukle();