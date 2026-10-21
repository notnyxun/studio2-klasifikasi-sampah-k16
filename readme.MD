# Studio 2 - Klasifikasi Sampah · Kelompok 16/FARID 16 PRO MAX 256 GB NATURAL TITANIUM

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|---:|---|---|---|
| 1 | BAGAS DWI AJITYA | 123140181 | Setup Project |
| 2 | MUHAMMAD FATAHILLAH FARID | 123140203 | Test Dataset |
| 3 | ADITYA HOT MARTUA SIHITE | 123140114 | Train Dataset |

## Ringkasan Pendekatan

- **Arsitektur / backbone:**
  - **Preview 1:** ResNet-18 pretrained ImageNet.
  - **Preview 2:** EfficientNet-B0 pretrained (`torchvision.models.efficientnet_b0`). Lapisan classifier bawaan diganti dengan `Linear` yang menghasilkan **5 kelas** (`nn.Linear(1280, 5)`).
- **Strategi pelatihan:** Fine-tuning bobot pretrained pada dataset klasifikasi sampah. Model terbaik dipilih berdasarkan metrik evaluasi validasi lokal dan disimpan ke `submission/model.pth`.
- **Learning rate, optimizer, scheduler, dan augmentasi:** Menggunakan optimasi PyTorch dengan augmentasi rotasi/crop pada data latih serta cosine scheduler.
- **Data tambahan:** Tidak ada data tambahan di luar dataset yang disediakan untuk tugas Studio 2.
- **Ukuran input & preprocessing:** Preprocessing inferensi pada `submission/model.py` menerapkan transformasi:
  1. `Resize(256)`
  2. `CenterCrop(224)`
  3. `ToTensor()`
  4. Normalisasi kanal RGB dengan mean `[0.485, 0.456, 0.406]` dan standard deviation `[0.229, 0.224, 0.225]`.
- **Framework:** PyTorch dan Torchvision.
- **Lingkungan pelatihan:** Google Colab dengan GPU CUDA.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---|---|---:|---:|
| Preview 1 (14 Okt) | f33515f | 0.9255 | — |
| Preview 2 (21 Okt) | — | 0.9929 | — |
| Final (25 Okt) | — | — | — |

**Catatan:** Macro-F1 Preview 2 sebesar **0.9929** dan akurasi **0.9927** diperoleh dari evaluasi lokal pada 410 gambar split validasi (15% stratified split dari dataset train dengan seed 42). Nilai pada kolom Macro-F1 yang diumumkan asisten diisi setelah hasil resmi tersedia. Nilai validasi lokal bukan skor final kompetisi.

### F1-Score per Kelas

| Kelas | F1-Score (Preview 1 - ResNet18) | F1-Score (Preview 2 - EfficientNet-B0) |
|---|---:|---:|
| `botol_plastik` | 0.9474 | 1.0000 |
| `gelas_plastik` | 0.9620 | 0.9935 |
| `kertas_kardus` | 0.8462 | 0.9908 |
| `saset_kemasan` | 0.9307 | 0.9950 |
| `styrofoam` | 0.9412 | 0.9851 |

### Catatan Eksperimen

- **Preview 1:** Model baseline ResNet-18 pretrained dilatih selama 8 epoch. Akurasi validasi lokal tercatat 0.9317 dan Macro-F1 sebesar 0.9255. Kelas `kertas_kardus` memiliki F1-Score terendah (0.8462) dan menjadi kandidat utama perbaikan.
- **Preview 2:** Eksperimen beralih ke arsitektur **EfficientNet-B0**. Model ini memberikan representasi fitur yang jauh lebih baik dengan jumlah parameter yang efisien. Evaluasi lokal menunjukkan peningkatan signifikan di semua kelas: Macro-F1 naik ke **0.9929** dan akurasi mencapai **0.9927**. Kelas `kertas_kardus` melonjak ke F1 0.9908. Bobot terbaik disimpan pada `submission/model.pth`.

## Cara Reproduksi

Jalankan perintah berikut dari direktori utama repositori.

### 1. Instalasi Dependensi

Pastikan Python, PyTorch, dan Torchvision tersedia:

```bash
pip install -r submission/requirements.txt
```

### 2. Evaluasi Model Resmi

Jalankan evaluasi menggunakan evaluator resmi Studio 2:

```bash
python evaluate.py \
  --submission ./submission \
  --data ./my_val \
  --batch-size 32 \
  --out ./evaluation.json
```

Folder `my_val` harus berisi validation set lokal dengan struktur folder kelas yang sesuai (`<data>/<nama_kelas>/*.jpg`).

### 3. Penggunaan Inferensi Mandiri

```python
import torch
from PIL import Image
from submission.model import get_model, get_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
state_dict = torch.load("submission/model.pth", map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

transform = get_transform()
image = Image.open("path/to/image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

with torch.inference_mode():
    logits = model(input_tensor)
    pred_idx = logits.argmax(1).item()

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
print("Prediksi:", CLASSES[pred_idx])
```

## Struktur Repositori

```text
.
├── readme.MD
├── train_baseline.py
├── evaluate.py
└── submission/
    ├── model.py          # Definisi model EfficientNet-B0 & get_transform()
    ├── model.pth         # Bobot state_dict terbaik Preview 2 (~16.3 MB)
    └── requirements.txt  # Dependensi inferensi
```

## Catatan

- Dataset dan validation set lokal tidak disertakan dalam repositori.
- Skor resmi mengikuti evaluasi asisten pada validation set tersembunyi.
- Seluruh eksperimen dan submission mengikuti ketentuan tugas Studio 2.
