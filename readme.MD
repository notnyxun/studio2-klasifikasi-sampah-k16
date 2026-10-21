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
| Preview 2 (21 Okt) | — | 0.9551 | — |
| Final (25 Okt) | — | — | — |

**Catatan:** Macro-F1 Preview 2 sebesar **0.9551**, akurasi **0.9541**, dan Weighted-F1 **0.9540** diperoleh pada epoch terbaik ke-11 dari evaluasi lokal pada 414 gambar validasi. Nilai pada kolom Macro-F1 yang diumumkan asisten diisi setelah hasil resmi tersedia. Nilai validasi lokal bukan skor final kompetisi.

### Classification Report (Preview 2 - EfficientNet-B0)

| Kelas | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `botol_plastik` | 0.94 | 0.97 | 0.96 | 79 |
| `gelas_plastik` | 0.95 | 0.96 | 0.96 | 78 |
| `kertas_kardus` | 0.95 | 0.98 | 0.96 | 55 |
| `saset_kemasan` | 0.99 | 0.91 | 0.95 | 101 |
| `styrofoam` | 0.94 | 0.96 | 0.95 | 101 |
| **Accuracy** | | | **0.95** (0.9541) | 414 |
| **Macro Avg** | **0.95** | **0.96** | **0.96** (0.9551) | 414 |
| **Weighted Avg** | **0.95** | **0.95** | **0.95** (0.9540) | 414 |

### Catatan Eksperimen

- **Preview 1:** Model baseline ResNet-18 pretrained dilatih selama 8 epoch. Akurasi validasi lokal tercatat 0.9317 dan Macro-F1 sebesar 0.9255. Kelas `kertas_kardus` memiliki F1-Score terendah (0.8462) dan menjadi kandidat utama perbaikan.
- **Preview 2:** Eksperimen beralih ke arsitektur **EfficientNet-B0** yang dilatih hingga epoch 11. Model menghasilkan performa yang lebih seimbang dan meningkat di seluruh kelas: Macro-F1 naik menjadi **0.9551**, akurasi **0.9541**, dan Weighted-F1 **0.9540**. Kelas `kertas_kardus` mengalami peningkatan signifikan dari F1 0.8462 menjadi 0.96. Bobot terbaik disimpan pada `submission/model.pth`.

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
