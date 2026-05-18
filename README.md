---

title: Prediksi Kemiskinan Indonesia
emoji: 🇮🇩
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false

---

#  Prediksi Tingkat Kemiskinan di Indonesia
**UTS Praktikum Kecerdasan Buatan**

> Platform prediksi berbasis Machine Learning untuk mengestimasi persentase penduduk miskin per Kabupaten/Kota di Indonesia menggunakan 5 algoritma terintegrasi.

---

##  Deskripsi Proyek

Aplikasi web Flask yang mengimplementasikan **5 algoritma Machine Learning** untuk memprediksi tingkat kemiskinan (% penduduk miskin) berdasarkan indikator sosial-ekonomi tingkat Kabupaten/Kota di Indonesia.

| # | Algoritma | Jenis | Framework |
|---|-----------|-------|-----------|
| 1 | Linear Regression | Supervised Regression | Scikit-learn |
| 2 | Artificial Neural Network (ANN) | Deep Learning | TensorFlow/Keras |
| 3 | LSTM / RNN | Sequential Deep Learning | TensorFlow/Keras |
| 4 | K-Means Clustering | Unsupervised Clustering | Scikit-learn |
| 5 | Backpropagation (Manual) | Custom NumPy | Pure NumPy |

---

##  Dataset

- **Sumber**: Badan Pusat Statistik (BPS) Indonesia / Kaggle
- **Cakupan**: 514 Kabupaten/Kota dari 34 Provinsi
- **Target**: `Persentase Penduduk Miskin (P0)` dalam %
- **Rentang**: 2.38% — 41.66%

### Fitur Prediktor (9 Fitur)

| Fitur | Satuan |
|-------|--------|
| Rata-rata Lama Sekolah Penduduk 15+ | Tahun |
| Pengeluaran per Kapita Disesuaikan | Ribu Rp/Orang/Tahun |
| Indeks Pembangunan Manusia (IPM) | Indeks (0–100) |
| Umur Harapan Hidup | Tahun |
| Akses Sanitasi Layak | % Rumah Tangga |
| Akses Air Minum Layak | % Rumah Tangga |
| Tingkat Pengangguran Terbuka | % |
| Tingkat Partisipasi Angkatan Kerja | % |
| PDRB Harga Konstan | Rupiah |

---

##  Struktur Proyek

```
poverty-prediction/
├── 📁 app/
│   ├── 📁 static/
│   │   ├── css/          # Stylesheet tambahan
│   │   └── js/           # Script tambahan
│   ├── 📁 templates/
│   │   ├── base.html     # Template dasar (navbar + footer)
│   │   ├── index.html    # Halaman Beranda + EDA charts
│   │   ├── predict.html  # Halaman Prediksi Real-Time
│   │   └── comparison.html # Dashboard Perbandingan Model
│   └── app.py            # Flask application + 5 ML models
├── 📁 data/
│   └── Klasifikasi_Tingkat_Kemiskinan_di_Indonesia.csv
├── 📁 models/            # Saved model artifacts
├── 📁 notebooks/         # Jupyter Notebook EDA 
├── wsgi.py               # Entry point (Hugging Face / Gunicorn)
├── Procfile              # Konfigurasi deployment
├── requirements.txt      # Dependensi Python
└── README.md
```

---

##  Cara Menjalankan Lokal

### 1. Clone / Download Proyek

```bash
git clone <repo-url>
cd poverty-prediction
```

### 2. Buat Virtual Environment (Python 3.11)

```bash
python3.11 -m venv venv
source venv/bin/activate        # Linux/macOS
# atau
venv\Scripts\activate           # Windows
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
# Dari root direktori proyek:
python wsgi.py

# Atau langsung dari folder app:
cd app
python app.py 
```

Buka browser: **http://localhost:7860**

---

##  Deployment ke Hugging Face Spaces

### Persiapan

1. Buat akun di [huggingface.co](https://huggingface.co)
2. Buat Space baru → pilih **Docker** atau **Gradio/Flask** (gunakan SDK: `docker`)

### Struktur File yang Diunggah

Upload seluruh isi folder proyek. Hugging Face akan otomatis mendeteksi `Procfile` dan `requirements.txt`.

### Konfigurasi `README.md` untuk Hugging Face

Tambahkan header YAML di bagian paling atas `README.md`:

```yaml
---
title: Prediksi Kemiskinan Indonesia
emoji: 🇮🇩
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

### Dockerfile (Opsional — jika Space tipe Docker)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["gunicorn", "--chdir", "app", "app:app", "--bind", "0.0.0.0:7860", \
     "--workers", "1", "--timeout", "300"]
```

---

##  Metrik Evaluasi

| Model | Metrik Utama |
|-------|-------------|
| Linear Regression | MAE, RMSE, R² Score |
| ANN (Keras) | MAE, RMSE, R² Score |
| LSTM / RNN | MAE, RMSE, MAPE |
| K-Means | Inertia, Silhouette Score |
| Backpropagation | Initial/Final Loss, Convergence Rate, MAE, RMSE |

---

##  Halaman Aplikasi

| Halaman | URL | Deskripsi |
|---------|-----|-----------|
| Beranda | `/` | Welcome, statistik dataset, EDA charts (distribusi, provinsi, korelasi) |
| Prediksi | `/predict` | Form input 9 indikator → output prediksi 5 model + klaster K-Means |
| Perbandingan | `/comparison` | Dashboard metrik, chart MAE/RMSE, kurva konvergensi backpropagation |

---

##  Teknologi

- **Backend**: Flask 3.0, Python 3.11
- **Machine Learning**: Scikit-learn, TensorFlow/Keras, NumPy
- **Frontend**: Bootstrap 5, Chart.js 4, Bootstrap Icons
- **Deployment**: Gunicorn, Hugging Face Spaces

---

##  Informasi UTS

- **Mata Kuliah**: Praktikum Kecerdasan Buatan
- **Topik**: Prediksi Tingkat Kemiskinan di Indonesia
- **Dataset**: BPS Indonesia (514 Kab/Kota, 34 Provinsi)
