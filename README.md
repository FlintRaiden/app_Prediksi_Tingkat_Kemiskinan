---
title: Prediksi Kemiskinan Indonesia
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Prediksi Tingkat Kemiskinan di Indonesia
**UTS Praktikum Kecerdasan Buatan**

> Platform prediksi berbasis Machine Learning untuk mengestimasi persentase penduduk miskin per Kabupaten/Kota di Indonesia menggunakan 5 algoritma terintegrasi.

---

## Deskripsi Proyek

Aplikasi web Flask yang mengimplementasikan **5 algoritma Machine Learning** untuk memprediksi tingkat kemiskinan (% penduduk miskin) berdasarkan indikator sosial-ekonomi tingkat Kabupaten/Kota di Indonesia.

| # | Algoritma | Jenis | Framework |
|---|-----------|-------|-----------|
| 1 | Linear Regression | Supervised Regression | Scikit-learn |
| 2 | Artificial Neural Network (ANN) | Deep Learning | TensorFlow/Keras |
| 3 | LSTM / RNN | Sequential Deep Learning | TensorFlow/Keras |
| 4 | K-Means Clustering | Unsupervised Clustering | Scikit-learn |
| 5 | Backpropagation (Manual) | Custom NumPy | Pure NumPy |

---

## Dataset

- **Sumber**: Badan Pusat Statistik (BPS) Indonesia
- **Cakupan**: 514 Kabupaten/Kota dari 34 Provinsi
- **Target**: Persentase Penduduk Miskin (P0) dalam %
- **Rentang**: 2.38% - 41.66%

### Fitur Prediktor (9 Fitur)

| Fitur | Satuan |
|-------|--------|
| Rata-rata Lama Sekolah Penduduk 15+ | Tahun |
| Pengeluaran per Kapita Disesuaikan | Ribu Rp/Orang/Tahun |
| Indeks Pembangunan Manusia (IPM) | Indeks (0-100) |
| Umur Harapan Hidup | Tahun |
| Akses Sanitasi Layak | % Rumah Tangga |
| Akses Air Minum Layak | % Rumah Tangga |
| Tingkat Pengangguran Terbuka | % |
| Tingkat Partisipasi Angkatan Kerja | % |
| PDRB Harga Konstan | Rupiah |

---

## Struktur Proyek

```
poverty-prediction/
├── app/
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── predict.html
│   │   └── comparison.html
│   └── app.py
├── data/
│   └── Klasifikasi_Tingkat_Kemiskinan_di_Indonesia.csv
├── models/
├── notebooks/
│   └── 01_EDA_Training.ipynb
├── docs/
│   └── Laporan_UTS_Kemiskinan_Indonesia.docx
├── Dockerfile
├── Procfile
├── wsgi.py
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan Lokal

### 1. Clone Proyek

```bash
git clone <repo-url>
cd poverty-prediction
```

### 2. Buat Virtual Environment (Python 3.11)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
python wsgi.py
```

Buka browser: **http://localhost:7860**

---

## Halaman Aplikasi

| Halaman | URL | Deskripsi |
|---------|-----|-----------|
| Beranda | `/` | Statistik dataset dan EDA charts |
| Prediksi | `/predict` | Form input 9 indikator, output prediksi 5 model |
| Perbandingan | `/comparison` | Dashboard metrik dan kurva konvergensi |

---

## Metrik Evaluasi

| Model | Metrik |
|-------|--------|
| Linear Regression | MAE, RMSE, R2 Score |
| ANN (Keras) | MAE, RMSE, R2 Score |
| LSTM / RNN | MAE, RMSE, MAPE |
| K-Means | Inertia, Silhouette Score |
| Backpropagation | Loss, Convergence Rate, MAE, RMSE |

---

## Teknologi

- **Backend**: Flask 3.0, Python 3.11
- **Machine Learning**: Scikit-learn, TensorFlow/Keras, NumPy
- **Frontend**: Bootstrap 5, Chart.js 4, Bootstrap Icons
- **Deployment**: Gunicorn, Hugging Face Spaces

---

## Informasi UTS

- **Mata Kuliah**: Praktikum Kecerdasan Buatan
- **Topik**: Prediksi Tingkat Kemiskinan di Indonesia
- **Dataset**: BPS Indonesia (514 Kab/Kota, 34 Provinsi)