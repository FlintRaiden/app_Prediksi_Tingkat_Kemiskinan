# =============================================================================
# app/app.py — Prediksi Tingkat Kemiskinan di Indonesia
# UTS Praktikum Kecerdasan Buatan
# =============================================================================

import os
import sys
import json
import threading
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore")

# ── Flask Setup ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# Make Python builtins available in Jinja2 templates
app.jinja_env.globals.update(enumerate=enumerate, zip=zip, len=len, round=round)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = ROOT_DIR / "data" / "Klasifikasi_Tingkat_Kemiskinan_di_Indonesia.csv"

FEATURE_COLS = [
    "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)",
    "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)",
    "Indeks Pembangunan Manusia",
    "Umur Harapan Hidup (Tahun)",
    "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak",
    "Persentase rumah tangga yang memiliki akses terhadap air minum layak",
    "Tingkat Pengangguran Terbuka",
    "Tingkat Partisipasi Angkatan Kerja",
    "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)",
]

TARGET_COL = "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)"

FORM_FIELDS = [
    {
        "name": "lama_sekolah",
        "label": "Rata-rata Lama Sekolah (Tahun)",
        "unit": "Tahun",
        "min": 0, "max": 20, "step": 0.01,
        "placeholder": "Contoh: 9.50",
        "col": FEATURE_COLS[0],
        "icon": "bi-book",
        "tip": "Rata-rata tahun sekolah penduduk usia 15+ di wilayah tersebut.",
    },
    {
        "name": "pengeluaran",
        "label": "Pengeluaran per Kapita (Ribu Rp/Orang/Tahun)",
        "unit": "Ribu Rp",
        "min": 1000, "max": 30000, "step": 1,
        "placeholder": "Contoh: 8500",
        "col": FEATURE_COLS[1],
        "icon": "bi-cash-stack",
        "tip": "Total pengeluaran konsumsi per kapita yang disesuaikan.",
    },
    {
        "name": "ipm",
        "label": "Indeks Pembangunan Manusia (IPM)",
        "unit": "",
        "min": 0, "max": 100, "step": 0.01,
        "placeholder": "Contoh: 70.50",
        "col": FEATURE_COLS[2],
        "icon": "bi-people",
        "tip": "Indikator komposit pendidikan, kesehatan, dan standar hidup.",
    },
    {
        "name": "harapan_hidup",
        "label": "Umur Harapan Hidup (Tahun)",
        "unit": "Tahun",
        "min": 50, "max": 85, "step": 0.01,
        "placeholder": "Contoh: 68.00",
        "col": FEATURE_COLS[3],
        "icon": "bi-heart-pulse",
        "tip": "Angka harapan hidup saat lahir di wilayah tersebut.",
    },
    {
        "name": "sanitasi",
        "label": "Akses Sanitasi Layak (%)",
        "unit": "%",
        "min": 0, "max": 100, "step": 0.01,
        "placeholder": "Contoh: 75.00",
        "col": FEATURE_COLS[4],
        "icon": "bi-droplet",
        "tip": "Persentase rumah tangga yang menggunakan fasilitas sanitasi layak.",
    },
    {
        "name": "air_minum",
        "label": "Akses Air Minum Layak (%)",
        "unit": "%",
        "min": 0, "max": 100, "step": 0.01,
        "placeholder": "Contoh: 80.00",
        "col": FEATURE_COLS[5],
        "icon": "bi-water",
        "tip": "Persentase rumah tangga yang memiliki akses air minum bersih.",
    },
    {
        "name": "pengangguran",
        "label": "Tingkat Pengangguran Terbuka (%)",
        "unit": "%",
        "min": 0, "max": 30, "step": 0.01,
        "placeholder": "Contoh: 5.50",
        "col": FEATURE_COLS[6],
        "icon": "bi-person-x",
        "tip": "Persentase angkatan kerja yang sedang mencari pekerjaan.",
    },
    {
        "name": "tpak",
        "label": "Tingkat Partisipasi Angkatan Kerja (%)",
        "unit": "%",
        "min": 0, "max": 100, "step": 0.01,
        "placeholder": "Contoh: 65.00",
        "col": FEATURE_COLS[7],
        "icon": "bi-briefcase",
        "tip": "Persentase penduduk usia kerja yang aktif dalam pasar kerja.",
    },
    {
        "name": "pdrb",
        "label": "PDRB Harga Konstan (Rupiah)",
        "unit": "Rp",
        "min": 100000, "max": 5000000000, "step": 1000,
        "placeholder": "Contoh: 5000000",
        "col": FEATURE_COLS[8],
        "icon": "bi-graph-up-arrow",
        "tip": "Produk Domestik Regional Bruto berdasarkan harga konstan.",
    },
]

# ── Global application state ──────────────────────────────────────────────────
_state = {
    "models": {},
    "scalers": {},
    "metrics": {},
    "eda_data": {},
    "cluster_names": {},
    "bp_losses": [],
    "ready": False,
    "training": True,
    "error": None,
}


# =============================================================================
# Manual Backpropagation Neural Network (Pure NumPy)
# =============================================================================

class BackpropNN:
    """
    Fully manual feedforward neural network implementing backpropagation
    from scratch using NumPy. Demonstrates the core weight-update loop.
    """

    def __init__(self, layer_sizes, lr=0.005, epochs=400):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.epochs = epochs
        self.losses = []
        self.weights = []
        self.biases = []
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            self.weights.append(np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in))
            self.biases.append(np.zeros((1, n_out)))

    # ── Activation ───────────────────────────────────────────────────────────
    @staticmethod
    def _relu(z):       return np.maximum(0, z)
    @staticmethod
    def _relu_d(z):     return (z > 0).astype(float)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def _forward(self, X):
        self._a = [X]
        self._z = []
        cur = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = cur @ W + b
            self._z.append(z)
            cur = self._relu(z) if i < len(self.weights) - 1 else z
            self._a.append(cur)
        return cur

    # ── Backward pass ────────────────────────────────────────────────────────
    def _backward(self, y):
        m = self._a[0].shape[0]
        delta = (self._a[-1] - y.reshape(-1, 1)) * (2.0 / m)
        for i in reversed(range(len(self.weights))):
            dW = self._a[i].T @ delta
            db = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = delta @ self.weights[i].T * self._relu_d(self._z[i - 1])
            self.weights[i] -= self.lr * dW
            self.biases[i]  -= self.lr * db

    # ── Training ─────────────────────────────────────────────────────────────
    def fit(self, X, y):
        for _ in range(self.epochs):
            y_hat = self._forward(X).flatten()
            self.losses.append(float(np.mean((y_hat - y) ** 2)))
            self._backward(y)
        return self

    def predict(self, X):
        return self._forward(X).flatten()

    def convergence_rate(self):
        if len(self.losses) < 2:
            return 0.0
        return float((self.losses[0] - self.losses[-1]) / max(self.losses[0], 1e-9) * 100)


# =============================================================================
# Data Loading & Preprocessing
# =============================================================================

def load_data():
    df = pd.read_csv(DATA_PATH, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    for col in [TARGET_COL] + FEATURE_COLS:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".").str.strip(),
            errors="coerce",
        )
    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS)
    return df.reset_index(drop=True)


# =============================================================================
# Model Training (runs once in background thread at startup)
# =============================================================================

def train_all_models():
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            mean_absolute_error, mean_squared_error, r2_score, silhouette_score
        )

        # ── Load & split data ─────────────────────────────────────────────────
        df = load_data()
        X = df[FEATURE_COLS].values.astype(np.float64)
        y = df[TARGET_COL].values.astype(np.float64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        _state["scalers"]["standard"] = scaler

        # ── 1. Linear Regression ─────────────────────────────────────────────
        lr_model = LinearRegression()
        lr_model.fit(X_tr, y_train)
        y_lr = lr_model.predict(X_te)
        _state["models"]["linear"] = lr_model
        _state["metrics"]["linear"] = {
            "mae":  round(float(mean_absolute_error(y_test, y_lr)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_lr))), 4),
            "r2":   round(float(r2_score(y_test, y_lr)), 4),
        }

        # ── 2. ANN (Keras/TensorFlow) ────────────────────────────────────────
        try:
            import tensorflow as tf
            tf.random.set_seed(42)

            ann = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X_tr.shape[1],)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1),
            ])
            ann.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss="mse", metrics=["mae"]
            )
            ann.fit(
                X_tr, y_train,
                epochs=80, batch_size=32, verbose=0,
                validation_split=0.1,
            )
            y_ann = ann.predict(X_te, verbose=0).flatten()
            _state["models"]["ann"] = ("keras", ann)
            _state["metrics"]["ann"] = {
                "mae":  round(float(mean_absolute_error(y_test, y_ann)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_ann))), 4),
                "r2":   round(float(r2_score(y_test, y_ann)), 4),
                "backend": "TensorFlow/Keras",
            }
        except Exception:
            from sklearn.neural_network import MLPRegressor
            ann_sk = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), max_iter=300,
                random_state=42, early_stopping=True
            )
            ann_sk.fit(X_tr, y_train)
            y_ann = ann_sk.predict(X_te)
            _state["models"]["ann"] = ("sklearn", ann_sk)
            _state["metrics"]["ann"] = {
                "mae":  round(float(mean_absolute_error(y_test, y_ann)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_ann))), 4),
                "r2":   round(float(r2_score(y_test, y_ann)), 4),
                "backend": "Scikit-learn MLP (fallback)",
            }

        # ── 3. RNN / LSTM (Keras/TensorFlow) ────────────────────────────────
        try:
            import tensorflow as tf
            tf.random.set_seed(42)
            n_feat = X_tr.shape[1]

            lstm = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(n_feat, 1)),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ])
            lstm.compile(optimizer="adam", loss="mse", metrics=["mae"])
            lstm.fit(
                X_tr.reshape(-1, n_feat, 1), y_train,
                epochs=80, batch_size=32, verbose=0,
                validation_split=0.1,
            )
            y_lstm = lstm.predict(
                X_te.reshape(-1, n_feat, 1), verbose=0
            ).flatten()
            _state["models"]["lstm"] = ("keras", lstm)
            _state["metrics"]["lstm"] = {
                "mae":  round(float(mean_absolute_error(y_test, y_lstm)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_lstm))), 4),
                "mape": round(float(
                    np.mean(np.abs((y_test - y_lstm) / np.maximum(y_test, 1e-9))) * 100
                ), 4),
                "backend": "TensorFlow/Keras LSTM",
            }
        except Exception:
            from sklearn.neural_network import MLPRegressor
            lstm_sk = MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=300,
                random_state=0, early_stopping=True
            )
            lstm_sk.fit(X_tr, y_train)
            y_lstm = lstm_sk.predict(X_te)
            _state["models"]["lstm"] = ("sklearn", lstm_sk)
            _state["metrics"]["lstm"] = {
                "mae":  round(float(mean_absolute_error(y_test, y_lstm)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_lstm))), 4),
                "mape": round(float(
                    np.mean(np.abs((y_test - y_lstm) / np.maximum(y_test, 1e-9))) * 100
                ), 4),
                "backend": "Scikit-learn MLP (fallback)",
            }

        # ── 4. K-Means Clustering ────────────────────────────────────────────
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        X_all_sc = scaler.transform(X)
        kmeans.fit(X_all_sc)
        labels_all = kmeans.labels_
        sil = float(silhouette_score(X_all_sc, labels_all, sample_size=min(400, len(X))))

        # Map cluster IDs → poverty category
        cluster_poverty_avg = {
            c: float(y[labels_all == c].mean()) for c in range(3)
        }
        sorted_c = sorted(cluster_poverty_avg, key=cluster_poverty_avg.get)
        cluster_names = {
            sorted_c[0]: ("Kemiskinan Rendah",  "success",  "↓ Rendah"),
            sorted_c[1]: ("Kemiskinan Sedang",  "warning",  "~ Sedang"),
            sorted_c[2]: ("Kemiskinan Tinggi",  "danger",   "↑ Tinggi"),
        }
        _state["models"]["kmeans"] = kmeans
        _state["cluster_names"] = cluster_names
        _state["metrics"]["kmeans"] = {
            "inertia":    round(float(kmeans.inertia_), 2),
            "silhouette": round(sil, 4),
            "n_clusters": 3,
            "cluster_avgs": {
                cluster_names[c][0]: round(cluster_poverty_avg[c], 2)
                for c in range(3)
            },
        }

        # ── 5. Backpropagation (Manual NumPy) ───────────────────────────────
        bp = BackpropNN(
            layer_sizes=[X_tr.shape[1], 64, 32, 1],
            lr=0.005, epochs=400,
        )
        bp.fit(X_tr, y_train)
        y_bp = bp.predict(X_te)

        _state["models"]["backprop"] = bp
        _state["bp_losses"] = bp.losses
        _state["metrics"]["backprop"] = {
            "initial_loss":     round(float(bp.losses[0]), 4),
            "final_loss":       round(float(bp.losses[-1]), 4),
            "convergence_rate": round(float(bp.convergence_rate()), 2),
            "mae":              round(float(mean_absolute_error(y_test, y_bp)), 4),
            "rmse":             round(float(np.sqrt(mean_squared_error(y_test, y_bp))), 4),
        }

        # ── EDA Aggregates (for homepage charts) ─────────────────────────────
        prov_poverty = (
            df.groupby("Provinsi")[TARGET_COL]
            .mean()
            .sort_values(ascending=False)
        )
        hist_counts, hist_edges = np.histogram(y, bins=15)
        corr = {
            col: round(float(df[[TARGET_COL, col]].corr().iloc[0, 1]), 3)
            for col in FEATURE_COLS
        }

        _state["eda_data"] = {
            "n_rows":       int(len(df)),
            "n_provinces":  int(df["Provinsi"].nunique()),
            "poverty_mean": round(float(y.mean()), 2),
            "poverty_min":  round(float(y.min()),  2),
            "poverty_max":  round(float(y.max()),  2),
            "hist_counts":  hist_counts.tolist(),
            "hist_labels":  [
                f"{hist_edges[i]:.1f}–{hist_edges[i+1]:.1f}"
                for i in range(len(hist_counts))
            ],
            "prov_labels":  prov_poverty.index[:12].tolist(),
            "prov_values":  [round(v, 2) for v in prov_poverty.values[:12].tolist()],
            "corr_short":   [
                "Lama Sekolah", "Pengeluaran", "IPM", "Harapan Hidup",
                "Sanitasi", "Air Minum", "Pengangguran", "TPAK", "PDRB"
            ],
            "corr_values":  list(corr.values()),
        }

        _state["ready"]    = True
        _state["training"] = False
        print("[✓] Semua model berhasil dilatih.")

    except Exception as exc:
        import traceback
        _state["error"]    = str(exc)
        _state["training"] = False
        _state["ready"]    = False
        print(f"[✗] Pelatihan model gagal: {exc}")
        traceback.print_exc()


# =============================================================================
# Prediction Helper
# =============================================================================

def _predict_single(X_raw: np.ndarray) -> dict:
    """Run all 5 models on a single (1, n_features) input."""
    scaler = _state["scalers"]["standard"]
    X_sc   = scaler.transform(X_raw)          # (1, 9)
    preds  = {}

    # Linear Regression
    preds["linear"] = float(_state["models"]["linear"].predict(X_sc)[0])

    # ANN
    kind, ann_model = _state["models"]["ann"]
    if kind == "keras":
        preds["ann"] = float(ann_model.predict(X_sc, verbose=0).flatten()[0])
    else:
        preds["ann"] = float(ann_model.predict(X_sc)[0])

    # LSTM
    kind, lstm_model = _state["models"]["lstm"]
    if kind == "keras":
        X_lstm = X_sc.reshape(-1, X_sc.shape[1], 1)
        preds["lstm"] = float(lstm_model.predict(X_lstm, verbose=0).flatten()[0])
    else:
        preds["lstm"] = float(lstm_model.predict(X_sc)[0])

    # Backpropagation
    preds["backprop"] = float(_state["models"]["backprop"].predict(X_sc)[0])

    # Ensemble average (regression models only)
    preds["ensemble"] = float(np.mean([
        preds["linear"], preds["ann"], preds["lstm"], preds["backprop"]
    ]))

    # K-Means cluster
    cluster_id   = int(_state["models"]["kmeans"].predict(X_sc)[0])
    cluster_info = _state["cluster_names"].get(cluster_id, ("Unknown", "secondary", "?"))
    preds["cluster_id"]   = cluster_id
    preds["cluster_name"] = cluster_info[0]
    preds["cluster_color"]= cluster_info[1]
    preds["cluster_icon"] = cluster_info[2]

    # Round predictions
    for k in ["linear", "ann", "lstm", "backprop", "ensemble"]:
        preds[k] = round(max(0.0, preds[k]), 2)

    return preds


# =============================================================================
# Routes
# =============================================================================

@app.route("/")
def index():
    eda = _state["eda_data"]
    return render_template(
        "index.html",
        ready   = _state["ready"],
        training= _state["training"],
        error   = _state["error"],
        eda     = eda,
        hist_labels  = json.dumps(eda.get("hist_labels",  [])),
        hist_counts  = json.dumps(eda.get("hist_counts",  [])),
        prov_labels  = json.dumps(eda.get("prov_labels",  [])),
        prov_values  = json.dumps(eda.get("prov_values",  [])),
        corr_short   = json.dumps(eda.get("corr_short",   [])),
        corr_values  = json.dumps(eda.get("corr_values",  [])),
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error  = None
    form_data = {}

    if request.method == "POST":
        if not _state["ready"]:
            error = "Model sedang dalam proses pelatihan. Silakan coba beberapa saat lagi."
        else:
            try:
                values = []
                for f in FORM_FIELDS:
                    raw = request.form.get(f["name"], "0").replace(",", ".")
                    val = float(raw)
                    form_data[f["name"]] = val
                    values.append(val)

                X_raw = np.array(values, dtype=np.float64).reshape(1, -1)
                preds = _predict_single(X_raw)

                # Poverty level label
                ens = preds["ensemble"]
                if ens < 10:
                    level, level_color = "Rendah",  "success"
                elif ens < 20:
                    level, level_color = "Sedang",  "warning"
                else:
                    level, level_color = "Tinggi",  "danger"

                result = {
                    **preds,
                    "level":       level,
                    "level_color": level_color,
                    "input_labels": [f["label"] for f in FORM_FIELDS],
                    "input_values": values,
                    "pred_labels":  json.dumps(
                        ["Linear Regression", "ANN", "LSTM/RNN", "Backpropagation"]
                    ),
                    "pred_values":  json.dumps([
                        preds["linear"], preds["ann"],
                        preds["lstm"],   preds["backprop"],
                    ]),
                }
            except Exception as exc:
                error = f"Kesalahan prediksi: {exc}"

    return render_template(
        "predict.html",
        form_fields = FORM_FIELDS,
        result      = result,
        error       = error,
        form_data   = form_data,
        ready       = _state["ready"],
        training    = _state["training"],
    )


@app.route("/comparison")
def comparison():
    m    = _state["metrics"]
    bpl  = _state["bp_losses"]

    # Sample loss curve for chart (max 60 points)
    step = max(1, len(bpl) // 60)
    bp_ep  = list(range(0, len(bpl), step))
    bp_lv  = [round(bpl[i], 4) for i in bp_ep]

    reg_labels = ["Linear Regression", "ANN", "LSTM/RNN", "Backpropagation"]
    reg_mae    = [
        m.get("linear",  {}).get("mae",  0),
        m.get("ann",     {}).get("mae",  0),
        m.get("lstm",    {}).get("mae",  0),
        m.get("backprop",{}).get("mae",  0),
    ]
    reg_rmse   = [
        m.get("linear",  {}).get("rmse", 0),
        m.get("ann",     {}).get("rmse", 0),
        m.get("lstm",    {}).get("rmse", 0),
        m.get("backprop",{}).get("rmse", 0),
    ]

    return render_template(
        "comparison.html",
        metrics    = m,
        ready      = _state["ready"],
        training   = _state["training"],
        reg_labels = json.dumps(reg_labels),
        reg_mae    = json.dumps(reg_mae),
        reg_rmse   = json.dumps(reg_rmse),
        bp_epochs  = json.dumps(bp_ep),
        bp_losses  = json.dumps(bp_lv),
    )


@app.route("/api/status")
def api_status():
    return jsonify({
        "ready":    _state["ready"],
        "training": _state["training"],
        "error":    _state["error"],
    })


# =============================================================================
# Boot — Train models in background
# =============================================================================
threading.Thread(target=train_all_models, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
