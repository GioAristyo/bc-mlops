# Deteksi Tumor Ganas - MLOps dengan MLflow

Proyek ini mengikuti **Skenario: Deteksi Tumor Ganas (Breast Cancer Wisconsin)** pada materi *Model Experimentation & Tracking in MLOps*, khususnya slide 54-81. Kode dibuat agar bisa dijalankan lokal di Ubuntu + Miniforge/Conda.

> **Batasan penting:** ini proyek pembelajaran dengan dataset benchmark kecil, bukan alat diagnosis/skrining klinis. Angka pada tabel hasil acuan di laporan berasal dari slide 71-72 materi kuliah dan harus dibandingkan dengan hasil run lokal sebelum laporan difinalisasi.

## Isi proyek

```text
bc-mlops/
├── data/                         # train.csv/test.csv dibuat lokal oleh src/data.py
├── outputs/                      # hasil seleksi, evaluasi, confusion matrix
├── reports/                      # laporan PDF
├── src/
│   ├── data.py                   # L1: split tetap, label, hash dan metadata
│   ├── common.py                 # konfigurasi bersama
│   ├── tracking.py               # logging CV, metrik, model, artefak ke MLflow
│   ├── train.py                  # L2-L4: baseline, seleksi model, 12 trial SVC
│   ├── analyze.py                # L5: seleksi kandidat dengan aturan metrik
│   ├── evaluate.py               # L6: evaluasi final test sekali
│   ├── register.py               # L7: registry dan quality gate
│   ├── predict_api.py            # L8: contoh pemanggilan REST API
│   └── reproduce.py              # L9: periksa lineage/data hash
├── tests/test_data.py
├── environment.yml               # paket utama dikelola Conda/Mamba
├── requirements.txt              # versi paket yang dicatat sebagai artefak run
└── .gitignore
```

## 1. Environment Ubuntu + Miniforge

Buka terminal di root folder proyek dan aktifkan environment. Jika kamu sudah membuat environment `bc-mlops` dan impor semua library berhasil, **tidak perlu membuat environment baru**.

```bash
conda activate bc-mlops
python --version
python -c "import pandas, sklearn, matplotlib, mlflow, requests; print('semua library terimpor')"
```

Untuk membuat environment baru di komputer lain:

```bash
mamba env create -f environment.yml
mamba activate bc-mlops
```

File `environment.yml` memakai `conda-forge` dan mem-pin versi paket utama. `requirements.txt` disimpan sebagai metadata lingkungan yang bisa dilampirkan ke setiap run MLflow; penginstalan utama proyek ini menggunakan Conda/Mamba, bukan pip.

## 2. L1 - siapkan data

```bash
python -m src.data
cat data/metadata.json
ls -lh data/
```

Target: 569 sampel, 30 fitur, 455 train dan 114 test. Label dibalik agar `0=benign` dan `1=malignant`. Split dibuat stratified dengan seed 42. `data/metadata.json` menyimpan fingerprint, seed, ukuran split dan mapping label. Fingerprint MD5 di sini untuk penelusuran eksperimen, bukan keamanan kriptografis.

## 3. Terminal 1 - jalankan MLflow Tracking Server

Jalankan dari root project dengan environment `bc-mlops` aktif, lalu biarkan terminal tetap terbuka:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

Buka `http://127.0.0.1:5000` di browser untuk melihat MLflow UI. Database dan artefak disimpan lokal dan diabaikan oleh Git.

## 4. Terminal 2 - eksperimen L2-L4

Buka terminal kedua, aktifkan environment, masuk ke root project, dan set tracking URI:

```bash
conda activate bc-mlops
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m src.train
```

Run yang dibuat mencakup DummyClassifier, Logistic Regression tanpa scaling, Logistic Regression + StandardScaler, KNN, Decision Tree, Random Forest, Gradient Boosting, SVC default, dan GridSearchCV SVC dengan 12 kombinasi `C` dan `gamma`. Semua model dibandingkan memakai 5-fold Stratified CV dengan seed/fold yang sama. Scaling berada di dalam `Pipeline` untuk menghindari data leakage antar-fold.

## 5. L5 - pilih kandidat otomatis

```bash
python -m src.analyze
```

Aturan ditentukan sebelum test: pilih recall CV tertinggi di antara run dengan `cv_precision_mean >= 0.90`, lalu gunakan F1 sebagai tie-breaker. Kandidat tersimpan di `outputs/selected_candidate.json`.

## 6. L6 - evaluasi final di test set

```bash
python -m src.evaluate
```

Script menguji kandidat terpilih, menyimpan metrik ke `outputs/test_metrics.json`, membuat confusion matrix, serta mengaitkan metrik test ke run MLflow yang sama. Evaluasi berulang ditolak secara default untuk menekan risiko test leakage. Jangan mengubah hyperparameter berdasarkan hasil test lalu mengklaim test yang sama sebagai evaluasi final independen.

## 7. L7 - registry dan quality gate

```bash
python -m src.register
```

Model didaftarkan sebagai `bc-malignant-classifier`, diberi alias `challenger`, dan dipromosikan menjadi `champion` jika precision test minimal 0.90 serta recall mengalahkan champion sebelumnya pada test split yang sama dengan margin 0.005. Jika quality gate gagal, model tidak dipromosikan.

## 8. L8 - REST serving

Jalankan hanya jika alias `champion` sudah ada. Terminal 3:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve \
  -m 'models:/bc-malignant-classifier@champion' \
  -p 5001 \
  --env-manager local
```

Terminal lain:

```bash
python -m src.predict_api --url http://127.0.0.1:5001/invocations --rows 3
```

## 9. L9 - cek asal-usul/reproduksibilitas

```bash
python -m src.reproduce
```

Script menampilkan versi champion, run ID, commit (jika tercatat), fingerprint data, parameter dan metrik, lalu memeriksa kecocokan fingerprint.

## 10. Tes kode

```bash
pytest -q
```

Pastikan menjalankan perintah dari root proyek. Jika ada error, baca traceback dan pastikan server MLflow hidup untuk langkah yang memerlukan Tracking API.

## 11. Bukti dan GitHub

Screenshot yang dianjurkan untuk laporan: (1) MLflow table/compare view yang memuat semua model utama, (2) parent `10-svc-gridsearch` dan 12 child trials, (3) test metrics/confusion matrix, dan (4) registry alias `champion` serta respons REST API bila langkah ini berhasil.

Di repository lokal yang sudah memiliki `.git`, **jangan menjalankan `git init` lagi**. Setelah memeriksa `git status`, commit kode dengan:

```bash
git add .gitignore README.md environment.yml requirements.txt src tests reports
# Data CSV, outputs, db dan artefak lokal memang diabaikan lewat .gitignore.
git status
git commit -m "Add breast cancer MLOps experiment pipeline"
```

Buat repository kosong di GitHub, lalu hubungkan dan push (ganti USERNAME):

```bash
git remote add origin https://github.com/USERNAME/bc-mlops.git
git push -u origin main
```

Jika remote sudah ada, cek `git remote -v` sebelum menambahkannya lagi. Jangan mengunggah token/password, `.env`, database MLflow atau seluruh environment Conda.

## Hasil acuan yang tercantum di materi kuliah

Angka ini diambil dari tabel materi kuliah slide 71-72, bukan klaim hasil lokal mahasiswa. Verifikasi melalui MLflow UI dan ganti/konfirmasi di laporan berdasarkan eksperimen yang benar-benar dijalankan.

| Kandidat | Recall CV | Precision CV | F1 CV |
|---|---:|---:|---:|
| Dummy (most frequent) | 0.000 | 0.000 | 0.000 |
| Logistic Regression tanpa scaling | 0.918 | 0.958 | 0.937 |
| Logistic Regression + scaling | 0.953 | 0.977 | 0.964 |
| KNN | 0.918 | 0.988 | 0.951 |
| Decision Tree | 0.853 | 0.894 | 0.872 |
| Random Forest | 0.935 | 0.964 | 0.949 |
| Gradient Boosting | 0.947 | 0.965 | 0.955 |
| SVC default | 0.947 | 0.977 | 0.961 |
| SVC GridSearchCV | 0.965 | 0.971 | 0.967 |

Materi juga mencantumkan contoh evaluasi test untuk SVC: recall 0.929, precision 1.000, ROC-AUC 0.993. Itu adalah nilai acuan dari slide; gunakan metrik yang ditampilkan run lokalmu untuk laporan final.

## Catatan medis dan etika

Dataset ini untuk pendidikan dan eksperimen ML. Model benchmark kecil ini belum divalidasi secara klinis, bukan untuk memutuskan diagnosis atau perawatan, dan tidak boleh menggantikan dokter. Recall/precision pada test split ini tidak menjamin kinerja terhadap pasien/populasi lain.

## Referensi

1. Materi kuliah *Model Experimentation & Tracking in MLOps*, Skenario Deteksi Tumor Ganas, slide 54-81 (file kuliah yang diberikan).
2. scikit-learn, `load_breast_cancer`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
3. MLflow Tracking: https://mlflow.org/docs/latest/ml/tracking/
4. MLflow Model Registry: https://mlflow.org/docs/latest/ml/model-registry/
5. MLflow Model Serving: https://mlflow.org/docs/latest/ml/deployment/
