# Final Project Data Mining (T)
**Anggota Kelompok**

| Nama               | NRP           | Kelompok |
|--------------------|---------------|----------|
| Audrey Sasqhia Wijaya    | 5025221055   | 09        |
| Amanda Illona Farrel     | 5025221056   | 09        |

**Video Demo** <br>
https://www.youtube.com/watch?v=BTx47l8Iv2Y&feature=youtu.be


# Restaurant Rating Prediction (1–5)
**TF-IDF vs Transformer vs Augmentation (EDA / Modified EDA / BERT Aug / Backtranslation)**

Skenario:
1) **Skenario 1: TF-IDF Baseline**  
2) **Skenario 2: Transformer Baseline**  
3) **Skenario 3: Training dari data augmentasi yang disimpan sebagai CSV**

Fokus utama evaluasi adalah **F1 Macro** untuk menangkap performa pada kelas minoritas.

# Dataset
Dataset berisi minimal kolom:
- `text` : teks ulasan restoran
- `rating` : label rating 1–5


# Struktur Skenario

### S1 — TF-IDF Baseline
Model:
- Linear SVM
- Naive Bayes
- Logistic Regression

**Hasil S1 (TF-IDF baseline):**
| Model | Accuracy | F1 Macro | F1 Weighted |
|---|---:|---:|---:|
| Linear SVM (S1 baseline) | 0.500000 | 0.375765 | 0.473433 |
| Logistic Regression (S1 baseline) | 0.477273 | 0.239589 | 0.402541 |
| Naive Bayes (S1 baseline) | 0.436364 | 0.147927 | 0.291245 |

### S2 — Transformer Baseline
Model:
- BERT (`bert-base-uncased`)
- DistilBERT (`distilbert-base-uncased`)

**Hasil S2 (Transformer baseline):**
| Model | Accuracy | F1 Macro | F1 Weighted | Eval Loss | Epoch |
|---|---:|---:|---:|---:|---:|
| bert-base-uncased (S2 baseline) | 0.536364 | 0.420316 | 0.533823 | 1.376944 | 6.0 |
| distilbert-base-uncased (S2 baseline) | 0.540909 | 0.384541 | 0.523294 | 1.119390 | 5.0 |

### S3 — Augmentation from CSV
Pada skenario ini, data train **sudah di-augment terlebih dahulu** dan disimpan sebagai CSV, lalu langsung dipakai untuk training model.

Metode augmentasi:
- **EDA**
- **Modified EDA**
- **BERT Augmentation**
- **Backtranslation**

Model yang dievaluasi dari CSV augmented:
- Linear SVM
- Naive Bayes
- Logistic Regression
- BERT
- DistilBERT

**Hasil S3 (ringkas):**

1. BERT (S3)

| Strategy        | Accuracy |     F1 Macro | F1 Weighted | Eval Loss | Epoch |
| --------------- | -------: | -----------: | ----------: | --------: | ----: |
| EDA             | 0.563636 | **0.476050** |    0.552760 |  1.325081 |   4.0 |
| Modified EDA    | 0.540909 |     0.375052 |    0.502462 |  1.214526 |   3.0 |
| BERT Aug        | 0.486364 |     0.347210 |    0.442112 |  1.236302 |   2.0 |
| Backtranslation | 0.563636 |     0.400881 |    0.552043 |  1.211754 |   3.0 |

2. DistilBERT (S3)

| Strategy        | Accuracy |     F1 Macro | F1 Weighted | Eval Loss | Epoch |
| --------------- | -------: | -----------: | ----------: | --------: | ----: |
| EDA             | 0.531818 |     0.445806 |    0.529768 |  1.134219 |   3.0 |
| Modified EDA    | 0.518182 |     0.415016 |    0.518361 |  1.182725 |   3.0 |
| BERT Aug        | 0.509091 |     0.419016 |    0.512371 |  1.266814 |   4.0 |
| Backtranslation | 0.554545 | **0.446246** |    0.555131 |  1.170164 |   3.0 |

3. Linear SVM (S3)

| Strategy        | Accuracy |     F1 Macro | F1 Weighted | 
| --------------- | -------: | -----------: | ----------: | 
| EDA             | 0.472727 | **0.375121** |    0.464937 | 
| Modified EDA    | 0.468182 | **0.380895** |    0.460067 | 
| BERT Aug        | 0.440909 |     0.372659 |    0.437744 | 
| Backtranslation | 0.459091 |     0.373903 |    0.453903 | 

4. Naive Bayes (S3)

| Strategy        | Accuracy |     F1 Macro | F1 Weighted |
| --------------- | -------: | -----------: | ----------: |
| EDA             | 0.472727 | **0.405282** |    0.483077 |
| Modified EDA    | 0.463636 | **0.406881** |    0.477030 |
| BERT Aug        | 0.445455 |     0.369150 |    0.439468 |
| Backtranslation | 0.404545 |     0.345135 |    0.424870 |

5. Logistic Regression (S3)

| Strategy        | Accuracy |     F1 Macro | F1 Weighted |
| --------------- | -------: | -----------: | ----------: |
| EDA             | 0.472727 |     0.370317 |    0.473298 |
| Modified EDA    | 0.477273 | **0.383209** |    0.478697 |
| BERT Aug        | 0.450000 |     0.374484 |    0.444189 |
| Backtranslation | 0.486364 |     0.381717 |    0.481611 |

## Summary Utama
Berdasarkan rekap keseluruhan (sorted by F1 Macro):
- **Eksperimen terbaik:** `bert-base-uncased (S3 aug=eda)`  
  - Accuracy: **0.563636**
  - **F1 Macro: 0.476050**
- Augmentasi **lebih terasa manfaatnya di Transformer** dibanding TF-IDF.
- Performa TF-IDF baseline relatif lebih rendah pada kelas minoritas.

## Kenapa Akurasi & F1 Tidak Tinggi?
Untuk klasifikasi rating 1–5:
- Distribusi label umumnya **tidak seimbang** (bias ke rating tinggi).
- Model cenderung “aman” dengan memprediksi kelas dominan.
- **F1 Macro** menjadi metrik penting karena menilai rata-rata performa tiap kelas.
