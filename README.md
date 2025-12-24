# UAP-ML
# Kania Kusvania-20221037031108

# 🎩 Klasifikasi Jenis Topi Menggunakan Deep Learning

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk membangun sistem **klasifikasi citra** guna mendeteksi **keberadaan topi** pada gambar serta mengklasifikasikan **jenis topi** menggunakan metode **Deep Learning berbasis Convolutional Neural Network (CNN)**.

Sistem dikembangkan secara end-to-end mulai dari preprocessing data, pelatihan model, evaluasi, hingga **deployment aplikasi web menggunakan Streamlit**.

---

## 🎯 Tujuan Proyek
- Menerapkan preprocessing citra digital
- Melatih dan mengevaluasi model deep learning
- Melakukan klasifikasi:
  - **Binary Classification**: Ada Topi / Tidak Ada Topi
  - **Multiclass Classification**: Jenis-jenis topi
- Membangun aplikasi web interaktif berbasis Streamlit

---

## 🧠 Model yang Digunakan
Model deep learning yang digunakan pada proyek ini:
- **CNN (Custom Convolutional Neural Network)**
- **MobileNetV2**
- **ResNet50**
- **Multiclass CNN** (khusus untuk klasifikasi jenis topi)

Model disimpan dalam format `.h5`, sedangkan metadata kelas disimpan dalam format `.pkl`.

---

## 📂 Dataset
- Dataset berupa kumpulan **gambar topi**
- Dataset telah dibagi menjadi:
  - Data Train
  - Data Validation
  - Data Test
- Setiap kelas direpresentasikan dalam folder terpisah
- Berikut link dataset: https://drive.google.com/drive/folders/1ycRsELyjKkaYKOENF2x7xPcNwXM6fFG7?usp=sharing
- Berikut link model: https://drive.google.com/drive/folders/1Aahvc-4chz4DkwkRAVUO7LMzYdksWmkQ?usp=sharing

---

## ⚙️ Tahapan Pengerjaan Proyek
1. Import library
2. Load dataset gambar
3. Preprocessing data:
   - Resize gambar ke 224 × 224
   - Normalisasi pixel (0–1)
4. Split dataset:
   - Training
   - Validation
   - Testing
5. Pembuatan model 
6. Training model
7. Evaluasi performa model
8. Penyimpanan model dan metadata
9. Deployment aplikasi menggunakan Streamlit

---

## 🖼️ Preprocessing Gambar
Setiap gambar yang diinput akan melalui proses:
- Resize gambar ke ukuran **224 × 224**
- Konversi ke array NumPy
- Normalisasi nilai pixel
- Penambahan dimensi batch untuk input model

---

## 📊 Evaluasi Model
Evaluasi model dilakukan menggunakan:
- Akurasi
- Loss
- Confidence score pada hasil prediksi

Model terbaik dipilih berdasarkan performa pada data validation.

---
