import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import pickle
import time

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Klasifikasi Jenis Topi",
    page_icon="🎩",
    layout="centered"
)

# =============================
# PATH PROYEK
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# =============================
# PATH MODEL (FINAL & VALID)
# =============================
MODEL_PATHS = {
    "CNN Base": os.path.join(MODEL_DIR, "cnn_base_model.h5"),
    "MobileNet": os.path.join(MODEL_DIR, "mobilenet_model.keras"),
    "ResNet": os.path.join(MODEL_DIR, "resnet_model.keras")
}

METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")

# =============================
# LOAD LABEL KELAS (AMAN)
# =============================
CLASS_NAMES = [
    "Baseball Cap",
    "Bucket Hat",
    "Fedora",
    "Beanie",
    "Topi Pantai"
]

if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)

        if isinstance(metadata, dict):
            for key in ["class_names", "classes", "labels"]:
                if key in metadata:
                    CLASS_NAMES = metadata[key]
                    break
    except:
        pass

IMG_SIZE = (224, 224)

# =============================
# STYLE UI (KONTRAS & MODERN)
# =============================
st.markdown("""
<style>
body {
    background-color: #0F172A;
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    color: #FFFFFF;
}
.subtitle {
    text-align: center;
    color: #CBD5E1;
    margin-bottom: 30px;
}
.result-card {
    background: linear-gradient(135deg, #1E293B, #020617);
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 15px 35px rgba(0,0,0,0.45);
}
.result-title {
    font-size: 18px;
    color: #60A5FA;
    margin-bottom: 10px;
    font-weight: 600;
}
.result-class {
    font-size: 34px;
    color: #FFFFFF;
    font-weight: 800;
    margin-bottom: 12px;
}
.result-confidence {
    font-size: 18px;
    color: #22C55E;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown('<div class="title">🎩 Klasifikasi Jenis Topi</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">CNN Base • MobileNet • ResNet</div>', unsafe_allow_html=True)

# =============================
# PILIH MODEL
# =============================
model_choice = st.selectbox(
    "Pilih Model Deep Learning",
    list(MODEL_PATHS.keys())
)

model_path = MODEL_PATHS[model_choice]

# =============================
# LOAD MODEL (AMAN TOTAL)
# =============================
@st.cache_resource
def load_model_safe(path):
    try:
        return load_model(path, compile=False), None
    except Exception as e:
        return None, str(e)

model = None
error_msg = None

if not os.path.exists(model_path):
    error_msg = "File model tidak ditemukan."
else:
    model, error_msg = load_model_safe(model_path)

if model is None:
    st.error(f"❌ Model {model_choice} gagal dimuat")

    st.warning(
        f"""
        Model **{model_choice}** tidak dapat digunakan.

        📌 Kemungkinan penyebab:
        - File model rusak
        - Model tidak disimpan dengan `model.save()`
        - Versi TensorFlow berbeda saat training
        """
    )

    st.info("👉 Silakan pilih model lain.")

else:
    st.success(f"✅ Model {model_choice} berhasil dimuat")

# =============================
# UPLOAD GAMBAR
# =============================
uploaded_file = st.file_uploader(
    "Upload gambar topi (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# =============================
# PREPROCESSING
# =============================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# =============================
# PREDIKSI
# =============================
if uploaded_file and model is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Gambar Input", use_column_width=True)

    with col2:
        with st.spinner("🔍 Menganalisis gambar..."):
            time.sleep(1)
            preds = model.predict(preprocess_image(image))
            idx = np.argmax(preds)
            confidence = float(np.max(preds) * 100)
            predicted_class = CLASS_NAMES[idx]

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">🧢 Jenis Topi</div>
                <div class="result-class">{predicted_class}</div>
                <div class="result-confidence">
                    Confidence: {confidence:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("UAP Machine Learning • Klasifikasi Jenis Topi")

