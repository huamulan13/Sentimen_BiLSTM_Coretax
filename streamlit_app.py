import streamlit as st
import numpy as np
import re
import pickle
import nltk
from tensorflow.keras.models import load_model
from afinn import Afinn
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- FUNGSI-FUNGSI HELPER (Tidak ada perubahan) ---

@st.cache_resource
def download_nltk_resources():
    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
    }
    for path, pkg_id in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg_id)
    print("✅ Resource NLTK siap.")

@st.cache_resource
def load_resources():
    try:
        model = load_model('bilstm_model_from_tfidf.h5')
        with open('tfidf_vectorizer_for_bilstm.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('label_encoder_for_bilstm.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, vectorizer, encoder
    except FileNotFoundError:
        st.error("Gagal memuat file model/vectorizer/encoder. Pastikan file tersebut ada di repository GitHub Anda.")
        return None, None, None

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|@\w+|#\w*', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).strip()
    if not text: return ""
    try:
        translated_text = GoogleTranslator(source='id', target='en').translate(text)
    except Exception:
        translated_text = text
    tokens = word_tokenize(translated_text.lower())
    stop_words_en = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words_en]
    return " ".join(lemmatized_tokens)

# --- EKSEKUSI UTAMA APLIKASI ---

# 1. Pastikan resource NLTK dan Model sudah siap
download_nltk_resources()
model, vectorizer, encoder = load_resources()

# 2. Menyuntikkan CSS untuk mengubah tampilan
st.markdown("""
<style>
    /* Mengubah latar belakang utama aplikasi */
    .stApp {
        background-color: #f0f2f6;
    }
    /* Membuat container utama menjadi kartu di tengah */
    .main .block-container {
        max-width: 700px; /* Lebar maksimum kartu */
        margin: 2rem auto; /* Membuat kartu berada di tengah */
        padding: 2rem;
        background-color: white;
        border-radius: 1rem; /* Sudut melengkung */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); /* Efek bayangan */
    }
    /* Mengubah gaya tombol agar memenuhi lebar */
    .stButton button {
        background-color: #1d4ed8; /* Warna biru */
        color: white;
        width: 100%;
        border-radius: 0.5rem;
        border: none;
        padding: 0.75rem 0;
    }
    .stButton button:hover {
        background-color: #1e40af;
        color: white;
    }
    /* Mengubah gaya judul */
    h1 {
        text-align: center;
        font-weight: bold;
        color: #1f2937;
    }
    /* Mengubah gaya subjudul (deskripsi di bawah judul) */
    .st-emotion-cache-1629p8f p {
        text-align: center;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)


# 3. Tampilkan judul dan antarmuka utama
st.title("Dashboard Analisis Sentimen")
st.write("Deploy Web Sederhana Bi-LSTM dengan AFINN")

user_input = st.text_area("Masukkan teks di sini:", "Contoh: aplikasi ini error terus, payah!", label_visibility="collapsed", placeholder="Contoh: aplikasi ini error terus, payah!")

# 4. Logika Tombol Analisis
if st.button("Analisis Sekarang"):
    if not model or not vectorizer or not encoder:
        st.error("Aplikasi belum siap karena gagal memuat resource. Periksa log untuk detail.")
    elif user_input:
        with st.spinner("Sedang memproses..."):
            processed_text = preprocess_text(user_input)
            afinn_score = Afinn().score(processed_text)
            tfidf_vector = vectorizer.transform([processed_text])
            reshaped_vector = np.expand_dims(tfidf_vector.toarray(), axis=1)
            prediction_probs = model.predict(reshaped_vector)
            predicted_index = np.argmax(prediction_probs, axis=1)[0]
            final_prediction = encoder.inverse_transform([predicted_index])[0]

        st.subheader("Hasil Prediksi:")
        if final_prediction.lower() == 'positif':
            st.success(f"Sentimen: **Positif** 👍")
        else:
            st.error(f"Sentimen: **Negatif** 👎")

        with st.expander("Lihat Detail Proses"):
            st.write(f"**Teks Setelah Preprocessing:** `{processed_text}`")
            st.write(f"**Skor AFINN:** `{afinn_score}`")
            st.write(f"**Probabilitas Prediksi (Negatif, Positif):** `{prediction_probs[0]}`")

    else:
        st.warning("Harap masukkan teks untuk dianalisis.")

