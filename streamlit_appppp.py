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

# --- FUNGSI-FUNGSI HELPER ---

@st.cache_resource
def download_nltk_resources():
    """
    Memeriksa dan mengunduh resource NLTK yang dibutuhkan.
    Hanya dijalankan sekali saat aplikasi pertama kali dimulai.
    """
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
    """
    Memuat model ML, vectorizer, dan encoder.
    Hanya dijalankan sekali dan hasilnya disimpan di cache.
    """
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
    """
    Membersihkan dan memproses teks input dari Bahasa Indonesia ke Bahasa Inggris
    yang siap untuk model.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|@\w+|#\w*', '', text) # Hapus URL, mention, hashtag
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).strip() # Hapus karakter non-alfabet
    
    if not text:
        return ""
        
    try:
        translated_text = GoogleTranslator(source='id', target='en').translate(text)
    except Exception:
        translated_text = text # Fallback jika translasi gagal
    
    tokens = word_tokenize(translated_text.lower())
    stop_words_en = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words_en]
    
    return " ".join(lemmatized_tokens)

# --- EKSEKUSI UTAMA APLIKASI ---

# 1. Pastikan resource NLTK dan Model sudah siap saat aplikasi dimulai
download_nltk_resources()
model, vectorizer, encoder = load_resources()

# 2. Tampilkan judul dan antarmuka utama
st.title("Analisis Sentimen Menggunakan Bi-LSTM")
st.write("Aplikasi ini memprediksi sentimen (positif/negatif) dari ulasan atau teks berbahasa Indonesia.")

user_input = st.text_area("Masukkan teks di sini:", "Contoh: Pelayanannya cepat dan memuaskan!")

# 3. Logika Tombol Analisis (HANYA SATU BLOK)
if st.button("Analisis Sekarang"):
    # Pengecekan 1: Pastikan model dan semua komponennya sudah termuat
    if not model or not vectorizer or not encoder:
        st.error("Aplikasi belum siap karena gagal memuat resource. Periksa log untuk detail.")
    
    # Pengecekan 2: Pastikan ada teks yang dimasukkan oleh pengguna
    elif user_input:
        with st.spinner("Sedang memproses..."):
            # Langkah-langkah pemrosesan
            processed_text = preprocess_text(user_input)
            afinn_score = Afinn().score(processed_text)
            tfidf_vector = vectorizer.transform([processed_text])
            reshaped_vector = np.expand_dims(tfidf_vector.toarray(), axis=1)
            prediction_probs = model.predict(reshaped_vector)
            predicted_index = np.argmax(prediction_probs, axis=1)[0]
            final_prediction = encoder.inverse_transform([predicted_index])[0]

        # Tampilkan hasil prediksi utama
        st.subheader("Hasil Prediksi:")
        if final_prediction.lower() == 'positif':
            st.success(f"Sentimen: **Positif** 👍")
        else:
            st.error(f"Sentimen: **Negatif** 👎")

        # Tampilkan semua detail proses di dalam expander
        with st.expander("Lihat Detail Proses"):
            st.write(f"**Teks Setelah Preprocessing:** `{processed_text}`")
            #st.write(f"**Skor AFINN:** `{afinn_score}`")
            #st.write(f"**Probabilitas Prediksi (Negatif, Positif):** `{prediction_probs[0]}`")

    # Pengecekan 3 (else): Jika tidak ada input teks
    else:
        st.warning("Harap masukkan teks untuk dianalisis.")
