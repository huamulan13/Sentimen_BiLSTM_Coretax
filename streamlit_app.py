import streamlit as st
import nltk
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from afinn import Afinn

# Fungsi-fungsi preprocessing yang sama persis dari skrip pelatihan
# (Kita letakkan di sini agar skripnya mandiri)
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

@st.cache_resource
def download_nltk_resources():
    try:
        # Memeriksa apakah paket sudah ada
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        # Jika belum ada, unduh paketnya
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
    print("✅ Resource NLTK siap.")

# Pastikan resource NLTK sudah ada
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet'], quiet=True)

# ----------------- LOGIKA INTI APLIKASI -----------------

# Gunakan cache agar model tidak di-load ulang setiap kali ada interaksi
# Di dalam streamlit_app.py

@st.cache_resource
def load_resources():
    """Fungsi untuk me-load model, vectorizer, dan encoder."""
    try:
        model = load_model('bilstm_model_from_tfidf.h5')
        with open('tfidf_vectorizer_for_bilstm.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('label_encoder_for_bilstm.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, vectorizer, encoder
    except FileNotFoundError:
        st.error("Gagal memuat file model/vectorizer/encoder. Pastikan file tersebut ada.")
        return None, None, None

def preprocess_text(text):
    """Fungsi untuk membersihkan dan memproses teks input."""
    # (Ini adalah pipeline yang sama dari skrip pelatihan)
    text = str(text).lower()
    text = re.sub(r'http\\S+|\\@\\w+|\\#\\w*', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', ' ', text).strip()
    if not text:
        return ""
    try:
        # Untuk performa lebih cepat, translasi bisa dilewati jika input sudah bahasa inggris
        translated_text = GoogleTranslator(source='id', target='en').translate(text)
    except Exception:
       translated_text = text # Fallback
    
    tokens = word_tokenize(translated_text.lower())
    stop_words_en = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words_en]
    return " ".join(lemmatized_tokens)

# Load model dan resource lainnya
model, vectorizer, encoder = load_resources()

# ----------------- TAMPILAN APLIKASI STREAMLIT -----------------

st.title("Analisis Sentimen Menggunakan Bi-LSTM")
st.write("Aplikasi ini memprediksi sentimen (positif/negatif) dari ulasan atau teks berbahasa Indonesia.")

# Area input dari pengguna
user_input = st.text_area("Masukkan teks di sini:", "Contoh: Pelayanannya cepat dan memuaskan!")
if st.button("Analisis Sekarang"):
    # Cek apakah resource sudah siap dan ada input dari pengguna
    if not model or not vectorizer or not encoder:
        st.warning("Aplikasi belum siap karena gagal memuat resource.")
    elif user_input:
        # Tampilkan spinner saat proses berjalan
        with st.spinner("Sedang memproses..."):
            
            # 1. Preprocess teks input
            processed_text = preprocess_text(user_input)

            # 2. Hitung skor AFINN
            afinn_analyzer = Afinn()
            afinn_score = afinn_analyzer.score(processed_text)

            # 3. Transformasi TF-IDF
            tfidf_vector = vectorizer.transform([processed_text])
            
            # 4. Reshape vektor agar cocok untuk input LSTM
            reshaped_vector = np.expand_dims(tfidf_vector.toarray(), axis=1)

            # 5. Prediksi menggunakan model
            prediction_probs = model.predict(reshaped_vector)
            predicted_index = np.argmax(prediction_probs, axis=1)[0]
            
            # 6. Decode label
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
            st.write(f"**Skor AFINN:** `{afinn_score}`")
            st.write(f"**Probabilitas Prediksi:** `{prediction_probs[0]}`")

    else:
        st.warning("Harap masukkan teks untuk dianalisis.")
        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        if final_prediction.lower() == 'positif':
            st.success(f"Sentimen: **Positif** 👍")
        else:
            st.error(f"Sentimen: **Negatif** 👎")

        # Tampilkan detail proses jika ingin
        with st.expander("Lihat Detail Proses"):
            st.write(f"**Teks Setelah Preprocessing:** `{processed_text}`")
            st.write(f"**Probabilitas Prediksi:** `{prediction_probs[0]}`")

    else:
        st.warning("Harap masukkan teks untuk dianalisis.")
