import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Load dataset
menstruasi_dataset = pd.read_csv('menstruasi_dataset.csv')
keputihan_dataset = pd.read_csv('keputihan_dataset.csv')

# Standarisasi data
scaler = StandardScaler()
menstruasi_dataset[['Durasi', 'Gejala']] = scaler.fit_transform(menstruasi_dataset[['Durasi', 'Gejala']])
keputihan_dataset[['Warna', 'Karakteristik', 'Penyebab']] = scaler.fit_transform(keputihan_dataset[['Warna', 'Karakteristik', 'Penyebab']])

# Split data train dan test
X_menstruasi = menstruasi_dataset.drop('Fase', axis=1)
y_menstruasi = menstruasi_dataset['Fase']
X_train_menstruasi, X_test_menstruasi, y_train_menstruasi, y_test_menstruasi = train_test_split(X_menstruasi, y_menstruasi, test_size=0.2, random_state=42)

X_keputihan = keputihan_dataset.drop('Jenis Keputihan', axis=1)
y_keputihan = keputihan_dataset['Jenis Keputihan']
X_train_keputihan, X_test_keputihan, y_train_keputihan, y_test_keputihan = train_test_split(X_keputihan, y_keputihan, test_size=0.2, random_state=42)

# Membuat model menggunakan algoritma Random Forest
rf_menstruasi = RandomForestClassifier(n_estimators=100, random_state=42)
rf_menstruasi.fit(X_train_menstruasi, y_train_menstruasi)

rf_keputihan = RandomForestClassifier(n_estimators=100, random_state=42)
rf_keputihan.fit(X_train_keputihan, y_train_keputihan)

# Membuat model evaluasi untuk uji akurasi
y_pred_menstruasi = rf_menstruasi.predict(X_test_menstruasi)
y_pred_keputihan = rf_keputihan.predict(X_test_keputihan)

print("Akurasi Menstruasi:", accuracy_score(y_test_menstruasi, y_pred_menstruasi))
print("Laporan Klasifikasi Menstruasi:\n", classification_report(y_test_menstruasi, y_pred_menstruasi))

print("Akurasi Keputihan:", accuracy_score(y_test_keputihan, y_pred_keputihan))
print("Laporan Klasifikasi Keputihan:\n", classification_report(y_test_keputihan, y_pred_keputihan))

# Membuat model aplikasi dengan streamlit
st.title("Aplikasi Kesehatan Wanita")

st.header("Registrasi dan Profil Pengguna")
usia = st.number_input("Usia")
riwayat_kesehatan = st.text_input("Riwayat Kesehatan")
siklus_menstruasi = st.text_input("Siklus Menstruasi")

st.header("Input Data Kesehatan")
siklus_menstruasi_input = st.text_input("Siklus Menstruasi")
keputihan_input = st.text_input("Keputihan")

if st.button("Analisis"):
    # Membuat prediksi menggunakan model
    X_input_menstruasi = pd.DataFrame({'Durasi': [siklus_menstruasi_input], 'Gejala': ['']})
    X_input_menstruasi[['Durasi', 'Gejala']] = scaler.transform(X_input_menstruasi[['Durasi', 'Gejala']])
    y_pred_menstruasi = rf_menstruasi.predict(X_input_menstruasi)

    X_input_keputihan = pd.DataFrame({'Warna': [keputihan_input], 'Karakteristik': [''], 'Penyebab': ['']})
    X_input_keputihan[['Warna', 'Karakteristik', 'Penyebab']] = scaler.transform(X_input_keputihan[['Warna', 'Karakteristik', 'Penyebab']])
    y_pred_keputihan = rf_keputihan.predict(X_input_keputihan)

    # Menampilkan hasil analisis
    st.write("Hasil Analisis:")
    st.write("Fase Menstruasi:", y_pred_menstruasi[0])
    st.write("Jenis Keputihan:", y_pred_keputihan[0])

    # Menampilkan saran medis yang dipersonalisasi
    st.write("Saran Medis:")
    # TODO: implementasi saran medis yang dipersonalisasi

    # Menampilkan artikel dan video edukasi yang relevan
    st.write("Artikel dan Video Edukasi:")
    # TODO: implementasi artikel dan video edukasi yang relevan

    # Menampilkan konsultasi dengan dokter
    st.write("Konsultasi dengan Dokter:")
    # TODO: implementasi konsultasi dengan dokter

    # Menampilkan akses layanan kesehatan
    st.write("Akses Layanan Kesehatan:")
    # TODO: implementasi akses layanan kesehatan
