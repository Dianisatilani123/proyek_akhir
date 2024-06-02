# 1. Tentukan library yang digunakan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# 2. Load dataset
data = pd.read_csv('aug_train.csv')
# Tampilkan beberapa baris pertama dari dataset untuk memahami strukturnya
print(data.head())

# 3. Standarisasi data
# Pilih fitur yang relevan dan target
features = ['gender', 'education_level', 'experience', 'city_development_index']
target = 'target'

# Menghapus baris dengan nilai yang hilang pada fitur yang dipilih dan target
data = data.dropna(subset=features + [target])

# Encoding gender menjadi numerik
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})

# Encoding education_level menjadi numerik
education_level_mapping = {
    'Primary School': 1,
    'High School': 2,
    'Graduate': 3,
    'Masters': 4,
    'Phd': 5
}
data['education_level'] = data['education_level'].map(education_level_mapping)

# Encoding experience menjadi numerik
experience_mapping = {
    '<1': 0,
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '>20': 21
}
data['experience'] = data['experience'].map(experience_mapping)

# Memisahkan fitur dan target
X = data[features]
y = data[target]

# Impute missing values with mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standarisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 4. Split data train dan test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Membuat model menggunakan algoritma Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Membuat model evaluasi untuk uji akurasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 7. Membuat model untuk aplikasi
def predict_acceptance(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    return prediction

# 8. Deploy aplikasi AI dengan streamlit
def main():
    st.title("AI Deteksi Bias Gender pada Perekrutan Kerja")
    
    st.write("Masukkan fitur-fitur untuk memprediksi apakah kandidat diterima:")
    
    city_development_index = st.number_input("City Development Index", min_value=0.0, max_value=1.0, step=0.01)
    education_level = st.selectbox("Education Level", list(education_level_mapping.keys()))
    education_level = education_level_mapping[education_level]
    experience = st.selectbox("Experience", list(experience_mapping.keys()))
    experience = experience_mapping[experience]
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 0 if gender == "Male" else 1
    
    if st.button("Prediksi"):
        result = predict_acceptance([gender, education_level, experience, city_development_index])
        if result == 1:
            st.success("Kandidat diterima")
        else:
            st.error("Kandidat ditolak")
    
    st.write(f"Akurasi model: {accuracy * 100:.2f}%")
    st.write("Laporan Klasifikasi:")
    st.text(report)

if __name__ == "__main__":
    main()
