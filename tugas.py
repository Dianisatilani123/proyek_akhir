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
features = ['enrollee_id', 'city_development_index', 'elevent_experience', 
            'enrolled_university', 'last_new_job', 'training_hours']
target = 'target'

# Menghapus baris dengan nilai yang hilang pada fitur yang dipilih dan target
data = data.dropna(subset=features + [target])

# Encoding relevent_experience menjadi numerik
data['relevent_experience'] = data['relevent_experience'].map({'Has relevent experience': 1, 'No relevent experience': 0})

# Encoding enrolled_university menjadi numerik
enrolled_university_mapping = {
    'no_enrollment': 0,
    'Full time course': 1,
    'Part time course': 2
}
data['enrolled_university'] = data['enrolled_university'].map(enrolled_university_mapping)

# Encoding last_new_job menjadi numerik
last_new_job_mapping = {
    'never': 0,
    '1': 1, '2': 2, '3': 3, '4': 4, '>4': 5
}
data['last_new_job'] = data['last_new_job'].map(last_new_job_mapping)

# 4. Split data train dan test
X = data[features]
y = data[target]

# Remove rows with NaN values
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)

# 5. Membuat model menggunakan algoritma Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Membuat model evaluasi untuk uji akurasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)  # Set zero_division to 0

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
    
    enrollee_id = st.text_input("Enrollee ID")
    city_development_index = st.number_input("City Development Index", min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    relevent_experience = st.selectbox("Relevent Experience", ["Has relevent experience", "No relevent experience"])
    relevent_experience = 1 if relevent_experience == "Has relevent experience" else 0
    enrolled_university = st.selectbox("Enrolled University", list(enrolled_university_mapping.keys()))
    enrolled_university = enrolled_university_mapping[enrolled_university]
    last_new_job = st.selectbox("Last New Job", list(last_new_job_mapping.keys()))
    last_new_job = last_new_job_mapping[last_new_job]
    training_hours = st.number_input("Training Hours", min_value=0, step=1)
    
    gender = st.selectbox("Gender", ["Male", "Female"])  # Inputan gender masih ada, tapi tidak mempengaruhi prediksi
    
    if st.button("Prediksi"):
        result = predict_acceptance([enrollee_id, city_development_index, relevent_experience, 
                                     enrolled_university, last_new_job, training_hours])
        if result == 1:
            st.success("Kandidat diterima")
        else:
            st.error("Kandidat ditolak")
    
    st.write(f"Akurasi model: {accuracy * 100:.2f}%")
    st.write("Laporan Klasifikasi:")
    st.text(report)

if __name__ == "__main__":
    main()
