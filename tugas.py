import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import streamlit as st

# Load dataset
data = pd.read_csv('aug_train.csv')
print(data.head())

# Standarisasi data
features = ['city_development_index', 'enrolled_university', 
            'last_new_job', 'training_hours']
target = 'target'

# Menghapus baris dengan nilai yang hilang pada fitur yang dipilih dan target
data = data.dropna(subset=features + [target])

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

# Split data train dan test
X = data[features]
y = data[target]

# Check for infinity values
inf_count = np.isinf(X).sum().sum()
neginf_count = np.isneginf(X).sum().sum()
if inf_count > 0 or neginf_count > 0:
    print(f"Found {inf_count} infinity values and {neginf_count} negative infinity values.")
    X = X.replace([np.inf, -np.inf], np.nan)

# Check for missing values
na_count = X.isna().sum().sum()
if na_count > 0:
    print(f"Found {na_count} missing values.")
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

print("Shape of X after imputing NaN values:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membuat model menggunakan algoritma Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Membuat model evaluasi untuk uji akurasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

def predict_acceptance(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("AI Deteksi Bias Gender pada Perekrutan Kerja")

    st.write("Masukkan fitur-fitur untuk memprediksi apakah kandidat diterima:")

    city_development_index = st.slider("City Development Index", min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    enrolled_university = st.selectbox("Enrolled University", list(enrolled_university_mapping.keys()), index=0)
    enrolled_university = enrolled_university_mapping[enrolled_university]
    last_new_job = st.selectbox("Last New Job", list(last_new_job_mapping.keys()), index=0)
    last_new_job = last_new_job_mapping[last_new_job]
    training_hours = st.slider("Training Hours", min_value=0, step=1)

    if st.button("Prediksi"):
        result = predict_acceptance([city_development_index, enrolled_university, 
                                     last_new_job, training_hours])
        if result == 1:
            st.write("Kandidat diterima")
        else:
            st.write("Kandidat ditolak")

    st.write(f"Akurasi model: {accuracy * 100:.2f}%")
    st.write(report)

if __name__ == "__main__":
    main()
