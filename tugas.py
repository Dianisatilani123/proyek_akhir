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
features = ['relevent_experience', 'enrolled_university', 
            'education_level', 'training_hours']
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

# Encoding education_level menjadi numerik
education_level_mapping = {
    'Graduate': 0,
    'Masters': 1,
    'Phd': 2
}
data['education_level'] = data['education_level'].map(education_level_mapping)

# Split data train dan test
X = data[features]
y = data[target]

# Check for infinity values
X_num = X.select_dtypes(include=[np.number])  # Select only numeric columns
inf_count = np.isinf(X_num).sum().sum()
neginf_count = np.isneginf(X_num).sum().sum()
if inf_count > 0 or neginf_count > 0:
    print(f"Found {inf_count} infinity values and {neginf_count} negative infinity values.")
    X_num = X_num.replace([np.inf, -np.inf], np.nan)  # Replace infinity values with NaN

# Check for missing values
na_count = X.isna().sum().sum()
if na_count > 0:
    print(f"Found {na_count} missing values.")
    imputer = SimpleImputer(strategy='most_frequent')  # Use most frequent strategy
    X[['relevent_experience', 'enrolled_university', 'education_level']] = imputer.fit_transform(X[['relevent_experience', 'enrolled_university', 'education_level']])
    imputer = SimpleImputer(strategy='mean')  # Use mean strategy for numeric columns
    X[['training_hours']] = imputer.fit_transform(X[['training_hours']])

print("Shape of X after imputing NaN values:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_array = X_train.to_numpy()
X_train = scaler.fit_transform(X_train_array)

X_test_array = X_test.to_numpy()
X_test = scaler.transform(X_test_array)

# Membuat model menggunakan algoritma Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Membuat model evaluasi untuk uji akurasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

def predict_acceptance(input_data):
    relevent_experience_mapping = {
        "Has relevent experience": 1,
        "No relevent experience": 0
    }
    enrolled_university_mapping = {
        'no_enrollment': 0,
        'Full time course': 1,
        'Part time course': 2
    }
    education_level_mapping = {
        'Graduate': 0,
        'Masters': 1,
        'Phd': 2
    }
    input_data[0] = relevent_experience_mapping[input_data[0]]
    input_data[1] = enrolled_university_mapping[input_data[1]]
    input_data[2] = education_level_mapping[input_data[2]]
    input_data = np.array(input_data, dtype=float)  # Convert to float array
    input_data = scaler.transform([input_data])  # Reshape and transform
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("AI Deteksi Bias Gender pada Perekrutan Kerja")

    st.write("Masukkan fitur-fitur untuk memprediksi apakah kandidat diterima:")

    relevent_experience = st.selectbox("Relevent Experience", ["Has relevent experience", "No relevent experience"])
    enrolled_university = st.selectbox("Enrolled University", list(enrolled_university_mapping.keys()), index=0)
    education_level = st.selectbox("Education Level", list(education_level_mapping.keys()), index=0)

    training_hours = st.slider("Training Hours", min_value=0, step=1)

    if st.button("Prediksi"):
        result = predict_acceptance([relevent_experience, enrolled_university, education_level, training_hours])
        if result == 1:
            st.write("Kandidat diterima")
        else:
            st.write("Kandidat ditolak")

        st.write(f"Akurasi model: {accuracy * 100:.2f}%")
        st.write(report)

if __name__ == "__main__":
    main()
