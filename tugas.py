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

# Print column names
print("Column names:", data.columns)

features = ['city_development_index', 'enrolled_university', 
            'last_new_job', 'training_hours', 'elevent_experience', 'education_level', 'ajor_discipline', 'experience']
target = 'target'

for feature in features + [target]:
    if feature not in data.columns:
        raise ValueError(f"Column '{feature}' does not exist in the dataset.")

# Standarisasi data

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

# Encoding relevent_experience menjadi numerik
relevent_experience_mapping = {
    'Has relevent experience': 1,
    'No relevent experience': 0
}
data['relevent_experience'] = data['relevent_experience'].map(relevent_experience_mapping)

# Encoding education_level menjadi numerik
education_level_mapping = {
    'Graduate': 1,
    'Masters': 2,
    'Phd': 3,
    'High School': 0
}
data['education_level'] = data['education_level'].map(education_level_mapping)

# Encoding major_discipline menjadi numerik
major_discipline_mapping = {
    'STEM': 1,
    'Business': 2,
    'Humanities': 3,
    'Others': 0
}
data['major_discipline'] = data['major_discipline'].map(major_discipline_mapping)

# Split data train dan test
X = data[features]
y = data[target]

# Check for infinity values
inf_count = 0
neginf_count = 0
for col in X.columns:
    if X[col].dtype.kind in 'bifc':  # Check if column is numeric
        X_array = X[col].to_numpy()  # Convert column to a numpy array
        inf_count += np.isinf(X_array).sum()
        neginf_count += np.sum(np.isinf(-X_array))
if inf_count > 0 or neginf_count > 0:
    print(f"Found {inf_count} infinity values and {neginf_count} negative infinity values.")
    X = X.replace([np.inf, -np.inf], np.nan)  # Replace infinity values with NaN

# Check for missing values
na_count = X.isna().sum().sum()
if na_count > 0:
    print(f"Found {na_count} missing values.")
    X_array = X.to_numpy()  # Convert to numpy array
    imputer = SimpleImputer(strategy='mean')
    X_array = imputer.fit_transform(X_array)  # Fit and transform
else:
    X_array = X.to_numpy()  # Convert to numpy array if no missing values.

print("Shape of X after imputing NaN values:", X_array.shape)

X_train, X_test, y_train, y_test = train_test_split(X_array, y, test_size=0.1, random_state=42)
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

    enrollee_id = st.text_input("Enrollee ID")
    gender = st.selectbox("Gender", ["Male", "Female"])

    city_development_index = st.text_input("City Development Index")
    enrolled_university = st.selectbox("Enrolled University", list(enrolled_university_mapping.keys()), index=0)
    enrolled_university = enrolled_university_mapping[enrolled_university]
    last_new_job = st.selectbox("Last New Job", list(last_new_job_mapping.keys()), index=0)
    last_new_job = last_new_job_mapping[last_new_job]
    training_hours = st.number_input("Training Hours", min_value=0, step=1)
    relevent_experience = st.selectbox("Relevent Experience", list(relevent_experience_mapping.keys()), index=0)
    relevent_experience = relevent_experience_mapping[relevent_experience]
    education_level = st.selectbox("Education Level", list(education_level_mapping.keys()), index=0)
    education_level = education_level_mapping[education_level]
    major_discipline = st.selectbox("Major Discipline", list(major_discipline_mapping.keys()), index=0)
    major_discipline = major_discipline_mapping[major_discipline]
    experience = st.number_input("Experience", min_value=0, step=1)

    if st.button("Prediksi"):
        result = predict_acceptance([float(city_development_index), enrolled_university, 
                                     last_new_job, training_hours, relevent_experience, education_level, major_discipline, experience])
        if result == 1:
            st.write("Kandidat diterima")
        else:
            st.write("Kandidat ditolak")

    st.write(f"Akurasi model: {accuracy * 100:.2f}%")
    st.write(report)

if __name__ == "__main__":
    main()
