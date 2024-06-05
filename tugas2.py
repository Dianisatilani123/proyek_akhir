import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st

# Load dataset
data = pd.read_csv('dataset.csv')

# Tampilkan beberapa baris data
st.write("Dataset:")
st.write(data.head())

# Fungsi untuk mengubah ukuran perusahaan menjadi integer
def company_size_to_int(size):
    try:
        if '-' in size:
            return int(size.split('-')[0])
        elif '<' in size:
            return 0
        elif '>' in size:
            return 10001
        else:
            return int(size)
    except ValueError:
        return np.nan  # Mengembalikan NaN untuk nilai yang tidak valid

# Filter sesuai syarat kandidat yang diterima
filtered_data = data[
    (data['relevent_experience'] == 'Has relevent experience') &
    (data['education_level'].isin(['Graduate', 'Masters'])) &
    (data['major_discipline'] == 'STEM') &
    (data['experience'].apply(lambda x: x if x.isdigit() else '21').astype(int) > 10) &
    (data['company_size'].apply(company_size_to_int) >= 100) &
    (data['company_type'].isin(['Pvt Ltd', 'Early Stage Startup', 'Funded Startup'])) &
    (data['training_hours'] > 50) &
    (data['last_new_job'].apply(lambda x: x if x.isdigit() else '6').astype(int).between(1, 5))
]

# Periksa apakah filtered_data kosong
if filtered_data.empty:
    st.write("No data matches the filter criteria.")
else:
    # Definisikan fitur dan target (buat kolom target sementara untuk demo)
    filtered_data['target_column'] = np.random.randint(2, size=len(filtered_data))  # Hapus ini saat target sebenarnya tersedia

    # Encode categorical variables
    categorical_cols = ['gender', 'relevent_experience', 'enrolled_university', 'education_level', 
                        'major_discipline', 'company_size', 'company_type', 'last_new_job']
    le = LabelEncoder()
    for col in categorical_cols:
        filtered_data[col] = le.fit_transform(filtered_data[col])

    # Standarisasi data
    numerical_cols = ['city_development_index', 'training_hours']
    scaler = StandardScaler()
    filtered_data[numerical_cols] = scaler.fit_transform(filtered_data[numerical_cols])

    # Drop kolom yang tidak diperlukan
    features = filtered_data.drop(['enrollee_id', 'city', 'target_column'], axis=1)
    target = filtered_data['target_column']

    # Split data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Membuat data latih menggunakan algoritma machine learning
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Membuat model evaluasi untuk uji akurasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Simpan model dan scaler
    joblib.dump(model, 'recruitment_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Evaluasi Model
    st.write(f'Accuracy: {accuracy}')
    st.write('Classification Report:')
    st.text(report)

    # Load model dan scaler untuk Streamlit
    model = joblib.load('recruitment_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Input data dari user melalui sidebar Streamlit
    def user_input_features():
        city_development_index = st.sidebar.slider('City Development Index', 0.0, 1.0, 0.5)
        training_hours = st.sidebar.slider('Training Hours', 0, 500, 50)
        # Tambahkan input lainnya sesuai dengan dataset
        data = {
            'city_development_index': city_development_index,
            'training_hours': training_hours,
            'gender': st.sidebar.selectbox('Gender', options=['Male', 'Female', 'Other']),
            'relevent_experience': st.sidebar.selectbox('Relevant Experience', options=['Has relevent experience', 'No relevent experience']),
            'enrolled_university': st.sidebar.selectbox('Enrolled University', options=['no_enrollment', 'Full time course', 'Part time course']),
            'education_level': st.sidebar.selectbox('Education Level', options=['Graduate', 'Masters', 'Phd']),
            'major_discipline': st.sidebar.selectbox('Major Discipline', options=['STEM', 'Business Degree', 'Arts', 'Humanities', 'No Major']),
            'experience': st.sidebar.selectbox('Experience', options=['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '>20']),
            'company_size': st.sidebar.selectbox('Company Size', options=['<10', '10-49', '50-99', '100-500', '500-999', '1000-4999', '10000+']),
            'company_type': st.sidebar.selectbox('Company Type', options=['Pvt Ltd', 'Public Sector', 'Early Stage Startup', 'Funded Startup', 'Other']),
            'last_new_job': st.sidebar.selectbox('Last New Job', options=['never', '1', '2', '3', '4', '>4'])
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Encode input data
    for col in categorical_cols:
        input_df[col] = le.fit_transform(input_df[col])

    # Preprocessing
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Predict
    prediction = model.predict(input_df)

    st.write('Hasil Prediksi:')
    st.write(prediction)
