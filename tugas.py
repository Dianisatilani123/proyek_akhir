import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st

# Langkah 2: Memuat Dataset
df = pd.read_csv('aug_train.csv')

# Langkah 3: Standarisasi Data
# Menghapus kolom yang tidak diperlukan
df = df.drop(columns=['gender'])

# Mengisi nilai yang hilang untuk kolom numerik dengan median
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Mengisi nilai yang hilang untuk kolom kategorikal dengan modus
categorical_columns = df.select_dtypes(exclude=[np.number]).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Encode categorical variables
df = pd.get_dummies(df, columns=categorical_columns)

# Langkah 4: Split Data
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Langkah 5: Membuat Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Langkah 6: Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Langkah 7: Simpan Model
joblib.dump(model, 'model_rekrutmen.pkl')

# Langkah 8: Deploy Aplikasi AI dengan Streamlit

# Memuat model
model = joblib.load('model_rekrutmen.pkl')

# Judul aplikasi
st.title('Aplikasi Rekrutmen Tanpa Bias')

# Input pengguna
enrollee_id = st.text_input('Enrollee ID')
city_development_index = st.number_input('City Development Index', min_value=0.0, max_value=1.0)
relevent_experience = st.selectbox('Relevent Experience', ['Has relevent experience', 'No relevent experience'])

enrolled_university = 'Unknown'
if 'enrolled_university' in df.columns:
    enrolled_university = st.selectbox('Enrolled University', df['enrolled_university'].unique())

education_level = 'Unknown'
if 'education_level' in df.columns:
    education_level = st.selectbox('Education Level', df['education_level'].unique())

company_size = 'Unknown'
if 'company_size' in df.columns:
    company_size = st.selectbox('Company Size', df['company_size'].unique())

company_type = 'Unknown'
if 'company_type' in df.columns:
    company_type = st.selectbox('Company Type', df['company_type'].unique())

last_new_job = 'Unknown'
if 'last_new_job' in df.columns:
    last_new_job = st.selectbox('Last New Job', df['last_new_job'].unique())

experience = st.number_input('Experience', min_value=0, max_value=20)
training_hours = st.number_input('Training Hours', min_value=0)

# Menggabungkan input pengguna menjadi dataframe
input_data = pd.DataFrame({
    'enrollee_id': [enrollee_id],
    'city_development_index': [city_development_index],
    'elevent_experience': [relevent_experience],
    'enrolled_university': [enrolled_university],
    'education_level': [education_level],
    'experience': [experience],
    'company_size': [company_size],
    'company_type': [company_type],
    'last_new_job': [last_new_job],
    'training_hours': [training_hours]
})

# Prediksi
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.success('Kandidat Berpotensi Diterima')
    else:
        st.error('Kandidat Tidak Berpotensi Diterima')
