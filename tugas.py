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
# Menghapus kolom yang dapat memicu bias
df = df.drop(columns=['enrollee_id', 'gender', 'major_discipline'])

# Mengisi nilai yang hilang untuk kolom numerik dengan median
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Mengisi nilai yang hilang untuk kolom kategorikal dengan modus
categorical_columns = df.select_dtypes(exclude=[np.number]).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# One-hot encoding untuk kolom kategorikal
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
city = st.selectbox('City', ['city_103', 'city_40', 'city_21', 'city_115', 'city_162', 'city_176', 'city_160'])
city_development_index = st.number_input('City Development Index', min_value=0.0, max_value=1.0)
relevent_experience = st.selectbox('Relevent Experience', ['Has relevent experience', 'No relevent experience'])
enrolled_university = st.selectbox('Enrolled University', ['no_enrollment', 'Full time course', 'Part time course'])
education_level = st.selectbox('Education Level', ['Graduate', 'High School', 'Masters', 'Primary School'])
experience = st.number_input('Experience', min_value=0, max_value=20)
company_size = st.selectbox('Company Size', ['<10', '10-49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'])
company_type = st.selectbox('Company Type', ['Pvt Ltd', 'Funded Startup', 'Public Sector', 'Early Stage Startup'])
last_new_job = st.selectbox('Last New Job', ['never', '1', '2', '3', '4', '>4'])
training_hours = st.number_input('Training Hours', min_value=0)

# Menggabungkan input pengguna menjadi dataframe
input_data = pd.DataFrame({
    'city': [city],
    'city_development_index': [city_development_index],
    'relevent_experience': [relevent_experience],
    'enrolled_university': [enrolled_university],
    'education_level': [education_level],
    'experience': [experience],
    'company_size': [company_size],
    'company_type': [company_type],
    'last_new_job': [last_new_job],
    'training_hours': [training_hours]
})

# One-hot encoding untuk input data
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Prediksi
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.success('Kandidat Berpotensi Diterima')
    else:
        st.error('Kandidat Tidak Berpotensi Diterima')

