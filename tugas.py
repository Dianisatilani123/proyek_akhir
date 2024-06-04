import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Memuat data
data = pd.read_csv('aug_train.csv')
print(data.head())


# 2. Pra-pemrosesan data
# Mengisi nilai yang hilang (jika ada)
data.fillna('', inplace=True)

# Encoder untuk fitur kategorikal
label_encoders = {}
categorical_features = ['city', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'company_size', 'company_type', 'last_new_job']

for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# Fitur dan target
X = data.drop(['target', 'enrollee_id', 'gender'], axis=1)
y = data['target']

# Standarisasi fitur numerik
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Melatih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Membuat prediksi
def predict(data):
    # Melakukan encoding untuk fitur kategorikal
    for feature, le in label_encoders.items():
        data[feature] = le.transform([data[feature]])[0]
    
    # Membuat dataframe dari input data
    df = pd.DataFrame([data])
    
    # Menghapus kolom yang tidak digunakan
    df = df.drop(['enrollee_id', 'gender'], axis=1)
    
    # Standarisasi fitur numerik
    df = scaler.transform(df)
    
    # Melakukan prediksi
    prediction = model.predict(df)
    
    return prediction[0]

# Streamlit antarmuka pengguna
st.title("Job Candidate Acceptance Prediction")

enrollee_id = st.text_input("Enrollee ID")
city = st.selectbox("City", label_encoders['city'].classes_)
city_development_index = st.slider("City Development Index", 0.0, 1.0, 0.5)
gender = st.selectbox("Gender", label_encoders['gender'].classes_)
relevent_experience = st.selectbox("Relevent Experience", label_encoders['relevent_experience'].classes_)
enrolled_university = st.selectbox("Enrolled University", label_encoders['enrolled_university'].classes_)
education_level = st.selectbox("Education Level", label_encoders['education_level'].classes_)
major_discipline = st.selectbox("Major Discipline", label_encoders['major_discipline'].classes_)
experience = st.selectbox("Experience", ['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '>20'])
company_size = st.selectbox("Company Size", label_encoders['company_size'].classes_)
company_type = st.selectbox("Company Type", label_encoders['company_type'].classes_)
last_new_job = st.selectbox("Last New Job", label_encoders['last_new_job'].classes_)
training_hours = st.number_input("Training Hours", min_value=0)

if st.button("Predict"):
    data = {
        'enrollee_id': enrollee_id,
        'city': city,
        'city_development_index': city_development_index,
        'relevent_experience': relevent_experience,
        'enrolled_university': enrolled_university,
        'education_level': education_level,
        'major_discipline': major_discipline,
        'experience': experience,
        'company_size': company_size,
        'company_type': company_type,
        'last_new_job': last_new_job,
        'training_hours': training_hours
    }
    result = predict(data)
    if result == 1:
        st.success("The candidate is likely to be accepted.")
    else:
        st.error("The candidate is not likely to be accepted.")
