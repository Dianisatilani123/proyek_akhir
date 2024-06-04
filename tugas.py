import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# 1. Load data
data = pd.read_csv('aug_train.csv')
print(data.head())

# 2. Preprocess data
# Fill missing values with mean
data.fillna(data.mean(), inplace=True)

# Encode categorical features
categorical_features = ['city', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'company_size', 'company_type', 'last_new_job']
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# Scale numerical features
numeric_features = data.select_dtypes(include=['int', 'float']).columns
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Split data into training and testing sets
X = data.drop(['target', 'enrollee_id', 'gender'], axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions
def predict(data):
    # Encode categorical features
    for feature, le in label_encoders.items():
        data[feature] = le.transform([data[feature]])[0]
    
    # Scale numerical features
    numeric_features = data.select_dtypes(include=['int', 'float']).columns
    data[numeric_features] = scaler.transform(data[numeric_features])
    
    # Make prediction
    prediction = model.predict(data)
    return prediction[0]

# Streamlit user interface
st.title("Job Candidate Acceptance Prediction")

enrollee_id = st.text_input("Enrollee ID")
city = st.selectbox("City", label_encoders['city'].classes_)
city_development_index = st.slider("City Development Index", 0.0, 1.0, 0.5)
gender = st.selectbox("Gender", label_encoders['gender'].classes_)
relevent_experience = st.selectbox("Relevent Experience", label_encoders['relevent_experience'].classes_)
enrolled_university = st.selectbox("Enrolled University", label_encoders['enrolled_university'].classes_)
education_level = st.selectbox("Education Level", label_encoders['education_level'].classes_)
major_discipline = st.selectbox("Major Discipline", label_encoders['major_discipline'].classes_)

experience_mapping = {'<1': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '>20': 21}
experience = st.selectbox("Experience", list(experience_mapping.keys()))

company_size = st.selectbox("Company Size", label_encoders['company_size'].classes_)
company_type = st.selectbox("Company Type", label_encoders['company_type'].classes_)
last_new_job = st.selectbox("Last New Job", label_encoders['last_new_job'].classes_)
training_hours = st.number_input("Training Hours", min_value=0)

if st.button("Predict"):
    data = {
        'enrollee_id': enrollee_id,
        'city': city,
        'city_development_index': city_development_index,
        'gender': gender,
        'relevent_experience': relevent_experience,
        'enrolled_university': enrolled_university,
        'education_level': education_level,
        'major_discipline': major_discipline,
        'experience': experience_mapping[experience],
        'company_size': company_size,
        'company_type': company_type,
        'last_new_job': last_new_job,
        'training_hours': training_hours
    }
    result = predict(pd.DataFrame([data]))
    if result == 1:
        st.success("The candidate is likely to be accepted.")
    else:
        st.error("The candidate is not likely to be accepted.")
