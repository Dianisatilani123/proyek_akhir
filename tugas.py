import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st

# Load dataset
df = pd.read_csv('aug_train.csv')

# Drop unnecessary columns
df = df.drop(columns=['gender'])

# Fill missing values for numeric columns with median
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Fill missing values for categorical columns with mode
categorical_columns = df.select_dtypes(exclude=[np.number]).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Encode categorical variables
df = pd.get_dummies(df, columns=categorical_columns)

# Scale numeric columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Split data
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"Classification Report:\n{report}")

# Deploy application with Streamlit

# Title
st.title('Aplikasi Rekrutmen Tanpa Bias')

# Input fields
enrollee_id = st.text_input('Enrollee ID')
city_development_index = st.number_input('City Development Index', min_value=0.0, max_value=1.0)
relevent_experience = st.selectbox('Relevent Experience', ['Has relevent experience', 'No relevent experience'])

enrolled_university = st.selectbox('Enrolled University', df.columns if 'enrolled_university' in df.columns else ['Unknown'])
education_level = st.selectbox('Education Level', df.columns if 'education_level' in df.columns else ['Unknown'])
company_size = st.selectbox('Company Size', df.columns if 'company_size' in df.columns else ['Unknown'])
company_type = st.selectbox('Company Type', df.columns if 'company_type' in df.columns else ['Unknown'])
last_new_job = st.selectbox('Last New Job', df.columns if 'last_new_job' in df.columns else ['Unknown'])

experience = st.number_input('Experience', min_value=0, max_value=20)
training_hours = st.number_input('Training Hours', min_value=0)

# Create input data
input_data = pd.DataFrame({
    'city_development_index': [city_development_index],
    'relevent_experience': [1 if relevent_experience == 'Has relevent experience' else 0]
}, index=[0])

if 'enrolled_university' in df.columns:
    input_data['enrolled_university_' + enrolled_university] = [1]
if 'education_level' in df.columns:
    input_data['education_level_' + education_level] = [1]
if 'company_size' in df.columns:
    input_data['company_size_' + company_size] = [1]
if 'company_type' in df.columns:
    input_data['company_type_' + company_type] = [1]
if 'last_new_job' in df.columns:
    input_data['last_new_job_' + last_new_job] = [1]

input_data['experience'] = [experience]
input_data['training_hours'] = [training_hours]

# Scale input data
input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])

# Predict
if st.button('Predict'):
    try:
        prediction = model.predict(input_data)
        if prediction == 1:
            st.success('Kandidat Berpotensi Diterima')
        else:
            st.error('Kandidat Tidak Berpotensi Diterima')
    except Exception as e:
        st.error(f"Error: {str(e)}")
