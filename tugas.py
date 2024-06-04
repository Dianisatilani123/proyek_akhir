import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Save model
joblib.dump(model, 'model_rekrutmen.pkl')

# Deploy AI application with Streamlit

# Load model
model = joblib.load('model_rekrutmen.pkl')

# App title
st.title('Aplikasi Rekrutmen Tanpa Bias')

# User input
enrollee_id = st.text_input('Enrollee ID')
city_development_index = st.number_input('City Development Index', min_value=0.0, max_value=1.0)
relevent_experience = st.selectbox('Relevent Experience', ['Has relevent experience', 'No relevent experience'])

enrolled_university = st.selectbox('Enrolled University', df['enrolled_university'].unique())
education_level = st.selectbox('Education Level', df['education_level'].unique())
company_size = st.selectbox('Company Size', df['company_size'].unique())
company_type = st.selectbox('Company Type', df['company_type'].unique())
last_new_job = st.selectbox('Last New Job', df['last_new_job'].unique())

experience = st.number_input('Experience', min_value=0, max_value=20)
training_hours = st.number_input('Training Hours', min_value=0)

# Combine user input into dataframe
input_data = pd.DataFrame({
    'city_development_index': [city_development_index],
    'relevent_experience': [1 if relevent_experience == 'Has relevent experience' else 0],
    'enrolled_university_' + enrolled_university: [1],
    'education_level_' + education_level: [1],
    'experience': [experience],
    'company_size_' + company_size: [1],
    'company_type_' + company_type: [1],
    'last_new_job_' + last_new_job: [1],
    'training_hours': [training_hours]
}, index=[0])

# Predict
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.success('Kandidat Berpotensi Diterima')
    else:
        st.error('Kandidat Tidak Berpotensi Diterima')
