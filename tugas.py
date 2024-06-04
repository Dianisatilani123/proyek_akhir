import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load dataset
df = pd.read_csv('aug_train.csv')

# Display some data from the dataset
print(df.head())

# Standardize data
df.drop(['enrollee_id', 'city', 'gender'], axis=1, inplace=True)  # remove columns that can cause bias

# Fill missing values in categorical columns with a placeholder
df.loc[df['relevent_experience'].isna(), 'relevent_experience'] = 'Unknown'
df.loc[df['enrolled_university'].isna(), 'enrolled_university'] = 'Unknown'
df.loc[df['education_level'].isna(), 'education_level'] = 'Unknown'
df.loc[df['major_discipline'].isna(), 'major_discipline'] = 'Unknown'
df.loc[df['company_size'].isna(), 'company_size'] = 'Unknown'
df.loc[df['company_type'].isna(), 'company_type'] = 'Unknown'
df.loc[df['last_new_job'].isna(), 'last_new_job'] = 'Unknown'

# Fill missing values in 'experience' with '0'
df.loc[df['experience'].isna(), 'experience'] = '0'

# Convert 'experience' column to numerical values
def convert_experience(exp):
    if exp == '>20':
        return 21
    elif exp == '<1':
        return 0
    else:
        return int(exp)

df['experience'] = df['experience'].apply(convert_experience)

# Possible categories for each categorical column
possible_relevent_experience = ['Has relevent experience', 'No relevent experience', 'Unknown']
possible_enrolled_university = ['no_enrollment', 'Full time course', 'Part time course', 'Unknown']
possible_education_level = ['Graduate', 'Masters', 'Phd', 'High School', 'Primary School', 'Unknown']  # Added 'Phd'
possible_major_discipline = ['STEM', 'Business Degree', 'Humanities', 'Arts', 'No Major', 'Unknown']  # Added 'No Major'
possible_company_size = ['<1', '1-49', '50-99', '100-500', '500-999', 'Unknown']
possible_company_type = ['Pvt Ltd', 'Funded Startup', 'Public Sector', 'Unknown']
possible_last_new_job = ['never', '>4', '>2', '>1', '<1', 'Unknown']

# Fit LabelEncoders with predefined categories
le_relevent_experience = LabelEncoder().fit(df['relevent_experience'])
le_enrolled_university = LabelEncoder().fit(df['enrolled_university'])
le_education_level = LabelEncoder().fit(df['education_level'])
le_major_discipline = LabelEncoder().fit(df['major_discipline'])
le_company_size = LabelEncoder().fit(df['company_size'])
le_company_type = LabelEncoder().fit(df['company_type'])
le_last_new_job = LabelEncoder().fit(df['last_new_job'])

df['relevent_experience'] = le_relevent_experience.transform(df['relevent_experience'])
df['enrolled_university'] = le_enrolled_university.transform(df['enrolled_university'])
df['education_level'] = le_education_level.transform(df['education_level'])
df['major_discipline'] = le_major_discipline.transform(df['major_discipline'])
df['company_size'] = le_company_size.transform(df['company_size'])
df['company_type'] = le_company_type.transform(df['company_type'])
df['last_new_job'] = le_last_new_job.transform(df['last_new_job'])

# Create a new feature that combines 'relevent_experience' and 'experience'
df['experience_score'] = df['relevent_experience'] * df['experience']

# Split data into training and testing sets
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a model using Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("Presisi:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Deploy the application using Streamlit
st.title("Aplikasi Rekrutmen AI")
st.write("Masukkan data kandidat untuk memprediksi kelayakan")

city_development_index = st.selectbox("City Development Index", [0.624, 0.762, 0.764, 0.767, 0.776, 0.789, 0.804, 0.827, 0.843, 0.913, 0.92, 0.926]
)
relevent_experience = st.selectbox("Relevent Experience", possible_relevent_experience)
enrolled_university = st.selectbox("Enrolled University", possible_enrolled_university)
education_level = st.selectbox("Education Level", possible_education_level)
major_discipline = st.selectbox("Major Discipline", possible_major_discipline)
experience = st.number_input('Experience', min_value=0, max_value=21)
company_size = st.selectbox("Company Size", possible_company_size) 
company_type = st.selectbox("Company Type", possible_company_type)
last_new_job = st.selectbox("Last New Job", possible_last_new_job)
training_hours = st.slider("Training Hours", 0, 100, 20)

# Check if user-entered company_size is in the list of predefined categories
possible_company_size = ['<1', '1-49', '50-99', '100-500', '500-999', 'Unknown']
le_company_size = LabelEncoder().fit(possible_company_size)

input_data = pd.DataFrame({'city_development_index': [city_development_index],
                           'relevent_experience': [relevent_experience],
                           'enrolled_university': [enrolled_university],
                           'education_level': [education_level],
                           'major_discipline': [major_discipline],
                           'experience': [experience],
                           'company_size': [company_size],
                           'company_type': [company_type],
                           'last_new_job': [last_new_job],
                           'training_hours': [training_hours],
                           'experience_score': [0]})  # Added 'experience_score' column

# Convert categorical data into numerical data using the same LabelEncoders
input_data['relevent_experience'] = le_relevent_experience.transform([relevent_experience])
input_data['enrolled_university'] = le_enrolled_university.transform([enrolled_university])
input_data['education_level'] = le_education_level.transform([education_level])
input_data['major_discipline'] = le_major_discipline.transform([major_discipline])
input_data['company_size'] = le_company_size.transform([company_size])
input_data['company_type'] = le_company_type.transform([company_type])
input_data['last_new_job'] = le_last_new_job.transform([last_new_job])

# Calculate 'experience_score' for input data
input_data['experience_score'] = input_data['relevent_experience'] * input_data['experience']

# Check if company_size is None before making a prediction
if company_size is not None:
    prediction = model.predict(input_data)
    st.write("Prediksi kelayakan:", prediction[0])
else:
    st.error("Company size is not in the list of predefined categories. Please select a valid option.")
