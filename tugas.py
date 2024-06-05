import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Load dataset
df = pd.read_csv('aug_train.csv')

# Check DataFrame structure
print(df.columns)

# Check for the existence of the 'target' column
if 'target' not in df.columns:
    raise ValueError("The 'target' column does not exist in the DataFrame")

# Drop columns that can cause bias
df = df.drop(columns=['enrollee_id', 'gender'])

# Fill missing values for numerical columns with median
numerical_cols = ['city_development_index', 'training_hours']
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Fill missing values for categorical columns with mode
categorical_cols = ['city', 'elevent_experience', 'enrolled_university', 'education_level', 'ajor_discipline', 'experience', 'company_size', 'company_type', 'last_new_job']
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Separate features and target
y = df['target']
X = df.drop(columns=['target'])

# Preprocessing for numerical and categorical columns
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Create pipeline for training
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier())])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Streamlit for web interface
st.title('Aplikasi AI Rekrutmen Tanpa Bias')

# Display some rows from the dataset
st.subheader('Dataset')
st.write(df.head())

# Display model evaluation
st.subheader('Model Evaluation')
st.write(f'Accuracy: {accuracy}')
st.write(f'Classification Report: \n{report}')

# Input form for new candidate data
def user_input_features():
    city_development_index = st.sidebar.slider('City Development Index', 0.0, 1.0, 0.5)
    city = st.sidebar.selectbox('City', df['city'].unique())
    relevent_experience = st.sidebar.selectbox('Relevant Experience', ['No relevent experience', 'Has relevent experience'])
    enrolled_university = st.sidebar.selectbox('Enrolled University', ['no_enrollment', 'Part time course', 'Full time course'])
    education_level = st.sidebar.selectbox('Education Level', ['High School', 'Graduate', 'Masters', 'Phd'])
    major_discipline = st.sidebar.selectbox('Major Discipline', ['STEM', 'Business Degree', 'Arts', 'Humanities', 'Other'])
    experience = st.sidebar.slider('Experience (years)', 0, 20, 0)
    company_size = st.sidebar.selectbox('Company Size', ['<10', '10-49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'])
    company_type = st.sidebar.selectbox('Company Type', ['Pvt Ltd', 'Funded Startup', 'Early Stage Startup', 'Public Sector', 'NGO', 'Other'])
    last_new_job = st.sidebar.selectbox('Last New Job', ['never', '1', '2', '3', '4', '>4'])
    training_hours = st.sidebar.slider('Training Hours', 0, 500, 0)

    data = {'city_development_index': city_development_index,
            'city': city,
            'elevent_experience': relevent_experience,
            'enrolled_university': enrolled_university,
            'education_level': education_level,
            'ajor_discipline': major_discipline,
            'experience': experience,
            'company_size': company_size,
            'company_type': company_type,
            'last_new_job': last_new_job,
            'training_hours': training_hours}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Predict function with handling for missing 'city' column and encoding categorical columns
def predict(data):
    # Encode categorical columns
    data_encoded = pd.get_dummies(data, columns=categorical_cols)
    
    # Ensure all columns from the training data are present
    missing_cols = set(X_train.columns) - set(data_encoded.columns)
    for col in missing_cols:
        data_encoded[col] = 0
    
    # Reorder columns to match the training data
    data_encoded = data_encoded[X_train.columns]
    
    # Predict using the model
    prediction = model.predict(data_encoded)
    return prediction

# Predict
prediction = predict(input_df)
if prediction is not None:
    st.subheader('Prediction')
    st.write('Hired' if prediction == 1 else 'Not Hired')
